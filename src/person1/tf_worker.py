"""Original ADJSCC TensorFlow modules with deterministic common-data adapter."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from .protocol import Images, fingerprint


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def build(options, strategy):
    import tensorflow as tf
    # Load the installed Linux wheel before the author's path is prepended:
    # upstream also contains a macOS-only tensorflow_compression snapshot.
    import tensorflow_compression
    from .imports import upstream
    with upstream(options['source_dir'],('util_module','util_channel')):
        from util_module import Attention_Encoder,Attention_Decoder
        from util_channel import Channel
        with strategy.scope():
            image=tf.keras.Input((None,None,3));snr=tf.keras.Input((1,))
            normalized=tf.keras.layers.Lambda(lambda x:x/255.,name='normal')(image)
            latent=Attention_Encoder(normalized,snr,options['latent_channels'])
            received=Channel(channel_type='awgn')(latent,snr)
            decoded=Attention_Decoder(received,snr)
            output=tf.keras.layers.Lambda(lambda x:x*255.,name='denormal')(decoded)
            model=tf.keras.Model([image,snr],output)
            optimizer=tf.keras.optimizers.Adam(options['learning_rate'])
            optimizer.build(model.trainable_variables)
    return model,optimizer


def train(options,run,pilot=False):
    import tensorflow as tf
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(180,repeat=True)
    gpus=tf.config.list_physical_devices('GPU')
    if options.get('require_two_gpus',True) and len(gpus)!=2:raise RuntimeError('ADJSCC requires two T4s')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    # Explicit reduction tests a suspected NCCL collective stall on Kaggle T4x2.
    # It changes transport of gradients only, not native model/loss mathematics.
    strategy=tf.distribute.MirroredStrategy(devices=['/gpu:0','/gpu:1'] if gpus else ['/cpu:0'],
        cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device='/gpu:0') if gpus else None)
    tf.keras.utils.set_random_seed(options['seed'])
    model,optimizer=build(options,strategy)
    data=Images(options['manifest'],'train',options['crop']);dev=Images(options['manifest'],'val',options['crop'])
    readers=ThreadPoolExecutor(max_workers=max(1,int(options.get('workers',4))))
    contract=fingerprint(options,data.manifest['identity']);run=Path(run);(run/'checkpoints').mkdir(parents=True,exist_ok=True)
    state=dict(epoch=0,offset=0,step=0,images_seen=0,best=None,contract=contract,completed_epochs={'codec':0})
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    if pilot:
        result=graph_flops(model,options,options['crop'],options['crop'],options['batch_size'],True)
        write(run/'training_profile-codec.json',result)
    def restore(path):
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    (Path(td)/item.filename).resolve().relative_to(Path(td).resolve())
                archive.extractall(td)
            saved=json.loads((Path(td)/'progress.json').read_text())
            if saved['contract']!=contract:raise ValueError('ADJSCC continuation contract changed')
            # Checkpoint.write pairs with read: restore would create an
            # unsaved save_counter and fail strict matching unnecessarily.
            checkpoint.read(str(Path(td)/'state')).assert_consumed()
            return saved
    if options.get('resume_checkpoint') and not pilot:state=restore(options['resume_checkpoint'])
    def save():
        path=run/'checkpoints/last-tf.zip';temp=path.with_suffix('.tmp')
        with tempfile.TemporaryDirectory() as td:
            checkpoint.write(str(Path(td)/'state'));write(Path(td)/'progress.json',state)
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED) as archive:
                for item in Path(td).iterdir():archive.write(item,item.name)
        temp.replace(path);write(run/'progress.json',dict(state,status='partial'))
    started=time.monotonic();deadline=started+options['max_seconds'];last_save=started;observed=set();stopped=False
    while state['epoch']<options['epochs']:
        order=data.order(state['epoch'],options['seed'])
        while state['offset']<len(order):
            if time.monotonic()>=deadline or (options.get('stop_after_steps') and state['step']>=options['stop_after_steps']):stopped=True;break
            selected=order[state['offset']:state['offset']+options['batch_size']]
            array=np.stack(list(readers.map(lambda i:data.array(i,state['epoch'],options['seed']),selected)))*255
            batch=tf.convert_to_tensor(array);size=len(array)
            # Stateless step identity controls native tf.random.normal calls in
            # eager MirroredStrategy; the author channel source is untouched.
            tf.random.set_seed(options['seed']+state['step'])
            def values(ctx):
                lo=size*ctx.replica_id_in_sync_group//strategy.num_replicas_in_sync
                hi=size*(ctx.replica_id_in_sync_group+1)//strategy.num_replicas_in_sync
                with tf.device('/gpu:'+str(ctx.replica_id_in_sync_group) if gpus else '/cpu:0'):
                    return tf.identity(batch[lo:hi])
            distributed=strategy.experimental_distribute_values_from_function(values)
            def step(x):
                observed.add(x.device)
                if pilot:print('Replica forward start: '+x.device,flush=True)
                with tf.GradientTape() as tape:
                    y=model([x,tf.fill([tf.shape(x)[0],1],tf.cast(options['snr_db'],tf.float32))],training=True)
                    losses=tf.reduce_mean(tf.square(y-x),axis=[1,2,3])
                    loss=tf.nn.compute_average_loss(losses,global_batch_size=size)
                gradients=tape.gradient(loss,model.trainable_variables)
                if pilot:print('Replica backward complete: '+x.device,flush=True)
                for gradient in gradients:tf.debugging.assert_all_finite(gradient,'ADJSCC nonfinite gradient')
                optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                if pilot:print('Replica optimizer complete: '+x.device,flush=True)
                return loss
            tick=time.monotonic();loss=strategy.reduce(tf.distribute.ReduceOp.SUM,strategy.run(step,args=(distributed,)),axis=None)
            state['step']+=1;state['offset']+=size;state['images_seen']+=size
            event=dict(epoch=state['epoch']+1,step=state['step'],offset=state['offset'],loss=float(loss),seconds=time.monotonic()-tick)
            with (run/'training.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            if state['step']<=2 or state['step']%25==0:print(json.dumps(event),flush=True)
            if time.monotonic()-last_save>=300:save();last_save=time.monotonic()
            if pilot and state['step']>=2:break
        if stopped or pilot:break
        state['epoch']+=1;state['offset']=0;state['completed_epochs']['codec']=state['epoch']
        values=[]
        for i in range(min(16,len(dev))):
            x=dev.array(i)[None]*255;tf.random.set_seed(options['seed']+900000+i)
            y=model([x,np.full((1,1),options['snr_db'],np.float32)],training=False).numpy()
            values.append(float(np.mean((x-y)**2)))
        validation=float(np.mean(values));improved=state['best'] is None or validation<state['best']
        if improved:state['best']=validation
        save()
        if improved:
            import shutil
            shutil.copy2(run/'checkpoints/last-tf.zip',run/'checkpoints/best-tf.zip')
        print(json.dumps(dict(epoch_completed=state['epoch'],parents=len(data),validation=validation)),flush=True)
    readers.shutdown(wait=True)
    save();before=[v.numpy().copy() for v in model.weights];restore(run/'checkpoints/last-tf.zip')
    if any(not np.array_equal(a,b.numpy()) for a,b in zip(before,model.weights)):raise ValueError('TF strict checkpoint reload changed weights')
    if gpus and not all(any(('GPU:'+str(i)) in d.upper() for d in observed) for i in range(2)):
        raise RuntimeError('MirroredStrategy did not observe both GPU replicas')
    summary=dict(status='pilot_completed' if pilot else 'training_completed' if state['epoch']==options['epochs'] else 'partial',
                 completed_epochs=state['completed_epochs'],target_epochs={'codec':options['epochs']},steps=state['step'],
                 images_seen=state['images_seen'],elapsed_seconds=time.monotonic()-started,
                 strict_reload=True,resume_supported=True,observed_devices=sorted(observed),contract=contract,
                 rng_policy='native eager channel seeded by persisted global optimizer step',paper_convergence_reproduced=False)
    write(run/'model.json',dict(architecture='alexxu1988/ADJSCC original TensorFlow Attention_Encoder/Decoder + Channel',
         source_commit='e5332e95faf592aab9f440992de96029162dc7dd',input_range='native [0,255]',
         loss='native MSE on [0,255]',rate_scope='complex latent pairs; no framing',parameters=model.count_params()))
    write(run/'training_summary.json',summary)
    faulthandler.cancel_dump_traceback_later()
    if pilot:
        x=dev.array(0)[None]*255;tf.random.set_seed(22)
        y=model([x,np.full((1,1),options['snr_db'],np.float32)],training=False).numpy()
        if y.shape!=x.shape or not np.isfinite(y).all():raise ValueError('TF pilot reconstruction invalid')
        geometry=[]
        for h,w in ((512,768),(768,512),(513,769)):
            ph,pw=(-h)%4,(-w)%4
            x=np.random.RandomState(22).rand(1,h+ph,w+pw,3).astype(np.float32)*255
            y=model([x,np.full((1,1),options['snr_db'],np.float32)],training=False).numpy()[:,:h,:w]
            if y.shape!=(1,h,w,3) or not np.isfinite(y).all():raise ValueError('TF geometry pilot failed')
            geometry.append(dict(original=[h,w],padded=[h+ph,w+pw]))
        write(run/'geometry_probe.json',dict(passed=True,synthetic=True,images=geometry))
        write(run.parent/'pilot_passed.json',dict(real_data=True,strict_reload=True,observed_devices=sorted(observed)))
    return summary


def evaluate(options,run):
    import tensorflow as tf
    # The author channel creates random ops while the model is built. Seed
    # before build as well as per trial when deterministic execution is enabled.
    tf.keras.utils.set_random_seed(options['seed'])
    for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
    strategy=tf.distribute.OneDeviceStrategy('/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0')
    model,optimizer=build(options,strategy);checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(Path(run)/'checkpoints/last-tf.zip') as archive:
            for item in archive.infolist():(Path(td)/item.filename).resolve().relative_to(Path(td).resolve())
            archive.extractall(td)
        checkpoint.read(str(Path(td)/'state')).assert_consumed()
    dataset=Images(options['manifest'],'test',options['crop']);folder=Path(run)/'predictions';folder.mkdir(exist_ok=False);rows=[]
    for i in range(len(dataset)):
        original=dataset.array(i);h,w=original.shape[:2];ph,pw=(-h)%4,(-w)%4
        padded=np.pad(original,((0,ph),(0,pw),(0,0)))[None]*255
        for trial in range(options['trials']):
            seed=options['seed']+10000*i+trial;tf.random.set_seed(seed)
            tick=time.perf_counter();value=model([padded,np.full((1,1),options['snr_db'],np.float32)],training=False).numpy()[0,:h,:w]/255
            elapsed=time.perf_counter()-tick
            if not np.isfinite(value).all() or value.shape!=original.shape:raise ValueError('Invalid native TF reconstruction')
            path=folder/('%03d-%02d.npy'%(i,trial));np.save(path,value)
            rows.append(dict(sample=i,trial=trial,seed=seed,path=path.name,seconds=elapsed,
                             complex_channel_uses=(h+ph)*(w+pw)*options['latent_channels']/32,
                             geometry=dict(original=[h,w],padded=[h+ph,w+pw],padding=[0,pw,0,ph])))
            write(Path(run)/'predictions.json',dict(rows=rows,pilot=False))
        if i==0:profile(model,padded,options,run)
    return rows


def graph_flops(model,options,h,w,batch,training=False):
    import tensorflow as tf
    result=dict(batch_size=batch,input_shape=[batch,h,w,3],complete_training_flops=False,
                coverage='TF graph registered float ops only; custom/unsupported ops and optimizer arithmetic excluded;2FLOPs/MAC')
    try:
        @tf.function
        def measured(x):
            with tf.GradientTape() as tape:
                y=model([x,tf.fill([batch,1],float(options['snr_db']))],training=training)
                loss=tf.reduce_mean(tf.square(y-x))
            return tape.gradient(loss,model.trainable_variables) if training else y
        graph=measured.get_concrete_function(tf.TensorSpec([batch,h,w,3],tf.float32)).graph
        settings=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation();settings['output']='none'
        measured_ops=tf.compat.v1.profiler.profile(graph=graph,options=settings)
        result['registered_kernel_flops']=int(measured_ops.total_float_ops) if measured_ops else None
    except Exception as error:result.update(registered_kernel_flops=None,error=type(error).__name__+': '+str(error))
    return result


def profile(model,image,options,run):
    import tensorflow as tf
    snr=np.full((1,1),options['snr_db'],np.float32)
    def call():return model([image,snr],training=False).numpy()
    for _ in range(3):call()
    timings=[]
    for _ in range(10):
        tick=time.perf_counter();call();timings.append(time.perf_counter()-tick)
    result=graph_flops(model,options,image.shape[1],image.shape[2],1)
    result['flops_coverage']=result['coverage']
    result.update(warmup=3,repeats=10,latency_p50_seconds=float(np.median(timings)),
                  latency_p95_seconds=float(np.quantile(timings,.95)),precision='float32',device='GPU0')
    try:
        encoder=tf.keras.Model(model.inputs,model.get_layer('en5_conv').output)
        decoder=tf.keras.Model([model.get_layer('channel').output,model.inputs[1]],model.output)
        latent=encoder([image,snr],training=False);received=model.get_layer('channel')(latent,snr)
        functions=dict(encoder=lambda:encoder([image,snr],training=False).numpy(),
                       channel=lambda:model.get_layer('channel')(latent,snr).numpy(),
                       decoder=lambda:decoder([received,snr],training=False).numpy())
        result['component_seconds']={}
        for name,function in functions.items():
            function();values=[]
            for _ in range(10):tick=time.perf_counter();function();values.append(time.perf_counter()-tick)
            result['component_seconds'][name]=dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    except Exception as error:result['component_error']=type(error).__name__+': '+str(error)
    write(Path(run)/'profile.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','train','evaluate','resume-check']);p.add_argument('--config',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    options=json.loads(Path(a.config).read_text());run=Path(a.run);run.mkdir(parents=True,exist_ok=True)
    if a.stage=='resume-check':
        import tensorflow as tf
        import gc
        from .protocol import resume_fixture
        probe=resume_fixture(options,run);folder=run/'resume-check';folder.mkdir(exist_ok=False)
        for name,extra in [('full',{}),('part',{'stop_after_steps':1}),('resumed',{'resume_checkpoint':str(folder/'part/checkpoints/last-tf.zip')})]:
            tf.keras.backend.clear_session();gc.collect();train(dict(probe,**extra),folder/name)
        with tempfile.TemporaryDirectory() as td:
            for name in ('full','resumed'):
                destination=Path(td)/name;destination.mkdir()
                with zipfile.ZipFile(folder/name/'checkpoints/last-tf.zip') as archive:archive.extractall(destination)
            one=tf.train.load_checkpoint(str(Path(td)/'full/state'));two=tf.train.load_checkpoint(str(Path(td)/'resumed/state'))
            keys=one.get_variable_to_shape_map()
            if keys!=two.get_variable_to_shape_map():raise ValueError('TF continuation checkpoint keys/shapes differ')
            maximum=0.
            for key in keys:
                if key=='_CHECKPOINTABLE_OBJECT_GRAPH':continue
                a1=one.get_tensor(key);a2=two.get_tensor(key)
                if not np.allclose(a1,a2,rtol=1e-6,atol=1e-7):raise ValueError('TF continuation mismatch: '+key)
                if np.size(a1):maximum=max(maximum,float(np.max(np.abs(a1-a2))))
        write(run/'resume_probe.json',dict(passed=True,scope='native TensorFlow mirrored model;4 real-parent crop fixture;uninterrupted vs split',
              rtol=1e-6,atol=1e-7,max_absolute_tensor_difference=maximum))
    elif a.stage=='pilot':
        pilot=run/'pilot';pilot.mkdir(exist_ok=False);train(options,pilot,True)
    elif a.stage=='train':
        if not (run/'pilot_passed.json').is_file():raise RuntimeError('Real two-GPU pilot required')
        train(options,run)
    else:
        from .protocol import require_complete
        require_complete(json.loads((run/'training_summary.json').read_text()),{'codec':options['epochs']})
        evaluate(options,run)


if __name__=='__main__':main()
