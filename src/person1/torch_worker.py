"""Real author-model updates with replayable data and strict chunk continuation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .protocol import Images, fingerprint


def validate_parallel_environment(options):
    if (options.get('data_parallel',True) and options.get('require_two_gpus',True)
            and os.environ.get('CUDA_LAUNCH_BLOCKING')=='1'):
        raise RuntimeError('CUDA_LAUNCH_BLOCKING=1 is not allowed for the two-GPU collective pilot; use phase logs and stack dumps')


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def load(path, device='cpu'):
    try:return torch.load(path,map_location=device,weights_only=False)
    except TypeError:return torch.load(path,map_location=device)


def build(options, device):
    name=options['algorithm']
    if name=='deepjscc':
        from .paper_deepjscc import DeepJSCC
        return DeepJSCC(options).to(device)
    if name=='isec':
        from .isec import build as factory
    elif name=='ntscc_plus':
        from .ntscc import build as factory
    else:raise ValueError(name)
    return factory(options,device,options['source_dir'])


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]])


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))


def stage_model(model, stage):
    if hasattr(model,'stage'):
        model.stage=stage
        for module in (model.encoder,model.decoder):
            for p in module.parameters():p.requires_grad_(stage=='jscc');p.grad=None
        for name,p in model.denoiser.named_parameters():
            p.requires_grad_(stage=='denoiser' and not name.startswith('running_sd.'));p.grad=None


def verify_replica_reduction(actual, shards, loss_gradients):
    """Check actual pre-reduction gradients; no independent-graph tolerance."""
    if not actual:raise AssertionError('No native parameter gradients captured')
    if set(shards)!={0,1} or set(loss_gradients)!={0,1}:
        raise AssertionError('Both replica gradients and loss coefficients are required')
    for coefficient in loss_gradients.values():
        if not torch.equal(coefficient,torch.full_like(coefficient,.5)):
            raise AssertionError('Replica loss must receive exact global-mean coefficient0.5')
    if set(shards[0])!=set(shards[1]) or set(actual)-set(shards[0]):
        raise AssertionError('Replica/root gradient parameter coverage differs')
    maximum=0.
    for name in shards[0]:
        gradient=actual.get(name);left=shards[0][name];right=shards[1][name]
        if gradient is None:
            if left is not None or right is not None:raise AssertionError('A used replica gradient was lost: '+name)
            continue
        # Broadcast backward may materialize zero for a native unused parameter.
        expected=torch.zeros_like(gradient) if left is None else left.to(gradient.device)
        if right is not None:expected=expected+right.to(gradient.device)
        if not torch.isfinite(expected).all() or not torch.isfinite(gradient).all():
            raise AssertionError('Nonfinite captured gradient: '+name)
        delta=float((gradient-expected).abs().max())
        maximum=max(maximum,delta)
        if not torch.equal(gradient,expected):
            raise AssertionError('Actual replica reduction differs at %s; maxdiff=%s'%(name,delta))
    return maximum


def capture_parameter_gradient(gradient,captured,index,key):
    # Autograd may call a broadcast-tensor hook with None for native parameters
    # unused by this objective (e.g. entropy auxiliary parameters).
    if key in captured[index]:raise AssertionError('Duplicate replica parameter gradient: '+key)
    captured[index][key]=None if gradient is None else gradient.detach().clone()


def capture_replica_loss(replica,ordinal,coefficients):
    # Module replication shallow-copies hook registries. Isolate this registry
    # so an ordinal-specific hook cannot run on the other replica or base model.
    replica._forward_hooks=replica._forward_hooks.copy()
    def capture_output(module,args,output):
        def loss_coefficient(gradient):
            if gradient.device.type=='cuda' and gradient.device.index!=ordinal:
                raise AssertionError('Loss coefficient captured from wrong GPU')
            if ordinal in coefficients:raise AssertionError('Duplicate replica loss coefficient')
            coefficients[ordinal]=gradient.detach().clone()
        output['native_loss'].register_hook(loss_coefficient)
    replica.register_forward_hook(capture_output)


def independent_gradient_probe(model,options,run):
    """Retain the qualified DeepJSCC probe for its native global RGB-MSE loss."""
    import copy
    probe=copy.deepcopy(model);probe.train();probe.zero_grad()
    data=Images(options['manifest'],'train',options['crop'],options.get('horizontal_flip',False))
    image=torch.from_numpy(np.stack([data.array(i,0,options['seed']) for i in range(2)])).permute(0,3,1,2).contiguous().cuda(0)
    references=[copy.deepcopy(probe).to('cuda:'+str(i)) for i in range(2)]
    before=rng_state()
    try:
        out=torch.nn.DataParallel(probe,[0,1])(image)
        probe.loss(out,{'image':image}).backward()
        actual={n:p.grad.detach().cpu().clone() for n,p in probe.named_parameters() if p.grad is not None}
        restore_rng(before)
        for i,reference in enumerate(references):
            part=image[i:i+1].to('cuda:'+str(i))
            with torch.cuda.device(i):reference.loss(reference(part),{'image':part}).mul(.5).backward()
        parameters=[dict(r.named_parameters()) for r in references];maximum=0.
        for name,gradient in actual.items():
            expected=parameters[0][name].grad.detach().cpu()+parameters[1][name].grad.detach().cpu()
            if not torch.allclose(gradient,expected,rtol=1e-4,atol=1e-5):
                raise AssertionError('Independent DeepJSCC gradient differs: '+name)
            maximum=max(maximum,float((gradient-expected).abs().max()))
        write(Path(run)/'gradient_aggregation.json',dict(passed=True,
            scope='DeepJSCC native global RGB-MSE: actualDP gradients versus weighted independent replicas',
            parameter_gradients=len(actual),max_absolute_difference=maximum,rtol=1e-4,atol=1e-5,optimizer_updates=0))
    finally:restore_rng(before)


def gradient_probe(model, options, run):
    """Verify actual native replica weighting/reduction and independent forwards."""
    import copy
    if options['algorithm']=='deepjscc':return independent_gradient_probe(model,options,run)
    if next(model.parameters()).device.type!='cuda':
        raise RuntimeError('Gradient aggregation qualification requires two real GPUs')
    data=Images(options['manifest'],'train',options['crop'],options.get('horizontal_flip',False))
    image=torch.from_numpy(np.stack([data.array(i,0,options['seed']) for i in range(2)])).permute(0,3,1,2).contiguous().cuda(0)
    random_state=rng_state();previous=getattr(model,'stage',None);records=[]
    try:
        for stage in (['jscc','denoiser'] if previous is not None else ['codec']):
            probe=copy.deepcopy(model);stage_model(probe,stage);probe.train();probe.zero_grad()
            if stage=='denoiser':probe.encoder.eval();probe.decoder.eval()
            references=[copy.deepcopy(probe).to('cuda:'+str(i)) for i in range(2)]
            captured={0:{},1:{}};coefficients={}
            class CapturingParallel(torch.nn.DataParallel):
                def replicate(self,module,device_ids):
                    replicas=super().replicate(module,device_ids)
                    for ordinal,replica in enumerate(replicas):
                        for prefix,child in replica.named_modules():
                            for local,parameter in getattr(child,'_former_parameters',{}).items():
                                if not parameter.requires_grad:continue
                                name=(prefix+'.' if prefix else '')+local
                                def capture(gradient,key=name,index=ordinal):
                                    capture_parameter_gradient(gradient,captured,index,key)
                                parameter.register_hook(capture)
                        capture_replica_loss(replica,ordinal,coefficients)
                    return replicas
            before=rng_state()
            output=CapturingParallel(probe,[0,1])(image)
            loss=probe.loss(output,{'image':image})
            actual_loss=loss.detach().clone();reconstruction=output['reconstruction'].detach().cpu().clone()
            loss.backward()
            actual={n:p.grad.detach().clone() for n,p in probe.named_parameters() if p.grad is not None}
            maximum=verify_replica_reduction(actual,captured,coefficients)
            restore_rng(before);reference_losses=[];reference_images=[]
            for device,reference in enumerate(references):
                part=image[device:device+1].to('cuda:'+str(device))
                with torch.cuda.device(device):
                    out=reference(part);local=reference.loss(out,{'image':part})
                    reference_losses.append(local.detach().to('cuda:0'))
                    reference_images.append(out['reconstruction'].detach().cpu())
                    local.mul(.5).backward()
            expected_loss=torch.stack(reference_losses).mean()
            if not torch.equal(actual_loss,expected_loss) or not torch.equal(reconstruction,torch.cat(reference_images)):
                raise AssertionError('Independent native forward/loss differs from actual replicas')
            independent_max=0.;independent_pointwise_failures=0
            one=dict(references[0].named_parameters());two=dict(references[1].named_parameters())
            for name,gradient in actual.items():
                expected=torch.zeros_like(gradient) if one[name].grad is None else one[name].grad.detach().to('cuda:0')
                if two[name].grad is not None:expected=expected+two[name].grad.detach().to('cuda:0')
                independent_max=max(independent_max,float((gradient-expected).abs().max()))
                independent_pointwise_failures+=int(not torch.allclose(gradient,expected,rtol=1e-4,atol=1e-5))
            records.append(dict(stage=stage,parameter_gradients=len(actual),max_absolute_difference=maximum,
                exact_loss_coefficients=True,independent_forward_exact=True,
                independent_backward_maxdiff=independent_max,
                independent_backward_pointwise_failures=independent_pointwise_failures,
                independent_backward_scope='different backward execution graph; diagnostic only; original pointwise test retained and reported'))
            del probe,references,output,actual,captured,coefficients
    finally:restore_rng(random_state)
    write(Path(run)/'gradient_aggregation.json',dict(passed=True,devices=['cuda:0','cuda:1'],
        scope='actual replica native-loss coefficient0.5 and exact float32 pre-reduction gradient sum; independent forwards exact',
        microbatch_per_device=1,aggregation_comparison='torch.equal; no fitted tolerance',stages=records,optimizer_updates=0))


def train(model, options, run, pilot=False):
    run=Path(run);(run/'checkpoints').mkdir(parents=True,exist_ok=True)
    seed=int(options['seed']);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    # Determinism is required for the continuation acceptance check. Native
    # unsupported nondeterministic kernels remain disclosed rather than hidden.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dataset=Images(options['manifest'],'train',options['crop'],options.get('horizontal_flip',False))
    development=Images(options['manifest'],'val',options['crop'])
    readers=ThreadPoolExecutor(max_workers=max(1,int(options.get('workers',4))))
    contract=fingerprint(options,dataset.manifest['identity'])
    stages=['jscc','denoiser'] if hasattr(model,'stage') else ['codec']
    targets={s:int(options['epochs']) for s in stages}
    state=dict(contract=contract,stage_index=0,epoch=0,offset=0,step=0,images_seen=0,
               completed_epochs={s:0 for s in stages},optimizer_states={},scheduler_states={},
               auxiliary_states={},best={s:None for s in stages},elapsed_seconds=0.)
    checkpoint=options.get('resume_checkpoint') if not pilot else None
    if checkpoint:
        saved=load(checkpoint)
        if saved['progress']['contract']!=contract:raise ValueError('Training contract changed; resume rejected')
        model.load_state_dict(saved['state_dict'],strict=True)
        state=saved['progress'];restore_rng(saved['rng'])
        write(run/'resume_verified.json',dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            contract=contract,completed_epochs=state['completed_epochs'],offset=state['offset'],strict_reload=True))
    device=next(model.parameters()).device
    multi=device.type=='cuda' and options.get('data_parallel',True)
    if multi and (torch.cuda.device_count()!=2 or options['batch_size']<2):
        raise RuntimeError('Person1 requires exactly two visible CUDA GPUs and global batch>=2')
    parallel=torch.nn.DataParallel(model,[0,1]) if multi else model
    observed=set();hooks=[]
    for name,module in model.named_modules():
        if name in {'encoder','core.ntc.g_a0'}:
            def observe(module,args):
                if args:observed.add(str(args[0].device))
            hooks.append(module.register_forward_pre_hook(observe))
    started=time.monotonic();deadline=started+float(options['max_seconds']);last_save=started
    pilot_steps={s:0 for s in stages};optimizer=scheduler=aux=None;stage=None

    def save():
        if optimizer is not None and stage is not None:
            state['optimizer_states'][stage]=optimizer.state_dict()
            state['scheduler_states'][stage]=scheduler.state_dict() if scheduler else None
            state['auxiliary_states'][stage]=aux.state_dict() if aux else None
        state['elapsed_seconds']+=time.monotonic()-save.last_clock;save.last_clock=time.monotonic()
        path=run/'checkpoints/last.pt';temp=path.with_suffix('.tmp')
        torch.save(dict(state_dict=model.state_dict(),progress=state,rng=rng_state(),options=options,
                        resume_supported=True),temp);temp.replace(path)
        write(run/'progress.json',dict(**{k:v for k,v in state.items() if not k.endswith('_states')},
                                      target_epochs=targets,status='partial'))
    save.last_clock=started
    stopped=False
    for stage_index in range(state['stage_index'],len(stages)):
        stage=stages[stage_index];stage_model(model,stage)
        optimizer=model.optimizer();scheduler=model.scheduler(optimizer) if hasattr(model,'scheduler') else None
        aux=model.auxiliary_optimizer() if hasattr(model,'auxiliary_optimizer') else None
        if stage in state['optimizer_states']:
            optimizer.load_state_dict(state['optimizer_states'][stage])
            if scheduler and state['scheduler_states'].get(stage):scheduler.load_state_dict(state['scheduler_states'][stage])
            if aux and state['auxiliary_states'].get(stage):aux.load_state_dict(state['auxiliary_states'][stage])
        while state['epoch']<targets[stage]:
            epoch=state['epoch'];order=dataset.order(epoch,seed)
            while state['offset']<len(order):
                if time.monotonic()>=deadline or (options.get('stop_after_steps') and state['step']>=options['stop_after_steps']):
                    stopped=True;break
                selected=order[state['offset']:state['offset']+options['batch_size']]
                data_started=time.monotonic()
                array=np.stack(list(readers.map(lambda i:dataset.array(i,epoch,seed),selected)))
                image=torch.from_numpy(array).permute(0,3,1,2).to(device)
                data_seconds=time.monotonic()-data_started
                parallel.train()
                if stage=='denoiser':model.encoder.eval();model.decoder.eval()
                optimizer.zero_grad()
                if aux:aux.zero_grad()
                step_started=time.monotonic()
                if pilot:print('Pilot forward start: '+stage+' '+str(tuple(image.shape)),flush=True)
                output=parallel(image);loss=model.loss(output,{'image':image})
                if pilot:print('Pilot forward complete: '+stage,flush=True)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                loss.backward()
                if pilot:print('Pilot backward complete: '+stage,flush=True)
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite gradients')
                if options.get('clip_norm',0)>0:torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm'])
                optimizer.step()
                if pilot:print('Pilot optimizer complete: '+stage,flush=True)
                if aux:
                    auxiliary=model.aux_loss();auxiliary.backward();aux.step()
                if scheduler and getattr(model,'scheduler_per_step',False):scheduler.step()
                if device.type=='cuda':torch.cuda.synchronize(device)
                state['step']+=1;state['offset']+=len(selected);state['images_seen']+=len(selected)
                event=dict(stage=stage,epoch=epoch+1,step=state['step'],sample_offset=state['offset'],
                           images_seen=state['images_seen'],loss=float(loss.detach()),
                           data_seconds=data_seconds,step_seconds=time.monotonic()-step_started)
                with (run/'training.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                if state['step']<=2 or state['step']%25==0:print(json.dumps(event),flush=True)
                if time.monotonic()-last_save>=300:save();last_save=time.monotonic()
                if pilot:
                    pilot_steps[stage]+=1
                    if pilot_steps[stage]>=2:break
            if stopped:break
            if pilot:break
            if state['offset']!=len(order):raise RuntimeError('Incomplete epoch cannot advance')
            if scheduler and not getattr(model,'scheduler_per_step',False):scheduler.step()
            state['epoch']+=1;state['offset']=0;state['completed_epochs'][stage]=state['epoch']
            # Fixed development parents; no Kodak selection. Preserve channel RNG
            # so a chunk ending here and an uninterrupted run have identical state.
            before=rng_state();parallel.eval();values=[]
            for i in range(min(len(development),16)):
                torch.manual_seed(seed+900000+i)
                x=torch.from_numpy(development.array(i)).permute(2,0,1)[None].to(device)
                with torch.no_grad():o=model(x);values.append(float(model.loss(o,{'image':x})))
            restore_rng(before)
            score=float(np.mean(values));previous=state['best'][stage]
            improved=previous is None or score<previous
            if improved:state['best'][stage]=score
            save()
            if improved:
                import shutil
                shutil.copy2(run/'checkpoints/last.pt',run/('checkpoints/best-'+stage+'.pt'))
            print(json.dumps(dict(epoch_completed=state['epoch'],stage=stage,parents=len(order),validation=score)),flush=True)
        if stopped:break
        if not pilot:
            state['stage_index']=stage_index+1;state['epoch']=0;state['offset']=0
        save()
    save()
    for hook in hooks:hook.remove()
    complete=all(state['completed_epochs'][s]==targets[s] for s in stages)
    if pilot and any(pilot_steps[s]!=2 for s in stages):raise RuntimeError('Two updates per native stage were not completed')
    if multi and not {'cuda:0','cuda:1'}.issubset(observed):raise RuntimeError('Both GPUs did not execute encoder work')
    model.load_state_dict(load(run/'checkpoints/last.pt',device)['state_dict'],strict=True)
    summary=dict(status='pilot_completed' if pilot else 'training_completed' if complete else 'partial',
                 completed_epochs=state['completed_epochs'],target_epochs=targets,steps=state['step'],
                 images_seen=state['images_seen'],elapsed_seconds=state['elapsed_seconds'],
                 strict_reload=True,observed_encoder_devices=sorted(observed),contract=contract,
                 resume_supported=True,partial_epoch_offset=state['offset'],paper_convergence_reproduced=False)
    readers.shutdown(wait=True)
    write(run/'training_summary.json',summary);print(json.dumps(summary),flush=True)
    return summary


def predict(model, image, options):
    h,w=image.shape[-2:]
    ah,aw=(128,64) if options['algorithm']=='ntscc_plus' else (4,4)
    ph,pw=(-h)%ah,(-w)%aw;l=pw//2;t=ph//2
    padded=F.pad(image,(l,pw-l,t,ph-t),value=0)
    out=model(padded)
    out['reconstruction']=out['reconstruction'][...,t:t+h,l:l+w]
    out['geometry']=dict(original=[h,w],padded=list(padded.shape[-2:]),padding=[l,pw-l,t,ph-t])
    return out


def profile_training(model,options,run):
    """Profile one real-data microbatch on one replica, without an update.

    Dispatch modes do not reliably enter DataParallel's Python threads. This
    explicit measurement avoids publishing a backward-only count as a full step.
    """
    device=next(model.parameters()).device
    data=Images(options['manifest'],'train',options['crop'],options.get('horizontal_flip',False))
    batch=max(1,options['batch_size']//2)
    image=torch.from_numpy(np.stack([data.array(i,0,options['seed']) for i in range(batch)])).permute(0,3,1,2).to(device)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};random_state=rng_state()
    previous_stage=getattr(model,'stage',None);modes={name:module.training for name,module in model.named_modules()}
    for stage in (['jscc','denoiser'] if previous_stage is not None else ['codec']):
        stage_model(model,stage);model.train()
        if stage=='denoiser':model.encoder.eval();model.decoder.eval()
        model.zero_grad();counter=None;value=None;error=None
        def measured():
            out=model(image);model.loss(out,{'image':image}).backward()
            if hasattr(model,'aux_loss'):model.aux_loss().backward()
        try:
            try:
                from torch.utils.flop_counter import FlopCounterMode
                counter=FlopCounterMode(display=False)
                with counter:measured()
                value=counter.get_total_flops()
            except ImportError:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as counter:measured()
                value=sum(getattr(event,'flops',0) for event in counter.key_averages())
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            model.load_state_dict(state,strict=True);model.zero_grad();restore_rng(random_state)
        write(Path(run)/('training_profile-'+stage+'.json'),dict(measured_microbatch=batch,
              training_global_batch=options['batch_size'],input_shape=list(image.shape),precision=str(image.dtype),
              registered_forward_backward_auxiliary_flops=value if value and not error else None,
              registered_flops_per_image=value/batch if value and not error else None,error=error,
              coverage='one replica, registered operators only; optimizer arithmetic and unsupported kernels excluded;2FLOPs/MAC',
              complete_training_flops=False,weights_restored=True,optimizer_updates=0))
    if previous_stage is not None:stage_model(model,previous_stage)
    for name,module in model.named_modules():module.training=modes[name]


def evaluate(model, options, run, pilot=False):
    from PIL import Image
    import io
    run=Path(run);folder=run/'predictions';folder.mkdir(parents=True,exist_ok=False)
    device=next(model.parameters()).device
    dataset=Images(options['manifest'],'val' if pilot else 'test',options['crop'])
    if hasattr(model,'stage'):model.stage='denoiser'
    model.eval();rows=[]
    for i in range(1 if pilot else len(dataset)):
        array=dataset.array(i);x=torch.from_numpy(array).permute(2,0,1)[None].to(device)
        for trial in range(1 if pilot else options['trials']):
            seed=options['seed']+10000*i+trial;torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter()
            with torch.no_grad():out=predict(model,x,options)
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started;y=out['reconstruction']
            if y.shape!=x.shape or not torch.isfinite(y).all():raise ValueError('Invalid reconstruction geometry/values')
            path=folder/('%03d-%02d.npy'%(i,trial));np.save(path,y[0].detach().cpu().permute(1,2,0).numpy())
            row=dict(sample=i,trial=trial,seed=seed,path=path.name,seconds=elapsed,
                     complex_channel_uses=float(out['complex_uses'].sum()),geometry=out['geometry'])
            if 'indexes' in out:
                values=out['indexes'].detach().cpu().numpy().astype(np.uint8)
                shape=out['geometry']['padded'];values=values.reshape(shape[0]//16,shape[1]//16)
                stream=io.BytesIO();Image.fromarray(values).save(stream,format='PNG');bits=len(stream.getvalue())*8
                (folder/('%03d-%02d-side.png'%(i,trial))).write_bytes(stream.getvalue())
                row.update(serialized_side_bits=bits,assumed_side_complex_symbols=int(np.ceil(bits*3/8)),
                           side_reliability='author assumes correct LDPC2/3+16QAM delivery; no released decoder',
                           complete_link_certified=False)
            rows.append(row);write(run/'predictions.json',dict(rows=rows,pilot=pilot))
            print(json.dumps(row),flush=True)
        if i==0:
            profile(model,x,options,run)
    return rows


def profile(model,image,options,run):
    device=image.device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    def call():
        with torch.no_grad():return predict(model,image,options)
    for _ in range(3):call()
    timings=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        sync();tick=time.perf_counter();call();sync();timings.append(time.perf_counter()-tick)
    counter=None;error=None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter=FlopCounterMode(display=False)
        with counter:call();sync()
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    components={};handles=[];ticks={}
    for name,module in model.named_modules():
        if name in {'encoder','decoder','denoiser','core.channel'}:
            def before(mod,args,key=name):sync();ticks[key]=time.perf_counter()
            def after(mod,args,out,key=name):
                sync();components.setdefault(key,[]).append(time.perf_counter()-ticks[key])
            handles += [module.register_forward_pre_hook(before),module.register_forward_hook(after)]
    try:call();sync()
    finally:
        for handle in handles:handle.remove()
    write(Path(run)/'profile.json',dict(input_shape=list(image.shape),batch_size=1,precision=str(image.dtype),
        device=str(device),warmup=3,repeats=10,latency_p50_seconds=float(np.median(timings)),
        latency_p95_seconds=float(np.quantile(timings,.95)),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        registered_kernel_flops=counter.get_total_flops() if counter and not error else None,
        flop_error=error,flops_coverage='registered PyTorch kernels;2FLOPs/MAC;partial, no claimed total',
        component_instrumented_seconds={k:dict(total=sum(v),calls=len(v)) for k,v in components.items()},
        component_timing_note='one synchronized instrumented pass; hook overhead included; absent native components unavailable',
        includes_iterative_latent_gradients=bool(getattr(model,'needs_grad_evaluation',False))))


def main():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(180,repeat=True)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','train','evaluate','resume-check'])
    p.add_argument('--config',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    options=json.loads(Path(a.config).read_text());run=Path(a.run);run.mkdir(parents=True,exist_ok=True)
    validate_parallel_environment(options)
    if options['algorithm'] in ('isec','ntscc_plus'):
        # Native convolution mathematics is unchanged; require replayable kernels.
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if options.get('require_two_gpus',True) and (device=='cpu' or torch.cuda.device_count()!=2):
        raise RuntimeError('Two-T4 qualification is required')
    torch.manual_seed(options['seed']);np.random.seed(options['seed']);random.seed(options['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(options['seed'])
    model=build(options,device);write(run/'model.json',model.metadata)
    if a.stage=='resume-check':
        from .protocol import resume_fixture
        probe=resume_fixture(options,run)
        folder=run/'resume-check';folder.mkdir(exist_ok=False)
        def fresh():
            import gc
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            torch.manual_seed(probe['seed']);np.random.seed(probe['seed']);random.seed(probe['seed'])
            if torch.cuda.is_available():torch.cuda.manual_seed_all(probe['seed'])
            return build(probe,device)
        del model
        full=fresh();train(full,probe,folder/'full');del full
        interrupted=fresh();train(interrupted,dict(probe,stop_after_steps=3 if options['algorithm']=='isec' else 1),folder/'part');del interrupted
        resumed=fresh();train(resumed,dict(probe,resume_checkpoint=str(folder/'part/checkpoints/last.pt')),folder/'resumed');del resumed
        one=load(folder/'full/checkpoints/last.pt');two=load(folder/'resumed/checkpoints/last.pt');maximum=0.
        def compare(a,b,path='state'):
            nonlocal maximum
            if torch.is_tensor(a):
                if a.shape!=b.shape or not torch.allclose(a,b,rtol=1e-6,atol=1e-7):
                    delta=float((a.double()-b.double()).abs().max()) if a.shape==b.shape and a.numel() else None
                    raise ValueError('Continuation tensor mismatch at %s; maxdiff=%s'%(path,delta))
                if a.numel():maximum=max(maximum,float((a.double()-b.double()).abs().max()))
            elif isinstance(a,dict):
                if a.keys()!=b.keys():raise ValueError('Continuation state keys differ')
                for k in a:compare(a[k],b[k],path+'.'+str(k))
            elif isinstance(a,(list,tuple)):
                if len(a)!=len(b):raise ValueError('Continuation state lengths differ')
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
            elif a!=b:raise ValueError('Continuation scalar state differs')
        compare(one['state_dict'],two['state_dict'])
        for key in ('optimizer_states','scheduler_states','auxiliary_states','completed_epochs','step','images_seen'):
            compare(one['progress'][key],two['progress'][key])
        write(run/'resume_probe.json',dict(passed=True,scope='selected model implementation;4 real-parent crop fixture;uninterrupted vs split',
             rtol=1e-6,atol=1e-7,max_absolute_tensor_difference=maximum))
    elif a.stage=='pilot':
        target=run/'pilot';target.mkdir(exist_ok=False)
        native_stats={n:p.detach().cpu().clone() for n,p in model.named_parameters() if '.running_sd.' in n}
        train(model,options,target,True)
        if native_stats:
            after=dict(model.named_parameters())
            changed=[n for n,before in native_stats.items() if not torch.equal(before,after[n].detach().cpu())]
            if not changed:raise RuntimeError('Native denoiser running statistics did not persist through DataParallel')
            if any(after[n].requires_grad for n in native_stats):raise RuntimeError('Author running statistics became trainable')
            saved=load(target/'checkpoints/last.pt')['state_dict']
            if any(not torch.equal(saved[n].cpu(),after[n].detach().cpu()) for n in native_stats):
                raise RuntimeError('Denoiser running statistics missing from strict checkpoint')
            write(target/'running_statistics_probe.json',dict(passed=True,changed=len(changed),total=len(native_stats),
                  checkpoint_equal=True,author_nontrainable=True,scope='replica0 native running statistics'))
        gradient_probe(model,options,target)
        profile_training(model,options,target)
        model.eval()
        if hasattr(model,'stage'):model.stage='denoiser'
        geometry=[]
        for h,w in ((512,768),(768,512),(513,769)):
            x=torch.rand(1,3,h,w,device=device)
            with torch.no_grad():out=predict(model,x,options)
            if out['reconstruction'].shape!=x.shape or not torch.isfinite(out['reconstruction']).all():
                raise ValueError('Native/odd portrait-landscape geometry pilot failed')
            geometry.append(out['geometry'])
        write(target/'geometry_probe.json',dict(passed=True,synthetic=True,images=geometry))
        evaluate(model,options,target,True)
        write(run/'pilot_passed.json',dict(real_data=True,stages='two updates each',strict_reload=True,
              device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    elif a.stage=='train':
        if not (run/'pilot_passed.json').is_file():raise RuntimeError('Pilot must pass first')
        train(model,options,run)
    else:
        saved=load(run/'checkpoints/last.pt',device);model.load_state_dict(saved['state_dict'],strict=True)
        from .protocol import require_complete
        require_complete(json.loads((run/'training_summary.json').read_text()),
                         {s:options['epochs'] for s in saved['progress']['completed_epochs']})
        evaluate(model,options,run)


if __name__=='__main__':main()
