"""One model per saved Kaggle version; shared collector-compatible artifacts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from .protocol import options, Images, DATASET_IDENTITY, require_complete
from src.runtime import write_json,environment,new_session,export,stage
from src.dataset import discover,freeze,digest


def command(argv,log,timeout,env=None):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as stream:
        stream.write('\nCOMMAND '+repr(argv)+'\n');stream.flush()
        process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                 env=env,bufsize=1,start_new_session=sys.platform!='win32')
        def drain():
            for line in process.stdout:print(line,end='',flush=True);stream.write(line);stream.flush()
        reader=threading.Thread(target=drain,daemon=True);reader.start()
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if sys.platform!='win32':os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            reader.join(timeout=10);raise TimeoutError('Stage timed out; evidence retained: '+str(log))
        reader.join(timeout=10)
    if code:raise RuntimeError('Subprocess failed (%s); see %s'%(code,log))


def setup(root,name,mode,delivery_id,seconds):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError('Select Kaggle T4x2; two CUDA devices required')
    if any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('Requested comparison hardware is two TeslaT4 GPUs')
    root=Path(root);session=new_session('/kaggle/working/team_results','1');run=session/name;run.mkdir()
    write_json(session/'delivery_identity.json',dict(delivery_run_id=delivery_id,team='1',source_revision=environment(root)['revision']))
    write_json(run/'environment.json',environment(root))
    write_json(run/'person1_run.json',dict(algorithm=name,mode=mode,teammate='1',run_id=session.name))
    manifest=session/'common_dataset.json'
    with stage(run,'dataset'):
        frozen=freeze(discover('/kaggle/input'),manifest,workers=2)
        if frozen['identity']!=DATASET_IDENTITY:raise ValueError('Shared frozen dataset identity differs')
        write_json(run/'dataset_identity.json',{k:frozen[k] for k in ('identity','counts','protocol','duplicate_policy')})
    config=options(name,manifest);config.update(teammate='1',run_id=session.name,models_root='/kaggle/working/models',
                  max_seconds=seconds,require_two_gpus=True,source_revision=environment(root)['revision'])
    overlay=root/'private_runtime_overlay.json'
    if overlay.is_file():
        record=json.loads(overlay.read_text())
        for relative,expected in record['files'].items():
            path=(root/relative).resolve();path.relative_to(root.resolve())
            if digest(path)!=expected:raise ValueError('Private runtime file hash differs: '+relative)
        config['private_runtime_overlay_sha256']=record['sha256']
        write_json(run/'private_runtime_overlay.json',record)
    from src.sources import fetch_source,source_path
    source_name={'adjscc':'adjscc_author','isec':'isec','ntscc_plus':'ntscc_plus'}.get(name)
    if source_name:
        receipt=fetch_source(source_name);config['source_dir']=str(source_path(source_name));write_json(run/'source.json',receipt)
    else:
        write_json(run/'source.json',dict(repository='paper reproduction',commit=None,
                   paper='https://arxiv.org/abs/1809.01733v4',figure=2))
    files=list((root/'src/person1').glob('*.py'))
    config['runner_sha256']={p.name:digest(p) for p in sorted(files)}
    resumes=[]
    for path in Path('/kaggle/input').rglob('resume_receipt.json'):
        receipt=json.loads(path.read_text())
        if receipt.get('algorithm')==name:
            checkpoint=path.parent/receipt['checkpoint']
            if digest(checkpoint)!=receipt['sha256']:raise ValueError('Continuation checkpoint hash mismatch')
            resumes.append(checkpoint)
    if len(resumes)>1:raise ValueError('Multiple continuation checkpoints attached for '+name)
    if resumes:config['resume_checkpoint']=str(resumes[0])
    write_json(run/'requested_config.json',config);write_json(run/'resolved_config.json',config)
    from .environments import prepare
    with stage(run,'prepare'):
        python,env=prepare(name,root,run/'dependencies.log',command)
        if name in ('isec','ntscc_plus'):
            # Blocking CUDA launches can deadlock multi-device collectives.
            # Use stack dumps and phase logs for the two-GPU pilot instead.
            env.pop('CUDA_LAUNCH_BLOCKING',None)
            env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        # Required author interpreter and CUDA identities are written before work.
        probe=('import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("GPU"))'
               if name=='adjscc' else 'import torch;print(torch.__version__,torch.version.cuda);print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])')
        command([python,'-c',probe],run/'dependency_probe.log',120,env)
    return dict(root=str(root),session=str(session),run=str(run),config=config,python=python,env=env,mode=mode)


def invoke(context,action):
    run=Path(context['run']);name=context['config']['algorithm']
    module='src.person1.tf_worker' if name=='adjscc' else 'src.person1.torch_worker'
    with stage(run,action):
        command([context['python'],'-u','-m',module,action,'--config',str(run/'resolved_config.json'),'--run',str(run)],
                run/(action+'_cell.log'),18000 if action=='train' else 600 if action=='pilot' and name=='adjscc' else 1200 if action=='pilot' and name=='isec' else 7200,context['env'])


def score(context):
    import torch
    from PIL import Image
    from src.evaluate import image_quality
    run=Path(context['run']);config=context['config'];dataset=Images(config['manifest'],'test',config['crop'])
    predictions=json.loads((run/'predictions.json').read_text())['rows'];rows=[]
    folder=run/'metrics';folder.mkdir();images=run/'reconstructions';images.mkdir()
    for item in predictions:
        original=dataset.array(item['sample']);recon=np.load(run/'predictions'/item['path'])
        a=torch.from_numpy(original).permute(2,0,1)[None];b=torch.from_numpy(recon).permute(2,0,1)[None]
        measured=image_quality(a,b,True)
        for key in ('psnr_db','ssim','ms_ssim','lpips','mse'):
            if measured.get(key) is None or not np.isfinite(measured[key]):raise ValueError('Required metric unavailable: '+key)
        h,w=original.shape[:2];main=item['complex_channel_uses'];side=item.get('assumed_side_complex_symbols',0)
        rows.append(dict(algorithm=config['algorithm'],sample=item['sample'],trial=item['trial'],seed=item['seed'],
             height=h,width=w,snr_db=10,channel='awgn',complex_channel_uses=main,
             assumed_reliable_side_symbols=side,serialized_side_bits=item.get('serialized_side_bits'),
             cbr_complex_per_rgb_sample=(main+side)/(3*h*w),complete_link_certified=config['algorithm']!='ntscc_plus',
             one_pass_seconds=item['seconds'],**measured))
        Image.fromarray((np.clip(recon,0,1)*255).round().astype(np.uint8)).save(images/('%03d-%02d.png'%(item['sample'],item['trial'])))
    expected=24*config['trials']
    if len(rows)!=expected:raise ValueError('Incomplete Kodak trials: %s/%s'%(len(rows),expected))
    with (folder/'per_image.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(folder/'per_image.json',dict(rows=rows,failures=[]))
    summary=dict(requested_trials=expected,successful_trials=expected,failed_trials=0,
                 split='Kodak24 test',pilot=False,complete_link_certified=config['algorithm']!='ntscc_plus',
                 ci_method='bootstrap24 image means;2000 resamples;seed22')
    if (run/'profile.json').is_file():summary.update(json.loads((run/'profile.json').read_text()))
    summary['training_profiles']={p.name:json.loads(p.read_text()) for p in run.glob('training_profile-*.json')}
    for key in ('psnr_db','ssim','ms_ssim','lpips','mse','cbr_complex_per_rgb_sample','one_pass_seconds'):
        means=np.array([np.mean([r[key] for r in rows if r['sample']==i]) for i in range(24)])
        rng=np.random.RandomState(22);bootstrap=means[rng.randint(0,24,(2000,24))].mean(1)
        summary[key]=float(means.mean());summary[key+'_ci95']=np.quantile(bootstrap,[.025,.975]).tolist()
    write_json(folder/'summary.json',summary)


def execute(context):
    run=Path(context['run']);config=context['config'];session=Path(context['session'])
    try:
        invoke(context,'pilot')
        invoke(context,'resume-check')
        if not json.loads((run/'resume_probe.json').read_text()).get('passed'):
            raise RuntimeError('GPU continuation qualification failed')
        import shutil
        for path in (run/'pilot').glob('training_profile-*.json'):shutil.copy2(path,run/path.name)
        if context['mode']=='pilot':
            write_json(run/'pilot_only.json',dict(status='pilot_completed',training_complete=False));return
        invoke(context,'train')
        summary=json.loads((run/'training_summary.json').read_text())
        if summary['status']!='training_completed':
            path=run/'checkpoints'/('last-tf.zip' if config['algorithm']=='adjscc' else 'last.pt')
            write_json(run/'chunk_checkpoint.json',dict(algorithm=config['algorithm'],checkpoint=str(path.relative_to(run)),
                       sha256=digest(path),status='partial',completed_epochs=summary['completed_epochs']))
            print('PARTIAL CHUNK: saved checkpoint; continuation required',flush=True);return
        require_complete(summary,{s:config['epochs'] for s in summary['completed_epochs']})
        invoke(context,'evaluate')
        with stage(run,'evaluate'):
            score(context)
        from src.sharing import package_run
        target=package_run(run,config['models_root'],'1',config['run_id'])
        write_json(run/'package_result.json',dict(produced=True,bundle_path=str(target)))
    finally:
        training=json.loads((run/'training_summary.json').read_text()) if (run/'training_summary.json').is_file() else {}
        packaged=json.loads((run/'package_result.json').read_text()).get('produced',False) if (run/'package_result.json').is_file() else False
        write_json(session/'completed_models.json',dict(models={config['algorithm']:dict(
            training_completed=training.get('status')=='training_completed',
            completed_epochs=training.get('completed_epochs',{}),evaluation_completed=packaged,model_packaged=packaged)},
            delivery='saved notebook outputs only; private checkpoint input after verified local collection'))
        export(run);export(session)
