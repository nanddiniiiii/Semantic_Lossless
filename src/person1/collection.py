"""Preserve partial weights and create immutable private continuation inputs."""
import hashlib
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from src.dataset import digest
from src.runtime import write_json
from src.sharing import MODEL_NAMES


def atomic_copy(source,target,expected=None):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    expected=expected or digest(source)
    if target.is_file() and digest(target)==expected:return
    temporary=target.with_name(target.name+'.tmp')
    shutil.copy2(source,temporary)
    if digest(temporary)!=expected:raise ValueError('Interrupted/changed checkpoint copy')
    if target.exists():target.rename(target.with_name(target.name+'.interrupted-'+uuid.uuid4().hex[:8]))
    temporary.replace(target)


def owned_dataset_exists(ref,env):
    slug=ref.split('/')[1]
    listing=subprocess.run(['kaggle','datasets','list','--mine','--search',slug,'--page-size','100','--csv'],
                           env=env,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=90)
    if listing.returncode:raise RuntimeError('Cannot verify private checkpoint inventory; no upload attempted')
    if listing.stdout.strip()=='No datasets found':return False
    lines=listing.stdout.splitlines();header=next((i for i,line in enumerate(lines) if line.lstrip('\ufeff').startswith('ref,')),None)
    if header is None:raise RuntimeError('Owned Dataset inventory header unavailable')
    rows=list(csv.DictReader(io.StringIO('\n'.join(lines[header:]).lstrip('\ufeff'))))
    if len(rows)>=100:raise RuntimeError('Private Dataset search requires pagination; refusing to infer absence')
    return any(row.get('ref')==ref for row in rows)


def preserve_chunks(root,destination,env):
    root=Path(root);destination=Path(destination)
    submitted=json.loads((destination/'submitted_version.json').read_text())
    username=submitted['kernel'].split('/')[0]
    raw=destination/'published_output'
    for marker in raw.rglob('chunk_checkpoint.json'):
        run=marker.parent;spec=json.loads(marker.read_text());name=spec['algorithm']
        checkpoint=(run/spec['checkpoint']).resolve();checkpoint.relative_to(run.resolve())
        if digest(checkpoint)!=spec['sha256']:raise ValueError('Partial checkpoint changed after retrieval')
        identity=json.loads((run/'person1_run.json').read_text());run_id=identity['run_id']
        target=(root/'Models/teammate_1_models'/MODEL_NAMES[name]/run_id/'partial').resolve()
        target.relative_to((root/'Models/teammate_1_models').resolve())
        target.mkdir(parents=True,exist_ok=True)
        atomic_copy(checkpoint,target/checkpoint.name,spec['sha256'])
        for filename in ('training_summary.json','resolved_config.json','source.json','model.json','person1_run.json','chunk_checkpoint.json'):
            if (run/filename).is_file():atomic_copy(run/filename,target/filename)
        if digest(target/checkpoint.name)!=spec['sha256']:raise ValueError('Local preserved checkpoint failed hash')
        receipt=destination/('checkpoint_dataset-'+name+'.json')
        opaque_archive=checkpoint.suffix.lower()=='.zip'
        if receipt.is_file() and json.loads(receipt.read_text()).get('status')=='ready':
            if json.loads(receipt.read_text()).get('sha256')!=spec['sha256']:raise ValueError('Continuation receipt checkpoint differs')
            if not opaque_archive or json.loads(receipt.read_text()).get('payload_filename')=='checkpoint.bin':continue
            atomic_copy(receipt,receipt.with_name(receipt.stem+'-archive-ingestion.json'))
        slug='p1-'+name.replace('_','-')+'-'+hashlib.sha256(run_id.encode()).hexdigest()[:12]
        if opaque_archive:slug+='-raw'
        ref=username+'/'+slug
        upload=root/'Results/person1_checkpoints'/slug;upload.mkdir(parents=True,exist_ok=True)
        payload_name='checkpoint.bin' if opaque_archive else checkpoint.name
        atomic_copy(checkpoint,upload/payload_name,spec['sha256'])
        write_json(upload/'resume_receipt.json',dict(algorithm=name,checkpoint=payload_name,sha256=spec['sha256'],
                   contract=json.loads((run/'training_summary.json').read_text())['contract'],run_id=run_id))
        write_json(upload/'dataset-metadata.json',dict(id=ref,title='Person1 private '+name+' checkpoint',
                   licenses=[{'name':'other'}]))
        # No --public flag: checkpoint inputs must stay private. Unique immutable
        # identities prevent replacing an older chunk used in another version.
        if not owned_dataset_exists(ref,env):
            write_json(receipt,dict(dataset=ref,status='upload_started',sha256=spec['sha256']))
            result=subprocess.run(['kaggle','datasets','create','-p',str(upload),'-t'],env=env,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=7200)
            (destination/('checkpoint_upload-'+name+'.log')).write_text((result.stdout or '')+'\n'+(result.stderr or ''),encoding='utf-8')
            if result.returncode:raise RuntimeError('Private checkpoint upload failed; inspect saved log before retry')
        status=subprocess.run(['kaggle','datasets','status',ref],env=env,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=90)
        if status.returncode or 'ready' not in status.stdout.lower():
            raise RuntimeError('Private checkpoint Dataset not ready yet; collector will retry without duplicate creation')
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest
        api=KaggleApi();api.authenticate();request=ApiGetDatasetMetadataRequest()
        request.owner_slug=username;request.dataset_slug=slug
        with api.build_kaggle_client() as client:
            view=client.datasets.dataset_api_client.get_dataset_metadata(request)
        private=view.info.is_private if view.info else None
        if private is not True:raise ValueError('Continuation Dataset privacy could not be verified')
        write_json(receipt,dict(dataset=ref,status='ready',sha256=spec['sha256'],is_private=True,payload_filename=payload_name,
                               remote_checkpoint_hash_verified=False,verification='next Kaggle loader verifies actual checkpoint bytes'))
