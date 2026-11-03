"""Embed the reviewed checkpoint I/O adapter outside immutable model sources."""
import base64
import hashlib
import json
from pathlib import Path


def prepare_checkpoint_notebook(book, module, expected_sha, chunk_seconds=5400, resource_module=None, resource_sha=None, cudnn_plan_cache_limit=None):
    book,module=Path(book),Path(module)
    data=module.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('Checkpoint adapter hash changed')
    notebook=json.loads(book.read_text(encoding='utf-8'))
    existing=notebook.get('metadata',{}).get('checkpoint_io_adapter')
    if existing:
        if (existing['sha256']==expected_sha and existing.get('chunk_seconds')==chunk_seconds
                and existing.get('resource_sha256')==resource_sha
                and existing.get('cudnn_plan_cache_limit')==cudnn_plan_cache_limit):return book
        # Explicitly replace only our recorded runtime cell. Preserve the prior
        # submitted notebook and original author/source cells as provenance.
        matches=[i for i,c in enumerate(notebook['cells']) if c['cell_type']=='code'
                 and ''.join(c['source']).startswith('# Checkpoint serialization recovery;')]
        if len(matches)!=1:raise ValueError('Ambiguous prior checkpoint adapter; refusing replacement')
        notebook['cells'].pop(matches[0])
        notebook['metadata'].setdefault('checkpoint_io_lineage',[]).append(dict(existing,
            parent_notebook_sha256=hashlib.sha256(book.read_bytes()).hexdigest()))
        notebook['metadata'].pop('checkpoint_io_adapter')
    if not notebook.get('metadata',{}).get('decoded_cache_adapter'):raise ValueError('Expected reviewed decoded cache bootstrap')
    encoded=base64.b64encode(data).decode('ascii')
    resource_code=''
    resource_bootstrap=''
    if cudnn_plan_cache_limit not in (None,-1):raise ValueError('Only reviewed cuDNN plan-cache disabling is supported')
    if resource_module:
        resource_data=Path(resource_module).read_bytes()
        if hashlib.sha256(resource_data).hexdigest()!=resource_sha:raise ValueError('Resource diagnostics hash changed')
        resource_code=(f"_resource_bytes=_io_b64.b64decode({base64.b64encode(resource_data).decode('ascii')!r})\n"
                       f"assert _io_hash.sha256(_resource_bytes).hexdigest()=={resource_sha!r}\n"
                       "(_io_root/'person1_resource_runtime.py').write_bytes(_resource_bytes)\n")
        resource_bootstrap='import person1_resource_runtime;person1_resource_runtime.install();'
    source=f'''# Checkpoint serialization recovery; no architecture/optimizer changes.
import base64 as _io_b64, hashlib as _io_hash, json as _io_json
from pathlib import Path as _IOPath
_io_root=_IOPath('/tmp/person1-cache-adapter')
_io_root.mkdir(parents=True,exist_ok=True)
_io_bytes=_io_b64.b64decode({encoded!r})
assert _io_hash.sha256(_io_bytes).hexdigest()=={expected_sha!r}
(_io_root/'person1_checkpoint_runtime.py').write_bytes(_io_bytes)
{resource_code}
from src.person1 import controller as _io_controller
_io_plan_cache_limit={cudnn_plan_cache_limit!r}
_io_previous_command=_io_controller.command
def _io_command(argv,log,timeout,env=None):
    if len(argv)>=4 and argv[1:4]==['-u','-m','src.person1.torch_worker']:
        if _io_plan_cache_limit is not None and len(argv)>4 and argv[4] in ('pilot','resume-check','train'):
            import os as _io_os
            env=dict(_io_os.environ if env is None else env)
            env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']=str({cudnn_plan_cache_limit!r})
        if len(argv)>4 and argv[4]=='train':timeout=min(timeout,{chunk_seconds+900})
        bootstrap="import sys,runpy,os;print('cuDNN plan cache limit: '+str(os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT')),flush=True);sys.path.insert(0,'/tmp/person1-cache-adapter');{resource_bootstrap}import person1_checkpoint_runtime,person1_decoded_cache;person1_checkpoint_runtime.install();person1_decoded_cache.install();runpy.run_module('src.person1.torch_worker',run_name='__main__',alter_sys=True)"
        argv=[argv[0],'-u','-c',bootstrap]+argv[4:]
    return _io_previous_command(argv,log,timeout,env)
_io_controller.command=_io_command
_io_previous_setup=_io_controller.setup
def _io_setup(*args,**kwargs):
    context=_io_previous_setup(*args,**kwargs)
    (_IOPath(context['run'])/'checkpoint_io_adaptation.json').write_text(_io_json.dumps(dict(
        sha256={expected_sha!r},parent_notebook_sha256={hashlib.sha256(book.read_bytes()).hexdigest()!r},
        scope='non-mutating CPU checkpoint snapshot, native torch.save and atomic rename,120s save watchdog',
        train_subprocess_timeout_seconds={chunk_seconds+900},resource_sha256={resource_sha!r},cudnn_plan_cache_limit={cudnn_plan_cache_limit!r},cudnn_cache_scope='pilot/resume-check/train; normal evaluation cache unchanged',model_and_optimizer_math_unchanged=True,
        native_gpu_resume_verification_required=True),indent=2))
    return context
_io_controller.setup=_io_setup
'''
    index=next(i for i,c in enumerate(notebook['cells']) if c['cell_type']=='code' and 'CONTEXT=setup(' in ''.join(c['source']))
    notebook['cells'].insert(index,dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=source.splitlines(keepends=True)))
    notebook.setdefault('metadata',{})['checkpoint_io_adapter']=dict(sha256=expected_sha,chunk_seconds=chunk_seconds,resource_sha256=resource_sha,cudnn_plan_cache_limit=cudnn_plan_cache_limit)
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<checkpoint-notebook>','exec')
    path=book.with_name('checkpoint-recovery-notebook.ipynb');path.write_text(json.dumps(notebook),encoding='utf-8')
    return path
