"""Derive an explicitly recorded I/O adapter without editing frozen sources."""
import base64
import hashlib
import json
from pathlib import Path


def prepare_cache_notebook(book, module, expected_sha):
    book, module = Path(book), Path(module)
    data = module.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Reviewed cache adapter hash changed')
    notebook = json.loads(book.read_text(encoding='utf-8'))
    if notebook.get('metadata', {}).get('decoded_cache_adapter'):
        if notebook['metadata']['decoded_cache_adapter']['sha256'] != expected_sha:
            raise ValueError('Continuation already uses a different cache adapter')
        return book
    encoded = base64.b64encode(data).decode('ascii')
    source = f'''# Reviewed I/O-only adapter; frozen source files and author model are unchanged.
import base64 as _cache_b64, hashlib as _cache_hash, json as _cache_json
from pathlib import Path as _CachePath
_cache_root = _CachePath('/tmp/person1-cache-adapter')
_cache_root.mkdir(parents=True, exist_ok=True)
_cache_bytes = _cache_b64.b64decode({encoded!r})
assert _cache_hash.sha256(_cache_bytes).hexdigest() == {expected_sha!r}
(_cache_root/'person1_decoded_cache.py').write_bytes(_cache_bytes)
from src.person1 import controller as _cache_controller
_cache_original_command = _cache_controller.command
def _cache_command(argv, log, timeout, env=None):
    if len(argv) >= 4 and argv[1:4] == ['-u', '-m', 'src.person1.torch_worker']:
        bootstrap = "import sys,runpy;sys.path.insert(0,'/tmp/person1-cache-adapter');import person1_decoded_cache;person1_decoded_cache.install();runpy.run_module('src.person1.torch_worker',run_name='__main__',alter_sys=True)"
        argv = [argv[0], '-u', '-c', bootstrap] + argv[4:]
    return _cache_original_command(argv, log, timeout, env)
_cache_controller.command = _cache_command
_cache_original_setup = _cache_controller.setup
def _cache_setup(*args, **kwargs):
    context = _cache_original_setup(*args, **kwargs)
    (_CachePath(context['run'])/'decoded_cache_adaptation.json').write_text(_cache_json.dumps(dict(
        sha256={expected_sha!r}, source='reviewed private notebook runtime adapter',
        parent_notebook_sha256={hashlib.sha256(book.read_bytes()).hexdigest()!r},
        scope='EXIF-corrected RGB uint8 decoded parents on local scratch; identical crop/flip/float conversion',
        author_model_unchanged=True, frozen_runtime_files_unchanged=True,
        logical_training_contract_unchanged=True, gpu_speedup_measured=False), indent=2))
    return context
_cache_controller.setup = _cache_setup
'''
    index = next(i for i, cell in enumerate(notebook['cells'])
                 if cell['cell_type'] == 'code' and 'CONTEXT=setup(' in ''.join(cell['source']))
    notebook['cells'].insert(index, dict(cell_type='code', execution_count=None,
                metadata={}, outputs=[], source=source.splitlines(keepends=True)))
    notebook.setdefault('metadata', {})['decoded_cache_adapter'] = dict(sha256=expected_sha,
           parent_notebook_sha256=hashlib.sha256(book.read_bytes()).hexdigest(),
           scope='I/O only; original overlay unmodified')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':compile(''.join(cell['source']), '<cache-notebook>', 'exec')
    destination = book.with_name('cache-continuation-notebook.ipynb')
    destination.write_text(json.dumps(notebook), encoding='utf-8')
    return destination
