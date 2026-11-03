"""Create immutable model folders that can be zipped and shared with teammates."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from .dataset import digest
from .runtime import write_json

MODEL_NAMES = {
    'deepjscc': 'DeepJSCC', 'adjscc': 'ADJSCC', 'swinjscc': 'SwinJSCC',
    'isec': 'ISEC', 'hjscc': 'HJSCC', 'jcm': 'JCM', 'mdjcm': 'MDJCM',
    'ntscc_plus':'NTSCC-plus','ssc':'SSC','ofdm_guided_jscc':'OFDM-guided-JSCC',
    'deepjscc_mimo':'DeepJSCC-MIMO','deepjscc_wz':'DeepJSCC-WZ',
    'sqgan': 'SQ-GAN', 'shared_semantic_codebook': 'SharedSemanticCodebook',
}


def _safe_component(value):
    value = str(value)
    if not value or value in {'.', '..'} or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value):
        raise ValueError('Invalid folder identifier: ' + value)
    return value


def verify_bundle(folder):
    folder = Path(folder).resolve()
    metadata = json.loads((folder / 'bundle.json').read_text(encoding='utf-8'))
    for item in metadata['files']:
        path = (folder / item['path']).resolve()
        path.relative_to(folder)
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Model bundle verification failed: ' + str(path))
    if metadata.get('pipeline_completed') is not True:
        raise ValueError('Bundle is not a completed pipeline result')
    return metadata


def package_run(run, models_root, teammate, run_id):
    """Package one trained run, excluding pilot checkpoints and raw datasets."""
    run = Path(run).resolve()
    teammate = _safe_component(teammate)
    if teammate not in {'1', '2', '3'}:
        raise ValueError('Teammate must be 1, 2 or 3')
    run_id = _safe_component(run_id)
    config_path = run / 'resolved_config.json'
    options = json.loads(config_path.read_text(encoding='utf-8'))
    algorithm = options['algorithm']
    if algorithm not in MODEL_NAMES:
        raise ValueError('No shareable bundle specification for ' + algorithm)
    checkpoints = sorted((run / 'checkpoints').glob('*'))
    checkpoints = [path for path in checkpoints if path.is_file() and not path.name.endswith('.tmp')]
    if not checkpoints:
        return None
    for stage_name in ('train', 'evaluate'):
        status_path = run / f'{stage_name}_status.json'
        if not status_path.is_file() or json.loads(status_path.read_text())['status'] != 'completed':
            raise ValueError('Only completed training and evaluation can be shared; missing successful '+stage_name)
    root = Path(models_root).resolve()
    parent = root / f'teammate_{teammate}_models' / MODEL_NAMES[algorithm]
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / run_id
    destination.resolve().relative_to(root)
    if destination.exists():
        raise FileExistsError('Model bundle already exists: ' + str(destination))
    with tempfile.TemporaryDirectory(prefix='.bundle-', dir=parent) as temporary:
        staging = Path(temporary)
        for checkpoint in checkpoints:
            shutil.copy2(checkpoint, staging / checkpoint.name)
        for filename in ('requested_config.json', 'resolved_config.json', 'model.json', 'source.json',
                         'dataset_identity.json', 'environment.json', 'environment_plan.json',
                         'training_summary.json', 'evaluation_reload.json', 'pseudo_label_coverage.json', 'native_recipe.json', 'native_data_identity.json'):
            if (run / filename).is_file():
                shutil.copy2(run / filename, staging / filename)
        if (run / 'metrics').is_dir():
            target = staging / 'metrics'
            target.mkdir()
            for path in (run / 'metrics').glob('*'):
                if path.is_file() and path.suffix in {'.json', '.csv'}:
                    shutil.copy2(path, target / path.name)
        portable = dict(options)
        portable['manifest'] = None  # Original dataset paths are provenance, not a portable dataset.
        missing_assets = []
        for key in ('checkpoint', 'classifier', 'codebook'):
            value = options.get(key)
            if not value:
                continue
            asset = Path(value)
            if key == 'checkpoint' and algorithm == 'sqgan' and asset.is_dir():
                paths = sorted(asset.glob('*.ckpt'))
                if len(paths) != 10:
                    raise ValueError('SQ-GAN constructor requires its ten checkpoint components')
            else:
                paths = [asset] if asset.is_file() else []
            if not paths:
                missing_assets.append(key)
                portable[key] = None
                continue
            target = staging / 'assets' / key
            target.mkdir(parents=True)
            for path in paths:
                shutil.copy2(path, target / path.name)
            portable[key] = (target.relative_to(staging) / paths[0].name).as_posix() if len(paths) == 1 else target.relative_to(staging).as_posix()
        write_json(staging / 'portable_config.json', portable)
        (staging / 'README.md').write_text(
            f'# {MODEL_NAMES[algorithm]} — teammate {teammate}\n\n'
            f'Run: `{run_id}`. Training/evaluation status is in the accompanying JSON files.\n\n'
            'Zip this run folder or the whole teammate folder for Drive/OneDrive. '
            'Keep each run separate. `last.pt` is the final saved checkpoint; '
            '`best.pt`, if present, is the development-selected checkpoint. '
            'These files include optimizer/progress state where the trainer supports it, '
            'but exact training resume is not implemented.\n\n'
            'Use the matching repository revision and model dependencies recorded in '
            '`environment.json`/`source.json`. Paths in `portable_config.json` are relative '
            'to this folder. Dataset inputs are not bundled. Some author code and frozen '
            'teacher configurations are fetched from their pinned upstream releases. '
            f'Missing constructor assets: {missing_assets or "none recorded"}.\n\n'
            'After installing the recorded model dependencies, restore the model with:\n\n'
            '```python\nfrom src.sharing import load_bundle\n'
            'model, options = load_bundle("path/to/this/run", device="cuda:0")\n```\n'
            'Some author implementations require CUDA even for inference.\n', encoding='utf-8')
        files = [{'path': path.relative_to(staging).as_posix(), 'bytes': path.stat().st_size,
                  'sha256': digest(path)} for path in sorted(staging.rglob('*')) if path.is_file()]
        write_json(staging / 'bundle.json', {'algorithm': algorithm, 'model_name': MODEL_NAMES[algorithm],
            'teammate': teammate, 'run_id': run_id, 'missing_constructor_assets': missing_assets,
            'files': files, 'pipeline_completed': True, 'scientific_validity_certified': False})
        verify_bundle(staging)
        staging.rename(destination)
    return destination


def import_bundles(download_root, models_root, teammate):
    """Copy only verified bundles for this teammate; do not overwrite older runs."""
    teammate = _safe_component(teammate)
    imported = []
    root = Path(models_root).resolve()
    for manifest in sorted(Path(download_root).rglob('bundle.json')):
        folder = manifest.parent
        metadata = verify_bundle(folder)
        if metadata['teammate'] != teammate:
            raise ValueError('Downloaded model belongs to a different teammate')
        algorithm = metadata['algorithm']
        if algorithm not in MODEL_NAMES:
            raise ValueError('Unknown model bundle algorithm')
        run_id = _safe_component(metadata['run_id'])
        target = root / f'teammate_{teammate}_models' / MODEL_NAMES[algorithm] / run_id
        target.resolve().relative_to(root)
        if target.exists():
            if verify_bundle(target) != metadata:
                raise FileExistsError('A different model bundle already exists: ' + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.import-',dir=target.parent) as temporary:
                staging=Path(temporary)/'bundle'
                shutil.copytree(folder,staging)
                verify_bundle(staging)
                staging.rename(target)
        imported.append(str(target))
    return imported


def load_bundle(folder, device='cpu', checkpoint='last.pt'):
    """Restore a packaged model using this repository and its pinned upstreams."""
    import torch
    from .models import build, load_training_state
    from .sources import fetch_source
    folder=Path(folder).resolve()
    metadata=verify_bundle(folder)
    if metadata['missing_constructor_assets']:
        raise FileNotFoundError('Bundle needs constructor assets: '+str(metadata['missing_constructor_assets']))
    options=json.loads((folder/'portable_config.json').read_text(encoding='utf-8'))
    for key in ('checkpoint','classifier','codebook'):
        if options.get(key):
            path=(folder/options[key]).resolve()
            path.relative_to(folder)
            options[key]=str(path)
    options['device']=device
    name=metadata['algorithm']
    fetch_source('diffcom' if name=='adjscc' else name)
    if name=='deepjscc_wz':
        import hydra
        from .codecs import upstream_import
        from .sources import source_path
        recipe=json.loads((folder/'native_recipe.json').read_text(encoding='utf-8'))
        with upstream_import(source_path(name),('src',)):
            model=hydra.utils.instantiate(recipe['model'],input_dims=tuple(recipe['input_dims']),output_dir=str(folder))
        checkpoint='last.ckpt' if checkpoint=='last.pt' else checkpoint
        checkpoint_path=(folder/checkpoint).resolve();checkpoint_path.relative_to(folder)
        if checkpoint not in {item['path'] for item in metadata['files']}:raise ValueError('WZ checkpoint not verified')
        state=torch.load(checkpoint_path,map_location=device,weights_only=False)
        model.load_state_dict(state['state_dict'],strict=True)
        model.to(device).eval()
        return model,options
    if name=='sqgan':
        from .sqgan_train import build as build_sqgan
        model=build_sqgan(options)
    else:
        model=build(name,options,device)
    checkpoint_path=(folder/checkpoint).resolve()
    checkpoint_path.relative_to(folder)
    if checkpoint_path.relative_to(folder).as_posix() not in {item['path'] for item in metadata['files']}:
        raise ValueError('Checkpoint is not listed in the verified bundle')
    state=torch.load(checkpoint_path,map_location=device,weights_only=True)
    load_training_state(model,state.get('ema_state_dict',state['state_dict']) if name=='ssc' else state['state_dict'])
    if name=='isec':model.stage='denoiser'
    model.eval()
    return model,options
