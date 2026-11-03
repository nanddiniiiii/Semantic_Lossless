"""Complete local ZIP, with a streaming hash verification of every member."""
import hashlib
import json
from pathlib import Path
import zipfile
from src.dataset import digest
from src.sharing import verify_bundle
from .protocol import ORDER,require_complete


def create(root,output=None):
    root=Path(root).resolve()
    output=Path(output) if output else root.parent/'SharedModelHandoff'
    output.mkdir(parents=True,exist_ok=True)
    models=root/'Models/teammate_1_models'
    verified=[]
    for name in ORDER:
        from src.sharing import MODEL_NAMES
        matches=list((models/MODEL_NAMES[name]).glob('*/bundle.json'))
        accepted=[]
        for marker in matches:
            metadata=verify_bundle(marker.parent)
            summary=json.loads((marker.parent/'training_summary.json').read_text())
            require_complete(summary,{'jscc':150,'denoiser':150} if name=='isec' else {'codec':150})
            accepted.append(metadata)
        if not accepted:raise ValueError('No completed150-epoch bundle for '+name)
        verified.extend(accepted)
    paths=[]
    for relative in ('Models/teammate_1_models','Results/team_downloads/1','Results/person1_campaign','src','Infrastructure','configs','scripts'):
        paths.extend(p for p in (root/relative).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    paths += [root/'Notebooks/1.ipynb',root/'pyproject.toml',root/'AGENTS.md',root/'Docs/TEAMMATE_HANDOFF.md',root/'Docs/RESULT_COLLECTION.md',root/'Docs/PERSON1_IMPLEMENTATION.md',root/'Results/comparison.csv']
    paths=sorted(set(p for p in paths if p.is_file() and not p.name.endswith(('.lock','.tmp'))))
    manifest=dict(files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in paths],
                  accepted_models=verified,credentials_included=False,datasets_included=False)
    target=output/'teammate_1_full_handoff.zip';temporary=target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for path in paths:archive.write(path,path.relative_to(root).as_posix())
        archive.writestr('HANDOFF_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC verification failed')
        for item in manifest['files']:
            sha=hashlib.sha256()
            with archive.open(item['path']) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
            if sha.hexdigest()!=item['sha256']:raise ValueError('ZIP hash mismatch: '+item['path'])
    temporary.replace(target)
    (output/'teammate_1_zip_verification.json').write_text(json.dumps(dict(zip_sha256=digest(target),files=len(paths),
        all_member_hashes_passed=True,model_reload_evidence='Strict model reload is recorded by each hash-verified Kaggle bundle'),indent=2))
    print('Verified handoff ZIP:',target,flush=True);return target
