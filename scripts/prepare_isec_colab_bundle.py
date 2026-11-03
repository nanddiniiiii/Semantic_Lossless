"""Build an explicit, credential-free evaluation transfer for Amrita Colab."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/'Results/team_downloads/1/hmmkk20__semantic-notebook-1-isec/05a60cc829cb47a986e620fbba637425'
PARENT=RECEIPT/'published_output/team_results/1/20261103T101702Z-e89966bf/isec'
META=ROOT/'Results/person1_colab/isec_parent_metadata/team_results/1/20261103T101702Z-e89966bf/isec'
OUT=ROOT/'Results/person1_colab/isec-evaluation-input'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    entries={r['path']:r for r in json.loads((PARENT/'artifacts.json').read_text())['files']}
    for name in ['checkpoints/last.pt']:
        path=PARENT/name
        if path.stat().st_size!=entries[name]['bytes'] or sha(path)!=entries[name]['sha256']:
            raise ValueError('Checkpoint not yet downloaded or hash differs')
    for name in ['resolved_config.json','predictions.json','training_summary.json']:
        if sha(META/name)!=entries[name]['sha256']:raise ValueError('Metadata hash differs: '+name)
    saved=torch.load(PARENT/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    if saved['progress']['completed_epochs']!={'jscc':150,'denoiser':150}:raise ValueError('Training not complete')
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(PARENT/'checkpoints/last.pt',OUT/'last.pt')
    for name in ['resolved_config.json','training_summary.json']:shutil.copy2(META/name,OUT/name)
    shutil.copy2(META/'predictions.json',OUT/'parent_predictions.json')
    book=json.loads((RECEIPT/'submitted-notebook.ipynb').read_text())
    code=''.join(book['cells'][2]['source']);tree=ast.parse(code)
    payload=next(n.args[0].args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr=='decompress')
    raw=zlib.decompress(base64.b64decode(payload));overlay=json.loads(raw)
    expected='08e48202e0302107fac28dd458aad82248e7e14c8f77257c4c1e560fe0bdb395'
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Frozen overlay differs')
    revision='20278596c152de431fbc5478ebe5392612cfdd42'
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'src'],cwd=ROOT,text=True).splitlines()
    for relative in files:
        if not relative.endswith('.py'):continue
        target=OUT/'runtime'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(subprocess.check_output(['git','show',revision+':'+relative],cwd=ROOT))
    for relative, encoded in overlay.items():
        target=(OUT/'runtime'/relative).resolve();target.relative_to((OUT/'runtime').resolve())
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(encoded))
    for file in (ROOT/'.upstream/isec').rglob('*.py'):
        if '.git' in file.parts or '__pycache__' in file.parts:continue
        target=OUT/'author'/file.relative_to(ROOT/'.upstream/isec');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    shutil.copy2(ROOT/'.upstream/isec/source_receipt.json',OUT/'author/source_receipt.json')
    manifest_path=ROOT/'Results/team_downloads/1/hmmkk20__semantic-notebook-1-isec/239f4f84ac0a411b9b30d38130d8c576/published_output/team_results/1/20261103T082337Z-fc92bd40/common_dataset.json'
    manifest=json.loads(manifest_path.read_text());shutil.copy2(manifest_path,OUT/'frozen_dataset.json')
    rows=[x for x in manifest['images'] if x['split']=='test']
    if len(rows)!=24:raise ValueError('Kodak membership differs')
    images=ROOT.parent/'references/SemanticSchedulerEnd2End_team/data/parameter_codebook/train'
    (OUT/'kodak').mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        path=images/Path(row['relative_path']).name
        if sha(path)!=row['sha256']:raise ValueError('Kodak raw hash differs')
        with Image.open(path) as image:arr=np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
        height,width=arr.shape[:2]
        rgb=hashlib.sha256(f'{width}x{height}:RGB:'.encode()+arr.tobytes()).hexdigest()
        if rgb!=row['rgb_sha256']:raise ValueError('Kodak RGB hash differs')
        np.save(OUT/'kodak'/('%02d.npy'%i),arr)
    shutil.copy2(ROOT/'scripts/isec_colab_evaluation.py',OUT/'evaluate.py')
    manifest_out=dict(files={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='bundle_manifest.json'},
        checkpoint_sha256=sha(OUT/'last.pt'),completed_epochs=saved['progress']['completed_epochs'],
        dataset_identity=manifest['identity'],source_revision=revision,overlay_sha256=expected,
        missing_trials=240-len(json.loads((META/'predictions.json').read_text())['rows']),
        contains_credentials=False,contains_training_images=False,destination='Amrita Google Colab ephemeral runtime')
    (OUT/'bundle_manifest.json').write_text(json.dumps(manifest_out,indent=2))
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Transfer ZIP corrupt')
    report=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),**{k:v for k,v in manifest_out.items() if k!='files'})
    (OUT.parent/'transfer_review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
