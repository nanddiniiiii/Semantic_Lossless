"""Copy a bounded code/result publication into an isolated latest-main checkout."""
import json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/'Semantic_Lossless_delivery_20261103'
copied=[]
def copy(source,target=None):
    relative=Path(target) if target else source.relative_to(ROOT)
    if source.suffix.lower() in {'.pt','.pth','.ckpt','.bin','.zip','.npy','.safetensors'}:raise ValueError('Weight/data upload rejected')
    if source.stat().st_size>5*1024*1024:raise ValueError('Public file exceeds review limit')
    out=DEST/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out);copied.append(relative.as_posix())
for folder in ['src/person1','tests/person1']:
    for p in (ROOT/folder).glob('*.py'):copy(p)
for p in (ROOT/'scripts').glob('*.py'):
    if p.name not in {'1.py','2.py','3.py'} and (p.name.startswith(('adjscc_','isec_','person1_','flops_','timing_','finalize_','package_','prepare_','complete_two_','two_model_','continue_isec_','collect_person1_','transfer_isec_','build_person1_'))):copy(p)
for name in ['test_flops_isec_audit.py','test_person1_checkpoint_runtime.py','test_person1_chunk_policy.py','test_person1_decoded_cache.py','test_person1_speed_report.py']:copy(ROOT/'tests'/name)
for p in (ROOT/'Docs').glob('PERSON1*.md'):copy(p)
copy(ROOT/'Notebooks/1.ipynb')
for p in (ROOT/'Results/three_model_final').glob('*'):
    if p.suffix in {'.json','.csv','.md'}:copy(p)
for name in ['deepjscc_logical_arithmetic.json','isec_final150_logical_arithmetic.json']:
    copy(ROOT/'Results/person1_flops'/name,Path('Results/three_model_final/complexity')/name)
for filename in ['deepjscc.json','adjscc.json','comparison.json']:
    copy(ROOT/'Results/person1_timing/outputs/person1-timing'/filename,Path('Results/three_model_final/timing')/filename)
for model,path in [('DeepJSCC',ROOT/'Results/team_downloads/1/hmmmkk__semantic-notebook-1-deepjscc/7a5a49a0c063462fa6f3d63a86217fd6/published_output/team_results/1/20261031T003230Z-cc4c86d3/deepjscc'),
                   ('ADJSCC',ROOT/'Results/person1_final/adjscc/combined_run'),('ISEC',ROOT/'Results/person1_final/isec/combined_run')]:
    for filename in ['summary.json','per_image.csv','per_image.json']:
        copy(path/'metrics'/filename,Path('Results/three_model_final/raw_metrics')/model/filename)
    for filename in ['training_summary.json','profile.json','evaluation_reload.json']:
        if (path/filename).exists():copy(path/filename,Path('Results/three_model_final/provenance')/model/filename)
readme=subprocess.check_output(['git','show','HEAD:README.md'],cwd=DEST).decode()
notice='''
## Verified Person1 three-model results

DeepJSCC and ADJSCC completed150 full epochs each; ISEC completed150 JSCC
and150 denoiser epochs. Kodak24x10 outputs, checkpoint hashes, quality metrics,
declared logical FLOPs and measured timing scopes are documented in
[the three-model report](Results/three_model_final/README.md).
ISEC used native CBR1/16, DeepJSCC/ADJSCC1/6; this is not an equal-rate ranking.
ISEC measured14.87dB PSNR; no paper-convergence claim is made.
Deep/AD timing shares one CPU host; ISEC T4 timing is separately labelled.
Weights are excluded from Git and retained in the private OneDrive handoff.
NTSCC+ remains pending and is excluded from this completion scope.

'''
if '## Verified Person1 three-model results' not in readme:
    first,rest=readme.split('\n',1);(DEST/'README.md').write_text(first+'\n'+notice+rest)
copied.append('README.md')
original=subprocess.check_output(['git','show','HEAD:.gitignore'],cwd=DEST).decode()
(DEST/'.gitignore').write_text(original+'\n# Person1 private checkpoints, raw downloads and local execution state\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n.pytest*/\nResults/person1_colab/\nResults/person1_final/\nResults/NO_GITHUB_PUBLICATION\n')
copied.append('.gitignore')
(ROOT/'Results/three_model_final/publication_allowlist.json').write_text(json.dumps(dict(paths=copied,repository='nanddiniiiii/Semantic_Lossless',weights_included=False,approval='User explicitly requested publication after completed three-model retrieval'),indent=2))
print('Curated publication files',len(copied))
