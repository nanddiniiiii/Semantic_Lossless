"""Load the matching person1 implementation rather than historical adapters."""
import json
from pathlib import Path
import tempfile
import zipfile


def bundle(folder,options,device,checkpoint='last.pt'):
    from src.sources import fetch_source,source_path
    name=options['algorithm'];folder=Path(folder)
    source={'adjscc':'adjscc_author','isec':'isec','ntscc_plus':'ntscc_plus'}.get(name)
    if source:fetch_source(source);options['source_dir']=str(source_path(source))
    if name=='adjscc':
        import tensorflow as tf
        from .tf_worker import build
        strategy=tf.distribute.OneDeviceStrategy('/gpu:0' if device.startswith('cuda') else '/cpu:0')
        model,optimizer=build(options,strategy)
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(folder/'last-tf.zip') as archive:
                for item in archive.infolist():(Path(td)/item.filename).resolve().relative_to(Path(td).resolve())
                archive.extractall(td)
            tf.train.Checkpoint(model=model,optimizer=optimizer).read(str(Path(td)/'state')).assert_consumed()
        return model,options
    from .torch_worker import build,load
    model=build(options,device);model.load_state_dict(load(folder/checkpoint,device)['state_dict'],strict=True)
    if hasattr(model,'stage'):model.stage='denoiser'
    model.eval();return model,options
