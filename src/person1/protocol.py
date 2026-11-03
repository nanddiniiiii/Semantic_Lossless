"""Portable deterministic parent order/crops and strict continuation contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ORDER = ['deepjscc', 'adjscc', 'isec', 'ntscc_plus']
DATASET_IDENTITY = 'add977eb3277909e5ad1447770abdb368458445ab42b2f48a91c24f4c31f7fa0'


def options(name, manifest):
    if name not in ORDER:
        raise ValueError(name)
    value = dict(algorithm=name, manifest=str(manifest), protocol='person1-author-v1',
                 epochs=150, snr_db=10., train_channel='awgn', test_channel='awgn',
                 train_snr_db=10., test_snr_db=10., seed=22, batch_size=2, crop=256,
                 learning_rate=1e-4, max_seconds=14400, trials=10, data_parallel=True,
                 workers=4,
                 from_scratch=True, clip_norm=0., resume_checkpoint=None,
                 dataset_adaptation=True, paper_convergence_reproduced=False,
                 person1_runtime=True)
    if name == 'deepjscc':
        value.update(crop=128, batch_size=32, latent_channels=16,
                     architecture='paper2019-figure2', prelu_parameterization='per-channel')
    elif name == 'adjscc':
        value.update(crop=128, batch_size=16, latent_channels=16, backend='author_tensorflow')
    elif name == 'isec':
        value.update(crop=128, batch_size=2, latent_channels=6, learning_rate=1e-3,
                     denoiser_learning_rate=2e-4, iterations=50,horizontal_flip=True,
                     refinement_lr=.001, alpha=4., delta=1.)
    else:
        value.update(clip_norm=1., train_lambda=.72, eta=.2, backend='author_natten0146')
    return value


def fingerprint(options, identity):
    ignored = {'manifest', 'max_seconds', 'resume_checkpoint', 'run_id', 'models_root',
               'teammate', 'source_revision', 'pilot', 'stop_after_steps','workers'}
    payload = {k: v for k, v in options.items() if k not in ignored}
    payload['dataset_identity'] = identity
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Images:
    """Each training parent appears once per epoch; crop randomness is replayable.

    The crop adapter is declared, not an assertion of the authors' RNG sequence.
    An epoch/sample seed makes worker prefetch and interrupted chunks immaterial.
    """
    def __init__(self, manifest, split, crop=128, horizontal_flip=False):
        self.manifest = json.loads(Path(manifest).read_text())
        canonical = json.dumps(self.manifest['images'], sort_keys=True, separators=(',', ':'))
        if hashlib.sha256(canonical.encode()).hexdigest() != self.manifest['identity']:
            raise ValueError('Frozen dataset manifest was modified')
        self.rows = [r for r in self.manifest['images'] if r['split'] == split]
        if not self.rows:
            raise ValueError('Empty ' + split)
        self.split, self.crop = split, crop
        self.horizontal_flip=horizontal_flip

    def __len__(self):
        return len(self.rows)

    def order(self, epoch, seed):
        return np.random.RandomState(seed + 1000003 * epoch).permutation(len(self.rows)).tolist()

    def array(self, index, epoch=0, seed=22):
        row = self.rows[index]
        path = Path(self.manifest['roots'][row['source']]) / row['relative_path']
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
            if self.split != 'test':
                size = self.crop
                if min(image.size) < size:
                    raise ValueError('Parent smaller than declared crop: ' + str(path))
                if self.split == 'train':
                    rng = np.random.RandomState((seed + epoch*1000003 + index*9176) % (2**32))
                    x, y = rng.randint(image.width-size+1), rng.randint(image.height-size+1)
                else:
                    x, y = (image.width-size)//2, (image.height-size)//2
                image = image.crop((x, y, x+size, y+size))
                if self.split=='train' and self.horizontal_flip and rng.rand()<.5:
                    image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return np.asarray(image, dtype=np.float32).copy() / 255.


def require_complete(summary, targets):
    completed = summary.get('completed_epochs', {})
    if any(completed.get(stage, 0) != epochs for stage, epochs in targets.items()):
        raise ValueError('Training is partial: ' + json.dumps(completed))
    if summary.get('status') != 'training_completed':
        raise ValueError('Training completion status missing')


def resume_fixture(options,run):
    """Small real-parent crops for continuation mechanics, never quality results."""
    source=Images(options['manifest'],'train',options['crop'],options.get('horizontal_flip',False))
    folder=Path(run)/'resume-fixture';folder.mkdir(exist_ok=True)
    rows=[]
    for i in range(5):
        path=folder/(str(i)+'.png')
        Image.fromarray((source.array(i%len(source),0,options['seed'])*255).round().astype(np.uint8)).save(path)
        rows.append(dict(source='resume-fixture',relative_path=path.name,split='train' if i<4 else 'val'))
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(dict(images=rows,identity=identity,roots={'resume-fixture':str(folder)})))
    return dict(options,manifest=str(manifest),epochs=1,batch_size=2,max_seconds=600,resume_checkpoint=None)
