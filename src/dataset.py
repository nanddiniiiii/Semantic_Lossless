"""Content-identified common parents; Kodak is never a development split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def discover(input_root, keys=None):
    """Find unambiguous official HR folder names, never confuse LR/SR copies."""
    aliases={'div2k_train':{'div2k_train_hr'},'div2k_val':{'div2k_valid_hr','div2k_val_hr'},
             'flickr2k_train':{'flickr2k_hr','flickr2k'},'kodak_test':{'kodak','kodak24','kodak-dataset'}}
    expected={'div2k_train':800,'div2k_val':100,'flickr2k_train':2650,'kodak_test':24}
    selected=set(aliases) if keys is None else set(keys)
    if selected-set(aliases):raise ValueError('Unknown dataset keys: '+str(sorted(selected-set(aliases))))
    root=Path(input_root);found={}
    if not root.is_dir():return found
    folders=[p for p in root.rglob('*') if p.is_dir()]
    for key,names in aliases.items():
        if key not in selected:continue
        candidates=[]
        for path in folders:
            if path.name.lower() not in names:continue
            files=[p for p in path.rglob('*') if p.suffix.lower() in SUFFIXES]
            if len(files)==expected[key]:candidates.append(path.resolve())
        # Duplicate mount layouts may resolve to the same symlink target.
        candidates=sorted(set(candidates))
        # Nested aliases with equal recursive counts identify the same image set.
        candidates=[path for path in candidates if not any(
            other != path and path in other.parents for other in candidates)]
        if len(candidates)==1:found[key]=str(candidates[0])
        elif len(candidates)>1:
            raise ValueError('Multiple candidate official roots for '+key+': '+str(candidates))
    return found


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def freeze(roots, destination, strict_counts=True):
    """Official DIV2K800/Flickr2650 train, DIV2K100 dev, Kodak24 test.

    Hash decoded RGB pixels as well as file bytes: lossless re-encodings must not
    silently enter another split. No crop is chosen until after membership freezes.
    """
    expected = {'div2k_train': 800, 'flickr2k_train': 2650,
                'div2k_val': 100, 'kodak_test': 24}
    if set(roots) != set(expected):
        raise ValueError('Provide exactly these roots: ' + ', '.join(expected))
    records, seen, training_duplicates = [], {}, []
    for source, count in expected.items():
        root = Path(roots[source]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        files = sorted(p for p in root.rglob('*') if p.suffix.lower() in SUFFIXES)
        if not files or (strict_counts and len(files) != count):
            raise ValueError(f'{source}: expected {count} images, found {len(files)} in {root}')
        split = 'test' if source.endswith('_test') else 'val' if source.endswith('_val') else 'train'
        for position,path in enumerate(files,1):
            with Image.open(path) as handle:
                rgb = ImageOps.exif_transpose(handle).convert('RGB')
                width, height = rgb.size
                pixels = hashlib.sha256(f'{width}x{height}:RGB:'.encode() + rgb.tobytes()).hexdigest()
            if pixels in seen:
                earlier=seen[pixels]
                if split!='train' or earlier['split']!='train':
                    raise ValueError(f'Duplicate RGB parent across protected split: {path}; already seen in {earlier["path"]}')
                training_duplicates.append({'path':str(path),'earlier_path':earlier['path'],'rgb_sha256':pixels})
                print(f'Training duplicate retained and recorded: {path}; first {earlier["path"]}',flush=True)
            else:
                seen[pixels] = {'path':str(path),'split':split}
            records.append({'source': source, 'split': split,
                            'relative_path': path.relative_to(root).as_posix(),
                            'sha256': digest(path), 'rgb_sha256': pixels,
                            'width': width, 'height': height})
            if position%100==0 or position==len(files):
                print(f'Hashing dataset {source}: {position}/{len(files)} parents',flush=True)
        print(f'Dataset checked: {source}, {len(files)} parents, split={split}', flush=True)
    identity = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    value = {'protocol': 'div2k-flickr2k-kodak-v1', 'identity': identity,
             'roots': {k: str(Path(v).resolve()) for k, v in roots.items()},
             'counts': {s: sum(r['split'] == s for r in records) for s in ('train', 'val', 'test')},
             'images': records, 'official_counts_enforced': strict_counts,
             'training_duplicates_retained':training_duplicates,
             'duplicate_policy':'retain and record within training; reject duplicates involving validation or test',
             'test_used_for_selection': False}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old['identity'] != identity:
            raise FileExistsError('Frozen membership changed; use a new manifest/version')
    destination.write_text(json.dumps(value, indent=2), encoding='utf-8')
    return value


class ParentImages:
    """No silent resize; fixed center crop for dev, full resolution for test."""
    def __init__(self, manifest, split, crop=256, verify=True):
        value = json.loads(Path(manifest).read_text())
        canonical = json.dumps(value['images'], sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != value['identity']:
            raise ValueError('Manifest identity mismatch')
        self.rows = [r for r in value['images'] if r['split'] == split]
        if not self.rows:
            raise ValueError('Empty split: ' + split)
        self.roots = value['roots']
        self.split, self.crop = split, crop
        if split!='test':
            ch,cw=crop if isinstance(crop,(list,tuple)) else (crop,crop)
            incompatible=[row['source']+'/'+row['relative_path'] for row in self.rows
                          if row['width']<cw or row['height']<ch]
            if incompatible:
                raise ValueError(f'Parents smaller than native crop {ch}x{cw}: {incompatible[:5]}')
        if verify:
            for position,row in enumerate(self.rows,1):
                path = Path(self.roots[row['source']]) / row['relative_path']
                if digest(path) != row['sha256']:
                    raise ValueError('Changed dataset file: ' + str(path))
                if position%100==0 or position==len(self.rows):
                    print(f'Verified {split} image files: {position}/{len(self.rows)}',flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import random
        import numpy as np
        import torch
        row = self.rows[index]
        with Image.open(Path(self.roots[row['source']]) / row['relative_path']) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
            if self.split != 'test':
                ch,cw = self.crop if isinstance(self.crop,(list,tuple)) else (self.crop,self.crop)
                if image.width < cw or image.height < ch:
                    raise ValueError('Image smaller than declared crop: ' + row['relative_path'])
                left = random.randrange(image.width-cw+1) if self.split == 'train' else (image.width-cw)//2
                top = random.randrange(image.height-ch+1) if self.split == 'train' else (image.height-ch)//2
                image = image.crop((left, top, left+cw, top+ch))
            array = np.asarray(image, dtype=np.float32).copy() / 255
        return {'image': torch.from_numpy(array).permute(2, 0, 1),
                'sample': row['source'] + '/' + row['relative_path']}
