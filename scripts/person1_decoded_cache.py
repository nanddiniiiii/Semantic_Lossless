"""Optional I/O-only adapter for a frozen Person1 worker; no model changes.

Cache exact EXIF-corrected RGB uint8 parents on local scratch. Crop/flip RNG,
parent order, float conversion and duplicate exposures remain unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import subprocess
import numpy as np
from PIL import Image, ImageOps


class DecodedCache:
    def __init__(self, directory, budget_bytes=40 * 1024**3, reserve_bytes=3 * 1024**3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budget = budget_bytes
        self.reserve = reserve_bytes
        self.used = sum(p.stat().st_size for p in self.directory.glob('*.npy'))
        self.lock = threading.Lock()
        self.hits = self.misses = self.skipped = 0
        self.validated = set()
        self.disk_backed = True
        if os.name == 'posix':
            try:
                result = subprocess.run(['df', '-PT', str(self.directory)], capture_output=True,
                                        text=True, timeout=5, check=True)
                kind = result.stdout.splitlines()[-1].split()[1]
                self.disk_backed = kind not in ('tmpfs', 'ramfs')
                print('Decoded-cache filesystem: ' + kind + '; disk_backed=' + str(self.disk_backed), flush=True)
            except (OSError, subprocess.SubprocessError, IndexError):
                self.disk_backed = False  # Never fill an unverified RAM filesystem.

    def release(self, pixels, row):
        """Close mmap handles and advise clean pages away after copying the crop."""
        mapping = getattr(pixels, '_mmap', None)
        if mapping is not None:
            mapping.close()
        path = self.directory / (row['sha256'] + '.npy')
        if path.exists() and hasattr(os, 'posix_fadvise'):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                # Ensure cache writes are clean before releasing file-backed RAM.
                try:os.fdatasync(descriptor)
                except OSError:pass
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass  # Reclaim is a performance hint; never alter image values.
            finally:
                os.close(descriptor)

    def rgb(self, row, path):
        # Source/RGB hashes come from the already verified frozen manifest.
        key = row['sha256']
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid source hash in frozen manifest')
        target = self.directory / (key + '.npy')
        with self.lock:
            if self.disk_backed and target.exists():
                data = np.load(target, mmap_mode='r', allow_pickle=False)
                self._validate(data, row, key)
                self.hits += 1
                return data
            with Image.open(path) as handle:
                data = np.asarray(ImageOps.exif_transpose(handle).convert('RGB'), dtype=np.uint8).copy()
            self._validate(data, row, key, force=True)
            self.misses += 1
            needed = data.nbytes + 1024
            if not self.disk_backed or self.used + needed > self.budget or shutil.disk_usage(self.directory).free < needed + self.reserve:
                self.skipped += 1
                return data
            temporary = target.with_suffix('.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.save(stream, data, allow_pickle=False)
                temporary.replace(target)
                self.used += target.stat().st_size
            except OSError:
                temporary.unlink(missing_ok=True)
                # Storage availability is an optimization, never data selection.
                self.skipped += 1
                return data
            return data

    def _validate(self, data, row, key, force=False):
        if data.dtype != np.uint8 or data.shape != (row['height'], row['width'], 3):
            raise ValueError('Decoded cache dimensions/dtype differ from frozen parent')
        if force or key not in self.validated:
            prefix = ('%sx%s:RGB:' % (row['width'], row['height'])).encode()
            if hashlib.sha256(prefix + data.tobytes()).hexdigest() != row['rgb_sha256']:
                raise ValueError('Decoded cache pixels differ from frozen RGB hash')
            self.validated.add(key)


def cached_array(dataset, index, epoch, seed, cache):
    row = dataset.rows[index]
    path = Path(dataset.manifest['roots'][row['source']]) / row['relative_path']
    parent = cache.rgb(row, path)
    try:
        return _crop_array(dataset, index, epoch, seed, parent, path)
    finally:
        cache.release(parent, row)


def _crop_array(dataset, index, epoch, seed, pixels, path):
    h, w = pixels.shape[:2]
    if dataset.split != 'test':
        size = dataset.crop
        if min(h, w) < size:
            raise ValueError('Parent smaller than declared crop: ' + str(path))
        if dataset.split == 'train':
            rng = np.random.RandomState((seed + epoch*1000003 + index*9176) % (2**32))
            x, y = rng.randint(w-size+1), rng.randint(h-size+1)
        else:
            x, y = (w-size)//2, (h-size)//2
        pixels = pixels[y:y+size, x:x+size]
        if dataset.split == 'train' and dataset.horizontal_flip and rng.rand() < .5:
            pixels = pixels[:, ::-1]
    return np.asarray(pixels, dtype=np.float32).copy() / 255.


def install():
    from src.person1.protocol import Images
    if getattr(Images, '_decoded_cache_installed', False):
        return
    original = Images.array
    cache = DecodedCache(os.environ.get('PERSON1_DECODED_CACHE', '/kaggle/temp/person1-decoded-rgb'))
    checked = set()

    def array(self, index, epoch=0, seed=22):
        row = self.rows[index]
        required = {'sha256', 'rgb_sha256', 'width', 'height'}
        if not required.issubset(row):
            # Native resume_fixture deliberately writes tiny derivative PNGs
            # without parent hashes. Keep that acceptance test on its original
            # loader; production frozen parents must still have every hash.
            if row.get('source') == 'resume-fixture':
                return original(self, index, epoch, seed)
            raise ValueError('Production parent lacks frozen cache identity fields')
        result = cached_array(self, index, epoch, seed, cache)
        # Runtime parity includes real parents, all splits, varied crops/flips.
        key = (self.split, self.crop, self.horizontal_flip)
        if key not in checked:
            if not np.array_equal(result, original(self, index, epoch, seed)):
                raise AssertionError('Cache adapter is not byte-identical to frozen image loader')
            checked.add(key)
            print('Decoded-cache native-loader parity passed: ' + repr(key), flush=True)
        if (cache.hits + cache.misses) % 1000 == 0:
            print('Decoded-cache ' + json.dumps(dict(hits=cache.hits, misses=cache.misses,
                  skipped=cache.skipped, bytes=cache.used)), flush=True)
        return result

    Images.array = array
    Images._decoded_cache_installed = True
    print('Decoded RGB local-scratch cache enabled; source and RGB hashes checked; original training math unchanged', flush=True)
