import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageOps
from scripts.person1_decoded_cache import DecodedCache, cached_array, install
from src.person1.protocol import Images, resume_fixture
from src.dataset import _parent_identity


class CacheParity(unittest.TestCase):
    def test_ram_filesystem_fallback_and_clean_page_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'one.png'
            pixels=np.arange(12*13*3,dtype=np.uint8).reshape(12,13,3);Image.fromarray(pixels).save(path)
            w,h,rgb,sha=_parent_identity(path)
            row=dict(sha256=sha,rgb_sha256=rgb,width=w,height=h)
            cache=DecodedCache(root/'cache',reserve_bytes=0)
            cache.rgb(row,path)
            with patch('os.posix_fadvise',create=True) as advise,patch('os.fdatasync',create=True),patch('os.POSIX_FADV_DONTNEED',4,create=True):
                mapped=cache.rgb(row,path);cache.release(mapped,row)
                self.assertTrue(mapped._mmap.closed)
                self.assertEqual(advise.call_args.args[1:],(0,0,4))
            cache.disk_backed=False
            # A RAM-backed directory must not even reuse a preexisting cache.
            np.save(cache.directory/(sha+'.npy'),np.zeros_like(pixels))
            self.assertTrue(np.array_equal(cache.rgb(row,path),pixels))
            self.assertEqual(cache.skipped,1)

    def test_actual_native_resume_fixture_uses_original_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);path = root/'one.png'
            Image.fromarray(np.random.RandomState(9).randint(0,256,(47,53,3),dtype=np.uint8)).save(path)
            width,height,pixels_sha,file_sha = _parent_identity(path)
            rows = [dict(source='fixture',split='train',relative_path=path.name,
                sha256=file_sha,rgb_sha256=pixels_sha,width=width,height=height)]
            identity=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(images=rows,identity=identity,roots={'fixture':folder})))
            options=dict(manifest=str(manifest),crop=31,horizontal_flip=True,seed=22)
            native=root/'native';native.mkdir();cached=root/'cached';cached.mkdir()
            native_options=resume_fixture(options,native)
            original=Images.array
            try:
                with patch.dict('os.environ',{'PERSON1_DECODED_CACHE':str(root/'cache')}):install()
                cached_options=resume_fixture(options,cached)
                a=Images(native_options['manifest'],'train',31,True)
                b=Images(cached_options['manifest'],'train',31,True)
                for epoch in [0,1,76]:
                    for i in range(len(a)):
                        self.assertTrue(np.array_equal(original(a,i,epoch,22),b.array(i,epoch,22)))
                b.rows[0]['source']='production-with-missing-hash'
                with self.assertRaisesRegex(ValueError,'Production parent lacks'):
                    b.array(0)
            finally:
                Images.array=original
                if hasattr(Images,'_decoded_cache_installed'):del Images._decoded_cache_installed

    def test_native_loader_parity_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.RandomState(5)
            for name, mode, orientation in [('portrait.png', 'RGB', None), ('landscape.jpg', 'RGB', 6),
                                            ('gray.png', 'L', None), ('rgba.png', 'RGBA', None)]:
                shape = (51, 73) if mode == 'L' else (73, 51, len(mode))
                pixels = rng.randint(0, 256, shape, dtype=np.uint8)
                image = Image.fromarray(pixels)
                if orientation:
                    exif = image.getexif();exif[274] = orientation
                    image.save(root / name, exif=exif)
                else:
                    image.save(root / name)
            rows = []
            for p in sorted(root.iterdir()):
                with Image.open(p) as handle:
                    rgb = np.asarray(ImageOps.exif_transpose(handle).convert('RGB'))
                width, height, pixels_sha, file_sha = _parent_identity(p)
                rows.append(dict(source='fixture', split='train', relative_path=p.name,
                     sha256=file_sha, rgb_sha256=pixels_sha, width=width, height=height))
            rows.append(dict(rows[0]))  # Retain duplicate memberships/exposures.
            manifest = dict(images=rows, roots={'fixture': str(root)},
                identity=hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
            path = root/'manifest.json';path.write_text(json.dumps(manifest))
            cache = DecodedCache(root/'cache', reserve_bytes=0)
            for split in ['train', 'val', 'test']:
                dataset = Images(path, 'train', crop=31, horizontal_flip=True)
                dataset.split = split
                for epoch in [0, 76, 149]:
                    for seed in [22, 79]:
                        for index in range(len(dataset)):
                            expected = dataset.array(index, epoch, seed)
                            actual = cached_array(dataset, index, epoch, seed, cache)
                            self.assertTrue(np.array_equal(expected, actual))
                            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(cache.misses, 4)
            self.assertGreater(cache.hits, 60)
            with patch.object(cache,'release',wraps=cache.release) as release:
                actual=cached_array(dataset,0,3,22,cache)
                mapped=release.call_args.args[0]
                self.assertTrue(mapped._mmap.closed)
                self.assertTrue(np.array_equal(actual,dataset.array(0,3,22)))
            # Persisted cache corruption must fail, never silently change pixels.
            target = cache.directory/(rows[0]['sha256']+'.npy')
            data = np.load(target);data[0, 0, 0] ^= 1
            np.save(target, data)
            fresh = DecodedCache(cache.directory, reserve_bytes=0)
            with self.assertRaisesRegex(ValueError, 'pixels differ'):
                fresh.rgb(rows[0], root/rows[0]['relative_path'])

    def test_storage_budget_fallback_keeps_every_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);path = root/'one.png'
            rgb = np.arange(12*13*3, dtype=np.uint8).reshape(12, 13, 3)
            Image.fromarray(rgb).save(path)
            width, height, pixels_sha, file_sha = _parent_identity(path)
            row = dict(sha256=file_sha, rgb_sha256=pixels_sha, width=width, height=height)
            cache = DecodedCache(root/'cache', budget_bytes=0, reserve_bytes=0)
            for _ in range(2):self.assertTrue(np.array_equal(cache.rgb(row, path), rgb))
            self.assertEqual(cache.skipped, 2)
            self.assertEqual(list(cache.directory.glob('*.npy')), [])


if __name__ == '__main__': unittest.main()
