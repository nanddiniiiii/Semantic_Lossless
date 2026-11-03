import shutil
import pytest
from PIL import Image
from src.dataset import freeze


def test_parallel_checks_preserve_exact_membership_hashes_and_duplicates(tmp_path):
    roots={}
    for i,name in enumerate(('div2k_train','flickr2k_train','div2k_val','kodak_test')):
        folder=tmp_path/name;folder.mkdir();roots[name]=str(folder)
        Image.new('RGB',(17+i,19+i),(i,4+i,9+i)).save(folder/'image.png')
    shutil.copy2(tmp_path/'div2k_train/image.png',tmp_path/'flickr2k_train/duplicate.png')
    one=freeze(roots,tmp_path/'serial.json',strict_counts=False,workers=1)
    two=freeze(roots,tmp_path/'parallel.json',strict_counts=False,workers=2)
    assert one==two
    assert len(two['training_duplicates_retained'])==1
    shutil.copy2(tmp_path/'div2k_train/image.png',tmp_path/'kodak_test/leak.png')
    with pytest.raises(ValueError,match='protected split'):
        freeze(roots,tmp_path/'leak.json',strict_counts=False,workers=2)
