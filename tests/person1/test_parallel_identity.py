from types import SimpleNamespace
import pytest
from src.launch import notebook_slug


def test_independent_training_identity_is_scoped_to_assigned_model():
    assert notebook_slug('1',SimpleNamespace())=='semantic-notebook-1'
    args=SimpleNamespace(slug='semantic-notebook-1-deepjscc',model='deepjscc',mode='train')
    assert notebook_slug('1',args)==args.slug
    args.mode='pilot'
    with pytest.raises(ValueError):notebook_slug('1',args)
    args.mode='train';args.slug='someone-elses-notebook'
    with pytest.raises(ValueError):notebook_slug('1',args)
