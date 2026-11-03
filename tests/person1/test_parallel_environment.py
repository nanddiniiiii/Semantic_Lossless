import pytest
from src.person1.torch_worker import validate_parallel_environment


def test_blocking_debug_flag_cannot_silently_stall_parallel_run(monkeypatch):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING','1')
    with pytest.raises(RuntimeError,match='two-GPU'):
        validate_parallel_environment({'require_two_gpus':True,'data_parallel':True})
    validate_parallel_environment({'require_two_gpus':False,'data_parallel':False})
    monkeypatch.delenv('CUDA_LAUNCH_BLOCKING')
    validate_parallel_environment({'require_two_gpus':True,'data_parallel':True})
