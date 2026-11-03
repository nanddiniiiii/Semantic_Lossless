import torch
import pytest
from src.person1.torch_worker import verify_replica_reduction


def test_unused_native_parameter_hook_preserves_none_and_checks_real_gradients():
    from src.person1.torch_worker import capture_parameter_gradient
    captured={0:{},1:{}}
    capture_parameter_gradient(None,captured,0,'unused_auxiliary')
    assert captured[0]=={'unused_auxiliary':None}
    capture_parameter_gradient(torch.tensor([2.]),captured,0,'weight')
    assert torch.equal(captured[0]['weight'],torch.tensor([2.]))
    with pytest.raises(AssertionError,match='Duplicate'):
        capture_parameter_gradient(torch.tensor([2.]),captured,0,'weight')


@pytest.mark.parametrize('materialized',[False,True])
def test_explicit_unused_gradients_preserve_zero_or_none_semantics(materialized):
    actual,shards,coefficients=fixture()
    shards[0]['aux']=None;shards[1]['aux']=None
    if materialized:actual['aux']=torch.zeros(2)
    assert verify_replica_reduction(actual,shards,coefficients)==0.


def test_used_gradient_cannot_be_lost_under_unused_parameter_handling():
    actual,shards,coefficients=fixture();shards[0]['lost']=torch.ones(1);shards[1]['lost']=None
    with pytest.raises(AssertionError,match='lost'):verify_replica_reduction(actual,shards,coefficients)


def test_deepjscc_uses_its_global_mse_verification_contract(monkeypatch):
    from src.person1 import torch_worker
    calls=[]
    monkeypatch.setattr(torch_worker,'independent_gradient_probe',lambda *a:calls.append(1))
    torch_worker.gradient_probe(None,{'algorithm':'deepjscc'},None)
    assert calls==[1]


def test_replica_loss_hooks_do_not_share_or_overwrite_ordinals():
    from src.person1.torch_worker import capture_replica_loss
    class Loss(torch.nn.Module):
        def forward(self,x):return {'native_loss':2*x}
    root=Loss();left=root._replicate_for_data_parallel();right=root._replicate_for_data_parallel()
    coefficients={}
    capture_replica_loss(left,0,coefficients);capture_replica_loss(right,1,coefficients)
    left(torch.ones(1,requires_grad=True))['native_loss'].backward(torch.tensor([.25]))
    right(torch.ones(1,requires_grad=True))['native_loss'].backward(torch.tensor([.75]))
    assert torch.equal(coefficients[0],torch.tensor([.25]))
    assert torch.equal(coefficients[1],torch.tensor([.75]))
    assert not root._forward_hooks


def fixture():
    left=torch.tensor([1000.,-.25,1e-7]);right=torch.tensor([-999.,.5,2e-7])
    return {'w':left+right},{0:{'w':left},1:{'w':right}},{0:torch.tensor([.5]),1:torch.tensor([.5])}


def test_exact_actual_replica_sum_is_accepted():
    actual,shards,coefficients=fixture()
    assert verify_replica_reduction(actual,shards,coefficients)==0.


@pytest.mark.parametrize('fault',['unequal_weights','missing_shard','wrong_reduction','missing_parameter','no_gradients'])
def test_incorrect_weighting_or_aggregation_is_rejected(fault):
    actual,shards,coefficients=fixture()
    if fault=='unequal_weights':coefficients[0]=torch.tensor([.4])
    elif fault=='missing_shard':del shards[1]
    elif fault=='wrong_reduction':actual['w']*=2
    elif fault=='missing_parameter':del shards[1]['w']
    else:actual.clear()
    with pytest.raises(AssertionError):verify_replica_reduction(actual,shards,coefficients)
