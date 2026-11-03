"""Independent arithmetic edge cases and tiny true native graph coverage."""
import hashlib
import json
import torch
import pytest

from scripts.flops_isec_audit import Census, hand_checks, inference, make_model, rule, training_case, verify_image_manifest


def test_hand_counts_and_unknown_fail_closed():
    assert hand_checks()['passed']
    with pytest.raises(ValueError, match='Unknown operator'):
        rule('aten.unreviewed.default', (), {}, torch.empty(1))


def test_boolean_accumulation_counts_broadcast_destinations():
    target=torch.zeros(3,4)
    index=torch.tensor([True,False,True])
    assert rule('aten.index_put.default',(target,[index],torch.ones(4),True),{},target)[0]==8
    with pytest.raises(ValueError, match='Boolean first-axis'):
        rule('aten.index_put.default',(target,[torch.tensor([0,0])],torch.ones(2,4),True),{},target)


def test_unitnorm_records_projected_and_empty_subsets():
    model=make_model({'latent_channels':6},None,'jscc',False)
    unitnorm=model.encoder[-1]
    for values,expected in [(torch.tensor([[.01,.01],[3.,4.]]),1),(torch.full((2,2),.01),0)]:
        c=Census()
        with c:
            unitnorm(values)
        assert len(c.norms)==2
        assert c.norms[0]['input_shape']==[2,2]
        assert c.norms[1]['input_shape']==[expected,2]


def test_native_refinement_and_both_training_stages():
    torch.set_num_threads(2)
    options={'latent_channels':6,'iterations':2,'snr_db':10.}
    out=inference(options,None,torch.rand(1,3,8,8),22)
    assert out['logical_scope_complete']
    assert not out['parameter_gradients_accumulated']
    # Initial encoder plus exactly two refinement encoders, two norms each.
    assert len(out['unitnorm_calls'])==6
    multipliers=[r for r in out['ledger'] if r['operator']=='python.unitnorm_multiplier']
    assert sum(r['calls'] for r in multipliers)==3
    assert sum(r['basic'] for r in multipliers)==3
    assert sum(r['special'] for r in multipliers)==3
    assert all(r['component']=='encoder' for r in multipliers)
    metadata=[r for r in out['ledger'] if r['operator']=='python.complex_uses_metadata']
    assert len(metadata)==1 and metadata[0]['metadata_basic']==1
    assert metadata[0]['basic']==metadata[0]['special']==0
    assert out['total']['metadata_basic']==1
    back=[r for r in out['ledger'] if r['operator']=='aten.native_batch_norm_backward.default']
    assert sum(r['calls'] for r in back)==36
    assert all(r['example']['inputs'][-1]==[True,False,False] for r in back)
    cases=training_case(options,None,8,8,22)
    assert set(cases)=={'jscc','denoiser'}
    for case in cases.values():
        assert case['logical_scope_complete']
        assert case['root']['mean_forward_basic']==2
        assert case['root']['mean_backward_basic']==2
        assert case['total']['metadata_basic']==2
        assert case['replicas'][0]['summary']['adam']['logical_flops']>0
        assert 'adam' not in case['replicas'][1]['summary']
        assert all(r['backward_seed']==.5 and not r['local_extra_mean'] for r in case['replicas'])
        for replica in case['replicas']:
            host=[r for r in replica['ledger'] if r['operator']=='python.unitnorm_multiplier']
            assert sum(r['calls'] for r in host)==1
            assert sum(r['basic'] for r in host)==1
            assert sum(r['special'] for r in host)==1


def test_frozen_image_verification_rejects_modified_manifest(tmp_path):
    from PIL import Image
    image=Image.new('RGB',(2,3),(7,11,13))
    path=tmp_path/'fixture.png';image.save(path)
    rows=[dict(source='kodak_test',split='test',relative_path='fixture.png',width=2,height=3,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               rgb_sha256=hashlib.sha256(b'2x3:RGB:'+image.tobytes()).hexdigest())]
    manifest=tmp_path/'fixture_manifest.json'
    payload=dict(images=rows,identity=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    manifest.write_text(json.dumps(payload))
    assert verify_image_manifest(path,manifest)['canonical_rgb_sha256_match']
    payload['images'][0]['split']='train'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='manifest identity mismatch'):
        verify_image_manifest(path,manifest)
