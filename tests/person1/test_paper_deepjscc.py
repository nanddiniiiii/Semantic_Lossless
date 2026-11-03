import torch
from src.person1.paper_deepjscc import DeepJSCC,SameConv,SameTranspose


def test_figure2_shapes_kernels_and_power():
    model=DeepJSCC(dict(latent_channels=16,snr_db=1000,learning_rate=1e-4))
    kernels=[m for m in model.modules() if isinstance(m,(SameConv,SameTranspose))]
    assert len(kernels)==10 and all(m.kernel_size==(5,5) for m in kernels)
    received=[];handle=model.decoder.register_forward_pre_hook(lambda m,args:received.append(args[0].detach()))
    x=torch.rand(2,3,12,20);y=model(x);handle.remove()
    assert y['reconstruction'].shape==x.shape
    z=received[0].flatten(1);k=z.shape[1]/2
    assert torch.allclose(z.square().sum(1),torch.full((2,),k),rtol=1e-5)
    assert torch.allclose(y['complex_uses']/x[0].numel(),torch.full((2,),1/6))


def test_same_transpose_is_adjoint_for_even_geometry():
    for stride in (1,2):
        conv=SameConv(2,3,5,stride=stride,bias=False)
        trans=SameTranspose(3,2,5,stride=stride,bias=False);trans.weight.data.copy_(conv.weight.data)
        x=torch.randn(1,2,8,12,requires_grad=True);out=conv(x);probe=torch.randn_like(out)
        (out*probe).sum().backward()
        assert torch.allclose(x.grad,trans(probe),atol=1e-6)


def test_paper_pixel_mse_uses_denormalized_8bit_range():
    model=DeepJSCC({})
    loss=model.loss({'reconstruction':torch.zeros(1,3,4,4)},{'image':torch.ones(1,3,4,4)})
    assert loss.item()==255**2
