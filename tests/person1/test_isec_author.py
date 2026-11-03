"""Focused CPU checks of the ISEC author equations and adapter contract."""
from pathlib import Path

import pytest
import torch
from torch import nn

from src.person1.isec import _refine, build


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return self.scale * x


@pytest.mark.parametrize("test_snr", [7, 13])
def test_refinement_matches_author_eval_update_with_snr_mismatch(test_snr):
    encoder, decoder, denoiser = Scale(1.3), Scale(0.8), Scale(0.15)
    received = torch.tensor([[[[0.4, -0.2]]]])
    sigma = 10 ** (-test_snr / 20)
    base_var = 10 ** (-10 / 10)
    ratio = sigma**2 / base_var
    native = received.detach().clone().requires_grad_()
    for _ in range(3):
        obj = (encoder(decoder(native)) - received).square().sum() / (2 * sigma**2)
        obj.backward()
        with torch.no_grad():
            # Direct transcription of pinned author eval.py lines 209-220.
            grad = -native.grad + 1.0 * max(0.1, ratio**2) * denoiser(native)
            delta = 1.5 if ratio > 1 else 1.0
            step = 0.002 / max(0.1, ratio**delta)
            native = (native + step * grad).detach().requires_grad_()
    actual = _refine(received, encoder, decoder, denoiser,
                     test_snr_db=test_snr, training_snr_db=10, iterations=3,
                     learning_rate=0.002, alpha=1.0, delta=1.5)
    torch.testing.assert_close(actual, native.detach(), rtol=1e-6, atol=1e-7)


def _fake_source(root: Path):
    package = root / "models"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "autoencoders.py").write_text(
        "import torch\nfrom torch import nn\n"
        "class Encoder(nn.Module):\n"
        " def __init__(self, width, hidden, conv, residual):\n"
        "  super().__init__(); self.layer=nn.Conv2d(3,width,1)\n"
        " def forward(self,x): return self.layer(x)\n"
        "class Decoder(nn.Module):\n"
        " def __init__(self, width, hidden, conv, residual):\n"
        "  super().__init__(); self.layer=nn.Conv2d(width,3,1)\n"
        " def forward(self,x): return self.layer(x)\n"
    )
    (package / "bfcnn.py").write_text(
        "from torch import nn\n"
        "class BF_CNN(nn.Module):\n"
        " def __init__(self,padding,kernels,kernel_size,layers,width):\n"
        "  super().__init__(); self.layer=nn.Conv2d(width,width,1,bias=False)\n"
        " def forward(self,x): return self.layer(x)\n"
    )


def test_stage_losses_optimizers_and_rate(tmp_path):
    _fake_source(tmp_path)
    options = {"latent_channels": 6, "snr_db": 10, "iterations": 1, "epochs": 150}
    model = build(options, "cpu", tmp_path)
    image = torch.rand(2, 3, 4, 4)
    layouts=[]
    model.encoder.register_forward_pre_hook(lambda module,args:layouts.append(args[0].is_contiguous()))
    hwc=torch.rand(2,4,4,3).permute(0,3,1,2)
    assert not hwc.is_contiguous()
    model(hwc)
    assert layouts[-1]
    model.train()
    jscc = model(image)
    assert jscc["native_loss"].shape == (2,)
    assert jscc["reconstruction"].shape == image.shape
    torch.testing.assert_close(jscc["complex_uses"], torch.full((2,), 48.0))
    assert model.optimizer().param_groups[0]["betas"] == (0.0, 0.9)
    assert model.scheduler(model.optimizer()).step_size == 30
    model.stage = "denoiser"
    denoiser = model(image)
    assert denoiser["native_loss"].shape == (2,)
    assert model.loss(denoiser, {"image": image}).ndim == 0
    assert model.optimizer().param_groups[0]["betas"] == (0.9, 0.999)
    assert model.scheduler(model.optimizer()).step_size == 75
    model.eval()
    with torch.no_grad():
        refined = model(image)
    assert refined["reconstruction"].shape == image.shape


def test_missing_or_wrong_source_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        build({}, "cpu", tmp_path)
    _fake_source(tmp_path)
    (tmp_path / "source_receipt.json").write_text(
        '{"repository":"another/repo","commit":"deadbeef"}'
    )
    with pytest.raises(ValueError, match="approved author revision"):
        build({}, "cpu", tmp_path)


def test_native_bf_statistics_survive_replica_storage_and_preserve_math():
    import copy
    import importlib.util
    from src.person1.isec import preserve_running_statistics
    source=Path(__file__).resolve().parents[2]/'.upstream/isec/models/bfcnn.py'
    if not source.is_file():
        pytest.skip('Pinned author source required for native equivalence check')
    spec=importlib.util.spec_from_file_location('native_bf_stat_test',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    torch.manual_seed(21)
    reference=native.BF_CNN(1,4,3,4,2)
    replica=preserve_running_statistics(copy.deepcopy(reference))
    # Emulate DP: separate parameter handles share the persistent base storage.
    persistent=[p.detach() for p in replica.running_sd]
    replica.running_sd=nn.ParameterList([nn.Parameter(p.detach(),requires_grad=False) for p in replica.running_sd])
    x=torch.randn(2,2,8,8,requires_grad=True);y=x.detach().clone().requires_grad_()
    expected=reference(x);actual=replica(y)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    expected.square().mean().backward();actual.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    for (name,a),(other,b) in zip(reference.named_parameters(),replica.named_parameters()):
        assert name==other
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    for saved,updated,native_stat in zip(persistent,replica.running_sd,reference.running_sd):
        torch.testing.assert_close(saved,native_stat,rtol=0,atol=0)
        assert not torch.equal(saved,torch.ones_like(saved))
        assert not updated.requires_grad
    replica.eval()
    z=torch.randn(2,2,8,8,requires_grad=True)
    replica(z).sum().backward()  # Eval hooks must not invalidate autograd tensors.
    assert z.grad is not None


def test_stage_switch_keeps_native_statistics_nontrainable():
    from types import SimpleNamespace
    from src.person1.torch_worker import stage_model
    denoiser=nn.Linear(2,2)
    denoiser.running_sd=nn.ParameterList([nn.Parameter(torch.ones(1),requires_grad=False)])
    model=SimpleNamespace(stage='jscc',encoder=nn.Linear(2,2),decoder=nn.Linear(2,2),denoiser=denoiser)
    stage_model(model,'denoiser')
    assert model.denoiser.weight.requires_grad
    assert not model.denoiser.running_sd[0].requires_grad
    assert not model.encoder.weight.requires_grad
    stage_model(model,'jscc')
    assert model.encoder.weight.requires_grad
    assert not model.denoiser.weight.requires_grad
