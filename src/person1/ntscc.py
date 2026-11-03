"""Released NTSCC+ modules, objective and optimizer schedule; native channel.

Side-rate indices are serialized at evaluation. The author assumes reliable
delivery with LDPC/16QAM; it does not release a noisy side-bit decoder. This
limitation is recorded and prevents a certified complete-link ranking.
"""
import importlib
from types import SimpleNamespace
import torch
from torch import nn
from .imports import upstream


def build(options, device, source_dir):
    import natten
    from importlib.metadata import version
    if version('natten').split('+')[0] != '0.14.6':
        raise RuntimeError('Requires author NATTEN0.14.6')
    if not str(device).startswith('cuda'):
        raise RuntimeError('Native NTSCC+ requires CUDA')
    rates=[1,4,8,12,16,20,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320]
    config=SimpleNamespace(device=torch.device(device),logger=None,channel_type='awgn',
                           SNR=options['snr_db'],eta=options.get('eta',.2),multiple_rate=rates)
    with upstream(source_dir,('net','layer','channel','data','utils','main')):
        core=importlib.import_module('net.ntscc').NTSCC_plus(config)
        criterion=importlib.import_module('main').RateDistortionLoss(options.get('train_lambda',.72))

    class NTSCC(nn.Module):
        scheduler_per_step=True
        def __init__(self):
            super().__init__();self.core=core;self.criterion=criterion
            self.metadata=dict(source_commit='765c6f6c11cf3bbe249a26b624dbaf47a4f0676d',
                architecture='native NTSCC_plus / NatCheckerboard2',channel='native active-mask AWGN',
                training_parallelism='DataParallel; native normalization within each replica',
                side_transport='PNG serialized indices; author reliable side-channel assumption',
                complete_link_certified=False,rate_scope='main symbols plus separately reported assumed-reliable side symbols',
                optimizer='Adam excluding quantiles + auxiliary Adam; MultiStepLR600000 per update')
        def forward(self,image):
            out=self.core(image)
            loss=self.criterion(out,image)['loss']
            return dict(reconstruction=out['x_hat'],native_loss=loss.expand(len(image)),
                        complex_uses=image.new_full((len(image),),float(out['k'])/len(image)),
                        indexes=out['indexes'].reshape(len(image),-1))
        def loss(self,out,batch):return out['native_loss'].mean()
        def optimizer(self):
            return torch.optim.Adam([p for n,p in self.core.named_parameters() if not n.endswith('.quantiles')],lr=options.get('learning_rate',1e-4))
        def auxiliary_optimizer(self):
            return torch.optim.Adam([p for n,p in self.core.named_parameters() if n.endswith('.quantiles')],lr=1e-3)
        def aux_loss(self):return self.core.ntc.aux_loss()
        def scheduler(self,optimizer):
            return torch.optim.lr_scheduler.MultiStepLR(optimizer,[600000],gamma=.1)
    return NTSCC().to(device)
