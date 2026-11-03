"""2019 DeepJSCC Figure 2 reproduction, not original-author source.

https://arxiv.org/abs/1809.01733v4, Fig 2 and equations 1/3/5.
All ten kernels are 5x5; TensorFlow SAME alignment is explicit. The paper
does not specify PReLU sharing/initializer details, so per-channel PReLU and
PyTorch initializers are disclosed implementation choices, not verified facts.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SameConv(nn.Conv2d):
    def forward(self, x):
        h, w = x.shape[-2:]
        s = self.stride[0]
        ph = max(0, ((h+s-1)//s-1)*s + 5-h)
        pw = max(0, ((w+s-1)//s-1)*s + 5-w)
        return super().forward(F.pad(x, (pw//2, pw-pw//2, ph//2, ph-ph//2)))


class SameTranspose(nn.ConvTranspose2d):
    def forward(self, x):
        # TF SAME is the adjoint of the asymmetrically padded forward op.
        y = super().forward(x)
        s = self.stride[0]
        extra = 5-s
        left = extra//2
        return y[..., left:left+x.shape[-2]*s, left:left+x.shape[-1]*s]


class DeepJSCC(nn.Module):
    def __init__(self, options):
        super().__init__()
        c = int(options.get('latent_channels', 16))
        if c < 1:
            raise ValueError('latent_channels must be positive')
        layers = []
        for a, b, s in [(3,16,2),(16,32,2),(32,32,1),(32,32,1),(32,c,1)]:
            layers.extend([SameConv(a,b,5,stride=s),nn.PReLU(b)])
        self.encoder = nn.Sequential(*layers)
        layers = []
        for i, (a,b,s) in enumerate([(c,32,1),(32,32,1),(32,32,1),(32,16,2),(16,3,2)]):
            layers.extend([SameTranspose(a,b,5,stride=s),nn.Sigmoid() if i==4 else nn.PReLU(b)])
        self.decoder = nn.Sequential(*layers)
        self.snr = float(options.get('snr_db',10))
        self.lr = float(options.get('learning_rate',1e-4))
        self.metadata = dict(lineage='paper-faithful reproduction; original-author code unverified',
            paper='https://arxiv.org/abs/1809.01733v4', architecture='Figure2',
            encoder_filters=[16,32,32,32,c],decoder_filters=[32,32,32,16,3],
            kernel_sizes=[5]*10,encoder_strides=[2,2,1,1,1],decoder_strides=[1,1,1,2,2],
            prelu='per-channel; paper sharing unspecified',initializers='PyTorch defaults; paper unspecified',
            rate_scope='all complex latent pairs; no framing/CSI overhead',native_input_range='[0,1]',
            objective='MSE after native255 denormalization, paper Eq3')

    def forward(self, image):
        if image.shape[-2]%4 or image.shape[-1]%4:
            raise ValueError('Explicit alignment padding to multiple4 is required')
        z = self.encoder(image)
        flat = z.flatten(1)
        if flat.shape[1]%2:
            raise ValueError('Odd real latent length cannot form complex symbols')
        k = flat.shape[1]//2
        signal = flat * (k / flat.square().sum(1,keepdim=True).clamp_min(1e-12)).sqrt()
        received = signal + torch.randn_like(signal)*(10**(-self.snr/20)/2**.5)
        reconstruction = self.decoder(received.reshape_as(z))
        return {'reconstruction':reconstruction,
                'complex_uses':image.new_full((len(image),),float(k))}

    def loss(self, output, batch):
        return F.mse_loss(output['reconstruction']*255.,batch['image']*255.)

    def optimizer(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
