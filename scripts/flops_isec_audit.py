"""Checkpoint/input-specific, fail-closed logical arithmetic audit of native ISEC.

CPU execution only. Checkpoints are read, never saved or modified. The logical
algorithms below declare fused ATen primitives explicitly, rather than claiming
to count CPU/GPU instructions. Two real batch-one training graphs compose the
requested global-batch-two DataParallel logical case.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from src.person1.isec import build
from scripts.flops_deepjscc_audit import arithmetic_rule as shared_rule, description, digest, tensors


FIELDS = ('basic', 'special', 'comparisons', 'random_draws')
EXTRA_ZERO = {
    'aten.empty.memory_format', 'aten.unsqueeze.default', 'aten.squeeze.dim',
    'aten.new_zeros.default', 'aten.as_strided.default', 'aten.new_empty_strided.default',
    'aten.zeros.default', 'aten.select.int', 'aten.t.default', 'aten.transpose.int',
    'aten.permute.default', 'aten._to_copy.default', 'aten.empty_strided.default',
    'aten.alias.default', 'aten.index.Tensor', 'aten.nonzero.default',
    'aten.masked_fill_.Scalar', 'aten.masked_fill.Scalar',
}
CONVENTION = {
    'scope': 'Actual CPU-dispatched graph plus explicit Python scalar formulas; input/checkpoint/seed-specific logical arithmetic, not universal FLOPs.',
    'basic': 'add/sub/mul/div/negate/reciprocal each one; convolution dense full-kernel MAC=2 including zero padding; bias add separate',
    'special': 'sqrt, exp, general power, tanh each one logical special primitive, reported separately',
    'batchnorm': 'Canonical two-pass variance, sqrt then reciprocal; affine separate; training running-statistics updates included. Backward reuses centered=x-mean, honors requested gradient masks. Local backend kernels may use another algorithm.',
    'variance': 'Mean then centered squares then sum/divide:4N basic; correction denominator shape arithmetic excluded',
    'norm_p2': 'N squares plus N-R reduction adds plus R sqrt; actual composed autograd backward counted by dispatched operations',
    'unitnorm_host': 'Every actually executed native UnitNorm invocation evaluates max_power*np.sqrt(latent_elements_per_image):1 basic multiply +1 special sqrt, beyond the dispatched tensor norms. Counted by native module hook.',
    'bilinear': 'align_corners=False only; nested value blends9/output, gradient four contributions each two multiplies and scatter-add12/output; axis weights precomputed canonically5*(OH+OW)+2 basic per operator, including two scale divisions. Coordinate clamps comparisons counted. Index casts/floors/control excluded as coordinate indexing; no special floor primitive claimed.',
    'tanh': 'one special primitive forward; g*(1-y*y):3 basic backward',
    'relu': 'one comparison per output; selection zero arithmetic',
    'random': 'normal draws counted; RNG internal algorithm excluded',
    'zero': 'allocation/copy/views/indexing(selection or replacement, not accumulation)/integer shape and control arithmetic excluded; floating rate metadata arithmetic disclosed as metadata_basic but excluded from signal logical FLOPs',
    'not_counted': ['hardware instructions/FMA fusion/backend convolution algorithms', 'memory traffic', 'RNG internals', 'device transfer and GPU replication', 'integer/control/index arithmetic'],
    'hardware_scope_complete': False,
}


def rule(name, args, kwargs, output):
    """Return basic,special,comparisons,random,rule; unknowns raise immediately."""
    out = tensors(output)
    n = out[0].numel() if out else 0
    if name in EXTRA_ZERO:
        return 0, 0, 0, 0, 'memory/view/index/control/selection; no floating arithmetic'
    if name in {'aten.index_put.default', 'aten.index_put_.default'}:
        if kwargs.get('accumulate', args[3] if len(args)>3 else False):
            target,indices,values=args[:3]
            if len(indices)!=1 or indices[0] is None or indices[0].dtype!=torch.bool or indices[0].ndim!=1 or indices[0].numel()!=target.shape[0]:
                raise ValueError('Only Boolean first-axis accumulating index_put reviewed')
            selected=int(indices[0].sum().item())
            additions=selected*math.prod(target.shape[1:])
            return additions,0,0,0,'Boolean first-axis scatter accumulation:one add per selected destination, including broadcast values'
        return 0, 0, 0, 0, 'index replacement; accumulate=False'
    if name in {'aten.eq.Scalar', 'aten.lt.Scalar', 'aten.le.Scalar', 'aten.gt.Tensor', 'aten.ge.Tensor'}:
        return 0, 0, n, 0, 'one comparison per output'
    if name in {'aten.relu.default', 'aten.relu_.default', 'aten.threshold_backward.default'}:
        return 0, 0, n, 0, 'threshold comparison; selection zero arithmetic'
    if name in {'aten.leaky_relu.default', 'aten.leaky_relu_.default', 'aten.leaky_relu_backward.default'}:
        return n, 0, n, 0, 'dense slope-multiply candidate and sign comparison'
    if name == 'aten.tanh.default':
        return 0, n, 0, 0, 'one tanh special primitive'
    if name == 'aten.tanh_backward.default':
        return 3*n, 0, 0, 0, 'g*(1-y*y)'
    if name in {'aten.mean.dim', 'aten.mean.default'}:
        x = args[0]
        if n and x.numel() == 0:
            raise ValueError('Empty mean has no declared finite arithmetic')
        return x.numel(), 0, 0, 0, 'N-R reduction adds + R divisions'
    if name == 'aten.var.correction':
        x = args[0]
        correction = kwargs.get('correction', 1)
        correction = 1 if correction is None else correction
        if n and x.numel()/n <= correction:
            raise ValueError('Nonpositive variance denominator')
        return 4*x.numel(), 0, 0, 0, 'two-pass centered variance: mean N + center N + square N + reduce/normalize N'
    if name == 'aten.linalg_vector_norm.default':
        if len(args)>1 and args[1] != 2:
            raise ValueError('Only p2 vector norm reviewed')
        return 2*args[0].numel()-n, n, 0, 0, 'N squares, N-R additions, R sqrt (empty subset zero)'
    if name == 'aten.native_batch_norm.default':
        x, gamma, beta, running_mean, running_var, training, momentum, eps = args
        total = x.numel(); c = x.shape[1]; m = total//c
        affine = total*(int(gamma is not None)+int(beta is not None))
        if training:
            if m <= 1:
                raise ValueError('Training BN requires >1 value/channel')
            basic = 6*total + affine + 2*c  # var_mean4N + centering/normalizing2N + eps/reciprocal2C
            host = 0
            if running_mean is not None:
                basic += 3*c
                host = 1  # 1-momentum, shared across two updates
            if running_var is not None:
                basic += 4*c
                host = 2  # 1-momentum, shape ratio n/(n-1); denominator integer arithmetic excluded
            return basic+host, c, 0, 0, 'train BN:two-pass var_mean, sqrt+reciprocal, normalize+affine, running updates and host constants'
        if running_mean is None or running_var is None:
            raise ValueError('Eval BN requires running statistics')
        return 2*total+affine+2*c, c, 0, 0, 'eval BN:(x-mean)*invstd*gamma+beta; eps, sqrt, reciprocal'
    if name == 'aten.native_batch_norm_backward.default':
        g, x, gamma, running_mean, running_var, save_mean, save_invstd, training, eps, mask = args
        total=x.numel(); c=x.shape[1]
        dx,dg,db=map(bool,mask)
        if dg and gamma is None:
            raise ValueError('BN dgamma without gamma not reviewed')
        basic=special=0
        if training:
            if dx or db:
                basic += total-c  # s1=sum(g)
            if dx or dg:
                basic += 3*total-c  # centered, multiply, sum
            if dx:
                basic += c+3*c+c+4*total+1  # gradmean,projscale,gradscale,dx,host1/m
            if dg:
                basic += c
            return basic,0,0,0,'train BN backward requested masks; centered reused; scalar1/m included when dx requested'
        if dx or dg:
            basic += 2*c; special += c  # eval recomputes invstd
        if dx:
            basic += total+c  # gradscale + gx
        if dg:
            basic += 3*total  # centered*gradient reduce then invstd
        if db:
            basic += total-c
        return basic,special,0,0,'eval BN backward masks; sqrt+reciprocal recomputed; dx=g*(gamma*invstd)'
    if name in {'aten.upsample_bilinear2d.default', 'aten.upsample_bilinear2d_backward.default'}:
        if len(args)>2 and args[2] is True and name.endswith('2d.default'):
            raise ValueError('align_corners=True not reviewed')
        backward='backward' in name
        if backward:
            if args[3]:
                raise ValueError('align_corners=True not reviewed')
            oh,ow=args[0].shape[-2:]
            values=args[0].numel()
        else:
            oh,ow=output.shape[-2:]
            values=output.numel()
        coords=5*(oh+ow)+2
        return (12 if backward else 9)*values+coords,0,oh+ow,0,'nested bilinear values '+('12N gradient scatter-add' if backward else '9N')+' + canonical axis coordinates5(OH+OW)+2; coordinate lower clamps comparisons'
    # Integer tensors such as BN num_batches_tracked are explicit control state.
    if name in {'aten.add_.Tensor','aten.add.Tensor','aten.mul.Tensor','aten.sub.Tensor'} and out and not out[0].is_floating_point():
        return 0,0,0,0,'integer/control arithmetic excluded'
    return shared_rule(name,args,kwargs,output)


class Census(TorchDispatchMode):
    def __init__(self):
        self.phase='forward'; self.component='adapter_channel_loss'
        self.groups={}; self.norms=[]

    def add(self, operator, count, info=None):
        basic,special,comparison,random,formula=count
        key=(self.phase,self.component,operator,formula)
        row=self.groups.setdefault(key,dict(phase=self.phase,component=self.component,operator=operator,
            calls=0,basic=0,special=0,comparisons=0,random_draws=0,rule=formula,example=info))
        row['calls']+=1
        for field,value in zip(FIELDS,(basic,special,comparison,random)):
            row[field]+=value

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        out=func(*args,**kwargs)
        name=str(func)
        try:
            counted=rule(name,args,kwargs,out)
        except (ValueError,TypeError,IndexError) as error:
            raise RuntimeError(f'Unreviewed arithmetic: {name}, phase={self.phase}, component={self.component}: {error}') from error
        self.add(name,counted,{'inputs':description(args),'kwargs':description(kwargs),'outputs':description(out)})
        if name=='aten.linalg_vector_norm.default':
            self.norms.append(dict(phase=self.phase,component=self.component,input_shape=list(args[0].shape),
                                   output_shape=list(out.shape),input_elements=args[0].numel()))
        return out

    def host(self,name,basic=0,special=0,comparisons=0,rule_text='explicit Python scalar arithmetic',metadata_basic=0):
        self.add('python.'+name,(basic,special,comparisons,0,rule_text))
        if metadata_basic:
            row=self.groups[(self.phase,self.component,'python.'+name,rule_text)]
            row['metadata_basic']=row.get('metadata_basic',0)+metadata_basic

    def report(self):
        rows=list(self.groups.values()); summary={}
        for phase in sorted({r['phase'] for r in rows}):
            subset=[r for r in rows if r['phase']==phase]
            totals={f:sum(r[f] for r in subset) for f in FIELDS}
            totals['metadata_basic']=sum(r.get('metadata_basic',0) for r in subset)
            totals['logical_flops']=totals['basic']+totals['special']
            summary[phase]=totals
        total={f:sum(r[f] for r in rows) for f in FIELDS}
        total['metadata_basic']=sum(r.get('metadata_basic',0) for r in rows)
        total['logical_flops']=total['basic']+total['special']
        return dict(summary=summary,total=total,ledger=rows,unitnorm_calls=self.norms,
                    logical_scope_complete=True,unknown_operators=[])


def install_hooks(model,census):
    handles=[]
    for name in ('encoder','decoder','denoiser'):
        module=getattr(model,name)
        def pre(mod,args,component=name):
            census.component=component
        def post(mod,args,out):
            census.component='adapter_channel_loss'
        handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    def unitnorm_host(module,inputs):
        census.host('unitnorm_multiplier',1,1,
            rule_text='native max_power*np.sqrt(latent_elements_per_image):one Python/NumPy multiply+one scalar sqrt per actually executed UnitNorm invocation')
    # The pinned native OpenImages Encoder ends in exactly this UnitNorm. A
    # source/architecture change must fail rather than silently lose its cost.
    if type(model.encoder[-1]).__name__!='UnitNorm':
        raise RuntimeError('Native UnitNorm location changed; host arithmetic requires review')
    handles.append(model.encoder[-1].register_forward_pre_hook(unitnorm_host))
    return handles


def adam_host(census,optimizer):
    module=importlib.import_module('torch.optim.adam')
    source=inspect.getsource(module._single_tensor_adam)
    required=['exp_avg.lerp_(grad, 1 - device_beta1)','bias_correction1 = 1 - beta1**step',
              'bias_correction2 = 1 - beta2**step','bias_correction2_sqrt = bias_correction2**0.5',
              'param.addcdiv_(exp_avg, denom, value=-step_size)']
    if not all(term in source for term in required):
        raise RuntimeError('Adam host source changed; independent scalar review required')
    active=sum(p.grad is not None for group in optimizer.param_groups for p in group['params'])
    census.host('adam_bias_corrections',6*active,3*active,
                rule_text='per active tensor:1-beta1,1-beta2, two bias-correction subtracts, lr/bc1, negate; beta1**step,beta2**step,sqrt(bc2); same declared DeepJSCC convention')
    return hashlib.sha256(source.encode()).hexdigest()


def make_model(options,state,stage,training):
    model=build(options,'cpu',ROOT/'.upstream/isec')
    if state is not None:
        model.load_state_dict(state,strict=True)
    model.stage=stage; model.train(training)
    return model


def replica(options,state,image,stage,do_adam):
    model=make_model(options,state,stage,True)
    optimizer=model.optimizer()
    c=Census(); handles=install_hooks(model,c)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    with c:
        out=model(image)
        c.phase='backward'; c.component='native_loss_and_model_backward'
        torch.autograd.backward(out['native_loss'],torch.tensor([.5]))
        if do_adam:
            c.phase='adam'; c.component='root_optimizer'
            optimizer.step()
    if do_adam:
        adam_sha=adam_host(c,optimizer)
    else:
        adam_sha=None
    c.phase='forward';c.component='adapter_channel_loss';c.host('noise_sigma',2,1)
    c.host('complex_uses_metadata',0,0,metadata_basic=1,
        rule_text='native adapter code[0].numel()/2:one Python true division; metadata_basic disclosed and excluded from signal logical FLOPs')
    for handle in handles:handle.remove()
    result=c.report()
    result.update(stage=stage,input_shape=list(image.shape),synthetic_input=True,
        parameter_scalars_with_grad=sum(p.numel() for p in model.parameters() if p.grad is not None),
        parameter_tensors_with_grad=sum(p.grad is not None for p in model.parameters()),
        adam_source_sha256=adam_sha,
        running_statistic_changed_keys=[k for k,v in model.state_dict().items()
            if ('running_' in k or 'num_batches_tracked' in k) and not torch.equal(before[k],v)],
        backward_seed=.5,local_extra_mean=False,
        optimizer_scope='Fresh native Adam state, one actual step. State allocation/copies cost zero in this logical convention. This audits arithmetic, not resumed optimization or training quality.')
    return result


def training_case(options,state,height,width,seed):
    torch.manual_seed(seed)
    images=torch.rand(2,3,height,width)
    if state is None:
        # DataParallel replicas start with identical state, even in a fresh
        # synthetic smoke run. Each stage/replica loads copies of this state.
        reference=make_model(options,None,'jscc',True)
        state={key:value.detach().clone() for key,value in reference.state_dict().items()}
    results={}
    for stage in ('jscc','denoiser'):
        replicas=[replica(options,state,images[i:i+1],stage,i==0) for i in range(2)]
        p=replicas[0]['parameter_scalars_with_grad']
        if replicas[1]['parameter_scalars_with_grad'] != p:
            raise RuntimeError('Replica parameter gradient coverage differs')
        phases=collections.defaultdict(lambda:{f:0 for f in FIELDS})
        for case in replicas:
            for phase,counts in case['summary'].items():
                for f in FIELDS:phases[phase][f]+=counts[f]
        # Root mean forward is sum of2 then divide by2:2 operations. Its backward
        # multiplies two incoming scalar sensitivities:2 operations. These are
        # separate from applying .5 to the actual native per-image-loss backward.
        root=dict(mean_forward_basic=2,mean_backward_basic=2,gradient_reduction_additions=p,
            scope='two actual replica batch1 forward/backward graphs; replica0 Adam once; one gradient add per active scalar; rootglobal2 mean4 ops; .5 seed supplies each replica exact chain coefficient')
        total={f:sum(counts[f] for counts in phases.values()) for f in FIELDS}
        total['metadata_basic']=sum(case['total']['metadata_basic'] for case in replicas)
        total['basic']+=4+p;total['logical_flops']=total['basic']+total['special']
        total['logical_flops_per_image']=total['logical_flops']/2
        persistence=dict(scope='replica0 native running statistics persist; replica1 updates counted as executed but discarded; storage-preservation hooks copy only, zero floating arithmetic',
            replica0_changed_keys=replicas[0]['running_statistic_changed_keys'],replica1_changed_keys=replicas[1]['running_statistic_changed_keys'])
        results[stage]=dict(replicas=replicas,root=root,total=total,running_statistics=persistence,
                            logical_scope_complete=True)
    return results


def inference(options,state,image,seed):
    torch.manual_seed(seed)
    model=make_model(options,state,'denoiser',False)
    c=Census(); handles=install_hooks(model,c)
    actual_grad=torch.autograd.grad
    def counted_grad(*args,**kwargs):
        previous=c.phase
        c.phase='refinement_backward';c.component='latent_input_gradient'
        try:return actual_grad(*args,**kwargs)
        finally:
            c.phase=previous;c.component='adapter_channel_loss'
    with patch('torch.autograd.grad',counted_grad):
        with c,torch.no_grad():
            model(image)
    for handle in handles:handle.remove()
    c.phase='forward';c.component='adapter_channel_loss';c.host('noise_sigma',2,1)
    c.host('complex_uses_metadata',0,0,metadata_basic=1,
        rule_text='native adapter code[0].numel()/2:one Python true division; metadata_basic disclosed and excluded from signal logical FLOPs')
    iterations=int(options['iterations'])
    train_snr=float(options.get('training_snr_db',options.get('snr_db',10)))
    test_snr=float(options.get('test_snr_db',options.get('snr_db',10)))
    sigma=10**(-test_snr/20)
    base_var=10**(-train_snr/10)
    ratio=sigma**2/base_var
    effective_delta=float(options.get('delta',1)) if ratio>1 else 1.
    power_basic=int(effective_delta==2)
    power_special=int(effective_delta not in (0,1,2))
    # _refine is entered even with zero iterations and computes these constants.
    c.host('refinement_constants',9+power_basic,2+power_special,3,
        'sigma2basic+pow; basevar2basic+pow; ratio square/div2; prior square/mul2; stepdiv1+ratio**effective_delta(identity/zero power0,square1basic,otherwise1special); ratio comparison and two max comparisons')
    c.host('nll_denominator',2*iterations,0,
        rule_text='each real executed iteration computes2*sigma**2:one square+one multiply; tensor division dispatched separately')
    result=c.report()
    result.update(input_shape=list(image.shape),training=False,iterations=iterations,
        seed=seed,parameter_gradients_accumulated=any(p.grad is not None for p in model.parameters()),
        warning='Exact executed representative graph; UnitNorm branch and thus arithmetic depend on checkpoint,input,noise seed and iteration. No iteration extrapolation.',
        unitnorm_interpretation='Every encoder invocation computes first norm on full batch then second norm on selected projected samples; consecutive pairs show actual selected sizes.')
    return result


def hand_checks():
    x=torch.empty(1,2,2,2); g=torch.empty_like(x); w=torch.empty(2)
    y=torch.empty_like(x); stat=torch.empty(2)
    bn=(x,w,w,stat,stat,True,.1,1e-5)
    assert rule('aten.native_batch_norm.default',bn,{},(y,stat,stat))[:2]==(84,2)
    bn_eval=(*bn[:5],False,.1,1e-5)
    assert rule('aten.native_batch_norm.default',bn_eval,{},(y,stat,stat))[:2]==(36,2)
    back=(g,x,w,stat,stat,stat,stat,True,1e-5,[True,True,True])
    assert rule('aten.native_batch_norm_backward.default',back,{},(x,w,w))[:2]==(73,0)
    back_eval=(*back[:7],False,1e-5,[True,False,False])
    assert rule('aten.native_batch_norm_backward.default',back_eval,{},(x,None,None))[:2]==(14,2)
    assert rule('aten.var.correction',(x,[0,2,3]),{'correction':0},stat)[0]==32
    assert rule('aten.linalg_vector_norm.default',(torch.empty(2,3),2,[1]),{},torch.empty(2))[:2]==(10,2)
    up=torch.empty(1,1,4,4)
    assert rule('aten.upsample_bilinear2d.default',(torch.empty(1,1,2,2),[4,4],False),{},up)[0]==186
    assert rule('aten.upsample_bilinear2d_backward.default',(up,[4,4],[1,1,2,2],False),{},torch.empty(1,1,2,2))[0]==234
    try:rule('aten.unreviewed.default',(),{},x)
    except ValueError:pass
    else:raise AssertionError('unknown operators must fail closed')
    return {'passed':True,'checks':['BNtrain84basic+2sqrt including2host','BNeval36basic+2sqrt',
        'BNtrainback73basic including1host','BNevaldx14basic+2sqrt','variance32','norm10basic+2sqrt','bilinear186forward234backward','unknown_rejection']}


def verify_image_manifest(image_path,manifest_path):
    from PIL import Image, ImageOps
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    canonical=json.dumps(manifest['images'],sort_keys=True,separators=(',',':'))
    if hashlib.sha256(canonical.encode()).hexdigest()!=manifest['identity']:
        raise ValueError('Frozen dataset manifest identity mismatch')
    raw=digest(image_path)
    rows=[row for row in manifest['images'] if row['sha256']==raw
          and row['source']=='kodak_test' and row['split']=='test']
    if len(rows)!=1:
        raise ValueError('Image must match exactly one frozen Kodak test row')
    row=rows[0]
    with Image.open(image_path) as handle:
        rgb=ImageOps.exif_transpose(handle).convert('RGB')
        width,height=rgb.size
        rgb_sha=hashlib.sha256(f'{width}x{height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
    if rgb_sha!=row['rgb_sha256'] or (width,height)!=(row['width'],row['height']):
        raise ValueError('Image decoded pixels/dimensions differ from frozen row')
    return dict(manifest_path=str(Path(manifest_path).resolve()),manifest_sha256=digest(manifest_path),
        manifest_identity=manifest['identity'],matched_row=row,raw_sha256_match=True,
        canonical_rgb_sha256_match=True,dimensions_match=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--dataset-manifest',type=Path,help='verify actual image raw/RGB hashes and test split against frozen common_dataset.json')
    parser.add_argument('--output',type=Path,default=ROOT/'Results/person1_flops/isec_logical_arithmetic.json')
    parser.add_argument('--height',type=int,default=128,help='synthetic training/synthetic inference height; actual --image is never resized')
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=22)
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--skip-inference',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    checks=hand_checks()
    state=None; checkpoint_sha=None; progress=None
    options=dict(latent_channels=6,snr_db=10.,training_snr_db=10.,test_snr_db=10.,iterations=args.iterations)
    if args.checkpoint:
        checkpoint_sha=digest(args.checkpoint)
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        state=saved['state_dict']
        progress={key:value for key,value in saved.get('progress',{}).items()
                  if key in {'contract','stage_index','epoch','offset','step','images_seen',
                             'completed_epochs','best','elapsed_seconds'}}
        options.update(saved.get('options',{}));options['iterations']=args.iterations
    if args.image:
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(args.image) as im:
            image=torch.from_numpy(np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()).permute(2,0,1).unsqueeze(0).float()/255
        image_info=dict(path=str(args.image.resolve()),sha256=digest(args.image),synthetic=False,resize=False)
        if args.dataset_manifest:
            image_info['frozen_test_verification']=verify_image_manifest(args.image,args.dataset_manifest)
    else:
        if args.dataset_manifest:
            raise ValueError('--dataset-manifest requires --image')
        image=torch.rand(1,3,args.height,args.width)
        image_info=dict(synthetic=True)
    result=dict(model='isec',torch_version=torch.__version__,device='CPU',precision='float32',seed=args.seed,
        convention=CONVENTION,formula_checks=checks,checkpoint_loaded=bool(args.checkpoint),checkpoint_written=False,
        checkpoint_path=str(args.checkpoint.resolve()) if args.checkpoint else None,checkpoint_sha256=checkpoint_sha,
        checkpoint_progress=progress,source_sha256={p:digest(ROOT/p) for p in ['src/person1/isec.py','.upstream/isec/models/autoencoders.py','.upstream/isec/models/bfcnn.py','scripts/flops_deepjscc_audit.py']},
        tool_sha256=digest(__file__),options=options,image=image_info,hardware_scope_complete=False)
    if not args.skip_training:
        result['two_replica_global2_training']=training_case(options,state,args.height,args.width,args.seed)
    if not args.skip_inference:
        result['inference']=inference(options,state,image,args.seed)
    result['logical_scope_complete']=True
    if args.checkpoint and digest(args.checkpoint)!=checkpoint_sha:
        raise RuntimeError('Checkpoint changed during audit; refuse output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'logical_scope_complete':True,
        'inference_total':result.get('inference',{}).get('total'),
        'training_totals':{k:v['total'] for k,v in result.get('two_replica_global2_training',{}).items()}},indent=2))


if __name__=='__main__':main()
