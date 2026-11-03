"""Fail-closed logical arithmetic census of the current DeepJSCC implementation.

Uses real CPU execution of a fresh model, never reads or writes a checkpoint.
Dense convolutions use the declared 2 operations/MAC convention, not CPU/GPU
instruction counts. Every dispatched operator gets a coverage-ledger entry.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from src.person1.paper_deepjscc import DeepJSCC


CONVENTION = {
    'scope': 'logical arithmetic of observed ATen graph plus explicitly enumerated Python scalar formulas',
    'basic': 'add/subtract/multiply/divide/negate/reciprocal each 1; dense convolution MAC=2',
    'special': 'sqrt, exp, general power each 1 logical special-function operation; separately reported',
    'convolution': 'dense direct full-kernel MAC convention, including zero padding; bias add counted separately',
    'backward': 'actual requested input/weight/bias gradient masks; no forward multiplier shortcut',
    'prelu': 'dense piecewise candidate arithmetic: 1 multiply/element forward; 2 backward; comparisons separate',
    'sigmoid': '1 negate + 1 exp + 1 add + 1 reciprocal per element; backward 3 basic operations/element',
    'mse': 'mean squared error 3N basic operations (subtract, square, N-1 adds, divide); backward 3N plus scalar 2/N',
    'reductions': 'sum of R elements into O outputs counts R-O additions',
    'random': 'normal draws counted as random primitives, RNG implementation internal arithmetic excluded',
    'zero_cost': 'allocation, copying, views, indexing/padding, scalar extraction and profiler/control metadata excluded from floating arithmetic',
    'hardware_scope_complete': False,
    'not_counted': ['backend instruction algorithms, vectorization and FMA fusion', 'RNG internal implementation',
                    'Python indexing/shape/integer/control-loop arithmetic', 'memory traffic', 'GPU replication/communication'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in tensors(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in tensors(item)]
    return []


def description(value):
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [description(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


ZERO = {
    'aten.zeros_like.default', 'aten.ones_like.default', 'aten.new_full.default',
    'aten.detach.default', 'aten.lift_fresh.default', 'aten._local_scalar_dense.default',
    'aten.view.default', 'aten.reshape.default', 'aten.slice.Tensor', 'aten.slice_backward.default',
    'aten.constant_pad_nd.default', 'aten.expand.default', 'aten.scalar_tensor.default',
    'aten.empty_like.default', 'aten.copy_.default', 'aten.clone.default',
    'profiler._record_function_enter_new.default', 'profiler._record_function_exit._RecordFunction',
}


def convolution_macs(image, weight, result, transposed, groups):
    if transposed:
        return image.numel() * weight.shape[1] * math.prod(weight.shape[2:])
    return result.numel() * weight.shape[1] * math.prod(weight.shape[2:])


def arithmetic_rule(name, args, kwargs, output):
    """Return basic, special, comparisons, random draws, description; reject unknowns."""
    ts = tensors(output)
    n = ts[0].numel() if ts else 0
    if name in ZERO:
        return 0, 0, 0, 0, 'memory/shape/control only'
    if name == 'aten.convolution.default':
        x, w, bias, stride, padding, dilation, transpose, output_padding, groups = args
        mac = convolution_macs(x, w, output, transpose, groups)
        return 2*mac + (n if bias is not None else 0), 0, 0, 0, f'2*{mac} MAC + bias output elements'
    if name == 'aten.convolution_backward.default':
        grad, x, w, bias_size, stride, padding, dilation, transpose, output_padding, groups, mask = args
        mac = convolution_macs(x, w, grad, transpose, groups)
        count = 2*mac*(int(mask[0])+int(mask[1]))
        bias_add = grad.numel()-math.prod(bias_size) if mask[2] else 0
        return count+bias_add, 0, 0, 0, f'2*{mac} MAC for each requested dInput/dWeight; mask={mask}; dBias adds={bias_add}'
    if name == 'aten._prelu_kernel.default':
        return n, 0, n, 0, 'dense conditional multiply candidate + sign comparison'
    if name == 'aten._prelu_kernel_backward.default':
        return 2*n, 0, n, 0, 'dInput and un-reduced dWeight multiply candidates; sign comparison'
    if name == 'aten.sigmoid.default':
        return 3*n, n, 0, 0, 'negate, exp, add, reciprocal'
    if name == 'aten.sigmoid_backward.default':
        return 3*n, 0, 0, 0, 'grad*y*(1-y)'
    if name == 'aten.mse_loss.default':
        if len(args)>2 and args[2] != 1:
            raise ValueError('Only mean MSE is specified')
        return 3*args[0].numel(), 0, 0, 0, '2N pointwise + N-1 reduction + 1 normalization'
    if name == 'aten.mse_loss_backward.default':
        if args[3] != 1:
            raise ValueError('Only mean MSE backward is specified')
        return 3*n+1, 0, 0, 0, 'difference, normalization multiply, upstream-gradient multiply + scalar 2/N'
    if name in {'aten.sum.dim_IntList', 'aten.sum.default'}:
        return args[0].numel()-n, 0, 0, 0, 'input elements minus output elements reduction additions'
    if name == 'aten.pow.Tensor_Scalar':
        exponent = args[1]
        if exponent == 2:
            return n, 0, 0, 0, 'square: one multiplication'
        if exponent in (0, 1):
            return 0, 0, 0, 0, 'constant/identity power'
        return 0, n, 0, 0, 'general power special primitive'
    if name in {'aten.sqrt.default', 'aten.exp.default'}:
        return 0, n, 0, 0, 'one special primitive per element'
    if name in {'aten.gt.Scalar', 'aten.ge.Scalar', 'aten.clamp_min.default'}:
        return 0, 0, n, 0, 'one comparison per output element'
    if name == 'aten.where.self':
        return 0, 0, 0, 0, 'selection only; comparison dispatched separately'
    if name == 'aten.randn_like.default':
        return 0, 0, 0, n, 'one normal random draw primitive per element'
    if name == 'aten.lerp_.Scalar':
        return 3*n, 0, 0, 0, 'start + weight*(end-start)'
    if name in {'aten.addcmul_.default', 'aten.addcdiv_.default'}:
        return 3*n, 0, 0, 0, 'multiply/divide + scalar multiply + addition'
    if name in {'aten.mul.Tensor', 'aten.mul.Scalar', 'aten.mul_.Tensor', 'aten.div.Tensor',
                'aten.div.Scalar', 'aten.add.Tensor', 'aten.add_.Tensor', 'aten.sub.Tensor',
                'aten.neg.default', 'aten.reciprocal.default'}:
        count=n
        if name in {'aten.add.Tensor','aten.add_.Tensor','aten.sub.Tensor'} and kwargs.get('alpha',1)!=1:
            count+=n
        return count, 0, 0, 0, 'one pointwise basic operation per output (+nonunit alpha if present)'
    raise ValueError('Unknown operator arithmetic: '+name)


class Census(TorchDispatchMode):
    def __init__(self):
        self.phase='unassigned'
        self.component='channel'
        self.rows=[]
        self.unknown=[]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        out=func(*args, **kwargs)
        name=str(func)
        if self.phase=='backward' and name=='aten.sigmoid_backward.default':
            self.component='model_backward'
        try:
            basic,special,comparison,random,rule=arithmetic_rule(name,args,kwargs,out)
        except ValueError as error:
            self.unknown.append({'phase':self.phase,'operator':name,'error':str(error)})
            basic=special=comparison=random=0
            rule='UNKNOWN: NOT COUNTED'
        self.rows.append(dict(phase=self.phase,component=self.component if self.phase in {'forward','backward'} else None,
                              operator=name,inputs=description(args),outputs=description(out),
                              basic=basic,special=special,comparisons=comparison,random_draws=random,rule=rule))
        return out

    def host(self, phase, name, basic, special, rule):
        self.rows.append(dict(phase=phase,component='channel' if phase=='forward' else None,
                              operator='python.'+name,inputs=[],outputs=[],
                              basic=basic,special=special,comparisons=0,random_draws=0,rule=rule))

    def summary(self, batch):
        result={}
        for phase in sorted({r['phase'] for r in self.rows}):
            group=[r for r in self.rows if r['phase']==phase]
            total={key:sum(r[key] for r in group) for key in ('basic','special','comparisons','random_draws')}
            total['logical_flops']=total['basic']+total['special']
            total['per_image']={k:v/batch for k,v in total.items() if isinstance(v,(int,float))}
            result[phase]=total
        return result


def host_forward(census):
    census.host('forward','channel_noise_scale',3,1,'-snr/20; 10**exponent; divide by constant sqrt(2); constant folding excluded')


def formula_checks():
    """Small hand-enumerated cases plus a mandatory unknown-op rejection."""
    x=torch.empty(1,2,3,3);w=torch.empty(3,2,1,1);y=torch.empty(1,3,3,3)
    args=(x,w,torch.empty(3),[1,1],[0,0],[1,1],False,[0,0],1)
    assert arithmetic_rule('aten.convolution.default',args,{},y)[0]==135
    back=(y,x,w,[3],[1,1],[0,0],[1,1],False,[0,0],1,[False,True,True])
    assert arithmetic_rule('aten.convolution_backward.default',back,{},(None,w,torch.empty(3)))[0]==132
    tw=torch.empty(2,3,2,2);ty=torch.empty(1,3,4,4)
    targs=(x,tw,torch.empty(3),[1,1],[0,0],[1,1],True,[0,0],1)
    assert arithmetic_rule('aten.convolution.default',targs,{},ty)[0]==480
    assert arithmetic_rule('aten.sum.dim_IntList',(torch.empty(2,3),[1],False),{},torch.empty(2))[0]==4
    assert arithmetic_rule('aten.sigmoid.default',(torch.empty(2,3),),{},torch.empty(2,3))[:2]==(18,6)
    for operator in ('aten.addcmul_.default','aten.addcdiv_.default','aten.lerp_.Scalar'):
        assert arithmetic_rule(operator,(),{},torch.empty(2,3))[0]==18
    try:
        arithmetic_rule('aten.unreviewed_arithmetic.default',(),{},torch.empty(1))
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown arithmetic must be rejected')
    return {'passed':True,'cases':['conv+bias135','backward-mask132','transpose+bias480',
                                  'reduction4','sigmoid18basic+6special','optimizer-fused-primitives',
                                  'unknown-operator-fails-closed']}


def audit(batch, height, width, training):
    torch.manual_seed(22)
    model=DeepJSCC({'latent_channels':16,'snr_db':10.,'learning_rate':1e-4})
    optimizer=model.optimizer() if training else None
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    image=torch.rand(batch,3,height,width)
    census=Census()
    def before_encoder(module,args):census.component='encoder'
    def before_decoder(module,args):census.component='decoder'
    def after_component(module,args,out):census.component='channel'
    handles=[model.encoder.register_forward_pre_hook(before_encoder),
             model.encoder.register_forward_hook(after_component),
             model.decoder.register_forward_pre_hook(before_decoder),
             model.decoder.register_forward_hook(after_component)]
    with census:
        census.phase='forward'
        if training:
            model.train();out=model(image)
        else:
            model.eval()
            with torch.no_grad():out=model(image)
        if training:
            census.phase='loss';loss=model.loss(out,{'image':image})
            census.phase='backward';census.component='loss_backward';loss.backward()
            census.phase='adam';optimizer.step()
    for handle in handles:handle.remove()
    host_forward(census)
    if training:
        active=sum(p.grad is not None for p in model.parameters())
        census.host('adam','single_tensor_bias_corrections',6*active,3*active,
                    'per active parameter tensor: 1-beta1,1-beta2, two bias-correction subtracts, lr/bc1, negate step_size; two beta**step + sqrt(bc2)')
    changed=any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    backward_components={component:sum(r['basic']+r['special'] for r in census.rows
                                      if r['phase']=='backward' and r.get('component')==component)
                         for component in ('loss_backward','model_backward')}
    breakdown={}
    for component in ('encoder','channel','decoder'):
        group=[r for r in census.rows if r.get('component')==component and r['phase']=='forward']
        total={key:sum(r[key] for r in group) for key in ('basic','special','comparisons','random_draws')}
        total['logical_flops']=total['basic']+total['special']
        total['logical_flops_per_image']=total['logical_flops']/batch
        breakdown[component]=total
    return dict(input_shape=list(image.shape),synthetic=True,training=training,parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                parameters_with_gradients=sum(p.numel() for p in model.parameters() if p.grad is not None),
                parameter_tensors_with_gradients=sum(p.grad is not None for p in model.parameters()),
                forward_components=breakdown,
                backward_components=backward_components,
                checkpoint_loaded=False,checkpoint_written=False,fresh_model_updated=changed,
                actual_convolution_backward_calls=sum(r['operator']=='aten.convolution_backward.default' for r in census.rows),
                logical_scope_complete=not census.unknown,hardware_scope_complete=False,
                unknown_operators=census.unknown,summary=census.summary(batch),ledger=census.rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'Results/person1_flops/deepjscc_logical_arithmetic.json')
    parser.add_argument('--training-batch',type=int,default=32)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    checks=formula_checks()
    adam_module=importlib.import_module('torch.optim.adam')
    adam_source=inspect.getsource(adam_module._single_tensor_adam)
    required=['exp_avg.lerp_(grad, 1 - device_beta1)','bias_correction1 = 1 - beta1**step',
              'bias_correction2 = 1 - beta2**step','bias_correction2_sqrt = bias_correction2**0.5',
              'param.addcdiv_(exp_avg, denom, value=-step_size)']
    if not all(text in adam_source for text in required):
        raise RuntimeError('Adam host formula source changed; re-audit scalar accounting')
    result=dict(model='deepjscc',torch_version=torch.__version__,precision='float32',device='CPU',seed=22,
                lineage='2019 Figure2 reproduction; original author source unverified; per-channel PReLU disclosed',
                convention=CONVENTION,formula_checks=checks,model_source_sha256=digest(ROOT/'src/person1/paper_deepjscc.py'),
                tool_sha256=digest(__file__),adam_source_sha256=hashlib.sha256(adam_source.encode()).hexdigest(),
                cases={'training_128':audit(args.training_batch,128,128,True),
                       'training_replica16_128':audit(16,128,128,True),
                       'inference_512x768':audit(1,512,768,False)})
    result['logical_scope_complete']=all(case['logical_scope_complete'] for case in result['cases'].values())
    for case in result['cases'].values():
        case['total_logical_flops']=sum(p['logical_flops'] for p in case['summary'].values())
        case['total_logical_flops_per_image']=case['total_logical_flops']/case['input_shape'][0]
    global_case=result['cases']['training_128']
    replica=result['cases']['training_replica16_128']
    if global_case['input_shape'][0]==32:
        assert global_case['parameters_with_gradients']==global_case['trainable_parameters']
        assert replica['parameters_with_gradients']==global_case['trainable_parameters']
        parts=replica['summary'];root=global_case['summary']
        dp=dict(replica_forward=2*parts['forward']['logical_flops'],
                replica_model_backward=2*replica['backward_components']['model_backward'],
                root_global_loss_backward=global_case['backward_components']['loss_backward'],
                root_global_loss=root['loss']['logical_flops'],
                root_adam=root['adam']['logical_flops'],
                gradient_reduction_additions=global_case['parameters_with_gradients'])
        dp['total_logical_flops']=sum(dp.values())
        dp['per_image']=dp['total_logical_flops']/32
        dp['scope']='logical two-replica composition from actual CPU microbatch16 graphs, root global-batch32 loss/Adam, one gradient addition/trainable scalar; no GPU execution or transfers counted'
        result['two_replica_global32_composition']=dp
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    coverage=args.output.with_name('deepjscc_operator_coverage.csv')
    grouped={}
    for case_name,case in result['cases'].items():
        for row in case['ledger']:
            key=(case_name,row['phase'],row['operator'])
            entry=grouped.setdefault(key,dict(case=case_name,phase=row['phase'],operator=row['operator'],calls=0,
                basic=0,special=0,comparisons=0,random_draws=0,rule=row['rule']))
            entry['calls']+=1
            for field in ('basic','special','comparisons','random_draws'):entry[field]+=row[field]
    with coverage.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(next(iter(grouped.values()))))
        writer.writeheader();writer.writerows(grouped.values())
    print(json.dumps({name:{'complete':case['logical_scope_complete'],'summary':case['summary'],
                           'unknown':case['unknown_operators']} for name,case in result['cases'].items()},indent=2))
    if not result['logical_scope_complete']:
        raise SystemExit('Incomplete logical arithmetic coverage; unknown operators in ledger')


if __name__=='__main__':
    main()
