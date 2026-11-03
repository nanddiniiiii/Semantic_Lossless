"""Independent tiny CPU coverage review; no checkpoint or benchmark claims.

This intentionally does not turn unsupported operators into zero-cost arithmetic.
It inventories the native adapter's actual forward/backward/Adam dispatches against
the existing DeepJSCC rules and records examples for implementing new rules.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from src.person1.isec import build
from scripts.flops_deepjscc_audit import arithmetic_rule, description


class Review(TorchDispatchMode):
    def __init__(self):
        self.phase = 'forward'
        self.rows = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        key = (self.phase, str(func))
        if key not in self.rows:
            try:
                arithmetic_rule(str(func), args, kwargs, out)
                supported = True
            except (ValueError, TypeError):
                supported = False
            self.rows[key] = dict(phase=self.phase, operator=str(func), calls=0,
                                 shared_rule_supported=supported,
                                 example_inputs=description(args),
                                 example_outputs=description(out))
        self.rows[key]['calls'] += 1
        return out


def probe(stage, training, iterations=0):
    torch.manual_seed(22)
    model = build({'latent_channels': 6, 'iterations': iterations}, 'cpu', ROOT/'.upstream/isec')
    model.stage = stage
    model.train(training)
    optimizer = model.optimizer() if training else None
    image = torch.rand(2, 3, 8, 8)
    review = Review()
    with review:
        if training:
            out = model(image)
            review.phase = 'loss'
            loss = model.loss(out, {'image': image})
            review.phase = 'backward'
            loss.backward()
            review.phase = 'adam'
            optimizer.step()
        else:
            with torch.no_grad():
                model(image)
    return dict(synthetic=True, device='CPU', shape=list(image.shape), stage=stage,
                training=training, refinement_iterations=iterations,
                parameter_scalars_with_grad=sum(p.numel() for p in model.parameters() if p.grad is not None),
                parameter_tensors_with_grad=sum(p.grad is not None for p in model.parameters()),
                logical_scope_complete=False,
                rows=list(review.rows.values()))


def main():
    torch.set_num_threads(2)
    result = dict(torch_version=torch.__version__, checkpoint_loaded=False,
                  checkpoint_written=False, quality_or_timing_benchmark=False,
                  purpose='Operator coverage review only; unsupported operators fail coverage, never treated as zero.',
                  cases={name: probe(*args) for name, args in {
                      'jscc_train': ('jscc', True, 0),
                      'denoiser_train': ('denoiser', True, 0),
                      'eval_one_refinement': ('denoiser', False, 1),
                  }.items()})
    result['reviewed_formulas'] = {
        'scope': 'Canonical explicit logical algorithms; not backend instruction counts. Multiplication/addition/division each one. sqrt separate special; reciprocal basic. Cached floating coefficients must be declared. N=input elements, C=reduced groups/channels, m=N/C. Hand counts below exclude explicitly listed host constants.',
        'batch_norm_forward_train': {
            'basic': '8*N+9*C with affine gamma/beta and both running statistics',
            'special': 'C sqrt',
            'steps': ['two-pass var_mean:4N', 'eps and reciprocal:2C', 'center, normalize, gamma, beta:4N', 'running_mean:3C', 'biased-to-unbiased variance:C', 'running_var update:3C'],
            'host': '1-momentum:1 subtraction; n/(n-1):1 subtraction and1 division when n is floating, or1 division if denominator integer arithmetic excluded',
            'restrictions': 'training m>1; remove N if gamma absent, N if beta absent; remove running updates when statistics absent',
            'example': {'N': 8, 'C': 2, 'steps_basic': [32, 4, 32, 6, 2, 6], 'basic': 82, 'sqrt': 2},
        },
        'batch_norm_forward_eval': {
            'basic': '4*N+2*C', 'special': 'C sqrt',
            'steps': ['var+eps:C', 'sqrt:C special', 'reciprocal:C', '(x-mean)*invstd*gamma+beta:4N'],
            'example': {'N': 8, 'C': 2, 'basic': 36, 'sqrt': 2},
        },
        'batch_norm_backward_train': {
            'algorithm': 'Reuse centered=x-mean; s1=sum(g); s2=sum(g*centered); gradmean=s1/m; projscale=(s2/m)*(invstd*invstd); gradscale=invstd*gamma; dx=((g-centered*projscale)-gradmean)*gradscale; dgamma=s2*invstd; dbeta=s1.',
            'full_basic': '8*N+4*C', 'dx_only_basic': '8*N+3*C',
            'dgamma_only_basic': '3*N', 'dbeta_only_basic': 'N-C',
            'dgamma_dbeta_only_basic': '4*N-C',
            'host': '1/m:1 division; computed once per operator if not cached',
            'important': 'Formula follows an explicitly reused centered temporary. Local torch decomposition recomputes (input-mean) for dx, adding N. Either declared algorithm is valid logical scope; never mix them silently. Gate every output by native output_mask; refinement normally asks dx only.',
            'example': {'N': 8, 'C': 2, 'full_steps_basic': [6, 22, 2, 6, 2, 32, 2], 'full_basic': 72, 'dx_only_basic': 70},
        },
        'batch_norm_backward_eval': {
            'algorithm': 'Recompute invstd=1/sqrt(running_var+eps); dx=g*(gamma*invstd); dgamma=invstd*sum(g*(x-mean)); dbeta=sum(g). Share invstd.',
            'dx_only_basic': 'N+3*C', 'full_basic': '5*N+2*C',
            'special': 'C sqrt when any dx or dgamma requested; dbeta alone needs no invstd',
            'example': {'N': 8, 'C': 2, 'dx_only_basic': 14, 'full_basic': 44, 'sqrt': 2},
        },
        'var_correction_forward': {
            'basic': '4*N',
            'algorithm': 'mean: N-C additions+C divides=N; deviation squares:N subtracts+N multiplies; square reduction:N-C additions; divide by m-correction:C divisions. The denominator is shape/integer control when correction integer.',
            'restrictions': 'Reject m-correction<=0; preserve actual correction; BF_CNN passes correction=0. Do not use unbiased default.',
            'example': {'N': 8, 'C': 2, 'steps_basic': [8, 8, 8, 6, 2], 'basic': 32},
        },
        'vector_norm_p2': {
            'forward_basic': '2*N-R', 'forward_special': 'R sqrt; R nonempty output groups',
            'backward': 'Actual CPU ATen probe computes x/norm (N divides), norm==0 (R comparisons), masked_fill to0 (selection), multiply by expanded upstream grad (N multiplies). Hence2N basic and R comparisons; dispatch arithmetic already accounts this and must not add a second norm-backward total.',
            'empty': 'N=R=0:zero arithmetic; an empty projected subset is valid',
            'example': {'N': 6, 'R': 2, 'forward_basic': 10, 'sqrt': 2, 'backward_basic': 12, 'backward_comparisons': 2},
        },
        'bilinear_align_corners_false': {
            'forward_value_algorithm': 'h0*(w0*x00+w1*x01)+h1*(w0*x10+w1*x11):6 multiplies+3 adds=9 per output scalar.',
            'backward_value_algorithm': 'For each output gradient and4 neighbor contributions, g*hweight*wweight then scatter-add:8 multiplies+4 adds=12 per output gradient scalar. Count even the first addition into initialized zero, under this declared scatter algorithm.',
            'coordinates': 'If computed once per output coordinate per axis: scale=I/O:one divide per axis; source=(o+0.5)*scale-0.5:3 basic per coordinate; low=int(max(source,0)) and index clamps:control/comparisons separately; w1=source-low and w0=1-w1:2 basic per coordinate. Thus5*(OH+OW)+2 basic per invocation with scales not supplied, plus comparison/index control. This is canonical precomputed-axis accounting, not a claim about kernel caching. size-one/degenerate/corner cases may skip operations in a backend; use this declared dense logical rule.',
            'warning': '7N weighted4tap sums omit4 pairwise weight products per output pixel unless already materialized and counted;7N alone is incomplete. Coordinate coefficients are floating arithmetic and cannot silently be dropped as integer shape work.',
            'example': {'input_shape': [1, 1, 2, 2], 'output_shape': [1, 1, 4, 4], 'forward_value_basic': 144, 'coordinate_basic': 42, 'forward_total_basic': 186, 'backward_value_basic': 192, 'backward_total_if_coordinates_recomputed': 234},
        },
        'other_rules': {
            'mean': 'N-R reduction adds+R divisions=N basic, with nonempty reduction groups; backward actual expand/divide dispatches counted directly',
            'tanh': 'Either declare tanh a separate special primitive, or exp canonical formula; do not include both. Backward g*(1-y*y):3N basic.',
            'relu': 'N comparisons forward/backward, selection zero basic; threshold_backward threshold comparison per element.',
            'leaky_relu': 'Not observed in this native ISEC adapter. If added:dense forward candidate multiply N basic+N comparisons; input backward N basic+N comparisons. Parameterized PReLU differs: backward also emits slope gradient contributions and slope reductions.',
            'conv_transpose': 'Not observed in this native ISEC decoder. Shared dense full-kernel rule uses2*input.numel()*weight.shape[1]*prod(kernel) basic, plus output bias adds. Backward applies same MAC total per requested dx/dw mask, dBias reduction=sum(grad)-channels; account padding under declared dense convention.',
            'index_put': 'Zero floating arithmetic only for accumulate=False; accumulate=True needs indexed additions and overlapping index semantics review.',
            'adam': 'Tiny probe had no unsupported dispatched Adam operators; host beta powers/bias corrections still require pinned source review. First and later steps should be probed; do not count running_sd parameters without gradients as Adam updates.',
        },
    }
    target = ROOT/'Results/person1_flops/isec_review.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, case in result['cases'].items():
        missing = collections.defaultdict(list)
        for row in case['rows']:
            if not row['shared_rule_supported']:
                missing[row['phase']].append(row['operator'])
        print(name, json.dumps(dict(missing), indent=2))


if __name__ == '__main__':
    main()
