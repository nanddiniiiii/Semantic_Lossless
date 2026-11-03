"""Expand TF graph coverage without disguising unresolved FFT arithmetic."""
from pathlib import Path
import json,math,csv,collections,os
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Results/person1_flops/adjscc'
ZERO={'Const','Placeholder','Identity','IdentityN','NoOp','ReadVariableOp','Reshape','Transpose','ExpandDims','Squeeze','Pack','ConcatV2','ConcatOffset','StridedSlice','StridedSliceGrad','Slice','Tile','Fill','Shape','ShapeN','Size','BroadcastGradientArgs','InvertPermutation','Cast','Real','Imag','Complex'}
COMPARE={'Equal','NotEqual','Less','Greater','LessEqual','GreaterEqual','LogicalOr','LogicalAnd','LogicalNot','Maximum','Minimum','Relu','ReluGrad'}
def elements(t):
    s=t['shape']
    if s is None or any(x is None for x in s):raise ValueError('Unknown shape')
    return math.prod(s)
results=[];ledger=[]
for path in sorted(Path(os.environ.get('PERSON1_AD_GRAPH_ROOT',str(BASE/'outputs/adjscc-flops'))).glob('*forward*.json')):
    raw=json.loads(path.read_text())
    if not isinstance(raw,dict) or 'nodes' not in raw:continue
    totals=collections.Counter();pending=[]
    for n in raw['nodes']:
        kind=n['type'];flop=0;special=0;comparison=0;random=0;classification='';reason=None
        try:
            dtype=next((t['dtype'] for t in n['outputs'] if t['dtype'].startswith(('float','complex'))),None)
            if kind in ZERO:classification='data movement or metadata'
            elif kind in COMPARE:
                comparison=elements(n['outputs'][0]);classification='comparison or selection'
            elif kind=='RandomStandardNormal':random=elements(n['outputs'][0]);classification='normal random primitives'
            elif kind in {'RFFT2D','IRFFT2D'}:reason='FFT backend arithmetic not exposed by this graph census'
            elif kind=='StatelessIf':
                if any(t['dtype'].startswith(('float','complex')) for t in n['outputs']):reason='floating branch function requires expansion'
                else:classification='boolean control flow'
            elif kind=='Conj':flop=elements(n['outputs'][0]);classification='one imaginary negation per complex element'
            elif kind=='Sigmoid':
                e=elements(n['outputs'][0]);flop=3*e;special=e;classification='negate,exp,add,reciprocal'
            elif kind=='SigmoidGrad':flop=3*elements(n['outputs'][0]);classification='three scalar operations per element'
            elif kind=='Sqrt':special=elements(n['outputs'][0]);classification='one special function per element'
            elif kind=='SqrtGrad':flop=2*elements(n['outputs'][0]);classification='dy divided by twice output'
            elif kind in {'AssignAddVariableOp','AssignSubVariableOp'}:
                t=n['inputs'][1]
                if t['dtype'].startswith('float'):flop=elements(t)
                classification='resource update arithmetic; integer step counters separate'
            elif dtype is None:classification='integer or boolean metadata arithmetic'
            elif dtype.startswith('complex') and kind in {'Mul','AddV2','Sub','Neg','RealDiv'}:
                flop=elements(n['outputs'][0])*{'Mul':6,'AddV2':2,'Sub':2,'Neg':2,'RealDiv':11}[kind];classification='expanded real arithmetic for complex operation'
            elif dtype.startswith('complex') and kind in {'Sum','Mean','AddN'}:
                if n['registered_flops'] is None:raise ValueError('Complex reduction lacks registered shape census')
                flop=2*int(n['registered_flops']);classification='independent real and imaginary reductions'
            elif kind=='DivNoNan':
                flop=elements(n['outputs'][0]);comparison=flop;classification='one guarded scalar division; comparison separate'
            elif kind=='Pow':special=elements(n['outputs'][0]);classification='one logical special-function power per element'
            elif n['registered_flops'] is not None:flop=int(n['registered_flops']);classification='TF registered real arithmetic'
            else:reason='unclassified arithmetic'
        except (ValueError,IndexError):reason='unresolved shape or operands'
        totals.update(basic=flop,special=special,comparisons=comparison,normal_draws=random)
        row=dict(case=raw['case'],scope=raw['scope'],name=n['name'],type=kind,basic=flop,special=special,comparisons=comparison,normal_draws=random,classification=classification,unresolved=reason)
        ledger.append(row)
        if reason:pending.append(row)
    results.append(dict(case=raw['case'],scope=raw['scope'],batch=raw['batch'],accounted_logical_arithmetic=totals['basic']+totals['special'],categories=dict(totals),unresolved_nodes=pending,logical_scope_complete=not pending))
(BASE/'expanded_coverage.json').write_text(json.dumps(results,indent=2))
with (BASE/'operator_coverage.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
print(json.dumps([dict(case=r['case'],scope=r['scope'],unresolved=len(r['unresolved_nodes']),complete=r['logical_scope_complete']) for r in results]))
