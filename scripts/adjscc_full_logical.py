"""Source-based CPU-reference FFT/GDN supplements; fail closed on unknown branches."""
from pathlib import Path
import json,math,base64,struct,collections,os
from adjscc_reachable_graph import live_function_nodes
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'Results/person1_flops/adjscc';DATA=Path(os.environ.get('PERSON1_AD_GRAPH_ROOT',str(BASE/'outputs/adjscc-flops')))
def butterfly(n,reverse=False):
    if n==1:return 0
    if n==2:return 4
    if n==4:return 22
    if n==8:return 78
    return 2*butterfly(n//2,reverse)+20+8*n+int(reverse)
def fft_axis(size,n,reverse):
    lines=size//n
    if n&(n-1):
        m=1<<(2*n-2).bit_length()
        transform=14*n+8*m+2*butterfly(m)+butterfly(m,True)
        setup=5*(n+1) # two special functions and three basic operations/chirp.
    else:transform=butterfly(n,reverse);setup=0
    return setup+lines*(transform+(6*n+1 if reverse else 0))
def fft_cost(node):
    if node['type']=='IRFFT2D':
        shape=node['outputs'][0]['shape'];batch=math.prod(shape[:-2]);h,w=shape[-2:];half=w//2+1
        # TensorFlow fft_ops.cc: inverse outer axes on positive spectrum,
        # Hermitian reconstruction, then inverse inner axis.
        return fft_axis(batch*h*half,h,True)+batch*h*(w-half)+fft_axis(batch*h*w,w,True)
    shape=node['inputs'][0]['shape'];size=math.prod(shape);h,w=shape[-2:]
    return fft_axis(size,h,False)+fft_axis(size,w,False)
def count_shape(n):
    a=n.get('attr',{});shapes=a.get('_output_shapes',{}).get('list',{}).get('shape',[])
    if not shapes:raise ValueError('Arithmetic output shape is missing')
    dims=[int(d.get('size',0)) for d in shapes[0].get('dim',[])]
    if any(d<0 for d in dims):raise ValueError('Dynamic branch output shape')
    return math.prod(dims)
def constant(n):
    t=n['attr']['value']['tensor']
    for key in ('boolVal','floatVal','doubleVal','intVal','int64Val'):
        if t.get(key):return t[key][0]
    if t.get('tensorContent'):
        fmt={'DT_FLOAT':'f','DT_DOUBLE':'d','DT_BOOL':'?','DT_INT32':'i','DT_INT64':'q'}[t['dtype']]
        raw=base64.b64decode(t['tensorContent']);return struct.unpack('<'+fmt,raw[:struct.calcsize(fmt)])[0]
    return None
class Branches:
    def __init__(self,graph):
        self.nodes={n['name']:n for n in graph['node']};self.functions={f['signature']['name']:f for f in graph['library']['function']};self.cache={}
    def value(self,ref,nodes=None,args=None):
        nodes=self.nodes if nodes is None else nodes;args={} if args is None else args
        key=ref.lstrip('^').split(':')[0]
        port=int(ref.split(':')[-1]) if ':' in ref and ref.split(':')[-1].isdigit() else 0
        if key in args:return args[key]
        n=nodes.get(key)
        if n is None:return None
        op=n['op'];inputs=n.get('input',[])
        if op=='Const':return constant(n)
        if op=='OptionalNone':return ('optional',None)
        if op=='OptionalFromValue':return ('optional',[self.value(i,nodes,args) for i in inputs])
        if op in ('OptionalGetValue','OptionalHasValue'):
            value=self.value(inputs[0],nodes,args)
            if not isinstance(value,tuple) or value[0]!='optional':return None
            if op=='OptionalHasValue':return value[1] is not None
            return value[1][port] if value[1] is not None and port<len(value[1]) else None
        if op=='IdentityN':return self.value(inputs[port],nodes,args)
        if op in ('Identity','Cast'):return self.value(inputs[0],nodes,args)
        if op in ('Equal','NotEqual','Greater','Less','LogicalAnd','LogicalOr'):
            a,b=[self.value(i,nodes,args) for i in inputs[:2]]
            if a is None or b is None:return None
            return {'Equal':lambda:a==b,'NotEqual':lambda:a!=b,'Greater':lambda:a>b,'Less':lambda:a<b,'LogicalAnd':lambda:a and b,'LogicalOr':lambda:a or b}[op]()
        if op=='StatelessIf':
            selected=self.value(inputs[0],nodes,args)
            if selected is None:return None
            f=self.functions[n['attr']['then_branch' if selected else 'else_branch']['func']['name']]
            passed={a['name']:self.value(i,nodes,args) for a,i in zip(f['signature'].get('inputArg',[]),inputs[1:])}
            output=f['signature']['outputArg'][port]['name']
            return self.value(f['ret'][output],{x['name']:x for x in f['nodeDef']},passed)
        return None
    def branch(self,node,nodes=None,args=None):
        nodes=self.nodes if nodes is None else nodes;args={} if args is None else args
        inputs=node['input'];selected=self.value(inputs[0],nodes,args)
        if selected is None:raise ValueError('Unknown branch selector '+node['name']+' input='+inputs[0]+' args='+repr({k:v for k,v in args.items() if v is not None}))
        f=self.functions[node['attr']['then_branch' if selected else 'else_branch']['func']['name']]
        passed={a['name']:self.value(i,nodes,args) for a,i in zip(f['signature'].get('inputArg',[]),inputs[1:])}
        scope={n['name']:n for n in f['nodeDef']};total=0
        for n in live_function_nodes(f):
            op=n['op'];e=count_shape(n) if op in {'Square','Abs','Neg','Mul','AddV2','Sub','RealDiv','Pow','Sqrt','Exp','Rsqrt','SqrtGrad','SigmoidGrad'} else 0
            dtype=n.get('attr',{}).get('T',{}).get('type')
            if op=='StatelessIf':total+=self.branch(n,scope,passed)
            elif op in {'Const','Identity','IdentityN','Cast','Shape','Reshape','ZerosLike','Fill','OptionalNone','OptionalFromValue','OptionalGetValue','OptionalHasValue','Sign','Equal','LogicalAnd','LogicalOr','Greater','Less','SelectV2'}:pass
            elif op in {'Square','Abs','Neg','Mul','AddV2','Sub','RealDiv','Pow','Sqrt','Exp'}:total+=e
            elif op in {'Rsqrt','SqrtGrad'}:total+=2*e
            elif op=='SigmoidGrad':total+=3*e
            elif dtype in ('DT_BOOL','DT_INT32','DT_INT64'):pass
            else:raise ValueError('Unknown function arithmetic '+op+' '+n['name'])
        return total
expanded=json.loads((BASE/'expanded_coverage.json').read_text());reports=[]
for r in expanded:
    label=r['case']+'-'+r['scope'];raw=json.loads((DATA/(label+'.json')).read_text());graph=json.loads((DATA/(label+'-graph.json')).read_text());branches=Branches(graph);top={n['name']:n for n in graph['node']};nodes={n['name']:n for n in raw['nodes']};extra=0;unresolved=[];supplements=[]
    for pending in r['unresolved_nodes']:
        name=pending['name'];n=nodes[name]
        try:
            cost=fft_cost(n) if n['type'] in ('IRFFT2D','RFFT2D') else branches.branch(top[name]);extra+=cost;supplements.append(dict(name=name,type=n['type'],logical_ops=cost))
        except Exception as error:unresolved.append(dict(name=name,error=str(error)))
    reports.append(dict(case=r['case'],scope=r['scope'],batch=r['batch'],accounted_logical_ops=r['accounted_logical_arithmetic']+extra,supplements=supplements,unresolved=unresolved,logical_scope_complete=not unresolved,convention='CPU Eigen TensorFFT source-reference arithmetic, generic complex multiply=6,add=2; source-pinned Bluestein/radix2; backend compiler simplifications excluded; not measured GPU cuFFT instructions'))
(BASE/'source_based_logical.json').write_text(json.dumps(reports,indent=2))
print(json.dumps([dict(case=r['case'],scope=r['scope'],unresolved=len(r['unresolved']),complete=r['logical_scope_complete']) for r in reports]))
