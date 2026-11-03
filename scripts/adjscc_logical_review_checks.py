"""Independent arithmetic examples/fault checks; never rewrites audit reports."""
import ast
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'scripts/adjscc_full_logical.py'
tree=ast.parse(source.read_text())
declarations=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
namespace={'__file__':str(source),'__name__':'independent_logical_review'}
exec(compile(ast.Module(body=declarations,type_ignores=[]),str(source),'exec'),namespace)
butterfly=namespace['butterfly'];fft_axis=namespace['fft_axis'];fft_cost=namespace['fft_cost'];Branches=namespace['Branches']


def shaped(name,kind,dimensions,inputs=None,**attrs):
    return dict(name=name,op=kind,input=inputs or [],attr=dict(
        _output_shapes={'list':{'shape':[{'dim':[{'size':str(x)} for x in dimensions]}]}},**attrs))


def function(name,nodes,output):
    return dict(signature={'name':name,'inputArg':[],'outputArg':[{'name':'out'}]},nodeDef=nodes,ret={'out':output})


def expect_raises(call,message):
    try:call()
    except ValueError:return
    raise AssertionError(message)


def main():
    cases=[]
    assert [butterfly(n) for n in (1,2,4,8,16,32)]==[0,4,22,78,304,884]
    assert butterfly(16,True)==305 and butterfly(32,True)==887
    cases.append('pinned-source radix2 base and recursive examples')
    # n5 pads to16: transform=14*5+8*16+2*304+305=1111,
    # chirps=5*(5+1)=30; inverse outer scaling adds6*5+1/line.
    assert fft_axis(10,5,False)==2252
    assert fft_axis(10,5,True)==2314
    assert fft_axis(16,4,True)==188
    cases.append('Bluestein forward/reverse and normalization examples')
    assert fft_cost({'type':'IRFFT2D','outputs':[{'shape':[2,3,4]}]})==2462
    assert fft_cost({'type':'RFFT2D','inputs':[{'shape':[2,3,4]}]})==2872
    cases.append('two-dimensional half-spectrum and Hermitian reconstruction')
    condition=shaped('condition','Const',[],value={'tensor':{'dtype':'DT_BOOL','boolVal':[True]}})
    branch=shaped('branch','StatelessIf',[2,3],inputs=['condition'],
                  then_branch={'func':{'name':'yes'}},else_branch={'func':{'name':'no'}})
    functions=[function('yes',[shaped('multiply','Mul',[2,3])],'multiply'),
               function('no',[shaped('sigmoid_grad','SigmoidGrad',[2,3])],'sigmoid_grad')]
    graph={'node':[condition,branch],'library':{'function':functions}}
    parser=Branches(graph)
    assert parser.branch(branch)==6
    condition['attr']['value']['tensor']['boolVal']=[False]
    assert parser.branch(branch)==18
    cases.append('only selected branch contributes arithmetic')
    condition['attr']['value']['tensor']['boolVal']=[True]
    functions[0]['nodeDef']=[shaped('unknown','UnreviewedFloatingArithmetic',[2,3])]
    expect_raises(lambda:Branches(graph).branch(branch),'Unknown arithmetic silently accepted')
    functions[0]['nodeDef']=[shaped('dynamic','Mul',[-1,3])]
    expect_raises(lambda:Branches(graph).branch(branch),'Dynamic arithmetic shape silently accepted')
    condition['op']='Placeholder'
    expect_raises(lambda:Branches(graph).branch(branch),'Unknown selector silently accepted')
    cases.append('unknown op, unknown selector and dynamic shape fail closed')
    optional_graph={'node':[shaped('none','OptionalNone',[]),shaped('has','OptionalHasValue',[],['none'])],
                    'library':{'function':[]}}
    assert Branches(optional_graph).value('has') is False
    cases.append('empty optional selector resolves false')
    port_graph={'node':[shaped('yes','Const',[],value={'tensor':{'dtype':'DT_BOOL','boolVal':[True]}}),
                        shaped('no','Const',[],value={'tensor':{'dtype':'DT_BOOL','boolVal':[False]}}),
                        shaped('ports','IdentityN',[],['yes','no'])],
                'library':{'function':[]}}
    assert Branches(port_graph).value('ports:0') is True
    assert Branches(port_graph).value('ports:1') is False
    expect_raises(lambda:namespace['count_shape']({'op':'Mul','attr':{}}),'Missing shape must not default to scalar')
    cases.append('IdentityN output ports and missing-shape rejection')
    # Current exports must supply shapes for every arithmetic node in functions.
    arithmetic={'Square','Abs','Neg','Mul','AddV2','Sub','RealDiv','Pow','Sqrt','Exp','Rsqrt','SqrtGrad','SigmoidGrad'}
    graph_root=Path(os.environ.get('PERSON1_AD_GRAPH_ROOT',str(ROOT/'Results/person1_flops/adjscc/pruned-v4')))
    files=list(graph_root.glob('*-graph.json'))
    if len(files)!=5:raise ValueError('Expected all five graph cases for review')
    checked=0
    for path in files:
        g=json.loads(path.read_text())
        for f in g.get('library',{}).get('function',[]):
            for n in f.get('nodeDef',[]):
                if n['op'] in arithmetic:
                    assert n.get('attr',{}).get('_output_shapes',{}).get('list',{}).get('shape'),(path,n['name'])
                    checked+=1
    result=dict(passed=True,cases=cases,arithmetic_function_nodes_shape_checked=checked,
                graphs_checked=len(files),graph_root=str(graph_root),scope='independent source/formula and fail-closed checks; no TensorFlow execution or artifact changes')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
