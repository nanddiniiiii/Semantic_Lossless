"""Fail-closed dependency pruning of the frozen one-output AD audit functions.

Preserves immutable raw exports. Follow data AND control edges. Top roots are
verified against the audit's single forward/loss return, not arbitrary terminal
nodes. Selected FunctionDef bodies use their declared ret/control_ret roots.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def node_name(reference):
    return reference.lstrip('^').split(':')[0]


def dependency_names(nodes, roots, external=()):
    by_name={node['name']:node for node in nodes}
    if len(by_name)!=len(nodes):raise ValueError('Duplicate graph node names')
    external=set(external)
    if not roots:raise ValueError('Graph has no verified return/control roots')
    live=set();stack=list(roots)
    while stack:
        name=node_name(stack.pop())
        if name in live or name in external:continue
        if name not in by_name:raise ValueError('Unresolved dependency: '+name)
        live.add(name);stack.extend(by_name[name].get('input',[]))
    return live


def top_live_names(graph):
    nodes=graph['node'];by_name={node['name']:node for node in nodes}
    roots=[n['name'] for n in nodes if n['op']=='Identity' and '^NoOp' in n.get('input',[])]
    if len(roots)!=1:
        raise ValueError('Expected exactly one exported forward/loss return Identity with ^NoOp; got '+repr(roots))
    if by_name.get('NoOp',{}).get('op')!='NoOp':raise ValueError('Expected explicit stateful-control NoOp')
    if len([i for i in by_name[roots[0]]['input'] if not i.startswith('^')])!=1:
        raise ValueError('Expected one tensor returned by frozen audit function')
    return dependency_names(nodes,roots),roots


def live_function_nodes(function):
    roots=list(function.get('ret',{}).values())+list(function.get('controlRet',function.get('control_ret',{})).values())
    args=[a['name'] for a in function['signature'].get('inputArg',function['signature'].get('input_arg',[]))]
    nodes=function.get('nodeDef',function.get('node_def',[]))
    live=dependency_names(nodes,roots,args)
    return [n for n in nodes if n['name'] in live]


def fault_checks():
    nodes=[{'name':'x','op':'Placeholder'},
           {'name':'live_mul','op':'Mul','input':['x','x']},
           {'name':'dead_conv','op':'Conv2DBackpropInput','input':['x']},
           {'name':'update','op':'AssignAddVariableOp','input':['live_mul']},
           {'name':'NoOp','op':'NoOp','input':['^update']},
           {'name':'result','op':'Identity','input':['live_mul','^NoOp']},
           {'name':'dead_identity','op':'Identity','input':['dead_conv']}]
    graph={'node':nodes}
    live,roots=top_live_names(graph)
    assert live=={'x','live_mul','update','NoOp','result'} and roots==['result']
    f={'signature':{'inputArg':[{'name':'arg'}]},
       'nodeDef':[{'name':'live','op':'Mul','input':['arg','arg']},
                  {'name':'dead','op':'Exp','input':['arg']},
                  {'name':'side_effect','op':'AssignAddVariableOp','input':['arg']}],
       'ret':{'output':'live:z:0'},'controlRet':{'update':'side_effect'}}
    assert {n['name'] for n in live_function_nodes(f)}=={'live','side_effect'}
    for bad in ({'node':[n for n in nodes if n['name']!='result']},
                {'node':nodes+[{'name':'ambiguous','op':'Identity','input':['live_mul','^NoOp']}]},
                {'node':[dict(n,input=['missing']) if n['name']=='live_mul' else n for n in nodes]}):
        try:top_live_names(bad)
        except ValueError:pass
        else:raise AssertionError('Ambiguous/missing roots or dependencies must fail closed')
    return dict(passed=True,cases=['dead convolution and dead identity excluded',
        'stateful updates retained through control dependency',
        'function ret and controlRet roots retained, dead arithmetic excluded',
        'missing and ambiguous top roots rejected','missing dependency rejected'])


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prune_corpus(source,destination):
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Raw corpus must remain immutable')
    destination.mkdir(parents=True,exist_ok=True)
    reports=[];raw_hashes={}
    for path in sorted(source.glob('*forward*.json')):
        raw=json.loads(path.read_text())
        if not isinstance(raw,dict) or 'nodes' not in raw:continue
        graph_path=path.with_name(path.stem+'-graph.json')
        graph=json.loads(graph_path.read_text());live,roots=top_live_names(graph)
        if {n['name'] for n in raw['nodes']}!={n['name'] for n in graph['node']}:
            raise ValueError('Raw inventory and GraphDef node sets differ: '+path.name)
        raw_hashes[path.name]=sha(path);raw_hashes[graph_path.name]=sha(graph_path)
        before=len(raw['nodes']);removed=[n for n in raw['nodes'] if n['name'] not in live]
        raw['nodes']=[n for n in raw['nodes'] if n['name'] in live]
        raw['operation_inventory']=dict(collections.Counter(n['type'] for n in raw['nodes']))
        raw['registered_flops']=sum(n.get('registered_flops') or 0 for n in raw['nodes'])
        raw['unregistered_types']=sorted({n['type'] for n in raw['nodes'] if n.get('registered_flops') is None})
        raw['dependency_pruning']=dict(verified_return_nodes=roots,raw_node_count=before,live_node_count=len(live),
            raw_inventory_sha256=raw_hashes[path.name],raw_graph_sha256=raw_hashes[graph_path.name],
            convention='return-dependent data/control graph before backend compiler optimization; all stateful update dependencies retained')
        graph['node']=[n for n in graph['node'] if n['name'] in live]
        (destination/path.name).write_text(json.dumps(raw))
        (destination/graph_path.name).write_text(json.dumps(graph))
        branches_path=path.with_name(path.stem+'-branches.json')
        if branches_path.is_file():
            branches=json.loads(branches_path.read_text())
            (destination/branches_path.name).write_text(json.dumps([b for b in branches if b['name'] in live]))
        reports.append(dict(case=raw['case'],scope=raw['scope'],roots=roots,raw_nodes=before,live_nodes=len(live),
            removed_operation_counts=dict(collections.Counter(n['type'] for n in removed)),
            removed_nodes=[dict(name=n['name'],type=n['type'],registered_flops=n.get('registered_flops')) for n in removed]))
    if len(reports)!=5:raise ValueError('Expected exactly five frozen audit cases; got '+str(len(reports)))
    if not all(sha(source/name)==expected for name,expected in raw_hashes.items()):
        raise ValueError('Raw export changed during pruning')
    report=dict(fault_checks=fault_checks(),source=str(source.resolve()),raw_exports_unchanged=True,cases=reports)
    (destination/'pruning_ledger.json').write_text(json.dumps(report,indent=2))
    if (source/'summary.json').is_file():
        (destination/'raw_summary.json').write_bytes((source/'summary.json').read_bytes())
    (destination/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='removed_nodes'} for r in reports],indent=2))
    return report


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=root/'Results/person1_flops/adjscc/outputs-v4/adjscc-flops')
    parser.add_argument('--destination',type=Path,default=root/'Results/person1_flops/adjscc/pruned-v4')
    args=parser.parse_args();result=prune_corpus(args.source,args.destination)
    print(json.dumps(dict(fault_checks=result['fault_checks'],cases=[{k:v for k,v in r.items() if k!='removed_nodes'} for r in result['cases']]),indent=2))
