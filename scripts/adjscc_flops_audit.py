"""Prepare separate CPU graph accounting; never train or overwrite saved weights."""
from pathlib import Path
import json,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ['PATH']=str(ROOT/'.venv-collector/Scripts')+os.pathsep+os.environ['PATH']
from src.person1.auth import ensure_fresh_authorization
ensure_fresh_authorization('hmmmkk')
base=ROOT/'Results/person1_campaign/hmmmkk/adjscc-evaluation-only'
folder=ROOT/'Results/person1_flops/adjscc';folder.mkdir(parents=True,exist_ok=True)
nb=json.loads((base/'notebook.ipynb').read_text());nb['cells']=nb['cells'][:3]
code=r'''
import json,collections
from src.person1.controller import command
from src.person1.environments import prepare
from src.sources import fetch_source,source_path
from src.runtime import export
run=Path('/kaggle/working/adjscc-flops');run.mkdir()
asset=next(Path('/kaggle/input').rglob('checkpoint.bin')).parent
config=json.loads((asset/'config.json').read_text());fetch_source('adjscc_author');config['source_dir']=str(source_path('adjscc_author'))
(run/'config.json').write_text(json.dumps(config))
python,env=prepare('adjscc',REPO,run/'dependencies.log',command)
program=r"""
import tensorflow as tf,json,collections,sys
from pathlib import Path
from src.person1.tf_worker import build
from tensorflow.python.framework import ops
tf.keras.utils.set_random_seed(22)
run=Path(sys.argv[1]);options=json.loads((run/'config.json').read_text());results=[]
for name,training,batch,h,w in [('inference',False,1,512,768),('train_global',True,16,128,128),('train_replica',True,8,128,128)]:
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(22)
 model,optimizer=build(options,tf.distribute.OneDeviceStrategy('/cpu:0'))
 def forward(x):return model([x,tf.fill([batch,1],10.)],training=training)
 @tf.function
 def full_step(x):
  with tf.GradientTape() as tape:
   y=forward(x);losses=tf.reduce_mean(tf.square(y-x),axis=[1,2,3]);loss=tf.nn.compute_average_loss(losses,global_batch_size=16)
  gradients=tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(gradients,model.trainable_variables))
  return loss
 functions={'forward':tf.function(forward)}
 if training:functions['forward_backward_adam']=full_step
 for scope,function in functions.items():
  graph=function.get_concrete_function(tf.TensorSpec([batch,h,w,3],tf.float32)).graph
  from google.protobuf.json_format import MessageToJson
  (run/(name+'-'+scope+'-graph.json')).write_text(MessageToJson(graph.as_graph_def(add_shapes=True)))
  nodes=[]
  for op in graph.get_operations():
   try:count=ops.get_stats_for_node_def(graph,op.node_def,'flops').value
   except Exception:count=None
   if count is not None:count=int(count)
   nodes.append(dict(name=op.name,type=op.type,registered_flops=count,inputs=[dict(shape=t.shape.as_list() if t.shape.rank is not None else None,dtype=t.dtype.name) for t in op.inputs],outputs=[dict(shape=t.shape.as_list() if t.shape.rank is not None else None,dtype=t.dtype.name) for t in op.outputs]))
  branch_constants=[]
  for op in graph.get_operations():
   if op.type=='StatelessIf':
    value=tf.get_static_value(op.inputs[0])
    branch_constants.append(dict(name=op.name,static_predicate=None if value is None else bool(value),then_branch=op.get_attr('then_branch').name,else_branch=op.get_attr('else_branch').name))
  (run/(name+'-'+scope+'-branches.json')).write_text(json.dumps(branch_constants))
  result=dict(case=name,scope=scope,batch=batch,height=h,width=w,trainable_parameters=sum(int(v.shape.num_elements()) for v in model.trainable_variables),loss_definition="native per-image MSE then compute_average_loss globalbatch16",registered_flops=sum(n['registered_flops'] or 0 for n in nodes),operation_inventory=dict(collections.Counter(n['type'] for n in nodes)),nodes=nodes,unregistered_types=sorted(set(n['type'] for n in nodes if n['registered_flops'] is None)),complete=False,convention='logical arithmetic;2operations/MAC;unregistered arithmetic requires explicit accounting')
  (run/(name+'-'+scope+'.json')).write_text(json.dumps(result));results.append({k:v for k,v in result.items() if k!='nodes'})
(run/'summary.json').write_text(json.dumps(results,indent=2))
"""
(run/'audit.py').write_text(program)
try:command([python,'-u',str(run/'audit.py'),str(run)],run/'audit.log',18000,env)
finally:export(run)
'''
nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(True)))
for c in nb['cells']:
 if c['cell_type']=='code':compile(''.join(c['source']),'flops-notebook','exec')
(folder/'notebook.ipynb').write_text(json.dumps(nb))
meta=json.loads((base/'kernel-metadata.json').read_text());meta.update(id='hmmmkk/person1-adjscc-arithmetic-audit',title='Person1 ADJSCC arithmetic audit')
(folder/'kernel-metadata.json').write_text(json.dumps(meta))
if (folder/'submission.json').exists():raise RuntimeError('Audit already submitted; inspect existing job')
subprocess.run(['kaggle','kernels','push','-p',str(folder)],check=True)
(folder/'submission.json').write_text(json.dumps(dict(kernel=meta['id'],device='CPU')))
