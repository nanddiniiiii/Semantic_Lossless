from collections import OrderedDict
import io
import subprocess
import sys
import unittest
import weakref
import random
import torch
from scripts.person1_checkpoint_runtime import cpu_snapshot, checkpoint_save, reclaim_unused_host_memory


class CheckpointRecovery(unittest.TestCase):
    def test_reclamation_keeps_live_tensors_and_rng_unchanged(self):
        class Cycle:pass
        cycle=Cycle();cycle.self=cycle;reference=weakref.ref(cycle);del cycle
        live=torch.tensor([1.,2.]);before=live.clone()
        torch_rng=torch.get_rng_state().clone();python_rng=random.getstate()
        result=reclaim_unused_host_memory()
        self.assertIsNone(reference())
        self.assertTrue(torch.equal(live,before))
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_rng))
        self.assertEqual(random.getstate(),python_rng)
        self.assertGreater(result['unreachable_objects_collected'],0)

    def test_snapshot_preserves_metadata_alias_and_live_values(self):
        tensor=torch.tensor([1.,2.],requires_grad=True)
        state=OrderedDict(weight=tensor);state._metadata={'':{'version':2}}
        source=dict(state_dict=state,alias=tensor,progress=dict(optimizer_states={'step':torch.tensor(3.)}),resume_supported=True)
        result=cpu_snapshot(source)
        self.assertEqual(result['state_dict']._metadata,state._metadata)
        self.assertIs(result['alias'],result['state_dict']['weight'])
        self.assertIsNot(result['alias'],tensor)
        result['alias'][0]=9
        self.assertEqual(tensor[0].item(),1.)

    def test_roundtrip_optimizer_continuation(self):
        torch.manual_seed(22);model=torch.nn.Linear(3,2);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.randn(4,3)
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
        value=dict(state_dict=model.state_dict(),progress=dict(optimizer_states=optimizer.state_dict()),resume_supported=True)
        stream=io.BytesIO();checkpoint_save(torch.save,value,stream)
        stream.seek(0);saved=torch.load(stream,weights_only=False)
        other=torch.nn.Linear(3,2);other.load_state_dict(saved['state_dict'],strict=True)
        restored=torch.optim.Adam(other.parameters(),lr=.001);restored.load_state_dict(saved['progress']['optimizer_states'])
        for network,opt in [(model,optimizer),(other,restored)]:
            opt.zero_grad();network(x).square().mean().backward();opt.step()
        for a,b in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))

    def test_stalled_save_exits_bounded_subprocess(self):
        code="from scripts.person1_checkpoint_runtime import checkpoint_save;import time;checkpoint_save(lambda *a:time.sleep(10),{'resume_supported':True},None,timeout_seconds=.05)"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,86)
        self.assertIn('CHECKPOINT_IO_TIMEOUT',result.stdout)


if __name__=='__main__':unittest.main()
