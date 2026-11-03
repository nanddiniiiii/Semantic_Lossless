import json
from pathlib import Path
import tempfile
import unittest
from scripts.person1_chunk_policy import choose_chunk_seconds


class ChunkPolicy(unittest.TestCase):
    def test_extended_chunk_requires_complete_matching_safe_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'checkpoint_io_adaptation.json').write_text(json.dumps(dict(sha256='reviewed')))
            rows=[dict(time_unix=i*30,process='VmRSS: 2097152 kB\n',
                   cgroup_memory_events='oom 0\noom_kill 0\noom_group_kill 0\n') for i in range(50)]
            saves=[dict(host_memory_reclamation=dict(process_rss_after=2*1024**3)) for _ in range(4)]
            (p/'training.jsonl').write_text(json.dumps(dict(data_seconds=.01,step_seconds=.07)))
            def write():
                (p/'resource_diagnostics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
                (p/'checkpoint_io.jsonl').write_text('\n'.join(map(json.dumps,saves)))
            write();self.assertEqual(choose_chunk_seconds(p,'reviewed'),5400)
            self.assertEqual(choose_chunk_seconds(p,'reviewed',cudnn_plan_cache_limit=-1),2700)
            (p/'training.jsonl').write_text(json.dumps(dict(data_seconds=.5,step_seconds=.07)))
            self.assertEqual(choose_chunk_seconds(p,'reviewed'),2700)
            (p/'training.jsonl').write_text(json.dumps(dict(data_seconds=.01,step_seconds=.07)))
            self.assertEqual(choose_chunk_seconds(p,'different-source'),2700)
            rows[-1]['process']='VmRSS: 19922944 kB\n';write()
            self.assertEqual(choose_chunk_seconds(p,'reviewed'),2700)
            rows[-1]['process']='VmRSS: 2097152 kB\n'
            rows[-1]['cgroup_memory_events']='oom 1\noom_kill 1\n';write()
            self.assertEqual(choose_chunk_seconds(p,'reviewed'),2700)
            rows[-1]['cgroup_memory_events']='unavailable';write()
            self.assertEqual(choose_chunk_seconds(p,'reviewed'),2700)
            rows[-1]['cgroup_memory_events']='oom 0\noom_kill 0\n'
            saves[-1]['host_memory_reclamation']['process_rss_after']=None;write()
            self.assertEqual(choose_chunk_seconds(p,'reviewed'),2700)
