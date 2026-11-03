import json
from pathlib import Path
import tempfile
import unittest
from scripts.person1_speed_report import save_speed_report, speed_report


class SpeedEvidence(unittest.TestCase):
    def test_measured_values_comparison_and_rejection_of_failed_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';run.mkdir();receipt=root/'receipt';receipt.mkdir();reports=root/'reports';reports.mkdir()
            def write(name,value):(run/name).write_text(json.dumps(value))
            write('training_summary.json',dict(contract='same',completed_epochs={'denoiser':20}))
            write('train_status.json',dict(status='completed',elapsed_seconds=10))
            write('resolved_config.json',dict(batch_size=2))
            write('checkpoint_io_adaptation.json',dict(sha256='reviewed'))
            rows=[dict(stage='denoiser',data_seconds=x,step_seconds=y) for x,y in [(.1,.07),(.3,.09)]]
            (run/'training.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            (run/'resource_diagnostics.jsonl').write_text(json.dumps(dict(process='VmRSS: 3145728 kB',cgroup_memory_events='oom 0')))
            (run/'checkpoint_io.jsonl').write_text(json.dumps(dict(host_memory_reclamation=dict(process_rss_after=1024))))
            first=save_speed_report(run,receipt,reports)
            self.assertEqual(first['measured_steps'],2);self.assertEqual(first['training_steps_per_second'],.2)
            self.assertAlmostEqual(first['tail_loading_seconds_median'],.2)
            self.assertEqual(first['peak_process_rss_bytes'],3*1024**3)
            other=root/'other';other.mkdir();write('train_status.json',dict(status='completed',elapsed_seconds=5))
            second=save_speed_report(run,other,reports)
            self.assertEqual(second['comparison']['training_stage_rate_ratio'],2)
            write('train_status.json',dict(status='failed',elapsed_seconds=5))
            with self.assertRaises(ValueError):speed_report(run)
