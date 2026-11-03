"""Measured per-chunk speed/memory evidence, separate from quality results."""
import json
from pathlib import Path
import re
import statistics
import math


def speed_report(run):
    run=Path(run)
    read=lambda name:json.loads((run/name).read_text())
    summary=read('training_summary.json');status=read('train_status.json');config=read('resolved_config.json')
    rows=[json.loads(l) for l in (run/'training.jsonl').read_text().splitlines() if l.strip()]
    if status.get('status')!='completed' or not rows:raise ValueError('Speed comparison needs a normal training chunk')
    elapsed=float(status['elapsed_seconds'])
    if not math.isfinite(elapsed) or elapsed<=0:raise ValueError('Invalid training duration')
    tail=rows[-min(1000,len(rows)):];memory=[];after=[];oom=[]
    for line in (run/'resource_diagnostics.jsonl').read_text().splitlines():
        row=json.loads(line);match=re.search(r'VmRSS:\s+(\d+)',row.get('process',''))
        if match:memory.append(int(match[1])*1024)
        oom.append(row.get('cgroup_memory_events','unavailable'))
    for line in (run/'checkpoint_io.jsonl').read_text().splitlines():
        value=json.loads(line).get('host_memory_reclamation',{}).get('process_rss_after')
        if value is not None:after.append(value)
    io=read('checkpoint_io_adaptation.json')
    return dict(contract=summary['contract'],completed_epochs=summary['completed_epochs'],
        stages=sorted({x['stage'] for x in rows}),global_batch=config['batch_size'],
        measured_steps=len(rows),training_stage_seconds=elapsed,
        training_steps_per_second=len(rows)/elapsed,
        tail_window_steps=len(tail),tail_loading_seconds_median=statistics.median(x['data_seconds'] for x in tail),
        tail_compute_seconds_median=statistics.median(x['step_seconds'] for x in tail),
        peak_process_rss_bytes=max(memory) if memory else None,
        post_reclaim_rss_peak_bytes=max(after) if after else None,
        checkpoint_adapter_sha256=io['sha256'],oom_event_samples=oom,
        scope='Measured normal chunk; training-stage rate includes its validation/checkpoint overhead, excludes startup, queues and transfer',
        gpu_speedup_certified=False)


def save_speed_report(run, receipt, folder):
    receipt,folder=Path(receipt),Path(folder)
    report=speed_report(run);report['receipt']=receipt.name
    previous=folder/'latest_speed_report.json'
    if previous.exists():
        baseline=json.loads(previous.read_text())
        if (baseline.get('receipt')!=receipt.name and baseline.get('contract')==report['contract']
                and baseline.get('global_batch')==report['global_batch'] and baseline.get('stages')==report['stages']):
            report['comparison']=dict(baseline_receipt=baseline['receipt'],
                 training_stage_rate_ratio=report['training_steps_per_second']/baseline['training_steps_per_second'],
                 note='Observed comparison; host variability, cache warmup and partial-epoch position also affect rates')
    text=json.dumps(report,indent=2)
    (receipt/'speed_report.json').write_text(text);previous.write_text(text)
    return report
