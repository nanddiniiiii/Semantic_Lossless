"""Allow fewer restarts only after measured, bounded-memory recovery evidence."""
import json
from pathlib import Path
import re
import statistics
import math


def choose_chunk_seconds(run, adapter_sha, default=2700, cudnn_plan_cache_limit=None):
    run=Path(run)
    try:
        identity=json.loads((run/'checkpoint_io_adaptation.json').read_text())
        if identity.get('sha256')!=adapter_sha:return default
        if identity.get('cudnn_plan_cache_limit')!=cudnn_plan_cache_limit:return default
        rows=[json.loads(l) for l in (run/'resource_diagnostics.jsonl').read_text().splitlines() if l.strip()]
        saves=[json.loads(l) for l in (run/'checkpoint_io.jsonl').read_text().splitlines() if l.strip()]
        steps=[json.loads(l) for l in (run/'training.jsonl').read_text().splitlines() if l.strip()][-1000:]
        data=statistics.median(x['data_seconds'] for x in steps)
        compute=statistics.median(x['step_seconds'] for x in steps)
        if not math.isfinite(data) or not math.isfinite(compute) or data>.03 or compute>.12:return default
        if len(rows)<40 or rows[-1]['time_unix']-rows[0]['time_unix']<1200 or len(saves)<3:return default
        for row in rows:
            rss=re.search(r'VmRSS:\s+(\d+)',row.get('process',''))
            events=dict(re.findall(r'^(oom|oom_kill|oom_group_kill)\s+(\d+)\s*$',row.get('cgroup_memory_events',''),re.M))
            if not rss or int(rss[1])*1024>8*1024**3:return default
            if not {'oom','oom_kill'}.issubset(events) or any(int(n) for n in events.values()):return default
        for save in saves:
            reclaim=save.get('host_memory_reclamation',{})
            after=reclaim.get('process_rss_after')
            if after is None or after>6*1024**3:return default
        return 5400  # Ninety minutes, still below the agreed four-hour maximum.
    except (OSError,ValueError,KeyError,TypeError):
        return default
