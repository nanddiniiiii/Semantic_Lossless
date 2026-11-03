"""Small persisted host/GPU diagnostics for externally killed Kaggle workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def snapshot():
    record=dict(time_unix=time.time(),pid=os.getpid())
    for filename,label in [('/proc/self/status','process'),('/proc/meminfo','host_memory'),
                           ('/proc/pressure/memory','memory_pressure'),('/proc/pressure/io','io_pressure'),
                           ('/sys/fs/cgroup/memory.current','cgroup_memory_current'),
                           ('/sys/fs/cgroup/memory.max','cgroup_memory_limit'),
                           ('/sys/fs/cgroup/memory.events','cgroup_memory_events')]:
        try:record[label]=Path(filename).read_text()[:12000]
        except OSError:pass
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,pstate',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
        record['gpu']=result.stdout.strip();record['gpu_query_returncode']=result.returncode
    except (OSError,subprocess.TimeoutExpired):record['gpu_query']='unavailable'
    return record


def install():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--run')
    args,_=parser.parse_known_args(sys.argv[1:])
    if not args.run:raise ValueError('Resource diagnostics require worker output directory')
    path=Path(args.run)/'resource_diagnostics.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    def sample():
        with path.open('a') as stream:stream.write(json.dumps(snapshot())+'\n');stream.flush()
    sample()
    def monitor():
        while True:
            time.sleep(30)
            try:sample()
            except OSError:pass
    threading.Thread(target=monitor,daemon=True).start()
    print('Persisted host/cgroup/GPU diagnostics enabled; 30-second sampling',flush=True)
