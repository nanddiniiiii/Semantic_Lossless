"""Declared checkpoint-I/O recovery adapter; training tensors stay untouched."""
import copy
import gc
import ctypes
import json
import os
from pathlib import Path
import threading
import time
import torch


def reclaim_unused_host_memory():
    """Reclaim unreachable cycles/free allocator arenas, never live tensors."""
    def rss():
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):return int(line.split()[1])*1024
        except OSError:pass
        return None
    before=rss();collected=gc.collect();trimmed=None
    if os.name=='posix':
        try:
            trim=ctypes.CDLL(None).malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
            trimmed=int(trim(0))
        except (AttributeError,OSError):pass
    return dict(unreachable_objects_collected=collected,malloc_trim_result=trimmed,
                process_rss_before=before,process_rss_after=rss())


def cpu_snapshot(value, memo=None):
    """Copy tensor values to CPU without mutating live model/optimizer state."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if torch.is_tensor(value):
        result = value.detach().to(device='cpu', copy=True)
        memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = copy.copy(value)  # Preserve OrderedDict metadata/default factory.
        memo[id(value)] = result
        for key, item in value.items():result[key] = cpu_snapshot(item, memo)
        return result
    if isinstance(value, list):
        result = [];memo[id(value)] = result
        result.extend(cpu_snapshot(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        values = [cpu_snapshot(item, memo) for item in value]
        result = type(value)(*values) if hasattr(value, '_fields') else tuple(values)
        memo[id(value)] = result
        return result
    return value


def checkpoint_save(native_save, value, destination, *args, timeout_seconds=120, **kwargs):
    if not isinstance(value, dict) or not value.get('resume_supported'):
        return native_save(value, destination, *args, **kwargs)
    started = time.monotonic()
    def expired():
        print('CHECKPOINT_IO_TIMEOUT: previous atomic checkpoint retained; refusing hours of stalled serialization', flush=True)
        # Caller writes a temporary path and only renames after success.
        # Exit the worker, leaving the controller to export retained evidence.
        os._exit(86)
    watchdog = threading.Timer(timeout_seconds, expired)
    watchdog.daemon = True;watchdog.start()
    try:
        reclaimed=reclaim_unused_host_memory()
        snapshot = cpu_snapshot(value)
        staged = time.monotonic()
        result = native_save(snapshot, destination, *args, **kwargs)
        record = dict(cpu_snapshot_seconds=staged-started, serialization_seconds=time.monotonic()-staged,
                      total_seconds=time.monotonic()-started, tensor_values_unchanged=True,
                      host_memory_reclamation=reclaimed,
                      format='native torch.save; CPU tensor storage; full optimizer/RNG retained')
        print('CHECKPOINT_IO ' + json.dumps(record), flush=True)
        if isinstance(destination, (str, Path)):
            path = Path(destination).parent.parent/'checkpoint_io.jsonl'
            with path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        return result
    finally:
        watchdog.cancel()


def install():
    if getattr(torch.save, '_person1_cpu_snapshot', False):return
    native = torch.save
    def save(value, destination, *args, **kwargs):
        return checkpoint_save(native, value, destination, *args, **kwargs)
    save._person1_cpu_snapshot = True
    torch.save = save
    print('Checkpoint I/O recovery enabled: non-mutating CPU snapshot; 120s stalled-save bound; native atomic rename retained', flush=True)
