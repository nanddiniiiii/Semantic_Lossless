"""Source-module isolation usable by legacy author interpreters."""
from contextlib import contextmanager
import sys


@contextmanager
def upstream(directory, prefixes):
    old = sys.path[:]
    matches = lambda n: any(n == p or n.startswith(p+'.') for p in prefixes)
    saved = {n:v for n,v in sys.modules.items() if matches(n)}
    for n in saved: del sys.modules[n]
    sys.path.insert(0,str(directory))
    try:
        yield
    finally:
        sys.path[:] = old
        for n in list(sys.modules):
            if matches(n): del sys.modules[n]
        sys.modules.update(saved)
