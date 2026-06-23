"""
debug.py — Global debug flag and @trace decorator for doit_agent.

Set DEBUG = True here, or export DOIT_DEBUG=1 in your shell to enable logs.
"""
from __future__ import annotations

import functools
import os
import time

# ──────────────────────────────────────────────
# Global switch — flip this or set DOIT_DEBUG=1
# ──────────────────────────────────────────────
DEBUG: bool = True#os.getenv("DOIT_DEBUG", "1").strip().lower() in {"1", "true", "yes", "y"}


def trace(fn):
    """
    Decorator that prints [DEBUG] entry and exit lines when DEBUG is True.
    Works on plain functions and methods alike.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not DEBUG:
            return fn(*args, **kwargs)

        # Build a readable qualified name  e.g. DoitAgent.decide
        qualname = fn.__qualname__

        # Show first arg as class name when it looks like a method
        print(f"[DEBUG] --> {qualname}()", flush=True)
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[DEBUG] <-- {qualname}() done [{elapsed:.1f}ms]", flush=True)
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[DEBUG] <-- {qualname}() RAISED {type(exc).__name__}: {exc} [{elapsed:.1f}ms]", flush=True)
            raise

    return wrapper
