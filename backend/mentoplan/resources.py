"""Resource safety: memory checks, thread caps and memory-capped child processes.

Heavy jobs (case processing, the nonlinear refinement) run in a separate process whose private memory
is hard-limited (RLIMIT_DATA), so an unexpectedly large problem fails cleanly with an error message
instead of exhausting the machine's RAM.  Only one heavy job runs at a time.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback

HEAVY_JOB_LOCK = threading.Lock()          # at most one heavy job per server
DEFAULT_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))


def cap_threads(n: int = DEFAULT_THREADS):
    """Limit BLAS / OpenMP / MKL threads (fewer threads also means less solver workspace)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(n))


def available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("inf")


def job_memory_limit_mb(requested: float | None = None) -> float:
    """Memory budget for a child job: the request, but never more than 60% of what is available now."""
    avail = available_mb()
    cap = 0.6 * avail if avail != float("inf") else (requested or 6000)
    return min(requested or cap, cap)


class InsufficientMemory(RuntimeError):
    pass


def ensure_memory(min_mb: float, what: str):
    avail = available_mb()
    if avail < min_mb:
        raise InsufficientMemory(f"{what} needs about {min_mb / 1024:.1f} GB of free memory, "
                                 f"only {avail / 1024:.1f} GB available - close other programs and retry")


def _child(fn, args, kwargs, limit_mb, conn):
    try:
        import resource
        b = int(limit_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_DATA, (b, b))
    except Exception:
        pass
    try:
        conn.send(("ok", fn(*args, **kwargs)))
    except MemoryError:
        conn.send(("error", f"memory limit of {limit_mb / 1024:.1f} GB reached"))
    except BaseException as e:      # pragma: no cover - reported to the parent
        conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"))
    finally:
        conn.close()


def run_isolated(fn, *args, limit_mb: float, timeout_s: float | None = None, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a fresh process with a hard private-memory limit.

    Returns fn's (picklable) result; raises RuntimeError with the child's error otherwise."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(fn, args, kwargs, limit_mb, child), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout_s):
            p.kill()
            raise RuntimeError("job timed out")
        status, payload = parent.recv()
    except EOFError:
        p.join(5)
        raise RuntimeError(f"job process ended unexpectedly (exit code {p.exitcode}); "
                           f"most likely the {limit_mb / 1024:.1f} GB memory limit")
    finally:
        p.join(10)
        if p.is_alive():
            p.kill()
    if status != "ok":
        raise RuntimeError(payload)
    return payload
