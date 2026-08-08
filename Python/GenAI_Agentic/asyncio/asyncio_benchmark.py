"""Benchmark: blocking sync vs. sequential await vs. asyncio.gather.

Two workloads:
  1. I/O-bound  (sleep -> yields to the event loop)
  2. CPU-bound  (busy loop -> never yields)
gather() only helps when the coroutines actually await something.

Expected results:
10 tasks, 0.1s simulated latency each

I/O-BOUND
  sync (blocking, one by one)         1.002s
  async sequential await              1.005s
  asyncio.gather                      0.101s <==== THIS
  -> gather is 9.9x faster than sync

CPU-BOUND
  sync (blocking, one by one)         1.120s
  async sequential await              1.061s
  asyncio.gather                      1.072s <==== THIS
  -> gather is 1.04x vs sync

"""
import asyncio
import time

# ====================================================================
#  CONFIG  -  which knob feeds which test
# ====================================================================
# BOTH tests: how many calls each of the 3 strategies makes per workload.
# Raising this widens the gather() gap on I/O (ceiling = connection limit),
# but changes nothing on CPU.
N_TASKS = 10

# I/O-BOUND test only: per-task fake latency, used by blocking_io() via
# time.sleep() and by async_io() via asyncio.sleep(). Also the theoretical
# floor that gather() should converge to no matter how big N_TASKS gets.
IO_DELAY = 0.1

# CPU-BOUND test only: busy-loop length in blocking_cpu()/async_cpu().
# Tune so the CPU total lands near the I/O total (N_TASKS * IO_DELAY = 1.0s
# by default), otherwise the two sections aren't comparable at a glance.
CPU_ITERS = 2_000_000

# ====================================================================
#  HELPERS
# ====================================================================
def timed(label, fn, *args):
    """Run fn, print elapsed wall-clock time, return the duration."""
    start = time.perf_counter()
    fn(*args)
    elapsed = time.perf_counter() - start
    print(f"  {label:<34} {elapsed:6.3f}s")
    return elapsed

# ====================================================================
#  I/O-BOUND WORKLOAD
# ====================================================================
def blocking_io(n):
    """Plain synchronous function - blocks the whole thread."""
    time.sleep(IO_DELAY)
    return n

async def async_io(n):
    """Yields control back to the event loop while waiting."""
    await asyncio.sleep(IO_DELAY)
    return n

def run_sync_io():
    return [blocking_io(i) for i in range(N_TASKS)]

async def run_sequential_io():
    """Awaiting one at a time - no overlap, same as sync."""
    return [await async_io(i) for i in range(N_TASKS)]

async def run_gather_io():
    """All coroutines in flight at once."""
    return await asyncio.gather(*(async_io(i) for i in range(N_TASKS)))

# ====================================================================
#  CPU-BOUND WORKLOAD
# ====================================================================
def blocking_cpu(n):
    return sum(i * i for i in range(CPU_ITERS))

async def async_cpu(n):
    """async in name only - there is no await, so it never yields."""
    return sum(i * i for i in range(CPU_ITERS))

def run_sync_cpu():
    return [blocking_cpu(i) for i in range(N_TASKS)]

async def run_sequential_cpu():
    return [await async_cpu(i) for i in range(N_TASKS)]

async def run_gather_cpu():
    return await asyncio.gather(*(async_cpu(i) for i in range(N_TASKS)))

# ====================================================================
#  MAIN
# ====================================================================
def main():
    print(f"\n{N_TASKS} tasks, {IO_DELAY}s simulated latency each\n")
    print("I/O-BOUND")
    t_sync = timed("sync (blocking, one by one)", run_sync_io)
    timed("async sequential await", lambda: asyncio.run(run_sequential_io()))
    t_gat = timed("asyncio.gather", lambda: asyncio.run(run_gather_io()))
    print(f"  -> gather is {t_sync / t_gat:.1f}x faster than sync")
    print(f"  -> theoretical floor: {IO_DELAY:.3f}s (one task's latency)\n")
    print("CPU-BOUND")
    c_sync = timed("sync (blocking, one by one)", run_sync_cpu)
    timed("async sequential await", lambda: asyncio.run(run_sequential_cpu()))
    c_gat = timed("asyncio.gather", lambda: asyncio.run(run_gather_cpu()))
    print(f"  -> gather is {c_sync / c_gat:.2f}x vs sync (expect ~1.0 or worse)\n")

if __name__ == "__main__":
    main()
