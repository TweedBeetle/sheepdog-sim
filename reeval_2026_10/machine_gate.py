#!/usr/bin/env python3
"""Allocator gate (07.10.): ps median < 2 s and swap-ins < 2,000 per 15 s. Exit 0 = OK."""
import re, statistics, subprocess, sys, time
def ps_time():
    t = time.time(); subprocess.run(["ps", "-axo", "pid,comm"], capture_output=True); return time.time() - t
def swapins():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Swapins:\s+(\d+)", out).group(1))
s0 = swapins(); ts = []
for _ in range(5):
    ts.append(ps_time()); time.sleep(2.5)
time.sleep(max(0, 15 - sum(ts) - 12.5))
s1 = swapins()
med = statistics.median(ts); d = s1 - s0
ok = med < 2.0 and d < 2000
print(f"ps median {med:.2f}s  swapins/15s {d}  -> {'OK' if ok else 'BUSY'}")
sys.exit(0 if ok else 1)
