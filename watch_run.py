#!/usr/bin/env python3
"""
Monitor a sim3d training log for known exploit patterns.

Usage: python watch_run.py /tmp/week8-run-01.log
Prints ALERT lines to stdout when suspicious patterns are detected.
"""

import sys
import time
import re

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/week8-run-01.log"

# Exploit thresholds
HANG_STILL_EP_LEN   = 400   # ep_len_mean above this...
HANG_STILL_COM_GAP  = 0.10  # ...while com_z hasn't improved by this much since last check
FARMING_REW         = 300   # ep_rew_mean above this while com_z still low
FARMING_COM_MAX     = 1.0   # "still low" = com_z below this (a full route is ~2m)
STALE_COM_STEPS     = 100_000  # steps without com_z improvement = stagnation

RE_EP_LEN  = re.compile(r"ep_len_mean\s*\|\s*([\d.]+)")
RE_EP_REW  = re.compile(r"ep_rew_mean\s*\|\s*(-?[\d.]+)")
RE_STEPS   = re.compile(r"total_timesteps\s*\|\s*([\d]+)")
RE_COM_Z   = re.compile(r"new (?:max com_z|best progress_z)=([\d.]+) m at step (\d+)")
RE_AVG_REW = re.compile(r"new best avg_rew=(-?[\d.]+) at step (\d+)")

def fmt(val, width=8):
    return str(val).rjust(width)

def check(ep_len, ep_rew, steps, progress_z, max_com_step):
    alerts = []
    steps_since_com = steps - max_com_step

    if ep_len > HANG_STILL_EP_LEN and steps_since_com > STALE_COM_STEPS:
        alerts.append(
            f"HANG-STILL EXPLOIT? ep_len={ep_len:.0f} but com_z stuck at "
            f"{progress_z:.3f}m for {steps_since_com:,} steps"
        )

    if ep_rew > FARMING_REW and progress_z < FARMING_COM_MAX:
        alerts.append(
            f"REWARD FARMING? ep_rew={ep_rew:.0f} but max com_z only "
            f"{progress_z:.3f}m (agent not actually climbing)"
        )

    return alerts

def main():
    print(f"Watching: {LOG_PATH}")
    print("NOTE: progress_z prefers max gripped-hold height; older logs may report max COM.")
    print(f"Thresholds — hang-still: ep_len>{HANG_STILL_EP_LEN} + no com_z gain in {STALE_COM_STEPS:,} steps")
    print(f"            farming:    ep_rew>{FARMING_REW} + progress_z<{FARMING_COM_MAX}m")
    print("-" * 60)

    ep_len = ep_rew = 0.0
    steps = max_com_step = 0
    progress_z = 0.0
    last_reported_steps = -1

    with open(LOG_PATH, "r") as f:
        # First pass: scan existing file to seed com_z / step state
        for line in f:
            m = RE_COM_Z.search(line)
            if m:
                z, s = float(m.group(1)), int(m.group(2))
                if z > progress_z:
                    progress_z, max_com_step = z, s
            m = RE_STEPS.search(line)
            if m:
                steps = int(m.group(1))
        print(f"(caught up to step {steps:,} — progress_z so far: {progress_z:.3f}m at step {max_com_step:,})")
        print("(tailing new output...)\n")

        while True:
            line = f.readline()
            if not line:
                time.sleep(2)
                continue

            line = line.strip()

            m = RE_EP_LEN.search(line)
            if m:
                ep_len = float(m.group(1))

            m = RE_EP_REW.search(line)
            if m:
                ep_rew = float(m.group(1))

            m = RE_STEPS.search(line)
            if m:
                steps = int(m.group(1))
                if steps != last_reported_steps:
                    last_reported_steps = steps
                    alerts = check(ep_len, ep_rew, steps, progress_z, max_com_step)
                    status = (
                        f"[{steps:>9,} steps] "
                        f"ep_len={ep_len:6.1f}  "
                        f"ep_rew={ep_rew:8.1f}  "
                        f"progress_z={progress_z:.3f}m  "
                        f"(no gain for {steps - max_com_step:,} steps)"
                    )
                    print(status)
                    for a in alerts:
                        print(f"  *** ALERT: {a} ***")
                    sys.stdout.flush()

            m = RE_COM_Z.search(line)
            if m:
                z, s = float(m.group(1)), int(m.group(2))
                if z > progress_z:
                    progress_z = z
                    max_com_step = s
                    print(f"  >> new max com_z={progress_z:.3f}m at step {max_com_step:,}")
                    sys.stdout.flush()

if __name__ == "__main__":
    main()
