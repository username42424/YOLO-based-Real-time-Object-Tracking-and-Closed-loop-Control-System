# -*- coding: utf-8 -*-
"""Measure the actual default-vertical-recoil delivery rate per second during
firing windows (LMB down → up) across 30ms vs 40ms log groups."""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

GROUPS = {
    "cadence30_robust": ["aim_20260906_135713.log", "aim_20260906_140550.log",
                         "aim_20260906_141313.log"],
    "cadence40_robust": ["aim_20260906_125713.log", "aim_20260906_130338.log",
                         "aim_20260906_131131.log"],
}
# [瞄准观测] rows carry observed_recoil per capture interval with capture_t.
MARKER = "[瞄准观测] "
LMB_DOWN = re.compile(r"RMB→LMB down t=([0-9.]+)|Raw LMB down t=([0-9.]+)")
LMB_UP = re.compile(r"LMB up t=([0-9.]+)")


def analyze(path):
    obs = []          # (capture_t, recoil_y, dt)
    fire_windows = [] # (start, end)
    firing = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if MARKER in line:
            try:
                row = json.loads(line.split(MARKER, 1)[1])
            except Exception:
                continue
            t = float(row["capture_t"])
            rec = row.get("observed_recoil") or [0, 0]
            obs.append((t, float(rec[1])))
        elif "RMB→LMB down" in line or "Raw LMB down" in line:
            m = LMB_DOWN.search(line)
            if m and firing is None:
                firing = float(m.group(1) or m.group(2))
        elif "LMB up" in line and firing is not None:
            m = LMB_UP.search(line)
            if m:
                fire_windows.append((firing, float(m.group(1))))
                firing = None
    if firing is not None and obs:
        fire_windows.append((firing, obs[-1][0]))
    # per-second recoil delivery inside firing windows
    total_counts = 0.0
    total_dt = 0.0
    for (t0, t1) in fire_windows:
        for i in range(1, len(obs)):
            t, rc = obs[i]
            pt = obs[i - 1][0]
            # the recoil in row i was sent during (pt, t]
            lo, hi = max(pt, t0), min(t, t1)
            if hi > lo and t > pt:
                frac = (hi - lo) / (t - pt)
                total_counts += rc * frac
                total_dt += t - pt
    rate = total_counts / total_dt if total_dt > 0 else None
    return {
        "fire_windows": len(fire_windows),
        "firing_seconds": round(total_dt, 2),
        "recoil_counts_total": round(total_counts, 1),
        "recoil_counts_per_second": round(rate, 1) if rate else None,
    }


def main():
    out = {}
    for g, logs in GROUPS.items():
        rates = []
        details = {}
        for name in logs:
            d = analyze(os.path.join(ROOT, "logs", name))
            details[name] = d
            if d["recoil_counts_per_second"]:
                rates.append(d["recoil_counts_per_second"])
        out[g] = {
            "logs": details,
            "rate_per_second_median": (statistics.median(rates) if rates else None),
        }
    for g, v in out.items():
        print(g, "median recoil rate during fire: %s counts/s" % v["rate_per_second_median"])
        for name, d in v["logs"].items():
            print("   ", name, d)
    os.makedirs(os.path.join(ROOT, "日志V4", "recoil"), exist_ok=True)
    with open(os.path.join(ROOT, "日志V4", "recoil", "recoil_rate_check.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved 日志V4/recoil/recoil_rate_check.json")


if __name__ == "__main__":
    import statistics
    main()
