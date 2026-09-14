# -*- coding: utf-8 -*-
"""Print per-log config snapshots and episode statistics."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logparse import load_log, fill_world, config_snapshot

LOGS = [
    "aim_20260906_003353.log",
    "aim_20260906_001842.log",
    "aim_20260906_000134.log",
    "aim_20260905_232312.log",
    "aim_20260905_231714.log",
    "aim_20260905_230935.log",
    "aim_20260905_230244.log",
]
ROOT = r"C:\Users\12951\Desktop\12323\logs"

import math


def pctl(v, q):
    if not v:
        return None
    s = sorted(v)
    return s[round(q * (len(s) - 1))]


for name in LOGS:
    path = os.path.join(ROOT, name)
    data = load_log(path)
    snap = data["snap"]
    eps = data["episodes"]
    vx_all, sp_all = [], []
    n_strafe = 0
    total_obs = 0
    for ep in eps:
        fill_world(ep)
        pts = ep["points"]
        total_obs += len(pts)
        for p in pts[1:]:
            if p["vx"] is None:
                continue
            vx_all.append(p["vx"])
            sp_all.append(math.hypot(p["vx"], p["vy"]))
            if abs(p["vx"]) >= 400.0 and abs(p["vx"]) >= 1.5 * abs(p["vy"]):
                n_strafe += 1
    print("=" * 100)
    print(name)
    print("  snap:", json.dumps({k: v for k, v in snap.items()}, ensure_ascii=False))
    print("  episodes=%d observations=%d mouse_ticks=%d" %
          (len(eps), total_obs, len(data["ticks"])))
    print("  |vx| p50/p75/p90/max: %.0f / %.0f / %.0f / %.0f px/s" % (
        pctl([abs(v) for v in vx_all], .5) or -1,
        pctl([abs(v) for v in vx_all], .75) or -1,
        pctl([abs(v) for v in vx_all], .9) or -1,
        pctl([abs(v) for v in vx_all], 1.0) or -1))
    print("  strafe samples (|vx|>=400 & |vx|>=1.5|vy|): %d" % n_strafe)
