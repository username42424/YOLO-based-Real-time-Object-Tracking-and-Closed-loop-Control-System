# -*- coding: utf-8 -*-
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from logparse import load_log
from harness import load_production_cfg
from trackers import _load_aim_main

name = "aim_20260906_003353.log"
mod = _load_aim_main()
engine = mod.MainEngine(load_production_cfg())
path = os.path.join(ROOT, "logs", name)
data = load_log(path)
found = 0
for ei, ep in enumerate(data["episodes"]):
    engine.reset()
    for pi, p in enumerate(ep["points"]):
        if p["net"] != (0, 0):
            engine.notify_net(*p["net"])
        dets = [{"cls": d["cls"], "conf": d["conf"], "bbox": list(d["bbox"])}
                for d in p["dets"]]
        out = engine.compute(dets, p["cx_ref"], max(1e-4, p["dt"]),
                             processing_latency_s=p["lat_s"])
        engine.end_frame()
        mm = str(out["prediction_reason"]) != str(p["reason"])
        if mm and found < 3:
            found += 1
            print("ep %d pt %d t=%.3f" % (ei, pi, p["t"]))
            print("  log: reason=%s vel_raw=(%.1f,%.1f) vel_f=(%.1f,%.1f) "
                  "lead=(%.1f,%.1f) hor=%.1f lr=%s tgt=%s"
                  % (p["reason"], p["vel_raw"][0], p["vel_raw"][1],
                     p["vel_f"][0], p["vel_f"][1], p["lead"][0], p["lead"][1],
                     p["horizon_ms"], p["lock_reason"], p["target"]))
            print("  sim: reason=%s vel_raw=(%.1f,%.1f) vel_f=(%.1f,%.1f) "
                  "lead=(%.1f,%.1f) hor=%.1f lr=%s tgt=%s"
                  % (out["prediction_reason"], out["velocity_raw"][0],
                     out["velocity_raw"][1], out["velocity_filtered"][0],
                     out["velocity_filtered"][1], out["lead"][0], out["lead"][1],
                     out["prediction_horizon_ms"], engine.diag_reason,
                     out["target"]))
            print("  axes_valid=%s frames=%s (sim) net=%s dt=%.4f"
                  % (engine._unit_velocity_valid_axes,
                     engine._unit_velocity_frames_axes, p["net"], p["dt"]))
    if found >= 3:
        break
