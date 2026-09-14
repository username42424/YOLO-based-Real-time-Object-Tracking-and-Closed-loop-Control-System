# -*- coding: utf-8 -*-
import copy
import os
import sys

ROOT = r"C:\Users\12951\Desktop\12323"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))
sys.path.insert(0, os.path.join(ROOT, "shootsim", "ts_v2"))

from aim_engine import MainEngine
from exp_engines import V3Engine, LEGACY_PARAMS
from logparse import load_log
from tests.test_tracking_v3 import base_cfg

cfg = base_cfg()
data = load_log(os.path.join(ROOT, "logs", "aim_20260906_003353.log"))
group = data["episodes"][5]
e1 = MainEngine(copy.deepcopy(cfg))
e2 = V3Engine(copy.deepcopy(cfg), dict(LEGACY_PARAMS))
for i, p in enumerate(group["points"][:12]):
    for eng in (e1, e2):
        if p["net"] != (0, 0):
            eng.notify_net(*p["net"])
    dets = [{"cls": d["cls"], "conf": d["conf"], "bbox": list(d["bbox"])}
            for d in p["dets"]]
    o1 = e1.compute(dets, p["cx_ref"], max(1e-4, p["dt"]),
                    processing_latency_s=p["lat_s"])
    o2 = e2.compute(dets, p["cx_ref"], max(1e-4, p["dt"]),
                    processing_latency_s=p["lat_s"])
    e1.end_frame()
    e2.end_frame()
    print("row %2d t=%.3f dt=%.4f ndet=%d lr=%s r=%s cmd1=(%7.1f,%6.1f) cmd2=(%7.1f,%6.1f)" % (
        i, p["t"], p["dt"], len(p["dets"]), p["lock_reason"][:20], p["reason"][:16],
        o1["dx"], o1["dy"], o2["dx"], o2["dy"]))
    print("   real uv=(%8.1f,%7.1f) va=%s fr=%s raw=(%8.1f,%7.1f) prev=%s" % (
        e1._unit_velocity[0], e1._unit_velocity[1], e1._unit_velocity_valid_axes,
        e1._unit_velocity_frames_axes, e1._unit_raw_velocity[0],
        e1._unit_raw_velocity[1],
        "None" if e1._prev_target[0] is None else "(%.1f,%.1f)" % e1._prev_target[0]))
    print("   v3   uv=(%8.1f,%7.1f) va=%s fr=%s st=%s prev=%s" % (
        e2._unit_velocity[0], e2._unit_velocity[1], e2._unit_velocity_valid_axes,
        e2._unit_velocity_frames_axes,
        [(round(a["v"], 1), a["valid"], a["frames"]) for a in e2.v3.axes],
        "None" if e2._prev_target[0] is None else "(%.1f,%.1f)" % e2._prev_target[0]))
