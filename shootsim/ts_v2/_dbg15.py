# -*- coding: utf-8 -*-
import copy
import json
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
for i, p in enumerate(group["points"]):
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
    fields = ("can_send", "dx", "dy", "lead", "prediction_reason",
              "observed_error", "velocity_raw", "velocity_filtered")
    mismatch = [f for f in fields if o1[f] != o2[f]]
    if mismatch or i < 3:
        print(f"--- row {i} t={p['t']:.3f} dt={p['dt']:.4f} "
              f"reason={p['lock_reason']} reason2={p['reason']}")
        for f in mismatch:
            print(f"    {f}: real={o1[f]}  v3={o2[f]}")
        print(f"    real: uv={e1._unit_velocity} va={e1._unit_velocity_valid_axes} "
              f"fr={e1._unit_velocity_frames_axes} raw={e1._unit_raw_velocity}")
        print(f"    v3  : uv={e2._unit_velocity} va={e2._unit_velocity_valid_axes} "
              f"fr={e2._unit_velocity_frames_axes} raw={e2._unit_raw_velocity} "
              f"st={[(a['v'], a['valid'], a['frames']) for a in e2.v3.axes]}")
        if mismatch and i > 3:
            break
