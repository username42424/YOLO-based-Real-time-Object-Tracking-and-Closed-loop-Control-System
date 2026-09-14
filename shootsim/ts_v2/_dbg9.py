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
n_ff = 0
mismatch_ff = mismatch_noff = tot_ff = tot_noff = 0
for ep in data["episodes"]:
    engine.reset()
    for p in ep["points"]:
        ff = False
        # recoil_feedforward flag is on the raw row; recover from cmd/lead?
        # quick recount: use obs rows via parse (needs flag) — approximate here
        if p["net"] != (0, 0):
            engine.notify_net(*p["net"])
        dets = [{"cls": d["cls"], "conf": d["conf"], "bbox": list(d["bbox"])}
                for d in p["dets"]]
        out = engine.compute(dets, p["cx_ref"], max(1e-4, p["dt"]),
                             processing_latency_s=p["lat_s"])
        engine.end_frame()
        mm = str(out["prediction_reason"]) != str(p["reason"])
        tot_noff += 1
        mismatch_noff += mm
print("rows:", tot_noff, "reason mismatches:", mismatch_noff,
      "rate: %.4f" % (mismatch_noff / tot_noff))

# count recoil_feedforward rows in raw log
import io
cnt = 0
for line in io.open(path, encoding="utf-8", errors="replace"):
    if '"recoil_feedforward":true' in line:
        cnt += 1
print("rows with recoil_feedforward=true:", cnt)
