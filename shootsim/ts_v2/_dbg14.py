# -*- coding: utf-8 -*-
import json
import math
import os
import sys

ROOT = r"C:\Users\12951\Desktop\12323"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))
sys.path.insert(0, os.path.join(ROOT, "shootsim", "ts_v2"))

from aim_engine import MainEngine
from tests.test_tracking_v3 import base_cfg, _MiniLoop, constant_track

cfg = base_cfg()
engine = MainEngine(cfg)
loop = _MiniLoop(cfg)
track = constant_track(200.0, 500.0, 40)
dts = [0.04] * 40
rows = loop.run(engine, track, dts)
for r in rows[:16]:
    print("t=%.2f screen_x=%7.1f cmd=(%8.1f,%8.1f) err=(%7.1f,%7.1f) lead=(%6.1f,%6.1f) "
          "vraw=(%8.0f,%8.0f) reason=%s" % (
              r["t"], r["screen_x"], r["cmd"][0], r["cmd"][1],
              r["err"][0], r["err"][1], r["lead"][0], r["lead"][1],
              r["vel_raw"][0], r["vel_raw"][1], r["reason"]))
