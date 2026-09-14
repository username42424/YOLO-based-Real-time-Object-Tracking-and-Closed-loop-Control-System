# -*- coding: utf-8 -*-
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from logparse import load_log
from harness import load_production_cfg
from trackers import _load_aim_main


def reason_mismatch_patterns(name):
    mod = _load_aim_main()
    engine = mod.MainEngine(load_production_cfg())
    path = os.path.join(ROOT, "logs", name)
    data = load_log(path)
    c = Counter()
    cmd_diffs = []
    for ep in data["episodes"]:
        engine.reset()
        for p in ep["points"]:
            if p["net"] != (0, 0):
                engine.notify_net(*p["net"])
            dets = [{"cls": d["cls"], "conf": d["conf"], "bbox": list(d["bbox"])}
                    for d in p["dets"]]
            out = engine.compute(dets, p["cx_ref"], max(1e-4, p["dt"]),
                                 processing_latency_s=p["lat_s"])
            engine.end_frame()
            lr, sr = str(p["reason"]), str(out["prediction_reason"])
            if lr != sr:
                c[(lr, sr)] += 1
                cmd_diffs.append(math.hypot(out["dx"] - p["cmd"][0],
                                            out["dy"] - p["cmd"][1]))
    return c, cmd_diffs


for name in ("aim_20260906_003353.log", "aim_20260906_001842.log",
             "aim_20260906_000134.log"):
    c, cd = reason_mismatch_patterns(name)
    print(name, "mismatch patterns:", dict(c))
    if cd:
        cd.sort()
        print("   cmd diff on mismatched rows: p50=%.2f p95=%.2f max=%.2f" % (
            cd[len(cd) // 2], cd[int(0.95 * (len(cd) - 1))], cd[-1]))
