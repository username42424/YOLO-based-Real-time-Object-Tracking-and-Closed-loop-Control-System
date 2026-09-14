# -*- coding: utf-8 -*-
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "tracking_logic_v3")
d = json.load(open(os.path.join(OUT, "LOGIC_ABLATION_RESULTS.json"), encoding="utf-8"))
base = d["candidates"][0]
for c in d["candidates"][:6] + d["candidates"][9:]:
    for sp in ("train", "validation"):
        integ = c["splits"][sp]
        t = integ["timeout_rate@0.5"]
        print("%-26s %-11s elig=%2d to_mean=%.3f to_min=%.3f to_max=%.3f "
              "per_phase=%s" % (
                  c["name"], sp, integ["per_phase"][0]["eligible@0.5"],
                  t["mean"], t["min"],
                  t["max"], [round(v, 2) for v in t["per_phase_values"]]))
    print()
