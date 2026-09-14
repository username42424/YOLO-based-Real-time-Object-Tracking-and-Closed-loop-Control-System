# -*- coding: utf-8 -*-
import os
d = r"C:\Users\12951\Desktop\12323\shootsim\results\tracking_sweep_20260906_v2"
for f in ("stage_a_results.jsonl", "state.json", "baseline_metrics.json"):
    p = os.path.join(d, f)
    if os.path.exists(p):
        os.remove(p)
        print("removed", f)
