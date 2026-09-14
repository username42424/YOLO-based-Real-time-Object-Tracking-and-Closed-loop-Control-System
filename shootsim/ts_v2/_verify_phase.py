# -*- coding: utf-8 -*-
import json
import os

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "tracking_logic_v3", "LOGIC_ABLATION_RESULTS.json")
d = json.load(open(p, encoding="utf-8"))
by = {c["name"]: c for c in d["candidates"]}
base = by["baseline_alpha020"]
for name in ("robust_applied", "C_conf_rise3_ff1.0", "D_keep0.6_h0.5_conf",
             "ROBUST+keep0.6_conf", "ROBUST+D_conf"):
    c = by[name]
    for sp in ("train", "validation"):
        per = c["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
        bper = base["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
        better = sum(1 for a, b in zip(per, bper) if a < b)
        print("%-26s %-11s better %d/9  per=%s" % (
            name, sp, better, [round(x, 1) for x in per]))
    dw_tr = (c["splits"]["train"]["dwell_inner60"]["mean"]
             - base["splits"]["train"]["dwell_inner60"]["mean"])
    dw_va = (c["splits"]["validation"]["dwell_inner60"]["mean"]
             - base["splits"]["validation"]["dwell_inner60"]["mean"])
    print("   dwell delta train %+.4f val %+.4f" % (dw_tr, dw_va))
