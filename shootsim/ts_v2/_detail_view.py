# -*- coding: utf-8 -*-
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "tracking_logic_v3")
d = json.load(open(os.path.join(OUT, "LOGIC_ABLATION_RESULTS.json"), encoding="utf-8"))
want = {"baseline_alpha020", "robust_applied", "C_conf_rise3_ff1.0",
        "D_keep0.6_h0.5_conf", "D_keep0.6_h0.5_conf_rise2",
        "ROBUST+D_conf", "ROBUST+C_conf_only"}
keys = ("strafe_xlag_p75", "err_p75", "strafe_err_p75", "first_inner60_median",
        "first_inner60_p75", "dwell_inner60", "stable_cmd_p95",
        "timeout_rate@0.5", "reversal_recovery_ms_median",
        "lead_dir_wrong_rate", "cap_saturation_rate")
for c in d["candidates"]:
    if c["name"] not in want:
        continue
    print("==", c["name"], c.get("kind"))
    for sp in ("train", "validation"):
        integ = c["splits"][sp]
        vals = []
        for k in keys:
            v = integ.get(k)
            if v is None:
                vals.append("%s=null" % k)
            else:
                vals.append("%s=%.3f(w%.3f)" % (k, v["mean"], v["max"] if k != "dwell_inner60" else v["min"]))
        print(" ", sp, " | ".join(vals))
