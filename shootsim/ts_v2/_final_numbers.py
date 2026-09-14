# -*- coding: utf-8 -*-
import json
import os

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "tracking_logic_v3", "LOGIC_ABLATION_RESULTS.json")
d = json.load(open(p, encoding="utf-8"))
by = {c["name"]: c for c in d["candidates"]}
for name in ("baseline_alpha020", "robust_applied", "D_keep0.6_h0.5_conf",
             "ROBUST+keep0.6_conf"):
    c = by[name]
    print(name)
    for sp in ("train", "validation"):
        it = c["splits"][sp]
        rev = it["reversal_recovery_ms_median"]["mean"] if it["reversal_recovery_ms_median"] else -1
        print("  %s: in60=%.3f rev=%.0fms sp95=%.1f err=%.2f xlag=%.2f dwell=%.3f" % (
            sp, it["first_inner60_median"]["mean"], rev,
            it["stable_cmd_p95"]["mean"], it["err_p75"]["mean"],
            it["strafe_xlag_p75"]["mean"], it["dwell_inner60"]["mean"]))
