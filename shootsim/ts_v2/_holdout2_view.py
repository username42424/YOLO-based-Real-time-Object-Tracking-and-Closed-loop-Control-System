# -*- coding: utf-8 -*-
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "tracking_sweep_20260906_v2")
d = json.load(open(os.path.join(OUT, "holdout_results2.json"), encoding="utf-8"))
for name, row in d["rows"].items():
    h = row["holdout"]
    print("==", name)
    print("  xlag75=%.2f err75=%.2f serr75=%.2f in60med=%s in60p75=%s "
          "dwell=%.3f quiet95=%.1f noent=%.3f cap_sat=%.4f" % (
              h["strafe_xlag_p75"], h["err_p75"], h["strafe_err_p75"],
              h["first_inner60_median"], h["first_inner60_p75"],
              h["dwell_inner60"], h["quiet_cmd_p95"], h["no_entry_rate"],
              h["cap_saturation_rate"]))
    sq = row["speed_quartiles"]
    for k in ("q1", "q2", "q3", "q4"):
        print("   %s n=%4d lag75=%7.2f err75=%6.2f" % (
            k, sq[k]["samples"], sq[k]["lag_p75"], sq[k]["err_p75"]))
    print("   reasons:", {k: v for k, v in sorted(h["reason_counts"].items())
                          if v and not k.startswith("missing")})
