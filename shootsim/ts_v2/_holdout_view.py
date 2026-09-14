# -*- coding: utf-8 -*-
"""Holdout results table + parameter boundary/stability analysis."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "tracking_sweep_20260906_v2")

with open(os.path.join(OUT, "holdout_results.json"), encoding="utf-8") as f:
    data = json.load(f)
rows = data["rows"]
base = next(r for r in rows if r["name"] == "production_baseline")["holdout"]

keys = ("strafe_xlag_p75", "err_p75", "strafe_err_p75", "first_inner60_median",
        "first_inner60_p75", "dwell_inner60", "quiet_cmd_p95", "no_entry_rate",
        "strafe_dir_wrong_rate", "cap_saturation_rate", "jump_count_total")
print("%-24s " % "name" + " ".join("%8s" % k[:8] for k in keys))
print("%-24s " % "BASELINE(holdout)" + " ".join(
    "%8.2f" % (base[k] if base[k] is not None else -1) for k in keys))
for r in rows:
    h = r["holdout"]
    if r["name"] == "production_baseline":
        continue
    rel = []
    for k in keys:
        v, b = h.get(k), base.get(k)
        if v is None or b in (None, 0):
            rel.append("%8.2f" % (v if v is not None else -1))
        else:
            rel.append("%7.1f%%" % (100 * (v - b) / abs(b)))
    print("%-24s " % r["name"][:24] + " ".join(rel))
print()
for r in rows:
    if r["name"] == "production_baseline":
        continue
    print(r["name"], "stage=", r["stage"])
    print("   ", json.dumps(r["values"], ensure_ascii=False, sort_keys=True))
