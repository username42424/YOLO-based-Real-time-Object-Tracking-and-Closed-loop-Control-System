# -*- coding: utf-8 -*-
"""Print stage results table: name, key metrics (train), score, fails."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "tracking_sweep_20260906_v2")
stage = sys.argv[1] if len(sys.argv) > 1 else "a"
path = os.path.join(OUT, f"stage_{stage}_results.jsonl")
rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

base = None
with open(os.path.join(OUT, "baseline_metrics.json"), encoding="utf-8") as f:
    base = json.load(f)["splits"]["train"]

keys = ("strafe_xlag_p75", "err_p75", "strafe_err_p75", "first_inner60_median",
        "dwell_inner60", "quiet_cmd_p95", "no_entry_rate",
        "strafe_dir_wrong_rate", "pred_active_rate", "cap_saturation_rate")
print("%-28s %8s %7s %7s %7s %7s %8s %7s %7s %7s %7s %6s %s" % (
    "name", "score", "xlag75", "err75", "serr75", "in60med", "dwell",
    "quiet95", "noent", "dirw", "predact", "pass", "fails"))
for r in rows:
    t = r["splits"]["train"]
    print("%-28s %8.3f %7.2f %7.2f %7.2f %7.3f %8.3f %7.1f %7.3f %7.3f %7.3f %6s %s" % (
        r["name"][:28], r["score"],
        t["strafe_xlag_p75"], t["err_p75"], t["strafe_err_p75"],
        t["first_inner60_median"] if t["first_inner60_median"] else -1,
        t["dwell_inner60"], t["quiet_cmd_p95"], t["no_entry_rate"],
        t["strafe_dir_wrong_rate"], t["pred_active_rate"],
        r["gate_pass"], ",".join(r["fails"])))
print("%-28s %8s %7.2f %7.2f %7.2f %7.3f %8.3f %7.1f %7.3f %7.3f %7.3f" % (
    "BASELINE", "-", base["strafe_xlag_p75"], base["err_p75"],
    base["strafe_err_p75"],
    base["first_inner60_median"] if base["first_inner60_median"] else -1,
    base["dwell_inner60"], base["quiet_cmd_p95"], base["no_entry_rate"],
    base["strafe_dir_wrong_rate"], base["pred_active_rate"]))
