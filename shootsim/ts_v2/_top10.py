# -*- coding: utf-8 -*-
"""Assemble Top-10 candidate list across stages (gate-passing, unique)."""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "tracking_sweep_20260906_v2")
rows = []
for stage in ("a", "b", "c", "d", "e"):
    p = os.path.join(OUT, f"stage_{stage}_results.jsonl")
    if not os.path.exists(p):
        continue
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        r["stage"] = stage
        rows.append(r)
pool = [r for r in rows if r["gate_pass"]]
pool.sort(key=lambda r: -r["score"])
seen, top = set(), []
for r in pool:
    key = json.dumps(r["values"], sort_keys=True)
    if key in seen:
        continue
    seen.add(key)
    top.append(r)
    if len(top) >= 10:
        break
with open(os.path.join(OUT, "top10_candidates.json"), "w", encoding="utf-8") as f:
    json.dump([{"stage": r["stage"], "name": r["name"], "score": round(r["score"], 3),
                "values": r["values"],
                "train": {k: r["splits"]["train"][k] for k in (
                    "strafe_xlag_p75", "err_p75", "strafe_err_p75",
                    "first_inner60_median", "dwell_inner60", "quiet_cmd_p95")}}
               for r in top], f, ensure_ascii=False, indent=1)
for r in top:
    t = r["splits"]["train"]
    print("%-10s %-10s score=%6.3f xlag=%5.2f err=%5.2f serr=%5.2f dwell=%.3f "
          "quiet=%6.1f" % (r["stage"], r["name"], r["score"],
                           t["strafe_xlag_p75"], t["err_p75"],
                           t["strafe_err_p75"], t["dwell_inner60"],
                           t["quiet_cmd_p95"]))
