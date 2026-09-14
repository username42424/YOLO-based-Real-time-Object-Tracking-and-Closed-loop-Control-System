# -*- coding: utf-8 -*-
"""Extract all numbers needed for the V4 final report from the raw JSONs."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v4")
abl = json.load(open(os.path.join(V4, "LOGIC_ABLATION_RESULTS.json"), encoding="utf-8"))
by = {c["name"]: c for c in abl["candidates"]}
base = by["baseline_alpha020"]
rob = by["robust_applied"]

KEYS = ("strafe_xlag_p75", "err_p75", "strafe_err_p75", "first_inner60_median",
        "first_inner60_p75", "first_inner60_n", "dwell_inner60",
        "stable_cmd_p95", "stable_samples", "movement_samples",
        "reversal_recovery_ms_median", "lead_dir_wrong_rate",
        "cap_saturation_rate", "flip_rate", "micro_jitter_rms_median")

print("== baseline vs robust (9-phase mean, per-phase for xlag)")
for sp in ("train", "validation"):
    for name, c in (("baseline", base), ("robust", rob)):
        it = c["splits"][sp]
        vals = {k: it[k]["mean"] if it.get(k) else None for k in KEYS}
        print(" %-9s %-8s" % (sp, name), {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in vals.items()})
    bx = base["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
    rx = rob["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
    better = sum(1 for a, b in zip(rx, bx) if a < b)
    print("  xlag 9-phase baseline:", [round(x, 2) for x in bx])
    print("  xlag 9-phase robust  :", [round(x, 2) for x in rx], "better %d/9" % better)

print()
print("== entry metrics corrected (phase 0 = exact fit)")
for name in ("baseline_alpha020", "robust_applied"):
    c = by[name]
    for sp in ("train", "validation"):
        ph = c["splits"][sp]["per_phase"][PHASE_OFFSETS_MS.index(0) if False else 4]
        row = {}
        for T in ("0.3", "0.5", "0.8"):
            row["to@" + T] = (ph.get("timeout@" + T), ph.get("eligible@" + T),
                              ph.get("censored@" + T),
                              ph.get("entry_time_anomalies@" + T))
        print(" %-9s %-18s in60med=%s n=%s started_in@0.5=%s %s" % (
            sp, name, ph.get("first_inner60_median"),
            ph.get("first_inner60_n"), ph.get("started_inside@0.5"), row))

print()
print("== dwell detail for lag winners (train/val mean, worst)")
for name in ("ROBUST+keep0.6_conf_blank0", "ROBUST+keep0.6_conf_blank1",
             "ROBUST+D_conf", "ROBUST+C_conf_only", "C_conf_rise3_ff1.0",
             "C_conf_ff1.1_only"):
    c = by[name]
    for sp in ("train", "validation"):
        it = c["splits"][sp]
        print(" %-30s %-11s dwell=%.4f(w%.4f) xlag=%.2f err=%.2f sp95=%.1f gates=%s" % (
            name, sp, it["dwell_inner60"]["mean"], it["dwell_inner60"]["min"],
            it["strafe_xlag_p75"]["mean"], it["err_p75"]["mean"],
            it["stable_cmd_p95"]["mean"], c["gates_" + sp]))

print()
print("== blank0 vs blank1 field diff count")
import itertools
b0 = by["ROBUST+keep0.6_conf_blank0"]
b1 = by["ROBUST+keep0.6_conf_blank1"]
diffs = 0
checked = 0
for sp in ("train", "validation"):
    for ph in range(9):
        for k, v in b0["splits"][sp]["per_phase"][ph].items():
            if k == "episodes":
                continue
            checked += 1
            if json.dumps(v, sort_keys=True) != json.dumps(
                    b1["splits"][sp]["per_phase"][ph].get(k), sort_keys=True):
                diffs += 1
print("per-phase fields differing:", diffs, "of", checked)
print("blank0 xlag mean: t=%.2f v=%.2f ; blank1: t=%.2f v=%.2f" % (
    b0["splits"]["train"]["strafe_xlag_p75"]["mean"],
    b0["splits"]["validation"]["strafe_xlag_p75"]["mean"],
    b1["splits"]["train"]["strafe_xlag_p75"]["mean"],
    b1["splits"]["validation"]["strafe_xlag_p75"]["mean"]))
