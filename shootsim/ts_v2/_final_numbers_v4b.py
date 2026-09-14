# -*- coding: utf-8 -*-
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v4")
abl = json.load(open(os.path.join(V4, "LOGIC_ABLATION_RESULTS.json"), encoding="utf-8"))
by = {c["name"]: c for c in abl["candidates"]}
base = by["baseline_alpha020"]
rob = by["robust_applied"]

names = ["baseline_alpha020", "robust_applied", "ROBUST+keep0.6_conf_blank0",
         "ROBUST+keep0.6_conf_blank1", "ROBUST+D_conf", "C_conf_rise3_ff1.0"]
keys = ["strafe_xlag_p75", "err_p75", "strafe_err_p75", "first_inner60_median",
        "first_inner60_n", "dwell_inner60", "stable_cmd_p95", "stable_samples",
        "stable_tick_diff_p95", "movement_samples", "flip_rate",
        "reversal_recovery_ms_median", "lead_dir_wrong_rate",
        "cap_saturation_rate"]
for name in names:
    c = by[name]
    print("==", name)
    for sp in ("train", "validation"):
        it = c["splits"][sp]
        d = {}
        for k in keys:
            v = it.get(k)
            d[k] = round(v["mean"], 4) if isinstance(v, dict) and v["mean"] is not None else (
                round(v, 4) if isinstance(v, (int, float)) else None)
        print("  ", sp, d)
        print("   gates:", c["gates_" + sp])

print()
print("== entry metrics corrected (phase 0)")
for name in names[:2]:
    c = by[name]
    for sp in ("train", "validation"):
        ph = c["splits"][sp]["per_phase"][4]
        row = {("to@" + T): (ph.get("timeout@" + T), ph.get("eligible@" + T),
                             ph.get("censored@" + T),
                             ph.get("entry_time_anomalies@" + T))
               for T in ("0.3", "0.5", "0.8")}
        print("  %-9s %-18s in60med=%s in60n=%s started@0.5=%s %s" % (
            sp, name, ph.get("first_inner60_median"), ph.get("first_inner60_n"),
            ph.get("started_inside@0.5"), row))

print()
print("== xlag per-phase better count (robust vs baseline)")
for sp in ("train", "validation"):
    bx = base["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
    rx = rob["splits"][sp]["strafe_xlag_p75"]["per_phase_values"]
    print("  ", sp, "better %d/9" % sum(1 for a, b in zip(rx, bx) if a < b),
          "baseline", [round(x, 2) for x in bx], "robust", [round(x, 2) for x in rx])

print()
print("== robust val stable detail (Q5)")
it = rob["splits"]["validation"]
bm = base["splits"]["validation"]
print("  robust stable_p95 mean=%.2f max=%.2f min=%.2f per=%s" % (
    it["stable_cmd_p95"]["mean"], it["stable_cmd_p95"]["max"],
    it["stable_cmd_p95"]["min"],
    [round(x, 1) for x in it["stable_cmd_p95"]["per_phase_values"]]))
print("  base   stable_p95 mean=%.2f max=%.2f min=%.2f per=%s" % (
    bm["stable_cmd_p95"]["mean"], bm["stable_cmd_p95"]["max"],
    bm["stable_cmd_p95"]["min"],
    [round(x, 1) for x in bm["stable_cmd_p95"]["per_phase_values"]]))
print("  robust stable_samples mean=%.1f base=%.1f" % (
    it["stable_samples"]["mean"], bm["stable_samples"]["mean"]))
print("  robust stable_tick_diff_p95=%.2f base=%.2f" % (
    it["stable_tick_diff_p95"]["mean"], bm["stable_tick_diff_p95"]["mean"]))

print()
print("== dwell detail blank0 (train) vs baseline")
it = by["ROBUST+keep0.6_conf_blank0"]["splits"]["train"]
print("  blank0 train dwell mean=%.4f min=%.4f ; base mean=%.4f min=%.4f" % (
    it["dwell_inner60"]["mean"], it["dwell_inner60"]["min"],
    base["splits"]["train"]["dwell_inner60"]["mean"],
    base["splits"]["train"]["dwell_inner60"]["min"]))
