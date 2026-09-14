# -*- coding: utf-8 -*-
"""Cadence comparison: 30ms logs vs 40ms robust logs vs alpha=0.2 reference.
All live-observed metrics from the logs themselves (corrected v4 definitions),
plus cadence diagnostics (dt distribution, horizon floor rate)."""
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))

from logparse import load_log, fill_world  # noqa: E402
from eval3 import percentile, STRAFE_VX  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
GROUPS = {
    "cadence30_robust": ["aim_20260906_135713.log", "aim_20260906_140550.log",
                         "aim_20260906_141313.log"],
    "cadence40_robust": ["aim_20260906_125713.log", "aim_20260906_130338.log",
                         "aim_20260906_131131.log"],
    "alpha020_reference": ["aim_20260906_003353.log"],
}


def pooled(points_lists):
    errs, serrs, slags, speeds, dts, hors = [], [], [], [], [], []
    floor = 0
    floor_total = 0
    locked = 0
    for pts in points_lists:
        prev_t = None
        for p in pts:
            if prev_t is not None:
                dts.append(p["t"] - prev_t)
            prev_t = p["t"]
            if not (p["lock_reason"] and str(p["lock_reason"]).startswith("locked")):
                continue
            if p["bcx"] is None:
                continue
            locked += 1
            vx, vy = p["vx"] or 0.0, p["vy"] or 0.0
            sp = math.hypot(vx, vy)
            speeds.append(sp)
            err = math.hypot(*p["obs_err"])
            if sp > 100.0:
                errs.append(err)
            if abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy) and sp > 1.0:
                serrs.append(err)
                slags.append((p["obs_err"][0] * vx + p["obs_err"][1] * vy) / sp)
            if p["horizon_ms"] > 0:
                hors.append(p["horizon_ms"])
                floor_total += 1
                if p["horizon_ms"] <= 30.5:
                    floor += 1
    return {
        "episodes": len(points_lists),
        "locked_obs": locked,
        "movement_samples": len(errs),
        "err_p50": percentile(errs, .5), "err_p75": percentile(errs, .75),
        "strafe_samples": len(serrs),
        "strafe_err_p75": percentile(serrs, .75),
        "strafe_xlag_p50": percentile(slags, .5),
        "strafe_xlag_p75": percentile(slags, .75),
        "speed_p90": percentile(speeds, .9),
        "dt_p50_ms": percentile(dts, .5) * 1000 if dts else None,
        "dt_p95_ms": percentile(dts, .95) * 1000 if dts else None,
        "dt_max_ms": max(dts) * 1000 if dts else None,
        "dt_gt45ms_rate": (sum(1 for d in dts if d > 0.045) / len(dts)) if dts else None,
        "horizon_p50_ms": percentile(hors, .5) if hors else None,
        "horizon_floor_rate": floor / floor_total if floor_total else None,
    }


def main():
    out = {}
    for gname, logs in GROUPS.items():
        pts_lists = []
        for name in logs:
            path = os.path.join(LOG_DIR, name)
            data = load_log(path, 0.44, 0.51)
            for ep in data["episodes"]:
                fill_world(ep)
                pts = ep["points"]
                if any(p["detected"] for p in pts) and \
                        sum(1 for p in pts if p["box"] is not None) >= 3:
                    pts_lists.append(pts)
        out[gname] = pooled(pts_lists)
        out[gname]["logs"] = logs

    os.makedirs(os.path.join(ROOT, "日志V4", "cadence"), exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    for base in (os.path.join(ROOT, "日志V4", "cadence"),
                 os.path.join(os.path.dirname(HERE), "results",
                              "tracking_logic_v4", "cadence")):
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "cadence_comparison.json"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(text)

    order = ["cadence40_robust", "cadence30_robust", "alpha020_reference"]
    print("%-20s %8s %8s %9s %9s %7s %7s %8s %7s" % (
        "组", "xlag75", "err75", "serr75", "sp_p90", "dt_p50", "dt_p95",
        "hor_p50", "样本(strafe)"))
    for g in order:
        d = out[g]
        print("%-20s %8s %8s %9s %9.0f %7.1f %7.1f %8.1f %7d" % (
            g, fmt(d["strafe_xlag_p75"]), fmt(d["err_p75"]),
            fmt(d["strafe_err_p75"]), d["speed_p90"],
            d["dt_p50_ms"], d["dt_p95_ms"], d["horizon_p50_ms"],
            d["strafe_samples"]))
    print()
    print("dt>45ms 占比: " + ", ".join(
        "%s=%.1f%%" % (g, 100 * out[g]["dt_gt45ms_rate"]) for g in order))
    print("horizon 贴 30ms 下限比例: " + ", ".join(
        "%s=%.1f%%" % (g, 100 * out[g]["horizon_floor_rate"]) for g in order))
    a, b = out["cadence40_robust"], out["cadence30_robust"]
    print()
    print("30ms vs 40ms (robust): xlag %s -> %s (%.1f%%), err %s -> %s (%.1f%%)" % (
        fmt(b["strafe_xlag_p75"]), fmt(a["strafe_xlag_p75"]),
        100 * (a["strafe_xlag_p75"] / b["strafe_xlag_p75"] - 1),
        fmt(b["err_p75"]), fmt(a["err_p75"]),
        100 * (a["err_p75"] / b["err_p75"] - 1)))


def fmt(v, nd=2):
    return "null" if v is None else ("%%.%df" % nd) % v


if __name__ == "__main__":
    main()
