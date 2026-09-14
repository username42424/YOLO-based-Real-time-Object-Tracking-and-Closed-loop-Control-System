# -*- coding: utf-8 -*-
"""Cadence sweep: auto-detect the recognition cadence from log headers and
compute corrected tracking metrics + loop-health indicators per cadence.

Usage:
    python eval_cadence_sweep.py            # scan logs/ for new logs
    python eval_cadence_sweep.py f1 f2 ...  # explicit log names

Groups logs by detected cadence (识别节拍 from the header) and writes
日志V4/cadence/sweep_results.json + sweep table to stdout.
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))

from logparse import load_log, fill_world  # noqa: E402
from eval3 import percentile, STRAFE_VX  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
KNOWN = {"aim_20260906_003353.log", "aim_20260906_125713.log",
         "aim_20260906_130338.log", "aim_20260906_131131.log",
         "aim_20260906_135713.log", "aim_20260906_140550.log",
         "aim_20260906_141313.log"}
CADENCE_RE = re.compile(r"YOLO识别节拍=(\d+)ms")
MOVE_SPEED_MIN = 100.0
SPEED_BANDS = ((0, 600), (600, 1100), (1100, 3000), (3000, 10**9))


def detect_cadence(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        if "[配置]" in line or "识别节拍" in line:
            m = CADENCE_RE.search(line)
            if m:
                return int(m.group(1))
        if "[瞄准观测]" in line or "▶" in line:
            break
    return None


def speed_band(sp):
    for lo, hi in SPEED_BANDS:
        if lo <= sp < hi:
            return "%d-%d" % (lo, hi) if hi < 10**9 else "%d+" % lo
    return "unknown"


def analyze_log(path):
    data = load_log(path, 0.44, 0.51)
    if data.get("integrity", {}).get("polluted"):
        return []
    eps = []
    for ep in data["episodes"]:
        fill_world(ep)
        pts = ep["points"]
        if any(p["detected"] for p in pts) and \
                sum(1 for p in pts if p["box"] is not None) >= 3:
            eps.append(pts)
    return eps


def group_metrics(points_lists):
    errs, raw_errs, serrs, slags, speeds, dts, lats, hors = [], [], [], [], [], [], [], []
    band_lag = {speed_band(lo): [] for lo, _ in SPEED_BANDS}
    band_err = {speed_band(lo): [] for lo, _ in SPEED_BANDS}
    locked = 0
    locked_time = 0.0
    observed_time = 0.0
    lock_runs = []
    lock_run = 0.0
    lock_losses = 0
    weighted_err_area = 0.0
    weighted_raw_err_area = 0.0
    weighted_error_time = 0.0
    for pts in points_lists:
        prev_t = None
        prev_locked = False
        for p in pts:
            if prev_t is not None:
                interval = max(0.0, p["t"] - prev_t)
                dts.append(interval)
                observed_time += interval
                is_locked = bool(p.get("lock_reason") and
                                 str(p["lock_reason"]).startswith("locked") and
                                 p.get("bcx") is not None)
                if is_locked:
                    locked_time += interval
                    lock_run += interval
                    if p.get("bcx") is not None:
                        weighted_err_area += math.hypot(*p["obs_err"]) * interval
                        if p.get("raw_error") is not None:
                            weighted_raw_err_area += math.hypot(
                                *p["raw_error"]) * interval
                        weighted_error_time += interval
                elif prev_locked and lock_run > 0.0:
                    lock_runs.append(lock_run)
                    lock_losses += 1
                    lock_run = 0.0
                prev_locked = is_locked
            prev_t = p["t"]
            lats.append(p["lat_s"] * 1000.0)
            if p["horizon_ms"] > 0:
                hors.append(p["horizon_ms"])
            if not (p["lock_reason"] and str(p["lock_reason"]).startswith("locked")):
                continue
            if p["bcx"] is None:
                continue
            locked += 1
            if not p.get("velocity_valid"):
                continue
            vx, vy = p["vx"] or 0.0, p["vy"] or 0.0
            sp = math.hypot(vx, vy)
            speeds.append(sp)
            err = math.hypot(*p["obs_err"])
            if p.get("raw_error") is not None:
                raw_errs.append(math.hypot(*p["raw_error"]))
            if sp > MOVE_SPEED_MIN:
                errs.append(err)
                band_err[speed_band(sp)].append(err)
            if abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy) and sp > 1.0:
                lag = (p["obs_err"][0] * vx + p["obs_err"][1] * vy) / sp
                serrs.append(err)
                slags.append(lag)
                band_lag[speed_band(sp)].append(lag)
        if lock_run > 0.0:
            lock_runs.append(lock_run)
            lock_run = 0.0
        prev_locked = False
    if observed_time > 0.0:
        obs_rate_hz = sum(max(0, len(pts) - 1) for pts in points_lists) / observed_time
    else:
        obs_rate_hz = None
    return {
        "episodes": len(points_lists),
        "locked_obs": locked,
        "movement_samples": len(errs),
        "raw_err_p75": percentile(raw_errs, .75),
        "err_p75": percentile(errs, .75),
        "strafe_samples": len(serrs),
        "strafe_err_p75": percentile(serrs, .75),
        "strafe_xlag_p75": percentile(slags, .75),
        "strafe_xlag_p50": percentile(slags, .5),
        "speed_p90": percentile(speeds, .9),
        "dt_p50_ms": percentile(dts, .5) * 1000 if dts else None,
        "dt_p95_ms": percentile(dts, .95) * 1000 if dts else None,
        "dt_max_ms": max(dts) * 1000 if dts else None,
        "observation_count": sum(len(pts) for pts in points_lists),
        "observed_duration_s": observed_time,
        "observation_rate_hz": obs_rate_hz,
        "locked_time_ratio": locked_time / observed_time if observed_time else None,
        "time_weighted_err_mean_px": (
            weighted_error_time and weighted_err_area / weighted_error_time),
        "time_weighted_raw_err_mean_px": (
            weighted_error_time and weighted_raw_err_area / weighted_error_time),
        "continuous_lock_p50_s": percentile(lock_runs, .5) if lock_runs else None,
        "lock_loss_count": lock_losses,
        "lat_p50_ms": percentile(lats, .5) if lats else None,
        "lat_p95_ms": percentile(lats, .95) if lats else None,
        "horizon_p50_ms": percentile(hors, .5) if hors else None,
        "xlag_by_speed_band": {
            k: {"p75": percentile(v, .75), "n": len(v)}
            for k, v in band_lag.items() if v},
        "err_by_speed_band": {
            k: {"p75": percentile(v, .75), "n": len(v)}
            for k, v in band_err.items() if v},
    }


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(set(os.listdir(LOG_DIR)) - KNOWN - {"recoil"})
        names = [n for n in names if n.endswith(".log")]
        if not names:
            print("no new logs found; pass log filenames as arguments")
            return
    groups = {}
    skipped = []
    for name in names:
        path = os.path.join(LOG_DIR, name)
        if not os.path.exists(path):
            skipped.append((name, "not found"))
            continue
        cad = detect_cadence(path)
        if cad is None:
            skipped.append((name, "cadence not found in header"))
            continue
        pts_lists = analyze_log(path)
        if pts_lists:
            groups.setdefault(cad, {"logs": [], "points": []})
            groups[cad]["logs"].append(name)
            groups[cad]["points"].extend(pts_lists)
        else:
            skipped.append((name, "no valid episodes"))
    results = {"detected_cadences": {}, "skipped": skipped}
    table = []
    for cad in sorted(groups):
        m = group_metrics(groups[cad]["points"])
        m["logs"] = groups[cad]["logs"]
        results["detected_cadences"][str(cad)] = m
        beat_hold = (m["dt_p95_ms"] / cad) if cad else None
        table.append((cad, m, beat_hold))

    print("%6s %4s %8s %8s %9s %7s %7s %7s %8s %7s" % (
        "节拍", "eps", "xlag75", "err75", "serr75", "sp_p90", "dt_p95",
        "lat_p95", "beat_hold", "strafeN"))
    for cad, m, hold in table:
        print("%6d %4d %8s %8s %9s %7.0f %7s %7s %7.2f %7d" % (
            cad, m["episodes"], fmt(m["strafe_xlag_p75"]), fmt(m["err_p75"]),
            fmt(m["strafe_err_p75"]), m["speed_p90"], fmt(m["dt_p95_ms"], 1),
            fmt(m["lat_p95_ms"], 1), hold if hold else 0, m["strafe_samples"]))
        for band, d in sorted(m["xlag_by_speed_band"].items()):
            be = m["err_by_speed_band"].get(band, {"p75": None})
            print("         %6s px/s: xlag75=%s (n=%d)  err75=%s" % (
                band, fmt(d["p75"]), d["n"], fmt(be["p75"])))
    if skipped:
        print("skipped:", skipped)

    os.makedirs(os.path.join(ROOT, "日志V4", "cadence"), exist_ok=True)
    text = json.dumps(results, ensure_ascii=False, indent=1)
    for b in (os.path.join(ROOT, "日志V4", "cadence"),
              os.path.join(os.path.dirname(HERE), "results",
                           "tracking_logic_v4", "cadence")):
        os.makedirs(b, exist_ok=True)
        with open(os.path.join(b, "sweep_results.json"), "w",
                  encoding="utf-8", newline="") as f:
            f.write(text)
    print("saved 日志V4/cadence/sweep_results.json")


def fmt(v, nd=2):
    return "null" if v is None else ("%%.%df" % nd) % v


if __name__ == "__main__":
    main()
