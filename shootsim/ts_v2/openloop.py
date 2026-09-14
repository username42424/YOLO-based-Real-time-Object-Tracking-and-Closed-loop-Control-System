# -*- coding: utf-8 -*-
"""Decomposed fidelity tests:

1. open-loop engine replication — feed the real MainEngine with the log's own
   detections/timestamps; its outputs must reproduce the log fields exactly.
   Tests: filter chain, velocity model, prediction, control, missing logic.
2. integer allocator test — replay the log's chunk stream through
   MotionArbiter.quantize_components; ints must match the log's 瞄准整数.
3. tick/freeze race rate — fraction of capture boundaries coinciding with the
   15ms output grid (the irreducible attribution noise in closed loop).
"""
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from logparse import load_log  # noqa: E402
from harness import load_production_cfg, percentile  # noqa: E402
from dataset import fit_grid_phase  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
from trackers import _load_aim_main  # noqa: E402


def open_loop_test(log_name, main_cfg):
    mod = _load_aim_main()
    MainEngine = mod.MainEngine
    path = os.path.join(ROOT, "logs", log_name)
    data = load_log(path)
    engine = MainEngine(main_cfg)
    diffs = {"target": [], "err": [], "vel": [], "vel_f": [], "cmd": [],
             "lead": [], "hor": [], "reason": 0, "lock": 0, "total": 0}
    for ep in data["episodes"]:
        engine.reset()
        for p in ep["points"]:
            aim = p["net"]
            if aim != (0, 0):
                engine.notify_net(*aim)
            dets = [{"cls": d["cls"], "conf": d["conf"], "bbox": list(d["bbox"])}
                    for d in p["dets"]]
            out = engine.compute(dets, p["cx_ref"], max(1e-4, p["dt"]),
                                 processing_latency_s=p["lat_s"])
            engine.end_frame()
            diffs["total"] += 1
            if p["target"] is not None and out["target"] is not None:
                diffs["target"].append(math.hypot(out["target"][0] - p["target"][0],
                                                  out["target"][1] - p["target"][1]))
            elif (p["target"] is None) != (out["target"] is None):
                diffs["reason"] += 1
            if str(out["prediction_reason"]) != str(p["reason"]):
                diffs["reason"] += 1
            if str(engine.diag_reason) != str(p["lock_reason"]):
                diffs["lock"] += 1
            oe = p["obs_err"]
            diffs["err"].append(math.hypot(out["observed_error"][0] - oe[0],
                                           out["observed_error"][1] - oe[1]))
            vr = p["vel_raw"]
            diffs["vel"].append(math.hypot(out["velocity_raw"][0] - vr[0],
                                           out["velocity_raw"][1] - vr[1]))
            vf = p["vel_f"]
            diffs["vel_f"].append(math.hypot(out["velocity_filtered"][0] - vf[0],
                                             out["velocity_filtered"][1] - vf[1]))
            ld = p["lead"]
            diffs["lead"].append(math.hypot(out["lead"][0] - ld[0],
                                            out["lead"][1] - ld[1]))
            cmd = p["cmd"]
            diffs["cmd"].append(math.hypot(out["dx"] - cmd[0], out["dy"] - cmd[1]))
            diffs["hor"].append(abs(out["prediction_horizon_ms"] - p["horizon_ms"]))
    n = max(1, diffs["total"])
    return {
        "rows": diffs["total"],
        "target_mismatch": diffs["reason"] / n,
        "lock_reason_mismatch": diffs["lock"] / n,
        "pred_reason_mismatch": diffs["reason"] / n,
        "err_max_diff": max(diffs["err"]) if diffs["err"] else None,
        "cmd_p50_diff": percentile(diffs["cmd"], .5),
        "cmd_p95_diff": percentile(diffs["cmd"], .95),
        "cmd_max_diff": max(diffs["cmd"]) if diffs["cmd"] else None,
        "vel_max_diff": max(diffs["vel"]) if diffs["vel"] else None,
        "velf_max_diff": max(diffs["vel_f"]) if diffs["vel_f"] else None,
        "lead_max_diff": max(diffs["lead"]) if diffs["lead"] else None,
        "horizon_max_diff": max(diffs["hor"]) if diffs["hor"] else None,
    }


def allocator_test(log_name):
    from logparse import parse_mouse_lines
    mod = _load_aim_main()
    arb = mod.MotionArbiter()
    ticks = parse_mouse_lines(os.path.join(ROOT, "logs", log_name))
    match = total = 0
    int_diffs = []
    for t in ticks:
        chunk = t["aim_chunk"]
        got_x, got_y = arb.quantize_components(chunk[0], chunk[1], 0.0, 0.0)
        exp = t["aim_int"]
        total += 1
        d = max(abs(got_x - exp[0]), abs(got_y - exp[1]))
        int_diffs.append(d)
        if d == 0:
            match += 1
    return {
        "ticks": total,
        "int_exact_match_rate": match / max(1, total),
        "int_diff_p50": percentile(int_diffs, .5),
        "int_diff_p95": percentile(int_diffs, .95),
    }


def race_rate(log_name, period_s=0.015):
    path = os.path.join(ROOT, "logs", log_name)
    phase, n_ticks, conc = fit_grid_phase(path, period_s)
    data = load_log(path)
    gaps = []
    for ep in data["episodes"]:
        t_abs0 = None
        for p in ep["points"]:
            # absolute capture time = t_rel + ep start; use cumsum of dt instead
            pass
    # reconstruct absolute capture times from dt chain per episode is not
    # possible without the absolute base; instead use capture_t directly via
    # a second parse
    from logparse import parse_rows
    rows = parse_rows(path)
    ts = [float(r["capture_t"]) for r in rows]
    # distance from each capture to the nearest grid point
    near = 0
    win = 0.003
    for t in ts:
        ph = (t / period_s) % 1.0
        d = min(ph, 1.0 - ph) * period_s
        gaps.append(d)
        if d <= win:
            near += 1
    return {
        "grid_phase_s": phase, "ticks_used": n_ticks, "phase_concentration": conc,
        "captures": len(ts),
        "capture_within_3ms_of_tick_rate": near / max(1, len(ts)),
        "capture_tick_dist_p50_ms": 1000 * statistics.median(gaps),
    }


def main():
    prod = load_production_cfg()
    logs = ["aim_20260906_003353.log", "aim_20260906_001842.log",
            "aim_20260906_000134.log"]
    out = {}
    for name in logs:
        ol = open_loop_test(name, prod)
        al = allocator_test(name)
        rr = race_rate(name)
        out[name] = {"open_loop": ol, "allocator": al, "race": rr}
        print(json.dumps({"log": name, **ol, **al, **rr}, ensure_ascii=False))
    out_dir = os.path.join(os.path.dirname(HERE), "results",
                           "tracking_sweep_20260906_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "openloop_allocator_tests.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
