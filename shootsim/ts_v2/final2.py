# -*- coding: utf-8 -*-
"""Final candidate definition, stability check, speed-quartile tables and the
single final holdout evaluation.  Writes recommended_configs.json,
holdout_results2.json and refreshes pareto_candidates.json."""
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import OUT_DIR, prepare_dataset, improvement_score, hard_gates  # noqa: E402
from harness import (load_production_cfg, patch_cfg, evaluate_set, run_episode,
                     episode_metrics, percentile, build_sim_cfg)  # noqa: E402

SEED = 20260906

BASE = {}  # production values stay as-is


def V(**kw):
    return {f"aim_control.{k}" if k in (
        "lock_box_filter_mode", "lock_box_smoothing_alpha", "prediction_min_ms",
        "prediction_max_ms", "prediction_box_ratio", "prediction_cap_px",
        "prediction_lock_frames", "prediction_vel_tau", "reversal_damp",
        "unit_max_counts", "missing_decay", "missing_max_counts") else
        f"unit.{k}": v for k, v in kw.items()}


ROBUST = {
    "aim_control.lock_box_filter_mode": "fixed",
    "aim_control.lock_box_smoothing_alpha": 0.18,
    "aim_control.prediction_min_ms": 30.0,
    "aim_control.prediction_max_ms": 105.0,
    "aim_control.prediction_box_ratio": 0.65,
    "aim_control.prediction_vel_tau": 0.06,
    "aim_control.reversal_damp": 0.45,
    "unit.far_gain": 1.05,
    "unit.near_gain": 0.55,
    "unit.gain_softness_px_x": 8.0,
}
BALANCED = {
    "aim_control.lock_box_filter_mode": "fixed",
    "aim_control.lock_box_smoothing_alpha": 0.1509,
    "aim_control.prediction_min_ms": 44.56,
    "aim_control.prediction_max_ms": 105.0,
    "aim_control.prediction_box_ratio": 0.6966,
    "aim_control.prediction_cap_px": 28.8,
    "aim_control.prediction_vel_tau": 0.0759,
    "aim_control.reversal_damp": 0.355,
    "aim_control.unit_max_counts": 254.9,
    "unit.far_gain": 1.089,
    "unit.near_gain": 0.6715,
    "unit.near_radius_px": 16.62,
    "unit.gain_softness_px_x": 17.79,
    "unit.stop_deadzone": 0.63,
    "unit.target_filter_alpha": 0.4722,
}
BALANCED_SIMPLE = dict(BALANCED)
BALANCED_SIMPLE["aim_control.prediction_min_ms"] = 30.0
AGGRESSIVE = {
    "aim_control.lock_box_filter_mode": "fixed",
    "aim_control.lock_box_smoothing_alpha": 0.1509,
    "aim_control.prediction_min_ms": 44.56,
    "aim_control.prediction_max_ms": 127.51,
    "aim_control.prediction_box_ratio": 0.6966,
    "aim_control.prediction_cap_px": 50.87,
    "aim_control.prediction_vel_tau": 0.0759,
    "aim_control.reversal_damp": 0.355,
    "aim_control.unit_max_counts": 254.9,
    "unit.deadzone": 4.98,
    "unit.far_gain": 1.089,
    "unit.near_gain": 0.6715,
    "unit.near_radius_px": 16.62,
    "unit.gain_softness_px_x": 17.79,
    "unit.stop_deadzone": 0.63,
    "unit.target_filter_alpha": 0.4722,
    "unit.move_steps": 1,
}


def speed_quartile_table(sim_cfg, cfg, episodes, phases, edges):
    """Bucket locked-row |vx| into quartiles; report lag/error P75 per bucket."""
    recs = [run_episode(sim_cfg, cfg, ep, ep_id=i, output_phase_s=phases[i])
            for i, ep in enumerate(episodes)]
    buckets = {k: {"lag": [], "err": []} for k in ("q1", "q2", "q3", "q4")}
    for rec in recs:
        for row in rec.obs:
            locked = (row["target"] is not None
                      and str(row["lock_reason"] or "").startswith("locked"))
            if not locked:
                continue
            vx, vy = row["vel_raw"]
            sp = math.hypot(vx, vy)
            ex, ey = row["obs_err"]
            err = math.hypot(ex, ey)
            lag = (ex * vx + ey * vy) / sp if sp > 1.0 else 0.0
            if sp < edges[0]:
                b = "q1"
            elif sp < edges[1]:
                b = "q2"
            elif sp < edges[2]:
                b = "q3"
            else:
                b = "q4"
            buckets[b]["lag"].append(lag)
            buckets[b]["err"].append(err)
    out = {}
    for k, v in buckets.items():
        out[k] = {
            "samples": len(v["lag"]),
            "lag_p75": percentile(v["lag"], .75),
            "err_p75": percentile(v["err"], .75),
        }
    return out


def main():
    metas, episodes, phases, idx = prepare_dataset()
    prod = load_production_cfg()
    sim_cfg = build_sim_cfg(["aim_20260906_003353.log", "aim_20260906_001842.log",
                             "aim_20260906_000134.log"])
    bjson = json.load(open(os.path.join(OUT_DIR, "baseline_metrics.json"),
                           encoding="utf-8"))
    base_train = bjson["splits"]["train"]
    base_val = bjson["splits"]["val"]

    named = {
        "robust": ROBUST,
        "balanced": BALANCED,
        "balanced_simple": BALANCED_SIMPLE,
        "aggressive": AGGRESSIVE,
    }
    results = {}
    for name, vals in named.items():
        cfg = patch_cfg(prod, vals)
        r = {"values": vals, "splits": {}}
        for sp in ("train", "val"):
            res = evaluate_set(sim_cfg, cfg, [episodes[i] for i in idx[sp]],
                               [phases[i] for i in idx[sp]])
            res.pop("_recs", None)
            res.pop("_per_ep", None)
            r["splits"][sp] = res
        r["fails_train"] = hard_gates(r["splits"]["train"], base_train)
        r["dwell_ok"] = all(
            r["splits"][sp]["dwell_inner60"] >=
            (base_train if sp == "train" else base_val)["dwell_inner60"] - 0.005
            for sp in ("train", "val"))
        r["score_train"] = improvement_score(r["splits"]["train"], base_train)
        results[name] = r
        t = r["splits"]["train"]
        print("%-16s score=%7.3f fails=%s dwell_ok=%s xlag=%.2f err=%.2f "
              "dwell=%.3f quiet=%.1f" % (
                  name, r["score_train"], r["fails_train"], r["dwell_ok"],
                  t["strafe_xlag_p75"], t["err_p75"], t["dwell_inner60"],
                  t["quiet_cmd_p95"]))

    # bootstrap stability on train (5 resamples)
    rng = random.Random(SEED)
    train_ids = idx["train"]
    stability = {}
    for name in ("robust", "balanced", "aggressive"):
        cfg = patch_cfg(prod, named[name])
        scores = []
        for b in range(5):
            sample = [rng.choice(train_ids) for _ in train_ids]
            res = evaluate_set(sim_cfg, cfg, [episodes[i] for i in sample],
                               [phases[i] for i in sample])
            res.pop("_recs", None)
            res.pop("_per_ep", None)
            scores.append(improvement_score(res, base_train))
        stability[name] = {
            "bootstrap_scores": [round(s, 3) for s in scores],
            "mean": round(statistics.mean(scores), 3),
            "stdev": round(statistics.stdev(scores), 3),
        }
        print("stability", name, stability[name])

    # final holdout evaluation (single pass) + speed quartiles
    hold_ids = idx["holdout"]
    hold_eps = [episodes[i] for i in hold_ids]
    hold_phases = [phases[i] for i in hold_ids]
    # speed edges from baseline holdout run speeds — reuse approximate edges
    edges = (120.0, 320.0, 700.0)
    holdout = {}
    base_res = evaluate_set(sim_cfg, prod, hold_eps, hold_phases)
    base_res.pop("_recs", None)
    base_res.pop("_per_ep", None)
    holdout["production_baseline"] = {"holdout": base_res,
                                      "speed_quartiles": speed_quartile_table(
                                          sim_cfg, prod, hold_eps, hold_phases,
                                          edges)}
    for name, vals in named.items():
        cfg = patch_cfg(prod, vals)
        res = evaluate_set(sim_cfg, cfg, hold_eps, hold_phases)
        res.pop("_recs", None)
        res.pop("_per_ep", None)
        res["score_vs_holdout_baseline"] = improvement_score(res, base_res)
        res["dwell_delta_pp"] = round(
            (res["dwell_inner60"] - base_res["dwell_inner60"]) * 100, 2)
        holdout[name] = {"holdout": res,
                         "speed_quartiles": speed_quartile_table(
                             sim_cfg, cfg, hold_eps, hold_phases, edges)}
        print("holdout %-16s score=%.3f dwell_dpp=%+.2f xlag=%.2f err=%.2f" % (
            name, res["score_vs_holdout_baseline"], res["dwell_delta_pp"],
            res["strafe_xlag_p75"], res["err_p75"]))

    with open(os.path.join(OUT_DIR, "holdout_results2.json"), "w",
              encoding="utf-8") as f:
        json.dump({"stability": stability, "rows": holdout}, f,
                  ensure_ascii=False, indent=1)
    rec = {
        "seed": SEED,
        "note": "参数值为完整配置增补片段（叠加在当前生产 config.json 之上）；物理标定 px_per_count/view_scale 未改动",
        "candidates": {k: {"patch": v} for k, v in named.items()},
        "control": {"values": {}, "note": "当前 alpha=0.2 生产配置作为对照组保留"},
    }
    rec["control"]["values"] = {}
    with open(os.path.join(OUT_DIR, "recommended_configs.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    # refresh pareto on final holdout rows
    from search import pareto_rank
    pr = pareto_rank(
        [{"name": k, "metrics": v["holdout"]} for k, v in holdout.items()],
        base_res)
    with open(os.path.join(OUT_DIR, "pareto_candidates.json"), "w",
              encoding="utf-8") as f:
        json.dump(pr, f, ensure_ascii=False, indent=1)
    print("final2 done")


if __name__ == "__main__":
    main()
