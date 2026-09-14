# -*- coding: utf-8 -*-
"""Run the staged search. Usage:
    python run_stages.py prepare      # dataset split + baseline metrics
    python run_stages.py A [budget]   # filter-chain ablation
    python run_stages.py B [budget]   # prediction params
    python run_stages.py C [budget]   # control params
    python run_stages.py D [budget]   # missing params
    python run_stages.py final        # holdout + pareto + candidates
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import (OUT_DIR, SEED, Sweep, TPE, prepare_dataset, hard_gates,
                    improvement_score, pareto_rank, fmt)  # noqa: E402
from harness import load_production_cfg, patch_cfg, evaluate_set  # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)

_STATE = {}


def load_state():
    if not _STATE:
        with open(os.path.join(OUT_DIR, "state.json"), encoding="utf-8") as f:
            _STATE.update(json.load(f))
    return _STATE


def save_state(state):
    with open(os.path.join(OUT_DIR, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def prepare():
    metas, episodes, phases, idx = prepare_dataset()
    prod = load_production_cfg()
    from harness import build_sim_cfg
    sim_cfg = build_sim_cfg(["aim_20260906_003353.log", "aim_20260906_001842.log",
                             "aim_20260906_000134.log"])
    baseline = {"name": "production_baseline", "values": {}, "splits": {}}
    for sp in ("train", "val", "holdout"):
        ep_ids = idx[sp]
        t0 = time.time()
        res = evaluate_set(sim_cfg, prod, [episodes[i] for i in ep_ids],
                           [phases[i] for i in ep_ids])
        res.pop("_recs", None)
        res.pop("_per_ep", None)
        res["eval_seconds"] = round(time.time() - t0, 2)
        baseline["splits"][sp] = res
        print(sp, "episodes=%d eval_s=%.1f err_p75=%s xlag_p75=%s "
              "inner60_med=%s dwell=%s to=%s dirwrong=%s" % (
                  len(ep_ids), res["eval_seconds"], res["err_p75"],
                  res["strafe_xlag_p75"], res["first_inner60_median"],
                  res["dwell_inner60"], res["timeout_rate"],
                  res["strafe_dir_wrong_rate"]))
    with open(os.path.join(OUT_DIR, "baseline_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=1)
    save_state({"baseline_train": baseline["splits"]["train"]})
    print("prepare done")


def anchors(sw):
    sw.evaluate("anchor_production", {}, splits=("train", "val"))
    sw.evaluate("anchor_alpha_03", {
        "aim_control.lock_box_filter_mode": "fixed",
        "aim_control.lock_box_smoothing_alpha": 0.30}, splits=("train", "val"))
    sw.evaluate("anchor_alpha_04", {
        "aim_control.lock_box_filter_mode": "fixed",
        "aim_control.lock_box_smoothing_alpha": 0.40}, splits=("train", "val"))


def top_from_stage(stage, n):
    path = os.path.join(OUT_DIR, f"stage_{stage}_results.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["gate_pass"]:
                rows.append(r)
    rows.sort(key=lambda r: -r["score"])
    seen = set()
    out = []
    for r in rows:
        key = json.dumps(r["values"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


def stage_a(budget=40):
    state = load_state()
    metas, episodes, phases, idx = prepare_dataset()
    base = state["baseline_train"]
    sw = Sweep("a", metas, episodes, phases, idx, base)
    anchors(sw)
    sw.evaluate("A0_full_chain", {}, splits=("train", "val"))
    sw.evaluate("A1_unit_filter_bypass", {"unit.target_filter_alpha": 1.0},
                splits=("train", "val"))
    sw.evaluate("A2_lock_filter_bypass",
                {"aim_control.lock_box_smoothing_alpha": 1.0},
                splits=("train", "val"))
    sw.evaluate("A3_both_kept", {}, splits=("train", "val"))
    # A4 (independent position/velocity filtering) is already the engine's
    # architecture (velocity estimated from the raw box) — audit finding.
    lock_alphas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60]
    unit_alphas = [0.30, 0.50, 0.80, 1.00]
    for la in lock_alphas:
        sw.evaluate(f"A_grid_lock_{la}", {
            "aim_control.lock_box_filter_mode": "fixed",
            "aim_control.lock_box_smoothing_alpha": la}, splits=("train", "val"))
    for ua in unit_alphas:
        sw.evaluate(f"A_grid_unit_{ua}", {
            "aim_control.lock_box_filter_mode": "fixed",
            "aim_control.lock_box_smoothing_alpha": 0.20,
            "unit.target_filter_alpha": ua}, splits=("train", "val"))
    tpe = TPE([("lock_alpha", 0.15, 0.60, False),
               ("unit_alpha", 0.25, 1.00, False)], seed=SEED)
    for i in range(budget):
        x = tpe.sample()
        r = sw.evaluate(f"A_tpe_{i}", {
            "aim_control.lock_box_filter_mode": "fixed",
            "aim_control.lock_box_smoothing_alpha": round(x["lock_alpha"], 4),
            "unit.target_filter_alpha": round(x["unit_alpha"], 4)},
            splits=("train", "val"))
        tpe.update(x, -r["score"])
        print("A tpe %d: lock=%.3f unit=%.3f score=%.3f fails=%s" % (
            i, x["lock_alpha"], x["unit_alpha"], r["score"], r["fails"]))
    print("stage A done")


def stage_b(budget=50):
    state = load_state()
    metas, episodes, phases, idx = prepare_dataset()
    base = state["baseline_train"]
    sw = Sweep("b", metas, episodes, phases, idx, base)
    anchors(sw)
    top_a = top_from_stage("a", 5)
    combos = []
    for r in top_a:
        v = dict(r["values"])
        v.setdefault("aim_control.lock_box_filter_mode", "fixed")
        combos.append(v)
    if not combos:
        combos = [{}]
    tpe = TPE([("pred_min_ms", 10.0, 55.0, False),
               ("pred_max_ms", 60.0, 150.0, False),
               ("box_ratio", 0.35, 0.85, False),
               ("cap_px", 18.0, 55.0, False),
               ("lock_frames", 2, 4, True),
               ("vel_tau", 0.025, 0.14, False)], seed=SEED + 1)
    for i in range(budget):
        x = tpe.sample()
        base_vals = combos[i % len(combos)]
        vals = dict(base_vals)
        vals.update({
            "aim_control.prediction_min_ms": round(x["pred_min_ms"], 2),
            "aim_control.prediction_max_ms": round(x["pred_max_ms"], 2),
            "aim_control.prediction_box_ratio": round(x["box_ratio"], 4),
            "aim_control.prediction_cap_px": round(x["cap_px"], 2),
            "aim_control.prediction_lock_frames": x["lock_frames"],
            "aim_control.prediction_vel_tau": round(x["vel_tau"], 4),
        })
        if vals.get("aim_control.prediction_min_ms", 20) >= \
                vals.get("aim_control.prediction_max_ms", 150):
            continue
        r = sw.evaluate(f"B_tpe_{i}", vals, splits=("train", "val"))
        tpe.update(x, -r["score"])
        print("B tpe %d: combo=%d score=%.3f fails=%s xlag=%s" % (
            i, i % len(combos), r["score"], r["fails"],
            r["splits"]["train"].get("strafe_xlag_p75")))
    print("stage B done")


def stage_c(budget=60):
    state = load_state()
    metas, episodes, phases, idx = prepare_dataset()
    base = state["baseline_train"]
    sw = Sweep("c", metas, episodes, phases, idx, base)
    anchors(sw)
    top_b = top_from_stage("b", 5)
    combos = [dict(r["values"]) for r in top_b] or [{}]
    tpe = TPE([("far_gain", 0.90, 1.25, False),
               ("near_gain", 0.35, 0.70, False),
               ("near_radius", 10.0, 45.0, False),
               ("soft_x", 1.0, 20.0, False),
               ("deadzone", 1.5, 5.0, False),
               ("stop_dz", 0.2, 1.5, False),
               ("reversal", 0.35, 0.75, False),
               ("max_counts", 220.0, 380.0, False),
               ("move_steps", 1, 3, True)], seed=SEED + 2)
    for i in range(budget):
        x = tpe.sample()
        if x["stop_dz"] >= x["deadzone"]:
            continue
        if x["near_gain"] > x["far_gain"]:
            continue
        base_vals = combos[i % len(combos)]
        vals = dict(base_vals)
        vals.update({
            "unit.far_gain": round(x["far_gain"], 4),
            "unit.near_gain": round(x["near_gain"], 4),
            "unit.near_radius_px": round(x["near_radius"], 2),
            "unit.gain_softness_px_x": round(x["soft_x"], 2),
            "unit.deadzone": round(x["deadzone"], 2),
            "unit.stop_deadzone": round(x["stop_dz"], 2),
            "aim_control.reversal_damp": round(x["reversal"], 3),
            "aim_control.unit_max_counts": round(x["max_counts"], 1),
            "unit.move_steps": x["move_steps"],
        })
        r = sw.evaluate(f"C_tpe_{i}", vals, splits=("train", "val"))
        tpe.update(x, -r["score"])
        print("C tpe %d: combo=%d score=%.3f fails=%s" % (
            i, i % len(combos), r["score"], r["fails"]))
    print("stage C done")


def stage_d(budget=16):
    state = load_state()
    metas, episodes, phases, idx = prepare_dataset()
    base = state["baseline_train"]
    sw = Sweep("d", metas, episodes, phases, idx, base)
    anchors(sw)
    top_c = top_from_stage("c", 3)
    combos = [dict(r["values"]) for r in top_c] or [{}]
    tpe = TPE([("decay", 0.10, 0.35, False),
               ("max_counts", 4.0, 12.0, False)], seed=SEED + 3)
    for i in range(budget):
        x = tpe.sample()
        vals = dict(combos[i % len(combos)])
        vals.update({
            "aim_control.missing_decay": round(x["decay"], 3),
            "aim_control.missing_max_counts": round(x["max_counts"], 2),
        })
        r = sw.evaluate(f"D_tpe_{i}", vals, splits=("train", "val"))
        tpe.update(x, -r["score"])
        print("D tpe %d: score=%.3f fails=%s" % (i, r["score"], r["fails"]))
    print("stage D done")


def final():
    state = load_state()
    metas, episodes, phases, idx = prepare_dataset()
    prod = load_production_cfg()
    from harness import build_sim_cfg
    sim_cfg = build_sim_cfg(["aim_20260906_003353.log", "aim_20260906_001842.log",
                             "aim_20260906_000134.log"])
    pool = []
    for stage in ("a", "b", "c", "d"):
        path = os.path.join(OUT_DIR, f"stage_{stage}_results.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["stage"] = stage
                if r["gate_pass"]:
                    pool.append(r)
    pool.sort(key=lambda r: -r["score"])
    seen = set()
    uniq = []
    for r in pool:
        key = json.dumps(r["values"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    uniq = uniq[:10]
    holdout_rows = []
    for r in uniq:
        main_cfg = patch_cfg(prod, r["values"])
        res = evaluate_set(sim_cfg, main_cfg, [episodes[i] for i in idx["holdout"]],
                           [phases[i] for i in idx["holdout"]])
        res.pop("_recs", None)
        res.pop("_per_ep", None)
        holdout_rows.append({
            "name": r["name"], "stage": r["stage"], "values": r["values"],
            "holdout": res,
        })
        print("holdout", r["name"], "score=%.3f" % improvement_score(
            res, state["baseline_train"]))
    res = evaluate_set(sim_cfg, prod, [episodes[i] for i in idx["holdout"]],
                       [phases[i] for i in idx["holdout"]])
    res.pop("_recs", None)
    res.pop("_per_ep", None)
    holdout_rows.append({"name": "production_baseline", "stage": "-",
                         "values": {}, "holdout": res})
    with open(os.path.join(OUT_DIR, "holdout_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"baseline_train": state["baseline_train"],
                   "rows": holdout_rows}, f, ensure_ascii=False, indent=1)
    pr = pareto_rank([{"name": r["name"], "metrics": r["holdout"]}
                      for r in holdout_rows], state["baseline_train"])
    with open(os.path.join(OUT_DIR, "pareto_candidates.json"), "w",
              encoding="utf-8") as f:
        json.dump(pr, f, ensure_ascii=False, indent=1, default=fmt)
    print("final done")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "prepare":
        prepare()
    elif cmd.lower() == "a":
        stage_a(budget or 40)
    elif cmd.lower() == "b":
        stage_b(budget or 50)
    elif cmd.lower() == "c":
        stage_c(budget or 60)
    elif cmd.lower() == "d":
        stage_d(budget or 16)
    elif cmd == "final":
        final()
    else:
        print("unknown cmd")
