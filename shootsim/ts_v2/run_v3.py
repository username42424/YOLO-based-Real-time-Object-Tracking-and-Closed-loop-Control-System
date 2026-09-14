# -*- coding: utf-8 -*-
"""tracking_logic_v3 runner.

Usage:
    python run_v3.py probe     # pipeline lag decomposition on train (baseline)
    python run_v3.py ablate    # A-D logic ablation with 9-phase integration

Baseline is ALWAYS built from config_backup_alpha020_20260906.json (SHA256
recorded); the live config.json is never read.
"""
import copy
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset3 import (build_split_manifest, baseline_config, baseline_sha256,
                      BASELINE_CONFIG_PATH, OUT_DIR, replay_episodes)  # noqa: E402
from harness import build_sim_cfg  # noqa: E402
from eval3 import evaluate_v3  # noqa: E402
from exp_engines import V3Engine  # noqa: E402

GATES_1PCT = 0.01


def robust_cfg():
    """The currently-applied 'robust' candidate, constructed from the backup
    baseline (never from the live config)."""
    cfg = baseline_config()
    cfg["aim_control"].update({
        "lock_box_filter_mode": "fixed",
        "lock_box_smoothing_alpha": 0.18,
        "prediction_min_ms": 30.0,
        "prediction_max_ms": 105.0,
        "prediction_box_ratio": 0.65,
        "prediction_vel_tau": 0.06,
        "reversal_damp": 0.45,
    })
    cfg["unit"].update({
        "far_gain": 1.05,
        "near_gain": 0.55,
        "gain_softness_px_x": 8.0,
    })
    return cfg


def make_plan_hook(state):
    def hook(ep_index, phase_offset_ms):
        def install(tracker):
            arb = tracker.motion_arbiter
            orig = arb.publish_aim

            def wrapped(dx, dy, steps=1, profile="linear"):
                key = phase_offset_ms
                st = state.setdefault(key, {"publishes": 0, "replaced": 0})
                st["publishes"] += 1
                if arb._chunks:
                    st["replaced"] += 1
                return orig(dx, dy, steps, profile)
            arb.publish_aim = wrapped
        return install
    return hook


CANDIDATES = [
    {"name": "A_ff_g1.0_lat60_tau50", "kind": "pred",
     "params": {"ff_gain": 1.0, "latency_s": 0.060, "vel_tau": 0.050}},
    {"name": "A_ff_g1.2_lat60_tau50", "kind": "pred",
     "params": {"ff_gain": 1.2, "latency_s": 0.060, "vel_tau": 0.050}},
    {"name": "A_ff_g0.8_lat60_tau50", "kind": "pred",
     "params": {"ff_gain": 0.8, "latency_s": 0.060, "vel_tau": 0.050}},
    {"name": "A_ff_g1.2_lat80_tau80", "kind": "pred",
     "params": {"ff_gain": 1.2, "latency_s": 0.080, "vel_tau": 0.080}},
    {"name": "A_ff_g1.0_lat45_tau35", "kind": "pred",
     "params": {"ff_gain": 1.0, "latency_s": 0.045, "vel_tau": 0.035}},
    {"name": "B_ab_a0.5_b0.25_gate30", "kind": "pred",
     "params": {"ab_enable": True, "ab_alpha": 0.5, "ab_beta": 0.25,
                "ab_innov_gate_px": 30.0, "vel_tau": 0.05}},
    {"name": "B_ab_a0.6_b0.35_gate40", "kind": "pred",
     "params": {"ab_enable": True, "ab_alpha": 0.6, "ab_beta": 0.35,
                "ab_innov_gate_px": 40.0, "vel_tau": 0.05}},
    {"name": "B_ab_a0.4_b0.2_gate25", "kind": "pred",
     "params": {"ab_enable": True, "ab_alpha": 0.4, "ab_beta": 0.2,
                "ab_innov_gate_px": 25.0, "vel_tau": 0.05}},
    {"name": "C_conf_rise3_ff1.0", "kind": "pred",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.0, "vel_tau": 0.05}},
    {"name": "C_conf_rise3_ff1.2_lat70", "kind": "pred",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.2, "latency_s": 0.070,
                "vel_tau": 0.05}},
    {"name": "D_confirm1_keep0", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.0,
                "vel_tau": 0.05}},
    {"name": "D_confirm2_keep0", "kind": "pred",
     "params": {"reversal_confirm_frames": 2, "reversal_keep_frac": 0.0,
                "vel_tau": 0.05}},
    {"name": "D_keep0.5", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.5,
                "vel_tau": 0.05}},
    {"name": "D_keep0.5_h0.6_boost", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.5,
                "post_reval_horizon_scale": 0.6, "post_reval_frames": 3,
                "post_reval_boost_frames": 3, "post_reval_tau_scale": 0.4,
                "vel_tau": 0.05}},
    {"name": "D_keep0.6_h0.5_conf", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.05}},
    {"name": "D_keep0.6_h0.5_noconf", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "vel_tau": 0.05}},
    {"name": "C_conf_ff1.1_only", "kind": "pred",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.05}},
    {"name": "D_keep0.6_only", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "vel_tau": 0.05}},
    {"name": "D_h0.5_only", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.0,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "vel_tau": 0.05}},
    {"name": "D_keep0.6_h0.5_conf_rise2", "kind": "pred",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 2,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.05}},
    {"name": "ROBUST+D_conf", "kind": "pred_on_robust",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.06}},
    {"name": "ROBUST+C_conf_only", "kind": "pred_on_robust",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.0, "vel_tau": 0.06}},
    {"name": "ROBUST+keep0.6_conf", "kind": "pred_on_robust",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.06}},
    {"name": "ROBUST+keep0.6_conf_blank0", "kind": "pred_on_robust",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 0, "ff_gain": 1.1, "vel_tau": 0.06}},
]


def rel(new, base, lower_is_better=True):
    if new is None or base in (None, 0):
        return 0.0
    if lower_is_better:
        return (base - new) / abs(base)
    return (new - base) / abs(base)


def gate_check(integ, base_integ):
    """Hard gates on phase-integrated mean AND worst phase."""
    fails = []

    def gv(integ, key, field="mean"):
        v = integ.get(key)
        return None if v is None else v[field]

    for key in ("timeout_rate@0.3", "timeout_rate@0.5", "timeout_rate@0.8"):
        m, w = gv(integ, key, "mean"), gv(integ, key, "max")
        bm = gv(base_integ, key, "mean")
        # measurement granularity: one episode's worth of the denominator
        elig = integ["per_phase"][0].get(f"eligible@{key.split('@')[1]}", 0) or 0
        tol1 = 1.0 / elig if elig else 0.0
        if m is not None and bm is not None and m > bm + tol1:
            fails.append(f"{key}_mean")
        if w is not None and bm is not None and w > bm + 2 * tol1:
            fails.append(f"{key}_worst")
    dm = gv(integ, "dwell_inner60", "mean")
    dw = gv(integ, "dwell_inner60", "min")
    bm = gv(base_integ, "dwell_inner60", "mean")
    if dm is not None and bm is not None and dm < bm - 0.01:
        fails.append("dwell_mean")
    if dw is not None and bm is not None and dw < bm - 0.02:
        fails.append("dwell_worst")
    sm = gv(integ, "stable_cmd_p95", "mean")
    sw = gv(integ, "stable_cmd_p95", "max")
    bm = gv(base_integ, "stable_cmd_p95", "mean")
    if sm is not None and bm is not None and sm > bm * 1.10:
        fails.append("stable_p95_mean")
    if sw is not None and bm is not None and sw > bm * 1.35:
        fails.append("stable_p95_worst")
    wm = gv(integ, "lead_dir_wrong_rate", "mean")
    ww = gv(integ, "lead_dir_wrong_rate", "max")
    if wm is not None and wm > GATES_1PCT:
        fails.append("dir_wrong_mean")
    if ww is not None and ww > 0.02:
        fails.append("dir_wrong_worst")
    em = gv(integ, "err_p75", "mean")
    ew = gv(integ, "err_p75", "max")
    bm = gv(base_integ, "err_p75", "mean")
    if em is not None and bm is not None and em > bm:
        fails.append("err_p75_mean")
    if ew is not None and bm is not None and ew > bm * 1.15:
        fails.append("err_p75_worst")
    xw = gv(integ, "strafe_xlag_p75", "max")
    bm = gv(base_integ, "strafe_xlag_p75", "mean")
    if xw is not None and bm is not None and xw > bm * 1.6:
        fails.append("xlag_worst_runaway")
    return fails


def score(integ, base_integ):
    def gv(integ, key):
        v = integ.get(key)
        return None if v is None else v["mean"]
    s = 0.0
    s += 3.0 * rel(gv(integ, "strafe_xlag_p75"), gv(base_integ, "strafe_xlag_p75"))
    s += 2.0 * rel(gv(integ, "err_p75"), gv(base_integ, "err_p75"))
    s += 2.0 * rel(gv(integ, "strafe_err_p75"), gv(base_integ, "strafe_err_p75"))
    s += 1.5 * rel(gv(integ, "first_inner60_median"), gv(base_integ, "first_inner60_median"))
    s += 1.5 * rel(gv(integ, "first_inner60_p75"), gv(base_integ, "first_inner60_p75"))
    s += 2.0 * rel(gv(integ, "dwell_inner60"), gv(base_integ, "dwell_inner60"),
                   lower_is_better=False)
    s += 1.0 * rel(gv(integ, "stable_cmd_p95"), gv(base_integ, "stable_cmd_p95"))
    s += 1.0 * rel(gv(integ, "reversal_recovery_ms_median"),
                   gv(base_integ, "reversal_recovery_ms_median"))
    return s


def eval_candidate(name, main_cfg, params, splits, metas, assign, sim_cfg):
    """Evaluate one candidate on the given splits with 9-phase integration."""
    out = {"name": name, "values": params or {}, "splits": {}}
    engine_factory = None
    if params is not None:
        engine_factory = lambda mc, _p=params: V3Engine(mc, _p)
    for sp in splits:
        ids = [i for i, a in enumerate(assign) if a == sp]
        metas_sp = [metas[i] for i in ids]
        eps, elig, phases = [], [], []
        for m, ep in zip(metas_sp, replay_episodes(metas_sp)):
            eps.append(ep)
            elig.append(m["elig"])
            phases.append(m["output_phase_s"])
        if engine_factory is not None:
            integ = _evaluate_with_engine(sim_cfg, main_cfg, eps, elig, phases,
                                          engine_factory)
        else:
            integ = evaluate_v3(sim_cfg, main_cfg, eps, elig, phases=phases)
        out["splits"][sp] = integ
    return out


def _evaluate_with_engine(sim_cfg, main_cfg, eps, elig, phases, engine_factory):
    """Same as evaluate_v3 but with an engine swap installed per run."""
    from eval3 import (aggregate_one_phase, integrate_phases,
                       PHASE_OFFSETS_MS, PHASE_PERIOD_S)
    cap_counts = float((main_cfg.get("aim_control") or {}).get(
        "unit_max_counts", 280.3026))
    per_phase = []
    for off in PHASE_OFFSETS_MS:
        recs = []
        for i, ep in enumerate(eps):
            ph = None if phases is None else phases[i]
            if ph is not None and off:
                ph = (ph + off / 1000.0) % PHASE_PERIOD_S
            recs.append(_run_with_engine(sim_cfg, main_cfg, ep, i, ph,
                                         engine_factory))
        per_phase.append(aggregate_one_phase(recs, elig, cap_counts))
    return integrate_phases(per_phase, PHASE_OFFSETS_MS)


def _run_with_engine(sim_cfg, main_cfg, ep, ep_id, phase, engine_factory):
    from harness import run_episode
    return run_episode(sim_cfg, main_cfg, ep, ep_id=ep_id,
                       output_phase_s=phase, collect_dets=False,
                       engine_factory=engine_factory)


def cmd_probe(metas, assign, sim_cfg):
    """Pipeline lag decomposition on the train split with the baseline."""
    base = baseline_config()
    ids = [i for i, a in enumerate(assign) if a == "train"]
    eps_all = replay_episodes([metas[i] for i in ids])
    eps, elig, phases = [], [], []
    for i, m, ep in zip(ids, [metas[i] for i in ids], eps_all):
        eps.append(ep)
        elig.append(m["elig"])
        phases.append(m["output_phase_s"])
    state = {}
    integ = evaluate_v3(sim_cfg, base, eps, elig, phases=phases,
                        tracker_hook=make_plan_hook(state))
    plan = {
        "publishes": sum(v["publishes"] for v in state.values()),
        "replaced_with_pending": sum(v["replaced"] for v in state.values()),
    }
    plan["replaced_rate"] = (plan["replaced_with_pending"] / plan["publishes"]
                             if plan["publishes"] else 0.0)
    result = {
        "baseline_sha256": baseline_sha256(),
        "baseline_config_path": BASELINE_CONFIG_PATH,
        "episodes": len(eps),
        "phase0_metrics": integ["per_phase"][4],   # exact-fit phase
        "plan_replacement": plan,
    }
    with open(os.path.join(OUT_DIR, "probe_baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    p = integ["per_phase"][4]
    print(json.dumps({k: p[k] for k in (
        "first_output_latency_ms_median", "velocity_ready_s_median",
        "lock_warmup_frames", "xlead_zero_on_motion", "cap_saturation_rate",
        "lead_dir_wrong_rate", "lead_insufficient_rate", "lead_mag_p50",
        "lead_mag_p75", "reversal_recovery_ms_median", "reversal_events",
        "flip_rate", "flip_pairs", "stable_cmd_p75", "stable_cmd_p95",
        "stable_tick_diff_p95", "stable_flip_rate", "stable_jump_count",
        "strafe_xlag_p75", "err_p75", "dwell_inner60",
        "timeout_rate@0.5", "eligible@0.5", "censored@0.5")},
        ensure_ascii=False, indent=1))
    print("plan_replacement:", plan)


def cmd_ablate(metas, assign, sim_cfg):
    base = baseline_config()
    robust = robust_cfg()
    rows = []
    spec = [("baseline_alpha020", base, None),
            ("robust_applied", robust, None)]
    for c in CANDIDATES:
        cfg = robust if c["kind"] == "pred_on_robust" else base
        spec.append((c["name"], cfg, c["params"]))
    for name, cfg, params in spec:
        t0 = time.time()
        res = eval_candidate(name, cfg, params, ("train", "validation"),
                             metas, assign, sim_cfg)
        res["kind"] = "config" if params is None else "predictor"
        res["baseline_sha256"] = baseline_sha256()
        b = rows[0]["splits"]["train"] if rows else None
        bv = rows[0]["splits"]["validation"] if rows else None
        res["gates_train"] = gate_check(res["splits"]["train"], b) if b else []
        res["gates_validation"] = (gate_check(res["splits"]["validation"], bv)
                                   if bv else [])
        res["score_train"] = score(res["splits"]["train"], b) if b else 0.0
        res["score_validation"] = (score(res["splits"]["validation"], bv)
                                   if bv else 0.0)
        res["eval_seconds"] = round(time.time() - t0, 1)
        rows.append(res)
        t = res["splits"]["train"]["strafe_xlag_p75"]["mean"]
        v = res["splits"]["validation"]["strafe_xlag_p75"]["mean"]
        print("%-28s train_xlag=%6.2f val_xlag=%6.2f score=%5.2f/%5.2f "
              "gates=%s|%s (%.1fs)" % (
                  name, t, v, res["score_train"], res["score_validation"],
                  res["gates_train"], res["gates_validation"],
                  res["eval_seconds"]))
        with open(os.path.join(OUT_DIR, "LOGIC_ABLATION_RESULTS.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"baseline_sha256": baseline_sha256(),
                       "candidates": rows}, f, ensure_ascii=False, indent=1)
    print("ablate done")


def main():
    metas, assign, idx = build_split_manifest()
    sim_cfg = build_sim_cfg(list(__import__("dataset3").LOG_META))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ablate"
    if cmd == "probe":
        cmd_probe(metas, assign, sim_cfg)
    elif cmd == "ablate":
        cmd_ablate(metas, assign, sim_cfg)
    else:
        print("unknown cmd")


if __name__ == "__main__":
    main()
