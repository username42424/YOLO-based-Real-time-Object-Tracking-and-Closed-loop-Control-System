# -*- coding: utf-8 -*-
"""tracking_logic_v4 runner.

Principles (per task spec):
  - baseline is ALWAYS config_backup_alpha020_20260906.json; live config.json
    is never read;
  - one immutable dataset manifest per run (SHA256 recorded);
  - the scoring formula is written to scoring_config.json BEFORE any
    candidate is evaluated and is never edited afterwards;
  - only train and validation run; holdout_ext is untouched;
  - results land in 日志V4 (primary) and are mirrored byte-identically to
    shootsim/results/tracking_logic_v4.

Usage:
    python run_v4.py all       # manifest + probe + ablation + reports
    python run_v4.py manifest
    python run_v4.py ablate
"""
import copy
import hashlib
import json
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "shootsim"))

from dataset3 import BASELINE_CONFIG_PATH, LOG_META, load_all_episodes  # noqa: E402
from dataset3 import replay_episodes, stratified_split  # noqa: E402
from harness import build_sim_cfg  # noqa: E402
from eval3 import evaluate_v3, sha256_of_file, PHASE_OFFSETS_MS  # noqa: E402
from exp_engines import V3Engine  # noqa: E402

RUN_DIR = os.path.join(ROOT, "日志V4")
MIRROR_DIR = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v4")
SEED = 20260906
CODE_FILES = ["eval3.py", "exp_engines.py", "dataset3.py", "harness.py",
              "logparse.py", "run_v4.py"]


def sha256_text(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def robust_cfg():
    cfg = json.load(open(BASELINE_CONFIG_PATH, encoding="utf-8"))
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


# ── preregistered scoring (written before any evaluation) ────────────────
SCORING = {
    "note": ("预注册评分公式：运行前写入，运行后不得修改。rel(x, base) = "
             "(base - x)/|base|（越低越好）或 (x - base)/|base|（驻留率越高越好）。"
             "strafe_err_p75 仅为横移子集诊断指标，不参与评分（避免与 err_p75 重复加权）。"),
    "terms": [
        {"metric": "strafe_xlag_p75", "weight": 3.0, "direction": "lower"},
        {"metric": "err_p75", "weight": 2.0, "direction": "lower"},
        {"metric": "first_inner60_median", "weight": 1.5, "direction": "lower"},
        {"metric": "first_inner60_p75", "weight": 1.5, "direction": "lower"},
        {"metric": "dwell_inner60", "weight": 2.0, "direction": "higher"},
        {"metric": "stable_cmd_p95", "weight": 1.0, "direction": "lower"},
        {"metric": "reversal_recovery_ms_median", "weight": 1.0,
         "direction": "lower"},
    ],
    "excluded_from_score": ["strafe_err_p75 (diagnostic only)"],
    "hard_gates": {
        "timeout_rate@T": "mean <= baseline_mean + 1/eligible; "
                          "worst <= baseline_mean + 2/eligible (T=0.3/0.5/0.8)",
        "dwell_inner60": "mean >= baseline_mean - 1pp; worst >= baseline_mean - 2pp",
        "stable_cmd_p95": "mean <= baseline_mean * 1.10; "
                          "worst <= baseline_mean * 1.35",
        "lead_dir_wrong_rate": "mean <= 1%",
        "err_p75": "mean <= baseline_mean; worst <= baseline_mean * 1.15",
        "xlag_worst_runaway": "worst <= baseline_mean * 1.6",
    },
}


def make_plan_hook(state):
    def hook(ep_index, phase_offset_ms):
        def install(tracker):
            arb = tracker.motion_arbiter
            orig = arb.publish_aim

            def wrapped(dx, dy, steps=1, profile="linear"):
                st = state.setdefault(phase_offset_ms,
                                      {"publishes": 0, "replaced": 0})
                st["publishes"] += 1
                if arb._chunks:
                    st["replaced"] += 1
                return orig(dx, dy, steps, profile)
            arb.publish_aim = wrapped
        return install
    return hook


CANDIDATES = [
    {"name": "A_ff_g1.0_lat60_tau50",
     "params": {"ff_gain": 1.0, "latency_s": 0.060, "vel_tau": 0.050}},
    {"name": "A_ff_g1.2_lat60_tau50",
     "params": {"ff_gain": 1.2, "latency_s": 0.060, "vel_tau": 0.050}},
    {"name": "A_ff_g0.8_lat60_tau50",
     "params": {"ff_gain": 0.8, "latency_s": 0.060, "vel_tau": 0.050}},
    {"name": "A_ff_g1.2_lat80_tau80",
     "params": {"ff_gain": 1.2, "latency_s": 0.080, "vel_tau": 0.080}},
    {"name": "A_ff_g1.0_lat45_tau35",
     "params": {"ff_gain": 1.0, "latency_s": 0.045, "vel_tau": 0.035}},
    {"name": "B_ab_a0.5_b0.25_gate30",
     "params": {"ab_enable": True, "ab_alpha": 0.5, "ab_beta": 0.25,
                "ab_innov_gate_px": 30.0, "vel_tau": 0.05}},
    {"name": "B_ab_a0.6_b0.35_gate40",
     "params": {"ab_enable": True, "ab_alpha": 0.6, "ab_beta": 0.35,
                "ab_innov_gate_px": 40.0, "vel_tau": 0.05}},
    {"name": "B_ab_a0.4_b0.2_gate25",
     "params": {"ab_enable": True, "ab_alpha": 0.4, "ab_beta": 0.2,
                "ab_innov_gate_px": 25.0, "vel_tau": 0.05}},
    {"name": "C_conf_rise3_ff1.0",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.0, "vel_tau": 0.05}},
    {"name": "C_conf_rise3_ff1.2_lat70",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.2, "latency_s": 0.070,
                "vel_tau": 0.05}},
    {"name": "D_confirm1_keep0",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.0,
                "vel_tau": 0.05}},
    {"name": "D_confirm2_keep0",
     "params": {"reversal_confirm_frames": 2, "reversal_keep_frac": 0.0,
                "vel_tau": 0.05}},
    {"name": "D_keep0.5",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.5,
                "vel_tau": 0.05}},
    {"name": "D_keep0.5_h0.6_boost",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.5,
                "post_reval_horizon_scale": 0.6, "post_reval_frames": 3,
                "post_reval_boost_frames": 3, "post_reval_tau_scale": 0.4,
                "vel_tau": 0.05}},
    {"name": "D_keep0.6_h0.5_conf",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.05}},
    {"name": "D_keep0.6_h0.5_noconf",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "vel_tau": 0.05}},
    {"name": "C_conf_ff1.1_only",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.05}},
    {"name": "D_keep0.6_only",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "vel_tau": 0.05}},
    {"name": "D_h0.5_only",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.0,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "vel_tau": 0.05}},
    {"name": "D_keep0.6_h0.5_conf_rise2",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 2,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.05}},
    {"name": "ROBUST+D_conf",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "post_reval_horizon_scale": 0.5, "post_reval_frames": 4,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.06},
     "on_robust": True},
    {"name": "ROBUST+C_conf_only",
     "params": {"conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.0, "vel_tau": 0.06},
     "on_robust": True},
    {"name": "ROBUST+keep0.6_conf_blank0",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 0, "ff_gain": 1.1, "vel_tau": 0.06},
     "on_robust": True},
    {"name": "ROBUST+keep0.6_conf_blank1",
     "params": {"reversal_confirm_frames": 1, "reversal_keep_frac": 0.6,
                "conf_enable": True, "conf_speed_low": 150.0,
                "conf_speed_high": 700.0, "conf_rise_frames": 3,
                "conf_blank_frames": 1, "ff_gain": 1.1, "vel_tau": 0.06},
     "on_robust": True},
]


def build_and_save_manifest():
    metas = load_all_episodes()
    assign = stratified_split(metas, seed=SEED)
    idx = {k: [i for i, a in enumerate(assign) if a == k]
           for k in ("train", "validation", "holdout_ext")}
    episodes_meta = []
    for i, m in enumerate(metas):
        episodes_meta.append({
            "index": i, "log": m["log"], "split": assign[i],
            "group": m["group"], "chest_ratio": m["chest_ratio"],
            "vertical_recoil": m["vertical_recoil"],
            "duration_s": m["duration_s"], "n_points": m["n_points"],
            "strafe_share": m["strafe_share"],
            "max_detected_streak_s": m["elig"]["max_detected_streak_s"],
            "started_outside_inner60": m["elig"]["started_outside_inner60"],
            "first_detection_t": m["elig"]["first_detection_t"],
        })
    manifest = {
        "seed": SEED,
        "episode_count": len(metas),
        "split_counts": {k: len(v) for k, v in idx.items()},
        "phases_ms": list(PHASE_OFFSETS_MS),
        "baseline_config_path": BASELINE_CONFIG_PATH,
        "baseline_config_sha256": sha256_of_file(BASELINE_CONFIG_PATH),
        "manifest_sha256": None,   # filled below (self-excluded)
        "code_sha256": {f: sha256_of_file(os.path.join(HERE, f))
                        for f in CODE_FILES},
        "episodes": episodes_meta,
    }
    manifest["manifest_sha256"] = sha256_text(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return metas, assign, idx, manifest


def rel(new, base, lower_is_better=True):
    if new is None or base in (None, 0):
        return 0.0
    if lower_is_better:
        return (base - new) / abs(base)
    return (new - base) / abs(base)


def gv(integ, key, field="mean"):
    v = integ.get(key)
    return None if v is None else v[field]


def gate_check(integ, base_integ):
    fails = []
    for key in ("timeout_rate@0.3", "timeout_rate@0.5", "timeout_rate@0.8"):
        m, w = gv(integ, key, "mean"), gv(integ, key, "max")
        bm = gv(base_integ, key, "mean")
        T = key.split("@")[1]
        elig = integ["per_phase"][0].get(f"eligible@{T}", 0) or 0
        tol1 = 1.0 / elig if elig else 0.0
        if m is not None and bm is not None and m > bm + tol1:
            fails.append(f"{key}_mean")
        if w is not None and bm is not None and w > bm + 2 * tol1:
            fails.append(f"{key}_worst")
    dm, dw = gv(integ, "dwell_inner60", "mean"), gv(integ, "dwell_inner60", "min")
    bm = gv(base_integ, "dwell_inner60", "mean")
    if dm is not None and bm is not None and dm < bm - 0.01:
        fails.append("dwell_mean")
    if dw is not None and bm is not None and dw < bm - 0.02:
        fails.append("dwell_worst")
    sm, sw = gv(integ, "stable_cmd_p95", "mean"), gv(integ, "stable_cmd_p95", "max")
    bm = gv(base_integ, "stable_cmd_p95", "mean")
    if sm is not None and bm is not None and sm > bm * 1.10:
        fails.append("stable_p95_mean")
    if sw is not None and bm is not None and sw > bm * 1.35:
        fails.append("stable_p95_worst")
    wm = gv(integ, "lead_dir_wrong_rate", "mean")
    ww = gv(integ, "lead_dir_wrong_rate", "max")
    if wm is not None and wm > 0.01:
        fails.append("dir_wrong_mean")
    if ww is not None and ww > 0.02:
        fails.append("dir_wrong_worst")
    em, ew = gv(integ, "err_p75", "mean"), gv(integ, "err_p75", "max")
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
    s = 0.0
    for term in SCORING["terms"]:
        k, w = term["metric"], term["weight"]
        lower = term["direction"] == "lower"
        s += w * rel(gv(integ, k), gv(base_integ, k), lower_is_better=lower)
    return s


def eval_candidate(name, main_cfg, params, splits, split_data):
    out = {"name": name, "kind": "config" if params is None else "predictor",
           "values": params or {}, "splits": {}}
    engine_factory = None
    if params is not None:
        engine_factory = lambda mc, _p=params: V3Engine(mc, _p)
    for sp in splits:
        eps, elig, phases = split_data[sp]
        integ = evaluate_v3(sim_cfg_global, main_cfg, eps, elig, phases=phases,
                            engine_factory=engine_factory)
        out["splits"][sp] = integ
    return out


sim_cfg_global = None


def write_both(relpath, obj):
    """Write JSON to 日志V4 (primary) and mirror byte-identically."""
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    for base in (RUN_DIR, MIRROR_DIR):
        path = os.path.join(base, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    return text


def main():
    global sim_cfg_global
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(MIRROR_DIR, exist_ok=True)
    sim_cfg_global = build_sim_cfg(list(LOG_META))

    # 1. immutable manifest
    metas, assign, idx, manifest = build_and_save_manifest()
    write_both("dataset_split_v4.json", manifest)
    print("manifest: episodes=%d train=%d validation=%d holdout_ext=%d "
          "manifest_sha=%s" % (
              manifest["episode_count"], manifest["split_counts"]["train"],
              manifest["split_counts"]["validation"],
              manifest["split_counts"]["holdout_ext"],
              manifest["manifest_sha256"][:16]))

    # 2. preregistered scoring (before any evaluation)
    write_both("scoring_config.json", SCORING)

    # 3. split data (replay episodes once, reuse everywhere)
    split_data = {}
    for sp in ("train", "validation"):
        ids = idx[sp]
        metas_sp = [metas[i] for i in ids]
        eps = replay_episodes(metas_sp)
        elig = [m["elig"] for m in metas_sp]
        phases = [m["output_phase_s"] for m in metas_sp]
        split_data[sp] = (eps, elig, phases)

    # 4. probe (baseline pipeline diagnostics, exact-fit phase)
    base_cfg = json.load(open(BASELINE_CONFIG_PATH, encoding="utf-8"))
    state = {}
    integ_probe = evaluate_v3(sim_cfg_global, base_cfg, split_data["train"][0],
                              split_data["train"][1],
                              phases=split_data["train"][2],
                              tracker_hook=make_plan_hook(state))
    probe = {
        "baseline_config_sha256": manifest["baseline_config_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "episodes": manifest["split_counts"]["train"],
        "plan_replacement": {
            "publishes": sum(v["publishes"] for v in state.values()),
            "replaced_with_pending": sum(v["replaced"] for v in state.values()),
        },
        "phase0_metrics": integ_probe["per_phase"][PHASE_OFFSETS_MS.index(0)],
    }
    probe["plan_replacement"]["replaced_rate"] = (
        probe["plan_replacement"]["replaced_with_pending"]
        / max(1, probe["plan_replacement"]["publishes"]))
    write_both("probe_baseline_v4.json", probe)
    print("probe done: replaced_rate=%.3f xlag=%.2f err=%.2f" % (
        probe["plan_replacement"]["replaced_rate"],
        probe["phase0_metrics"]["strafe_xlag_p75"],
        probe["phase0_metrics"]["err_p75"]))

    # 5. ablation
    spec = [("baseline_alpha020", base_cfg, None),
            ("robust_applied", robust_cfg(), None)]
    for c in CANDIDATES:
        cfg = robust_cfg() if c.get("on_robust") else copy.deepcopy(base_cfg)
        spec.append((c["name"], cfg, c["params"]))
    rows = []
    for name, cfg, params in spec:
        t0 = time.time()
        res = eval_candidate(name, cfg, params, ("train", "validation"),
                             split_data)
        res["kind"] = "config" if params is None else "predictor"
        res["baseline_config_sha256"] = manifest["baseline_config_sha256"]
        res["manifest_sha256"] = manifest["manifest_sha256"]
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
        print("%-30s train_xlag=%6.2f val_xlag=%6.2f err=%6.2f score=%5.2f/%5.2f "
              "gates=%s|%s (%.1fs)" % (
                  name, t, v,
                  res["splits"]["train"]["err_p75"]["mean"],
                  res["score_train"], res["score_validation"],
                  res["gates_train"], res["gates_validation"],
                  res["eval_seconds"]))
    write_both("LOGIC_ABLATION_RESULTS.json", {
        "run_id": "tracking_logic_v4",
        "seed": SEED,
        "candidate_count": len(rows),
        "phase_count": len(PHASE_OFFSETS_MS),
        "phases_ms": list(PHASE_OFFSETS_MS),
        "split_counts": manifest["split_counts"],
        "manifest_sha256": manifest["manifest_sha256"],
        "baseline_config_sha256": manifest["baseline_config_sha256"],
        "code_sha256": manifest["code_sha256"],
        "scoring": SCORING,
        "candidates": rows,
    })
    print("ablate done: %d candidates" % len(rows))


if __name__ == "__main__":
    main()
