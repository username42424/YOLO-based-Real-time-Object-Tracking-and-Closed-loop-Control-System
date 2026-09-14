# -*- coding: utf-8 -*-
"""Staged parameter search (TPE-lite over numpy) for the 20260906 v2 sweep.

Stages:
  A  filter-chain ablation: lock_box_smoothing_alpha x unit.target_filter_alpha
     plus structural bypasses (A1: second layer off; A2: lock filter off)
  B  prediction params (fixed top-5 filter combos)
  C  control params (fixed top-5 A+B combos)
  D  missing/lost-frame params (low budget, last)

Scoring: hard gates from the sweep spec, then weighted improvement vs the
production baseline.  Anchors (production, alpha=0.3, alpha=0.4) are evaluated
in every stage.  All results appended to stage_*.jsonl.
"""
import copy
import json
import math
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (build_sim_cfg, load_production_cfg, patch_cfg,
                     evaluate_set, percentile)  # noqa: E402
from dataset import (load_all_episodes, build_replay_episodes, episode_features,
                     stratify_split, fit_grid_phase)  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "tracking_sweep_20260906_v2")
SEED = 20260906

# ------------------------------------------------------------ dataset


def prepare_dataset():
    logs = ["aim_20260906_003353.log", "aim_20260906_001842.log",
            "aim_20260906_000134.log"]
    metas = load_all_episodes(logs)
    feats = [episode_features(m) for m in metas]
    keep = [i for i, f in enumerate(feats) if not f.get("invalid")]
    metas = [metas[i] for i in keep]
    feats = [feats[i] for i in keep]
    assign = stratify_split(feats, seed=SEED)
    episodes = build_replay_episodes(metas)
    phases = [m["output_phase_s"] for m in metas]
    split = {"seed": SEED, "episodes": []}
    for i, (f, a) in enumerate(zip(feats, assign)):
        split["episodes"].append({
            "idx": i, "log": f["log"], "group": f["group"],
            "duration_s": f["duration_s"], "n_points": f["n_points"],
            "strafe_share": round(f["strafe_share"], 3),
            "abs_vx_p75": f["abs_vx_p75"],
            "split": a,
        })
    with open(os.path.join(OUT_DIR, "dataset_split.json"), "w",
              encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
    idx = {k: [i for i, a in enumerate(assign) if a == k]
           for k in ("train", "val", "holdout")}
    return metas, episodes, phases, idx


# ------------------------------------------------------------ scoring

GATE_KEYS = ("timeout_rate", "no_entry_rate", "dwell_inner60", "quiet_cmd_p95",
             "strafe_dir_wrong_rate", "err_p75", "jump_count_total")


def hard_gates(res, base):
    """Return list of violated gate names (empty = pass)."""
    fails = []
    if res["timeout_rate"] > max(base["timeout_rate"], 0.05):
        fails.append("timeout_rate")
    if res["no_entry_rate"] > base["no_entry_rate"] + 1e-9:
        fails.append("no_entry_rate")
    if res["dwell_inner60"] < base["dwell_inner60"] - 0.01:
        fails.append("dwell_inner60")
    q0, q1 = base.get("quiet_cmd_p95"), res.get("quiet_cmd_p95")
    if q0 and q1 and q1 > q0 * 1.10:
        fails.append("quiet_cmd_p95")
    if res["strafe_dir_wrong_rate"] > max(0.01, base["strafe_dir_wrong_rate"] * 1.25):
        fails.append("strafe_dir_wrong_rate")
    if res["err_p75"] is not None and base["err_p75"] is not None \
            and res["err_p75"] > base["err_p75"]:
        fails.append("err_p75")
    if res["jump_count_total"] > base["jump_count_total"] * 1.5:
        fails.append("jump_count_total")
    return fails


def improvement_score(res, base):
    """Weighted relative improvement (positive = better)."""
    def rel(key, lower_is_better=True, ref=None):
        a, b = res.get(key), (ref if ref is not None else base.get(key))
        if a is None or b is None or b == 0:
            return 0.0
        if lower_is_better:
            return (b - a) / abs(b)
        return (a - b) / abs(b)

    s = 0.0
    s += 3.0 * rel("strafe_xlag_p75")           # 高速横移 X 滞后
    s += 2.0 * rel("err_p75")                   # 整体移动误差
    s += 2.0 * rel("strafe_err_p75")            # 高速移动误差
    s += 1.5 * rel("first_inner60_median")      # 首入时间
    s += 1.5 * rel("first_inner60_p75")
    s += 2.0 * rel("dwell_inner60", lower_is_better=False)
    s += 1.0 * rel("quiet_cmd_p95")             # 静止抖动（越低越好）
    return s


def pareto_rank(rows, base):
    """Nondominated sort on (min) xlag_p75, err_p75, first_inner60_median,
    (max) dwell_inner60, (min) quiet_cmd_p95."""
    keys_min = ("strafe_xlag_p75", "err_p75", "first_inner60_median",
                "quiet_cmd_p95")
    objs = []
    for r in rows:
        o = []
        for k in keys_min:
            v = r["metrics"].get(k)
            o.append(v if v is not None else math.inf)
        d = r["metrics"].get("dwell_inner60")
        o.append(-d if d is not None else math.inf)
        objs.append(o)
    n = len(rows)
    dominated = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if all(a <= b for a, b in zip(objs[j], objs[i])) and \
                    any(a < b for a, b in zip(objs[j], objs[i])):
                dominated[i] += 1
    for i, r in enumerate(rows):
        r["pareto_layer"] = dominated[i]
    return rows


# ------------------------------------------------------------ TPE-lite

class TPE:
    """Simple parzen-estimator TPE over independent per-param distributions."""

    def __init__(self, space, seed=SEED, gamma=None):
        self.space = space          # list of (name, low, high, is_int)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.obs = []               # (x_dict, score)

    def _sample_one(self, low, high, is_int):
        v = self.rng.uniform(low, high)
        return int(round(v)) if is_int else float(v)

    def sample(self):
        n = len(self.obs)
        gamma = self.gamma or max(5, min(25, n // 3))
        if n < 10:
            return {name: self._sample_one(lo, hi, it)
                    for name, lo, hi, it in self.space}
        order = sorted(range(n), key=lambda i: self.obs[i][1])
        good_idx = order[:gamma]
        bad_idx = order[gamma:]
        out = {}
        for name, lo, hi, is_int in self.space:
            g = np.array([self.obs[i][0][name] for i in good_idx], dtype=float)
            b = np.array([self.obs[i][0][name] for i in bad_idx], dtype=float)
            best = None
            for _ in range(24):
                v = self._sample_one(lo, hi, is_int)
                # log-likelihood ratio: p_good(x)/p_bad(x) via KDE
                def kde(x, pts, bw):
                    if len(pts) == 0:
                        return 1e-12
                    z = (pts - x) / bw
                    return float(np.sum(np.exp(-0.5 * z * z)) / (
                        len(pts) * bw * math.sqrt(2 * math.pi)))
                span = max(1e-6, hi - lo)
                bw_g = max(0.35 * span / math.sqrt(max(4, len(g)) + 4), 1e-3)
                bw_b = max(0.35 * span / math.sqrt(max(4, len(b)) + 4), 1e-3)
                lg = math.log(max(kde(v, g, bw_g), 1e-300))
                lb = math.log(max(kde(v, b, bw_b), 1e-300))
                score = lg - lb
                if best is None or score > best[0]:
                    best = (score, v)
            out[name] = best[1]
        return out

    def update(self, x, score):
        self.obs.append((dict(x), score))


# ------------------------------------------------------------ driver

class Sweep:
    def __init__(self, stage_name, metas, episodes, phases, idx, base_metrics):
        self.stage = stage_name
        self.metas = metas
        self.episodes = episodes
        self.phases = phases
        self.idx = idx
        self.base = base_metrics
        self.sim_cfg = build_sim_cfg(
            ["aim_20260906_003353.log", "aim_20260906_001842.log",
             "aim_20260906_000134.log"])
        self.jsonl = os.path.join(OUT_DIR, f"stage_{stage_name}_results.jsonl")
        self.evaluated = {}

    def evaluate(self, name, values, splits=("train",)):
        key = json.dumps(values, sort_keys=True)
        if key in self.evaluated:
            return self.evaluated[key]
        main_cfg = patch_cfg(load_production_cfg(), values)
        out = {"name": name, "values": values, "splits": {}}
        for sp in splits:
            ep_ids = self.idx[sp]
            episodes = [self.episodes[i] for i in ep_ids]
            phases = [self.phases[i] for i in ep_ids]
            t0 = time.time()
            res = evaluate_set(self.sim_cfg, main_cfg, episodes, phases)
            res.pop("_recs", None)
            res.pop("_per_ep", None)
            res["eval_seconds"] = round(time.time() - t0, 2)
            out["splits"][sp] = res
        train = out["splits"].get("train", out["splits"][list(out["splits"])[0]])
        out["fails"] = hard_gates(train, self.base)
        out["score"] = improvement_score(train, self.base)
        out["gate_pass"] = not out["fails"]
        self.evaluated[key] = out
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        return out


def fmt(v):
    if isinstance(v, float):
        return round(v, 4)
    return v
