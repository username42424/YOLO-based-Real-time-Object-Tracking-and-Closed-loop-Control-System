# -*- coding: utf-8 -*-
"""分阶段 unit 参数与跟踪质量寻优。

阶段(每阶段独立结果文件, 后阶段固定前阶段最优):
  stage1  12维: 识别节拍unit.frame_ms, 缓出分配周期unit.move_steps, far/near_gain,
          near_radius, deadzone, stop_deadzone, target_filter_alpha,
          锁框平滑、横向增益软区、reversal_damp、unit_max_counts
  stage2   5维: missing_decay, missing_max_counts, target_lock_distance,
          target_lock_iou, lock_ambiguity_margin
  stage3  10维: 预测6参数(prediction_min/max_ms, box_ratio, cap_px, lock_frames,
          vel_tau) + px_per_count(x/y, 当前值0.75~1.25倍)
  final   Top8 在独立验证集(60回合)复测 + 与当前生产配置对比

约束: stop_deadzone < deadzone; near_gain <= far_gain;
      prediction_min_ms < prediction_max_ms (违反时自动修复)。
评价指标: 首次进入目标框内60%区域、进入失败率、进入后60%区域驻留率、
      真实超时率，以及低权重的反向/过冲/丢失移动安全项。chest_ratio 固定取
      生产配置值，不允许搜索器通过改变瞄点来迎合几何评分。

预算: 6小时版 stage1 170min, stage2 80min, stage3 80min, final 其余。
进度: results/staged_v6/staged_sweep_progress.json (每15s刷新)。
结果: results/staged_v6/stage{1,2,3}_results.jsonl / stage{1,2,3}_best.json /
      final_validation.json / final_report.json
断点续跑: 重跑时自动跳过 jsonl 中已评估参数组。
复现: 固定回合池(seed42 抽140回合搜索 / seed77 抽60回合验证, 互斥), 每组
      逐回合 seed=回合号, 多worker仅并行不影响结果。
"""
import copy
import json
import math
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT_CFG = os.path.join(os.path.dirname(HERE), "config.json")
RESULTS = os.path.join(HERE, "results", "staged_v8_tracking")
PROGRESS = os.path.join(RESULTS, "staged_sweep_progress.json")

SEARCH_EPS_N = 10 ** 6   # 用满全部训练回合(248, holdout 日志除外)
HOLDOUT_EPS_N = 10 ** 6  # 用满全部 holdout 回合(aim_20260904_220626 的 21 个)
# 6小时版时间预算(与 --workers 1 配合): stage1 170 + stage2 80 + stage3 80,
# 剩余给基线/终评/扩界。组数上限仅作保险。
STAGE_BUDGET_MIN = {"stage1": 170, "stage2": 80, "stage3": 80}
STAGE_GROUP_CAP = {"stage1": 120000, "stage2": 80000, "stage3": 100000}

# (键, lo, hi, 是否取整) — 每阶段独立空间
STAGE_SPACE = {
    "stage1": [
        ("unit.frame_ms", 40.0, 100.0, False),
        ("unit.move_steps", 1.0, 3.0, True),
        ("unit.far_gain", 0.70, 1.05, False),
        ("unit.near_gain", 0.30, 0.75, False),
        ("unit.near_radius_px", 15.0, 60.0, False),
        ("unit.deadzone", 3.0, 12.0, False),
        ("unit.stop_deadzone", 0.30, 1.50, False),
        ("unit.target_filter_alpha", 0.15, 0.60, False),
        ("unit.gain_softness_px_x", 1.0, 30.0, False),
        ("aim_control.reversal_damp", 0.40, 0.90, False),
        ("aim_control.unit_max_counts", 180.0, 360.0, False),
    ],
    "stage2": [
        ("aim_control.missing_decay", 0.25, 0.70, False),
        ("aim_control.missing_max_counts", 8.0, 40.0, False),
        ("target_lock_distance", 60.0, 140.0, False),
        ("target_lock_iou", 0.02, 0.25, False),
        ("lock_ambiguity_margin", 0.10, 0.35, False),
    ],
    "stage3": [
        ("aim_control.prediction_min_ms", 20.0, 60.0, False),
        ("aim_control.prediction_max_ms", 90.0, 160.0, False),
        ("aim_control.prediction_box_ratio", 0.60, 0.90, False),
        ("aim_control.prediction_cap_px", 20.0, 60.0, False),
        ("aim_control.prediction_lock_frames", 1.0, 4.0, True),
        ("aim_control.prediction_vel_tau", 0.05, 0.25, False),
        ("unit.px_per_count", 0.3011 * 0.75, 0.3011 * 1.25, False),
        ("unit.px_per_count_y", 0.3011 * 0.75, 0.3011 * 1.25, False),
    ],
}
# 硬限(边界扩展也不得越过用户给定范围)
STAGE_HARD = {k: (lo, hi) for k, (lo, hi) in
              ((s[0], (s[1], s[2])) for sp in STAGE_SPACE.values() for s in sp)}

_G = {}


def _min_frame_ms():
    """搜索器计算出的最小可辨识 frame_ms; 未知时返回 0(不裁剪)。"""
    try:
        return float(_G.get("min_resolvable_frame_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def repair(stage, p):
    """阶段内约束修复(在采样向量转参数后调用)。"""
    if stage == "stage1":
        # 日志观测周期(~60ms)决定可辨识下限: 低于下限的 frame_ms 直接裁到下限,
        # 不让 40/45ms 这类日志无法分辨的值以偶然高分进入候选。
        mf = _min_frame_ms()
        if mf > 0.0 and float(p.get("unit.frame_ms", 0.0)) < mf:
            p["unit.frame_ms"] = round(mf, 3)
        if p["unit.stop_deadzone"] >= p["unit.deadzone"] - 0.5:
            p["unit.stop_deadzone"] = max(0.30, p["unit.deadzone"] - 0.5)
        if p["unit.near_gain"] > p["unit.far_gain"]:
            p["unit.near_gain"] = p["unit.far_gain"]
    if stage == "stage3":
        if p["aim_control.prediction_min_ms"] >= p["aim_control.prediction_max_ms"] - 5:
            p["aim_control.prediction_min_ms"] = p["aim_control.prediction_max_ms"] - 5
    return p


def priority_score(m):
    """Tracking score; simulated kills and shot cadence are intentionally absent."""
    if m.get("error"):
        return 1e9
    return (
        3.0 * m["first_inner60_median"]
        + 10.0 * m["inner60_entry_failure_rate"]
        + 8.0 * (1.0 - m["target_dwell_rate"])
        + 12.0 * m["timeout_rate"]
        + 50.0 * float(m.get("frame_ms_unresolvable", False))
        + 3.0 * m["flip_rate"]                     # 大幅反向惩罚
        + 0.15 * m["overshoot_per_ep"]             # 过冲惩罚
        + 0.004 * m["lost_move_per_ep"]            # 丢失目标错误移动惩罚
        + 1.0 * m.get("mouse_fail_per_ep", 0.0)    # 鼠标输出失败惩罚
    )


def acceptance_failures(m, eval_cfg):
    """返回未满足的硬门槛；空列表表示可作为最终冠军。"""
    checks = (
        ("first_inner60", m["first_inner60_median"], "<=",
         eval_cfg.get("first_inner60_max", eval_cfg.get("first_hit_max", 0.3))),
        ("inner60_entry_failure", m["inner60_entry_failure_rate"], "<=",
         eval_cfg.get("inner60_entry_failure_max", 0.25)),
        ("target_dwell_rate", m.get("target_dwell_rate", m.get("dwell_reddot", 0.0)),
         ">=", eval_cfg.get("dwell_min", 0.3)),
        ("timeout", m["timeout_rate"], "<=", eval_cfg.get("timeout_max", 0.05)),
    )
    failures = [name for name, value, op, limit in checks
                if (value > limit if op == "<=" else value < limit)]
    if m.get("frame_ms_unresolvable"):
        failures.append("frame_ms_unresolvable")
    return failures


def _patch(cfg_main, params):
    for dotted, v in params.items():
        parts = dotted.split(".")
        node = cfg_main
        for pp in parts[:-1]:
            node = node[pp]
        node[parts[-1]] = v
    return cfg_main


def _init_worker():
    from config import load_config
    from logger import SilentLogger
    import run_sim
    import replay as replay_mod
    cfg = load_config(os.path.join(HERE, "config.json"))
    base_main = json.load(open(ROOT_CFG, encoding="utf-8"))
    episodes = replay_mod.load_episodes(cfg["replay"])
    n_eps = len(episodes)
    holdout_names = {os.path.basename(p).lower()
                     for p in cfg["replay"].get("holdout_logs", [])}
    holdout_pool = [i for i, ep in enumerate(episodes)
                    if os.path.basename(ep.src).lower() in holdout_names]
    search_pool = [i for i in range(n_eps) if i not in set(holdout_pool)]
    if not holdout_pool:
        search_pool = sorted(random.Random(42).sample(
            range(n_eps), min(SEARCH_EPS_N, n_eps)))
        search_set = set(search_pool)
        holdout_pool = [i for i in range(n_eps) if i not in search_set]
    search_eps = sorted(random.Random(42).sample(
        search_pool, min(SEARCH_EPS_N, len(search_pool))))
    holdout_eps = sorted(random.Random(77).sample(
        holdout_pool, min(HOLDOUT_EPS_N, len(holdout_pool))))
    _G["cfg"] = cfg
    _G["base_main"] = base_main
    _G["search_eps"] = search_eps
    _G["holdout_eps"] = holdout_eps
    _G["max_s"] = float(cfg["episode"]["max_seconds"])
    _G["eval"] = cfg.get("eval", {})
    # 日志本身不能创造更密集的截图。用训练集观测周期 P10 的90%作保守
    # 下界，避免把 40~45ms 这类日志无法分辨的数值误当精确最优。
    source_periods_ms = sorted(episodes[i].iter * 1000.0 for i in search_eps)
    p10 = source_periods_ms[int(0.10 * (len(source_periods_ms) - 1))]
    _G["min_resolvable_frame_ms"] = round(p10 * 0.90, 3)
    _G["run_episode"] = run_sim.run_episode
    _G["SilentLogger"] = SilentLogger


class _EpStats:
    """逐回合累计: 过冲/丢失错误移动/大幅反向, 经 command_hook(dx,dy,out)。"""
    __slots__ = ("last_move", "last_any", "pairs", "flips", "overshoot",
                 "lost_move", "mouse_fail", "move_count", "step_sum",
                 "peak_step", "delta_pairs", "delta_sq", "micro_pairs",
                 "micro_delta_sq")

    def __init__(self):
        self.last_move = None
        self.last_any = None
        self.pairs = 0
        self.flips = 0
        self.overshoot = 0
        self.lost_move = 0.0
        self.mouse_fail = 0
        self.move_count = 0
        self.step_sum = 0.0
        self.peak_step = 0.0
        self.delta_pairs = 0
        self.delta_sq = 0.0
        self.micro_pairs = 0
        self.micro_delta_sq = 0.0

    def feed(self, dx, dy, out):
        mag = math.hypot(dx, dy)
        if mag > 0.0:
            self.move_count += 1
            self.step_sum += mag
            self.peak_step = max(self.peak_step, mag)
            if self.last_any is not None:
                delta_sq = ((dx - self.last_any[0]) ** 2
                            + (dy - self.last_any[1]) ** 2)
                self.delta_pairs += 1
                self.delta_sq += delta_sq
                if mag < 20.0 and math.hypot(*self.last_any) < 20.0:
                    self.micro_pairs += 1
                    self.micro_delta_sq += delta_sq
            self.last_any = (dx, dy)
        # 丢失目标后的错误移动: 引擎在丢失宽限期内沿预测继续发送(prediction_reason
        # = missing_decay_N); 真正无目标时引擎 can_send=False 不会发送。
        if out is not None and str(out.get("prediction_reason", "")).startswith(
                "missing_decay"):
            self.lost_move += mag
        if mag >= 20.0:
            if self.last_move is not None:
                self.pairs += 1
                if dx * self.last_move[0] + dy * self.last_move[1] < 0.0:
                    self.flips += 1
                    if math.hypot(*self.last_move) >= 60.0:
                        self.overshoot += 1
            self.last_move = (dx, dy)


def run_group(task):
    gid, stage, params, ep_ids = task
    t0 = time.time()
    run_episode = _G["run_episode"]
    SilentLogger = _G["SilentLogger"]
    cfg = _G["cfg"]
    max_s = _G["max_s"]
    cfg_main = _patch(copy.deepcopy(_G["base_main"]), params)
    box_entries, inner_entries = [], []
    dwell_rd_s = dwell_ch_s = observed_s = above_s = 0.0
    post_inner_s = post_observed_s = 0.0
    to = censored = exhausted = box_timeouts = 0
    flips = pairs = overshoot = 0
    move_count = delta_pairs = micro_pairs = 0
    step_sum = delta_sq = micro_delta_sq = peak_step = 0.0
    lost_move = 0.0
    mouse_fail = 0
    timeout_eps = []
    evaluated = 0
    for ep_idx in ep_ids:
        st = _EpStats()

        def _hook(dx, dy, out=None, _st=st):
            _st.feed(dx, dy, out)
        try:
            sm = run_episode(cfg, "auto", ep_idx, renderer=None,
                             logger=SilentLogger(), quiet=True,
                             main_cfg_path=cfg_main, send_hook=_hook)
        except Exception:
            continue
        evaluated += 1
        if sm.get("true_timeout", False):
            to += 1
            timeout_eps.append(ep_idx)
            if sm.get("first_box_entry_time") is None:
                box_timeouts += 1
        if sm.get("entry_censored", False):
            censored += 1
        if sm.get("replay_exhausted"):
            exhausted += 1
        first_box = sm.get("first_box_entry_time")
        if first_box is not None:
            box_entries.append(first_box)
        inner = sm.get("first_inner60_time")
        if inner is not None:
            inner_entries.append(inner)
        dwell_rd_s += sm.get("dwell_reddot_seconds", 0.0)
        dwell_ch_s += sm.get("dwell_cross_seconds", 0.0)
        post_inner_s += sm.get("post_entry_inner60_seconds", 0.0)
        post_observed_s += sm.get("post_entry_observed_seconds", 0.0)
        above_s += sm.get("above_box_seconds", 0.0)
        observed_s += sm.get("replay_observed_seconds", 0.0)
        flips += st.flips
        pairs += st.pairs
        overshoot += st.overshoot
        lost_move += st.lost_move
        mouse_fail += st.mouse_fail
        move_count += st.move_count
        step_sum += st.step_sum
        peak_step = max(peak_step, st.peak_step)
        delta_pairs += st.delta_pairs
        delta_sq += st.delta_sq
        micro_pairs += st.micro_pairs
        micro_delta_sq += st.micro_delta_sq
    n = evaluated
    if n == 0:
        return gid, stage, params, {"error": "all_failed"}, 1e9, time.time() - t0, []
    first_box_median = statistics.median(box_entries) if box_entries else max_s
    first_inner_median = statistics.median(inner_entries) if inner_entries else max_s
    entry_trials = len(inner_entries) + to
    box_trials = len(box_entries) + box_timeouts
    m = {
        "first_box_entry_median": round(first_box_median, 4),
        "first_box_entry_success_rate": (
            round(len(box_entries) / box_trials, 4) if box_trials else 0.0),
        "first_inner60_median": round(first_inner_median, 4),
        "inner60_entry_failure_rate": (
            round(to / entry_trials, 4) if entry_trials else 1.0),
        "entry_trials": entry_trials,
        "entry_censored_rate": round(censored / n, 4),
        "target_dwell_rate": (
            round(post_inner_s / post_observed_s, 4) if post_observed_s else 0.0),
        "dwell_reddot": round(dwell_rd_s / observed_s, 4) if observed_s else 0.0,
        "dwell_cross": round(dwell_ch_s / observed_s, 4) if observed_s else 0.0,
        "above_box_ratio": round(above_s / observed_s, 4) if observed_s else 0.0,
        "observed_seconds": round(observed_s, 3),
        "timeout_rate": round(to / entry_trials, 4) if entry_trials else 1.0,
        "replay_exhaustion_rate": round(exhausted / n, 4),
        "flip_rate": round(flips / pairs, 4) if pairs else 0.0,
        "overshoot_per_ep": round(overshoot / n, 3),
        "lost_move_per_ep": round(lost_move / n, 1),
        "mouse_fail_per_ep": round(mouse_fail / n, 3),
        "mean_step_counts": round(step_sum / move_count, 3) if move_count else 0.0,
        "peak_step_counts": round(peak_step, 3),
        "command_delta_rms_counts": (
            round(math.sqrt(delta_sq / delta_pairs), 3) if delta_pairs else 0.0),
        "micro_jitter_rms_counts": (
            round(math.sqrt(micro_delta_sq / micro_pairs), 3) if micro_pairs else 0.0),
        "n": n,
        "failed_runs": len(ep_ids) - n,
    }
    frame_ms = float(params.get("unit.frame_ms", _G["base_main"]["unit"].get("frame_ms", 0)))
    m["min_resolvable_frame_ms"] = _G["min_resolvable_frame_ms"]
    m["frame_ms_unresolvable"] = frame_ms < _G["min_resolvable_frame_ms"]
    m["acceptance_failures"] = acceptance_failures(m, _G["eval"])
    m["accepted"] = not m["acceptance_failures"]
    return gid, stage, params, m, round(priority_score(m), 4), \
        round(time.time() - t0, 2), timeout_eps


# ── 采样: LHS + 独立维 TPE (与 sweep_bayes 同思路) ──
def make_lhs(n, seed, bounds):
    rng = random.Random(seed)
    d = len(bounds)
    cols = []
    for j in range(d):
        perm = list(range(n))
        rng.shuffle(perm)
        cols.append([(perm[i] + rng.random()) / n for i in range(n)])
    return [np.array([bounds[j][0] + cols[j][i] * (bounds[j][1] - bounds[j][0])
                      for j in range(d)]) for i in range(n)]


class TPE:
    def __init__(self, bounds, seed):
        self.bounds = bounds
        self.rng = random.Random(seed)
        self.X, self.S = [], []
        self._bw = (bounds[:, 1] - bounds[:, 0]) * 0.08

    def tell(self, v, score):
        self.X.append(np.asarray(v, float))
        self.S.append(float(score))

    def _dens(self, row, pts):
        z = (row[None, :] - pts) / self._bw[None, :]
        return (np.exp(-0.5 * z * z) / (self._bw[None, :] * 2.50662827)).mean(axis=0)

    def suggest(self, k=32, n_cand=48):
        if len(self.S) < 16:
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            return [lo + np.array([self.rng.random() for _ in range(len(lo))]) * (hi - lo)
                    for _ in range(k)]
        X, S = np.array(self.X), np.array(self.S)
        order = np.argsort(S)
        gamma = max(8, int(len(S) * 0.25))
        topX, restX = X[order[:gamma]], X[order[gamma:]]
        out = []
        for _ in range(k):
            best_c, best_s = None, None
            for _c in range(n_cand):
                base = topX[self.rng.randrange(len(topX))]
                cand = np.clip(base + np.array([self.rng.gauss(0, 1) for _ in range(len(base))])
                               * self._bw, self.bounds[:, 0], self.bounds[:, 1])
                l = float(np.sum(np.log(np.maximum(1e-12, self._dens(cand, topX)))))
                g = float(np.sum(np.log(np.maximum(1e-12, self._dens(cand, restX)))))
                s = l - g
                if best_s is None or s > best_s:
                    best_s, best_c = s, cand
            out.append(best_c)
        return out


def vec_to_params(stage, space, v):
    p = {}
    for (k, lo, hi, isint), val in zip(space, v):
        p[k] = int(round(val)) if isint else round(float(val), 4)
    return repair(stage, p)


def write_progress(**kw):
    kw.setdefault("updated", time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PROGRESS)


def _result_iter(pool, tasks):
    return map(run_group, tasks) if pool is None else pool.imap_unordered(run_group, tasks)


def load_history(path):
    recs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("score") is not None:
                    recs.append(r)
    return recs


def run_stage(stage, pool, best, t_deadline, group_cap, gid_base):
    space = STAGE_SPACE[stage]
    bounds = np.array([[s[1], s[2]] for s in space], dtype=float)
    out_jsonl = os.path.join(RESULTS, "%s_results.jsonl" % stage)
    seen = set()
    history = []
    for r in load_history(out_jsonl):
        key = json.dumps(r["params"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        history.append((r["params"], r["score"]))
    base_params = {}
    for k, v in (best.get("stage1") or {}).items():
        base_params[k] = v
    for k, v in (best.get("stage2") or {}).items():
        base_params[k] = v
    tpe = TPE(bounds, seed=20260904 + len(stage))
    for p, s in history:
        tpe.tell(np.array([float(p[k]) for k, _, _, _ in space]), s)
    n_done = len(history)
    stage_start = time.time()
    write_progress(stage=stage, stage_groups=n_done, stage_cap=group_cap,
                   stage_best=min(history, key=lambda t: t[1]) if history else None,
                   stage_elapsed_min=0.0, phase=stage)
    print("[%s] 续跑 %d 组历史" % (stage, n_done), flush=True)

    def evaluate(plist, tag):
        nonlocal n_done
        tasks = []
        for i, p in enumerate(plist):
            full = dict(base_params)
            full.update(p)
            key = json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tasks.append((gid_base + n_done + i, stage, full, _G["search_eps"]))
        got = []
        for gid, st, full_p, m, score, dur, to_eps in _result_iter(pool, tasks):
            search_p = {k: v for k, v in full_p.items() if k in dict(
                ((s[0], None) for s in space))}
            rec = {"gid": gid, "stage": stage, "params": full_p, "metrics": m,
                   "score": score, "dur_s": dur, "timeout_eps": to_eps[:8]}
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if m.get("error"):
                continue
            history.append((full_p, score))
            tpe.tell(np.array([float(search_p.get(k, full_p[k]))
                               for k, _, _, _ in space]), score)
            if score < best["score"]:
                best.update({"score": score, "params": full_p, "metrics": m,
                             "stage": stage})
            got.append((full_p, score))
        return got

    # LHS 粗搜
    lhs_n = {"stage1": 600, "stage2": 300, "stage3": 350}[stage]
    lhs_n = min(lhs_n, group_cap)
    plist = [vec_to_params(stage, space, v) for v in make_lhs(lhs_n, 20260904, bounds)]
    got = evaluate(plist, "lhs")
    print("[%s] LHS %d 组, 最优 %.4f, 用时 %.1fmin"
          % (stage, len(got), best["score"], (time.time() - stage_start) / 60), flush=True)

    # TPE 精搜直到时间/组数预算; 预留 15% 阶段时间给贴边扩界重搜
    # (否则 TPE 会耗尽预算, 扩界永远不会执行——v2~v5 实测均如此)
    reserve_s = 0.15 * STAGE_BUDGET_MIN[stage] * 60
    t_tpe = t_deadline - reserve_s
    rnd = 0
    while time.time() < t_tpe and n_done < group_cap:
        cands = tpe.suggest(k=min(32, group_cap - n_done))
        plist = [vec_to_params(stage, space, v) for v in cands]
        evaluate(plist, "bo")
        rnd += 1
        if rnd % 5 == 0:
            st_best = min(history, key=lambda t: t[1]) if history else None
            elapsed = (time.time() - stage_start) / 60
            write_progress(stage=stage, stage_groups=n_done, stage_cap=group_cap,
                           stage_elapsed_min=round(elapsed, 1),
                           stage_best=({"params": st_best[0], "score": st_best[1]}
                                       if st_best else None),
                           overall_best=best, phase=stage)
            print("[%s] %d 组, 阶段用时 %.1fmin, 总最优 %.4f"
                  % (stage, n_done, elapsed, best["score"]), flush=True)

    st_best = min(history, key=lambda t: t[1])
    # 边界贴近检查
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng_ = hi - lo
    vec = np.array([float(st_best[0][k]) for k, _, _, _ in space])
    near = [space[i][0] for i in range(len(space))
            if vec[i] <= lo[i] + 0.05 * rng_[i] or vec[i] >= hi[i] - 0.05 * rng_[i]]
    # 边界扩展: 贴边维按用户硬限内扩 15% 幅宽, 重搜一批
    expanded = False
    if near and time.time() < t_deadline:
        expanded = True
        lo2, hi2 = lo.copy(), hi.copy()
        for i, s in enumerate(space):
            if s[0] in near:
                hard_lo, hard_hi = STAGE_HARD[s[0]]
                lo2[i] = max(hard_lo, lo[i] - 0.15 * rng_[i])
                hi2[i] = min(hard_hi, hi[i] + 0.15 * rng_[i])
        tpe2 = TPE(np.stack([lo2, hi2], axis=1), seed=555)
        for p, s in history:
            tpe2.tell(np.array([float(p[k]) for k, _, _, _ in space]), s)
        ext = 0
        while time.time() < t_deadline and ext < 2000:
            cands_raw = tpe2.suggest(k=32)
            evaluate_ext(cands_raw, tpe2, stage, space, out_jsonl, base_params,
                         best, history, seen, gid_base, n_done)
            ext += 32
        st_best = min(history, key=lambda t: t[1])
        vec = np.array([float(st_best[0][k]) for k, _, _, _ in space])
        near = [space[i][0] for i in range(len(space))
                if vec[i] <= lo2[i] + 0.05 * (hi2[i] - lo2[i])
                or vec[i] >= hi2[i] - 0.05 * (hi2[i] - lo2[i])]

    best_p, best_s = st_best
    # 从 jsonl 找回最优组的完整 metrics
    full_m = None
    for r in load_history(out_jsonl):
        if r["score"] == best_s:
            full_m = r["metrics"]
            break
    stage_best_doc = {
        "stage": stage, "search_space": [{"key": k, "lo": l, "hi": h}
                                         for k, l, h, _ in space],
        "best_params": best_p, "score": best_s, "metrics": full_m,
        "groups": n_done, "boundary_keys": near,
        "boundary_expanded": expanded,
    }
    with open(os.path.join(RESULTS, "%s_best.json" % stage), "w", encoding="utf-8") as f:
        json.dump(stage_best_doc, f, ensure_ascii=False, indent=1)
    best["stage" + stage[-1]] = best_p
    write_progress(stage=stage, stage_groups=n_done, phase=stage + "_done",
                   overall_best=best)
    print("[%s] 完成: %d 组, 最优 %.4f, 贴边参数 %s"
          % (stage, n_done, best_s, near or "无"), flush=True)
    return best_p


def evaluate_ext(plist, tpe2, stage, space, out_jsonl, base_params, best,
                 history, seen, gid_base, n_done):
    """边界扩展后的 TPE 批次评估(允许向量超出原界)。"""
    tasks = []
    for i, v in enumerate(plist):
        p = {}
        for (k, l, h, isint), val in zip(space, v):
            p[k] = int(round(val)) if isint else round(float(val), 4)
        p = repair(stage, p)
        full = dict(base_params)
        full.update(p)
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        tasks.append((gid_base + 500000 + n_done + i, stage, full, _G["search_eps"]))
    for gid, st, full_p, m, score, dur, to_eps in _result_iter(pool_global, tasks):
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"gid": gid, "stage": stage, "params": full_p,
                                "metrics": m, "score": score, "dur_s": dur,
                                "timeout_eps": to_eps[:8]}, ensure_ascii=False) + "\n")
        if m.get("error"):
            continue
        history.append((full_p, score))
        tpe2.tell(np.array([float(full_p[k]) for k, _, _, _ in space]), score)
        if score < best["score"]:
            best.update({"score": score, "params": full_p, "metrics": m, "stage": stage})


pool_global = None


def main():
    global pool_global
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, min(12, (os.cpu_count() or 8))))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        STAGE_BUDGET_MIN.update({"stage1": 0.08, "stage2": 0.05, "stage3": 0.05})
        STAGE_GROUP_CAP.update({"stage1": 24, "stage2": 16, "stage3": 16})
    os.makedirs(RESULTS, exist_ok=True)
    _init_worker()
    best = {"score": 1e9, "params": None, "metrics": None, "stage": None}
    # 基线: 当前生产配置(不 patch)
    t0 = time.time()
    gid, st, p, m, score, dur, to_eps = run_group(
        (0, "baseline", {}, _G["search_eps"]))
    baseline = {"params": "当前生产配置(无patch)", "metrics": m, "score": score,
                "search_eps": _G["search_eps"], "holdout_eps": _G["holdout_eps"]}
    with open(os.path.join(RESULTS, "baseline_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=1)
    print("[基线] 当前配置 score=%.4f %s" % (score, json.dumps(m)), flush=True)

    pool = (None if args.workers <= 1 else
            Pool(processes=args.workers, initializer=_init_worker))
    pool_global = pool
    stage_seq = ["stage1", "stage2", "stage3"]
    if args.smoke:
        stage_seq = ["stage1"]
    for stage in stage_seq:
        deadline = time.time() + STAGE_BUDGET_MIN[stage] * 60
        run_stage(stage, pool, best, deadline, STAGE_GROUP_CAP[stage],
                  1000000 + 1000000 * stage_seq.index(stage))
        if time.time() - t0 > 5.4 * 3600:
            print("总时长接近6小时, 提前进入终评", flush=True)
            break
    # ── 终评: Top8 (跨阶段去重) 在独立验证集复测 ──
    all_hist = []
    for stage in stage_seq:
        for r in load_history(os.path.join(RESULTS, "%s_results.jsonl" % stage)):
            all_hist.append((stage, r["params"], r["score"], r["metrics"]))
    all_hist.sort(key=lambda t: t[2])
    finals, seen_p = [], set()
    for stage, p, s, m in all_hist:
        key = json.dumps(p, sort_keys=True)
        if key in seen_p:
            continue
        seen_p.add(key)
        finals.append((stage, p, s, m))
        if len(finals) >= 8:
            break
    tasks = [(9000000 + i, "final", p, _G["holdout_eps"])
             for i, (stage, p, s, m) in enumerate(finals)]
    rows = []
    for gid, st, p, m, score, dur, to_eps in _result_iter(pool, tasks):
        rows.append({"from_stage": st, "params": p, "holdout_metrics": m,
                     "holdout_score": score, "timeout_eps": to_eps})
        print("[终评] %s score=%.4f 首入60%%=%.3f 60%%驻留=%.1f%% 超时=%.1f%% %s"
              % (st, score, m.get("first_inner60_median", -1),
                 m.get("target_dwell_rate", 0) * 100,
                 m.get("timeout_rate", 0) * 100,
                 "通过" if m.get("accepted") else
                 "未通过:" + ",".join(m.get("acceptance_failures", []))),
              flush=True)
    # 基线在验证集
    gid, st, p, bm, bscore, dur, to_eps = run_group((9999999, "baseline", {},
                                                    _G["holdout_eps"]))
    rows.sort(key=lambda r: r["holdout_score"])
    passing_rows = [r for r in rows if r["holdout_metrics"].get("accepted")]
    champ = passing_rows[0] if passing_rows else None
    best_candidate = rows[0] if rows else None
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_groups": sum(1 for _ in all_hist),
        "baseline": {"search": baseline["metrics"], "holdout": bm,
                     "holdout_score": bscore},
        "stage_best": {st: json.load(open(os.path.join(
            RESULTS, "%s_best.json" % st), encoding="utf-8"))
            for st in stage_seq if os.path.exists(os.path.join(
                RESULTS, "%s_best.json" % st))},
        "final_ranking_holdout": rows,
        "champion": champ,
        "best_candidate": best_candidate,
        "episodes": {"search": _G["search_eps"], "holdout": _G["holdout_eps"]},
        "writeback": ("未写回生产 config.json (按用户要求, 需明确指令)"
                      if champ else
                      "拒绝产生冠军或写回：验证集没有参数组通过全部硬门槛"),
    }
    with open(os.path.join(RESULTS, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    write_progress(phase="all_done", overall_best=best,
                   total_min=round((time.time() - t0) / 60, 1))
    if pool is not None:
        pool.close()
        pool.join()
    print("全部完成: 总最优 %.4f (%s)" % (best["score"], best["stage"]), flush=True)
    if champ:
        print("验证集冠军: %s score=%.4f" % (champ["from_stage"],
                                            champ["holdout_score"]), flush=True)
    elif best_candidate:
        print("验证集无合格冠军；最低分候选仍未通过: %s"
              % ",".join(best_candidate["holdout_metrics"].get(
                  "acceptance_failures", [])), flush=True)


if __name__ == "__main__":
    main()
