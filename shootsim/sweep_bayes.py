# -*- coding: utf-8 -*-
"""三阶段 unit 参数寻优: LHS → TPE 贝叶斯精搜 → 最优邻域局部加密。

只搜索 aim_mode=unit 实际生效的参数，避免旧版把 smooth 专用参数当成有效变量。

每组 100 个固定种子(回合池随机抽 100, 全程配对可比); 终评 Top10 在全部回合上复核。
综合分采用与 sweep_staged.py 相同的修正版控制质量指标。日志自然结束不算超时。

用法:
    python sweep_bayes.py                 # 全量: LHS 800 + BO 3000 + 局部 8000 + 终评
    python sweep_bayes.py --smoke         # 冒烟: 小预算验证全流程
结果: results/bayes_unit_v7_results.jsonl / bayes_unit_v7_progress.json / bayes_unit_v7_final.json
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
from sweep_staged import acceptance_failures, priority_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_JSONL = os.path.join(HERE, "results", "bayes_unit_v7_results.jsonl")
PROGRESS = os.path.join(HERE, "results", "bayes_unit_v7_progress.json")
FINAL = os.path.join(HERE, "results", "bayes_unit_v7_final.json")

# (键, lo, hi, 是否取整)
SPACE = [
    ("chest_ratio", 0.05, 0.40, False),
    ("unit.far_gain", 0.75, 1.00, False),
    ("unit.near_gain", 0.25, 0.55, False),
    ("unit.near_radius_px", 10.0, 35.0, False),
    ("unit.deadzone", 5.0, 12.0, False),
    ("unit.stop_deadzone", 0.3, 1.5, False),
    ("unit.target_filter_alpha", 0.15, 0.60, False),
    ("aim_control.reversal_damp", 0.50, 0.95, False),
    ("aim_control.unit_max_counts", 200.0, 650.0, False),
    ("aim_control.missing_decay", 0.20, 0.70, False),
    ("aim_control.missing_max_counts", 8.0, 40.0, False),
    ("target_lock_distance", 45.0, 140.0, False),
    ("target_lock_iou", 0.05, 0.50, False),
    ("lock_ambiguity_margin", 0.05, 0.30, False),
]
NDIM = len(SPACE)
BOUNDS = np.array([[s[1], s[2]] for s in SPACE], dtype=float)

_G = {}


def _init_worker():
    from config import load_config
    from logger import SilentLogger
    import run_sim
    import replay as replay_mod
    cfg = load_config(os.path.join(HERE, "config.json"))
    base_main = json.load(open(os.path.join(os.path.dirname(HERE), "config.json"),
                               encoding="utf-8"))
    episodes = replay_mod.load_episodes(cfg["replay"])
    _G["cfg"] = cfg
    _G["base_main"] = base_main
    _G["episodes"] = episodes
    _G["n_eps"] = len(episodes)
    _G["max_s"] = float(cfg["episode"]["max_seconds"])
    _G["eval"] = cfg.get("eval", {})
    _G["run_episode"] = run_sim.run_episode
    _G["SilentLogger"] = SilentLogger


def _patch(cfg_main, params):
    for dotted, v in params.items():
        parts = dotted.split(".")
        node = cfg_main
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return cfg_main


def run_group(task):
    gid, stage, params, ep_ids = task
    t0 = time.time()
    run_episode = _G["run_episode"]
    SilentLogger = _G["SilentLogger"]
    cfg = _G["cfg"]
    max_s = _G["max_s"]
    cfg_main = _patch(copy.deepcopy(_G["base_main"]), params)
    kills, fhs, inner_entries = [], [], []
    dwell_rd_s = dwell_ch_s = observed_s = above_s = 0.0
    hits = shots = to = exhausted = flips = flip_pairs = evaluated = 0
    for ep_idx in ep_ids:
        last_move = [None]
        ep_flips = [0]
        ep_pairs = [0]

        def _send_hook(dx, dy):
            mag = math.hypot(dx, dy)
            if mag < 20.0:
                return
            if last_move[0] is not None:
                ep_pairs[0] += 1
                if dx * last_move[0][0] + dy * last_move[0][1] < 0.0:
                    ep_flips[0] += 1
            last_move[0] = (dx, dy)
        try:
            sm = run_episode(cfg, "auto", ep_idx, renderer=None,
                             logger=SilentLogger(), quiet=True,
                             main_cfg_path=cfg_main, send_hook=_send_hook)
        except Exception:
            continue
        evaluated += 1
        if sm.get("timeout"):
            to += 1
        if sm.get("replay_exhausted"):
            exhausted += 1
        if sm["kill_time"] is not None:
            kills.append(sm["kill_time"])
        fh = sm.get("first_hit_time")
        if fh is not None:
            fhs.append(fh)
        inner = sm.get("first_inner60_time")
        if inner is not None:
            inner_entries.append(inner)
        dwell_rd_s += sm.get("dwell_reddot_seconds", 0.0)
        dwell_ch_s += sm.get("dwell_cross_seconds", 0.0)
        above_s += sm.get("above_box_seconds", 0.0)
        observed_s += sm.get("replay_observed_seconds", 0.0)
        hits += sm["hits"]
        shots += sm["shots"]
        flips += ep_flips[0]
        flip_pairs += ep_pairs[0]
    n = evaluated
    if n == 0:
        return gid, stage, params, {"error": "all_failed"}, None, time.time() - t0
    metrics = {
        "median": round(statistics.median(kills), 4) if kills else None,
        "kill_completion_rate": round(len(kills) / n, 4),
        "first_hit_median": round(statistics.median(fhs), 4) if fhs else max_s,
        "first_hit_success_rate": round(len(fhs) / n, 4),
        "first_inner60_median": (round(statistics.median(inner_entries), 4)
                                  if inner_entries else max_s),
        "inner60_entry_failure_rate": round(1.0 - len(inner_entries) / n, 4),
        "dwell_reddot": round(dwell_rd_s / observed_s, 4) if observed_s else 0.0,
        "dwell_cross": round(dwell_ch_s / observed_s, 4) if observed_s else 0.0,
        "above_box_ratio": round(above_s / observed_s, 4) if observed_s else 0.0,
        "observed_seconds": round(observed_s, 3),
        "hit_rate": round(hits / shots, 4) if shots else 0.0,
        "timeout_rate": round(to / n, 4),
        "replay_exhaustion_rate": round(exhausted / n, 4),
        "flip_rate": round(flips / flip_pairs, 4) if flip_pairs else 0.0,
        "overshoot_per_ep": 0.0,
        "lost_move_per_ep": 0.0,
        "mouse_fail_per_ep": 0.0,
        "n": n,
        "failed_runs": len(ep_ids) - n,
    }
    metrics["acceptance_failures"] = acceptance_failures(metrics, _G["eval"])
    metrics["accepted"] = not metrics["acceptance_failures"]
    score = priority_score(metrics)
    return gid, stage, params, metrics, round(score, 4), time.time() - t0


# ── 参数向量 ↔ 字典 ──
def vec_to_params(v):
    p = {}
    for (k, lo, hi, isint), val in zip(SPACE, v):
        p[k] = int(round(val)) if isint else round(float(val), 4)
    return p


def params_to_vec(p):
    return np.array([float(p[k]) for k, _, _, _ in SPACE], dtype=float)


def clip_vec(v):
    return np.minimum(np.maximum(v, BOUNDS[:, 0]), BOUNDS[:, 1])


class TPE:
    """独立维 TPE(与 optuna 默认一致的建模思路): top KDE 采样, l/g 比值选点。"""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.X = []      # np vectors
        self.S = []      # scores
        self._bw = (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.08

    def tell(self, v, score):
        self.X.append(np.asarray(v, dtype=float))
        self.S.append(float(score))

    def _dens(self, row, pts, bw):
        z = (row[None, :] - pts) / bw[None, :]
        return (np.exp(-0.5 * z * z) / (bw[None, :] * 2.50662827)).mean(axis=0)

    def suggest(self, k=32, n_cand=64):
        n = len(self.S)
        if n < 16:
            return [clip_vec(BOUNDS[:, 0] + np.array([self.rng.random() for _ in range(NDIM)])
                             * (BOUNDS[:, 1] - BOUNDS[:, 0]))
                    for _ in range(k)]
        X = np.array(self.X)
        S = np.array(self.S)
        order = np.argsort(S)
        gamma = max(8, int(n * 0.25))
        topX, restX = X[order[:gamma]], X[order[gamma:]]
        cands = []
        for _ in range(k):
            best_c, best_s = None, None
            for _c in range(n_cand):
                base = topX[self.rng.randrange(len(topX))]
                cand = np.clip(base + np.array([self.rng.gauss(0, 1) for _ in range(NDIM)])
                               * self._bw,
                               BOUNDS[:, 0], BOUNDS[:, 1])
                l = float(np.sum(np.log(np.maximum(1e-12, self._dens(cand, topX, self._bw)))))
                g = float(np.sum(np.log(np.maximum(1e-12, self._dens(cand, restX, self._bw)))))
                s = l - g
                # TPE 选择 l(x)/g(x) 最大的候选，即更像优良样本、较不像其余样本。
                if best_s is None or s > best_s:
                    best_s, best_c = s, cand
            cands.append(best_c)
        return cands


def make_lhs(n, seed):
    rng = random.Random(seed)
    cols = []
    for d in range(NDIM):
        perm = list(range(n))
        rng.shuffle(perm)
        cols.append([(perm[i] + rng.random()) / n for i in range(n)])
    pts = []
    for i in range(n):
        v = np.array([SPACE[d][1] + cols[d][i] * (SPACE[d][2] - SPACE[d][1])
                      for d in range(NDIM)])
        pts.append(v)
    return pts


def rand_in_box(lo, hi, rng, n):
    return [lo + rng.random() * (hi - lo) for _ in range(n)]


class _Store:
    results = []


def _save_jsonl(rec):
    _Store.results.append(rec)
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_progress(**kw):
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2)


def fmt_params(p):
    return ("chest=%.3f far=%.2f near=%.2f nr=%.1f dz=%.1f stop=%.2f "
            "filter=%.2f rev=%.2f cap=%.0f miss=%.2f/%.0f "
            "lock=%.0f iou=%.2f amb=%.2f"
            % (p["chest_ratio"], p["unit.far_gain"], p["unit.near_gain"],
                p["unit.near_radius_px"], p["unit.deadzone"],
                p["unit.stop_deadzone"], p["unit.target_filter_alpha"],
                p["aim_control.reversal_damp"], p["aim_control.unit_max_counts"],
                p["aim_control.missing_decay"], p["aim_control.missing_max_counts"],
                p["target_lock_distance"], p["target_lock_iou"],
                p["lock_ambiguity_margin"]))


def _result_iter(pool, tasks):
    """workers=1 时不创建 Windows 管道，便于受限环境/CI 冒烟验证。"""
    return map(run_group, tasks) if pool is None else pool.imap_unordered(run_group, tasks)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, min(12, (os.cpu_count() or 8))))
    ap.add_argument("--lhs", type=int, default=800)
    ap.add_argument("--bo", type=int, default=3000)
    ap.add_argument("--local", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.lhs, args.bo, args.local = 8, 12, 12

    _init_worker()
    n_eps = _G["n_eps"]
    episodes = _G["episodes"]
    holdout_names = {os.path.basename(p).lower()
                     for p in _G["cfg"]["replay"].get("holdout_logs", [])}
    all_eps = [i for i, ep in enumerate(episodes)
               if os.path.basename(ep.src).lower() in holdout_names]
    search_pool = [i for i in range(n_eps) if i not in set(all_eps)]
    if not all_eps:
        fixed_eps = sorted(random.Random(42).sample(range(n_eps), min(100, n_eps)))
        fixed_set = set(fixed_eps)
        all_eps = [i for i in range(n_eps) if i not in fixed_set]
    else:
        fixed_eps = sorted(random.Random(42).sample(search_pool,
                                                    min(100, len(search_pool))))
    print("回合池=%d 优化种子=%d 独立终评=%d 工作进程=%d" %
          (n_eps, len(fixed_eps), len(all_eps), args.workers),
          flush=True)

    t0 = time.time()
    groups_done = 0
    total_planned = args.lhs + args.bo + args.local
    tpe = TPE(seed=20260829)
    seen = set()
    history = []   # (params, score) 全历史
    pool = (None if args.workers <= 1 else
            Pool(processes=args.workers, initializer=_init_worker))

    def evaluate(params_list, stage, ep_ids, gid_base, pool):
        nonlocal groups_done, last_prog
        tasks = []
        for i, p in enumerate(params_list):
            key = json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tasks.append((gid_base + i, stage, p, ep_ids))
        out = []
        if not tasks:
            return out
        for gid, stage_, p, m, score, _s in _result_iter(pool, tasks):
            rec = {"gid": gid, "stage": stage_, "params": p, "metrics": m, "score": score}
            _save_jsonl(rec)
            groups_done += 1
            if m.get("error"):
                continue
            out.append((p, score))
            history.append((p, score))
            tpe.tell(params_to_vec(p), score)
            if score < best["score"]:
                best.update({"score": score, "params": p, "metrics": m})
            if time.time() - last_prog > 10:
                _progress(stage)
                last_prog = time.time()
        return out

    best = {"score": float("inf"), "params": None, "metrics": None}
    last_prog = 0.0

    # ── 断点续跑: 从历史 jsonl 恢复已评估组(不重复跑) ──
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("score") is None or r.get("stage") == "final":
                    continue
                key = json.dumps(r["params"], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                history.append((r["params"], r["score"]))
                tpe.tell(params_to_vec(r["params"]), r["score"])
                if r["score"] < best["score"]:
                    best.update({"score": r["score"], "params": r["params"],
                                 "metrics": r["metrics"]})
        groups_done = len(history)
        if groups_done:
            print("续跑: 已恢复 %d 组历史" % groups_done, flush=True)

    def _progress(stage):
        rate = groups_done / max(1e-6, time.time() - t0)
        write_progress(stage=stage, groups_done=groups_done, total=total_planned,
                       elapsed_min=round((time.time() - t0) / 60, 1),
                       groups_per_min=round(rate * 60, 1),
                       eta_min=round((total_planned - groups_done) / max(1e-9, rate) / 60, 1),
                       best=best, updated=time.strftime("%H:%M:%S"))

    # ── 阶段1: LHS ──
    pts = make_lhs(args.lhs, 20260829)
    plist = [vec_to_params(v) for v in pts]
    res1 = evaluate(plist, "lhs", fixed_eps, 1000000, pool)
    _progress("lhs_done")
    print("[LHS] %d 组完成, 当前最优 %.4f" % (len(res1), best["score"]), flush=True)

    # ── 阶段2: TPE 贝叶斯 ──
    bo_done = 0
    rnd = 0
    while bo_done < args.bo:
        k = min(32, args.bo - bo_done)
        cands = tpe.suggest(k=k)
        plist = [vec_to_params(v) for v in cands]
        res = evaluate(plist, "bo", fixed_eps, 2000000 + bo_done, pool)
        bo_done += len(res)
        rnd += 1
        if rnd % 10 == 0:
            _progress("bo")
            print("[BO] %d/%d  最优 %.4f" % (bo_done, args.bo, best["score"]), flush=True)
    _progress("bo_done")
    print("[BO] 完成, 最优 %.4f" % best["score"], flush=True)

    # ── 阶段3: 局部加密 ──
    def local_stage(n_groups, topk, stage, gid_base):
        hist_sorted = sorted(history, key=lambda t: t[1])
        top = hist_sorted[:topk]
        vecs = [params_to_vec(p) for p, _ in top]
        lo = clip_vec(np.min(vecs, axis=0) - 0.05 * (BOUNDS[:, 1] - BOUNDS[:, 0]))
        hi = clip_vec(np.max(vecs, axis=0) + 0.05 * (BOUNDS[:, 1] - BOUNDS[:, 0]))
        rng = random.Random(77001 if stage == "local_wide" else 77002)
        done = 0
        while done < n_groups:
            k = min(64, n_groups - done)
            plist = []
            for _ in range(k):
                v = np.array([rng.uniform(lo[d], hi[d]) for d in range(NDIM)])
                plist.append(vec_to_params(v))
            res = evaluate(plist, stage, fixed_eps, gid_base + done, pool)
            done += len(res)
            _progress(stage)
        print("[%s] 完成, 最优 %.4f" % (stage, best["score"]), flush=True)

    local_stage(args.local // 2, 20, "local_wide", 3000000)
    local_stage(args.local - args.local // 2, 5, "local_tight", 4000000)

    # ── 终评: Top10 全回合复核 ──
    hist_sorted = sorted(history, key=lambda t: t[1])
    seen_p = set()
    finals = []
    for p, s in hist_sorted:
        key = json.dumps(p, sort_keys=True)
        if key in seen_p:
            continue
        seen_p.add(key)
        finals.append(p)
        if len(finals) >= 10:
            break
    out = []
    tasks = [(9000000 + i, "final", p, all_eps) for i, p in enumerate(finals)]
    for gid, stage_, p, m, score, _s in _result_iter(pool, tasks):
        out.append({"rank": 0, "params": p, "metrics": m, "score": score})
        print("[终评] score=%.4f 首入60%%=%.3f 驻留=%.1f%% 偏上=%.1f%% %s"
              % (score, m["first_inner60_median"], m["dwell_reddot"] * 100,
                 m["above_box_ratio"] * 100,
                 "通过" if m["accepted"] else
                 "未通过:" + ",".join(m["acceptance_failures"])), flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    out.sort(key=lambda r: r["score"])
    for i, r in enumerate(out):
        r["rank"] = i + 1
    passing = [r for r in out if r["metrics"].get("accepted")]
    report = {
        "ranking": out,
        "champion": passing[0] if passing else None,
        "best_candidate": out[0] if out else None,
        "writeback": ("未写回生产 config.json，需明确指令"
                      if passing else
                      "拒绝产生冠军或写回：没有候选通过全部硬门槛"),
    }
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_progress(stage="final_done", groups_done=groups_done, total=total_planned,
                   elapsed_min=round((time.time() - t0) / 60, 1), best=best,
                   eta_min=0, done=True)
    print("全部完成: %d 组, 最低分(终评) %.4f" %
          (groups_done, out[0]["score"] if out else -1), flush=True)
    if passing:
        print("合格冠军参数:", fmt_params(passing[0]["params"]), flush=True)
    elif out:
        print("没有候选通过硬门槛，未产生冠军。最低分候选未通过:",
              ",".join(out[0]["metrics"]["acceptance_failures"]), flush=True)


if __name__ == "__main__":
    main()
