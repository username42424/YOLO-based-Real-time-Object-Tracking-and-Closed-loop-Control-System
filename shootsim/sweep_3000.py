# -*- coding: utf-8 -*-
"""已停用的旧版 unit 模式随机参数扫描。

只扫描 MainEngine unit 分支实际使用的参数。评分优先级严格为：
该脚本把日志自然结束误记为超时，且不使用独立验证集；保留源码仅供
历史追溯，不能用于当前回放寻优。请使用 sweep_staged.py。

用法:
    python sweep_3000.py --groups 30000 --seeds 200 --workers 8
    python sweep_3000.py --groups 4          # 冒烟
结果: results/sweep_results.jsonl (每行一组, 断点续跑)
进度: results/sweep_progress.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_JSONL = os.path.join(HERE, "results", "sweep_unit_v3_results.jsonl")
PROGRESS = os.path.join(HERE, "results", "sweep_unit_v3_progress.json")

PARAM_KEYS = [
    "chest_ratio",
    "unit.far_gain",
    "unit.near_gain",
    "unit.near_radius_px",
    "unit.deadzone",
    "aim_control.reversal_damp",
    "aim_control.unit_max_counts",
    "target_lock_distance",
    "target_lock_iou",
    "lock_ambiguity_margin",
]


def sample_params(rng):
    """从用户批准的 unit 有效范围随机采样。"""
    return {
        "chest_ratio": round(rng.uniform(0.10, 0.45), 4),
        "unit.far_gain": round(rng.uniform(0.75, 1.00), 4),
        "unit.near_gain": round(rng.uniform(0.25, 0.55), 4),
        "unit.near_radius_px": round(rng.uniform(10.0, 35.0), 2),
        "unit.deadzone": round(rng.uniform(5.0, 12.0), 2),
        "aim_control.reversal_damp": round(rng.uniform(0.50, 0.95), 4),
        "aim_control.unit_max_counts": round(rng.uniform(200.0, 650.0), 2),
        "target_lock_distance": round(rng.uniform(45.0, 140.0), 2),
        "target_lock_iou": round(rng.uniform(0.05, 0.50), 4),
        "lock_ambiguity_margin": round(rng.uniform(0.05, 0.30), 4),
    }


def priority_score(timeout_rate, flip_rate, hit_rate, dwell, first_hit, median):
    return (timeout_rate * 1e8 + flip_rate * 1e5 +
            (1.0 - hit_rate) * 100.0 + (1.0 - dwell) * 10.0 +
            first_hit + median * 0.1)


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
    _G["n_eps"] = len(episodes)
    _G["max_s"] = float(cfg["episode"]["max_seconds"])
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
    gid, params, ep_ids = task
    t0 = time.time()
    run_episode = _G["run_episode"]
    SilentLogger = _G["SilentLogger"]
    cfg = _G["cfg"]
    max_s = _G["max_s"]
    cfg_main = _patch(copy.deepcopy(_G["base_main"]), params)
    kills, fhs, dwells = [], [], []
    hits = shots = to = flips = flip_pairs = 0
    for ep_idx in ep_ids:
        last_move = [None]
        ep_flips = [0]
        ep_pairs = [0]

        def _send_hook(dx, dy):
            if math.hypot(dx, dy) < 20.0:
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
        if sm["kill_time"] is None:
            to += 1
            kills.append(max_s)
        else:
            kills.append(sm["kill_time"])
        fh = sm.get("first_hit_time")
        fhs.append(fh if fh is not None else max_s)
        dwells.append(sm.get("dwell_reddot", 0.0))
        hits += sm["hits"]
        shots += sm["shots"]
        flips += ep_flips[0]
        flip_pairs += ep_pairs[0]
    n = len(kills)
    if n == 0:
        return gid, params, {"error": "all_failed"}, None, 0
    median = statistics.median(kills)
    fh_med = statistics.median(fhs)
    dwell = statistics.mean(dwells)
    hit_rate = (hits / shots) if shots else 0.0
    to_rate = to / n
    flip_rate = flips / flip_pairs if flip_pairs else 0.0
    score = priority_score(to_rate, flip_rate, hit_rate, dwell, fh_med, median)
    metrics = {
        "median": round(median, 4), "first_hit_median": round(fh_med, 4),
        "dwell": round(dwell, 4), "hit_rate": round(hit_rate, 4),
        "timeout_rate": round(to_rate, 4), "flip_rate": round(flip_rate, 4), "n": n,
    }
    return gid, params, metrics, round(score, 4), time.time() - t0


def make_tasks(n_groups, n_seeds, done_gids):
    tasks = []
    for gid in range(n_groups):
        if gid in done_gids:
            continue
        rng = random.Random(888000 + gid)
        params = sample_params(rng)
        # 200 种子 > 回合池大小 → 有放回抽样
        eps = [rng.randrange(_G["n_eps"]) for _ in range(n_seeds)]
        tasks.append((gid, params, eps))
    return tasks


def load_done():
    done = {}
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["gid"]] = rec
                except Exception:
                    pass
    return done


def write_progress(done_n, total, best, n_err, t0):
    rate = done_n / max(1e-6, time.time() - t0)
    eta_min = (total - done_n) / rate / 60.0 if rate > 0 else None
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump({
            "done": done_n, "total": total, "elapsed_min": round((time.time() - t0) / 60, 1),
            "groups_per_min": round(rate * 60, 2), "eta_min": round(eta_min, 1) if eta_min else None,
            "best": best, "errors": n_err, "updated": time.strftime("%H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--legacy-unsafe", action="store_true",
                    help="仅用于复现历史结果；不会得到可用于生产的结论")
    args = ap.parse_args()

    if not args.legacy_unsafe:
        ap.error("sweep_3000.py 已停用：它使用修复前的回放评分。请运行 sweep_staged.py。")

    _init_worker()
    done = load_done()
    tasks = make_tasks(args.groups, args.seeds, set(done))
    total_done = len(done)
    print("扫描启动: 总组=%d 已完成=%d 本次=%d 工作进程=%d 回合池=%d"
          % (args.groups, total_done, len(tasks), args.workers, _G["n_eps"]), flush=True)

    best = None
    n_err = 0
    t0 = time.time()
    last_prog = 0.0
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    pool = (None if args.workers <= 1 else
            Pool(processes=args.workers, initializer=_init_worker))
    try:
        results = map(run_group, tasks) if pool is None else pool.imap_unordered(run_group, tasks)
        with open(OUT_JSONL, "a", encoding="utf-8") as fout:
            for gid, params, metrics, score, secs in results:
                rec = {"gid": gid, "params": params, "metrics": metrics, "score": score}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                total_done += 1
                if metrics.get("error"):
                    n_err += 1
                elif best is None or score < best["score"]:
                    best = {"gid": gid, "score": score, "params": params, "metrics": metrics}
                if time.time() - last_prog > 15:
                    write_progress(total_done, args.groups, best, n_err, t0)
                    last_prog = time.time()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    write_progress(total_done, args.groups, best, n_err, t0)
    print("扫描完成: %d 组, 错误 %d, 最优分=%s" % (total_done, n_err, best and best["score"]), flush=True)


if __name__ == "__main__":
    main()
