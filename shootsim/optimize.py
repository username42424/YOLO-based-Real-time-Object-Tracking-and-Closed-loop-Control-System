# -*- coding: utf-8 -*-
"""参数搜索与对比实验（headless 批量运行，输出对比结果）。

用法:
    python optimize.py                          # 随机搜索（SEARCH_SPACE）
    python optimize.py --iterations 300 --seeds 5
    python optimize.py --compare                # 预设对比实验（策略/瞄准点/射速等）
    python optimize.py --output results/search.csv

输出:
    results/search.csv        每组合的逐回合击杀时间 + 统计
    results/best_config.json  最优参数（可拷入 config.json 使用）
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_sim
from config import (DEFAULT, SEARCH_SPACE, deep_merge, get, load_config,
                    save_config, set_path)
from logger import SilentLogger

# ── main 策略的参数搜索空间 ──
# 对应根目录 main.py 的 config.json 字段。搜索出的最优会写回根目录 config.json。
# 注意：main.py 的平滑模式用 aim_control.smoothing 覆盖 humanize.smoothing，
#       所以这里搜 aim_control.smoothing（humanize.smoothing 是哑变量）。
MAIN_SEARCH_SPACE = [
    ("aim_control.smoothing",     "uniform", [0.2, 0.9]),
    ("aim_control.target_ema",    "uniform", [0.10, 0.45]),
    ("aim_control.ema_max_step",  "uniform", [20.0, 120.0]),
    ("aim_control.max_total_gain", "uniform", [0.6, 1.2]),
    ("humanize.sensitivity",      "uniform", [0.5, 2.0]),
    ("humanize.deadzone",         "uniform", [0.0, 10.0]),
    ("humanize.max_aim_delta",    "uniform", [40.0, 200.0]),
    ("max_target_step",           "uniform", [40.0, 180.0]),
    ("max_first_step",            "uniform", [80.0, 160.0]),
    ("mouse_step_ms",             "uniform", [16.0, 50.0]),
    ("step_min_gap_ms",           "uniform", [12.0, 30.0]),
    ("chest_ratio",               "uniform", [0.05, 0.30]),
    ("aim_control.lead_frames",   "uniform", [0.0, 5.0]),
    ("box_scale_gain.min_gain",   "uniform", [0.8, 2.0]),
    ("box_scale_gain.max_gain",   "uniform", [0.4, 1.4]),
]

# 局部细搜空间：围绕旧配置（config_user_backup）做 ±~20% 邻域搜索，
# 尤其把 max_target_step / max_first_step 收紧到旧值附近，避免再搜出过冲大步长。
LOCAL_MAIN_SEARCH_SPACE = [
    ("aim_control.smoothing",     "uniform", [0.49, 0.73]),
    ("aim_control.target_ema",    "uniform", [0.24, 0.36]),
    ("aim_control.ema_max_step",  "uniform", [49.0, 74.0]),
    ("aim_control.max_total_gain", "uniform", [0.85, 1.0]),
    ("humanize.sensitivity",      "uniform", [1.15, 1.73]),
    ("humanize.deadzone",         "uniform", [3.5, 6.0]),
    ("humanize.max_aim_delta",    "uniform", [95.0, 145.0]),
    ("max_target_step",           "uniform", [52.0, 78.0]),
    ("max_first_step",            "uniform", [72.0, 108.0]),
    ("mouse_step_ms",             "uniform", [30.0, 44.0]),
    ("step_min_gap_ms",           "uniform", [18.0, 28.0]),
    ("chest_ratio",               "uniform", [0.16, 0.24]),
    ("aim_control.lead_frames",   "uniform", [0.0, 2.0]),
    ("box_scale_gain.min_gain",   "uniform", [1.1, 1.65]),
    ("box_scale_gain.max_gain",   "uniform", [0.8, 1.25]),
]

MAIN_CONFIG_PATH = os.environ.get(
    "AIM_MAIN_CONFIG",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"),
)


def load_main_config():
    with open(MAIN_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_main_config(rng, search_space=MAIN_SEARCH_SPACE):
    """按给定搜索空间随机采样一份 main 配置，写回 MAIN_CONFIG_PATH 并返回。"""
    cfg = load_main_config()
    for dotted, kind, params in search_space:
        parts = dotted.split(".")
        if kind == "choice":
            val = rng.choice(params)
        else:
            val = round(rng.uniform(params[0], params[1]), 4)
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    # 约束：最小发送间隔 ≤ 基础间隔（否则 MainTracker gap 异常）
    if float(cfg.get("step_min_gap_ms", 18.0)) > float(cfg.get("mouse_step_ms", 24.0)):
        cfg["step_min_gap_ms"] = round(float(cfg.get("mouse_step_ms", 24.0)), 4)
    with open(MAIN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


def run_headless(cfg, seed):
    return run_sim.run_episode(cfg, "auto", seed, renderer=None,
                               logger=SilentLogger(), quiet=True)


def run_headless_main(seed):
    """main 策略：用 MAIN_CONFIG_PATH（已由 sample_main_config 更新），模拟用 main 追踪器。

    并行时每个进程用独立配置（AIM_MAIN_CONFIG 环境变量指定），避免争抢同一个文件。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sim_cfg = load_config(os.path.join(here, "config.json"))
    sim_cfg["tracker"]["strategy"] = "main"
    return run_sim.run_episode(sim_cfg, "auto", seed, renderer=None,
                               logger=SilentLogger(), quiet=True,
                               main_cfg_path=MAIN_CONFIG_PATH)


def kill_metric(summary, max_seconds):
    """击杀时间指标：超时按 max_seconds 计（惩罚）。"""
    return summary["kill_time"] if summary["kill_time"] is not None else max_seconds


def accuracy_metric(summaries):
    """只汇总非 oracle accuracy；oracle 的命中率不具备比较意义。"""
    valid = [s["accuracy"] for s in summaries
             if s.get("accuracy_valid", True)]
    return statistics.mean(valid) if valid else None


def sample_config(rng):
    """按 SEARCH_SPACE 随机采样一份配置。"""
    cfg = deep_merge(DEFAULT, {})
    for dotted, kind, params in SEARCH_SPACE:
        if kind == "choice":
            set_path(cfg, dotted, rng.choice(params))
        elif kind == "uniform":
            set_path(cfg, dotted, round(rng.uniform(params[0], params[1]), 4))
    return cfg


def evaluate_config(cfg, n_seeds, max_seconds, rng):
    """一组参数 × n_seeds 个独立随机种子 → (kill_times, summaries)。

    种子从 rng 现采样（避免固定种子池导致过拟合——"幸运扫射"配置
    在少数种子上靠震荡偶尔爆头取胜，换种子就现原形）。
    """
    kills, sums = [], []
    for _ in range(n_seeds):
        s = rng.randint(0, 10 ** 9)
        sm = run_headless(cfg, s)
        kills.append(kill_metric(sm, max_seconds))
        sums.append(sm)
    return kills, sums


def run_main_search(args, iters, n_seeds, max_seconds, out_csv, out_best,
                    search_space=MAIN_SEARCH_SPACE):
    """main 策略参数搜索：采样写回 MAIN_CONFIG_PATH → 模拟评估 → 记录。

    注意：MainTracker 在创建时读取 MAIN_CONFIG_PATH，因此每轮采样后
    必须重新加载 main 模块（缓存避免重复 exec）。这里通过 run_sim 内部
    每次 run_episode 都会 create_tracker，trackers._load_aim_main 每次
    都会重新 exec main.py 并读 config.json，故无需额外处理。

    排序：最差击杀时间优先（压制过冲大步长）→ 超时数 → 中位数。
    """
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    param_cols = [d for d, _, _ in search_space]
    print(f"=== main 参数搜索（{iters} 组 × {n_seeds} 种子，超时={max_seconds}s）===", flush=True)
    print(f"参数将写回 → {MAIN_CONFIG_PATH}\n", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(param_cols + ["mean_kill", "median_kill", "std_kill", "min_kill",
                                 "timeouts", "mean_accuracy", "kill_times"])
        results = []
        t0 = time.perf_counter()
        report_interval_s = 180.0  # 每 3 分钟汇报一次
        last_report = -report_interval_s
        best_so_far = None
        for i in range(iters):
            mcfg = sample_main_config(rng, search_space)
            kills = []   # 仅统计完成击杀的时间（超时=3s 的回合不计入平均，用户要求）
            sums = []
            for _ in range(n_seeds):
                s = rng.randint(0, 10 ** 9)
                sm = run_headless_main(s)
                if sm["kill_time"] is not None:
                    kills.append(sm["kill_time"])
                sums.append(sm)
            to = sum(1 for sm in sums if sm["kill_time"] is None)
            if kills:
                mean_k = statistics.mean(kills)
                med_k = statistics.median(kills)
                std_k = statistics.pstdev(kills)
                min_k = min(kills)
            else:
                mean_k = med_k = std_k = min_k = max_seconds
            acc = accuracy_metric(sums)
            row = [get(mcfg, d) for d in param_cols]
            w.writerow(row + [round(mean_k, 4), round(med_k, 4), round(std_k, 4),
                              round(min_k, 4), to,
                              round(acc, 4) if acc is not None else "N/A",
                              ";".join(str(round(k, 3)) for k in kills)])
            f.flush()  # 每轮落盘：CSV 缓冲导致进度查询"卡住"的历史问题
            results.append((med_k, to, mean_k, std_k, dict(mcfg), kills, sums))
            worst = max(kills) if kills else max_seconds
            key = (worst, to, med_k)
            if best_so_far is None or key < best_so_far[0]:
                best_so_far = (key, med_k, to, mean_k, dict(mcfg))
            el = time.perf_counter() - t0
            if (i + 1 == iters) or (el - last_report >= report_interval_s):
                last_report = el
                rate = el / (i + 1)
                print(f"\n[进度 {i + 1}/{iters}] 用时 {el/60:.1f} 分钟，"
                      f"速度 {rate:.2f}s/组，预计还需 {rate*(iters-i-1)/60:.1f} 分钟",
                      flush=True)
                if best_so_far is not None:
                    print(f"  当前最优：最差击杀 {best_so_far[0][0]:.3f}s / 中位 "
                          f"{best_so_far[1]:.3f}s / 超时 {best_so_far[2]}", flush=True)

    # 排序：最差击杀时间 → 超时数 → 中位数
    results.sort(key=lambda r: (max(r[5]) if r[5] else max_seconds, r[1], r[0]))
    print("\n=== main 参数搜索完成：最差击杀时间最小的前 8 组 ===")
    print(f"{'排名':<4}{'中位':<8}{'平均':<8}{'最差':<8}{'超时':<5}{'命中率':<8} 参数")
    for rank, (med, to, mk, sk, mcfg, kills, sums) in enumerate(results[:8], 1):
        acc = accuracy_metric(sums)
        worst = max(kills) if kills else max_seconds
        desc = " ".join(f"{d.split('.')[-1]}={get(mcfg, d)}" for d, _, _ in search_space)
        acc_text = "N/A" if acc is None else f"{acc:.1%}"
        print(f"{rank:<4}{med:<8.3f}{mk:<8.3f}{worst:<8.3f}{to:<5}{acc_text:<8} {desc}")

    # 最优配置已写回 MAIN_CONFIG_PATH（sample_main_config 每轮都写了，
    # 最后一轮写的就是当前最优附近；这里把最优参数明确保存到 out_best）
    best_mcfg = results[0][4]
    save_config(best_mcfg, out_best)
    print(f"\n最优 main 参数已保存 → {out_best}")
    print(f"搜索结果表 → {out_csv}")
    print(f"⚠️ {MAIN_CONFIG_PATH} 当前为最后一轮采样的参数（可用 best 回拷）")


def main():
    ap = argparse.ArgumentParser(description="ShootSim 参数搜索")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--best", default=None)
    ap.add_argument("--compare", action="store_true", help="预设对比实验")
    ap.add_argument("--main", action="store_true",
                    help="搜索 main.py 参数（写回 MAIN_CONFIG_PATH，模拟用 main 追踪器）")
    ap.add_argument("--local", action="store_true",
                    help="局部细搜：围绕旧配置邻域（LOCAL_MAIN_SEARCH_SPACE）")
    ap.add_argument("--seed", type=int, default=42, help="搜索随机种子")
    args = ap.parse_args()

    base = load_config(args.config)
    max_seconds = float(get(base, "episode.max_seconds", 30.0))
    iters = args.iterations or int(get(base, "optimize.iterations", 200))
    n_seeds = args.seeds or int(get(base, "optimize.seeds", 4))
    out_csv = args.output or get(base, "optimize.output", "results/search.csv")
    out_best = args.best or get(base, "optimize.best", "results/best_config.json")

    if args.compare:
        run_compare(base, n_seeds, max_seconds)
        return

    if args.main:
        space = LOCAL_MAIN_SEARCH_SPACE if args.local else MAIN_SEARCH_SPACE
        run_main_search(args, iters, n_seeds, max_seconds, out_csv, out_best,
                        search_space=space)
        return

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    param_cols = [d for d, _, _ in SEARCH_SPACE]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(param_cols + ["mean_kill", "median_kill", "std_kill", "min_kill",
                                 "timeouts", "mean_accuracy", "kill_times"])
        results = []
        for i in range(iters):
            cfg = sample_config(rng)
            kills, sums = evaluate_config(cfg, n_seeds, max_seconds, rng)
            mean_k = statistics.mean(kills)
            med_k = statistics.median(kills)
            std_k = statistics.pstdev(kills)
            to = sum(1 for s in sums if s["kill_time"] is None)
            acc = accuracy_metric(sums)
            row = [get(cfg, d) for d in param_cols]
            w.writerow(row + [round(mean_k, 4), round(med_k, 4), round(std_k, 4),
                              round(min(kills), 4), to,
                              round(acc, 4) if acc is not None else "N/A",
                              ";".join(str(round(k, 3)) for k in kills)])
            results.append((med_k, to, mean_k, std_k, cfg, kills, sums))
            if (i + 1) % 50 == 0:
                print(f"  已搜索 {i + 1}/{iters} …")

    # 排序：中位数击杀时间 → 超时数 → 标准差（鲁棒，抗"幸运扫射"过拟合）
    results.sort(key=lambda r: (r[0], r[1], r[3]))
    print("\n=== 搜索完成：击杀时间最短的前 10 组（按中位数排序）===")
    print(f"{'排名':<4}{'中位击杀s':<10}{'平均击杀s':<10}{'标准差':<8}{'超时':<5}{'命中率':<8} 参数")
    for rank, (med, to, mk, sk, cfg, _, sums) in enumerate(results[:10], 1):
        acc = accuracy_metric(sums)
        desc = " ".join(f"{d.split('.')[-1]}={get(cfg, d)}" for d in
                        ("tracker.strategy", "tracker.gain", "tracker.cap",
                         "tracker.aim_ratio", "tracker.lead_frames",
                         "shooter.policy", "target.speed", "target.mover"))
        acc_text = "N/A" if acc is None else f"{acc:.1%}"
        print(f"{rank:<4}{med:<10.3f}{mk:<10.3f}{sk:<8.3f}{to:<5}{acc_text:<8} {desc}")

    best = results[0][4]  # 元组: (med, to, mean, std, cfg, kills, sums)
    save_config(best, out_best)
    print(f"\n最优配置已保存 → {out_best}")
    print(f"搜索结果表 → {out_csv}")


def run_compare(base, n_seeds, max_seconds):
    """预设对比实验：控制变量，逐个维度对比击杀时间。"""
    rng = random.Random(2026)
    print(f"=== 对比实验（每组 {n_seeds} 个回合，超时={max_seconds}s）===\n")

    def cmp(name, muts, group=None):
        print(f"--- {name} ---")
        print(f"{'配置':<28}{'平均击杀s':<10}{'标准差':<8}{'超时':<5}{'命中率':<8}")
        results = []
        for label, mut in muts:
            cfg = deep_merge(base, mut)
            kills, sums = evaluate_config(cfg, n_seeds, max_seconds, rng)
            mk = statistics.mean(kills)
            sk = statistics.pstdev(kills)
            to = sum(1 for s in sums if s["kill_time"] is None)
            acc = accuracy_metric(sums)
            if acc is not None:
                results.append((mk, label))
            acc_text = "N/A(oracle)" if acc is None else f"{acc:.1%}"
            print(f"{label:<28}{mk:<10.3f}{sk:<8.3f}{to:<5}{acc_text:<14}")
        if results:
            results.sort()
            print(f"→ 最优: {results[0][1]} ({results[0][0]:.3f}s)\n")
        else:
            print("→ 本组没有可排名的非 oracle 配置\n")

    cmp("1. 追踪策略", [
        ("P 控制", {"tracker": {"strategy": "p", "gain": 0.35}}),
        ("PID(ki0.3,kd0.15)", {"tracker": {"strategy": "pid", "gain": 0.35, "ki": 0.3, "kd": 0.15}}),
        ("EMA(α=0.26)", {"tracker": {"strategy": "ema", "gain": 0.35}}),
        ("速度预测(2帧)", {"tracker": {"strategy": "predict", "gain": 0.35, "lead_frames": 2}}),
    ])
    cmp("2. 瞄准点高度（P 策略）", [
        (f"头部上沿({v:.2f})", {"tracker": {"strategy": "p", "aim_ratio": v}})
        for v in [0.08, 0.166, 0.30, 0.42, 0.50]
    ])
    cmp("3. 比例增益（瞄准头部 0.166）", [
        (f"gain={v}", {"tracker": {"strategy": "p", "gain": v}})
        for v in [0.2, 0.3, 0.4, 0.5, 0.7]
    ])
    cmp("4. 射击策略", [
        ("连续射击", {"shooter": {"policy": "always"}}),
        ("oracle_on_aim（不排名）", {"shooter": {"policy": "oracle_on_aim"}}),
    ])
    cmp("5. 目标移动速度", [
        (f"speed={v}", {"target": {"speed": v}})
        for v in [60, 120, 180, 260]
    ])
    cmp("6. 感知延迟（模拟回路延迟）", [
        (f"延迟{v}帧", {"tracker": {"obs_delay_frames": v}})
        for v in [0, 1, 2, 3, 4]
    ])


if __name__ == "__main__":
    main()
