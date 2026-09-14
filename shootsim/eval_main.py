# -*- coding: utf-8 -*-
"""固定 main 配置的评估（replay 模式: 回合=实战日志回放, "种子"=回合序号）。

用法:
    python eval_main.py <main配置.json> [回合数] [起始索引]

replay 模式下每个"种子"对应一个实战日志回合(按序循环, 超出取模);
输出评估标准: 首次进入完整框/框内60%区域、进入后的60%保持率、按可观测
时长加权的驻留与偏上占比，以及真实超时率。击杀和射击节拍不参与评分。
同时把模拟器每次实际发送按 main.py 的 [鼠标] 行格式写入 logging.mouse_log。
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_sim
from config import load_config, get
from logger import SilentLogger
import replay as replay_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    main_cfg_path = os.path.abspath(sys.argv[1])
    selector = sys.argv[2] if len(sys.argv) > 2 else None
    holdout_only = selector == "--holdout"
    n = None if selector is None or holdout_only else int(selector)
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cfg = load_config(os.path.join(HERE, "config.json"))
    mode = get(cfg, "perception.mode")
    if mode == "replay":
        eps = replay_mod.load_episodes(cfg["replay"])
        total = len(eps)
        if not total:
            print("replay 未解析到有效回合")
            sys.exit(1)
        if holdout_only:
            holdout_names = {os.path.basename(path).lower()
                             for path in cfg["replay"].get("holdout_logs", [])}
            episode_ids = [i for i, episode in enumerate(eps)
                           if os.path.basename(episode.src).lower() in holdout_names]
            if not episode_ids:
                print("replay 未配置有效留出回合")
                sys.exit(1)
        else:
            n = n or total
            episode_ids = [start + i for i in range(n)]
        n = len(episode_ids)
        scope = "留出集" if holdout_only else f"起始索引 {start}"
        print(f"评估 {main_cfg_path} × {n} 回合 (日志回放 {total} 个有效回合, {scope})")
        print(f"场景: 日志 {len(cfg['replay']['logs'])} 个  "
              f"scale=({cfg['replay'].get('scale_x')},{cfg['replay'].get('scale_y')})  "
              f"send_lag={cfg['replay'].get('send_lag', 2)}  "
              f"策略={get(cfg, 'tracker.strategy')}  射击={get(cfg, 'shooter.policy')}  "
              f"遮挡=已移除")
    else:
        if holdout_only:
            print("--holdout 只适用于 replay 模式")
            sys.exit(1)
        n = n or 30
        episode_ids = [start + i for i in range(n)]
        print(f"评估 {main_cfg_path} × {n} 种子 (场景: {mode})")

    mouse_log = None
    ml_path = (cfg.get("logging") or {}).get("mouse_log")
    if ml_path:
        mouse_log = replay_mod.MouseLog(os.path.join(HERE, ml_path))

    box_entries, inner_entries = [], []
    dwell_rd_s = dwell_ch_s = observed_s = above_s = 0.0
    post_inner_s = post_observed_s = 0.0
    to = censored = box_timeouts = flips = flip_pairs = 0
    move_count = delta_pairs = micro_pairs = 0
    step_sum = delta_sq = micro_delta_sq = peak_step = 0.0
    timeout_seeds, exhausted_seeds = [], []
    try:
        for ep_id in episode_ids:
            last_move = [None]
            last_any = [None]

            def _send_hook(dx, dy):
                nonlocal flips, flip_pairs, move_count, step_sum, peak_step
                nonlocal delta_pairs, delta_sq, micro_pairs, micro_delta_sq
                mag = math.hypot(dx, dy)
                if mag <= 0.0:
                    return
                move_count += 1
                step_sum += mag
                peak_step = max(peak_step, mag)
                if last_any[0] is not None:
                    change_sq = ((dx - last_any[0][0]) ** 2
                                 + (dy - last_any[0][1]) ** 2)
                    delta_pairs += 1
                    delta_sq += change_sq
                    if mag < 20.0 and math.hypot(*last_any[0]) < 20.0:
                        micro_pairs += 1
                        micro_delta_sq += change_sq
                last_any[0] = (dx, dy)
                if mag < 20.0:
                    return
                if last_move[0] is not None:
                    flip_pairs += 1
                    if dx * last_move[0][0] + dy * last_move[0][1] < 0.0:
                        flips += 1
                last_move[0] = (dx, dy)

            sm = run_sim.run_episode(cfg, "auto", ep_id, renderer=None,
                                     logger=SilentLogger(), quiet=True,
                                     main_cfg_path=main_cfg_path, mouse_log=mouse_log,
                                     send_hook=_send_hook)
            if sm.get("true_timeout", False):
                to += 1
                timeout_seeds.append(ep_id)
                if sm.get("first_box_entry_time") is None:
                    box_timeouts += 1
            if sm.get("entry_censored", False):
                censored += 1
            if sm.get("replay_exhausted"):
                exhausted_seeds.append(ep_id)
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
    finally:
        if mouse_log is not None:
            mouse_log.close()

    if box_entries:
        print(f"首次进入完整框(成功回合中位/平均): {statistics.median(box_entries):.3f}s / "
              f"{statistics.mean(box_entries):.3f}s")
    else:
        print("首次进入完整框: 无")
    if inner_entries:
        print(f"首次进入框内60%(仅成功回合中位/平均): "
              f"{statistics.median(inner_entries):.3f}s / {statistics.mean(inner_entries):.3f}s  "
              f"成功: {len(inner_entries)}")
    else:
        print(f"首次进入框内60%: 无  未进入: {n}/{n}")
    dwell_rd = dwell_rd_s / observed_s if observed_s > 0 else 0.0
    dwell_ch = dwell_ch_s / observed_s if observed_s > 0 else 0.0
    target_dwell = post_inner_s / post_observed_s if post_observed_s > 0 else 0.0
    above = above_s / observed_s if observed_s > 0 else 0.0
    print(f"60%区域停留占比: 红点 {dwell_rd * 100:.1f}%  "
          f"十字 {dwell_ch * 100:.1f}%  (按 {observed_s:.1f}s 可观测时长加权)")
    print(f"首次进入后的60%保持率: {target_dwell * 100:.1f}%  "
          f"(按 {post_observed_s:.1f}s 首入后时长加权)")
    print(f"准星位于整个目标框上方: {above * 100:.1f}%")
    print(f"大步反向率(相邻移动均≥20 counts): "
          f"{(flips / flip_pairs if flip_pairs else 0.0):.1%}  ({flips}/{flip_pairs})")
    print(f"步长: 平均 {(step_sum / move_count if move_count else 0.0):.1f}  "
          f"峰值 {peak_step:.1f} counts; 指令变化RMS "
          f"{(math.sqrt(delta_sq / delta_pairs) if delta_pairs else 0.0):.1f}; "
          f"微步抖动RMS "
          f"{(math.sqrt(micro_delta_sq / micro_pairs) if micro_pairs else 0.0):.1f}")
    entry_trials = len(inner_entries) + to
    box_trials = len(box_entries) + box_timeouts
    print(f"真实3秒超时: {to}/{entry_trials}个可判定回合  "
          f"截尾: {censored}/{n}  日志自然结束: {len(exhausted_seeds)}/{n}")
    if timeout_seeds:
        print(f"真实超时回放索引: {timeout_seeds}")
    # 供 eval_and_status 等自动化工具消费；避免再从中文展示文本反向解析。
    machine = {
        "episodes": n,
        "first_box_entry_success_count": len(box_entries),
        "first_box_entry_success_rate": (
            round(len(box_entries) / box_trials, 4) if box_trials else 0.0),
        "first_box_entry_median": (
            round(statistics.median(box_entries), 4) if box_entries else None),
        "first_inner60_success_count": len(inner_entries),
        "first_inner60_median": (round(statistics.median(inner_entries), 4)
                                 if inner_entries else None),
        "entry_trials": entry_trials,
        "inner60_entry_failure_rate": (
            round(to / entry_trials, 4) if entry_trials else 1.0),
        "entry_censored_count": censored,
        "entry_censored_rate": round(censored / n, 4),
        "target_dwell_rate": round(target_dwell, 4),
        "dwell_reddot": round(dwell_rd, 4),
        "dwell_cross": round(dwell_ch, 4),
        "above_box_ratio": round(above, 4),
        "timeout_count": to,
        "timeout_rate": round(to / entry_trials, 4) if entry_trials else 1.0,
        "replay_exhausted_count": len(exhausted_seeds),
        "flip_rate": round(flips / flip_pairs, 4) if flip_pairs else 0.0,
        "mean_step_counts": round(step_sum / move_count, 3) if move_count else 0.0,
        "peak_step_counts": round(peak_step, 3),
        "command_delta_rms_counts": (
            round(math.sqrt(delta_sq / delta_pairs), 3) if delta_pairs else 0.0),
        "micro_jitter_rms_counts": (
            round(math.sqrt(micro_delta_sq / micro_pairs), 3) if micro_pairs else 0.0),
    }
    print("EVAL_JSON=" + json.dumps(machine, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
