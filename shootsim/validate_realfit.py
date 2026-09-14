# -*- coding: utf-8 -*-
"""验证 RealFitMover("realfit") 生成的轨迹统计是否贴近实战日志拟合结果。

对比三方：实战拟合(fitted_motion.json) / realfit 生成 / 现役 jumpstrafe 配置生成。
只读，不改配置。
"""
import io
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_real_motion as fit  # noqa: E402
from movers import create_mover  # noqa: E402


def traj_stats(ts, xs, ys):
    """与 fit_real_motion.analyze 相同口径的统计。ts/xs/ys: numpy 逐点序列。"""
    dt = np.diff(ts)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    vx = fit.smooth(vx, 3)
    vy = fit.smooth(vy, 3)
    sp = np.hypot(vx, vy)
    # 横移方向段
    dead = 40.0
    holds, leg_speeds, leg_vys = [], [], []
    prev_dir, prev_t = None, None
    leg_vx, leg_vy = [], []
    for i in range(len(vx)):
        d = 0 if vx[i] > dead else (1 if vx[i] < -dead else prev_dir)
        if d is None:
            continue
        if prev_dir is None:
            prev_dir, prev_t = d, i
        elif d != prev_dir:
            hold = i - prev_t
            if 0.05 < hold * float(np.median(dt)) < 5.0:
                holds.append(hold * float(np.median(dt)))
                if leg_vx:
                    leg_speeds.append(float(np.median(np.abs(leg_vx))))
                    leg_vys.append(float(np.median(leg_vy)))
            leg_vx, leg_vy = [], []
            prev_dir, prev_t = d, i
        leg_vx.append(vx[i])
        leg_vy.append(vy[i])
    # 跳跃峰
    jump_peaks, jump_ints = [], []
    last_t = None
    thr = -220.0
    i = 1
    while i < len(vy) - 1:
        if vy[i] < thr and vy[i] <= vy[i - 1] and vy[i] < vy[i + 1]:
            jump_peaks.append(float(-vy[i]))
            if last_t is not None and 0.1 < ts[i] - last_t < 10:
                jump_ints.append(ts[i] - last_t)
            last_t = ts[i]
            j = i
            while j < len(vy) and vy[j] < thr * 0.3:
                j += 1
            i = j + 1
        else:
            i += 1
    return dict(
        speeds=sp, holds=np.array(holds), leg_speeds=np.array(leg_speeds),
        leg_vys=np.array(leg_vys), jump_peaks=np.array(jump_peaks),
        jump_ints=np.array(jump_ints), total_t=float(np.sum(dt)))


def gen_traj(mover_name, seed, duration=60.0):
    rng = random.Random(seed)
    # 模拟器典型帧步长: 视角/目标更新 60fps
    mover = create_mover(mover_name, {
        "speed": 240.0, "dir_change_min": 0.4, "dir_change_max": 1.2,
        "jump_interval_min": 0.8, "jump_interval_max": 2.2,
        "bounce": True,
    }, 4000, 4000, rng)
    dt = 1.0 / 60.0
    x, y = 2000.0, 2000.0
    mover.reset(x, y)
    ts, xs, ys = [0.0], [x], [y]
    t = 0.0
    while t < duration:
        x, y = mover.update(dt, x, y)
        t += dt
        ts.append(t)
        xs.append(x)
        ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)


def main():
    fitted = json.load(io.open(os.path.join(HERE, "fitted_motion.json"), encoding="utf-8"))
    # 逐 seed 统计后合并(不拼接轨迹, 避免接缝伪速度)
    trajs = {}
    for mover_name in ("realfit", "jumpstrafe"):
        acc = dict(speeds=[], holds=[], leg_speeds=[], leg_vys=[],
                   jump_peaks=[], jump_ints=[], total_t=0.0)
        for seed in range(8):
            ts, xs, ys = gen_traj(mover_name, seed)
            st = traj_stats(ts, xs, ys)
            for k in ("speeds", "holds", "leg_speeds", "leg_vys",
                      "jump_peaks", "jump_ints"):
                acc[k].extend(np.asarray(st[k]).tolist())
            acc["total_t"] += st["total_t"]
        trajs[mover_name] = {k: (np.array(v) if k != "total_t" else v)
                             for k, v in acc.items()}
    real_sp = np.array(fitted["samples"]["speed_px_s"])
    real_holds = np.array(fitted["samples"]["strafe_hold_s"])
    real_leg = np.array(fitted["samples"]["leg_speed_px_s"])
    real_jp = np.array(fitted["samples"]["jump_peak_vy"])
    real_ji = np.array(fitted["samples"]["jump_interval_s"])
    real_total = fitted["meta"]["total_motion_s"]

    def row(label, arr, scale_note=""):
        if len(arr) == 0:
            return "%-28s 无样本" % label
        return ("%-28s p25=%7.1f p50=%7.1f p75=%7.1f p90=%7.1f%s"
                % (label, np.percentile(arr, 25), np.percentile(arr, 50),
                   np.percentile(arr, 75), np.percentile(arr, 90), scale_note))

    print("=== 速度 |v| px/s 分位 ===")
    print(row("实战(拟合源)", real_sp))
    print(row("realfit 生成", trajs["realfit"]["speeds"]))
    print(row("jumpstrafe 生成", trajs["jumpstrafe"]["speeds"]))
    print("=== 横移段保持时长 s ===")
    print(row("实战", real_holds))
    print(row("realfit", trajs["realfit"]["holds"]))
    print(row("jumpstrafe", trajs["jumpstrafe"]["holds"]))
    print("=== 段速 |vx| px/s ===")
    print(row("实战", real_leg))
    print(row("realfit", trajs["realfit"]["leg_speeds"]))
    print(row("jumpstrafe", trajs["jumpstrafe"]["leg_speeds"]))
    print("=== 跳跃峰值 vy px/s ===")
    print(row("实战", real_jp))
    print(row("realfit", trajs["realfit"]["jump_peaks"]))
    print(row("jumpstrafe", trajs["jumpstrafe"]["jump_peaks"]))
    print("=== 跳跃频率 (次/s) 与静止占比 ===")
    for name, st in trajs.items():
        jr = len(st["jump_ints"]) / max(1e-6, st["total_t"])
        idle = float(np.mean(st["speeds"] < 30.0))
        print("%-12s 频率=%.3f  静止占比=%.3f" % (name, jr, idle))
    jr = len(real_ji) / max(1e-6, real_total)
    print("实战         频率=%.3f(配对) / %.3f(无偏峰值)  静止占比=%.3f"
          % (jr, fitted["jump_rate_unbiased"], fitted["idle_frac_below30"]))
    print("=== 跳跃间隔 s ===")
    print(row("实战", real_ji))
    print(row("realfit", trajs["realfit"]["jump_ints"]))
    print(row("jumpstrafe", trajs["jumpstrafe"]["jump_ints"]))


if __name__ == "__main__":
    main()
