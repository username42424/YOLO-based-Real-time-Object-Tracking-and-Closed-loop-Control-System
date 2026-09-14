# -*- coding: utf-8 -*-
"""从 logs/ 全部实战日志拟合目标真实运动规律。

流程：
  1. 用 replay.parse_log_file + ReplayEpisode 重建每个回合的目标世界轨迹
     （已扣除自身鼠标移动，见 replay.py 文件头说明）。
  2. 统计：速度分布、横移换向保持时间、静止占比、跳跃(垂直抛物段)频率与高度、
     框宽高分布、检出丢失(gap)规律。
  3. 把拟合结果写入 fitted_motion.json，供 RealFitMover("realfit") 在模拟器中
     生成统计特性与实战一致的目标运动。

只读日志、写 fitted_motion.json，不碰任何生产配置。
"""
import glob
import io
import json
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay as replay_mod  # noqa: E402

OUT_JSON = os.path.join(HERE, "fitted_motion.json")
OUT_REPORT = os.path.join(HERE, "results", "fit_real_motion_report.json")


def load_all_episodes():
    cfg = json.load(io.open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    rp = cfg["replay"]
    scale = (float(rp.get("scale_x", 0.3011)), float(rp.get("scale_y", 0.2684)))
    cap_c = float(rp.get("capture_center", 160.0))
    logs = sorted(glob.glob(os.path.join(HERE, "..", "logs", "*.log")))
    episodes = []
    n_files = 0
    for lp in logs:
        try:
            eps = replay_mod.parse_log_file(lp)
        except Exception:
            continue
        n_files += 1
        for e in eps:
            try:
                episodes.append(replay_mod.ReplayEpisode(
                    e, len(episodes), scale, cap_c, 5, 2))
            except ValueError:
                pass
    return episodes, n_files


def smooth(x, k=3):
    if len(x) < k:
        return np.asarray(x, dtype=float)
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def episode_kinematics(ep):
    """回合内点级世界速度(vx,vy)与时间。返回 (t[], vx[], vy[], wx[], wy[], bw[], bh[])。"""
    pts = ep.points
    t = np.array([p["t_rel"] for p in pts], dtype=float)
    wx = np.asarray(ep.wx, dtype=float)
    wy = np.asarray(ep.wy, dtype=float)
    bw = np.asarray(ep.bw, dtype=float)
    bh = np.asarray(ep.bh, dtype=float)
    if len(t) < 5:
        return None
    dt = np.diff(t)
    if np.median(dt) <= 0:
        return None
    vx = np.diff(wx) / dt
    vy = np.diff(wy) / dt
    tm = 0.5 * (t[:-1] + t[1:])
    # 剔除异常跳变（检测抖动/换锁导致的伪速度）
    ok = (np.abs(vx) < 3000) & (np.abs(vy) < 3000)
    return {"t": tm[ok], "vx": smooth(vx[ok], 3), "vy": smooth(vy[ok], 3),
            "wx": wx, "wy": wy, "bw": bw, "bh": bh, "dt": dt}


def analyze(episodes):
    speeds, vxs, vys = [], [], []
    holds = []          # 横移同方向保持时长
    idle_runs = []      # 静止段时长
    jump_ints = []      # 相邻跳跃间隔
    jump_peaks = []     # 跳跃峰值垂直速度
    jump_gravs = []     # 每跳重力 peak/half_airtime
    bws, bhs = [], []
    durations = []
    total_t = 0.0
    prev_dir = None
    prev_dir_t = None
    prev_leg_vx = []
    prev_leg_vy = []
    leg_speeds = []   # 每条横移段的 |vx| 中位数
    leg_vys = []      # 每条横移段的 vy 中位数(纵向漂移)
    last_jump_t = None
    for ep in episodes:
        k = episode_kinematics(ep)
        if k is None:
            continue
        durations.append(float(k["t"][-1] - k["t"][0]) if len(k["t"]) > 1 else 0.0)
        total_t += float(np.sum(k["dt"]))
        sp = np.hypot(k["vx"], k["vy"])
        speeds.extend(sp.tolist())
        vxs.extend(k["vx"].tolist())
        vys.extend(k["vy"].tolist())
        bws.extend([b for b in k["bw"] if 5 < b < 400])
        bhs.extend([b for b in k["bh"] if 5 < b < 500])
        # 横移方向段（用平滑 vx 的符号，带 40px/s 死区）
        dead = 40.0
        for i in range(len(k["t"])):
            d = 0 if k["vx"][i] > dead else (1 if k["vx"][i] < -dead else prev_dir)
            if d is None:
                d, prev_dir, prev_dir_t = 0, d, k["t"][i]
                continue
            if prev_dir is None:
                prev_dir, prev_dir_t = d, k["t"][i]
            elif d != prev_dir:
                hold = k["t"][i] - prev_dir_t
                if 0.05 < hold < 5.0:
                    holds.append(hold)
                    if prev_leg_vx:
                        leg_speeds.append(float(np.median(np.abs(prev_leg_vx))))
                        leg_vys.append(float(np.median(prev_leg_vy)))
                prev_leg_vx, prev_leg_vy = [], []
                prev_dir, prev_dir_t = d, k["t"][i]
            prev_leg_vx.append(k["vx"][i])
            prev_leg_vy.append(k["vy"][i])
        # 静止段
        idle = sp < 30.0
        run = 0.0
        for j, is_id in enumerate(idle):
            if is_id:
                run += k["dt"][j]
            else:
                if run > 0.05:
                    idle_runs.append(run)
                run = 0.0
        if run > 0.05:
            idle_runs.append(run)
        # 跳跃: vy 负峰(向上)超过阈值且随后回落 → 简单峰值检测
        vy = k["vy"]
        t = k["t"]
        thr = -220.0
        i = 1
        while i < len(vy) - 1:
            if vy[i] < thr and vy[i] <= vy[i - 1] and vy[i] < vy[i + 1]:
                jump_peaks.append(float(-vy[i]))
                # 半滞空: 峰值到 vy 过零(上升) → 重力 = peak / half
                j = i
                while j < len(vy) - 1 and vy[j] < 0:
                    j += 1
                if vy[j] >= 0 and t[j] > t[i]:
                    grav = float(-vy[i]) / (t[j] - t[i])
                    if 300.0 < grav < 12000.0:
                        jump_gravs.append(grav)
                if last_jump_t is not None:
                    gi = t[i] - last_jump_t
                    if 0.1 < gi < 10:
                        jump_ints.append(gi)
                last_jump_t = t[i]
                # 跳过回落段（0.6s 内不再触发）
                j2 = i
                while j2 < len(vy) and vy[j2] < thr * 0.3:
                    j2 += 1
                i = j2 + 1
            else:
                i += 1
    return dict(speeds=np.array(speeds), vxs=np.array(vxs), vys=np.array(vys),
                holds=np.array(holds), idle_runs=np.array(idle_runs),
                jump_ints=np.array(jump_ints), jump_peaks=np.array(jump_peaks),
                jump_gravs=np.array(jump_gravs),
                leg_speeds=np.array(leg_speeds), leg_vys=np.array(leg_vys),
                bws=np.array(bws), bhs=np.array(bhs), durations=np.array(durations),
                total_t=total_t)


def pct(a, ps=(5, 25, 50, 75, 90, 95)):
    a = np.asarray(a)
    if len(a) == 0:
        return {}
    return {str(p): round(float(np.percentile(a, p)), 2) for p in ps}


def main():
    episodes, n_files = load_all_episodes()
    print("解析日志文件 %d 个, 有效回合 %d 个" % (n_files, len(episodes)))
    s = analyze(episodes)
    print("总运动时长 %.1f s" % s["total_t"])
    sp = s["speeds"]
    speed_mean = float(np.mean(sp)) if len(sp) else 0.0
    idle_frac = float(np.mean(sp < 30.0)) if len(sp) else 0.0
    jump_rate = len(s["jump_ints"]) / max(1e-6, s["total_t"])
    jump_rate_unbiased = len(s["jump_peaks"]) / max(1e-6, s["total_t"])
    out = {
        "meta": {
            "source": "logs/*.log 全量实战日志",
            "n_logs": n_files,
            "n_episodes": len(episodes),
            "total_motion_s": round(s["total_t"], 1),
            "scale": [0.3011, 0.2684],
        },
        "speed_px_s": pct(sp),
        "speed_mean": round(speed_mean, 1),
        "vx_px_s": pct(s["vxs"]),
        "vy_px_s": pct(s["vys"]),
        "idle_frac_below30": round(idle_frac, 4),
        "strafe_hold_s": pct(s["holds"]),
        "strafe_hold_mean": round(float(np.mean(s["holds"])), 3) if len(s["holds"]) else None,
        "leg_speed_px_s": pct(s["leg_speeds"]),
        "leg_vy_px_s": pct(s["leg_vys"]),
        "idle_run_s": pct(s["idle_runs"]),
        "jump_interval_s": pct(s["jump_ints"]),
        "jump_rate_per_s": round(jump_rate, 3),
        "jump_rate_unbiased": round(jump_rate_unbiased, 3),
        "jump_peak_vy": pct(s["jump_peaks"]),
        "jump_gravity": pct(s["jump_gravs"]),
        "box_w": pct(s["bws"]),
        "box_h": pct(s["bhs"]),
        # 供 RealFitMover 使用的采样表
        "samples": {
            "strafe_hold_s": [round(v, 3) for v in s["holds"]],
            "leg_speed_px_s": [round(v, 1) for v in s["leg_speeds"]],
            "leg_vy_px_s": [round(v, 1) for v in s["leg_vys"]],
            "speed_px_s": [round(v, 1) for v in sp],
            "jump_interval_s": [round(v, 3) for v in s["jump_ints"]],
            "jump_peak_vy": [round(v, 1) for v in s["jump_peaks"]],
            "jump_gravity": [round(v, 1) for v in s["jump_gravs"]],
            "idle_run_s": [round(v, 3) for v in s["idle_runs"]],
        },
    }
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with io.open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # 摘要
    print("速度 px/s 分位:", out["speed_px_s"], " 均值", out["speed_mean"])
    print("静止占比(速度<30):", out["idle_frac_below30"])
    print("横移保持 s:", out["strafe_hold_s"], " 均值", out["strafe_hold_mean"])
    print("段速度 |vx| px/s 分位:", out["leg_speed_px_s"])
    print("段纵向漂移 vy px/s 分位:", out["leg_vy_px_s"])
    print("跳跃: 频率 %.3f/s (无偏峰值频率 %.3f/s), 峰值vy分位 %s"
          % (jump_rate, jump_rate_unbiased, out["jump_peak_vy"]))
    print("跳跃重力 px/s² 分位:", out["jump_gravity"])
    print("跳跃间隔 s:", out["jump_interval_s"])
    print("框宽分位:", out["box_w"], " 框高分位:", out["box_h"])
    print("已写入", OUT_JSON)


if __name__ == "__main__":
    main()
