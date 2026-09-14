# -*- coding: utf-8 -*-
"""回放保真验证(开环注入) + 目标框尺寸统计。

【开环验证】把日志里的真实发送量(净移 counts)原样注入模拟器视角,
扫描"发送生效延迟 q"(帧), 看模拟器看到的框能否逐帧复现日志里的锁定框:
  - 若最优 q 下平均偏差只有 ~1-2px → 证明回放的世界轨迹就是日志轨迹(扣除自身移动后),
    GIF 里"框不像日志"只是因为闭环时相机由算法驱动, 而日志里的相机由你的手驱动。
  - 结构化日志的 observed_aim 已在截图时点归属；q 仅用于诊断帧对齐，
    不能用来设置 view.apply_delay_frames。

【尺寸统计】日志锁定框的宽高分布(回答"框为什么这么小")。

用法: python replay_verify.py [注入回合数=全部]
"""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from env import Env
import replay as replay_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def box_stats(eps):
    ws, hs = [], []
    for e in eps:
        for p in e.points:
            if p["box"] is not None:
                ws.append(p["w"])
                hs.append(p["h"])
    n = len(hs)
    hs_s = sorted(hs)
    ws_s = sorted(ws)
    print("== 日志锁定框尺寸分布(共 %d 个框) ==" % n)
    print("  高 h: 中位 %.0fpx  均值 %.0f  P10=%.0f  P25=%.0f  P75=%.0f  P90=%.0f  最大 %.0f"
          % (statistics.median(hs), statistics.mean(hs),
             hs_s[int(0.10 * n)], hs_s[int(0.25 * n)], hs_s[int(0.75 * n)],
             hs_s[int(0.90 * n)], hs_s[-1]))
    print("  宽 w: 中位 %.0fpx  均值 %.0f" % (statistics.median(ws), statistics.mean(ws)))
    for th in (20, 30, 40, 60):
        print("  h<%dpx 占 %.0f%%" % (th, 100.0 * sum(1 for v in hs if v < th) / n))


def run_openloop(cfg, ep, q, max_seconds):
    """单回合开环注入: 返回 {frame: (sim_center, log_center)}。

    sim 框在模拟器屏幕坐标(中心 320), 日志框在 320 截图坐标(中心 160),
    对比前先把 sim 框中心换算回截图坐标(减 off)。
    """
    env = Env(cfg)
    off = env.cx - float(cfg["replay"]["capture_center"])
    player = replay_mod.ReplayPlayer(ep, max_seconds=max_seconds, on_end="stop")
    env.spawn_replay(player, 1234 + ep.idx)
    dev = {}
    queue = []   # (deliver_step, (nx, ny))
    step = 0
    while True:
        f = player.peek_frame_no()
        # 本帧"看到"的框(相机含此前所有已生效发送)
        seen = env.replay_detections(0)
        if f is not None and f in ep.net_by_frame:
            if seen:
                b = seen[0]
                sim_c = ((b[0] + b[2]) / 2.0 - off, (b[1] + b[3]) / 2.0 - off)
            else:
                sim_c = None
            p = ep.points[[i for i, pp in enumerate(ep.points) if pp["frame"] == f][0]]
            if p["box"] is not None:
                dev[f] = (sim_c, (p["cx"], p["cy"]))
            queue.append((step + q, ep.net_by_frame[f]))
        due = [n for d, n in queue if d <= step]
        queue = [(d, n) for d, n in queue if d > step]
        dx = sum(n[0] for n in due)
        dy = sum(n[1] for n in due)
        running = env.step(dx, dy)
        step += 1
        if not running or step > 600:
            break
    return dev


def verify(cfg, eps, max_seconds):
    print("== 开环注入验证: 注入日志真实发送量, 模拟器看到的框 vs 日志框 ==")
    print("(扫描 q 检查开环帧对齐；平均偏差越小=回放越忠实，不能据此增加延迟)")
    rows = []
    for q in (0, 1, 2, 3, 4):
        devs = []
        for ep in eps:
            d = run_openloop(cfg, ep, q, max_seconds)
            for sim_c, log_c in d.values():
                if sim_c is not None:
                    devs.append(math.hypot(sim_c[0] - log_c[0], sim_c[1] - log_c[1]))
        rows.append((q, statistics.mean(devs), sorted(devs)[int(0.95 * len(devs))], max(devs), len(devs)))
        print("  q=%d帧: 平均偏差 %.2fpx  P95 %.2fpx  最大 %.2fpx  (n=%d)"
              % (q, rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4]))
    best = min(rows, key=lambda r: r[1])
    print("  → 最优 q=%d (平均 %.2fpx)" % (best[0], best[1]))
    return best[0]


def sample_table(cfg, ep, q, max_seconds, rows=12):
    """打印一个回合逐帧对照(日志框中心 vs 注入后模拟框中心, 均为截图坐标)。"""
    env = Env(cfg)
    off = env.cx - float(cfg["replay"]["capture_center"])
    player = replay_mod.ReplayPlayer(ep, max_seconds=max_seconds, on_end="stop")
    env.spawn_replay(player, 4321 + ep.idx)
    queue = []
    step = 0
    out = []
    while len(out) < rows:
        f = player.peek_frame_no()
        seen = env.replay_detections(0)
        if f is not None and f in ep.net_by_frame:
            p = ep.points[[i for i, pp in enumerate(ep.points) if pp["frame"] == f][0]]
            if p["box"] is not None and seen:
                b = seen[0]
                out.append((f, p["cx"], p["cy"], (b[0] + b[2]) / 2.0 - off, (b[1] + b[3]) / 2.0 - off))
            queue.append((step + q, ep.net_by_frame[f]))
        due = [n for d, n in queue if d <= step]
        queue = [(d, n) for d, n in queue if d > step]
        running = env.step(sum(n[0] for n in due), sum(n[1] for n in due))
        step += 1
        if not running or step > 600:
            break
    print("== 回合 ep%02d 逐帧对照(日志框中心 vs 开环注入后模拟框中心) ==" % ep.idx)
    for f, lx, ly, sx, sy in out:
        print("  帧%3d  日志=(%6.1f,%6.1f)  模拟=(%6.1f,%6.1f)  偏差=%.1fpx"
              % (f, lx, ly, sx, sy, math.hypot(lx - sx, ly - sy)))


def main():
    cfg = load_config(os.path.join(HERE, "config.json"))
    eps_all = replay_mod.load_episodes(cfg["replay"])
    eps = [e for e in eps_all if e.duration <= 3.2 and e.t_first <= 1.0 and e.n_points >= 5]
    box_stats(eps_all)
    print()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(eps)
    q = verify(cfg, eps[:n], float(cfg["episode"]["max_seconds"]))
    print()
    sample_table(cfg, eps[0], q, float(cfg["episode"]["max_seconds"]))
    print()
    print("结构化日志保持 replay.send_lag=0 与 view.apply_delay_frames=0；"
          "不要把 q 写回延迟配置。")


if __name__ == "__main__":
    main()
