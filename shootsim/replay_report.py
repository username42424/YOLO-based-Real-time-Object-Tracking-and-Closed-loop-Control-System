# -*- coding: utf-8 -*-
"""回放数据校验报告：解析实战日志 → 检查回合数/时长/循环周期，
并用"世界轨迹平滑度"回归验证自身移动扣除比例(scale)是否与实机标定一致。
同时给出日志侧的 60% 区域停留占比(准星=红点参考 & 几何中心)作为模拟器对照基准。

用法:  python replay_report.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
import replay as R

HERE = os.path.dirname(os.path.abspath(__file__))


def _best_scale(pts, axis, lag=2):
    """网格搜索 scale(capture_i 反映截至 i-lag 点的发送, 与 replay.py 同约定):
    让世界轨迹(capture+Σ净移*s)的二阶差分最小(最平滑)。"""
    nets = [p["net"][axis] for p in pts]
    caps = [(p["cx"] if axis == 0 else p["cy"]) for p in pts]
    best_s, best_j = None, None
    s = 0.02
    while s <= 1.201:
        cum = 0.0
        prev = 0.0
        w = []
        for i in range(len(pts)):
            w.append(caps[i] - 160.0 + cum)
            cum += prev * s
            prev = nets[i]
        j = sum(abs(w[i + 1] - 2 * w[i] + w[i - 1]) for i in range(1, len(w) - 1)) / max(1, len(w) - 2)
        if best_j is None or j < best_j:
            best_j, best_s = j, s
        s += 0.01
    return best_s, best_j


def main():
    cfg = load_config(os.path.join(HERE, "config.json"))
    rp = cfg.get("replay")
    if not rp:
        print("config.json 缺少 replay 段")
        return
    print("== 解析统计 ==")
    total_raw = 0
    for lp in rp["logs"]:
        path = lp if os.path.isabs(lp) else os.path.normpath(os.path.join(HERE, lp))
        eps = R.parse_log_file(path)
        total_raw += len(eps)
        npts = sum(len(e["points"]) for e in eps)
        print("  %s: 回合 %d, 点 %d" % (os.path.basename(path), len(eps), npts))
    episodes = R.load_episodes(rp)
    print("  有效回合: %d (丢弃过短 %d)" % (len(episodes), getattr(R.load_episodes, "last_dropped", 0)))
    if not episodes:
        return
    print()
    print("== 回合明细(前 60) ==")
    print("  idx  src                  dur(s) 点  帧  iter(ms)  dw60准星 dw60中心  自移合量(px) 大跳变")
    dws_ref, dws_ctr, iters, durs = [], [], [], []
    for e in episodes[:60]:
        nref = nctr = 0
        jumps = 0
        netx = nety = 0.0
        for i, p in enumerate(e.points):
            if p["box"] is None:
                continue
            h60 = p["h"] * 0.3
            cy0, cy1 = p["cy"] - h60, p["cy"] + h60
            rx, ry = p["ref"]
            if p["box"][0] <= rx <= p["box"][2] and cy0 <= ry <= cy1:
                nref += 1
            if p["box"][0] <= 160.0 <= p["box"][2] and cy0 <= 160.0 <= cy1:
                nctr += 1
            if i > 0:
                dx = e.wx[i] - e.wx[i - 1]
                dy = e.wy[i] - e.wy[i - 1]
                if dx * dx + dy * dy > 100.0 ** 2:
                    jumps += 1
            netx += p["net"][0]
            nety += p["net"][1]
        n = sum(1 for p in e.points if p["box"] is not None)
        wr = nref / max(1, n)
        wc = nctr / max(1, n)
        dws_ref.append(wr)
        dws_ctr.append(wc)
        iters.append(e.iter)
        durs.append(e.duration)
        print("  %3d  %-20s %5.2f  %3d %4d  %6.1f   %6.1f%%  %6.1f%%   (%+6.0f,%+6.0f)  %d"
              % (e.idx, os.path.basename(e.src)[:20], e.duration, e.n_points, e.n_frames,
                 e.iter * 1000, wr * 100, wc * 100, netx, nety, jumps))
    print()
    print("== 汇总 ==")
    print("  回合时长: 中位 %.2fs  平均 %.2fs" % (statistics.median(durs), statistics.mean(durs)))
    print("  循环周期: 中位 %.1fms  (即实战识别/决策约 %.0f fps)"
          % (statistics.median(iters) * 1000, 1.0 / statistics.median(iters)))
    print("  日志侧 60%%区域停留占比: 准星(红点参考) 均值 %.1f%% | 中心 均值 %.1f%%  (按回合均值)"
          % (statistics.mean(dws_ref) * 100, statistics.mean(dws_ctr) * 100))
    print()
    print("== 自身移动扣除比例(scale) 验证 ==")
    bx, by = [], []
    for e in episodes:
        if e.n_points >= 8:
            sx, _ = _best_scale(e.points, 0)
            sy, _ = _best_scale(e.points, 1)
            bx.append(sx)
            by.append(sy)
    if bx:
        print("  水平: 配置 %.4f | 平滑度最优 中位 %.2f (n=%d)" % (rp.get("scale_x", 0.3011), statistics.median(bx), len(bx)))
        print("  垂直: 配置 %.4f | 平滑度最优 中位 %.2f (n=%d)" % (rp.get("scale_y", 0.2684), statistics.median(by), len(by)))
        print("  (最优值与配置接近 → 扣除比例可信; 偏差大 → 提示标定/坐标假设需复核)")


if __name__ == "__main__":
    main()
