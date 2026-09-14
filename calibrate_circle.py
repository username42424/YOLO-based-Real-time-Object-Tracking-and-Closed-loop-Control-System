# -*- coding: utf-8 -*-
"""自动标定 view_scale(屏幕px/鼠标px)——用截图比对自动识别 360° 回转,转满 2 圈取平均。

改进点(相对旧版 calibrate_view_scale.py):
  - 旧版:人工盯地标转回原位按键停,只测 1 圈,误差大。
  - 本版:每步截图,用归一化互相关与起始画面比对,自动检出"世界回到起点"(=360°),
    连续测 2 圈(720°)取平均,消除人工误差 + 提高置信度。

用法:
  1. 进入游戏,把准星对准一个纹理明显的地标(墙角/柱子/招牌),站着别动。
  2. 运行:python calibrate_circle.py [--fov 105] [--dir right|left] [--ads] [--write]
  3. 3 秒倒计时内切回游戏窗口,脚本自动向右(或左)旋转视角。
  4. 转满 2 圈后自动停下,报出:
     - 每圈 360° 用掉的鼠标px(第1圈 / 第2圈 / 平均)
     - 实测 view_scale(屏幕px/鼠标px) 与 config.json 里当前值的差距%
  5. 加 --write 才把新值写进 config.json(默认只报告不动配置)。
"""
import argparse
import ctypes
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

import main as aim  # 只用它的 SendInput 发鼠标(与瞄准软件同一套发送机制)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

# 步进(每步发的鼠标px)。越小精度越高但越慢:29150/50≈583 步/圈。
STEP_PX = 50
STEP_S = 0.035     # 每步间隔(秒);SendInput 太快游戏可能丢采样
GRAB_W = 320       # 比对用截图缩放宽度(速度)
SCORE_THR_OFF = 0.10   # 判定"峰值"的阈值:score >= 全局最高 - 0.10
MIN_LAP_PX = 8000.0    # 小于这个 px 的"回转"不可能是 360°,丢弃
DIP_REQ = 0.04         # 峰值前必须比峰值低这么多(防纯色场景处处高分)
RADIUS = 10            # 局部最大邻域(步数),同一峰只取最高点


def grab_gray():
    """全屏截图 → 灰度小图。用 PIL(与 main.py 的 _capture_from_pil 同源)。"""
    from PIL import ImageGrab
    img = ImageGrab.grab()
    w, h = img.size
    scale = GRAB_W / float(w)
    arr = np.asarray(img.convert("L").resize((GRAB_W, max(1, int(h * scale)))))
    return arr.astype(np.float32)


def corr(a, b):
    """两帧的皮尔逊相关系数(对亮度偏移鲁棒)。"""
    a = a.ravel()
    b = b.ravel()
    da = a - a.mean()
    db = b - b.mean()
    n = da.shape[0]
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if den < 1e-6:
        return 1.0 if n == 0 else 0.0
    return float((da * db).sum() / den)


def send_key(vk, down):
    """用 SendInput 发送按键(供 --ads 按住右键用)。"""
    u = ctypes.windll.user32
    _INPUT = aim._INPUT
    inp = _INPUT()
    inp.type = 1  # INPUT_KEYBOARD
    inp.ki.wVk = vk
    inp.ki.dwFlags = 0 if down else 0x0002  # KEYEVENTF_KEYUP
    u.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", type=float, default=105.0, help="水平FOV(度),默认105")
    ap.add_argument("--dir", default="right", choices=["right", "left"])
    ap.add_argument("--ads", action="store_true", help="按住右键(开镜灵敏度)测")
    ap.add_argument("--write", action="store_true", help="把水平标定值同步写进 config.json")
    ap.add_argument("--step", type=int, default=STEP_PX)
    args = ap.parse_args()

    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    cur_vs = float(cfg.get("aim_control", {}).get("view_scale", 1.0))

    screen_w = ctypes.windll.user32.GetSystemMetrics(0)  # 物理主屏宽
    lap_est = 360.0 * screen_w / (args.fov * (cur_vs or 1e-6))  # 当前配置预测的 360° 鼠标px

    print(f"=== view_scale 自动标定 === 屏幕宽 {screen_w}px  水平FOV {args.fov}°")
    print(f"当前 config.json: view_scale = {cur_vs}  (预测 360° = {lap_est:.0f} 鼠标px)")
    print(f"方向: {'右' if args.dir == 'right' else '左'}  开镜: {'是' if args.ads else '否'}  每步 {args.step}px")
    print("\n请把准星对准纹理明显的地标(墙角/柱子/招牌),站着别动。")
    print("3 秒后自动开始旋转,转满 2 圈停止。期间不要动鼠标/键盘。")
    print("(按任意键可随时中止)")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

    import msvcrt
    dx = -args.step if args.dir == "left" else args.step
    if args.ads:
        send_key(0x02, True)  # VK_RBUTTON down
    try:
        s0 = grab_gray()
        total = 0.0
        series = []      # [(total_px, score)]
        laps = []        # 检测到的每圈 360° 的 total_px
        gmax = -1.0
        prev_frame = s0
        max_change = 0.0  # 相邻帧最大变化量(诊断:画面到底动没动)
        n = 0
        while True:
            aim._send_relative(dx, 0)
            total += args.step
            n += 1
            time.sleep(STEP_S)
            f = grab_gray()
            sc = corr(s0, f)
            ch = corr(prev_frame, f)
            max_change = max(max_change, abs(ch - 1.0))
            prev_frame = f
            series.append((total, sc))
            if sc > gmax:
                gmax = sc

            # 实时峰检:score 接近全局最高,且在前一段明显更低(有"回转"特征)
            if (sc >= gmax - SCORE_THR_OFF and total >= MIN_LAP_PX and
                    total >= (laps[-1] if laps else 0) + max(0.5 * (laps[0] if laps else MIN_LAP_PX), MIN_LAP_PX)):
                # 找局部最高:回看最近 RADIUS 步
                lo = max(0, len(series) - RADIUS)
                seg = series[lo:]
                best = max(seg, key=lambda x: x[1])
                # 峰值前必须明显更低(λ 防纯色场景)
                if len(series) > RADIUS + 8:
                    pre = max(s for _, s in series[max(0, len(series) - 3 * RADIUS): lo])
                    if best[1] - pre < DIP_REQ:
                        pass  # 无回转特征,不记
                    else:
                        laps.append(best[0])
                        print(f"  [第{len(laps)}圈] 360° = {best[0]:.0f} 鼠标px  (score={best[1]:.3f})", flush=True)
                        if len(laps) >= 2:
                            break

            if total > 4.0 * max(lap_est, MIN_LAP_PX):
                print("[中止] 转得太多还没检出 2 圈;可能是窗口没聚焦或场景太均匀/无地标。")
                break
            if msvcrt.kbhit():
                print("[中止] 按键中断")
                break
            if n % 200 == 0:
                print(f"  已转 {total:.0f}px, 当前匹配 {sc:.3f}", flush=True)
    finally:
        if args.ads:
            send_key(0x02, False)  # VK_RBUTTON up

    print(f"\n画面变化诊断: 相邻帧最大偏差 {max_change:.3f}  (≈0=画面没动,即窗口没聚焦)")
    print(f"匹配度范围: 最低 {min(s for _, s in series):.3f} → 最高 {gmax:.3f}  "
          f"(最高-最低 <0.15 说明场景太均匀,结果不可信)")

    if len(laps) < 1:
        print("未检出任何 360° 回转,没有结果。请对准有明显纹理的地标重试。")
        return
    laps = laps[:2]
    N1 = laps[0]
    N2 = laps[1] - laps[0] if len(laps) > 1 else None
    N_avg = (N1 + (N2 or N1)) / (2 if N2 else 1)

    px_per_deg = screen_w / args.fov
    vs1 = 360.0 / N1 * px_per_deg
    vs2 = (360.0 / N2 * px_per_deg) if N2 else None
    vs_avg = 360.0 / N_avg * px_per_deg

    gap = (N_avg / lap_est - 1.0)  # 相对当前配置的比值差(小数,正=游戏灵敏度比配置低)

    print("\n================ 结果 ================")
    if N2:
        print(f"第1圈 360° = {N1:.0f}px  → view_scale {vs1:.4f}")
        print(f"第2圈 360° = {N2:.0f}px  → view_scale {vs2:.4f}  (两圈差 {abs(N2 - N1) / N2:.1%})")
    print(f"平均  360° = {N_avg:.0f} 鼠标px")
    print(f"=== 实测 view_scale = {vs_avg:.4f} (屏幕px/鼠标px) ===")
    print(f"当前配置 view_scale = {cur_vs}")
    print(f"差距 = {gap:+.1%}  ({'游戏灵敏度比标定时低,欠补偿→过冲侧' if gap > 0 else '游戏灵敏度比标定时高,过补偿→欠冲侧'})")
    print("(正差距=每 360° 要多花鼠标px,即游戏灵敏度变低了)")

    if args.write:
        value = round(vs_avg, 4)
        cfg.setdefault("aim_control", {})["view_scale"] = value
        cfg.setdefault("unit", {})["px_per_count"] = value
        json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"已写入 config.json: view_scale = px_per_count = {value:.4f}  (原 {cur_vs})")
    else:
        print(f"未写入(加 --write 才会更新 config.json)")


if __name__ == "__main__":
    main()
