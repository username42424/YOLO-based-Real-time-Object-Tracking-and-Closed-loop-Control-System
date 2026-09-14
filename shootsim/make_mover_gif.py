# -*- coding: utf-8 -*-
"""生成目标移动轨迹 GIF:展示 jumpstrafe(左右螃蟹步 + 垂直跳跃)的移动形态。

用法:
    python make_mover_gif.py [输出目录] [seed1 seed2 ...]
默认输出到 ../tu/mover_gifs/,种子 1 2 3。

每张 GIF 分左右两面板:
  左:目标在 2D 平面上的真实移动轨迹(横=左右,纵=上下),红点为当前帧,尾巴为近 0.4s 轨迹。
  右:时间序列 —— 蓝=左右(strafe),橙=上下(jump),竖线为当前时刻。
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from movers import create_mover

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    print("缺 Pillow:", e)
    sys.exit(1)

WORLD = 4000.0
SPAWN = (2000.0, 2000.0)
DUR = 6.0
DT = 0.01
FPS_GIF = 12
FRAME_STEP = int(round(1.0 / FPS_GIF / DT))  # 每个 GIF 帧采样的模拟步数

CFG = {
    "speed": 240.0,
    "dir_change_min": 0.4,
    "dir_change_max": 1.2,
    "bounce": True,
}

BG = (18, 20, 24)
GRID = (46, 50, 58)
TRACE = (120, 126, 134)
TRAIL0 = (60, 70, 90)
CUR = (255, 60, 60)
C_TX = (80, 180, 255)
C_TY = (255, 170, 60)
TXT = (210, 210, 210)


def simulate(seed):
    rng = random.Random(seed)
    m = create_mover("jumpstrafe", CFG, WORLD, WORLD, rng)
    tx, ty = SPAWN
    m.reset(tx, ty)
    n = int(DUR / DT)
    pts = []
    reversals = 0
    jumps = 0
    max_jump = 0.0
    prev_dir = m.dir
    ground = ty
    for i in range(n):
        tx, ty = m.update(DT, tx, ty)
        pts.append((tx, ty))
        if m.dir != prev_dir:
            reversals += 1
            prev_dir = m.dir
        if m.vjump < -1e-6 and getattr(m, "_prev_vjump", 0.0) >= -1e-6:
            jumps += 1
            ground = ty
        m._prev_vjump = m.vjump
        if ty < ground:
            max_jump = max(max_jump, ground - ty)
    txs = [p[0] for p in pts]
    tys = [p[1] for p in pts]
    return pts, {
        "seed": seed, "reversals": reversals, "jumps": jumps,
        "max_jump_px": max_jump,
        "strafe_amp_px": max(txs) - min(txs),
        "tx_range": (min(txs), max(txs)),
        "ty_range": (min(tys), max(tys)),
    }


def draw_frame(d, panelA, panelB, pts, idx, frame_i):
    """在两张面板上画第 frame_i 帧(模拟点索引 idx)。"""
    # ── 左面板:轨迹 ──
    A_x0, A_y0, A_w, A_h = panelA
    d.rectangle([A_x0, A_y0, A_x0 + A_w - 1, A_y0 + A_h - 1], fill=BG)
    for g in range(0, A_w, 50):
        d.line([A_x0 + g, A_y0, A_x0 + g, A_y0 + A_h - 1], fill=GRID)
    for g in range(0, A_h, 50):
        d.line([A_x0, A_y0 + g, A_x0 + A_w - 1, A_y0 + g], fill=GRID)
    # 全轨迹
    if len(pts) > 1:
        d.line([_map2d(p, pts, panelA) for p in pts], fill=TRACE, width=1)
    # 尾巴(近 0.4s)
    tail = pts[max(0, idx - 40):idx + 1]
    for k, p in enumerate(tail):
        c = k / max(1, len(tail))
        col = tuple(int(TRAIL0[j] + (CUR[j] - TRAIL0[j]) * c) for j in range(3))
        x, y = _map2d(p, pts, panelA)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)
    x, y = _map2d(pts[idx], pts, panelA)
    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=CUR)
    d.text((A_x0 + 6, A_y0 + 4), "轨迹 (左-右 / 上-下)", fill=TXT)

    # ── 右面板:时间序列 ──
    B_x0, B_y0, B_w, B_h = panelB
    d.rectangle([B_x0, B_y0, B_x0 + B_w - 1, B_y0 + B_h - 1], fill=BG)
    for g in range(0, B_h, 50):
        d.line([B_x0, B_y0 + g, B_x0 + B_w - 1, B_y0 + g], fill=GRID)
    d.text((B_x0 + 6, B_y0 + 4), "时间序列  蓝=左右  橙=上下", fill=TXT)
    n = len(pts)
    tx_pts = [(B_x0 + int(i / (n - 1) * (B_w - 1)), _map1d(pts[i][0], pts, B_y0, B_h)) for i in range(n)]
    ty_pts = [(B_x0 + int(i / (n - 1) * (B_w - 1)), _map1d(pts[i][1], pts, B_y0, B_h)) for i in range(n)]
    d.line(tx_pts, fill=C_TX, width=1)
    d.line(ty_pts, fill=C_TY, width=1)
    cx = B_x0 + int(idx / (n - 1) * (B_w - 1))
    d.line([cx, B_y0, cx, B_y0 + B_h - 1], fill=(120, 120, 120))
    d.text((B_x0 + 6, B_y0 + B_h - 14), f"t={idx * DT:.2f}s", fill=TXT)


def _map2d(p, pts, panel):
    x0, y0, w, h = panel
    txs = [q[0] for q in pts]
    tys = [q[1] for q in pts]
    rx = max(txs) - min(txs) or 1.0
    ry = max(tys) - min(tys) or 1.0
    sc = min((w - 20) / rx, (h - 20) / ry)
    cx = (min(txs) + max(txs)) / 2
    cy = (min(tys) + max(tys)) / 2
    px = x0 + w / 2 + (p[0] - cx) * sc
    py = y0 + h / 2 + (p[1] - cy) * sc
    return px, py


def _map1d(v, pts, y0, h):
    txs = [q[0] for q in pts]
    tys = [q[1] for q in pts]
    lo = min(min(txs), min(tys))
    hi = max(max(txs), max(tys))
    r = hi - lo or 1.0
    return y0 + h - 10 - (v - lo) / r * (h - 20)


def make_gif(seed, out_dir):
    pts, stats = simulate(seed)
    W, H = 1040, 430
    panelA = (0, 0, 430, 430)
    panelB = (450, 0, 570, 430)
    frames = []
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    n_frames = len(pts) // FRAME_STEP
    for f in range(n_frames):
        draw_frame(d, panelA, panelB, pts, f * FRAME_STEP, f)
        frames.append(img.copy())
        d.rectangle([0, 0, W, H], fill=BG)
    path = os.path.join(out_dir, f"mover_seed{seed}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS_GIF), loop=0)
    return path, stats, len(frames)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tu", "mover_gifs")
    seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else [1, 2, 3]
    os.makedirs(out_dir, exist_ok=True)
    for s in seeds:
        path, st, nf = make_gif(s, out_dir)
        print(f"seed={s} 帧={nf} 转向={st['reversals']}次 跳={st['jumps']}次 "
              f"跳高={st['max_jump_px']:.0f}px 横摆幅={st['strafe_amp_px']:.0f}px "
              f"tx={st['tx_range']} ty={st['ty_range']} -> {path}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
