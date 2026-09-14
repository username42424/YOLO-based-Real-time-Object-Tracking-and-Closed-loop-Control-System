# -*- coding: utf-8 -*-
"""
mouse_control — main.py 的鼠标"手"层（纯输出，无视觉/无目标决策）
=============================================================
本模块是从 main.py 拆分出来的全部"把位移变成真实鼠标输入"的逻辑：

  _INPUT / _MOUSEINPUT / _POINT   ctypes 结构（SendInput 载荷）
  _send_relative(dx, dy)          单次相对移动（Windows SendInput, MOUSEEVENTF_MOVE）
  aim_move(dx_px, dy_px, hu, ...) 一次瞄准移动的完整发送管线：
                                  死区 → 限幅(max_aim_delta) → 灵敏度缩放
                                  → 位移硬上限(max_target_step) → 人手仿真(抖动/跳过/
                                  分步/加速/过冲) → 分步发送，返回诊断

与 main.py 的接口：
  set_max_target_step(v)  由 run_engine 在读取配置后写入单帧位移硬上限(px)，0=不限制；
                          aim_move 在 cap=None 时自动使用该值。
  send_fn 注入            模拟器把 aim_move 的 send_fn 换成环境回调即可完整复用
                          本管线的死区/灵敏度/人手仿真/分步逻辑，调参直接对应真实 main。
"""

import ctypes
import math
import random

_IS_WINDOWS = hasattr(ctypes, "windll")
OWN_INPUT_MARKER = 0xA17E2026

if _IS_WINDOWS:
    user32 = ctypes.windll.user32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", _MOUSEINPUT)]


# 显式声明 SendInput 签名，避免 ctypes 在 64 位进程中发生参数/返回值截断。
try:
    user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
    user32.SendInput.restype = ctypes.c_uint
except Exception:
    pass


def _send_relative(dx, dy):
    """SendInput 相对移动 — FPS 游戏专用。返回值只代表 Windows 接收了输入。"""
    inp = _INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi.dx = int(dx)
    inp.mi.dy = int(dy)
    inp.mi.mouseData = 0
    inp.mi.dwFlags = 0x0001  # MOUSEEVENTF_MOVE (无 ABSOLUTE)
    inp.mi.time = 0
    inp.mi.dwExtraInfo = OWN_INPUT_MARKER
    result = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    return bool(result)


# 单帧最大位移（像素，灵敏度缩放前）——防止误检/闪烁导致准星飞天
_MAX_AIM_DELTA = 120

# 最终位移硬上限(px)：直接限制发给鼠标的实际位移，不依赖 smoothing/增益/灵敏度。
# 由 run_engine 从 config 读取后经 set_max_target_step() 设置；0=不限制。
_MAX_TARGET_STEP = 80.0


def split_move_conserving(dx, dy, steps, profile="ease"):
    """拆成整数子步，保证子步向量和严格等于四舍五入后的总指令。"""
    steps = max(1, int(steps))
    total_x, total_y = int(round(dx)), int(round(dy))
    if profile == "ease" and steps >= 2:
        weight_sum = steps * (steps + 1) / 2.0
        weights = [(steps - i) / weight_sum for i in range(steps)]
    else:
        weights = [1.0 / steps] * steps

    commands = []
    sent_x = sent_y = 0
    cumulative = 0.0
    for i, weight in enumerate(weights):
        cumulative += weight
        if i == steps - 1:
            target_x, target_y = total_x, total_y
        else:
            target_x = int(round(total_x * cumulative))
            target_y = int(round(total_y * cumulative))
        commands.append((target_x - sent_x, target_y - sent_y))
        sent_x, sent_y = target_x, target_y
    return commands


def set_max_target_step(v):
    """run_engine 读取 config 后写入单帧位移硬上限。0=不限制。"""
    global _MAX_TARGET_STEP
    _MAX_TARGET_STEP = max(0.0, float(v))


def set_right_hold(hold):
    """合成右键按下/松开(SendInput)。合成输入会被 GetAsyncKeyState 与低级钩子捕获,
    瞄准热键(aim=mouse_right)据此自动开/关瞄准管线(自动触发功能用)。
    返回是否投递成功。"""
    inp = _INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi.dx = 0
    inp.mi.dy = 0
    inp.mi.mouseData = 0
    inp.mi.dwFlags = 0x0008 if hold else 0x0010  # MOUSEEVENTF_RIGHTDOWN / RIGHTUP
    inp.mi.time = 0
    inp.mi.dwExtraInfo = OWN_INPUT_MARKER
    return bool(user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)))


def aim_move(dx_px, dy_px, hu, raw=False, send_fn=None, cap=None):
    """发送一次瞄准移动，并返回诊断状态。

    raw=True（暴力直瞄）：完全不做人手仿真——无抖动、无分步、无随机跳过，
    位移直接按灵敏度缩放取整后一步发送。

    send_fn=None：默认用 Windows SendInput 发真实鼠标。
    传入自定义 send_fn(dx, dy) 时（如 shootsim 模拟器），把分步后的位移
    交给 send_fn 发送——便于在模拟环境里完整复用 main 的瞄准管线
    （死区/限幅/灵敏度/人手仿真/分步/换向阻尼），调参结果直接对应真实 main。

    cap=None：用全局 _MAX_TARGET_STEP 限幅；传入数值则本次调用用该上限
    （首步大位移：新锁定/切换锁定后的第一步用 max_first_step）。

    人手仿真逻辑（humanize_level > 0 且 raw=False 时生效）：
    - 微抖动：最终位移叠加 ±(level*sens*0.15)px 随机偏移
    - 概率分步：位移 > 3px 时 20% 概率拆成 2 步发送
    - 加速曲线：第二步比第一步略大，模拟轻甩
    - 微小过冲：level>0.3 且 位移>5px 时 10% 概率过冲 5-15%
    - 反应慢：level>0.4 时 5% 概率本帧不发（模拟注意力不够）
    """
    if send_fn is None:
        send_fn = _send_relative
    sens = float(hu.get("sensitivity", 1.0))
    deadzone = float(hu.get("deadzone", 0.0))
    if raw:
        level = 0.0  # raw 模式完全关闭人手仿真
    else:
        level = max(0.0, min(1.0, float(hu.get("humanize_level", 0.15))))
    max_delta = max(1.0, float(hu.get("max_aim_delta", _MAX_AIM_DELTA)))

    # 死区
    if abs(dx_px) < deadzone and abs(dy_px) < deadzone:
        return {"sent": False, "reason": "deadzone", "successes": 0, "failures": 0}

    # 限幅
    if abs(dx_px) > max_delta or abs(dy_px) > max_delta:
        scale = max_delta / max(abs(dx_px), abs(dy_px))
        dx_px *= scale
        dy_px *= scale

    dx = dx_px * sens
    dy = dy_px * sens

    # ── 最终位移硬上限（平坦，不随误差放宽）──
    # 曾改为"随误差自适应"：误差>80px 时上限放宽到≈误差本身(最多240px)，意图
    # 快速咬住短时目标。但该公式让有效增益在 80px 处跳变——误差大→≈1倍误差，
    # 误差小→灵敏度倍数——形成"远目标反而慢、中近目标反而甩"的非单调响应，
    # 是大上摇/大下摇的诱因之一。恢复平坦上限：误差越大移动越大(单调)，
    # 顶到 _MAX_TARGET_STEP 为止(模拟器验证一致采用平坦上限)。
    _cap = _MAX_TARGET_STEP if cap is None else cap
    if _cap > 0.0:
        _final_dist = math.hypot(dx, dy)
        if _final_dist > _cap:
            _s = _cap / _final_dist
            dx *= _s
            dy *= _s

    # ── 人手仿真 ──
    if level > 0.01:
        # 1. 微抖动：最终位移加小幅随机偏移
        j_rms = max(0.05, level * sens * 0.15)
        dx += random.gauss(0, j_rms)
        dy += random.gauss(0, j_rms)

        # 2. 反应慢：偶尔本帧不移动（模拟注意力延迟）
        if level > 0.4 and random.random() < 0.05:
            return {"sent": False, "reason": "humanize_skip", "successes": 0, "failures": 0}

    total = math.hypot(dx, dy)
    if total < 0.5:
        return {"sent": False, "reason": "subpixel", "successes": 0, "failures": 0}

    int_dx, int_dy = int(round(dx)), int(round(dy))
    if int_dx == 0 and int_dy == 0:
        return {"sent": False, "reason": "rounded_zero", "successes": 0, "failures": 0}

    # ── 分步发送 ──
    commands = []
    # 大幅移动 → 多步（模拟分段微调）
    if total > 80 and level > 0.4:
        steps = 2 if level < 0.7 else 3
        base_x, base_y = int_dx // steps, int_dy // steps
        rem_x, rem_y = int_dx - base_x * steps, int_dy - base_y * steps
        for i in range(steps):
            sd = base_x + (1 if i < rem_x else 0)
            sy = base_y + (1 if i < rem_y else 0)
            # 加速曲线：第一步慢，越后越快
            if level > 0.5 and steps > 1:
                accel = 0.7 + 0.6 * (i / (steps - 1))
                sd = int(round(sd * accel))
                sy = int(round(sy * accel))
            if sd != 0 or sy != 0:
                commands.append((sd, sy))
    # 中等移动(3-80px) → 概率分步
    elif total > 3 and level > 0.2:
        if random.random() < level * 0.2:
            split = 0.55 + random.uniform(-0.1, 0.1)
            s1_x, s1_y = int(round(int_dx * split)), int(round(int_dy * split))
            s2_x, s2_y = int_dx - s1_x, int_dy - s1_y
            if s1_x != 0 or s1_y != 0:
                commands.append((s1_x, s1_y))
            if s2_x != 0 or s2_y != 0:
                commands.append((s2_x, s2_y))
        else:
            commands.append((int_dx, int_dy))
    else:
        commands.append((int_dx, int_dy))

    delivered = []
    for command in commands:
        if send_fn(*command):
            delivered.append(command)
    successes = len(delivered)
    failures = len(commands) - successes
    net_dx = sum(c[0] for c in delivered)
    net_dy = sum(c[1] for c in delivered)
    return {"sent": successes > 0, "reason": "sent" if successes > 0 else "send_failed",
            "successes": successes,
            "failures": failures, "net_dx": net_dx, "net_dy": net_dy, "steps": len(commands)}
