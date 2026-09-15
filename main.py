# -*- coding: utf-8 -*-
"""
Aim — YOLO FPS 视觉自瞄引擎
=============================
支持 YOLOv5 (V5-pro.onnx) 和 YOLOv11 (Dawan v11s) 自动切换。
热键: GetAsyncKeyState 主线程轮询
鼠标: SendInput 相对移动 (MOUSEEVENTF_MOVE)，用于 FPS 游戏视角旋转

运行:  python gui.py   (GUI 控制面板)
       python main.py   (CLI 模式)
"""

import ctypes
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import math
import threading
import json
import sys
import queue
import random
import traceback
import logging
from collections import deque

# ── 本地模块引导:确保本文件目录在 sys.path 中 ──
# shootsim/perception.py 等会用 importlib 以自定义模块名 exec 本文件(非常规 import),
# 此时绝对导入走 sys.path;不引导则 from mouse_control/aim_engine 会找不到同目录文件。
_MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

# ── 鼠标"手"层与瞄准"脑"层(从本文件拆分出的独立模块;re-export 保持对外接口不变)──
#   mouse_control.py   一切"把位移变成真实鼠标输入"的代码:SendInput 结构/_send_relative/
#                      aim_move(死区→限幅→灵敏度→硬上限→人手仿真)
#   aim_engine.py      MainEngine:目标选择 + 每帧瞄准计算(EMA/前导/节流/死区/增益钳制/换向阻尼)
#   main.py(run_engine)只负责编排:截图 → YOLO 检测 → 红点定位 → engine.compute → aim_move
from mouse_control import (
    _INPUT, _MOUSEINPUT, _POINT,
    _send_relative, aim_move,
    _MAX_AIM_DELTA, _MAX_TARGET_STEP,
    set_max_target_step, set_right_hold,
)
from aim_engine import MainEngine
from crosshair_tracker import CrosshairTracker
from motion_arbiter import MotionArbiter, MotionComponents, mix_motion_components
from async_control import (ObservationSnapshot, LatestObservation,
                           RemainingErrorController, AimControlEpoch,
                           RecoilControlEpoch)
from runtime_logging import RunLogSink, new_run_id
from asap_timing import (
    ObservationFreshness,
    bounded_rate_integration_dt,
    merge_config_defaults,
)
from recoil_controller import (
    ManualRecoilController,
    RecoilChordGate,
    RecoilController,
    apply_playback_blend,
)
from raw_mouse_input import RawMouseInput
from recoil_profiles import ProfileStoreError, RecoilProfileStore
from hotkey_config import HOTKEY_DEFAULTS, normalize_hotkeys


# ============================================================
# 0. 日志 — GUI 和 CLI 都能看到
# ============================================================

log = logging.getLogger("aim")
log.setLevel(logging.DEBUG)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%H:%M:%S"))
log.addHandler(_ch)

# ── 会话日志文件:每次引擎启动→停止,自动保存一份带毫秒时间戳的完整日志 ──
_LOG_FILE_HANDLER = None


def _start_file_log():
    """新建本次会话的日志文件 logs/aim_YYYYMMDD_HHMMSS.log 并挂到 logger 上。"""
    global _LOG_FILE_HANDLER
    _stop_file_log()
    try:
        d = os.path.join(_base_dir(), "logs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("aim_%Y%m%d_%H%M%S.log"))
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%H:%M:%S"))
        log.addHandler(fh)
        _LOG_FILE_HANDLER = fh
        return path
    except Exception:
        return None


def _stop_file_log():
    """会话结束:摘下并关闭日志文件句柄。"""
    global _LOG_FILE_HANDLER
    if _LOG_FILE_HANDLER is not None:
        try:
            log.removeHandler(_LOG_FILE_HANDLER)
            _LOG_FILE_HANDLER.close()
        except Exception:
            pass
        _LOG_FILE_HANDLER = None

_GUI_CALLBACK = None
_HOTKEY_MAP = None  # {name: hk_id} — 由 gui.py 设置
_GUI_HWND = None
_EXTERNAL_HK = None  # 由 gui.py 注入的 HotkeyPoller
_GUI_GENERATION = 0   # gui.py 启动时的代次，旧回调发现不匹配就退
_RECOIL_RESET_EVENT = threading.Event()
_RECOIL_STATE_CALLBACK = None
_MANUAL_RECOIL_CONTROLLER = None


def request_recoil_reset():
    """供GUI请求清空当前运行中的会话校准曲线。"""
    _RECOIL_RESET_EVENT.set()


def set_recoil_state_callback(fn):
    """Register a GUI callback receiving immutable manual-recoil snapshots."""
    global _RECOIL_STATE_CALLBACK
    _RECOIL_STATE_CALLBACK = fn


def get_manual_recoil_snapshot():
    controller = _MANUAL_RECOIL_CONTROLLER
    return controller.snapshot() if controller is not None else None


def _crosshair_in_target_box(center, box):
    """Whether the current screenshot-space crosshair is inside its lock box."""
    if center is None or box is None:
        return False
    try:
        x, y = float(center[0]), float(center[1])
        left, top, right, bottom = map(float, box)
    except (TypeError, ValueError, IndexError):
        return False
    return left <= x <= right and top <= y <= bottom


def _manual_recoil_gate_open(target_valid, center, box):
    """Manual recoil is allowed only after YOLO has put the crosshair in-box."""
    return bool(target_valid and _crosshair_in_target_box(center, box))


def _sample_default_vertical_recoil(strength, uniform=random.uniform):
    """Original fixed-strength recoil: a small randomized vertical-only pull."""
    strength = max(0.0, float(strength))
    base = strength * uniform(0.92, 1.08)
    jitter = uniform(-max(0.2, strength * 0.06),
                     max(0.2, strength * 0.06))
    return 0.0, base + jitter


def _default_recoil_tick_strength(strength, output_period_s, control_frame_s):
    """Convert the old per-control-frame strength to one output-worker tick."""
    frame_s = max(1e-4, float(control_frame_s))
    return max(0.0, float(strength)) * max(0.0, float(output_period_s)) / frame_s


def _frame_schedule_of(unit_cfg):
    """读取 unit.frame_schedule(fixed/asap); 缺失或非法值回退 fixed(向后兼容)。"""
    schedule = str((unit_cfg or {}).get("frame_schedule", "fixed")).lower()
    return schedule if schedule in ("fixed", "asap") else "fixed"


def _reference_frame_seconds(unit_cfg):
    """reference_frame_ms: 控制参数的标称整定周期(秒)。

    仅用于把"每帧参数"(每识别周期的后坐力强度/帧数阈值)换算到真实时间,
    在 asap 模式下绝不作为等待时间使用。缺省回退 unit.frame_ms。
    """
    ucfg = unit_cfg or {}
    ref_ms = float(ucfg.get("reference_frame_ms", ucfg.get("frame_ms", 22.0)))
    return max(0.002, ref_ms / 1000.0)


def _unit_frame_wait_seconds(frame_schedule, next_t, now, frame_s):
    """unit 模式本次识别帧开始前应等待的秒数。

    fixed: 等到下一节拍时刻 next_t(旧固定周期语义, 可被 fire_wake 提前唤醒);
    asap:  恒为 0 —— 上一帧"截图→推理→选点→发布"完成后立即开始下一帧,
           frame_ms 不参与等待, 识别频率完全由实际端到端耗时决定。
    """
    if str(frame_schedule).lower() == "asap":
        return 0.0
    return max(0.0, float(next_t) - float(now))


def _compose_recoil_components(default_vertical_y, manual_raw, manual_gate_open):
    """Combine exclusive recoil paths without applying the manual gate to default Y."""
    if manual_gate_open:
        manual_x, manual_y = map(float, manual_raw)
    else:
        manual_x = manual_y = 0.0
    return manual_x, float(default_vertical_y) + manual_y, manual_x, manual_y


def _publish_recoil_state(controller, event, profile_name="", message="",
                          raw_available=False):
    callback = _RECOIL_STATE_CALLBACK
    if callback is None:
        return
    snapshot = controller.snapshot()
    snapshot.update({
        "event": str(event),
        "profile_name": str(profile_name or ""),
        "message": str(message or ""),
        "raw_available": bool(raw_available),
    })
    try:
        callback(snapshot)
    except Exception:
        log.exception("手动轨迹状态回调失败")


def set_gui_generation(gen: int):
    """gui.py 启动引擎时写入代次，热键回调用于检测过期。"""
    global _GUI_GENERATION
    _GUI_GENERATION = gen


def set_gui_callback(fn):
    global _GUI_CALLBACK
    _GUI_CALLBACK = fn


def set_hotkey_map(hk_map: dict):
    """gui.py 调用: 设置 RegisterHotKey ID 映射 {name: hk_id}"""
    global _HOTKEY_MAP
    _HOTKEY_MAP = hk_map


def _emit(msg):
    log.info(msg)
    if _GUI_CALLBACK:
        try:
            _GUI_CALLBACK(msg)
        except Exception:
            pass


def execute_motion_transaction(send_token, desired_x, desired_y,
                               learned_x, learned_y, send_epoch,
                               motion_arbiter, send_fn, controller=None,
                               diagnostics=None, diagnostics_lock=None,
                               emit=None, transaction_lock=None,
                               clock=None):
    """Run the production quantize/auth/send/commit transaction.

    The output worker supplies its live arbiter, epoch, controller and
    ``_send_relative`` callback.  Keeping this as one production seam makes
    the cancellation and failure ordering directly testable with a fake
    sender.  A successful OS submission is never rolled back because a later
    feedback or diagnostic callback fails.
    """
    lock = transaction_lock or threading.RLock()
    now_fn = clock or time.perf_counter

    def increment(name):
        if diagnostics is None:
            return
        if diagnostics_lock is None:
            diagnostics[name] = diagnostics.get(name, 0) + 1
            return
        with diagnostics_lock:
            diagnostics[name] = diagnostics.get(name, 0) + 1

    def safe_emit(message, event_type=None, fields=None, send_id=None):
        if emit is None:
            return
        try:
            emit(message, event_type=event_type, fields=fields,
                 send_id=send_id)
        except Exception:
            # Logging/GUI failure must not change the physical-send result.
            if diagnostics is not None:
                increment("diagnostic_errors")

    def emit_error(exc, phase):
        safe_emit(
            f"[鼠标事务] {phase}异常: {type(exc).__name__}: {exc}",
            event_type="error",
            fields={"phase": phase, "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc()},
        )

    with lock:
        send_x = send_y = 0
        aim_step = recoil_step = (0, 0)
        try:
            send_x, send_y = motion_arbiter.quantize_components(
                desired_x - learned_x, desired_y - learned_y,
                learned_x, learned_y)
            aim_step, recoil_step = motion_arbiter.last_component_steps
        except Exception as exc:
            # quantize_components can mutate pending state before raising.
            # mark_send_failed restores the exact pre-transaction snapshot.
            try:
                motion_arbiter.mark_send_failed(send_x, send_y)
            except Exception as rollback_exc:
                emit_error(rollback_exc, "quantize_rollback")
            increment("send_failed")
            emit_error(exc, "quantize")
            t = now_fn()
            return (False, t, t, "transaction_error", int(send_x), int(send_y),
                    aim_step, recoil_step, t)

        # No integer packet is still a valid accumulator commit, but it is not
        # an input-send event.  Component cancellation is accounted separately
        # because aim/recoil may be non-zero while the physical net is zero.
        if not send_x and not send_y:
            increment("idle_tick")
            if not any(aim_step) and not any(recoil_step):
                if (desired_x - learned_x) or (desired_y - learned_y):
                    increment("fractional_wait")
                reason = "fractional_wait" if (
                    (desired_x - learned_x) or (desired_y - learned_y)
                ) else "no_packet"
            elif any(aim_step) or any(recoil_step):
                increment("component_cancel")
                reason = "component_cancel"
            else:
                reason = "no_packet"
            try:
                motion_arbiter.mark_send_succeeded(
                    sent_at_s=None, record_event=False)
            except Exception as exc:
                try:
                    motion_arbiter.mark_send_failed(send_x, send_y)
                except Exception as rollback_exc:
                    emit_error(rollback_exc, "idle_rollback")
                increment("send_failed")
                emit_error(exc, "idle_commit")
                t = now_fn()
                return (False, t, t, "transaction_error", 0, 0,
                        aim_step, recoil_step, t)
            t = now_fn()
            return (True, None, None, reason, 0, 0,
                    aim_step, recoil_step, t)

        increment("send_attempt")
        try:
            sent_ok, send_started_s, send_ended_s, send_reason = (
                send_epoch.run_send(send_token, send_fn, int(send_x), int(send_y)))
        except Exception as exc:
            # The epoch normally converts sender exceptions to send_failed;
            # retain this guard for alternate/test epochs.
            try:
                motion_arbiter.mark_send_failed(send_x, send_y)
            except Exception as rollback_exc:
                emit_error(rollback_exc, "send_rollback")
            increment("send_failed")
            emit_error(exc, "send")
            t = now_fn()
            return (False, t, t, "send_failed", int(send_x), int(send_y),
                    aim_step, recoil_step, t)

        if not sent_ok:
            try:
                motion_arbiter.mark_send_failed(send_x, send_y)
            except Exception as exc:
                emit_error(exc, "failed_rollback")
            increment("send_failed")
            safe_emit(
                f"[发送失败] delta=({int(send_x):+d},{int(send_y):+d}) "
                f"result={send_reason}",
                event_type=("send_rejected" if send_reason ==
                            "cancelled_before_send" else "send_failed"),
                fields={"result": send_reason, "net": [int(send_x), int(send_y)],
                        "actual_aim": list(aim_step),
                        "actual_recoil": list(recoil_step)},
            )
            return (False, send_started_s, send_ended_s, send_reason,
                    int(send_x), int(send_y), aim_step, recoil_step,
                    send_ended_s)

        # From here the OS accepted the input.  Never call mark_send_failed
        # below, even if a diagnostic or controller callback is broken.
        increment("send_success")
        commit_t = (send_ended_s if send_ended_s is not None else now_fn())
        event_id = None
        try:
            motion_arbiter.mark_send_succeeded(commit_t, record_event=True)
            event_id = motion_arbiter.last_event_id
        except Exception as exc:
            emit_error(exc, "successful_commit")
        if controller is not None:
            try:
                controller.on_send_succeeded(
                    aim_step[0], aim_step[1], commit_t, event_id)
            except Exception as exc:
                emit_error(exc, "controller_feedback")
        safe_emit(
            f"[发送] delta=({int(send_x):+d},{int(send_y):+d}) "
            f"result=submitted event_id={event_id if event_id is not None else '-'}",
            event_type="send",
            fields={"result": "submitted", "net": [int(send_x), int(send_y)],
                    "actual_aim": list(aim_step),
                    "actual_recoil": list(recoil_step),
                    "ledger_event_id": event_id},
            send_id=event_id,
        )
        return (True, send_started_s, send_ended_s, "submitted",
                int(send_x), int(send_y), aim_step, recoil_step, commit_t)

# ============================================================
# 1. Windows API 常量
# ============================================================

try:
    import ctypes.wintypes as _wt  # noqa: F401  (仅 Windows)
except Exception:
    _wt = None

_IS_WINDOWS = hasattr(ctypes, "windll")

if _IS_WINDOWS:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        winmm = ctypes.windll.winmm
    except Exception:
        winmm = None
else:
    # 非 Windows 平台（如模拟器在 Linux 上加载本模块复用检测器）：
    # 提供占位对象，让模块可导入、检测器工厂可用；鼠标/热键功能不可用。
    class _DummyWin:
        def __getattr__(self, _name):
            raise OSError("Windows API 仅在 Windows 上可用")

        def __call__(self, *a, **k):
            raise OSError("Windows API 仅在 Windows 上可用")
    user32 = _DummyWin()
    kernel32 = _DummyWin()
    winmm = None

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

VK = {}
for ch in "0123456789":
    VK[ch] = ord(ch)
# 字母键：Windows 虚拟键码 = 大写 ASCII（'p'=0x50=80），
# 不能用小写 ASCII（ord('p')=112=0x70=VK_F1）。
for c in range(ord("a"), ord("z") + 1):
    VK[chr(c)] = c - 32
for n in range(1, 13):
    VK[f"f{n}"] = 0x6F + n
VK.update({
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "backspace": 0x08,
    "delete": 0x2E, "esc": 0x1B,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "`": 0xC0,  # 反引号键（ESC 下方，VK_OEM_3）
    "mouse_left": 0x01, "mouse_right": 0x02,
    "mouse_middle": 0x04, "mouse_x1": 0x05, "mouse_x2": 0x06,
})

_MOUSE_PRESS = {0x0201, 0x0204, 0x0207, 0x020B}  # down 消息码
_MOUSE_RELEASE = {0x0202, 0x0205, 0x0208, 0x020C}  # up 消息码
_MOUSE_MSG_VK = {
    0x0201: 0x01, 0x0202: 0x01,  # 左键
    0x0204: 0x02, 0x0205: 0x02,  # 右键
    0x0207: 0x04, 0x0208: 0x04,  # 中键
    0x020B: None, 0x020C: None,  # X 键 — 由 mouseData 决定
}
_KB_PRESS = {WM_KEYDOWN, WM_SYSKEYDOWN}
_KB_RELEASE = {WM_KEYUP, WM_SYSKEYUP}

# ============================================================
# 2. 热键管理器 — GetAsyncKeyState 轮询
# ============================================================
# 不用 SetWindowsHookEx（需要管理员权限 + 消息循环）。
# GetAsyncKeyState 在用户态轮询，无论有没有焦点都能读到按键状态。
# 50 Hz 轮询 + 边缘检测 = 可靠的按下/释放语义。

class HotEvent:
    __slots__ = ("vk", "pressed")
    def __init__(self, vk, pressed):
        self.vk = vk
        self.pressed = pressed


class Hotkeys:
    """
    纯 GetAsyncKeyState 热键管理器。
    不需要 SetWindowsHookEx、不需要消息循环、不挑进程权限。
    """

    def __init__(self):
        self._wanted: dict[int, bool] = {}  # vk → last_pressed_state
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._thread = None
        self._ready = False
        self._aim_vk = None
        self._aim_state_callback = None

    def is_ready(self):
        return self._ready

    def register(self, name: str) -> int:
        """注册热键（键盘或鼠标），返回虚拟键码"""
        vk = VK.get(name.lower())
        if vk is not None:
            if vk not in self._wanted:
                self._wanted[vk] = False
            if str(name).lower() == "mouse_right":
                self._aim_vk = vk
        return vk

    def set_aim_state_callback(self, callback):
        """Notify the output side immediately on aim-key edges."""
        self._aim_state_callback = callback

    def set_aim_vk(self, vk):
        self._aim_vk = vk

    def poll(self, timeout: float = 0.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self):
        if self._running:
            return
        self._ready = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # 不 join — daemon 线程会在进程退出时自动结束

    # ---- 内部 ----

    def _loop(self):
        """50 Hz 轮询 GetAsyncKeyState，做边缘检测入队"""
        interval = 0.005  # 5ms = 200 Hz（热键轮询，与鼠标输出周期无关）
        while self._running:
            for vk, was_pressed in list(self._wanted.items()):
                state = user32.GetAsyncKeyState(vk)
                is_pressed = bool(state & 0x8000)

                if is_pressed != was_pressed:
                    self._wanted[vk] = is_pressed
                    if vk == self._aim_vk and self._aim_state_callback is not None:
                        self._aim_state_callback(is_pressed)
                    self._q.put(HotEvent(vk, is_pressed))

            kernel32.Sleep(int(interval * 1000))

# ============================================================
# 5. YOLO 模型封装 — 支持 v5、v11 和 YOLO26 端到端导出
# ============================================================

# 全局 precision 设置，由 run_engine 在加载模型前设置
_DETECTOR_PRECISION = "fp32"

# 所有 ONNX 推理(DML/CUDA/CPU)共用的全局互斥锁。
# DirectML 不允许两个会话并发 run: 瞄准管线与反应测试线程即使各用独立会话,
# 并发调用仍会让 DML 设备报错, 其 GBK 中文消息被按 UTF-8 解码冒泡成
# UnicodeDecodeError 并永久打残瞄准会话(每帧 [瞄准异常])。
# 独立会话只解决了"不共享会话", 没解决"并发 run"——全局串行化才是根治。
_INFER_LOCK = threading.Lock()

def _create_detector(model_path, conf, iou, emit=None):
    """
    根据模型文件名自动判断 v5 / v11，返回对应的检测器实例。
    统一输出格式: [{"bbox":[x1,y1,x2,y2],"conf":f,"cls":i}, ...]
    """
    basename = os.path.basename(model_path).lower()
    precision = _DETECTOR_PRECISION
    emit = emit or _emit
    emit(f"[模型] precision={precision}")
    if "yolo26" in basename or "yolodelta" in basename or "v26" in basename:
        emit(f"[模型] YOLO26 end-to-end detected: {model_path}")
        # Model metadata: 0=head, 1=person. YOLOv26 remaps these to the
        # project's stable internal convention: 0=body/person, 1=head.
        _CLASS_NAMES = ["enemy", "head"]
        return YOLOv26(model_path, conf, iou, _CLASS_NAMES, precision, emit)
    elif "v5" in basename or basename.startswith("v5"):
        emit(f"[模型] V5-pro detected: {model_path}")
        # 只保留项目约定的两类：0=敌人/身体框，1=头部框。
        # 即使 ONNX 模型仍输出更多类别，后处理也只读取前两类分数。
        _CLASS_NAMES = ["enemy", "head"]
        return YOLOv5(model_path, conf, iou, _CLASS_NAMES, precision, emit)
    elif "v11" in basename or "yolo11" in basename or "dawan" in basename.lower():
        emit(f"[模型] YOLOv11 detected: {model_path}")
        # 模型可能仍是 6 类输出，但项目只使用 0=敌人、1=头部。
        _CLASS_NAMES = ["enemy", "head"]
        return YOLOv11(model_path, conf, iou, _CLASS_NAMES, precision, emit)
    else:
        emit(f"[模型] 未知模型类型，默认 YOLOv11 推理: {model_path}")
        # fallback：探测输出维度
        _CLASS_NAMES = ["enemy", "head"]
        return YOLOv11(model_path, conf, iou, _CLASS_NAMES, precision, emit)

# ── 基类 ──

class _BaseYOLO:
    """公共部分：预处理、画框"""
    def __init__(self, model_path, conf, iou, class_names, precision="fp32",
                 emit=None):
        self._emit = emit or _emit
        self._class_names = class_names
        self._num_classes = len(class_names)
        if getattr(sys, "frozen", False) and not os.path.exists(model_path):
            model_path = os.path.join(sys._MEIPASS, os.path.basename(model_path))
        available = ort.get_available_providers()
        # GPU 优先: 尝试 DML(AMD) / CUDA(NVIDIA) / CPU
        pref = ["DmlExecutionProvider", "CUDAExecutionProvider",
                "TensorrtExecutionProvider", "OpenVINOExecutionProvider",
                "AzureExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in pref if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self._emit(f"  onnxruntime available: {available}")
        self._emit(f"  using: {providers[0]}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # ── DirectML / GPU 优化 ──
        is_gpu = providers[0] in ("DmlExecutionProvider", "CUDAExecutionProvider")
        if is_gpu:
            so.enable_mem_pattern = True
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # ── FP16 加速 (AMD 7700xt DirectML / NVIDIA CUDA 均支持) ──
        if precision == "fp16" and is_gpu:
            self._emit("  FP16 compute enabled")
            so.add_session_config_entry("session.intra_op.use_fp16_compute", "1")

        # 会话创建 + 预热推理一并纳入全局锁: 反应测试线程可能在瞄准已
        # 推理时加载自己的会话, DML 会话创建与在途推理并发同样会触发设备错误。
        with _INFER_LOCK:
            self._sess = ort.InferenceSession(model_path, so, providers=providers)
            self._inp_name = self._sess.get_inputs()[0].name
            _, _, self._h, self._w = self._sess.get_inputs()[0].shape
            self.conf = conf
            self.iou = iou
            self._precision = precision
            self._timing = {}
            dummy = np.zeros((1, 3, self._h, self._w), dtype=np.float32)
            self._sess.run(None, {self._inp_name: dummy})

    @property
    def input_size(self):
        return (self._w, self._h)

    def _preprocess(self, img_bgr):
        """通用预处理：letterbox → BGR2RGB → CHW → [0,1]"""
        ih, iw = img_bgr.shape[:2]
        r = min(self._w / iw, self._h / ih)
        nw, nh = int(round(iw * r)), int(round(ih * r))
        dw, dh = (self._w - nw) // 2, (self._h - nh) // 2

        # 性能优化：根据精度模式选择插值算法
        # INTER_AREA: 缩小时最快且质量好
        # INTER_LINEAR: 平衡速度和质量
        # INTER_LANCZOS4: 高质量但慢
        if self._precision == "high":
            interp = cv2.INTER_LANCZOS4
        elif self._precision == "fast":
            interp = cv2.INTER_AREA  # 更快的缩放
        else:
            interp = cv2.INTER_LINEAR

        img = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
        img = cv2.copyMakeBorder(img, dh, dh, dw, dw,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = img[..., ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[np.newaxis, ...]
        return blob, iw, ih, r, dw, dh

    def detect(self, img_bgr):
        t0 = time.perf_counter()
        blob, iw, ih, r, dw, dh = self._preprocess(img_bgr)
        t1 = time.perf_counter()
        try:
            with _INFER_LOCK:
                out = self._sess.run(None, {self._inp_name: blob})[0]
        except UnicodeDecodeError as e:
            # DirectML 设备错误的 GBK 中文消息按 UTF-8 解码必然失败,
            # 冒泡出来的是误导性的解码异常; 转成可读信息便于定位。
            raise RuntimeError(f"推理失败(疑似 DirectML 设备错误): {e}")
        t2 = time.perf_counter()
        dets = self._postprocess(out, iw, ih, r, dw, dh)
        t3 = time.perf_counter()
        self._timing = {"preprocess": (t1 - t0) * 1000,
                        "inference": (t2 - t1) * 1000,
                        "postprocess": (t3 - t2) * 1000}
        return dets


class YOLOv5(_BaseYOLO):
    """
    V5-pro.onnx 输出格式: [1, 6300, 9] → [n, 9] → cx,cy,w,h,obj,cls0..clsN
    """
    def __init__(self, model_path, conf, iou, class_names, precision="fp32",
                 emit=None):
        super().__init__(model_path, conf, iou, class_names, precision, emit)
        self._emit(f"  YOLOv5: input {self._w}x{self._h}  {self._num_classes}cls  {precision}")

    def _postprocess(self, out, iw, ih, r, dw, dh):
        out = out[0]  # [6300, 9] 或 [n, 5+classes]
        boxes = out[:, :4]
        obj_conf = out[:, 4:5]
        cls_scores = out[:, 5:5 + self._num_classes]
        scores = cls_scores * obj_conf
        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(scores)), cls_ids]

        mask = max_scores > self.conf
        boxes, cls_ids, max_scores = boxes[mask], cls_ids[mask], max_scores[mask]
        if len(boxes) == 0:
            return []

        cx = (boxes[:, 0] - dw) / r
        cy = (boxes[:, 1] - dh) / r
        w = boxes[:, 2] / r
        h = boxes[:, 3] / r
        x1 = np.clip(cx - w / 2, 0, iw)
        y1 = np.clip(cy - h / 2, 0, ih)
        x2 = np.clip(cx + w / 2, 0, iw)
        y2 = np.clip(cy + h / 2, 0, ih)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), max_scores.tolist(), self.conf, self.iou)
        dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                dets.append({
                    "bbox": xyxy[i].astype(int).tolist(),
                    "conf": float(max_scores[i]),
                    "cls": int(cls_ids[i]),
                })
        return dets


class YOLOv11(_BaseYOLO):
    """
    YOLOv8/v11 ONNX 输出格式: [1, C, N] where C = 5 + num_classes
    例如 Dawan_0121_v11s_320: [1, 11, 2100] → 6 classes
    """
    def __init__(self, model_path, conf, iou, class_names, precision="fp32",
                 emit=None):
        super().__init__(model_path, conf, iou, class_names, precision, emit)
        self._emit(f"  YOLOv11: input {self._w}x{self._h}  {self._num_classes}cls  {precision}")

    def _postprocess(self, out, iw, ih, r, dw, dh):
        # YOLOv8/v11 格式: [batch, channels, boxes] → 转置为 [boxes, channels]
        # Dawan v11输出: [1, 11, N] where N = 8400(640x640), 2100(320x320), 1344(256x256)
        # 通道布局: ch0=cx, ch1=cy, ch2=w, ch3=h, ch4-9=6个类别分数, ch10=未使用
        # 注意: YOLOv8/v11 没有单独的objectness通道！ch4就是第一个类别的分数
        preds = out[0]  # [11, N]
        preds = preds.T  # [N, 11]

        boxes = preds[:, :4]
        # ch4-9 是6个类别的分类分数（已经sigmoid过，值域[0,1]）
        cls_scores = preds[:, 4:4 + self._num_classes]

        # YOLOv8/v11直接用分类分数，不需要乘objectness
        scores = cls_scores
        cls_ids = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(scores)), cls_ids]

        mask = max_scores > self.conf
        boxes, cls_ids, max_scores = boxes[mask], cls_ids[mask], max_scores[mask]
        if len(boxes) == 0:
            return []

        cx = (boxes[:, 0] - dw) / r
        cy = (boxes[:, 1] - dh) / r
        w = boxes[:, 2] / r
        h = boxes[:, 3] / r
        x1 = np.clip(cx - w / 2, 0, iw)
        y1 = np.clip(cy - h / 2, 0, ih)
        x2 = np.clip(cx + w / 2, 0, iw)
        y2 = np.clip(cy + h / 2, 0, ih)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), max_scores.tolist(), self.conf, self.iou)
        dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                dets.append({
                    "bbox": xyxy[i].astype(int).tolist(),
                    "conf": float(max_scores[i]),
                    "cls": int(cls_ids[i]),
                })
        return dets


class YOLOv26(_BaseYOLO):
    """Ultralytics YOLO26 end-to-end output: [1, N, 6] = xyxy, score, class."""

    _CLASS_REMAP = {0: 1, 1: 0}  # exported head/person -> internal head/body IDs

    def __init__(self, model_path, conf, iou, class_names, precision="fp32",
                 emit=None):
        super().__init__(model_path, conf, iou, class_names, precision, emit)
        output_shape = self._sess.get_outputs()[0].shape
        if len(output_shape) != 3 or output_shape[-1] != 6:
            raise ValueError(
                f"YOLO26输出格式不兼容: 期望 [1,N,6], 实际 {output_shape}")
        self._emit(f"  YOLO26: input {self._w}x{self._h}  end-to-end {output_shape}  {precision}")

    def _postprocess(self, out, iw, ih, r, dw, dh):
        preds = np.asarray(out)[0]
        if preds.ndim != 2 or preds.shape[1] != 6:
            raise ValueError(f"YOLO26推理输出格式错误: {np.asarray(out).shape}")

        scores = preds[:, 4]
        exported_ids = preds[:, 5].astype(np.int64)
        valid = (np.isfinite(scores) & (scores > self.conf)
                 & np.isin(exported_ids, tuple(self._CLASS_REMAP)))
        preds = preds[valid]
        scores = scores[valid]
        exported_ids = exported_ids[valid]
        if len(preds) == 0:
            return []

        x1 = np.clip((preds[:, 0] - dw) / r, 0, iw)
        y1 = np.clip((preds[:, 1] - dh) / r, 0, ih)
        x2 = np.clip((preds[:, 2] - dw) / r, 0, iw)
        y2 = np.clip((preds[:, 3] - dh) / r, 0, ih)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        detections = []
        for box, score, exported_id in zip(boxes, scores, exported_ids):
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({
                "bbox": box.astype(int).tolist(),
                "conf": float(score),
                "cls": self._CLASS_REMAP[int(exported_id)],
            })
        return detections


def draw_boxes(img, dets, class_names=None, aim_center=None, dot_center=None):
    """调试用：在检测画面上叠加识别框和准星。

    aim_center：瞄准参考点（红点位置），画红色十字。
    dot_center：自动检测到的红点（子弹落点），画黄色圆点标记，方便核对。
    标签画在框下方（不遮挡准星/红点区域）。
    """
    if class_names is None:
        class_names = ["enemy", "head"]
    h, w = img.shape[:2]
    cx, cy = aim_center if aim_center else (w // 2, h // 2)
    cx, cy = int(cx), int(cy)

    # 画中心准星（蓝色：避免与游戏红点/红色刻度混淆，红点用黄圈标记）
    cv2.drawMarker(img, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
    # 画检测到的红点（子弹落点）标记
    if dot_center is not None:
        dx, dy = int(round(dot_center[0])), int(round(dot_center[1]))
        cv2.circle(img, (dx, dy), 5, (0, 255, 255), 2)
        cv2.drawMarker(img, (dx, dy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 1)

    # 画检测框
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cls = d["cls"]
        name = class_names[cls] if cls < len(class_names) else f"cls{cls}"
        label = f"{name}: {d['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 标签放在框下方（不遮挡准星/红点区域）；贴近底边时退回框内上方
        ly = y2 + th + bl + 2
        if ly > h - 2:
            ly = y1 - bl - 2 if y1 - bl - 2 >= th else y1 + th
        cv2.rectangle(img, (x1, ly - th - bl), (x1 + tw, ly), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, ly - bl), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)

    return img

def _base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(_base_dir(), "config.json")

SUPPORTED_TARGET_CLASSES = (0, 1)

def _normalize_target_classes(values):
    """Keep only the project's two supported classes: enemy and head."""
    normalized = set()
    for value in values or []:
        try:
            cls = int(value)
        except (TypeError, ValueError):
            continue
        if cls in SUPPORTED_TARGET_CLASSES:
            normalized.add(cls)
    return sorted(normalized)

# ============================================================
# 默认配置
# ============================================================

DEFAULT = {
    "model": "yolodeltav1.onnx",
    "output_dir": "outputs",
    "conf": 0.45,
    "iou": 0.45,
    "target_classes": [0, 1],
    "target_priority": "body",
    "chest_ratio": 0.3584,
    "head_ratio": 0.70,
    "anti_recoil": False,
    "recoil_profile": "unknown",
    "default_recoil": {"enabled": False, "strength": 4.0},
    "recoil": {"recovery_ms": 100.0, "output_period_ms": 30.0,
               "curve_bin_ms": 60.0, "calibration_runs": 5,
               "min_calibration_bins": 10, "max_target_gap_ms": 120.0},
    "manual_recoil": {
        "enabled": True,
        "profile_name": "",
        "playback_blend_percent": 70.0,
        "curve_bin_ms": 60.0,
        "calibration_runs": 5,
        "min_burst_ms": 300.0,
        "tail_vertical_ratio": 0.30,
        "tail_max_ms": 1500.0,
        "raw_self_test": True,
        "suppress_prediction_during_replay": True,
    },
    "hotkeys": dict(HOTKEY_DEFAULTS),
    "auto_trigger": {
        "enabled": False,
        "min_size": 80.0,
        "conf": 0.65,
        "interval_ms": 75,
        "size": 640,
        "grace_ms": 300,
    },
    "humanize": {
        "sensitivity": 1.5,
        "deadzone": 0.0,
        "humanize_level": 0.1,
        "max_aim_delta": 120.0,
        "micro_jitter": 0.02,
        "smoothing": 0.5,
    },
    "aim_control": {
        "smoothing": 0.5,
        "target_ema": 0.3,
        "ema_max_step": 60.0,     # EMA 单帧跳变限速(px)：目标点相对EMA跳变超过该值时分帧逼近，0=关闭(snap重锁)
        "max_total_gain": 0.95,   # 有效总增益钳制：smoothing×框增益×sensitivity 超过则按比例缩小，0=关闭
        "reversal_damp": 0.6044,  # 反向约保留60%，抑制越过目标后的满额纠正
        "send_every_n_frames": 1, # 发送周期(帧)：1=识别/移动1:1(每帧直发)；2=隔帧发送(识别每帧做,移动每2帧一次,抵消回路延迟振荡)
        "boost_lo_px": 0.0,       # 远距增益起点(px)：误差超过它开始放大增益，0=关闭分段增益
        "boost_hi_px": 130.0,     # 远距增益满点(px)：误差达到它增益升满
        "boost_max": 1.0,         # 远距增益倍数：误差≥boost_hi 时基础增益×该值，1=关闭
        "lead_frames": 0.0,       # 速度前导帧数：外推目标屏幕速度抵消回路延迟/稳态滞后，0=关闭（模拟器验证无增益，默认关）
        "delay_comp_ms": 0.0,     # 延迟补偿窗口(ms)：扣除在途未生效发送，0=关闭（实机视角响应≠42ms假设，默认关）
        "move_digest_ms": 55.0,   # 串行"等生效"的防死锁基量(ms)：实际等待 = max(200, 3×本值)，
                                  # 正常路径由"误差相对缩减达标"退出, 与游戏/识别延迟实时同步
        "unit_max_counts": 323.6093, # unit 单次硬上限；全量日志终评值
        "view_scale": 1.0,        # 屏幕px/鼠标px 换算系数：速度前导抵消自身移动时用；模拟器=1，实机按日志标定(≈0.24)
        "lock_box_filter_mode": "fixed",
        "lock_box_smoothing_alpha": 0.20,
        "lock_box_alpha_min": 0.35,
        "lock_box_alpha_max": 0.85,
        "lock_box_speed_low": 80.0,
        "lock_box_speed_high": 600.0,
        "unit_prediction_enabled": True,
        "prediction_mode": "arrival",
        "prediction_min_ms": 40.0,
        "prediction_max_ms": 120.0,
        "prediction_box_ratio": 0.75,
        "prediction_cap_px": 45.0,
        "prediction_lock_frames": 2,
        "recoil_prediction_box_ratio": 0.35,
        "recoil_prediction_cap_px": 20.0,
    },
    "precision": "fp16",
    "crosshair_mode": "dot_fallback",  # 瞄准参考点模式: center=只认屏幕中心 / dot_only=只认红点 / dot_fallback=允许红点退化(找不到用中心)
    "crosshair_search_radius": 80,    # 红点搜索半径(px)：以截图中心为圆心的搜索窗口
    "crosshair_max_offset": 45.0,
    "crosshair_max_jump": 12.0,
    "crosshair_confirm_frames": 2,
    "crosshair_hold_frames": 2,
    "crosshair_return_frames": 4,
    "crosshair_return_max_step": 24.0,
    "crosshair_ema_alpha": 0.35,
    "crosshair_fallback_decay": 1.0,
    "crosshair_offset_x": 0.0,
    "crosshair_offset_y": 0.0,
    "mouse_log_enabled": True,
    "max_target_step": 80.0,   # 单帧目标逼近上限(px)：目标点大幅跳变时准星每帧最多移动这么多
    "aim_mode": "smooth",
    "direct": {
        "max_delta": 400,
        "interval_ms": 24,
        "deadzone": 4,
        "sensitivity": 0.85,
        "radius": 300,
        "min_box_h": 25
    },
    "box_scale_gain_enabled": False,
    "box_scale_gain": {
        "min_size": 60,
        "max_size": 320,
        "min_gain": 1.3,
        "max_gain": 0.6
    },
    "target_lock_distance": 100.5581,
    "target_lock_iou": 0.0614,
    "lock_zero_iou_max_distance": 24.0,
    "lock_grace_frames": 2,
    "switch_grace_frames": 2,
    "lock_ambiguity_margin": 0.2043,
    "lock_prediction_max_step": 25.0,
    "edge_new_lock_margin": 1.0,
    "unit": {
        "px_per_count": 0.3011,
        "px_per_count_y": 0.2684,
        "far_gain": 0.8548,
        "near_gain": 0.5226,
        "near_radius_px": 31.4676,
        "deadzone": 6.5892,
        "frame_schedule": "fixed",
        "frame_ms": 100.0,
        "reference_frame_ms": 22.0,
        "move_steps": 2,
        "damped": False,
        "step_profile": "ease",
        # rate_hold preserves the existing ASAP controller. remaining_error
        # recomputes a finite residual budget in the output worker.
        "control_strategy": "rate_hold",
        "remaining_error": {
            "tau_ms": 40.0,
            "max_speed_counts_s": 4000.0,
            "max_accel_counts_s2": 0.0,
            "feedback_delay_ms": 20.0,
            "max_observation_age_ms": 120.0,
        },
    },
    "aim_interval_ms": 1,
    "capture_size": 640,
    "capture_center_mode": "screen",
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return merge_config_defaults({}, DEFAULT)
        cfg = merge_config_defaults(cfg, DEFAULT)
        cfg["target_classes"] = _normalize_target_classes(
            cfg.get("target_classes", SUPPORTED_TARGET_CLASSES))
        return cfg
    return merge_config_defaults({}, DEFAULT)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

# ============================================================
# 7. 引擎入口
# ============================================================

# 反应测试 worker 进程级单例(见 _reaction_test_worker)。
# run_engine 有多个启动源(CLI 入口 / GUI 开始按钮 / 引擎热键重启),
# 每次调用若各起一个 worker, 旧 worker 的 stop_event 几乎不会被 set,
# 会永远存活并在同一屏幕事件上重复记试次(日志表现: 每个试次号出现两次)。
# 新引擎启动时若检测到旧 worker 还活着, 先停旧再起新。
_RT_RUN = {"thread": None, "stop": None}

# 自动触发(独立识别到目标→合成按住右键→现有瞄准管线自动启动):
# enabled 是运行期开关(GUI 勾选/热键切换均改这里), worker 每轮读取;
# worker 单例与 _RT_RUN 同理, 防止多引擎实例各起一个 worker 重复按右键。
_AUTO_TG = {"enabled": False, "lock": threading.Lock()}
# args 保存 worker 启动参数(config, model_file, camera, cam_lock, stop_event, emit),
# 供热键开关在 worker 意外死亡时原地重建(引擎仍在运行的前提下)。
_AT_RUN = {"thread": None, "stop": None, "args": None}


def _auto_trigger_toggle(emit=None):
    """热键/GUI 切换自动触发开关, 返回切换后的状态。
    若 worker 已死但引擎仍在运行(检测会话报废等原因), 用保存的 args 原地重建。"""
    with _AUTO_TG["lock"]:
        _AUTO_TG["enabled"] = not _AUTO_TG["enabled"]
        on = _AUTO_TG["enabled"]
    th = _AT_RUN["thread"]
    alive = th is not None and th.is_alive()
    if not alive:
        args = _AT_RUN.get("args")
        st = _AT_RUN.get("stop")
        if args is not None and (st is None or not st.is_set()):
            t = threading.Thread(target=_auto_trigger_worker, args=args, daemon=True)
            _AT_RUN["thread"] = t
            t.start()
            if emit:
                emit("自动触发: worker 已重建, 当前%s" % ("开启" if on else "关闭"))
        elif emit:
            emit("自动触发: %s (worker 未运行: 引擎未启动或未勾选启用)"
                 % ("已开启" if on else "已关闭"))
        return on
    if emit:
        emit("自动触发: 已%s" % ("开启" if on else "已关闭"))
    return on


def _reaction_test_worker(config, model_file, camera, cam_lock, stop_event, emit):
    """反应时间测试线程: 与瞄准管线并行运行, 只识别+只记录按键时刻, 不动鼠标。

    共享主瞄准的 dxcam 相机(经 cam_lock 互斥, 按需抓 640x640 区域), 不另建实例。
    检测器为本线程**独立**的 ONNX 会话: DML 会话不允许并发调用——共享会话时
    DirectML 设备错误(GBK 中文消息)会以 UnicodeDecodeError 冒泡并永久打死瞄准会话;
    所有检测器的 run 再经全局 _INFER_LOCK 串行化(双会话并发同样会炸设备)。
    试次规则: 大框出现→记录右键/左键首次按下; 双键都按下立即完成; 1s 无任何按键
    作废; 目标消失超 lost_s(默认0.3s)时——已按下过则保留记录并完成, 一次都没按
    才作废; 消失超 lost_s 后才允许开新试次。ESC 仅结束本测试线程并打汇总, 不影响瞄准。
    """
    import statistics
    rt_cfg = config.get("reaction_test", {}) or {}
    min_size = float(rt_cfg.get("min_size", 80.0))
    interval = float(rt_cfg.get("interval_ms", 75.0)) / 1000.0
    size = max(128, int(rt_cfg.get("size", 640)))
    giveup = float(rt_cfg.get("window_s", 1.0))
    lost_s = float(rt_cfg.get("lost_s", 0.3))
    classes = set(config.get("target_classes", [0, 1]))
    conf = float(config.get("conf", 0.35))

    # 独立检测器会话(与瞄准互不干扰); 加载失败直接退出线程, 绝不碰瞄准
    try:
        detector = _create_detector(model_file, conf, float(config.get("iou", 0.5)),
                                    emit=emit)
        emit("反应测试: 独立检测器会话就绪(与瞄准分开)")
    except Exception as e:
        emit(f"反应测试: 独立检测器加载失败({e}), 线程退出(瞄准不受影响)")
        return

    if camera is None:
        emit("反应测试: 无 dxcam 相机, 回退 PIL 截屏")
    sw_, sh_ = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def grab():
        try:
            if camera is not None:
                with cam_lock:
                    fr = camera.grab(region=(max(0, sw_ // 2 - size // 2),
                                             max(0, sh_ // 2 - size // 2),
                                             min(sw_, sw_ // 2 + size // 2),
                                             min(sh_, sh_ // 2 + size // 2)),
                                     copy=True, new_frame_only=False)
                return np.asarray(fr) if fr is not None else None
            from PIL import ImageGrab
            img = np.asarray(ImageGrab.grab(
                bbox=(max(0, sw_ // 2 - size // 2), max(0, sh_ // 2 - size // 2),
                      min(sw_, sw_ // 2 + size // 2), min(sh_, sh_ // 2 + size // 2)),
                all_screens=True))
            return img[..., ::-1].copy()
        except Exception:
            return None

    VK_L, VK_R, VK_ESC = 0x01, 0x02, 0x1B
    rx, lx = [], []
    void_nokey = void_lost = 0
    trial = None
    big_absent_t = time.perf_counter()
    trial_end_t = 0.0
    prev_btn = {"L": False, "R": False}
    idx = 0
    _det_fail = 0
    emit(f"反应测试开始: 大框阈值 {min_size:.0f}px | 周期 {interval*1000:.0f}ms | "
         f"区域 {size}x{size} | 计时窗 {giveup:.1f}s | ESC 结束")

    while not stop_event.is_set():
        t_loop = time.perf_counter()
        if user32.GetAsyncKeyState(VK_ESC) & 0x8000:
            emit("反应测试: ESC -> 结束")
            break
        cur = {"L": bool(user32.GetAsyncKeyState(VK_L) & 0x8000),
               "R": bool(user32.GetAsyncKeyState(VK_R) & 0x8000)}
        press = {k: (cur[k] and not prev_btn[k]) for k in cur}
        prev_btn = cur

        frame = grab()
        big = None
        if frame is not None:
            try:
                dets = detector.detect(frame)
                _det_fail = 0
            except Exception:
                dets = []
                _det_fail += 1
                if _det_fail == 30:
                    emit("反应测试: 识别连续失败 30 次, 线程退出(瞄准不受影响)")
                    return
            best = None
            for d in dets or []:
                if int(d.get("cls", -1)) not in classes:
                    continue
                if float(d.get("conf", 0)) < conf:
                    continue
                bx = d["bbox"]
                if max(bx[2] - bx[0], bx[3] - bx[1]) < min_size:
                    continue
                if best is None or (bx[2] - bx[0]) * (bx[3] - bx[1]) > \
                        (best["bbox"][2] - best["bbox"][0]) * (best["bbox"][3] - best["bbox"][1]):
                    best = d
            big = best

        now = time.perf_counter()
        if big is None:
            big_absent_t = now

        if trial is None:
            if big is not None and (now - big_absent_t) >= lost_s \
                    and (now - trial_end_t) >= lost_s:
                trial = {"t0": now, "r": None, "l": None, "lost_t": None}
                idx += 1
                emit("[试次 #%d] 目标出现 (%dx%d)" % (
                    idx, big["bbox"][2] - big["bbox"][0], big["bbox"][3] - big["bbox"][1]))
        else:
            el = now - trial["t0"]
            for key, btn in (("R", "右键"), ("L", "左键")):
                if press[key] and trial[key.lower()] is None and el <= giveup:
                    trial[key.lower()] = el
                    emit(f"    {btn} 反应: {el:.3f}s")
            # 双键都已按下 → 立即完成, 不再等 giveup / 目标消失
            if trial["r"] is not None and trial["l"] is not None:
                rx.append(trial["r"])
                lx.append(trial["l"])
                emit("    [完成] 双键已记录")
                trial_end_t, trial = now, None
            elif big is None:
                if trial["lost_t"] is None:
                    trial["lost_t"] = now
                elif now - trial["lost_t"] >= lost_s and el <= giveup:
                    # 目标消失: 一次都没按过才作废; 已按过则保留记录完成
                    if trial["r"] is None and trial["l"] is None:
                        void_lost += 1
                        emit("    [作废] 目标消失未反应")
                    else:
                        if trial["r"] is not None:
                            rx.append(trial["r"])
                        if trial["l"] is not None:
                            lx.append(trial["l"])
                        emit("    [完成] 目标消失前已按下, 保留记录")
                    trial_end_t, trial = now, None
            elif el > giveup:
                if trial["r"] is None and trial["l"] is None:
                    void_nokey += 1
                    emit("    [作废] %.1fs 内无按键" % giveup)
                else:
                    if trial["r"] is not None:
                        rx.append(trial["r"])
                    if trial["l"] is not None:
                        lx.append(trial["l"])
                trial_end_t, trial = now, None

        wait = interval - (time.perf_counter() - t_loop)
        if wait > 0:
            time.sleep(wait)

    def _stat(tag, arr):
        if not arr:
            emit(f"  {tag}: 无有效样本")
            return
        arr_ms = [v * 1000.0 for v in arr]
        emit("  %s: 平均 %.0fms  中位 %.0fms  最快 %.0fms  最慢 %.0fms  (n=%d)"
             % (tag, statistics.mean(arr_ms), statistics.median(arr_ms),
                min(arr_ms), max(arr_ms), len(arr_ms)))

    emit("=" * 60)
    emit("反应测试汇总: 有效 右键 %d / 左键 %d (作废: 无按键 %d, 目标消失 %d)"
         % (len(rx), len(lx), void_nokey, void_lost))
    _stat("右键(开镜)", rx)
    _stat("左键(触发)", lx)
    emit(f"(测得值含 ~{interval*500:.0f}ms 采样/推理常数偏移, 相对快慢有效)")


def _auto_trigger_worker(config, model_file, camera, cam_lock, stop_event, emit):
    """自动触发线程: 独立 640x640 识别, 见到目标框(最大边≥min_size 且置信度≥conf)
    就合成**按住右键**(瞄准热键=鼠标右键, 现有轮询器会自动启动瞄准管线移动鼠标);
    目标消失≥grace_ms 后松开右键, 瞄准管线随之停止。

    - 运行期开关: _AUTO_TG["enabled"], GUI 勾选/热键(F6 可改)随时切换;
      关闭时若正按住右键, 立即松开。
    - 检测器为本线程独立 ONNX 会话, 推理经全局 _INFER_LOCK 串行化(见 _reaction_test_worker)。
    - 线程退出(finally)必定松开右键, 不会残留按住状态。
    """
    at = config.get("auto_trigger", {}) or {}
    min_size = float(at.get("min_size", 80.0))
    conf_th = float(at.get("conf", 0.65))
    interval = float(at.get("interval_ms", 75.0)) / 1000.0
    size = max(128, int(at.get("size", 640)))
    grace = float(at.get("grace_ms", 300.0)) / 1000.0
    classes = set(config.get("target_classes", [0, 1]))
    det_conf = float(config.get("conf", 0.35))

    try:
        detector = _create_detector(model_file, det_conf, float(config.get("iou", 0.5)),
                                    emit=emit)
        emit("自动触发: 检测器会话就绪(独立于瞄准)")
    except Exception as e:
        emit(f"自动触发: 检测器加载失败({e}), 线程退出")
        return

    sw_, sh_ = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def grab():
        try:
            if camera is not None:
                with cam_lock:
                    fr = camera.grab(region=(max(0, sw_ // 2 - size // 2),
                                             max(0, sh_ // 2 - size // 2),
                                             min(sw_, sw_ // 2 + size // 2),
                                             min(sh_, sh_ // 2 + size // 2)),
                                     copy=True, new_frame_only=False)
                if fr is not None:
                    return np.asarray(fr)
                # dxcam 拿不到帧时回落 PIL, 不让本线程空转
            from PIL import ImageGrab
            img = np.asarray(ImageGrab.grab(
                bbox=(max(0, sw_ // 2 - size // 2), max(0, sh_ // 2 - size // 2),
                      min(sw_, sw_ // 2 + size // 2), min(sh_, sh_ // 2 + size // 2)),
                all_screens=True))
            return img[..., ::-1].copy()
        except Exception:
            return None

    holding = False
    last_seen = 0.0
    _fail = 0
    try:
        while not stop_event.is_set():
            t_loop = time.perf_counter()
            try:
                with _AUTO_TG["lock"]:
                    enabled = _AUTO_TG["enabled"]
                if not enabled:
                    if holding:
                        holding = False
                        try:
                            set_right_hold(False)
                        except Exception:
                            pass
                        emit("自动触发: 已关闭, 松开右键")
                    time.sleep(interval)
                    continue

                frame = grab()
                big = None
                if frame is not None:
                    try:
                        dets = detector.detect(frame)
                        _fail = 0
                    except Exception as e:
                        dets = []
                        _fail += 1
                        if _fail == 1 or _fail % 10 == 0:
                            emit("自动触发: 识别失败 x%d (%s: %s)"
                                 % (_fail, type(e).__name__, e))
                        if _fail >= 10:
                            # 大概率是 DML 会话报废(设备错误): 重建会话自愈
                            emit("自动触发: 识别连续失败, 重建检测会话…")
                            try:
                                detector = _create_detector(
                                    model_file, det_conf, float(config.get("iou", 0.5)),
                                    emit=emit)
                                _fail = 0
                            except Exception as e2:
                                emit("自动触发: 会话重建失败(%s), 稍后重试" % e2)
                                _fail = 9  # 再失败一次就重试重建
                    best = None
                    for d in dets or []:
                        try:
                            if int(d.get("cls", -1)) not in classes:
                                continue
                            if float(d.get("conf", 0)) < conf_th:
                                continue
                            bx = d["bbox"]
                            if max(bx[2] - bx[0], bx[3] - bx[1]) < min_size:
                                continue
                            if best is None or (bx[2] - bx[0]) * (bx[3] - bx[1]) > \
                                    (best["bbox"][2] - best["bbox"][0]) * \
                                    (best["bbox"][3] - best["bbox"][1]):
                                best = d
                        except Exception:
                            continue  # 单条坏检测不致命
                    big = best

                if big is not None:
                    last_seen = time.perf_counter()
                    if not holding:
                        holding = True
                        try:
                            set_right_hold(True)
                        except Exception as e:
                            emit(f"自动触发: 按住右键失败({e})")
                            holding = False
                        else:
                            bx = big["bbox"]
                            emit("自动触发: 目标出现 (%dx%d, conf=%.2f), 按住右键"
                                 % (bx[2] - bx[0], bx[3] - bx[1], float(big.get("conf", 0))))
                elif holding and time.perf_counter() - last_seen >= grace:
                    holding = False
                    try:
                        set_right_hold(False)
                    except Exception:
                        pass
                    emit("自动触发: 目标消失, 松开右键")
            except Exception as e:
                # 最外层兜底: 任何未预期异常都只记日志并继续, 线程永不静默死亡
                emit("自动触发: 循环异常(%s: %s), 继续" % (type(e).__name__, e))

            # 瞄准进行中(正按住右键)时降频识别: 把 GPU 让给瞄准管线, 减少 _INFER_LOCK 争用;
            # 松开判定由 grace 控制(降频后检测间隔仍 < 宽限, 不会明显延迟松开)。
            _iv = interval * 3 if holding else interval
            wait = _iv - (time.perf_counter() - t_loop)
            if wait > 0:
                time.sleep(wait)
    finally:
        if holding:
            holding = False
            try:
                set_right_hold(False)
            except Exception:
                pass
            emit("自动触发: 线程退出, 已松开右键")


def run_engine(config=None, stop_event=None, run_id=None, gui_submit=None,
               ready_callback=None):
    global _MANUAL_RECOIL_CONTROLLER
    if stop_event is None or stop_event.is_set():
        stop_event = threading.Event()
    if config is None:
        config = load_config()
    else:
        # Callers such as the GUI normally pass load_config() output, but keep
        # direct callers on the same legacy-field compatibility path.
        config = merge_config_defaults(config, DEFAULT)

    # Each engine invocation owns its sink.  The local binding is captured by
    # all worker closures, so a later invocation cannot replace this run's
    # file handler or GUI receiver.  ``mouse_log_enabled`` is the GUI's
    # debug-log switch: the complete event stream is still retained in the
    # run file (and necessary status events can still reach the GUI), while
    # the background stdout channel is completely disabled when it is off.
    debug_logging_enabled = bool(
        config.get("mouse_log_enabled", DEFAULT["mouse_log_enabled"]))
    # Keep legacy module-level logging from bypassing the runtime sink.  The
    # GUI/file pipeline remains available, but the process stdout handler is
    # silent when debug logging is disabled.
    _ch.setLevel(logging.DEBUG if debug_logging_enabled else logging.CRITICAL + 1)
    _run_log = RunLogSink(
        _base_dir(), run_id=run_id or new_run_id(), gui_submit=gui_submit,
        stdout_enabled=debug_logging_enabled,
        stdout_minimal=True,
        gui_minimal=True).start()
    _emit = _run_log.emit
    _emit(f"[日志] 本次运行日志 → {_run_log.path}", event_type="lifecycle",
          fields={"run_id": _run_log.run_id})

    # 确保 humanize 完整
    hu = dict(DEFAULT["humanize"])
    hu.update(config.get("humanize", {}))
    # 从 aim_control 中读取 smoothing，注入 humanize 字典供 aim_move 使用
    ac = config.get("aim_control", {})
    hu["smoothing"] = float(ac.get("smoothing", 0.35))
    # 除 humanize 外,全部瞄准参数由 MainEngine 从 config 直接读取(见 aim_engine.py)。
    config["humanize"] = hu
    # ── 真实瞄准核心(共用,零漂移)──
    engine = MainEngine(config)
    _last_aim_t = [0.0]   # 引擎节流用真实帧间隔
    aim_mode = str(config.get("aim_mode", "smooth")).lower()

    # ── 串行管线(识别+移动=一节, 一等到底)──
    # 发完移动后连续抓帧, 直到"误差已按发送方向缩小"= 移动真正生效, 才进入下一节。
    # 生效前绝不拿旧画面算新误差(那是脱节振荡的来源); 生效延迟是多少由程序每轮
    # 实时感应(快就少等、慢就多等), 不依赖对游戏帧率/延迟的任何假设。
    # move_digest_ms 只是防死锁下限(目标消失/输入失效时防卡死), 正常路径不触发。
    move_digest_ms = float(ac.get("move_digest_ms", 55.0))
    # 满额纠错(unit)大位移硬上限(counts): 防误检直接甩飞; 串行验证下误差始终新鲜
    unit_max_counts = float(ac.get("unit_max_counts", 800.0))
    # 固定节拍串行参数(unit 块): 每 frame_ms 截图识别一次(默认75ms), 每张识别图
    # YOLO 每 frame_ms 发布一次最新计划；move_steps 表示共享鼠标输出器
    # 的缓出分配周期数，单周期由 recoil.output_period_ms 配置。
    _ucfg = config.get("unit", {}) or {}
    unit_frame_s = max(0.02, float(_ucfg.get("frame_ms", 75.0))) / 1000.0
    unit_move_steps = max(1, int(_ucfg.get("move_steps", 2)))
    unit_step_profile = str(_ucfg.get("step_profile", "ease"))  # ease=缓出(先快后慢) / even=匀速
    # ── 识别调度(fixed/asap)与标称整定周期 ──
    # fixed: 严格保持旧 frame_ms 固定节拍(基线对照); asap: 上一帧完成后立即开始
    # 下一帧, 识别频率由实际端到端耗时决定。reference_frame_ms 只用于把"每帧
    # 参数"(后坐力强度/TTL 等)换算到真实时间, 不作为 asap 的等待时间。
    frame_schedule = _frame_schedule_of(_ucfg)
    asap_mode = frame_schedule == "asap"
    unit_control_strategy = str(
        _ucfg.get("control_strategy", "rate_hold")).lower()
    if unit_control_strategy not in ("rate_hold", "remaining_error"):
        unit_control_strategy = "rate_hold"
    _remaining_cfg = _ucfg.get("remaining_error", {}) or {}
    # Resolve this section before the startup summary reads its output period.
    _recoil_cfg = config.get("recoil", {}) or {}
    remaining_controller = None
    _remaining_stale_latched = [False]
    effective_prediction_mode = str(
        _ucfg.get("prediction_mode", ac.get("prediction_mode", "current"))
    ).lower()
    if effective_prediction_mode not in ("current", "arrival"):
        effective_prediction_mode = "current"
    if aim_mode == "unit" and unit_control_strategy == "remaining_error":
        remaining_controller = RemainingErrorController(
            px_per_count=(
                float(_ucfg.get("px_per_count", 0.3011)),
                float(_ucfg.get("px_per_count_y", 0.2684)),
            ),
            tau_s=max(0.001, float(_remaining_cfg.get("tau_ms", 40.0)) / 1000.0),
            max_speed_counts_s=(
                float(_remaining_cfg.get("max_speed_counts_s", 4000.0)),
                float(_remaining_cfg.get("max_speed_counts_s", 4000.0)),
            ),
            max_accel_counts_s2=(
                float(_remaining_cfg.get("max_accel_counts_s2", 0.0)),
                float(_remaining_cfg.get("max_accel_counts_s2", 0.0)),
            ),
            deadzone_px=float(_ucfg.get("deadzone", 3.0)),
            stop_deadzone_px=float(_ucfg.get("stop_deadzone", 1.0)),
            feedback_delay_s=max(
                0.0, float(_remaining_cfg.get("feedback_delay_ms", 20.0)) / 1000.0),
            max_observation_age_s=max(
                0.001, float(_remaining_cfg.get("max_observation_age_ms", 120.0)) / 1000.0),
        )
        # Prediction is selected once here and is the only lead producer for
        # the remaining-error path.  The residual controller itself never adds
        # another delay term.
        engine.set_prediction_mode(effective_prediction_mode)
    reference_frame_s = _reference_frame_seconds(_ucfg)
    # 瞄准控制状态的新鲜度 TTL: 超过该时长未发布新控制即归零(防旧指令无限使用)。
    _aim_freshness = ObservationFreshness(reference_frame_s)
    _aim_fresh_ttl_s = [
        _aim_freshness.ttl_s if asap_mode else unit_frame_s * 2.5
    ]
    _seg_next_t = [0.0]   # 下一节(下一张识别图)的预定时刻(仅 fixed 使用)
    _move_sent_t = [0.0]   # 上次真实发送的时刻(0=未发送过)
    _last_net = [0.0, 0.0]  # 上发移动的净位移(counts, 判生效方向用)
    _E_sent = [0.0, 0.0]    # 上发时刻的目标-准星屏幕误差(纯几何, 与验证口径一致)

    # ── 模型路径 ──
    model_file = config.get("model", DEFAULT["model"])
    if getattr(sys, "frozen", False):
        alt = os.path.join(sys._MEIPASS, os.path.basename(model_file))
        if not os.path.exists(alt):
            alt = os.path.join(os.path.dirname(sys.executable), model_file)
        if os.path.exists(alt):
            model_file = alt
        else:
            raise FileNotFoundError(f"找不到模型: {model_file}")
    else:
        model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  os.path.basename(model_file))

    output_dir = config.get("output_dir", "outputs")
    conf_thres = config.get("conf", DEFAULT["conf"])
    iou_thres = config.get("iou", DEFAULT["iou"])
    hotkeys_cfg = normalize_hotkeys(config.get("hotkeys", DEFAULT["hotkeys"]), VK)
    _emit(
        f"[配置] 路径={os.path.abspath(CONFIG_PATH)} "
        f"瞄准={hotkeys_cfg.get('aim', '?')} "
        f"截图={hotkeys_cfg.get('screenshot', '?')} "
         f"压枪={hotkeys_cfg.get('recoil', '?')} "
         f"target_priority={str(config.get('target_priority', 'body')).lower()} "
         f"chest_ratio={config.get('chest_ratio', 0.10)} "
        f"head_ratio={config.get('head_ratio', 0.70)} "
        f"frame_schedule={frame_schedule} "
        f"frame_ms={unit_frame_s * 1000.0:.1f} "
        f"reference_frame_ms={reference_frame_s * 1000.0:.1f} "
        f"move_steps={unit_move_steps} "
        f"control_strategy={unit_control_strategy} "
        f"prediction_mode={effective_prediction_mode} "
        f"output_period_ms={float(_recoil_cfg.get('output_period_ms', 30.0)):.1f} "
        f"px_per_count=({float(_ucfg.get('px_per_count', 0.3011)):.4f},"
        f"{float(_ucfg.get('px_per_count_y', 0.2684)):.4f}) "
        f"target_filter={str(_ucfg.get('target_filter_mode', ac.get('lock_box_filter_mode', 'unknown')))} "
        f"target_filter_alpha={float(ac.get('lock_box_smoothing_alpha', 0.0)):.3f} "
        f"tau_ms={float(_remaining_cfg.get('tau_ms', 40.0)):.1f} "
        f"feedback_delay_ms={float(_remaining_cfg.get('feedback_delay_ms', 20.0)):.1f}"
    )
    anti_recoil = bool(config.get("anti_recoil", DEFAULT["anti_recoil"]))
    _default_recoil_cfg = config.get("default_recoil", {}) or {}
    default_recoil_enabled = bool(_default_recoil_cfg.get("enabled", False))
    default_recoil_strength = max(
        0.0, float(_default_recoil_cfg.get(
            "strength", config.get("recoil_strength", 4.0))))
    _manual_recoil_cfg = config.get("manual_recoil", {}) or {}
    manual_recoil_enabled = bool(
        _manual_recoil_cfg.get("enabled", True)) and not default_recoil_enabled
    # Only one user-facing recoil path may run. The experimental visual loop
    # stays hidden/off when either restored mode is selected.
    if manual_recoil_enabled or default_recoil_enabled:
        anti_recoil = False
    recoil_mode = (
        "manual" if manual_recoil_enabled else
        "default_vertical" if default_recoil_enabled else
        "legacy_visual" if anti_recoil else "off")
    _emit(
        f"[后坐力配置] 模式={recoil_mode} 手动={'开' if manual_recoil_enabled else '关'} "
        f"默认垂直={'开' if default_recoil_enabled else '关'} "
        f"曲线={str(_manual_recoil_cfg.get('profile_name', '') or '默认')}"
    )
    recoil_key = hotkeys_cfg.get("recoil", "mouse_left")
    precision_cfg = config.get("precision", "fp32")
    aim_interval_ms = config.get("aim_interval_ms", DEFAULT["aim_interval_ms"])
    aim_interval_sec = aim_interval_ms / 1000.0  # 转换为秒
    capture_size = max(128, int(config.get("capture_size", DEFAULT["capture_size"])))
    crosshair_offset_x = float(config.get("crosshair_offset_x", 0.0))
    crosshair_offset_y = float(config.get("crosshair_offset_y", 0.0))
    # 暴力直瞄模式：跳过 EMA/平滑/人手模拟，误差直接换算成位移一步到位
    recoil_controller = RecoilController(
        profile=str(config.get("recoil_profile", "unknown")),
        recovery_ms=float(_recoil_cfg.get("recovery_ms", 100.0)),
        bin_ms=float(_recoil_cfg.get("curve_bin_ms", 60.0)),
        calibration_runs=int(_recoil_cfg.get("calibration_runs", 5)),
        min_calibration_bins=int(_recoil_cfg.get("min_calibration_bins", 10)),
        max_target_gap_ms=float(_recoil_cfg.get("max_target_gap_ms", 120.0)),
    )
    manual_recoil_controller = ManualRecoilController(
        bin_ms=float(_manual_recoil_cfg.get("curve_bin_ms", 60.0)),
        calibration_runs=int(_manual_recoil_cfg.get("calibration_runs", 5)),
        min_burst_ms=float(_manual_recoil_cfg.get("min_burst_ms", 300.0)),
        tail_vertical_ratio=float(
            _manual_recoil_cfg.get("tail_vertical_ratio", 0.30)),
        tail_max_ms=float(_manual_recoil_cfg.get("tail_max_ms", 1500.0)),
    )
    _MANUAL_RECOIL_CONTROLLER = manual_recoil_controller
    selected_manual_profile = str(
        _manual_recoil_cfg.get("profile_name", "") or "").strip()
    manual_profile_loaded = False
    manual_profile_warning = ""
    manual_recoil_available = False
    raw_capture_ready = False
    if selected_manual_profile:
        try:
            _profile_store = RecoilProfileStore(
                os.path.join(_base_dir(), "recoil_profiles.json"))
            manual_recoil_controller.load_profile(
                _profile_store.load(selected_manual_profile),
                name=selected_manual_profile)
            manual_profile_loaded = True
        except (ProfileStoreError, ValueError) as exc:
            manual_profile_warning = str(exc)
            selected_manual_profile = ""
    recoil_gate = RecoilChordGate()
    motion_arbiter = MotionArbiter(namespace=_run_log.run_id)
    fire_wake = threading.Event()
    active = False
    recoil_active = False
    _control_lock = threading.RLock()
    # One transaction spans quantization, the final SendInput authorization,
    # and the successful-ledger commit.  Scope resets/cancel use the same
    # lock, so a reset cannot erase a packet between OS submission and commit.
    _motion_transaction_lock = threading.RLock()
    # Button/cancel generation is authoritative at capture, publish and send
    # boundaries.  It is separate from the mutable diagnostic state below.
    _aim_epoch = AimControlEpoch()
    _recoil_epoch = RecoilControlEpoch()
    _stop_watch_done = threading.Event()
    # Latest-value mailbox: vision replaces the complete snapshot; output
    # never consumes a queue of obsolete boxes/plans.
    _latest_observation = LatestObservation()
    _session_id = [0]
    _observation_id = [0]
    _target_id_next = [0]
    _current_target_id = [None]
    _last_real_observation_s = [None]
    _control_state = {
        "target_valid": False, "aim_active": False, "frame": 0,
        "session_id": 0, "target_id": None, "observation": None,
        "aim_token": None,
        "recoil_token": None,
        "diag": None, "last_publish": 0.0,
        "plan_active": False, "last_publish_history": 0.0,
        "last_clear_reason": "never_published", "last_clear_t": 0.0,
        "manual_recoil_blend_percent": max(
            0.0, min(100.0, float(
                _manual_recoil_cfg.get("playback_blend_percent", 70.0)))),
        "manual_recoil_suppress_prediction": bool(
            _manual_recoil_cfg.get("suppress_prediction_during_replay", True)),
    }
    # 最终位移硬上限(px)：直接限制发给鼠标的实际位移，不依赖 smoothing/增益/灵敏度。
    # 写入 mouse_control 模块的 _MAX_TARGET_STEP，aim_move(cap=None) 自动使用。0=不限制。
    set_max_target_step(config.get("max_target_step", DEFAULT.get("max_target_step", 80.0)))

    def _clear_current_aim(reason):
        """Atomically drop the active/pending aim plan and its active timestamp."""
        with _motion_transaction_lock:
            motion_arbiter.clear_aim()
            with _control_lock:
                was_active = bool(_control_state.get("plan_active"))
                _control_state["plan_active"] = False
                _control_state["last_publish"] = 0.0
                _control_state["last_clear_reason"] = (
                    reason if was_active else _control_state.get(
                        "last_clear_reason", "never_published"))
                _control_state["last_clear_t"] = time.perf_counter()

    def _cancel_aim(reason="aim_released"):
        """Invalidate old capture results before clearing their output plan."""
        # Lock ordering is transaction -> epoch everywhere.  This prevents a
        # cancel waiting on an output that is itself waiting for the epoch.
        with _motion_transaction_lock:
            _aim_epoch.cancel()  # waits for an in-flight SendInput to finish
            motion_arbiter.clear_aim()
            with _control_lock:
                _control_state["aim_active"] = False
                _control_state["target_valid"] = False
                _control_state["plan_active"] = False
                _control_state["aim_token"] = None
                _control_state["last_publish"] = 0.0
                _control_state["last_clear_reason"] = reason
                _control_state["last_clear_t"] = time.perf_counter()

    def _cancel_recoil(reason="recoil_released"):
        with _motion_transaction_lock:
            _recoil_epoch.cancel()
            with _control_lock:
                _control_state["recoil_token"] = None

    def _stop_watch():
        # Invalidate send authority as soon as the run stop is requested; do
        # not wait for the consumer/output loop's next scheduling point.
        stop_event.wait()
        if not _stop_watch_done.is_set():
            _cancel_aim("engine_stop")
            _cancel_recoil("engine_stop")

    _stop_watch_thread = threading.Thread(
        target=_stop_watch, name="AimStopWatch-%s" % _run_log.run_id,
        daemon=True)
    _stop_watch_thread.start()

    os.makedirs(output_dir, exist_ok=True)

    # ── 模型 (自动选择 optimized 版本) ──
    model_dir = os.path.dirname(model_file)
    model_name = os.path.basename(model_file)
    opt_name = model_name.replace(".onnx", "_optimized.onnx")
    opt_path = os.path.join(model_dir, opt_name)
    if os.path.exists(opt_path):
        _emit(f"使用 optimized 模型: {opt_name}")
        model_file = opt_path

    # 设置全局 precision 供 _create_detector 使用
    global _DETECTOR_PRECISION
    _DETECTOR_PRECISION = precision_cfg

    _emit(f"正在加载 {os.path.basename(model_file)} …")
    t0 = time.time()
    try:
        detector = _create_detector(model_file, conf_thres, iou_thres,
                                    emit=_emit)
    except Exception as e:
        _emit(f"模型加载失败: {e}")
        traceback.print_exc()
        _stop_watch_done.set()
        stop_event.set()
        _run_log.close(timeout_s=1.0)
        return
    _emit(f"模型就绪 ({(time.time() - t0) * 1000:.0f} ms)")
    _CLASS_NAMES = detector._class_names  # 用于画框

    # ── 高精度定时器（默认 ~15.6ms → 1ms，否则 poll/Sleep 会把 50Hz 压成 ~30Hz）──
    _timer_period_set = False
    if winmm is not None:
        try:
            if winmm.timeBeginPeriod(1) == 0:
                _timer_period_set = True
                _emit("系统定时器: 1ms 精度已启用")
        except Exception as e:
            _emit(f"系统定时器: timeBeginPeriod 失败 ({e})")

    # ── 截图 ──
    _cam_lock = threading.Lock()   # dxcam 相机为单一实例: 瞄准抓帧与反应测试抓帧互斥
    camera = None
    camera_ok = False
    try:
        import dxcam as _dc
        camera = _dc.create(output_color="BGR")
        # 使用单帧区域抓取；不启动全屏 video_mode 环形缓冲。
        camera_ok = True
        _emit("截图: dxcam 区域采集已就绪")
    except Exception as e:
        _emit(f"dxcam 不可用 ({e})，降级 PIL")
        camera = None
        try:
            from PIL import ImageGrab
            _emit("截图: PIL 备用")
        except ImportError:
            _emit("错误: dxcam 和 PIL 都不可用！")
            _stop_watch_done.set()
            stop_event.set()
            _run_log.close(timeout_s=1.0)
            return

    # ── 反应时间测试线程(可选, 与瞄准并行; 只识别记录, 不动鼠标) ──
    if (config.get("reaction_test", {}) or {}).get("enabled", False):
        prev = _RT_RUN["thread"]
        if prev is not None and prev.is_alive():
            # 旧 worker 还活着(上一次引擎实例未退出/双引擎并存): 先停旧再起新,
            # 避免同一屏幕事件被两个 worker 重复识别、试次号各记各的。
            old_stop = _RT_RUN["stop"]
            if old_stop is not None:
                old_stop.set()
            prev.join(timeout=0.8)
            _emit("反应测试: 已停旧线程, 重建新线程")
        _RT_RUN["stop"] = stop_event
        _RT_RUN["thread"] = threading.Thread(target=_reaction_test_worker,
                                             args=(config, model_file, camera,
                                                   _cam_lock, stop_event, _emit),
                                             daemon=True)
        _RT_RUN["thread"].start()
        _emit("反应测试线程已启动(与瞄准并行; 右侧日志看试次, ESC 仅结束测试)")

    # ── 自动触发线程(可选): 独立识别, 目标出现自动按住右键启动瞄准 ──
    if (config.get("auto_trigger", {}) or {}).get("enabled", False):
        prev_at = _AT_RUN["thread"]
        if prev_at is not None and prev_at.is_alive():
            old_stop = _AT_RUN["stop"]
            if old_stop is not None:
                old_stop.set()
            prev_at.join(timeout=0.8)
            _emit("自动触发: 已停旧线程, 重建新线程")
        _AT_RUN["stop"] = stop_event
        with _AUTO_TG["lock"]:
            _AUTO_TG["enabled"] = True
        _at_args = (config, model_file, camera, _cam_lock, stop_event, _emit)
        _AT_RUN["args"] = _at_args
        _AT_RUN["thread"] = threading.Thread(target=_auto_trigger_worker,
                                             args=_at_args, daemon=True)
        _AT_RUN["thread"].start()
        _emit("自动触发线程已启动(识别到目标自动按住右键; F6 可开关)")

    # ── 状态 ──
    lock = threading.Lock()
    counter = [1]
    _img_center = [0.0, 0.0]
    _capture_log_size = [None]
    # ── 时间节流(发送间隔)与全部瞄准状态已移入 MainEngine(见 aim_engine.py)。
    # 设计初衷(保留):按"时间"而非"帧数"节流,确保一次移动在截图里生效后才
    # 发下一次,避免旧误差堆积过冲。相关参数由引擎从 config 读取。
    mouse_log_enabled = bool(config.get("mouse_log_enabled", True))  # 每次实际移动都打日志

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # ── 性能计时器 ──
    perf_timings = {
        "capture": [],
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "find_target": [],
        "mouse_move": [],
        "end_to_end": [],
    }
    perf_log_interval = 30
    _yolo_idle_e2e = deque(maxlen=120)
    _yolo_fire_e2e = deque(maxlen=120)
    fps_counter = {"count": 0, "last_time": time.perf_counter(), "active": False}
    # ── 帧调度诊断: 截图/观测间隔、发布间隔、输出 tick、控制年龄、stale 清除 ──
    _sched_diag = {
        "capture_intervals": deque(maxlen=300),   # 两次截图时间戳间隔(ms)
        "observation_dts": deque(maxlen=300),     # 引擎实际使用的 observation dt(ms)
        "publish_intervals": deque(maxlen=300),   # 两次瞄准控制发布间隔(ms)
        "tick_dts": deque(maxlen=300),            # 输出线程真实 tick dt(ms)
        "control_ages": deque(maxlen=300),        # 输出时刻控制状态年龄(ms)
        "artificial_waits": deque(maxlen=300),    # fixed 节拍实际等待(ms), asap≈0
        "stale_rate_clears": 0,                   # TTL 超时导致的控制归零次数
        "stale_control_clears": 0,
        "tick_stall_count": 0,
        "discarded_dt_ms": 0.0,
        "max_actual_tick_dt_ms": 0.0,
        "recent_observation_dt_ms": reference_frame_s * 1000.0,
        "observation_dt_ewma_ms": reference_frame_s * 1000.0,
        "last_obs_t": 0.0,
        "last_publish_t": 0.0,
        "last_tick_log_t": 0.0,
    }
    diagnostics = {
        "frames": 0,
        "detections": 0,
        "detection_summary": [],
        "candidates": 0,
        "selection_reason": None,
        "target": None,
        "last_delta": None,
        "aim_center": None,
        "move_calls": 0,
        "image_shape": None,
        "move_sent": 0,
        "move_successes": 0,
        "move_failures": 0,
        "idle_tick": 0,
        "fractional_wait": 0,
        "component_cancel": 0,
        "send_attempt": 0,
        "send_success": 0,
        "send_failed": 0,
        "diagnostic_errors": 0,
        "output_fault": None,
        "move_skipped": {},
        "screenshot_frames": 0,
        "screenshot_detections": 0,
        "capture_errors": 0,
        "dxcam_empty": 0,
        "pil_fallbacks": 0,
        "capture_color": None,
        "aim_errors": 0,
        "last_error": None,
    }
    diagnostics_lock = threading.Lock()
    error_last_emit = {}

    def _diagnostic_error(stage, exc):
        """记录错误并限频输出，避免异常时日志线程再次被刷爆。"""
        now = time.perf_counter()
        with diagnostics_lock:
            diagnostics["last_error"] = f"{stage}: {type(exc).__name__}: {exc}"
            if stage == "capture":
                diagnostics["capture_errors"] += 1
            elif stage == "aim":
                diagnostics["aim_errors"] += 1
            last = error_last_emit.get(stage, 0.0)
            should_emit = now - last >= 1.0
            if should_emit:
                error_last_emit[stage] = now
        if should_emit:
            _emit(
                f"[异常][{stage}] {type(exc).__name__}: {exc}",
                event_type="error",
                fields={"stage": stage,
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc()},
            )

    def _diagnostic_snapshot():
        with diagnostics_lock:
            return dict(diagnostics)

    _emit(f"截图: 固定屏幕中心 {capture_size}x{capture_size}px  准星偏移=({crosshair_offset_x:.1f},{crosshair_offset_y:.1f})")
    _capture_aim_center = [capture_size / 2.0, capture_size / 2.0]  # fallback

    # ── 红点自动检测（子弹实际落点）──
    # 实战问题：枪战游戏开枪后瞄准镜向上飘，子弹跟着镜走——子弹实际落点
    # （镜上的红色小点）与截图中心不重合，且每把枪偏移不同。
    # 程序瞄准参考点必须跟着红点走，否则前几发总是打高。
    # 模式（crosshair_mode）：
    #   center       = 只认屏幕中心（不检测红点，用手动 crosshair_offset）
    #   dot_only     = 只认红点（找不到时保持上一帧红点位置，从未找到才用中心）
    #   dot_fallback = 允许红点退化（找不到时回退屏幕中心）
    # 兼容旧配置：crosshair_auto=true → dot_fallback，false → center。
    if config.get("crosshair_auto") is not None:
        crosshair_mode = "dot_fallback" if bool(config.get("crosshair_auto")) else "center"
    else:
        crosshair_mode = config.get("crosshair_mode", "dot_fallback")
    if crosshair_mode not in ("center", "dot_only", "dot_fallback"):
        crosshair_mode = "dot_fallback"
    crosshair_search_radius = max(40, int(config.get("crosshair_search_radius", 150)))
    crosshair_tracker = CrosshairTracker(
        max_center_offset=float(config.get("crosshair_max_offset", 45.0)),
        max_jump=float(config.get("crosshair_max_jump", 12.0)),
        confirm_frames=int(config.get("crosshair_confirm_frames", 2)),
        hold_frames=int(config.get("crosshair_hold_frames", 2)),
        ema_alpha=float(config.get("crosshair_ema_alpha", 0.35)),
        fallback_decay=float(config.get("crosshair_fallback_decay", 0.10)),
        shot_vertical_jump=float(_recoil_cfg.get("shot_vertical_jump_px", 60.0)),
        shot_horizontal_jump=float(_recoil_cfg.get("shot_horizontal_jump_px", 35.0)),
        shot_center_offset=float(_recoil_cfg.get("shot_center_offset_px", 85.0)),
        shot_ema_alpha=float(_recoil_cfg.get("shot_ema_alpha", 1.0)),
        return_frames=int(config.get("crosshair_return_frames", 4)),
        return_max_step=float(config.get("crosshair_return_max_step", 24.0)),
        frame_schedule=frame_schedule,
        reference_frame_ms=reference_frame_s * 1000.0,
    )
    _dot_status = ["center"]
    _dot_jump = [0.0, 0.0]
    _dot_last_obs_t = [0.0]

    def _detect_crosshair_dot(img, radius, prev=None, shot_active=False,
                              coordinate_offset=(0.0, 0.0), reference_center=None):
        """在截图中心 ±radius 区域内找子弹落点红点，返回 (x,y) 或 None。

        实战修正：狙击镜自带红色刻度线/刻度块，与红点同为红色——
        "离中心最近"会把刻度当红点（实测黄圈圈错）。红点的区分特征：
          - 圆润：外接框长宽比 0.6~1.7 且填充率 ≥0.45（刻度线细长/空心被过滤；
            程序画的调试十字（十字形）填充率 ~0.19 也被过滤）
          - 最大：满足圆润条件的红块里面积最大者（实测红点 36~49px 恒为最大）
          - 时间连续：若给了 prev（上一帧平滑位置），优先选离 prev ≤25px 的
            红块（镜漂移是连续过程，刻度是固定不动的）
        """
        try:
            ih, iw = img.shape[:2]
            r = img[..., 2].astype(np.int32)
            g = img[..., 1].astype(np.int32)
            b = img[..., 0].astype(np.int32)
            mask = (r > 100) & (r - g > 40) & (r - b > 40)
            local_cx, local_cy = iw / 2.0, ih / 2.0
            cx0, cy0 = (reference_center if reference_center is not None else
                        (local_cx + coordinate_offset[0], local_cy + coordinate_offset[1]))
            x0 = max(0, int(local_cx - radius))
            x1 = min(iw, int(local_cx + radius))
            y0 = max(0, int(local_cy - radius))
            y1 = min(ih, int(local_cy + radius))
            sub = mask[y0:y1, x0:x1]
            if int(sub.sum()) == 0:
                return None
            n, _, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
            cands = []
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                cw = int(stats[i, cv2.CC_STAT_WIDTH])
                ch = int(stats[i, cv2.CC_STAT_HEIGHT])
                if area < 3 or area > 200:
                    continue
                aspect = cw / max(1, ch)
                if aspect < 0.7 or aspect > 1.4:      # 更严格圆润度 → 刻度线/目标红块
                    continue
                fill = area / max(1, cw * ch)
                if fill < 0.55:                        # 更严格填充率 → 空心/十字形/红块
                    continue
                mx = coordinate_offset[0] + x0 + stats[i, cv2.CC_STAT_LEFT] + cw / 2.0
                my = coordinate_offset[1] + y0 + stats[i, cv2.CC_STAT_TOP] + ch / 2.0
                cands.append((area, mx, my))
            if not cands:
                return None
            # 时间先验：优先选择落在可信单帧跳变范围内的候选。
            # 镜红点是 HUD 元素几乎静止；目标红块/刻度随画面移动会被自然淘汰。
            if prev is not None:
                near = [c for c in cands
                        if math.hypot(c[1] - prev[0], c[2] - prev[1])
                        <= crosshair_tracker.max_jump]
                if near:
                    cands = near
                elif shot_active:
                    # 首发窗口只放行“近乎竖直向上”的红点跳变；横向大跳仍是误检。
                    shot = [c for c in cands
                            if math.hypot(c[1] - prev[0], c[2] - prev[1])
                            > crosshair_tracker.max_jump
                            and (c[2] - prev[1]) <= -3.0
                            and -(c[2] - prev[1]) <= crosshair_tracker.shot_vertical_jump
                            and abs(c[1] - prev[0]) <= crosshair_tracker.shot_horizontal_jump
                            and math.hypot(c[1] - cx0, c[2] - cy0)
                            <= crosshair_tracker.shot_center_offset]
                    if shot:
                        cands = shot
                    else:
                        return None
                else:
                    # 交给状态机判为 jump；只有连续两帧稳定后才允许重新采信。
                    cands = [c for c in cands
                             if math.hypot(c[1] - cx0, c[2] - cy0)
                             <= crosshair_tracker.max_center_offset]
                    if not cands:
                        return None
            else:
                # 首次/重新检测：日志最大可信偏移约26px，45px留出后坐力余量。
                cands = [c for c in cands
                         if math.hypot(c[1] - cx0, c[2] - cy0)
                         <= crosshair_tracker.max_center_offset]
                if not cands:
                    return None
            best = max(cands, key=lambda c: c[0])  # 面积最大者
            return (best[1], best[2])
        except Exception:
            return None

    def _get_mouse_capture_region():
        """以主屏幕中心为截图中心。FPS 游戏准星默认位于屏幕中心。
        截图区域与 Windows 鼠标位置完全无关。
        """
        center_x, center_y = screen_w / 2.0, screen_h / 2.0

        virtual_left = user32.GetSystemMetrics(76)
        virtual_top = user32.GetSystemMetrics(77)
        virtual_width = user32.GetSystemMetrics(78)
        virtual_height = user32.GetSystemMetrics(79)
        if virtual_width <= 0 or virtual_height <= 0:
            raise RuntimeError("无法读取虚拟屏幕尺寸")

        size = min(capture_size, virtual_width, virtual_height)
        left = int(round(center_x - size / 2.0))
        top = int(round(center_y - size / 2.0))
        left = max(virtual_left, min(left, virtual_left + virtual_width - size))
        top = max(virtual_top, min(top, virtual_top + virtual_height - size))

        # 准星在截图坐标系中的位置 = 截图正方形中心 + 用户偏移
        _capture_aim_center[0] = size / 2.0 + crosshair_offset_x
        _capture_aim_center[1] = size / 2.0 + crosshair_offset_y
        return left, top, left + size, top + size

    def _capture_from_pil(region):
        from PIL import ImageGrab as _ImageGrab
        image = np.asarray(_ImageGrab.grab(bbox=region, all_screens=True))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _camera_origin_and_size():
        """返回 dxcam 当前输出在虚拟桌面中的位置和尺寸。"""
        try:
            coords = camera._output.desc.DesktopCoordinates
            return (int(coords.left), int(coords.top),
                    int(coords.right - coords.left),
                    int(coords.bottom - coords.top))
        except Exception:
            return 0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def _capture_from_dxcam(region):
        """只向 dxcam 请求鼠标附近区域；无新帧时短暂重试并回退 PIL。"""
        origin_x, origin_y, width, height = _camera_origin_and_size()
        local_region = (region[0] - origin_x, region[1] - origin_y,
                        region[2] - origin_x, region[3] - origin_y)
        if (local_region[0] < 0 or local_region[1] < 0 or
                local_region[2] > width or local_region[3] > height):
            with diagnostics_lock:
                diagnostics["pil_fallbacks"] += 1
            return _capture_from_pil(region)

        for _ in range(2):
            # 动态区域沿用原来可用的最新帧模式；不强制等待新帧。
            with _cam_lock:
                frame = camera.grab(region=local_region, copy=True, new_frame_only=False)
            if frame is not None:
                return frame
            time.sleep(0.0005)

        # dxcam 连续拿不到新帧时，不能让整帧 aim 失败；PIL 使用同一绝对区域兜底。
        with diagnostics_lock:
            diagnostics["dxcam_empty"] += 1
            diagnostics["pil_fallbacks"] += 1
        return _capture_from_pil(region)

    def capture(include_motion_snapshot=False):
        t_cap_start = time.perf_counter()
        try:
            region = _get_mouse_capture_region()
            img = _capture_from_dxcam(region) if camera is not None else _capture_from_pil(region)
            t_cap_end = time.perf_counter()
            # Freeze motion attribution immediately after the image is obtained,
            # before red-dot processing and YOLO inference.  Output generated
            # during inference belongs to the next image, not this stale one.
            _motion_ledger = motion_arbiter.ledger_snapshot(t_cap_end)
            _motion_snapshot = MotionComponents(
                _motion_ledger.aim_total, _motion_ledger.recoil_total)
            _manual_feedforward_at_capture = bool(
                manual_recoil_enabled and manual_recoil_available
                and manual_recoil_controller.pressed
                and manual_recoil_controller.ready
                and manual_recoil_controller.phase in ("REPLAY", "TAIL"))

            # 检测框和控制器都使用局部截图坐标，画面中心就是截图中心。
            ih, iw = img.shape[:2]
            # 对实际返回尺寸做保护：目标/鼠标误差必须使用实际准星中心。
            if iw <= 0 or ih <= 0:
                raise RuntimeError(f"截图尺寸异常: {img.shape}")
            # ── 红点自动检测：瞄准参考点跟随子弹落点红点 ──
            # 开枪后瞄准镜上飘、红点离开截图中心；用 EMA 平滑红点位置作为
            # 瞄准参考点（误差=目标-红点），子弹跟着镜走就能打中。
            center = (iw / 2.0, ih / 2.0)
            if crosshair_mode == "center":
                _capture_aim_center[0] = center[0] + crosshair_offset_x
                _capture_aim_center[1] = center[1] + crosshair_offset_y
                _dot_status[0] = "center"
                _dot_jump[0] = _dot_jump[1] = 0.0
                dot_diag = None
            else:
                _shot_gate = bool(
                    anti_recoil and active and recoil_controller.pressed
                    and (engine.locked_box[0] is not None or recoil_controller.ready)
                    and recoil_controller.is_shot_window(time.perf_counter())
                )
                _dot = _detect_crosshair_dot(
                    img, crosshair_search_radius,
                    prev=crosshair_tracker.preferred_position,
                    shot_active=_shot_gate,
                )
                dot_result = crosshair_tracker.update(
                    _dot, center, crosshair_mode, shot_active=_shot_gate,
                    now_s=t_cap_end,
                )
                _capture_aim_center[0] = dot_result.x
                _capture_aim_center[1] = dot_result.y
                _dot_status[0] = dot_result.status
                _dot_jump[0], _dot_jump[1] = dot_result.jump_x, dot_result.jump_y
                if dot_result.status == "shot_trusted":
                    _obs_now = time.perf_counter()
                    _obs_gap_ms = ((_obs_now - _dot_last_obs_t[0]) * 1000.0
                                   if _dot_last_obs_t[0] else
                                   (reference_frame_s if asap_mode else unit_frame_s) * 1000.0)
                    _dot_last_obs_t[0] = _obs_now
                    _learned = recoil_controller.observe_recoil(
                        dot_result.jump_x, dot_result.jump_y,
                        float(_ucfg.get("px_per_count", 0.3011)),
                        float(_ucfg.get("px_per_count_y", 0.2684)),
                        _obs_now, sample_interval_ms=_obs_gap_ms,
                    )
                    if _learned:
                        _emit(
                            f"[后坐力] 红点跳变=({dot_result.jump_x:+.1f},{dot_result.jump_y:+.1f})px "
                            f"实时闭环已接收 校准={recoil_controller.calibration_count}/5 "
                            f"曲线={recoil_controller.profile_duration_ms:.0f}ms"
                        )
                dot_diag = ((round(dot_result.x, 1), round(dot_result.y, 1))
                            if dot_result.trusted else None)
            with diagnostics_lock:
                diagnostics["crosshair_dot"] = dot_diag
                diagnostics["crosshair_dot_status"] = _dot_status[0]
            _img_center[0] = min(max(_capture_aim_center[0], 0.0), float(iw))
            _img_center[1] = min(max(_capture_aim_center[1], 0.0), float(ih))
            dets = detector.detect(img)

            # 精简诊断：每帧只更新非核心字段，降锁开销
            with diagnostics_lock:
                diagnostics["frames"] += 1
                diagnostics["detections"] = len(dets)
                diagnostics["region"] = region
                diagnostics["image_shape"] = (iw, ih)
                diagnostics["aim_center"] = (round(_img_center[0], 1), round(_img_center[1], 1))
                # crosshair_dot 已在红点检测分支里按本帧结果更新
                if diagnostics["frames"] % perf_log_interval == 0:
                    diagnostics["detection_summary"] = [
                        {"cls": int(d["cls"]), "conf": round(float(d["conf"]), 3), "bbox": d["bbox"]}
                        for d in (dets or [])[:5]
                    ]

            t_cap = (t_cap_end - t_cap_start) * 1000
            perf_timings["capture"].append(t_cap)
            perf_timings["preprocess"].append(detector._timing.get("preprocess", 0))
            perf_timings["inference"].append(detector._timing.get("inference", 0))
            perf_timings["postprocess"].append(detector._timing.get("postprocess", 0))
            if _capture_log_size[0] != capture_size:
                _emit(f"截图区域: 屏幕中心 {capture_size}x{capture_size}px  准星偏移=({crosshair_offset_x:.1f},{crosshair_offset_y:.1f})")
                _capture_log_size[0] = capture_size
            if include_motion_snapshot:
                return (
                    dets, img, _motion_snapshot, t_cap_end,
                    _manual_feedforward_at_capture, _motion_ledger,
                )
            return dets, img
        except Exception as exc:
            _diagnostic_error("capture", exc)
            raise

    # GPU/DirectML 冷启动预热：消除首次瞄准的超高延迟
    _emit("预热推理中…")
    try:
        for _ in range(8):
            capture()
        # 清空预热产生的计时噪声
        for k in perf_timings:
            perf_timings[k].clear()
        fps_counter["count"] = 0
        fps_counter["last_time"] = time.perf_counter()
        _emit("预热完成")
    except Exception as e:
        _emit(f"预热跳过: {e}")

    # ── 目标选择/瞄准点/框增益已全部移入 MainEngine(aim_engine.py)，此处不再复制 ──
    def aim_frame():
        nonlocal fps_counter
        if not hasattr(aim_frame, "_c"):
            aim_frame._c = 0
        aim_frame._c += 1
        # Capture the cancellation/session generation before any blocking
        # screenshot or inference.  A release during inference makes this
        # token stale and the result is discarded before publication.
        _capture_token = _aim_epoch.token()
        # ── 识别帧调度 ──
        # fixed(固定节拍): 每 frame_ms 截图识别一次, 每张识别图只移动一次。
        #   节拍(默认22ms)大于游戏输入→画面延迟, 上一发在下一张识别图前
        #   已自然落地 —— 固定节拍本身就是串行, 不需要任何在途验证/等待机制。
        # asap(跟随推理速度): 无任何 frame_ms 人工等待, 上一帧"截图→预处理→
        #   推理→选点→发布"完成后立即开始下一帧; 人工等待恒为 0。
        if aim_mode == "unit" and not stop_event.is_set():
            _now_c = time.perf_counter()
            _wait_s = _unit_frame_wait_seconds(frame_schedule, _seg_next_t[0], _now_c, unit_frame_s)
            if _wait_s > 0.002:
                # 左键沿会唤醒固定节拍，首发无需白等完整一帧。
                fire_wake.wait(_wait_s)
                fire_wake.clear()
                _sched_diag["artificial_waits"].append(
                    (time.perf_counter() - _now_c) * 1000.0)
            else:
                _sched_diag["artificial_waits"].append(0.0)
            if not asap_mode:
                _seg_next_t[0] = time.perf_counter() + unit_frame_s
        t_e2e_start = time.perf_counter()
        try:
            (dets, img, _capture_motion, _observation_t,
             _manual_feedforward_active, _motion_ledger) = capture(include_motion_snapshot=True)
            if stop_event.is_set() or not _aim_epoch.is_current(_capture_token):
                return
            # _img_center 已在 capture() 中按截图区域与真实准星位置设置
            t_ft_start = time.perf_counter()
            _now_real = time.perf_counter()
            _dt = ((_observation_t - _last_aim_t[0])
                   if _last_aim_t[0] else (1.0 / 120.0))
            _dt = max(1e-4, min(0.5, _dt))
            if asap_mode:
                _aim_freshness.observe(_dt)
                _aim_fresh_ttl_s[0] = _aim_freshness.ttl_s
                _sched_diag["recent_observation_dt_ms"] = (
                    _aim_freshness.recent_dt * 1000.0)
                _sched_diag["observation_dt_ewma_ms"] = (
                    _aim_freshness.ewma_dt * 1000.0)
            if _sched_diag["last_obs_t"]:
                _sched_diag["capture_intervals"].append(
                    (_observation_t - _sched_diag["last_obs_t"]) * 1000.0)
            _sched_diag["last_obs_t"] = _observation_t
            _sched_diag["observation_dts"].append(_dt * 1000.0)
            _last_aim_t[0] = _observation_t
            _observed_motion = motion_arbiter.drain_components_through(
                _capture_motion)
            if _observed_motion.total != (0, 0):
                engine.notify_net(*_observed_motion.total)
            with _control_lock:
                _suppress_recoil_prediction = bool(
                    _control_state["manual_recoil_suppress_prediction"])
            _prediction_suppressed = bool(
                _manual_feedforward_active and _suppress_recoil_prediction)
            engine.set_last_verified(True)   # 固定节拍无在途验证概念; 自标定(auto_cal)默认关
            _recovery = bool(
                anti_recoil and active and recoil_controller.pressed
                and engine.locked_box[0] is not None
                and recoil_controller.recovery_active(_now_real)
            )
            out = engine.compute(
                dets, (_img_center[0], _img_center[1]), _dt,
                recoil_recovery=_recovery,
                processing_latency_s=max(0.0, time.perf_counter() - _observation_t),
                recoil_feedforward_active=_prediction_suppressed,
            )
            t_ft_end = time.perf_counter()
            if stop_event.is_set() or not _aim_epoch.is_current(_capture_token):
                _queue_mouse_log(
                    f"[观测丢弃] frame={aim_frame._c} reason=cancelled_or_new_session "
                    f"capture_t={_observation_t:.6f} finished_t={t_ft_end:.6f}")
                return
            _log_target = out.get("target")
            _log_reason = str(engine.diag_reason or "")
            _log_raw_error = (
                (float(_log_target[0]) - float(_img_center[0]),
                 float(_log_target[1]) - float(_img_center[1]))
                if _log_target is not None else (0.0, 0.0))
            _log_target_id = _current_target_id[0]
            if (_log_target is not None and
                    (_log_target_id is None or _log_reason.startswith("switched_"))):
                _log_target_id = _target_id_next[0] + 1
            if mouse_log_enabled:
                try:
                    _queue_mouse_log(
                        "[瞄准观测] " + json.dumps({
                            "session_id": _session_id[0],
                            "target_id": _log_target_id,
                            "observation_id": _observation_id[0] + 1,
                            "frame": aim_frame._c,
                            "capture_t": round(_observation_t, 6),
                            "image_time_s": round(_observation_t, 6),
                            "image_time_kind": "capture_completed",
                            "capture_completed_s": round(_observation_t, 6),
                            "published_s": round(t_ft_end, 6),
                            "dt": round(_dt, 6),
                            "processing_latency_ms": round(
                                max(0.0, t_ft_end - _observation_t) * 1000.0, 3),
                            "detections": [
                                {"cls": int(d["cls"]), "conf": round(float(d["conf"]), 4),
                                 "bbox": [round(float(v), 3) for v in d["bbox"]]}
                                for d in (dets or [])
                            ],
                            "crosshair": [round(float(_img_center[0]), 3),
                                          round(float(_img_center[1]), 3)],
                            "dot_status": _dot_status[0],
                            "observed_aim": list(_observed_motion.aim),
                            "observed_recoil": list(_observed_motion.recoil),
                            "recoil_feedforward": bool(_prediction_suppressed),
                            "target": ([round(float(target[0]), 3),
                                        round(float(target[1]), 3)]
                                       if (target := out.get("target")) is not None else None),
                            "lock_reason": engine.diag_reason,
                            "observed_error": [round(float(v), 3)
                                               for v in out["observed_error"]],
                            "raw_error": [round(float(v), 3)
                                           for v in _log_raw_error],
                            "filtered_error": [round(float(v), 3)
                                                for v in out["observed_error"]],
                            "control_error": [round(float(v), 3)
                                               for v in (out.get("raw_dx", _log_raw_error[0]),
                                                         out.get("raw_dy", _log_raw_error[1]))],
                            "velocity_raw": [round(float(v), 3)
                                             for v in out["velocity_raw"]],
                            "velocity_filtered": [round(float(v), 3)
                                                  for v in out["velocity_filtered"]],
                            "target_velocity_px_s": [round(float(v), 3)
                                                     for v in out["velocity_filtered"]],
                            "target_velocity_confidence": [
                                1.0 if bool(v) else 0.0 for v in getattr(
                                    engine, "_unit_velocity_valid_axes", (False, False))],
                            "lock_filter_mode": engine.lock_box_filter_mode,
                            "lock_filter_alpha": round(
                                float(out["lock_filter_alpha"]), 4),
                            "prediction_mode": out["prediction_mode"],
                            "prediction_reason": out["prediction_reason"],
                            "prediction_horizon_ms": round(
                                float(out["prediction_horizon_ms"]), 3),
                            "lead": [round(float(v), 3) for v in out["lead"]],
                            "command": [round(float(out["dx"]), 3),
                                        round(float(out["dy"]), 3)],
                            "can_send": bool(out["can_send"]),
                            "in_deadzone": bool(out["in_deadzone"]),
                        }, ensure_ascii=False, separators=(",", ":"))
                    )
                except Exception:
                    pass
            perf_timings["find_target"].append((t_ft_end - t_ft_start) * 1000)
            target = out["target"]

            # Publish one complete, immutable observation.  The target and
            # crosshair below were both read from this capture, so the output
            # worker never combines a new box with an older reference point.
            _visual_target_valid = bool(
                active and target is not None and
                engine.locked_box[0] is not None and
                not str(engine.diag_reason or "").startswith(
                    ("predict_missing", "ambiguous")))
            _reason = str(engine.diag_reason or "")
            if _visual_target_valid:
                if (_current_target_id[0] is None or
                        _reason.startswith("switched_")):
                    _target_id_next[0] += 1
                    _current_target_id[0] = _target_id_next[0]
                _last_real_observation_s[0] = _observation_t
                _obs_state = "real"
            elif target is not None and _reason.startswith("ambiguous"):
                _obs_state = "ambiguous"
            elif target is not None and _reason.startswith("predict_missing"):
                _obs_state = "predicted"
            else:
                _obs_state = "invalid"
            _observation_id[0] += 1
            _box = (list(engine.locked_box[0])
                    if engine.locked_box[0] is not None else [0.0] * 4)
            _box_size = (max(0.0, float(_box[2] - _box[0])),
                         max(0.0, float(_box[3] - _box[1])))
            _raw_error = (
                (float(target[0]) - float(_img_center[0]),
                 float(target[1]) - float(_img_center[1]))
                if target is not None else (0.0, 0.0))
            _observation = ObservationSnapshot(
                session_id=_session_id[0],
                target_id=_current_target_id[0],
                observation_id=_observation_id[0],
                source_frame_id=None,
                image_time_s=float(_observation_t),
                image_time_kind="capture_completed",
                capture_completed_s=float(_observation_t),
                published_s=float(t_ft_end),
                target_point_px=tuple(target) if target is not None else (0.0, 0.0),
                crosshair_px=(float(_img_center[0]), float(_img_center[1])),
                raw_error_px=_raw_error,
                filtered_error_px=tuple(out.get("observed_error", _raw_error)),
                box_size_px=_box_size,
                confidence=float(engine.locked_conf[0] or 0.0),
                state=_obs_state,
                last_real_observation_s=_last_real_observation_s[0],
                motion_ledger=_motion_ledger,
                # Keep the raw/filter/control error and prediction metadata
                # explicit; diagnostic velocity is never silently treated as
                # a control input.
                target_velocity_px_s=tuple(float(v) for v in
                                           out.get("velocity_filtered", (0.0, 0.0))),
                target_velocity_confidence=tuple(
                    1.0 if bool(v) else 0.0
                    for v in getattr(engine, "_unit_velocity_valid_axes", (False, False))),
                control_error_px=(float(out.get("raw_dx", _raw_error[0])),
                                  float(out.get("raw_dy", _raw_error[1]))),
                prediction_mode=str(out.get("prediction_mode", "current")),
                prediction_reason=str(out.get("prediction_reason", "unknown")),
                prediction_horizon_ms=float(out.get("prediction_horizon_ms", 0.0)),
            )
            if not _aim_epoch.is_current(_capture_token):
                _queue_mouse_log(
                    f"[观测丢弃] frame={aim_frame._c} reason=cancelled_before_publish "
                    f"capture_t={_observation_t:.6f} finished_t={t_ft_end:.6f}")
                return
            _aim_send = bool(
                out["can_send"] and not out["in_deadzone"]
                and (abs(out["dx"]) > 0.01 or abs(out["dy"]) > 0.01)
            )
            if aim_mode == "unit":
                _aim_dx = out["dx"] if _aim_send else 0.0
                _aim_dy = out["dy"] if _aim_send else 0.0
                _um = math.hypot(_aim_dx, _aim_dy)
                if _um > unit_max_counts:
                    _scale = unit_max_counts / _um
                    _aim_dx *= _scale
                    _aim_dy *= _scale
                with _control_lock:
                    # Publish the mailbox and its control metadata under the
                    # same state lock.  The output worker takes this lock
                    # before reading either, so it cannot pair two epochs.
                    _latest_observation.publish(_observation)
                    _target_valid = bool(
                        active and target is not None and engine.locked_box[0] is not None)
                    if unit_control_strategy == "remaining_error":
                        # The vision side publishes state only.  All actual
                        # aim motion for this strategy is generated by the
                        # independent output worker below.
                        with _motion_transaction_lock:
                            motion_arbiter.clear_aim()
                        _pub_t = time.perf_counter()
                        if _sched_diag["last_publish_t"]:
                            _sched_diag["publish_intervals"].append(
                                (_pub_t - _sched_diag["last_publish_t"]) * 1000.0)
                        _sched_diag["last_publish_t"] = _pub_t
                        _control_state["plan_active"] = _visual_target_valid
                        _control_state["last_publish"] = _pub_t
                        _control_state["last_publish_history"] = _pub_t
                        _control_state["last_clear_reason"] = (
                            "none" if _visual_target_valid else _obs_state)
                        _control_state["aim_token"] = _capture_token
                    elif _aim_send:
                        if asap_mode:
                            # asap: 参考周期下的一帧控制量转换为最新速率。
                            motion_arbiter.publish_aim_rate(
                                _aim_dx / reference_frame_s,
                                _aim_dy / reference_frame_s,
                            )
                        else:
                            motion_arbiter.publish_aim(
                                _aim_dx, _aim_dy, steps=unit_move_steps,
                                profile="ease" if unit_step_profile == "ease" else "linear",
                            )
                        _pub_t = time.perf_counter()
                        if _sched_diag["last_publish_t"]:
                            _sched_diag["publish_intervals"].append(
                                (_pub_t - _sched_diag["last_publish_t"]) * 1000.0)
                        _sched_diag["last_publish_t"] = _pub_t
                        _control_state["plan_active"] = True
                        _control_state["last_publish"] = _pub_t
                        _control_state["last_publish_history"] = _pub_t
                        _control_state["last_clear_reason"] = "none"
                        _control_state["aim_token"] = _capture_token
                    else:
                        with _motion_transaction_lock:
                            motion_arbiter.clear_aim()
                        _control_state["plan_active"] = False
                        _control_state["aim_token"] = None
                        _control_state["last_publish"] = 0.0
                        _control_state["last_clear_reason"] = (
                            "no_target" if not _target_valid else "deadzone")
                        _control_state["last_clear_t"] = time.perf_counter()
                    _control_state["target_valid"] = _target_valid
                    _control_state["aim_active"] = active
                    _control_state["session_id"] = _session_id[0]
                    _control_state["target_id"] = _current_target_id[0]
                    _control_state["observation"] = _observation
                    _control_state["frame"] = aim_frame._c
                    _control_state["diag"] = {
                        "target": target, "center": tuple(_img_center), "out": dict(out),
                        "ema": tuple(engine._target_ema),
                        "lock": list(engine.locked_box[0]) if engine.locked_box[0] else None,
                        "cls": engine.locked_cls[0], "conf": engine.locked_conf[0],
                        "reason": engine.diag_reason, "dot": _dot_status[0],
                        "capture_motion": _capture_motion,
                        "observed_motion": _observed_motion,
                        "manual_feedforward_active": _manual_feedforward_active,
                        "prediction_suppressed": _prediction_suppressed,
                    }
            else:
                # 非unit旧模式保留单次发送；共享定时输出只服务当前实战unit路径。
                with _motion_transaction_lock:
                    motion_arbiter.clear_aim()
                with _control_lock:
                    _control_state["target_valid"] = False
                if _aim_send and not stop_event.is_set():
                    move_result = (aim_move(out["dx"], out["dy"], hu, raw=True)
                                   if aim_mode == "direct" else
                                   aim_move(out["dx"], out["dy"], dict(hu, deadzone=0.0),
                                            cap=out["cap_hint"]))
                    if move_result["sent"]:
                        engine.notify_net(move_result.get("net_dx", 0), move_result.get("net_dy", 0))

            engine.end_frame()

            with diagnostics_lock:
                diagnostics["frames"] += 1
                diagnostics["candidates"] = engine.diag_candidates
                diagnostics["selection_reason"] = engine.diag_reason
                diagnostics["target"] = engine.diag_target
                diagnostics["locked_box"] = engine.locked_box[0]
                diagnostics["locked_cls"] = engine.locked_cls[0]
                diagnostics["locked_conf"] = engine.locked_conf[0]
                diagnostics["target_ema"] = (round(engine._target_ema[0], 1), round(engine._target_ema[1], 1))
                diagnostics["aim_center"] = (round(_img_center[0], 1), round(_img_center[1], 1))

            perf_timings["mouse_move"].append(0.0)
            _e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
            perf_timings["end_to_end"].append(_e2e_ms)
            _fire_pressed = recoil_controller.pressed or manual_recoil_controller.pressed
            (_yolo_fire_e2e if _fire_pressed else _yolo_idle_e2e).append(_e2e_ms)
            if aim_frame._c % 30 == 0:
                _recent_e2e = perf_timings["end_to_end"][-60:]
                _recent_cap = perf_timings["capture"][-60:]
                _recent_inf = perf_timings["inference"][-60:]
                _recent_civl = list(_sched_diag["capture_intervals"])[-60:]
                _recent_odt = list(_sched_diag["observation_dts"])[-60:]
                _recent_pivl = list(_sched_diag["publish_intervals"])[-60:]
                _recent_wait = list(_sched_diag["artificial_waits"])[-60:]
                _rate_now = motion_arbiter.aim_rate() if aim_mode == "unit" else (0.0, 0.0)
                _emit(
                    f"[YOLO性能] schedule={frame_schedule} "
                    f"reference_frame_ms={reference_frame_s * 1000.0:.1f} "
                    f"截图P50/P95={_percentile(_recent_cap,.5):.1f}/"
                    f"{_percentile(_recent_cap,.95):.1f}ms 推理P50/P95="
                    f"{_percentile(_recent_inf,.5):.1f}/{_percentile(_recent_inf,.95):.1f}ms "
                    f"端到端P50/P95={_percentile(_recent_e2e,.5):.1f}/"
                    f"{_percentile(_recent_e2e,.95):.1f}ms "
                    f"识别FPS={fps_counter.get('fps', 0.0):.1f} "
                    f"截图间隔P50/P95={_percentile(_recent_civl,.5):.1f}/"
                    f"{_percentile(_recent_civl,.95):.1f}ms "
                    f"观测dt P50/P95={_percentile(_recent_odt,.5):.1f}/"
                    f"{_percentile(_recent_odt,.95):.1f}ms "
                    f"发布间隔P50/P95={_percentile(_recent_pivl,.5):.1f}/"
                    f"{_percentile(_recent_pivl,.95):.1f}ms "
                    f"aim_rate=({_rate_now[0]:+.1f},{_rate_now[1]:+.1f})counts/s "
                    f"人工等待P50={_percentile(_recent_wait,.5):.1f}ms "
                    f"≤20ms={sum(v <= 20.0 for v in _recent_e2e) / max(1,len(_recent_e2e)):.0%}"
                )
            fps_counter["count"] += 1
            _now = time.perf_counter()
            if _now - fps_counter["last_time"] >= 1.0:
                fps_counter["fps"] = fps_counter["count"] / max(1e-3, _now - fps_counter["last_time"])
                fps_counter["count"] = 0
                fps_counter["last_time"] = _now
        except Exception as _e:
            with diagnostics_lock:
                diagnostics["aim_exceptions"] = diagnostics.get("aim_exceptions", 0) + 1
            _emit(f"[瞄准异常] {_e}")

    _output_thread = [None]

    def _queue_mouse_log(message, event_type=None, fields=None, send_id=None):
        """Submit detailed output diagnostics to the run-scoped sink."""
        _emit(message, event_type=event_type, fields=fields, send_id=send_id)

    def _send_motion_transaction(send_token, desired_x, desired_y,
                                 learned_x, learned_y, send_epoch):
        """Quantize, authorize, submit and commit one physical packet.

        The transaction lock is shared with target/session resets.  Thus a
        reset cannot clear the arbiter's pending component step after the OS
        accepted it but before the successful ledger event is recorded.
        """
        result = execute_motion_transaction(
            send_token=send_token, desired_x=desired_x, desired_y=desired_y,
            learned_x=learned_x, learned_y=learned_y,
            send_epoch=send_epoch, motion_arbiter=motion_arbiter,
            send_fn=_send_relative,
            controller=(remaining_controller
                        if unit_control_strategy == "remaining_error"
                        else None),
            diagnostics=diagnostics, diagnostics_lock=diagnostics_lock,
            emit=_queue_mouse_log,
            transaction_lock=_motion_transaction_lock,
        )
        if result[0] and (result[4] or result[5]):
            with diagnostics_lock:
                diagnostics["move_calls"] += 1
                diagnostics["last_delta"] = (result[4], result[5])
        return result

    def _output_worker_loop():
        """唯一的unit SendInput写入者；YOLO只发布计划，本线程按配置周期执行。

        fixed: 每 tick 消费一个瞄准 chunk(缓出计划)。
        asap:  每 tick 用"最新控制速率"(counts/s) × 本次真实 tick dt 积分出
               瞄准位移 —— 新观测整体替换速率, 绝不补发旧计划尾段; 控制状态
               超过 TTL 未更新时自动归零并计入 stale_rate_clears。
        """
        # The sole write path is `_send_motion_transaction`, whose contract is
        # quantize_components -> execute_motion_transaction -> ledger commit.
        nonlocal manual_recoil_available, selected_manual_profile
        # 5ms is the lower bound for the experimental high-cadence output
        # path.  Motion is still integrated from the measured dt, so lowering
        # the tick period does not multiply the configured counts/second.
        period_s = max(0.005, float(_recoil_cfg.get("output_period_ms", 30.0)) / 1000.0)
        default_tick_strength = _default_recoil_tick_strength(
            default_recoil_strength, period_s,
            reference_frame_s if asap_mode else unit_frame_s)
        next_t = time.perf_counter()
        consecutive_send_failures = 0
        last_failure_log_t = 0.0
        _last_tick_t = next_t
        # 首个周期日志从线程启动 1 秒后开始, 避免第一行只有亚毫秒 tick 样本。
        _sched_diag["last_tick_log_t"] = next_t
        while not stop_event.is_set():
            now = time.perf_counter()
            # 本次真实 tick dt: 输出线程按真实经过时间积分, 而非假设固定周期。
            _actual_tick_dt = max(0.0, now - _last_tick_t)
            _tick_dt, _discarded_dt, _tick_stalled = bounded_rate_integration_dt(
                _actual_tick_dt, period_s)
            _last_tick_t = now
            _sched_diag["tick_dts"].append(_actual_tick_dt * 1000.0)
            _sched_diag["max_actual_tick_dt_ms"] = max(
                _sched_diag["max_actual_tick_dt_ms"], _actual_tick_dt * 1000.0)
            if _tick_stalled:
                _sched_diag["tick_stall_count"] += 1
                _sched_diag["discarded_dt_ms"] += _discarded_dt * 1000.0
            if _RECOIL_RESET_EVENT.is_set():
                if _ensure_raw_capture():
                    recoil_controller.reset_learning()
                    manual_recoil_controller.reset_learning()
                    selected_manual_profile = ""
                    _emit("[手动轨迹校准] 已切换默认配置，进度=0/5")
                    _publish_recoil_state(
                        manual_recoil_controller, "reset", "",
                        "已进入0/5重新校准", raw_available=True)
                else:
                    _emit(
                        "[手动轨迹校准] 重置失败，已保留当前曲线："
                        f"{raw_mouse.last_error}")
                    _publish_recoil_state(
                        manual_recoil_controller, "reset_failed",
                        selected_manual_profile, raw_mouse.last_error,
                        raw_available=False)
                _RECOIL_RESET_EVENT.clear()
            with _control_lock:
                state = dict(_control_state)
                _snapshot = _latest_observation.read()
            # Aim commands carry the token captured when their snapshot was
            # published.  Never authorize an old snapshot with a newly read
            # epoch token.  Recoil-only packets retain the current chord token.
            _send_token = state.get("aim_token")
            _send_epoch = _aim_epoch
            if _send_token is None:
                # Aim cancellation is intentionally not a recoil cancellation;
                # an active left-button chord may continue on its own channel.
                _send_token = (state.get("recoil_token")
                               if recoil_active else _recoil_epoch.token())
                _send_epoch = _recoil_epoch if recoil_active else _aim_epoch
            _control_age_s = (
                now - state["last_publish"]
                if state.get("plan_active") and state["last_publish"] else None)
            if _control_age_s is not None:
                _sched_diag["control_ages"].append(_control_age_s * 1000.0)
            if unit_control_strategy == "remaining_error":
                _obs_age = (
                    now - _snapshot.last_real_observation_s
                    if _snapshot is not None and
                    _snapshot.last_real_observation_s is not None else None)
                valid = bool(
                    aim_mode == "unit" and state["aim_active"] and
                    state["target_valid"] and state.get("plan_active") and
                    _snapshot is not None and _snapshot.state == "real" and
                    _snapshot.session_id == state.get("session_id") and
                    _snapshot.target_id == state.get("target_id") and
                    _obs_age is not None and
                    _obs_age <= min(_aim_fresh_ttl_s[0],
                                    remaining_controller.max_observation_age_s))
                if valid:
                    remaining_controller.observe(_snapshot, now)
                    _remaining_stale_latched[0] = False
                else:
                    if (_control_age_s is not None and
                            _control_age_s > _aim_fresh_ttl_s[0] and
                            not _remaining_stale_latched[0]):
                        _sched_diag["stale_control_clears"] += 1
                        _remaining_stale_latched[0] = True
                    remaining_controller.invalidate("inactive_or_stale")
            else:
                valid = bool(
                    aim_mode == "unit" and state["aim_active"] and
                    state["target_valid"] and state.get("plan_active") and
                    _control_age_s is not None and
                    _control_age_s <= _aim_fresh_ttl_s[0]
                )
            # Raw Input records only the physical device, so YOLO aim output can
            # keep serving the user during all five calibration bursts.
            if not valid:
                # TTL 超时(仍在瞄准且目标曾有效)而非松键/目标无效时, 计入 stale 清除。
                _stale = bool(
                    state["aim_active"] and state["target_valid"] and
                    state.get("plan_active") and
                    _control_age_s is not None and
                    _control_age_s > _aim_fresh_ttl_s[0])
                if _stale and motion_arbiter.has_pending_aim():
                    _sched_diag["stale_rate_clears"] += 1
                if motion_arbiter.has_pending_aim():
                    _clear_current_aim("stale_clear" if _stale else "inactive")
            if unit_control_strategy == "remaining_error":
                aim_dx, aim_dy = remaining_controller.next_motion(
                    _tick_dt, now, valid=valid)
            elif asap_mode:
                aim_dx, aim_dy = motion_arbiter.next_aim_rate_motion(_tick_dt)
            else:
                aim_dx, aim_dy = motion_arbiter.next_aim_chunk()

            legacy_enabled = bool(
                anti_recoil and recoil_controller.pressed
                and (valid or recoil_controller.can_control_without_target))
            recoil_out = recoil_controller.step(now, enabled=legacy_enabled)
            manual_enabled_now = bool(
                manual_recoil_enabled and manual_recoil_available
                and manual_recoil_controller.pressed
                and manual_recoil_controller.ready)
            manual_out = manual_recoil_controller.step(
                time.perf_counter_ns(), enabled=manual_enabled_now)
            manual_raw_x, manual_raw_y = apply_playback_blend(
                manual_out, state["manual_recoil_blend_percent"])
            _, default_vertical_y = (_sample_default_vertical_recoil(
                default_tick_strength) if (
                    default_recoil_enabled and recoil_active) else (0.0, 0.0))
            diag = state.get("diag") or {}
            target_box = diag.get("lock")
            recoil_gate_open = _manual_recoil_gate_open(
                valid, diag.get("center"), target_box)
            # The manual profile clock advances regardless of its in-box gate;
            # the independent default controller remains vertical-only.
            recoil_x, recoil_y, manual_x, manual_y = _compose_recoil_components(
                default_vertical_y, (manual_raw_x, manual_raw_y), recoil_gate_open)
            desired_x, desired_y, learned_x, learned_y = mix_motion_components(
                aim_dx, aim_dy,
                recoil_x, recoil_y,
                unit_max_counts,
            )
            # Aim and recoil keep independent fractional ledgers.  The helper
            # makes quantization, SendInput and ledger commit one transaction.
            (sent_ok, _send_started_s, _send_ended_s, _send_reason,
             send_x, send_y, aim_step, recoil_step, _commit_t) = (
                _send_motion_transaction(_send_token, desired_x, desired_y,
                                         learned_x, learned_y, _send_epoch))
            if not sent_ok:
                consecutive_send_failures += 1
                if (_send_reason == "send_failed" and
                        (consecutive_send_failures == 1 or
                         now - last_failure_log_t >= 1.0)):
                    try:
                        win_error = int(ctypes.windll.kernel32.GetLastError())
                    except Exception:
                        win_error = -1
                    _emit(
                        "[鼠标输出器] SendInput失败 "
                        f"连续={consecutive_send_failures} error={win_error} "
                        f"delta=({send_x:+d},{send_y:+d})",
                        event_type="error",
                        fields={"stage": "input_submission",
                                "consecutive": consecutive_send_failures,
                                "win_error": win_error,
                                "net": [send_x, send_y]},
                    )
                    last_failure_log_t = now
            else:
                consecutive_send_failures = 0
            if anti_recoil and recoil_controller.pressed:
                recoil_controller.record_output(
                    learned_x, learned_y, now,
                    target_valid=valid, sent_ok=sent_ok,
                )
            if mouse_log_enabled and sent_ok and (send_x or send_y):
                diag = state.get("diag")
                if valid and diag and diag.get("target") is not None:
                    try:
                        target = diag["target"]
                        center = diag["center"]
                        out = diag["out"]
                        lock_box = diag["lock"] if diag["lock"] is not None else "无"
                        _queue_mouse_log(
                            f"[鼠标] 帧={state['frame']} 目标=({target[0]:.1f},{target[1]:.1f}) "
                            f"EMA=({diag['ema'][0]:.1f},{diag['ema'][1]:.1f}) "
                            f"准星=({center[0]:.1f},{center[1]:.1f}) "
                            f"误差=({out['raw_dx']:+.1f},{out['raw_dy']:+.1f}) "
                            f"前导=({out['lead'][0]:+.1f},{out['lead'][1]:+.1f}) "
                            f"速度原始=({out['velocity_raw'][0]:+.1f},{out['velocity_raw'][1]:+.1f}) "
                            f"速度滤波=({out['velocity_filtered'][0]:+.1f},{out['velocity_filtered'][1]:+.1f}) "
                            f"预测时域={out['prediction_horizon_ms']:.1f}ms "
                            f"预测模式={out['prediction_mode']} 原因={out['prediction_reason']} "
                            f"缺失时长={out.get('missing_duration_ms', 0.0):.1f}ms "
                            f"截图快照=({diag['capture_motion'].total[0]:+d},"
                            f"{diag['capture_motion'].total[1]:+d}) "
                            f"观测瞄准=({diag['observed_motion'].aim[0]:+d},"
                            f"{diag['observed_motion'].aim[1]:+d}) "
                            f"观测压枪=({diag['observed_motion'].recoil[0]:+d},"
                            f"{diag['observed_motion'].recoil[1]:+d}) "
                            f"前馈={int(diag['manual_feedforward_active'])} "
                            f"前导抑制={int(diag['prediction_suppressed'])} "
                            f"瞄准分量=({aim_dx:+.1f},{aim_dy:+.1f}) "
                            f"后坐力=({recoil_x:+.1f},{recoil_y:+.1f}) "
                            f"学习分量=({learned_x:+.1f},{learned_y:+.1f}) "
                            f"瞄准整数=({aim_step[0]:+d},{aim_step[1]:+d}) "
                            f"压枪整数=({recoil_step[0]:+d},{recoil_step[1]:+d}) "
                            f"手动曲线=({manual_x:+.1f},{manual_y:+.1f}) "
                            f"原始曲线=({manual_raw_x:+.1f},{manual_raw_y:+.1f}) "
                            f"入框门控={'放行' if recoil_gate_open else '等待准星入框'} "
                            f"闭环=({recoil_out.recovery_x:+.1f},{recoil_out.recovery_y:+.1f}) "
                            f"阶段={manual_out.phase} 时间格={manual_out.curve_bin} "
                            f"输出t={now:.6f} "
                            f"校准={manual_out.calibration_runs}/5 "
                            f"回放={state['manual_recoil_blend_percent']:.0f}% "
                            f"锁定={lock_box} cls={diag['cls']} conf={diag['conf']:.2f} "
                            f"净移=({send_x:+d},{send_y:+d}) 选择={diag['reason']} 红点={diag['dot']}"
                        )
                    except Exception:
                        pass
                elif manual_recoil_controller.ready:
                    _queue_mouse_log(
                        f"[鼠标] 无目标框 手动曲线=({manual_x:+.1f},"
                        f"{manual_y:+.1f}) 原始曲线=({manual_raw_x:+.1f},"
                        f"{manual_raw_y:+.1f}) 阶段={manual_out.phase} "
                        f"瞄准整数=({aim_step[0]:+d},{aim_step[1]:+d}) "
                        f"压枪整数=({recoil_step[0]:+d},{recoil_step[1]:+d}) "
                        f"时间格={manual_out.curve_bin} 校准=5/5 "
                        f"输出t={now:.6f} "
                        f"入框门控={'放行' if recoil_gate_open else '等待准星入框'} "
                        f"回放={state['manual_recoil_blend_percent']:.0f}% "
                        f"净移=({send_x:+d},{send_y:+d})"
                    )
            if now - _sched_diag["last_tick_log_t"] >= 1.0:
                _sched_diag["last_tick_log_t"] = now
                _recent_tick = list(_sched_diag["tick_dts"])[-100:]
                _recent_age = list(_sched_diag["control_ages"])[-100:]
                _recent_pivl = list(_sched_diag["publish_intervals"])[-100:]
                _recent_wait = list(_sched_diag["artificial_waits"])[-100:]
                _rate_now = motion_arbiter.aim_rate()
                _remaining_text = ""
                if unit_control_strategy == "remaining_error":
                    _rd = remaining_controller.diagnostics()
                    _remaining_text = (
                        f" residual=({_rd['remaining_counts'][0]:+.2f},"
                        f"{_rd['remaining_counts'][1]:+.2f})counts"
                        f" pending_pre=({_rd['pre_capture_pending_counts'][0]:+.1f},"
                        f"{_rd['pre_capture_pending_counts'][1]:+.1f})")
                _age_text = (
                    f"{_percentile(_recent_age,.5):.1f}ms"
                    if _recent_age else "N/A(inactive)")
                _pivl_text = (
                    f"{_percentile(_recent_pivl,.5):.1f}/"
                    f"{_percentile(_recent_pivl,.95):.1f}ms"
                    if _recent_pivl else "N/A(no-history)")
                _history_age = (
                    now - state["last_publish_history"]
                    if state.get("last_publish_history") else None)
                _history_age_text = (
                    f"{_history_age * 1000.0:.1f}ms"
                    if _history_age is not None else "N/A(no-history)")
                _plan_text = state.get("last_clear_reason", "never_published")
                if state.get("plan_active"):
                    _plan_text = "active"
                _emit(
                    f"[输出器] 周期={period_s * 1000.0:.1f}ms schedule={frame_schedule} "
                    f"tick_dt P50/P95={_percentile(_recent_tick,.5):.2f}/"
                    f"{_percentile(_recent_tick,.95):.2f}ms "
                    f"control_age P50={_age_text} "
                    f"last_publish_history_age={_history_age_text} "
                    f"发布间隔P50/P95={_pivl_text} plan={_plan_text} "
                    f"aim_rate=({_rate_now[0]:+.1f},{_rate_now[1]:+.1f})counts/s "
                    f"{_remaining_text} "
                    f"人工等待P50={_percentile(_recent_wait,.5):.1f}ms "
                    f"stale_clear={_sched_diag['stale_rate_clears']} "
                    f"stale_control={_sched_diag['stale_control_clears']} "
                    f"recent_obs={_sched_diag['recent_observation_dt_ms']:.2f}ms "
                    f"ttl={_aim_fresh_ttl_s[0] * 1000.0:.1f}ms "
                    f"tick_stall={_sched_diag['tick_stall_count']} "
                    f"discarded_dt={_sched_diag['discarded_dt_ms']:.1f}ms "
                    f"max_actual_tick_dt={_sched_diag['max_actual_tick_dt_ms']:.2f}ms "
                    f"idle={diagnostics.get('idle_tick', 0)} "
                    f"fractional_wait={diagnostics.get('fractional_wait', 0)} "
                    f"component_cancel={diagnostics.get('component_cancel', 0)} "
                    f"send_attempt={diagnostics.get('send_attempt', 0)} "
                    f"send_success={diagnostics.get('send_success', 0)} "
                    f"send_failed={diagnostics.get('send_failed', 0)}",
                    event_type="summary",
                )
            next_t += period_s
            delay = next_t - time.perf_counter()
            if delay <= 0.0:
                next_t = time.perf_counter() + period_s
                delay = period_s
            stop_event.wait(delay)

    def _output_worker():
        temporary_retries = 0
        while not stop_event.is_set():
            try:
                _output_worker_loop()
                return
            except (OSError, TimeoutError) as exc:
                temporary_retries += 1
                if temporary_retries <= 3:
                    _emit(
                        f"[鼠标输出器] 临时故障，将重试 {temporary_retries}/3: "
                        f"{type(exc).__name__}: {exc}",
                        event_type="warning",
                        fields={"exception_type": type(exc).__name__,
                                "retry": temporary_retries},
                    )
                    if stop_event.wait(0.10):
                        return
                    continue
                # A bounded retry budget is itself part of the output
                # contract; once exhausted, fail closed instead of spinning.
                with diagnostics_lock:
                    diagnostics["output_fault"] = f"{type(exc).__name__}: {exc}"
                _clear_current_aim("output_fault")
                _emit(
                    f"[鼠标输出器] 故障已停止（临时故障重试耗尽）: "
                    f"{type(exc).__name__}: {exc}",
                    event_type="error",
                    fields={"exception_type": type(exc).__name__,
                            "retry_exhausted": True,
                            "traceback": traceback.format_exc()},
                )
                stop_event.set()
                return
            except Exception as exc:
                # Deterministic program errors (including NameError and
                # UnboundLocalError) must not be retried every 100ms.
                with diagnostics_lock:
                    diagnostics["output_fault"] = f"{type(exc).__name__}: {exc}"
                try:
                    _clear_current_aim("output_fault")
                except Exception:
                    pass
                _emit(
                    f"[鼠标输出器] 故障已停止: {type(exc).__name__}: {exc}",
                    event_type="error",
                    fields={"exception_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                            "deterministic": True},
                )
                stop_event.set()
                return

    def _percentile(values, fraction):
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * fraction)]

    def do_screenshot():
        nonlocal output_dir, _CLASS_NAMES
        try:
            with lock:
                dets, img = capture()
                with diagnostics_lock:
                    diagnostics["screenshot_frames"] += 1
                    diagnostics["screenshot_detections"] = len(dets)
                out = draw_boxes(img, dets, _CLASS_NAMES,
                                 aim_center=tuple(_img_center),
                                 dot_center=(tuple(_capture_aim_center)
                                             if crosshair_mode != "center" else None))
            idx = counter[0]
            counter[0] += 1
            # 用绝对路径写入
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"output{idx}.jpg")
            ok = cv2.imwrite(path, out)
            t = detector._timing
            if ok:
                _emit(f"截图 #{idx} 已保存 → {path}  推理{t['inference']:.1f}ms  {len(dets)}目标")
            else:
                _emit(f"截图 #{idx} 写入失败 → {path} (权限？)")
        except Exception as exc:
            _diagnostic_error("screenshot", exc)

    # ── Physical mouse trajectory capture ──
    _raw_button_handler = [None]

    def _record_physical_mouse(dx, dy, event_ns, device):
        manual_recoil_controller.record_raw_delta(
            dx, dy, event_ns, device=device)

    def _record_physical_button(pressed, event_ns, device):
        handler = _raw_button_handler[0]
        if handler is not None:
            handler(bool(pressed), int(event_ns), int(device))

    raw_mouse = RawMouseInput(
        _record_physical_mouse, button_callback=_record_physical_button)

    def _ensure_raw_capture():
        nonlocal manual_recoil_available, raw_capture_ready
        if not manual_recoil_enabled or aim_mode != "unit":
            return False
        if raw_capture_ready and raw_mouse.healthy:
            manual_recoil_available = True
            return True
        if not raw_mouse.start():
            manual_recoil_available = False
            return False
        if bool(_manual_recoil_cfg.get("raw_self_test", True)) \
                and not raw_mouse.verify_injection_isolated(_send_relative):
            raw_mouse.stop()
            manual_recoil_available = False
            raw_capture_ready = False
            return False
        raw_capture_ready = True
        manual_recoil_available = True
        return True

    if manual_recoil_enabled and aim_mode == "unit":
        if manual_profile_loaded:
            manual_recoil_available = True
            _emit(
                f"[手动轨迹] 已加载曲线“{selected_manual_profile}”，"
                "状态=READY；已跳过 Raw Input 启动与自检")
        elif not _ensure_raw_capture():
            _emit(
                f"[手动轨迹] Raw Input 启动/隔离自检失败，功能已关闭: "
                f"{raw_mouse.last_error}")
        else:
            _emit(
                "[手动轨迹] Raw Input 已就绪；前5次≥"
                f"{float(_manual_recoil_cfg.get('min_burst_ms',300.0)):.0f}ms "
                "物理LMB按下即采集；YOLO识别与瞄准移动继续运行；"
                "回放仍使用RMB→LMB")
    elif manual_recoil_enabled:
        _emit("[手动轨迹] 仅 unit 瞄准模式支持，当前功能已关闭")
    if manual_profile_warning:
        _emit(
            f"[手动轨迹] 已保存曲线加载失败，已回退默认0/5："
            f"{manual_profile_warning}")
    _publish_recoil_state(
        manual_recoil_controller,
        "loaded" if manual_profile_loaded else "startup",
        selected_manual_profile,
        manual_profile_warning,
        raw_available=raw_capture_ready)
    _emit(
        f"[默认垂直压枪] {'开启' if default_recoil_enabled else '关闭'} "
        f"强度={default_recoil_strength:.1f}/识别周期")
    _emit(f"[旧视觉后坐力] {'开启' if anti_recoil else '已可靠关闭'}")

    # ── 热键 ──
    # gui.py 可通过 engine._EXTERNAL_HK 注入外部热键桥
    global _EXTERNAL_HK
    def _fast_aim_state(pressed):
        """Hotkey producer -> output mailbox, independent of inference."""
        if not pressed:
            _cancel_aim("aim_released")
            return
        with _control_lock:
            # Press is only a state hint here; session creation remains owned
            # by the consumer so a duplicate callback cannot create a session.
            _control_state["aim_active"] = True
    # CLI 模式下的 vk 占位（GUI 模式走 action 分支，不会用到这些）
    aim_vk = ss_vk = quit_vk = recoil_vk = trigger_vk = -1
    if _EXTERNAL_HK is not None:
        _emit("热键: GUI Poller (主线程 GetAsyncKeyState)")
        hk = _EXTERNAL_HK
    else:
        _emit("热键: GetAsyncKeyState 轮询 (无 GUI)")
        hk = Hotkeys()
        aim_key = hotkeys_cfg["aim"]
        ss_key = hotkeys_cfg["screenshot"]
        quit_key = hotkeys_cfg["quit"]
        recoil_key_local = recoil_key.lower()
        trigger_key = hotkeys_cfg["trigger"]
        aim_vk = hk.register(aim_key)
        ss_vk = hk.register(ss_key)
        quit_vk = hk.register(quit_key)
        recoil_vk = hk.register(recoil_key_local)
        trigger_vk = hk.register(trigger_key)
        hk.start()
        _emit(f"  瞄准={aim_key}(0x{aim_vk:02X})  截图={ss_key}(0x{ss_vk:02X})  退出={quit_key}(0x{quit_vk:02X})  压枪={recoil_key_local}(0x{recoil_vk:02X})  自动触发={trigger_key}(0x{trigger_vk:02X})")

    if hasattr(hk, "set_aim_state_callback"):
        hk.set_aim_state_callback(_fast_aim_state)
    if hasattr(hk, "set_aim_vk"):
        hk.set_aim_vk(aim_vk)

    _emit("  (图例: down=按键按下, up=按键松开)")

    aim_count = 0

    def _apply_recoil_state(pressed, now, source):
        nonlocal recoil_active
        pressed = bool(pressed)
        if pressed == recoil_active:
            return
        recoil_active = pressed
        _recoil_token = _recoil_epoch.set_pressed(pressed)
        with _control_lock:
            _control_state["recoil_token"] = (
                _recoil_token if pressed else None)
        accepted = False
        if anti_recoil:
            with _control_lock:
                state = dict(_control_state)
            calibration_eligible = bool(
                aim_mode == "unit" and state["aim_active"] and state["target_valid"]
                and state.get("plan_active") and state["last_publish"]
                and now - state["last_publish"] <= _aim_fresh_ttl_s[0])
            recoil_controller.on_fire(
                pressed, now, calibration_eligible=calibration_eligible)
        if manual_recoil_enabled and manual_recoil_available:
            accepted = manual_recoil_controller.on_fire(
                pressed, int(now * 1_000_000_000),
                capture_healthy=raw_mouse.healthy)
            if pressed and manual_recoil_controller.calibrating:
                # Drop queued aim chunks and fractional output debt before the
                # first raw sample so calibration contains physical input only.
                with _motion_transaction_lock:
                    motion_arbiter.reset()
        fire_wake.set()
        mode = (
            "手动采集" if (pressed and manual_recoil_enabled
                            and manual_recoil_controller.calibrating)
            else "轨迹回放" if (pressed and manual_recoil_enabled
                                 and manual_recoil_controller.ready)
            else "默认垂直" if pressed and default_recoil_enabled
            else "旧视觉闭环" if pressed and anti_recoil
            else "结束" if not pressed else "已关闭")
        _emit(
            f"[开火] {source} t={now:.6f} "
            f"阶段={manual_recoil_controller.phase} 模式={mode} "
            f"校准={manual_recoil_controller.calibration_count}/5 "
            f"曲线={manual_recoil_controller.profile_duration_ms:.0f}ms "
            f"结果={manual_recoil_controller.last_calibration_result}"
        )
        if not pressed and accepted:
            def _format_bins(points):
                return ";".join(
                    f"{index}:({x:+.1f},{y:+.1f})"
                    for index, (x, y) in enumerate(points))

            _emit(
                f"[手动轨迹样本] 次数={manual_recoil_controller.calibration_count}/5 "
                f"格宽={manual_recoil_controller.bin_ns / 1_000_000:.0f}ms "
                f"数据={_format_bins(manual_recoil_controller.last_accepted_run)}")
            _emit(
                f"[手动轨迹校准] 已接受 {manual_recoil_controller.calibration_count}/5 "
                f"共同曲线={manual_recoil_controller.profile_duration_ms:.0f}ms "
                f"状态={'已冻结；下次开火自动回放' if manual_recoil_controller.ready else '继续校准'}"
            )
            if manual_recoil_controller.ready:
                _emit(
                    f"[手动轨迹中位曲线] 格宽="
                    f"{manual_recoil_controller.bin_ns / 1_000_000:.0f}ms "
                    f"格数={manual_recoil_controller.curve_bins} "
                    f"数据={_format_bins(manual_recoil_controller.curve_points)}")
            _publish_recoil_state(
                manual_recoil_controller, "accepted", "",
                manual_recoil_controller.last_calibration_result,
                raw_available=raw_capture_ready)
        elif (not pressed and manual_recoil_enabled and manual_recoil_available
              and not manual_recoil_controller.ready
              and manual_recoil_controller.last_calibration_result.startswith("discarded_")):
            _emit(
                f"[手动轨迹校准] 未计入：{manual_recoil_controller.last_calibration_result} "
                f"进度={manual_recoil_controller.calibration_count}/5")
            _publish_recoil_state(
                manual_recoil_controller, "discarded", "",
                manual_recoil_controller.last_calibration_result,
                raw_available=raw_capture_ready)

    def _handle_recoil_edge(pressed, edge_time=None):
        now = time.perf_counter() if edge_time is None else float(edge_time)
        transition = recoil_gate.on_left(pressed)
        if transition.start:
            _apply_recoil_state(True, now, "RMB→LMB down")
        elif transition.stop:
            _apply_recoil_state(False, now, "LMB up")
        elif pressed and not recoil_gate.right_held:
            _emit(f"[开火] LMB down t={now:.6f} 已忽略：必须先按住右键")

    def _handle_right_edge(pressed, edge_time=None):
        now = time.perf_counter() if edge_time is None else float(edge_time)
        transition = recoil_gate.on_right(pressed)
        if transition.stop:
            _apply_recoil_state(False, now, "RMB up")

    def _handle_raw_left_edge(pressed, event_ns, _device):
        """Use physical LMB timing for calibration; replay keeps the chord."""
        if manual_recoil_controller.ready:
            return
        now = int(event_ns) / 1_000_000_000.0
        if pressed:
            _apply_recoil_state(True, now, "Raw LMB down")
        elif manual_recoil_controller.pressed:
            _apply_recoil_state(False, now, "Raw LMB up")

    _raw_button_handler[0] = _handle_raw_left_edge

    if hasattr(hk, "set_recoil_callback"):
        hk.set_recoil_callback(_handle_recoil_edge)

    if aim_mode == "unit":
        _output_thread[0] = threading.Thread(target=_output_worker, daemon=True)
        _output_thread[0].start()
        _output_period_ms = max(
            5.0, float(_recoil_cfg.get("output_period_ms", 30.0)))
        if asap_mode:
            _emit(
                f"鼠标输出器: 单写入, 共享周期={_output_period_ms:.0f}ms; "
                f"识别调度=asap(跟随推理速度, frame_ms 不等待); "
                f"输出语义={'剩余误差×真实tick dt' if unit_control_strategy == 'remaining_error' else '最新控制速率×真实tick dt积分'} "
                f"(move_steps={unit_move_steps} 仅固定周期模式生效)"
            )
        else:
            _emit(
                f"鼠标输出器: 单写入, 共享周期={_output_period_ms:.0f}ms; "
                f"YOLO识别节拍={unit_frame_s * 1000.0:.0f}ms; "
                f"缓出={unit_move_steps}周期(约{unit_move_steps * _output_period_ms:.0f}ms)"
            )
        _emit(
            "压枪-YOLO协调: 截图时点净位移归属=开; "
            f"回放横向受限前导={'开' if _control_state['manual_recoil_suppress_prediction'] else '关'}; "
            f"移动目标策略={engine.prediction_mode}"
        )
        _emit(
            "目标框滤波: "
            + ("自适应 alpha=0.35~0.85, speed=80~600px/s"
               if engine.lock_box_filter_mode == "adaptive"
               else f"固定 alpha={engine.lock_box_smoothing_alpha:.2f}")
        )
        if ready_callback is not None:
            try:
                ready_callback()
            except Exception as exc:
                _emit(f"[启动提示音] 回调失败: {type(exc).__name__}: {exc}",
                      event_type="warning")
        if asap_mode:
            _hold_ms = crosshair_tracker.hold_frames * crosshair_tracker.reference_dt * 1000.0
            _return_ms = crosshair_tracker.return_frames * crosshair_tracker.reference_dt * 1000.0
            _confirm_ms = (crosshair_tracker.confirm_frames
                           * crosshair_tracker.reference_dt * 1000.0)
            _emit(
                "红点闭环: 混合中心; "
                f"异常保持≈{_hold_ms:.0f}ms后在≈{_return_ms:.0f}ms内受限回归中心"
                f"(时间制, 候选确认≈{_confirm_ms:.0f}ms; 不随实际YOLO FPS变化)")
        else:
            _emit(
                "红点闭环: 混合中心; "
                f"异常保持{crosshair_tracker.hold_frames}帧后在"
                f"{crosshair_tracker.return_frames}帧内受限回归中心")

    def _reset_fps_meter():
        """瞄准开始时重置 FPS 计时/帧号,并重置引擎状态(EMA/锁定/速度等),
        避免新会话首帧残留上轮的旧 EMA/锁定(实战日志:帧1 目标已变,EMA 还是旧位置)。"""
        _session_id[0] += 1
        _session_token = _aim_epoch.start_session(_session_id[0])
        _current_target_id[0] = None
        _last_real_observation_s[0] = None
        _latest_observation.clear()
        engine.reset()
        crosshair_tracker.reset()
        # New session resets target-control state but retains committed input
        # history so the first new image can still account for in-flight view
        # motion from the previous session.
        with _motion_transaction_lock:
            motion_arbiter.reset_control_scope()
        _aim_freshness.reset()
        _aim_fresh_ttl_s[0] = (
            _aim_freshness.ttl_s if asap_mode else unit_frame_s * 2.5)
        with _control_lock:
            _control_state["session_id"] = _session_id[0]
            _control_state["aim_token"] = _session_token
            _control_state["target_id"] = None
            _control_state["observation"] = None
            _control_state["target_valid"] = False
            _control_state["aim_active"] = True
            _control_state["diag"] = None
            _control_state["plan_active"] = False
            _control_state["last_publish"] = 0.0
            _control_state["last_clear_reason"] = "reset"
            _control_state["last_clear_t"] = time.perf_counter()
        # A new hold-to-aim session has a new time origin.  Idle time must not
        # become the first filter/velocity/TTL dt or contaminate cadence stats.
        _last_aim_t[0] = 0.0
        _sched_diag["last_obs_t"] = 0.0
        _sched_diag["last_publish_t"] = 0.0
        _sched_diag["recent_observation_dt_ms"] = reference_frame_s * 1000.0
        _sched_diag["observation_dt_ewma_ms"] = reference_frame_s * 1000.0
        _sched_diag["observation_dts"].clear()
        _sched_diag["publish_intervals"].clear()
        fps_counter["count"] = 0
        fps_counter["last_time"] = time.perf_counter()
        fps_counter["active"] = True
        aim_frame._c = 0  # 无条件初始化，保证首帧日志显示帧号而非 '?'

    def consumer():
        nonlocal active, recoil_active, aim_count
        try:
            # 空闲时 poll 可以稍长一点省 CPU；瞄准时用短超时
            idle_timeout = max(0.01, aim_interval_sec)
            # 瞄准时：timeout=0 纯非阻塞，由 e2e 时间自然决定帧率
            # 若 e2e 很快，再 sleep 到 aim_interval，避免空转占满 CPU
            # aim_up/stop_event 时不再进入下一帧，避免停止后仍继续发送相对移动。
            while not stop_event.is_set():
                if active:
                    # 先非阻塞取热键事件，再立刻跑一帧
                    ev = hk.poll(timeout=0)
                    if ev is None:
                        t0 = time.perf_counter()
                        with lock:
                            aim_count += 1
                            aim_frame()
                        # 可选最小间隔：仅当 e2e 比设定间隔还快时才补 sleep
                        spent = time.perf_counter() - t0
                        remain = aim_interval_sec - spent
                        if remain > 0.0005:
                            time.sleep(remain)
                        continue
                else:
                    ev = hk.poll(timeout=idle_timeout)
                    if ev is None:
                        continue

                # ── HotkeyPoller (GUI) 事件 ──
                action = getattr(ev, "action", None)
                if action is not None:
                    if action == "aim_down":
                        _handle_right_edge(True)
                        if not active:
                            active = True
                            aim_count = 0
                            _reset_fps_meter()
                            _emit("▶ 瞄准开始")
                    elif action == "aim_up":
                        _handle_right_edge(False)
                        if active:
                            active = False
                            _cancel_aim("stopped")
                            fps_counter["active"] = False
                            _emit("■ 瞄准停止")
                    elif action == "recoil_down":
                        _handle_recoil_edge(True)
                    elif action == "recoil_up":
                        _handle_recoil_edge(False)
                    elif action == "manual_recoil_blend":
                        value = max(0.0, min(100.0, float(
                            getattr(ev, "value", 70.0))))
                        with _control_lock:
                            _control_state["manual_recoil_blend_percent"] = value
                    elif action == "manual_recoil_tail":
                        value = max(0.0, min(100.0, float(
                            getattr(ev, "value", 30.0))))
                        manual_recoil_controller.set_tail_vertical_ratio(
                            value / 100.0)
                    elif action == "manual_recoil_suppress_prediction":
                        with _control_lock:
                            _control_state["manual_recoil_suppress_prediction"] = bool(
                                getattr(ev, "value", True))
                    elif action == "prediction_mode":
                        value = str(getattr(ev, "value", "arrival")).lower()
                        if value in ("current", "arrival"):
                            engine.set_prediction_mode(value)
                            _emit(
                                "[瞄准策略] " +
                                ("到达时刻预测" if value == "arrival" else
                                 "当前跟踪点（无前导）"))
                    elif action == "screenshot":
                        threading.Thread(target=do_screenshot, daemon=True).start()
                    elif action == "trigger":
                        _auto_trigger_toggle(_emit)
                    elif action == "quit":
                        _emit("退出信号")
                        stop_event.set()
                        break
                    continue

                # ── Hotkeys (GAS) 事件 ──
                vk_id = getattr(ev, "vk", None)
                pressed = getattr(ev, "pressed", False)
                _emit(f"[HK] 收到 vk=0x{vk_id:02X}  pressed={pressed}")

                if ev.vk == aim_vk:
                    if ev.pressed and not active:
                        _handle_right_edge(True)
                        active = True
                        aim_count = 0
                        _reset_fps_meter()
                        _emit("▶ 瞄准开始")
                    elif not ev.pressed and active:
                        _handle_right_edge(False)
                        active = False
                        _cancel_aim("stopped")
                        fps_counter["active"] = False
                        _emit("■ 瞄准停止")
                elif ev.vk == recoil_vk:
                    _handle_recoil_edge(ev.pressed)
                elif ev.vk == trigger_vk and ev.pressed:
                    _auto_trigger_toggle(_emit)
                elif ev.vk == ss_vk and ev.pressed:
                    threading.Thread(target=do_screenshot, daemon=True).start()
                elif ev.vk == quit_vk and ev.pressed:
                    _emit("退出信号")
                    stop_event.set()
                    break
        except Exception as exc:
            _diagnostic_error("consumer", exc)
            stop_event.set()

    t_consumer = threading.Thread(target=consumer, daemon=True)
    t_consumer.start()

    try:
        # 等待 consumer 完成当前帧并退出，避免旧线程与下一次启动并发发送输入。
        while not stop_event.is_set():
            stop_event.wait(1.0)
        t_consumer.join(timeout=1.0)
    except KeyboardInterrupt:
        stop_event.set()
        _cancel_aim("keyboard_interrupt")

    _emit("清理中…")
    if _output_thread[0] is not None:
        _output_thread[0].join(timeout=1.0)
    try:
        if hasattr(hk, "set_recoil_callback"):
            hk.set_recoil_callback(None)
        hk.stop()
    except Exception:
        pass
    raw_mouse.stop()
    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass
    if _timer_period_set and winmm is not None:
        try:
            winmm.timeEndPeriod(1)
        except Exception:
            pass
    _emit("已退出")
    _log_stats = _run_log.stats()
    _emit(
        "[日志统计] phase=pre_close "
        f"run_id={_run_log.run_id} 产生={_log_stats.get('produced', 0)} "
        f"写入={_log_stats.get('file_written', 0)} "
        f"丢弃={_log_stats.get('file_dropped', 0)} "
        f"队列高水位={_log_stats.get('file_high_water', 0)} "
        f"文件队列年龄P95={_log_stats.get('file_queue_age_p95_s', 0.0) * 1000.0:.2f}ms "
        f"stdout队列年龄P95={_log_stats.get('stdout_queue_age_p95_s', 0.0) * 1000.0:.2f}ms "
        f"产生速率={_log_stats.get('production_rate_hz', 0.0):.1f}/s "
        f"写入速率={_log_stats.get('file_write_rate_hz', 0.0):.1f}/s",
        event_type="summary", fields=_log_stats)
    _stop_watch_done.set()
    if _stop_watch_thread is not None:
        _stop_watch_thread.join(timeout=0.2)
    _RECOIL_RESET_EVENT.clear()
    _MANUAL_RECOIL_CONTROLLER = None
    _run_log_result = _run_log.close(timeout_s=1.0)
    # The sink is already detached from GUI admission before close.  Keep the
    # final accounting in stdout only; it cannot be appended to a later run.
    if debug_logging_enabled:
        print("[日志最终] run_id=%s phase=closed close_reason=%s "
              "file_dropped=%d stdout_dropped=%d file_truncated=%d "
              "file_gap=%d close_elapsed_ms=%.1f" % (
                  _run_log.run_id, _run_log_result.get("close_reason"),
                  _run_log_result.get("file_dropped", 0),
                  _run_log_result.get("stdout_dropped", 0),
                  _run_log_result.get("truncated_count", 0),
                  _run_log_result.get("file_gap_count", 0),
                  _run_log_result.get("close_elapsed_s", 0.0) * 1000.0),
              flush=True)

# ============================================================
# 8. CLI 入口
# ============================================================

if __name__ == "__main__":
    run_engine()
