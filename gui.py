# gui.py — Aim 配置面板
# 退出键 = 按一次停止，再按一次启动

import copy, json, os, glob, time, threading, sys, traceback, ctypes, queue, re, math
import shutil, subprocess
from ctypes import wintypes
import tkinter as tk
from tkinter import ttk, messagebox
try:
    import winsound
except ImportError:  # 允许在非 Windows 环境复用 GUI 配置/测试模块
    winsound = None
from recoil_profiles import (
    DEFAULT_PROFILE_LABEL, ProfileStoreError, RecoilProfileStore,
    infer_recoil_path, validate_profile_name,
)
from asap_timing import merge_config_defaults
from runtime_logging import GuiLogPump, is_minimal_log_message, new_run_id
from hotkey_config import HOTKEY_DEFAULTS, normalize_hotkeys

try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
except: pass
user32 = ctypes.windll.user32; kernel32 = ctypes.windll.kernel32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, "frozen", False): BASE_DIR = os.path.dirname(sys.executable)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
RECOIL_PROFILES_PATH = os.path.join(BASE_DIR, "recoil_profiles.json")

_status_voice_lock = threading.Lock()
_status_voice_process = None


def _speak_engine_status(started):
    """Use the Windows built-in SAPI voice without blocking the GUI/engine."""
    global _status_voice_process
    if os.name != "nt":
        return False
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not powershell:
        return False
    phrase = "引擎已开启" if started else "引擎已关闭"
    command = f"(New-Object -ComObject SAPI.SpVoice).Speak('{phrase}')"
    with _status_voice_lock:
        previous = _status_voice_process
        try:
            if previous is not None and previous.poll() is None:
                previous.terminate()
        except Exception:
            pass
        try:
            _status_voice_process = subprocess.Popen(
                [powershell, "-NoProfile", "-NonInteractive", "-WindowStyle",
                 "Hidden", "-Command", command],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        except Exception:
            _status_voice_process = None
            return False


def play_engine_status_tone(started):
    """播放非阻塞系统提示音；声音失败不影响引擎或 GUI。"""
    if winsound is None:
        return False
    alias = "SystemAsterisk" if started else "SystemExclamation"
    try:
        winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
        return True
    except Exception:
        return False


def announce_engine_status(started):
    """Announce the state in words; use the old async tone only as fallback."""
    if _speak_engine_status(started):
        return "voice"
    return "tone" if play_engine_status_tone(started) else False

DEFAULT = {
    "model":"yolodeltav1.onnx","output_dir":"outputs","conf":0.15,"iou":0.55,
    "capture_size":640,"target_classes":[0,1],"target_priority":"body","chest_ratio":0.3584,"head_ratio":0.70,
    "anti_recoil":False,"recoil_profile":"unknown",
    "default_recoil":{"enabled":False,"strength":4.0},
    "recoil":{"recovery_ms":100.0,"output_period_ms":30.0,
              "curve_bin_ms":60.0,"calibration_runs":5,
              "min_calibration_bins":10,"max_target_gap_ms":120.0},
    "manual_recoil":{"enabled":True,"profile_name":"","playback_blend_percent":70.0,
                     "curve_bin_ms":60.0,"calibration_runs":5,
                     "min_burst_ms":300.0,"tail_vertical_ratio":0.30,
                     "tail_max_ms":1500.0,"raw_self_test":True,
                     "suppress_prediction_during_replay":True},
    "hotkeys":dict(HOTKEY_DEFAULTS),"precision":"fp16",
    "crosshair_offset_x":0.0,"crosshair_offset_y":0.0,
    "crosshair_mode":"dot_fallback","crosshair_search_radius":80,
    "crosshair_max_offset":45.0,"crosshair_max_jump":12.0,
    "crosshair_confirm_frames":2,"crosshair_hold_frames":2,
    "crosshair_return_frames":4,"crosshair_return_max_step":24.0,
    "crosshair_ema_alpha":0.35,"crosshair_fallback_decay":1.0,
    "humanize":{"sensitivity":1.5,"deadzone":0.0,
                "humanize_level":0.1,"max_aim_delta":120.0,"micro_jitter":0.02,"smoothing":0.5},
    "aim_control":{"smoothing":0.5,"target_ema":0.3,"ema_max_step":60.0,"max_total_gain":0.95,"reversal_damp":0.6044,"unit_max_counts":323.6093,"send_every_n_frames":1,"lead_frames":0.0,"delay_comp_ms":0.0,"move_digest_ms":55.0,"boost_lo_px":0.0,"boost_hi_px":130.0,"boost_max":1.0,"view_scale":0.3011,"view_scale_y":0.2684,"lock_box_filter_mode":"fixed","lock_box_smoothing_alpha":0.20,"lock_box_alpha_min":0.35,"lock_box_alpha_max":0.85,"lock_box_speed_low":80.0,"lock_box_speed_high":600.0,"unit_prediction_enabled":True,"prediction_mode":"arrival","prediction_min_ms":40.0,"prediction_max_ms":120.0,"prediction_box_ratio":0.75,"prediction_cap_px":45.0,"prediction_lock_frames":2,"recoil_prediction_box_ratio":0.35,"recoil_prediction_cap_px":20.0},
    "mouse_log_enabled":True,
    "max_target_step":80,"max_first_step":110,
    "box_scale_gain_enabled":False,
    "box_scale_gain":{"min_size":60,"max_size":320,"min_gain":1.3,"max_gain":0.6},
    "target_lock_distance":100.5581,"target_lock_iou":0.0614,"lock_grace_frames":2,
    "lock_zero_iou_max_distance":24.0,
    "switch_grace_frames":2,"lock_ambiguity_margin":0.2043,
    "lock_prediction_max_step":25.0,"edge_new_lock_margin":1.0,
    "unit":{"px_per_count":0.3011,"px_per_count_y":0.2684,"far_gain":0.8548,
            "near_gain":0.5226,"near_radius_px":31.4676,"deadzone":6.5892,"frame_schedule":"fixed",
            "frame_ms":100.0,"reference_frame_ms":22.0,
            "move_steps":2,"damped":False,"step_profile":"ease",
            "control_strategy":"rate_hold",
            "remaining_error":{"tau_ms":40.0,"max_speed_counts_s":4000.0,
                                "max_accel_counts_s2":0.0,"feedback_delay_ms":20.0,
                                "max_observation_age_ms":120.0}},
    "aim_interval_ms":1,"capture_center_mode":"screen",
    "auto_trigger":{"enabled":False,"min_size":80.0,"conf":0.65,"interval_ms":75,
                    "size":640,"grace_ms":300},
}
CLASS_NAMES=["敌人(0)","头部(1)"]
NUM_CLASSES=len(CLASS_NAMES)
KEY_OPTIONS = (
    [chr(c) for c in range(ord("a"),ord("z")+1)]+[str(n) for n in range(10)]
    +["`","ctrl","shift","alt","space","tab","enter","backspace","delete","esc","up","down","left","right"]
    +["f"+str(n) for n in range(1,13)]+["mouse_left","mouse_right","mouse_middle","mouse_x1","mouse_x2"]
)

_VK={}
for ch in "0123456789": _VK[ch]=ord(ch)
# 字母键：Windows 虚拟键码 = 大写 ASCII（'A'=0x41=65 … 'Z'=0x5A=90）。
# 旧代码用 ord(小写字母)（'p'=112=0x70=VK_F1），导致所有字母热键监听错键——
# 退出键"p"实际监听的是 F1，这就是 P 键完全没反应的真因。
for c in range(ord("a"),ord("z")+1): _VK[chr(c)]=c-32
for n in range(1,13): _VK[f"f{n}"]=0x6F+n
_VK.update({"space":0x20,"tab":0x09,"enter":0x0D,"backspace":0x08,"delete":0x2E,"esc":0x1B,
    "up":0x26,"down":0x28,"left":0x25,"right":0x27,"shift":0x10,"ctrl":0x11,"alt":0x12,
    "`":0xC0,  # 反引号键（ESC 下方，VK_OEM_3）
    "mouse_left":0x01,"mouse_right":0x02,"mouse_middle":0x04,"mouse_x1":0x05,"mouse_x2":0x06})

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH,"r",encoding="utf-8") as f: cfg=json.load(f)
        except: return merge_config_defaults({}, DEFAULT)
        cfg = merge_config_defaults(cfg, DEFAULT)
        cfg["target_classes"] = sorted({
            int(cls) for cls in cfg.get("target_classes", [])
            if int(cls) in range(NUM_CLASSES)
        })
        return cfg
    return merge_config_defaults({}, DEFAULT)

def save_config(cfg):
    with open(CONFIG_PATH,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)


def discover_onnx_models(dirs):
    """Return model filenames available to the GUI model selector."""
    return sorted(set(
        os.path.basename(path)
        for directory in dirs
        for path in glob.glob(os.path.join(directory, "*.onnx"))
    ))


def update_recoil_config(cfg, manual_enabled, blend_percent, tail_percent=None,
                         vertical_enabled=False):
    """Persist one mutually-exclusive recoil mode plus manual curve settings."""
    vertical_enabled = bool(vertical_enabled)
    cfg["anti_recoil"] = False
    default_recoil = cfg.setdefault("default_recoil", {})
    default_recoil["enabled"] = vertical_enabled
    default_recoil.setdefault("strength", 4.0)
    mr = cfg.setdefault("manual_recoil", {})
    mr["enabled"] = bool(manual_enabled) and not vertical_enabled
    mr.setdefault("profile_name", "")
    mr["playback_blend_percent"] = round(
        max(0.0, min(100.0, float(blend_percent))), 1)
    mr.setdefault("curve_bin_ms", 60.0)
    mr.setdefault("calibration_runs", 5)
    mr.setdefault("min_burst_ms", 300.0)
    if tail_percent is None:
        mr.setdefault("tail_vertical_ratio", 0.30)
    else:
        mr["tail_vertical_ratio"] = round(
            max(0.0, min(100.0, float(tail_percent))) / 100.0, 3)
    mr.setdefault("tail_max_ms", 1500.0)
    mr.setdefault("raw_self_test", True)
    mr.setdefault("suppress_prediction_during_replay", True)
    return cfg

class HotEvent:
    __slots__=("action","value")
    def __init__(self,action,value=None): self.action=action; self.value=value

class HotkeyPoller:
    """热键轮询器 — 由 GUI 主线程驱动，通过队列通知引擎线程。"""
    def __init__(self,hotkeys_cfg):
        self._q=queue.Queue(); self._running=False
        hotkeys_cfg=normalize_hotkeys(hotkeys_cfg,_VK)
        aim_s=hotkeys_cfg["aim"]
        ss_s=hotkeys_cfg["screenshot"]
        quit_s=hotkeys_cfg["quit"]
        recoil_s=hotkeys_cfg["recoil"]
        trigger_s=hotkeys_cfg["trigger"]
        self._keys=[]
        for s,name,is_hold in [(aim_s,"aim",True),(ss_s,"screenshot",False),
                                (quit_s,"quit",False),(recoil_s,"recoil",True),
                                (trigger_s,"trigger",False)]:
            vk=_VK.get(s)
            if vk is not None: self._keys.append((vk,name,is_hold))
        # Key edges belong to actions, not only to a VK.  Duplicate bindings
        # are rejected above; action-scoped state also prevents a future
        # caller from reintroducing the old shadowing bug accidentally.
        self._prev={name:False for _vk,name,_is_hold in self._keys}
        self._gui_cb=None; self._tick_count=0
        self._aim_state_callback=None
        for vk,name,is_hold in self._keys:
            if name=="quit": self._quit_vk=vk
    def set_gui_callback(self,cb): self._gui_cb=cb
    def set_aim_state_callback(self,cb): self._aim_state_callback=cb
    def set_aim_vk(self,vk):
        # GUI poller already knows the hold key; retained for the shared bridge.
        return None
    def set_manual_recoil_blend(self,percent):
        self._q.put(HotEvent("manual_recoil_blend",float(percent)))
    def set_manual_recoil_tail(self,percent):
        self._q.put(HotEvent("manual_recoil_tail",float(percent)))
    def set_manual_recoil_suppress_prediction(self,enabled):
        self._q.put(HotEvent(
            "manual_recoil_suppress_prediction",bool(enabled)))
    def set_prediction_mode(self,mode):
        self._q.put(HotEvent("prediction_mode",str(mode)))
    def start(self): self._running=True
    def stop(self): self._running=False
    def poll(self,timeout=0.01):
        try: return self._q.get(timeout=timeout)
        except queue.Empty: return None
    def tick(self):
        if not self._running: return
        self._tick_count+=1
        for vk,name,is_hold in self._keys:
            state=user32.GetAsyncKeyState(vk)
            pressed=bool(state&0x8000)
            was=self._prev.get(name,False)
            if pressed!=was:
                self._prev[name]=pressed
                if name=="quit": continue  # quit 键由 GUI 线程统一处理
                if is_hold:
                    if name=="aim" and self._aim_state_callback is not None:
                        self._aim_state_callback(pressed)
                    if name=="recoil":
                        self._q.put(HotEvent("recoil_down") if pressed else HotEvent("recoil_up"))
                    else:
                        self._q.put(HotEvent("aim_down") if pressed else HotEvent("aim_up"))
                elif pressed:
                    self._q.put(HotEvent(name))


class RecoilCurveCanvas(tk.Canvas):
    """Three diagnostic views of the same five-run trajectory dataset."""

    MODES = ("估算弹道趋势", "五次原始压枪", "X/Y 时间曲线")

    def __init__(self, parent, colors, **kwargs):
        super().__init__(
            parent, bg=colors["plot"], highlightthickness=1,
            highlightbackground=colors["line"], bd=0, **kwargs)
        self.colors = colors
        self.mode = self.MODES[0]
        self.snapshot = {}
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_mode(self, mode):
        self.mode = mode if mode in self.MODES else self.MODES[0]
        self.redraw()

    def set_snapshot(self, snapshot):
        self.snapshot = dict(snapshot or {})
        self.redraw()

    def _grid(self, width, height, pad):
        self.delete("all")
        for index in range(1, 6):
            x = pad + (width - 2 * pad) * index / 6
            y = pad + (height - 2 * pad) * index / 6
            self.create_line(x, pad, x, height - pad, fill=self.colors["grid"])
            self.create_line(pad, y, width - pad, y, fill=self.colors["grid"])
        self.create_rectangle(
            pad, pad, width - pad, height - pad,
            outline=self.colors["line"])

    @staticmethod
    def _cumulative(points):
        x = y = 0.0
        result = [(0.0, 0.0)]
        for dx, dy in points:
            x += float(dx); y += float(dy)
            result.append((x, y))
        return result

    def _fit(self, paths, width, height, pad, origin_bottom=False):
        points = [point for path in paths for point in path]
        if not points:
            return []
        if origin_bottom:
            max_x = max(1.0, max(abs(x) for x, _y in points))
            max_y = max(1.0, max(abs(y) for _x, y in points))
            scale = min((width - 2 * pad) / (2 * max_x),
                        (height - 2 * pad) / max_y)
            return [[
                (width / 2 + x * scale, height - pad + y * scale)
                for x, y in path
            ] for path in paths]
        xs = [x for x, _y in points]; ys = [y for _x, y in points]
        min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
        span_x = max(1.0, max_x - min_x); span_y = max(1.0, max_y - min_y)
        scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
        offset_x = (width - span_x * scale) / 2 - min_x * scale
        offset_y = (height - span_y * scale) / 2 - min_y * scale
        return [[(offset_x + x * scale, offset_y + y * scale)
                 for x, y in path] for path in paths]

    def _draw_path(self, points, color, width=2, dots=False, gradient=False):
        if len(points) >= 2:
            flat = [coordinate for point in points for coordinate in point]
            self.create_line(*flat, fill=color, width=width, smooth=True)
        if dots:
            count = max(1, len(points) - 1)
            for index, (x, y) in enumerate(points[1:], 1):
                ratio = index / count
                if gradient:
                    red = int(255)
                    green = int(116 + (74 * ratio))
                    blue = int(28 - (18 * ratio))
                    dot = f"#{red:02x}{green:02x}{blue:02x}"
                else:
                    dot = color
                radius = 3.0 if index < count else 4.5
                self.create_oval(
                    x-radius, y-radius, x+radius, y+radius,
                    fill=dot, outline="")

    def redraw(self):
        width = max(320, self.winfo_width())
        height = max(220, self.winfo_height())
        pad = 34
        self._grid(width, height, pad)
        curve = self.snapshot.get("curve") or []
        runs = self.snapshot.get("source_runs") or []
        if not curve and not runs:
            self.create_text(
                width/2, height/2-8, text="等待有效轨迹数据",
                fill=self.colors["muted"], font=("Microsoft YaHei UI", 12))
            self.create_text(
                width/2, height/2+20,
                text="按住物理左键完成 1/5…5/5 次连续扫射",
                fill=self.colors["dim"], font=("Microsoft YaHei UI", 9))
            return

        if self.mode == "估算弹道趋势":
            inferred = [(0.0, 0.0)] + infer_recoil_path(curve)
            fitted = self._fit([inferred], width, height, pad, origin_bottom=True)[0]
            self._draw_path(fitted, self.colors["accent_soft"], width=1,
                            dots=True, gradient=True)
            self.create_text(
                pad+8, pad+8, anchor="nw", text="估算趋势 · 非实测弹孔",
                fill=self.colors["warning"], font=("Microsoft YaHei UI", 9))
        elif self.mode == "五次原始压枪":
            paths = [self._cumulative(run) for run in runs]
            fitted = self._fit(paths, width, height, pad)
            palette = ("#ff6b1a", "#ff8a1f", "#e94618", "#f5a33a", "#c93816")
            for index, path in enumerate(fitted):
                self._draw_path(path, palette[index % len(palette)], width=2)
            self.create_text(
                pad+8, pad+8, anchor="nw", text=f"原始记录 · {len(runs)}/5",
                fill=self.colors["muted"], font=("Microsoft YaHei UI", 9))
        else:
            bin_ms = float(self.snapshot.get("bin_ms", 60.0))
            x_values = [(index * bin_ms, float(point[0]))
                        for index, point in enumerate(curve)]
            y_values = [(index * bin_ms, float(point[1]))
                        for index, point in enumerate(curve)]
            fitted = self._fit([x_values, y_values], width, height, pad)
            self._draw_path(fitted[0], self.colors["accent"], width=2, dots=True)
            self._draw_path(fitted[1], self.colors["danger"], width=2, dots=True)
            self.create_text(
                pad+8, pad+8, anchor="nw", text="X 水平",
                fill=self.colors["accent"], font=("Microsoft YaHei UI", 9))
            self.create_text(
                pad+78, pad+8, anchor="nw", text="Y 垂直",
                fill=self.colors["danger"], font=("Microsoft YaHei UI", 9))


class AimGUI:
    def __init__(self):
        self.root=tk.Tk(); self.root.title("YOLO 压枪与鼠标校准控制台"); self.root.resizable(True,True)
        self.root.geometry("1500x920"); self.root.minsize(1120,700)
        self.config=load_config()
        self._debug_logging_enabled=bool(
            self.config.get("mouse_log_enabled",DEFAULT["mouse_log_enabled"]))
        self.running=False; self.stop_event=None; self._poller=None
        self._running_loop=False; self._hk_after_id=None
        self._blend_after_id=None; self._tail_after_id=None
        self._hk_generation=0
        self._active_run_id=None
        self._runner_thread=None
        self._engine_ready_run_id=None
        self._engine_closed_tone_run_id=None
        self._gui_log_pump=GuiLogPump(maxsize=512)
        self._gui_log_after_id=None
        self._gui_log_max_lines=2000
        self._gui_log_line_count=0
        self._last_gui_refresh_ms=0.0
        self._quit_prev=False  # 退出键沿检测独立状态（不能共用 poller._prev，tick 会先消费按键沿）
        self._all_inputs=[]  # 所有需要启用/禁用的输入控件
        self._runtime_inputs=[]
        self._profile_store=RecoilProfileStore(RECOIL_PROFILES_PATH)
        self._profile_names=[]; self._profile_warning=""
        self._recoil_snapshot={}
        try:
            self._profile_names=self._profile_store.names()
        except ProfileStoreError as exc:
            self._profile_warning=f"曲线配置文件损坏，已回退默认：{exc}"
        selected=str(self.config.get("manual_recoil",{}).get("profile_name","") or "").strip()
        if selected and selected not in self._profile_names:
            self._profile_warning=f"找不到已选曲线“{selected}”，已回退默认"
            selected=""
        if selected:
            try:
                self._recoil_snapshot=self._profile_store.load(selected)
                self._recoil_snapshot.update({
                    "calibration_count":5,"ready":True,
                    "loaded_profile_name":selected,"profile_name":selected})
            except ProfileStoreError as exc:
                self._profile_warning=f"曲线“{selected}”损坏，已回退默认：{exc}"
                selected=""; self._recoil_snapshot={}
        self.config.setdefault("manual_recoil",{})["profile_name"]=selected
        self._build(); self._load_ui()
        self._gui_log_after_id=self.root.after(50,self._flush_gui_log)
        if self._profile_warning:
            self._do_log(f"[曲线配置] {self._profile_warning}")
            self._status_var.set("曲线已回退默认")
            save_config(self.config)
        self._hk_after_id=self.root.after(20,self._hk_poll_loop)
        self.root.protocol("WM_DELETE_WINDOW",self._on_close)

    # ── 控件注册辅助 ──
    def _reg(self, w): self._all_inputs.append(w); return w

    def _setup_theme(self):
        """集中定义工作台视觉令牌，避免每个标签页各自漂移。"""
        self._colors = {
            "canvas": "#151515", "panel": "#202020", "panel_alt": "#292929",
            "rail": "#191919", "line": "#454545", "grid": "#343434",
            "text": "#f4eee6", "muted": "#c5b8aa", "dim": "#8f8780",
            "accent": "#f2761b", "accent_soft": "#ff9a2f",
            "accent_dark": "#4c2110", "warning": "#ffb24a",
            "danger": "#e6461d", "success": "#74d66f",
            "log": "#111111", "plot": "#181818",
        }
        self.root.configure(bg=self._colors["canvas"])
        style=ttk.Style(self.root)
        style.theme_use("clam")
        c=self._colors
        style.configure("TFrame", background=c["panel"])
        style.configure("Panel.TFrame", background=c["panel_alt"])
        style.configure("Header.TFrame", background=c["canvas"])
        style.configure("Rail.TFrame", background=c["rail"])
        style.configure("Page.TFrame", background=c["panel"])
        style.configure("Log.TFrame", background=c["canvas"])
        style.configure("TLabel", background=c["panel"], foreground=c["text"], font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=c["panel"], foreground=c["muted"], font=("Segoe UI", 8))
        style.configure("Title.TLabel", background=c["canvas"], foreground=c["text"], font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Subtitle.TLabel", background=c["canvas"], foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=c["accent_dark"], foreground=c["warning"], padding=(12, 6), font=("Segoe UI Semibold", 9))
        style.configure("Section.TLabel", background=c["panel"], foreground=c["accent"], padding=(0, 9, 0, 4), font=("Segoe UI Semibold", 10))
        style.configure("PageTitle.TLabel", background=c["panel"], foreground=c["text"], font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Metric.TLabel", background=c["panel"], foreground=c["accent"], font=("Segoe UI Light", 22))
        style.configure("Success.TLabel", background=c["panel"], foreground=c["success"], font=("Segoe UI Semibold", 10))
        style.configure("TNotebook", background=c["canvas"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["panel_alt"], foreground=c["muted"], padding=(16, 8), font=("Segoe UI Semibold", 9))
        style.map("TNotebook.Tab", background=[("selected", c["accent_dark"]), ("active", c["line"])], foreground=[("selected", c["accent"]), ("active", c["text"])])
        style.configure("TButton", background=c["panel_alt"], foreground=c["text"], bordercolor=c["line"], padding=(12, 7), font=("Microsoft YaHei UI", 9))
        style.map("TButton", background=[("active", c["line"]), ("disabled", c["panel"])], foreground=[("disabled", c["muted"])])
        style.configure("Accent.TButton", background=c["accent"], foreground="#1a0b04", padding=(16, 8))
        style.map("Accent.TButton", background=[("active", "#ff9341"), ("disabled", c["line"])])
        style.configure("Danger.TButton", background="#542016", foreground="#ffd4c6", padding=(12, 7))
        style.map("Danger.TButton", background=[("active", c["danger"]), ("disabled", c["panel"])])
        style.configure("Nav.TButton", background=c["rail"], foreground=c["muted"], borderwidth=0,
                        anchor="w", padding=(18, 13), font=("Microsoft YaHei UI", 9))
        style.map("Nav.TButton", background=[("active", c["panel_alt"])], foreground=[("active", c["text"])])
        style.configure("NavActive.TButton", background=c["accent_dark"], foreground=c["accent_soft"],
                        borderwidth=0, anchor="w", padding=(18, 13), font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TEntry", fieldbackground=c["log"], foreground=c["text"], bordercolor=c["line"], padding=5)
        style.configure("TCombobox", fieldbackground=c["log"], background=c["panel_alt"], foreground=c["text"], arrowcolor=c["accent"], padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", c["log"]), ("disabled", c["panel"])], foreground=[("readonly", c["text"]), ("disabled", c["muted"])])
        style.configure("Horizontal.TScale", background=c["panel"], troughcolor=c["log"], slidercolor=c["accent"], sliderlength=18)
        style.configure("TCheckbutton", background=c["panel"], foreground=c["text"], indicatorbackground=c["log"], indicatorforeground=c["accent"], font=("Segoe UI", 9))
        style.map("TCheckbutton", foreground=[("disabled", c["muted"])])
        style.configure("Vertical.TScrollbar", background=c["panel_alt"], troughcolor=c["log"], arrowcolor=c["muted"])
        style.configure("TSeparator", background=c["line"])
        style.configure("TPanedwindow", background=c["canvas"])

    def _scrollable_tab(self, notebook, title):
        """让高密度调参页在小窗口和高 DPI 下仍可完整访问。"""
        outer=ttk.Frame(notebook)
        notebook.add(outer, text=title)
        canvas=tk.Canvas(outer, highlightthickness=0, bd=0, bg=self._colors["panel"])
        bar=ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner=ttk.Frame(canvas, padding=(16, 10, 18, 18))
        window=canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True); bar.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda _e, cv=canvas: cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta/120), "units")))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _scrollable_page(self, parent):
        outer=ttk.Frame(parent,style="Page.TFrame")
        canvas=tk.Canvas(outer,highlightthickness=0,bd=0,bg=self._colors["panel"])
        bar=ttk.Scrollbar(outer,orient="vertical",command=canvas.yview)
        inner=ttk.Frame(canvas,style="Page.TFrame",padding=(22,16,24,24))
        window=canvas.create_window((0,0),window=inner,anchor="nw")
        inner.bind("<Configure>",lambda _e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfigure(window,width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left",fill="both",expand=True); bar.pack(side="right",fill="y")
        canvas.bind("<Enter>",lambda _e,cv=canvas:cv.bind_all(
            "<MouseWheel>",lambda e:cv.yview_scroll(int(-e.delta/120),"units")))
        canvas.bind("<Leave>",lambda _e,cv=canvas:cv.unbind_all("<MouseWheel>"))
        return outer,inner

    def _show_page(self,key):
        if key not in self._pages:
            return
        for page_key,page in self._pages.items():
            if page_key==key: page.tkraise()
        for button_key,button in self._nav_buttons.items():
            button.configure(style="NavActive.TButton" if button_key==key else "Nav.TButton")
        self._active_page=key

    def _toggle_log_panel(self):
        if self._log_visible:
            self._log_panel.grid_remove()
            self._log_visible=False
            self._log_toggle_var.set("展开日志")
        else:
            self._log_panel.grid()
            self._log_visible=True
            self._log_toggle_var.set("收起日志")

    def _add_scale(self,parent,row,key,label,lo,hi,step,**kw):
        P={"padx":4,"pady":4}
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",**P)
        var=tk.DoubleVar(); self._scales[key]=var
        scale=self._reg(ttk.Scale(parent,from_=lo,to=hi,variable=var,orient="horizontal",
                                  length=kw.get("length",220),
                                  command=lambda v,k=key:self._on_scale(k,v)))
        scale.grid(row=row,column=1,sticky="ew",**P); self._scales[key+"_w"]=scale
        ent=self._reg(ttk.Entry(parent,textvariable=var,width=kw.get("width",9)))
        ent.grid(row=row,column=2,padx=2)
        ent.bind("<FocusOut>",lambda e,k=key,lo=lo,hi=hi:self._on_entry(k,lo,hi))
        ent.bind("<Return>",lambda e,k=key,lo=lo,hi=hi:self._on_entry(k,lo,hi))
        if kw.get("runtime",False):
            self._runtime_inputs.extend((scale,ent))
        return var

    def _section(self,parent,r,text):
        ttk.Separator(parent,orient="horizontal").grid(row=r,column=0,columnspan=3,sticky="ew",pady=(10,1)); r+=1
        ttk.Label(parent,text=text,style="Section.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",padx=3); r+=1
        return r

    def _build(self):
        self._setup_theme()
        self._scales={}; self._hotkey_vars={}; self._hotkey_widgets={}
        self._class_vars={}; self._class_widgets={}
        self.root.rowconfigure(1,weight=1); self.root.columnconfigure(0,weight=1)

        header=ttk.Frame(self.root,style="Header.TFrame",padding=(18,10))
        header.grid(row=0,column=0,sticky="ew")
        header.columnconfigure(1,weight=1)
        title_box=ttk.Frame(header,style="Header.TFrame")
        title_box.grid(row=0,column=0,rowspan=2,sticky="w",padx=(0,26))
        ttk.Label(title_box,text="YOLO 压枪与鼠标校准控制台",style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box,text="工业热控台 · 轨迹学习与实时回放",style="Subtitle.TLabel").pack(anchor="w",pady=(2,0))
        self._header_metrics=tk.StringVar(value="引擎：未启动    配置：默认    输入：等待")
        ttk.Label(header,textvariable=self._header_metrics,style="Subtitle.TLabel").grid(
            row=0,column=1,sticky="w")
        self._status_var=tk.StringVar(value="就绪")
        self._status_badge=ttk.Label(header,textvariable=self._status_var,style="Status.TLabel")
        self._status_badge.grid(row=0,column=2,sticky="e",padx=(12,8))
        self._btn_var=tk.StringVar(value="启动引擎")
        self._start_btn=ttk.Button(header,textvariable=self._btn_var,command=self._toggle,width=13,style="Accent.TButton")
        self._start_btn.grid(row=0,column=3,rowspan=2,sticky="e",padx=(8,0))
        self._log_toggle_var=tk.StringVar(value="收起日志")
        ttk.Button(header,textvariable=self._log_toggle_var,command=self._toggle_log_panel,
                   width=9).grid(row=1,column=2,sticky="e",padx=(12,8),pady=(5,0))

        body=ttk.Frame(self.root,style="Header.TFrame")
        body.grid(row=1,column=0,sticky="nsew")
        body.rowconfigure(0,weight=1); body.columnconfigure(1,weight=1)
        nav=ttk.Frame(body,style="Rail.TFrame",padding=(0,12))
        nav.grid(row=0,column=0,sticky="nsw")
        ttk.Label(nav,text="任务导航",background=self._colors["rail"],
                  foreground=self._colors["dim"],font=("Microsoft YaHei UI",8)).pack(
                      anchor="w",padx=18,pady=(2,8))
        self._nav_buttons={}
        nav_items=(
            ("basic","工作台"),("aim","瞄准与锁定"),
            ("manual","手动轨迹压枪"),("advanced","高级校准"),
            ("system","热键与实验"),
        )
        for key,label in nav_items:
            button=ttk.Button(nav,text=label,style="Nav.TButton",width=18,
                              command=lambda page=key:self._show_page(page))
            button.pack(fill="x",pady=1); self._nav_buttons[key]=button

        page_host=ttk.Frame(body,style="Page.TFrame")
        page_host.grid(row=0,column=1,sticky="nsew")
        page_host.rowconfigure(0,weight=1); page_host.columnconfigure(0,weight=1)
        self._pages={}
        page_basic,basic=self._scrollable_page(page_host)
        page_aim,aim=self._scrollable_page(page_host)
        page_advanced,advanced=self._scrollable_page(page_host)
        page_system,system=self._scrollable_page(page_host)
        page_manual=ttk.Frame(page_host,style="Page.TFrame",padding=(22,16,22,18))
        self._pages.update({"basic":page_basic,"aim":page_aim,"manual":page_manual,
                            "advanced":page_advanced,"system":page_system})
        for page in self._pages.values(): page.grid(row=0,column=0,sticky="nsew")
        self._build_basic(basic); self._build_aim(aim)
        self._build_manual_recoil(page_manual); self._build_advanced(advanced)
        reaction_box=ttk.Frame(system,style="Page.TFrame"); reaction_box.pack(fill="x")
        hotkey_box=ttk.Frame(system,style="Page.TFrame"); hotkey_box.pack(fill="x",pady=(18,0))
        self._build_reaction(reaction_box); self._build_hotkeys(hotkey_box)

        self._log_panel=ttk.Frame(body,style="Log.TFrame",padding=(12,10,14,12))
        self._log_panel.grid(row=0,column=2,sticky="nsew")
        self._log_visible=True
        ttk.Label(self._log_panel,text="运行日志",style="Title.TLabel").pack(anchor="w",pady=(0,2))
        ttk.Label(self._log_panel,text="引擎、目标锁定与鼠标输出诊断",style="Subtitle.TLabel").pack(anchor="w",pady=(0,8))
        lf=ttk.Frame(self._log_panel,style="Panel.TFrame",padding=6); lf.pack(fill="both",expand=True)
        self._log_text=tk.Text(lf,width=44,height=28,state="disabled",wrap="word",font=("Consolas",9),
                               bg=self._colors["log"],fg=self._colors["text"],insertbackground=self._colors["accent"],
                               selectbackground=self._colors["accent_dark"],relief="flat",padx=10,pady=10)
        sb=ttk.Scrollbar(lf,command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side="left",fill="both",expand=True); sb.pack(side="right",fill="y")

        footer=ttk.Frame(self.root,style="Header.TFrame",padding=(18,9))
        footer.grid(row=2,column=0,sticky="ew")
        ttk.Button(footer,text="保存全部配置",command=self._save,width=14).pack(side="left",padx=(0,6))
        ttk.Button(footer,text="重新采集压枪轨迹",command=self._reset_recoil_calibration,
                   width=17).pack(side="left",padx=3)
        ttk.Label(footer,text="高风险参数仅在引擎停止时可编辑；曲线命名与回放比例支持运行时操作",
                  style="Subtitle.TLabel").pack(side="right",padx=4)
        self._show_page("manual")

    def _build_basic(self,p):
        r=0; P={"padx":3,"pady":1}
        p.columnconfigure(1, weight=1)
        r=self._section(p,r,"▎模型")
        ttk.Label(p,text="  模型:").grid(row=r,column=0,sticky="w",**P)
        self._model_var=tk.StringVar()
        self._model_combo=self._reg(ttk.Combobox(p,textvariable=self._model_var,state="readonly",width=26))
        self._model_combo.grid(row=r,column=1,columnspan=2,sticky="ew",**P)
        self._refresh_models(); r+=1

        ttk.Label(p,text="  尺寸:").grid(row=r,column=0,sticky="w",**P)
        self._model_size_var=tk.StringVar()
        size_cb=self._reg(ttk.Combobox(p,textvariable=self._model_size_var,
                                        values=["256","320","416","512","640"],state="readonly",width=26))
        size_cb.grid(row=r,column=1,columnspan=2,sticky="ew",**P)
        size_cb.bind("<<ComboboxSelected>>",self._on_model_size_change); r+=1

        ttk.Label(p,text="  精度:").grid(row=r,column=0,sticky="w",**P)
        self._precision_var=tk.StringVar()
        prec_cb=self._reg(ttk.Combobox(p,textvariable=self._precision_var,
                                        values=["fp16","high","fast","fp32"],state="readonly",width=26))
        prec_cb.grid(row=r,column=1,columnspan=2,sticky="ew",**P); r+=1

        r=self._section(p,r,"▎检测")
        for key,label,lo,hi,step in [
            ("conf","置信度",0.05,0.95,0.05),
            ("capture_size","截图边长(px)",128,2000,16),
            ("chest_ratio","瞄准部位",0.0,1.0,0.05),
        ]:
            self._add_scale(p,r,key,label,lo,hi,step); r+=1

        r=self._section(p,r,"▎目标类别")
        cf=ttk.Frame(p); cf.grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        for i in range(NUM_CLASSES):
            var=tk.BooleanVar(); self._class_vars[i]=var
            cb=self._reg(ttk.Checkbutton(cf,text=CLASS_NAMES[i],variable=var))
            self._class_widgets[i]=cb
            cb.grid(row=i//3,column=i%3,sticky="w",padx=2,pady=1)
        ttk.Label(p,text="  同时检测头/身框时:").grid(row=r,column=0,sticky="w",**P)
        self._target_priority_var=tk.StringVar(value="优先身框")
        priority_cb=self._reg(ttk.Combobox(
            p,textvariable=self._target_priority_var,
            values=["优先身框","优先头框"],state="readonly",width=14))
        priority_cb.grid(row=r,column=1,sticky="w",**P)
        ttk.Label(p,text="(只影响新锁定候选；已锁定目标不强制切换)",
                  style="Muted.TLabel").grid(row=r,column=2,sticky="w",padx=2)

    def _build_hotkeys(self,p):
        r=0; P={"padx":3,"pady":1}
        p.columnconfigure(1, weight=1)
        r=self._section(p,r,"▎热键")
        for key,label in [("aim","瞄准键"),("recoil","压枪键"),("screenshot","截图键"),("trigger","自动触发开关"),("quit","退出键")]:
            ttk.Label(p,text=f"  {label}:").grid(row=r,column=0,sticky="w",**P)
            var=tk.StringVar(); self._hotkey_vars[key]=var
            cb=self._reg(ttk.Combobox(p,textvariable=var,values=KEY_OPTIONS,state="readonly",width=12))
            self._hotkey_widgets[key]=cb; cb.grid(row=r,column=1,sticky="w",**P)
            if key=="quit":
                ttk.Label(p,text="(切换启停)",style="Muted.TLabel").grid(row=r,column=2,sticky="w",padx=2)
            r+=1

        r=self._section(p,r,"▎日志")
        self._mouse_log_var=tk.BooleanVar()
        cb=self._reg(ttk.Checkbutton(
            p, text="  启用调试日志（后台输出/鼠标明细）",
            variable=self._mouse_log_var))
        cb.grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1

    def _build_aim(self,p):
        r=0; P={"padx":3,"pady":1}
        p.columnconfigure(1, weight=1)
        # ── 瞄准模式 ──
        ttk.Label(p,text="  瞄准模式:").grid(row=r,column=0,sticky="w",**P)
        self._aim_mode_var=tk.StringVar()
        cb=self._reg(ttk.Combobox(p,textvariable=self._aim_mode_var,
                                  values=["满额纠错(固定节拍)","平滑(旧架构)"],state="readonly",width=18))
        cb.grid(row=r,column=1,sticky="w",**P); r+=1
        ttk.Label(p,text="  (满额=固定节拍+每图一移到位; 平滑=旧连续架构)",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1

        r=self._section(p,r,"▎固定节拍")
        ttk.Label(p,text="  识别调度:").grid(row=r,column=0,sticky="w",**P)
        self._frame_schedule_var=tk.StringVar(value="固定周期")
        self._frame_schedule_cb=self._reg(ttk.Combobox(
            p,textvariable=self._frame_schedule_var,
            values=["跟随推理速度","固定周期"],state="readonly",width=18))
        self._frame_schedule_cb.grid(row=r,column=1,columnspan=2,sticky="ew",**P); r+=1
        ttk.Label(p,text="  (跟随推理=asap: 上一帧完成立即下一帧,无人工等待;"
                         "固定周期=按识别节拍等待)",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        ttk.Label(p,text="  输出控制策略:").grid(row=r,column=0,sticky="w",**P)
        self._unit_control_var=tk.StringVar(value="现有速率保持")
        self._unit_control_cb=self._reg(ttk.Combobox(
            p,textvariable=self._unit_control_var,
            values=["现有速率保持","异步剩余误差"],state="readonly",width=18))
        self._unit_control_cb.grid(row=r,column=1,columnspan=2,sticky="ew",**P); r+=1
        ttk.Label(p,text="  (剩余误差：视觉只发布快照；鼠标按成功提交量逐周期减速、到死区停止)",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        self._add_scale(p,r,"u_frame_ms","识别节拍(ms)",20,150,5); r+=1
        ttk.Label(p,text="  控制标称周期 reference_frame_ms 保留配置值；仅用于速率/滤波/预测归一化，ASAP 不参与等待",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        self._add_scale(p,r,"u_output_period_ms","共享鼠标输出周期(ms)",5,60,1); r+=1
        ttk.Label(p,text="  异步剩余误差参数").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        self._add_scale(p,r,"u_tau_ms","响应时间常数 tau(ms)",10,150,1); r+=1
        self._add_scale(p,r,"u_max_speed_counts_s","输出速度上限(counts/s)",500,8000,50); r+=1
        self._add_scale(p,r,"u_max_accel_counts_s2","加速度上限(counts/s²,0=关闭)",0,50000,100); r+=1
        self._add_scale(p,r,"u_feedback_delay_ms","输入反馈估计(ms)",0,100,1); r+=1
        self._add_scale(p,r,"u_max_observation_age_ms","观测最大年龄(ms)",40,300,1); r+=1
        self._add_scale(p,r,"u_kx","k标定·水平(px/count)",0.05,1.0,0.01); r+=1
        self._add_scale(p,r,"u_ky","k标定·纵向(px/count)",0.05,1.0,0.01); r+=1
        self._add_scale(p,r,"u_steps","缓出分配周期数",1,8,1); r+=1
        ttk.Label(p,text="  (缓出分配仅“固定周期”模式生效;"
                         "“跟随推理速度”按最新控制速率输出)",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        ttk.Label(p,text="  移动目标策略:").grid(row=r,column=0,sticky="w",**P)
        self._prediction_mode_var=tk.StringVar(value="到达时刻预测")
        self._prediction_mode_cb=self._reg(ttk.Combobox(
            p,textvariable=self._prediction_mode_var,
            values=["到达时刻预测","当前跟踪点（无前导）"],state="readonly",width=22))
        self._prediction_mode_cb.grid(row=r,column=1,columnspan=2,sticky="ew",**P)
        self._prediction_mode_cb.bind("<<ComboboxSelected>>",self._on_prediction_mode)
        self._runtime_inputs.append(self._prediction_mode_cb); r+=1
        ttk.Label(p,text="  目标框滤波:").grid(row=r,column=0,sticky="w",**P)
        self._lock_filter_mode_var=tk.StringVar(value="固定(手动alpha)")
        self._lock_filter_mode_cb=self._reg(ttk.Combobox(
            p,textvariable=self._lock_filter_mode_var,
            values=["固定(手动alpha)","自适应(0.35～0.85)"],
            state="readonly",width=22))
        self._lock_filter_mode_cb.grid(row=r,column=1,columnspan=2,sticky="ew",**P)
        r+=1
        self._add_scale(
            p,r,"lock_box_smoothing_alpha","固定滤波 alpha",0.05,1.0,0.01)
        r+=1

        ttk.Separator(p,orient="horizontal").grid(row=r,column=0,columnspan=3,sticky="ew",pady=(12,4)); r+=1
        self._smooth_toggle_var=tk.StringVar(value="展开平滑模式参数")
        ttk.Button(p,textvariable=self._smooth_toggle_var,command=self._toggle_smooth_panel).grid(
            row=r,column=0,columnspan=3,sticky="ew",padx=3,pady=2); r+=1
        self._smooth_body=ttk.Frame(p,style="Page.TFrame")
        self._smooth_body.grid(row=r,column=0,columnspan=3,sticky="ew",padx=3,pady=(2,8))
        self._smooth_body.columnconfigure(1,weight=1)
        rr=0
        for key,label,lo,hi,step in [
            ("smoothing","平滑度",0.01,1.0,0.01),
            ("target_ema","目标EMA",0.0,0.95,0.05),
            ("ema_max_step","EMA单帧限幅(px)",0,150,5),
            ("max_total_gain","总增益上限",0.0,1.5,0.05),
            ("reversal_damp","换向阻尼(0.5-2)",0.5,2.0,0.05),
            ("send_every_n_frames","发送周期(帧,1=每帧)",1,8,1),
            ("delay_comp_ms","延迟补偿(ms,0=关)",0,100,5),
            ("move_digest_ms","等待基量(ms)",0,150,5),
            ("boost_lo_px","远距增益起点(px,0=关)",0,150,5),
            ("boost_hi_px","远距增益满点(px)",60,300,5),
            ("boost_max","远距增益倍数(1=关)",1.0,3.0,0.1),
        ]:
            self._add_scale(self._smooth_body,rr,key,label,lo,hi,step); rr+=1
        self._smooth_body.grid_remove(); self._smooth_expanded=False

    def _toggle_smooth_panel(self):
        self._smooth_expanded=not self._smooth_expanded
        if self._smooth_expanded:
            self._smooth_body.grid()
            self._smooth_toggle_var.set("收起平滑模式参数")
        else:
            self._smooth_body.grid_remove()
            self._smooth_toggle_var.set("展开平滑模式参数")

    def _build_manual_recoil(self,p):
        p.columnconfigure(0,weight=1); p.rowconfigure(3,weight=1)
        title=ttk.Frame(p,style="Page.TFrame")
        title.grid(row=0,column=0,sticky="ew",pady=(0,10))
        ttk.Label(title,text="手动轨迹压枪",style="PageTitle.TLabel").pack(side="left")
        self._calibration_state_var=tk.StringVar(value="0 / 5  CALIBRATING")
        ttk.Label(title,textvariable=self._calibration_state_var,style="Metric.TLabel").pack(side="right")

        profile=ttk.Frame(p,style="Panel.TFrame",padding=(14,12))
        profile.grid(row=1,column=0,sticky="ew",pady=(0,10))
        profile.columnconfigure(1,weight=1); profile.columnconfigure(3,weight=1)
        ttk.Label(profile,text="配置档案",background=self._colors["panel_alt"]).grid(
            row=0,column=0,sticky="w",padx=(0,8),pady=4)
        self._profile_select_var=tk.StringVar(value=DEFAULT_PROFILE_LABEL)
        self._profile_selector=self._reg(ttk.Combobox(
            profile,textvariable=self._profile_select_var,state="readonly",width=26))
        self._profile_selector.grid(row=0,column=1,sticky="ew",padx=(0,14),pady=4)
        self._profile_selector.bind("<<ComboboxSelected>>",self._on_profile_selected)
        ttk.Label(profile,text="曲线名称",background=self._colors["panel_alt"]).grid(
            row=0,column=2,sticky="w",padx=(0,8),pady=4)
        self._profile_name_var=tk.StringVar()
        self._profile_name_entry=self._reg(ttk.Entry(
            profile,textvariable=self._profile_name_var,width=28))
        self._profile_name_entry.grid(row=0,column=3,sticky="ew",padx=(0,10),pady=4)
        self._runtime_inputs.append(self._profile_name_entry)
        actions=ttk.Frame(profile,style="Panel.TFrame")
        actions.grid(row=1,column=0,columnspan=4,sticky="ew",pady=(7,0))
        self._save_profile_button=ttk.Button(
            actions,text="保存当前",command=self._save_current_profile,style="Accent.TButton")
        self._save_as_button=ttk.Button(actions,text="另存为",command=self._save_profile_as)
        self._delete_profile_button=self._reg(ttk.Button(
            actions,text="删除",command=self._delete_profile,style="Danger.TButton"))
        self._save_profile_button.pack(side="left",padx=(0,6))
        self._save_as_button.pack(side="left",padx=3)
        self._delete_profile_button.pack(side="left",padx=3)
        self._profile_save_buttons=(self._save_profile_button,self._save_as_button)
        self._manual_recoil_enabled_var=tk.BooleanVar()
        cb=self._reg(ttk.Checkbutton(
            actions,text="启用手动轨迹压枪",
            variable=self._manual_recoil_enabled_var,
            command=self._on_manual_recoil_mode_toggle))
        cb.pack(side="right",padx=(10,0))
        self._vertical_recoil_enabled_var=tk.BooleanVar()
        vertical_cb=self._reg(ttk.Checkbutton(
            actions,text="启用默认垂直压枪",
            variable=self._vertical_recoil_enabled_var,
            command=self._on_vertical_recoil_mode_toggle))
        vertical_cb.pack(side="right",padx=(10,0))

        toolbar=ttk.Frame(p,style="Page.TFrame")
        toolbar.grid(row=2,column=0,sticky="ew",pady=(0,6))
        ttk.Label(toolbar,text="轨迹预览",style="Section.TLabel").pack(side="left")
        self._curve_meta_var=tk.StringVar(value="60 ms / bin    0 ms    0 bins")
        ttk.Label(toolbar,textvariable=self._curve_meta_var,style="Muted.TLabel").pack(side="left",padx=14)
        self._preview_mode_var=tk.StringVar(value=RecoilCurveCanvas.MODES[0])
        self._preview_mode_combo=self._reg(ttk.Combobox(
            toolbar,textvariable=self._preview_mode_var,
            values=RecoilCurveCanvas.MODES,state="readonly",width=16))
        self._preview_mode_combo.pack(side="right")
        self._preview_mode_combo.bind("<<ComboboxSelected>>",self._on_preview_mode)
        self._runtime_inputs.append(self._preview_mode_combo)

        self._curve_canvas=RecoilCurveCanvas(p,self._colors,height=390)
        self._curve_canvas.grid(row=3,column=0,sticky="nsew")

        controls=ttk.Frame(p,style="Panel.TFrame",padding=(14,9))
        controls.grid(row=4,column=0,sticky="ew",pady=(10,0))
        controls.columnconfigure(1,weight=1)
        self._add_scale(controls,0,"manual_recoil_blend","回放强度 (%)",0,100,1,runtime=True,length=300)
        self._add_scale(controls,1,"manual_recoil_tail","尾段垂直保留 (%)",0,100,1,runtime=True,length=300)
        self._add_scale(controls,2,"default_recoil_strength","默认垂直强度（每识别周期）",0,20,0.5,length=300)
        self._manual_recoil_suppress_prediction_var=tk.BooleanVar(
            value=bool(DEFAULT["manual_recoil"]["suppress_prediction_during_replay"]))
        suppress_cb=self._reg(ttk.Checkbutton(
            controls,text="回放时仅保留受限横向前导（抑制垂直假速度）",
            variable=self._manual_recoil_suppress_prediction_var,
            command=self._on_manual_recoil_suppress_prediction))
        suppress_cb.grid(row=3,column=0,columnspan=3,sticky="w",padx=4,pady=(4,1))
        self._runtime_inputs.append(suppress_cb)
        self._manual_hint_var=tk.StringVar(
            value="默认档案：物理左键按下立即记录；五次均≥300ms。回放使用右键→左键。")
        ttk.Label(controls,textvariable=self._manual_hint_var,style="Muted.TLabel").grid(
            row=4,column=0,columnspan=3,sticky="w",padx=4,pady=(5,1))
        self._refresh_profile_selector()
        self._update_recoil_preview(self._recoil_snapshot)

    def _build_advanced(self,p):
        # manual_recoil_tail 已迁入独立的“手动轨迹压枪”工作页。
        r=0
        p.columnconfigure(1, weight=1)
        ttk.Label(p,text="高级校准",style="PageTitle.TLabel").grid(
            row=r,column=0,columnspan=3,sticky="w",pady=(0,8)); r+=1
        r=self._section(p,r,"▎准星偏移")
        ttk.Label(p,text="  瞄准参考点:").grid(row=r,column=0,sticky="w",padx=3,pady=1)
        self._crosshair_mode_var=tk.StringVar()
        self._crosshair_mode_cb=self._reg(ttk.Combobox(p,textvariable=self._crosshair_mode_var,
            values=["只认红点","允许红点退化","只认屏幕中心"],state="readonly",width=14))
        self._crosshair_mode_cb.grid(row=r,column=1,columnspan=2,sticky="ew",padx=2,pady=1); r+=1
        self._add_scale(p,r,"crosshair_search_radius","红点搜索半径(px)",40,300,10); r+=1
        ttk.Label(p,text="  X偏移:").grid(row=r,column=0,sticky="w",padx=3,pady=1)
        self._crosshair_x_var=tk.DoubleVar(value=0.0)
        self._reg(ttk.Entry(p,textvariable=self._crosshair_x_var,width=7)).grid(row=r,column=1,sticky="w",padx=2,pady=1); r+=1
        ttk.Label(p,text="  Y偏移:").grid(row=r,column=0,sticky="w",padx=3,pady=1)
        self._crosshair_y_var=tk.DoubleVar(value=0.0)
        self._reg(ttk.Entry(p,textvariable=self._crosshair_y_var,width=7)).grid(row=r,column=1,sticky="w",padx=2,pady=1); r+=1

    # ── scale/entry helpers ──
    def _build_reaction(self,p):
        """反应时间测试: 并行线程, 只识别记录, 不动鼠标。"""
        r=0; P={"padx":3,"pady":1}
        p.columnconfigure(1, weight=1)
        r=self._section(p,r,"▎反应时间测试(与瞄准并行)")
        self._rt_en_var=tk.BooleanVar()
        cb=self._reg(ttk.Checkbutton(p,text="  启用(引擎启动时并行运行, 不影响瞄准)",
                                     variable=self._rt_en_var))
        cb.grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        self._add_scale(p,r,"rt_min_size","大框阈值(px,最大边≥此值)",40,300,5); r+=1
        self._add_scale(p,r,"rt_interval_ms","识别周期(ms,75=13fps)",40,200,5); r+=1
        ttk.Label(p,text="  规则: 大框出现→记录右键/左键首次按下; 双键都按下立即完成;",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        ttk.Label(p,text="  1s 无任何按键或目标消失(且未按下过)才作废。",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        ttk.Label(p,text="  试次实时输出在右侧日志; ESC 仅结束测试(瞄准继续)。",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1

        r=self._section(p,r,"▎自动触发(独立识别)")
        self._at_en_var=tk.BooleanVar()
        cb=self._reg(ttk.Checkbutton(p,text="  启用(识别到目标框自动按住右键+启动瞄准)",
                                     variable=self._at_en_var))
        cb.grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        self._add_scale(p,r,"at_min_size","触发大框阈值(px,最大边≥此值)",40,300,5); r+=1
        self._add_scale(p,r,"at_conf","触发置信度(0.3-0.95)",0.3,0.95,0.05); r+=1
        self._add_scale(p,r,"at_interval_ms","识别周期(ms)",40,200,5); r+=1
        self._add_scale(p,r,"at_grace_ms","目标消失宽限(ms,到期松开右键)",0,1000,10); r+=1
        ttk.Label(p,text="  条件: 目标框最大边≥阈值 且 置信度≥阈值 → 按住右键开始瞄准移动;",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1
        ttk.Label(p,text="  目标消失超过宽限期松开右键停止; 热键(默认F6)随时开关。",
                  style="Muted.TLabel").grid(row=r,column=0,columnspan=3,sticky="w",**P); r+=1

    def _on_scale(self,key,val):
        try: v=float(val)
        except: return
        fmt_keys={"conf"}
        if key in fmt_keys: v=round(v,3)
        elif key in("humanize_level","sensitivity","chest_ratio","reversal_damp",
                     "lock_box_smoothing_alpha"): v=round(v,2)
        self._scales[key].set(v)
        if key in ("manual_recoil_blend","manual_recoil_tail") and self.running and self._poller:
            self._schedule_manual_recoil_runtime(key)

    def _on_entry(self,key,lo,hi):
        try: v=float(self._scales[key].get()); v=max(lo,min(hi,v))
        except: return
        fmt_keys={"conf"}
        if key in fmt_keys: v=round(v,3)
        elif key in("humanize_level","sensitivity","chest_ratio","reversal_damp",
                     "lock_box_smoothing_alpha"): v=round(v,2)
        self._scales[key].set(v)
        if key in ("manual_recoil_blend","manual_recoil_tail") and self.running and self._poller:
            self._send_manual_recoil_runtime(key)

    def _schedule_manual_recoil_runtime(self,key):
        attr="_blend_after_id" if key=="manual_recoil_blend" else "_tail_after_id"
        pending=getattr(self,attr,None)
        if pending is not None:
            try: self.root.after_cancel(pending)
            except Exception: pass
        setattr(self,attr,self.root.after(
            30,lambda k=key:self._send_manual_recoil_runtime(k)))

    def _send_manual_recoil_runtime(self,key):
        attr="_blend_after_id" if key=="manual_recoil_blend" else "_tail_after_id"
        setattr(self,attr,None)
        if self.running and self._poller:
            value=self._scales[key].get()
            if key=="manual_recoil_blend": self._poller.set_manual_recoil_blend(value)
            else: self._poller.set_manual_recoil_tail(value)

    def _on_manual_recoil_suppress_prediction(self):
        if self.running and self._poller:
            self._poller.set_manual_recoil_suppress_prediction(
                self._manual_recoil_suppress_prediction_var.get())

    def _on_manual_recoil_mode_toggle(self):
        if self._manual_recoil_enabled_var.get():
            self._vertical_recoil_enabled_var.set(False)

    def _on_vertical_recoil_mode_toggle(self):
        if self._vertical_recoil_enabled_var.get():
            self._manual_recoil_enabled_var.set(False)

    def _on_prediction_mode(self,event=None):
        mode = ("arrival" if self._prediction_mode_var.get() == "到达时刻预测"
                else "current")
        if self._poller and self.running:
            self._poller.set_prediction_mode(mode)
            self._append_log(
                "[GUI] 移动目标策略=" +
                ("到达时刻预测" if mode == "arrival" else "当前跟踪点"))

    def _refresh_profile_selector(self,select_name=None):
        try:
            self._profile_names=self._profile_store.names()
        except ProfileStoreError as exc:
            self._profile_names=[]
            self._profile_warning=str(exc)
        values=[DEFAULT_PROFILE_LABEL]+self._profile_names
        if hasattr(self,"_profile_selector"):
            self._profile_selector.configure(values=values)
        selected=(select_name if select_name is not None else
                  str(self.config.get("manual_recoil",{}).get("profile_name","") or ""))
        if selected not in self._profile_names: selected=""
        self._profile_select_var.set(selected or DEFAULT_PROFILE_LABEL)
        self._profile_name_var.set(selected)

    def _on_profile_selected(self,_event=None):
        selected=self._profile_select_var.get()
        if selected==DEFAULT_PROFILE_LABEL:
            self._set_default_profile(persist=True)
            return
        try:
            snapshot=self._profile_store.load(selected)
        except ProfileStoreError as exc:
            messagebox.showwarning("曲线不可用",f"{exc}\n\n已回退到默认重新校准。")
            self._set_default_profile(persist=True)
            return
        snapshot.update({"calibration_count":5,"ready":True,
                         "loaded_profile_name":selected,"profile_name":selected})
        self._recoil_snapshot=snapshot
        self.config.setdefault("manual_recoil",{})["profile_name"]=selected
        self._profile_name_var.set(selected)
        # Choosing a saved manual curve is also choosing the recoil mode.
        # Previously the profile was persisted while an older
        # ``default_recoil.enabled=true`` selection remained active, so the
        # engine loaded the curve but every shot still used default vertical.
        self._manual_recoil_enabled_var.set(True)
        self._vertical_recoil_enabled_var.set(False)
        update_recoil_config(
            self.config, True,
            self._scales["manual_recoil_blend"].get(),
            self._scales["manual_recoil_tail"].get(),
            vertical_enabled=False)
        save_config(self.config)
        self._update_recoil_preview(snapshot)
        self._append_log(f"[曲线配置] 已选择“{selected}”；下次启动直接进入 READY")

    def _set_default_profile(self,persist=False):
        self.config.setdefault("manual_recoil",{})["profile_name"]=""
        self._recoil_snapshot={}
        self._profile_select_var.set(DEFAULT_PROFILE_LABEL)
        self._profile_name_var.set("")
        if hasattr(self, "_manual_recoil_enabled_var"):
            self._manual_recoil_enabled_var.set(True)
            self._vertical_recoil_enabled_var.set(False)
        if persist:
            update_recoil_config(
                self.config, True,
                self._scales["manual_recoil_blend"].get(),
                self._scales["manual_recoil_tail"].get(),
                vertical_enabled=False)
            save_config(self.config)
        self._update_recoil_preview({})

    def _on_preview_mode(self,_event=None):
        self._curve_canvas.set_mode(self._preview_mode_var.get())

    def _update_recoil_preview(self,snapshot):
        self._recoil_snapshot=dict(snapshot or {})
        count=int(self._recoil_snapshot.get("calibration_count",
                  len(self._recoil_snapshot.get("source_runs") or [])))
        ready=bool(self._recoil_snapshot.get("ready",count>=5))
        loaded=self._recoil_snapshot.get("loaded_profile_name")
        if ready:
            self._calibration_state_var.set("5 / 5  READY")
            self._manual_hint_var.set(
                f"已就绪{f' · 已加载 {loaded}' if loaded else ''}；回放使用右键→左键。")
        else:
            self._calibration_state_var.set(f"{count} / 5  CALIBRATING")
            self._manual_hint_var.set(
                "物理左键按下立即记录；五次均≥300ms。YOLO识别与瞄准移动保持运行。")
        curve=self._recoil_snapshot.get("curve") or []
        bin_ms=float(self._recoil_snapshot.get("bin_ms",60.0))
        duration=float(self._recoil_snapshot.get("duration_ms",len(curve)*bin_ms))
        self._curve_meta_var.set(f"{bin_ms:.0f} ms / bin    {duration:.0f} ms    {len(curve)} bins")
        self._curve_canvas.set_snapshot(self._recoil_snapshot)
        self._update_profile_action_states()

    def _update_profile_action_states(self):
        if not hasattr(self,"_profile_save_buttons"): return
        count=int(self._recoil_snapshot.get("calibration_count",
                  len(self._recoil_snapshot.get("source_runs") or [])))
        ready=bool(self._recoil_snapshot.get("ready",count>=5)) and count>=5
        for button in self._profile_save_buttons:
            button.configure(state="normal" if ready else "disabled")
        selected=self.config.get("manual_recoil",{}).get("profile_name","")
        self._delete_profile_button.configure(
            state="disabled" if self.running or not selected else "normal")

    def _profile_snapshot_for_save(self):
        snapshot=dict(self._recoil_snapshot or {})
        count=int(snapshot.get("calibration_count",
                  len(snapshot.get("source_runs") or [])))
        if not snapshot.get("ready",count>=5) or count<5:
            raise ValueError("需要先完成5次有效手动记录")
        return snapshot

    def _save_named_profile(self):
        try:
            name=validate_profile_name(self._profile_name_var.get())
            snapshot=self._profile_snapshot_for_save()
            if self._profile_store.exists(name) and not messagebox.askyesno(
                    "覆盖曲线",f"曲线“{name}”已经存在，是否覆盖？"):
                return
            saved=self._profile_store.save(name,snapshot)
        except (ValueError,ProfileStoreError) as exc:
            messagebox.showerror("无法保存曲线",str(exc)); return
        saved.update({"calibration_count":5,"ready":True,
                      "loaded_profile_name":name,"profile_name":name})
        self._recoil_snapshot=saved
        update_recoil_config(
            self.config,self._manual_recoil_enabled_var.get(),
            self._scales["manual_recoil_blend"].get(),
            self._scales["manual_recoil_tail"].get(),
            vertical_enabled=self._vertical_recoil_enabled_var.get())
        self.config.setdefault("manual_recoil",{})["profile_name"]=name
        save_config(self.config)
        self._refresh_profile_selector(select_name=name)
        self._update_recoil_preview(saved)
        self._append_log(f"[曲线配置] 已原子保存“{name}”，下次启动直接 READY")
        self._status_var.set("曲线已保存")

    def _save_current_profile(self):
        selected=str(self.config.get("manual_recoil",{}).get("profile_name","") or "")
        if selected:
            self._profile_name_var.set(selected)
        self._save_named_profile()

    def _save_profile_as(self):
        self._save_named_profile()

    def _delete_profile(self):
        name=str(self.config.get("manual_recoil",{}).get("profile_name","") or "")
        if not name: return
        if not messagebox.askyesno("删除曲线",f"确定删除曲线“{name}”？\n此操作无法撤销。"):
            return
        try:
            self._profile_store.delete(name)
        except ProfileStoreError as exc:
            messagebox.showerror("删除失败",str(exc)); return
        self._set_default_profile(persist=True)
        self._refresh_profile_selector(select_name="")
        self._append_log(f"[曲线配置] 已删除“{name}”，已切换默认重新校准")

    def _receive_recoil_state(self,snapshot):
        self.root.after(0,self._apply_recoil_state_snapshot,dict(snapshot or {}))

    def _apply_recoil_state_snapshot(self,snapshot):
        event=snapshot.get("event","")
        if event=="reset":
            self.config.setdefault("manual_recoil",{})["profile_name"]=""
            save_config(self.config)
            self._profile_select_var.set(DEFAULT_PROFILE_LABEL)
            self._profile_name_var.set("")
        elif event=="reset_failed":
            messagebox.showwarning(
                "无法重新校准",
                f"Raw Input 未能就绪，当前已保存曲线仍然有效。\n\n{snapshot.get('message','')}")
        elif event=="loaded" and snapshot.get("profile_name"):
            self._profile_name_var.set(snapshot["profile_name"])
        self._update_recoil_preview(snapshot)

    # ── UI 启停 ──
    def _set_inputs(self,enabled):
        """启用或禁用所有输入控件。Combobox/Entry/Scale/Checkbutton 状态统一处理。"""
        for w in self._all_inputs:
            try:
                if isinstance(w, ttk.Combobox):
                    w.configure(state="readonly" if enabled else "disabled")
                elif isinstance(w, ttk.Checkbutton):
                    w.configure(state="normal" if enabled else "disabled")
                elif isinstance(w, ttk.Scale):
                    w.configure(state="normal" if enabled else "disabled")
                elif isinstance(w, ttk.Entry):
                    w.configure(state="normal" if enabled else "disabled")
            except Exception:
                pass
        if not enabled:
            for w in self._runtime_inputs:
                try:
                    if isinstance(w,ttk.Combobox): w.configure(state="readonly")
                    else: w.configure(state="normal")
                except Exception: pass
        self._update_profile_action_states()

    # ── 日志 ──
    def _append_log(self,msg,run_id=None):
        """Thread-safe enqueue only; Tk is touched by _flush_gui_log."""
        visible_message = (
            msg.get("message", "") if isinstance(msg, dict) else msg)
        if (not self._debug_logging_enabled and
                not is_minimal_log_message(visible_message)):
            return False
        if isinstance(msg,dict):
            item=dict(msg)
            run_id=item.get("run_id",run_id)
            item["run_id"]=run_id
        else:
            item={"run_id":run_id,"message":str(msg)}
        # A stopped/changed run is never allowed to refill the new run's UI.
        if run_id is not None and str(run_id) != str(self._active_run_id):
            return False
        return self._gui_log_pump.submit(item)

    def _flush_gui_log(self):
        """Batch GUI log insertion with a row/time budget and bounded history."""
        started=time.perf_counter()
        batch=self._gui_log_pump.drain(max_items=80,max_ms=4.0,max_chars=12000)
        if batch:
            render_started=time.perf_counter()
            self._log_text.configure(state="normal")
            text="".join(str(item.get("message",item))+"\n" for item in batch)
            self._log_text.insert("end",text)
            self._gui_log_line_count+=len(batch)
            overflow=self._gui_log_line_count-self._gui_log_max_lines
            if overflow>0:
                self._log_text.delete("1.0",f"{overflow+1}.0")
                self._gui_log_line_count-=overflow
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
            self._gui_log_pump.record_display(
                batch, (time.perf_counter()-render_started)*1000.0)
        # The budget covers the complete Tk callback, including insert/delete/
        # see.  ``drain`` timing alone is not a rendering measurement.
        self._last_gui_refresh_ms=(time.perf_counter()-started)*1000.0
        if self._gui_log_after_id is not None:
            self._gui_log_after_id=self.root.after(50,self._flush_gui_log)

    def _clear_log_display(self):
        """Clear both queued and already-rendered GUI log rows."""
        self._gui_log_pump.clear()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._gui_log_line_count=0

    def _do_log(self,msg):
        if not self._debug_logging_enabled:
            return
        self._log_text.configure(state="normal"); self._log_text.insert("end",msg+"\n")
        self._gui_log_line_count+=1
        overflow=self._gui_log_line_count-self._gui_log_max_lines
        if overflow>0:
            self._log_text.delete("1.0",f"{overflow+1}.0")
            self._gui_log_line_count-=overflow
        self._log_text.see("end"); self._log_text.configure(state="disabled")

    def _reset_recoil_calibration(self):
        if self.running:
            import main as engine
            engine.request_recoil_reset()
            self._append_log("[GUI] 正在确认 Raw Input；成功后才会清空当前曲线")
            self._status_var.set("准备重新校准…")
        else:
            self._set_default_profile(persist=True)
            self._refresh_profile_selector(select_name="")
            self._append_log("[GUI] 已切换默认；下次启动从0/5重新采集，磁盘曲线未删除")

    # ── 加载 & 保存 ──
    def _load_ui(self):
        c=self.config; self._model_var.set(c.get("model",DEFAULT["model"]))
        for k,v in self._hotkey_vars.items(): v.set(c["hotkeys"].get(k,DEFAULT["hotkeys"][k]))
        # 检测参数
        for k in("conf","capture_size","chest_ratio","aim_interval_ms"):
            if k in self._scales: self._scales[k].set(c.get(k,DEFAULT.get(k,0)))
        u=c.get("unit",{}) or {}
        for ck,sk,dv in (("frame_ms","u_frame_ms",75.0),
                         ("px_per_count","u_kx",0.3011),("px_per_count_y","u_ky",0.2684),
                         ("move_steps","u_steps",2)):
            if sk in self._scales: self._scales[sk].set(u.get(ck,dv))
        rem=u.get("remaining_error",{}) or {}
        for ck,sk,dv in (("tau_ms","u_tau_ms",40.0),
                         ("max_speed_counts_s","u_max_speed_counts_s",4000.0),
                         ("max_accel_counts_s2","u_max_accel_counts_s2",0.0),
                         ("feedback_delay_ms","u_feedback_delay_ms",20.0),
                         ("max_observation_age_ms","u_max_observation_age_ms",120.0)):
            if sk in self._scales: self._scales[sk].set(rem.get(ck,dv))
        self._frame_schedule_var.set(
            "跟随推理速度" if str(u.get("frame_schedule", "fixed")).lower() == "asap"
            else "固定周期")
        self._unit_control_var.set(
            "异步剩余误差" if str(u.get("control_strategy", "rate_hold")).lower()
            == "remaining_error" else "现有速率保持")
        recoil=c.get("recoil",{}) or {}
        if "u_output_period_ms" in self._scales:
            self._scales["u_output_period_ms"].set(
                recoil.get("output_period_ms",DEFAULT["recoil"]["output_period_ms"]))
        # aim_control 参数
        ac=c.get("aim_control",{})
        for k in("smoothing","target_ema","ema_max_step","max_total_gain","reversal_damp","send_every_n_frames","delay_comp_ms","move_digest_ms","boost_lo_px","boost_hi_px","boost_max"):
            if k in self._scales: self._scales[k].set(ac.get(k,DEFAULT["aim_control"].get(k,0)))
        _pm = str(ac.get(
            "prediction_mode",
            "arrival" if ac.get("unit_prediction_enabled", True) else "current",
        )).lower()
        self._prediction_mode_var.set(
            "到达时刻预测" if _pm == "arrival" else "当前跟踪点（无前导）")
        self._lock_filter_mode_var.set(
            "固定(手动alpha)" if str(ac.get(
                "lock_box_filter_mode", "fixed")).lower() == "fixed"
            else "自适应(0.35～0.85)")
        self._scales["lock_box_smoothing_alpha"].set(
            ac.get("lock_box_smoothing_alpha", 0.20))
        self._aim_mode_var.set("满额纠错(固定节拍)" if str(c.get("aim_mode", "unit")) == "unit" else "平滑(旧架构)")
        # 日志
        self._mouse_log_var.set(c.get("mouse_log_enabled",DEFAULT["mouse_log_enabled"]))
        self._debug_logging_enabled=bool(self._mouse_log_var.get())
        rt=c.get("reaction_test",{}) or {}
        self._rt_en_var.set(bool(rt.get("enabled",False)))
        if "rt_min_size" in self._scales: self._scales["rt_min_size"].set(rt.get("min_size",80))
        if "rt_interval_ms" in self._scales: self._scales["rt_interval_ms"].set(rt.get("interval_ms",75))
        at=c.get("auto_trigger",{}) or {}
        self._at_en_var.set(bool(at.get("enabled",False)))
        if "at_min_size" in self._scales: self._scales["at_min_size"].set(at.get("min_size",80))
        if "at_conf" in self._scales: self._scales["at_conf"].set(at.get("conf",0.65))
        if "at_interval_ms" in self._scales: self._scales["at_interval_ms"].set(at.get("interval_ms",75))
        if "at_grace_ms" in self._scales: self._scales["at_grace_ms"].set(at.get("grace_ms",300))
        # 类别
        target_cls=c.get("target_classes",DEFAULT["target_classes"])
        for i in range(NUM_CLASSES): self._class_vars[i].set(i in target_cls)
        self._target_priority_var.set(
            "优先头框" if str(c.get("target_priority", "body")).lower() == "head"
            else "优先身框")
        # 其他
        self._precision_var.set(c.get("precision",DEFAULT["precision"]))
        model_name=c.get("model",DEFAULT["model"])
        size=self._extract_model_size(model_name); self._model_size_var.set(size)
        self._crosshair_x_var.set(c.get("crosshair_offset_x",0.0))
        self._crosshair_y_var.set(c.get("crosshair_offset_y",0.0))
        # 瞄准参考点模式（兼容旧 crosshair_auto 布尔键）
        _cm = c.get("crosshair_mode")
        if _cm is None:
            _cm = "dot_fallback" if c.get("crosshair_auto", False) else "center"
        self._crosshair_mode_var.set(
            {"dot_only": "只认红点", "dot_fallback": "允许红点退化",
             "center": "只认屏幕中心"}.get(_cm, "允许红点退化"))
        if "crosshair_search_radius" in self._scales:
            self._scales["crosshair_search_radius"].set(c.get("crosshair_search_radius",150))
        mr=c.get("manual_recoil",{}) or {}
        default_recoil=c.get("default_recoil",{}) or {}
        vertical_enabled=bool(default_recoil.get("enabled",False))
        self._vertical_recoil_enabled_var.set(vertical_enabled)
        self._manual_recoil_enabled_var.set(
            bool(mr.get("enabled",True)) and not vertical_enabled)
        self._manual_recoil_suppress_prediction_var.set(bool(
            mr.get("suppress_prediction_during_replay",True)))
        selected=str(mr.get("profile_name","") or "")
        self._refresh_profile_selector(select_name=selected)
        if "manual_recoil_blend" in self._scales:
            self._scales["manual_recoil_blend"].set(
                float(mr.get("playback_blend_percent",70.0)))
        if "manual_recoil_tail" in self._scales:
            self._scales["manual_recoil_tail"].set(
                float(mr.get("tail_vertical_ratio",0.30)) * 100.0)
        if "default_recoil_strength" in self._scales:
            self._scales["default_recoil_strength"].set(
                float(default_recoil.get("strength",4.0)))
        self._update_recoil_preview(self._recoil_snapshot)

    def _extract_model_size(self,model_name):
        match=re.search(r'_(\d{3,4})(?:_optimized)?\.onnx$',model_name)
        return match.group(1) if match else "320"

    def _on_model_size_change(self,event=None):
        size=self._model_size_var.get()
        current=self._model_var.get()
        new_model=re.sub(r'_\d{3,4}((?:_optimized)?\.onnx)$',f'_{size}\\1',current)
        if new_model in self._model_combo["values"]:
            self._model_var.set(new_model)

    def _ui_to_config(self):
        c=self.config
        c["model"]=self._model_var.get()
        for k in self._hotkey_vars: c["hotkeys"][k]=self._hotkey_vars[k].get()
        # 检测
        for k in("conf","capture_size","chest_ratio","aim_interval_ms"):
            if k in self._scales:
                v=self._scales[k].get()
                c[k]=int(v) if k in("capture_size","aim_interval_ms") else v
        # aim_control
        ac=c.setdefault("aim_control",{})
        for k in("smoothing","target_ema","ema_max_step","max_total_gain","reversal_damp","send_every_n_frames","delay_comp_ms","move_digest_ms","boost_lo_px","boost_hi_px","boost_max"):
            if k in self._scales: ac[k]=self._scales[k].get()
        ac["prediction_mode"] = (
            "arrival" if self._prediction_mode_var.get() == "到达时刻预测"
            else "current")
        ac["unit_prediction_enabled"] = ac["prediction_mode"] == "arrival"
        ac["lock_box_filter_mode"] = (
            "fixed" if self._lock_filter_mode_var.get().startswith("固定")
            else "adaptive")
        ac["lock_box_smoothing_alpha"] = round(
            self._scales["lock_box_smoothing_alpha"].get(), 2)
        ac["lock_box_alpha_min"] = 0.35
        ac["lock_box_alpha_max"] = 0.85
        ac["lock_box_speed_low"] = 80.0
        ac["lock_box_speed_high"] = 600.0
        c["aim_mode"] = "unit" if self._aim_mode_var.get() == "满额纠错(固定节拍)" else "smooth"
        u=c.setdefault("unit",{})
        u["frame_schedule"]=(
            "asap" if self._frame_schedule_var.get() == "跟随推理速度" else "fixed")
        u["control_strategy"] = (
            "remaining_error" if self._unit_control_var.get() == "异步剩余误差"
            else "rate_hold")
        u["frame_ms"]=float(self._scales["u_frame_ms"].get())
        # reference_frame_ms 是控制器标称整定周期，与 fixed 的识别节拍解耦。
        # GUI 当前不单独暴露它，保存时保留加载值；只有缺失时才按旧配置规则回退。
        u.setdefault("reference_frame_ms", float(u.get("frame_ms", 22.0)))
        u["px_per_count"]=round(self._scales["u_kx"].get(),4)
        u["px_per_count_y"]=round(self._scales["u_ky"].get(),4)
        # 每轴只有一个物理标定值。unit 用于控制换算，aim_control 用于
        # 自身视角运动补偿；二者必须从同一个 GUI 标定值同步。
        ac["view_scale"]=u["px_per_count"]
        ac["view_scale_y"]=u["px_per_count_y"]
        u["move_steps"]=int(self._scales["u_steps"].get())
        rem=u.setdefault("remaining_error",{})
        for ck,sk in (("tau_ms","u_tau_ms"),
                      ("max_speed_counts_s","u_max_speed_counts_s"),
                      ("max_accel_counts_s2","u_max_accel_counts_s2"),
                      ("feedback_delay_ms","u_feedback_delay_ms"),
                      ("max_observation_age_ms","u_max_observation_age_ms")):
            rem[ck]=float(self._scales[sk].get())
        u["prediction_mode"]=ac["prediction_mode"]
        recoil=c.setdefault("recoil",{})
        recoil["output_period_ms"]=float(
            self._scales["u_output_period_ms"].get())
        c.setdefault("default_recoil",{})["strength"] = float(
            self._scales["default_recoil_strength"].get())
        previous_debug_logging=self._debug_logging_enabled
        c["mouse_log_enabled"]=self._mouse_log_var.get()
        self._debug_logging_enabled=bool(c["mouse_log_enabled"])
        if previous_debug_logging and not self._debug_logging_enabled:
            self._clear_log_display()
        c["reaction_test"]={"enabled":self._rt_en_var.get(),
                            "min_size":round(self._scales["rt_min_size"].get(),1),
                            "interval_ms":int(self._scales["rt_interval_ms"].get()),
                            "size":640,"window_s":1.0,"lost_s":0.3}
        c["auto_trigger"]={"enabled":self._at_en_var.get(),
                           "min_size":round(self._scales["at_min_size"].get(),1),
                           "conf":round(self._scales["at_conf"].get(),3),
                           "interval_ms":int(self._scales["at_interval_ms"].get()),
                           "grace_ms":int(self._scales["at_grace_ms"].get()),
                           "size":640}
        # 其他
        c["precision"]=self._precision_var.get()
        c["crosshair_offset_x"]=self._crosshair_x_var.get()
        c["crosshair_offset_y"]=self._crosshair_y_var.get()
        c["crosshair_mode"]={"只认红点":"dot_only","允许红点退化":"dot_fallback",
                              "只认屏幕中心":"center"}.get(self._crosshair_mode_var.get(),"dot_fallback")
        c.pop("crosshair_auto",None)  # 旧布尔键废弃
        if "crosshair_search_radius" in self._scales:
            c["crosshair_search_radius"]=int(self._scales["crosshair_search_radius"].get())
        update_recoil_config(
            c,self._manual_recoil_enabled_var.get(),
            self._scales["manual_recoil_blend"].get(),
            self._scales["manual_recoil_tail"].get(),
            vertical_enabled=self._vertical_recoil_enabled_var.get())
        c.setdefault("manual_recoil",{})["suppress_prediction_during_replay"] = bool(
            self._manual_recoil_suppress_prediction_var.get())
        c["target_classes"]=[i for i in range(NUM_CLASSES) if self._class_vars[i].get()]
        c["target_priority"]=(
            "head" if self._target_priority_var.get() == "优先头框" else "body")
        return c

    def _refresh_models(self):
        dirs=[BASE_DIR]
        if getattr(sys,"frozen",False): dirs.insert(0,sys._MEIPASS)
        names=discover_onnx_models(dirs)
        if not names: names=[DEFAULT["model"]]
        self._model_combo["values"]=names
        if self._model_var.get() not in names: self._model_var.set(names[0])

    # ── 热键轮询 & 退出键切换 ──
    def _hk_poll_loop(self):
        # 常驻轮询：退出键要"停止后还能再启动"，不能因 generation 变化自杀。
        if self._poller:
            self._poller.tick()
            qvk = getattr(self._poller, "_quit_vk", None)
            if qvk is not None:
                state = user32.GetAsyncKeyState(qvk)
                pressed = bool(state & 0x8000)
                if pressed != self._quit_prev:
                    self._quit_prev = pressed
                    if pressed:
                        if self.running:
                            self._append_log("[热键] 退出键按下 → 停止引擎")
                            self._do_stop()
                        else:
                            self._append_log("[热键] 退出键按下 → 启动引擎")
                            self.root.after(50, self._start)
        self._hk_after_id = self.root.after(20, self._hk_poll_loop)

    # ── 按钮切换 ──
    def _toggle(self):
        if self.running: self._do_stop()
        else: self._start()

    def _save(self):
        hotkeys=self._validated_ui_hotkeys()
        if hotkeys is None: return
        self._ui_to_config(); self.config["hotkeys"]=hotkeys; save_config(self.config)
        self._status_var.set("已保存"); self._append_log("[GUI] 配置已保存")

    def _validated_ui_hotkeys(self):
        try:
            return normalize_hotkeys(
                {key:var.get() for key,var in self._hotkey_vars.items()}, _VK)
        except ValueError as exc:
            messagebox.showerror("热键配置无效", str(exc))
            self._status_var.set("热键冲突/无效")
            self._append_log(f"[热键] 拒绝保存：{exc}")
            return None

    # ── 停止 ──
    def _do_stop(self):
        """非阻塞停止：设标志位后立即返回，UI 重置由 _poll_engine_done 异步完成。"""
        if not self.running and not (self._runner_thread is not None and
                                     self._runner_thread.is_alive()):
            return
        self._status_var.set("停止中…"); self._append_log("[GUI] 正在停止引擎…")
        self._header_metrics.set("引擎：停止中    配置：保持    输入：释放")
        self.running = False; self._running_loop = False
        self._hk_generation += 1
        if self.stop_event:
            self.stop_event.set()
        # Do not replay a backlog of verbose details after stop.  The run file
        # sink remains responsible for the retained file events.
        self._gui_log_pump.clear()
        if self._poller:
            self._poller.stop()
            try:
                with self._poller._q.mutex: self._poller._q.queue.clear()
            except Exception: pass
            self._poller._prev.clear()
        # 异步等待引擎线程退出后重置 UI
        self._stop_gen = self._hk_generation
        self._stop_started = time.time()
        self.root.after(50, self._poll_engine_done)

    def _poll_engine_done(self):
        """轮询真实线程状态；超时只报告收尾中，不伪造已退出。"""
        # 如果在等待期间被新的启动覆盖（generation 变了），直接退
        if self._hk_generation != getattr(self, "_stop_gen", -1):
            return
        thread=self._runner_thread
        if thread is not None and thread.is_alive():
            elapsed=time.time()-getattr(self,"_stop_started",0)
            if elapsed < 3.0:
                self.root.after(50,self._poll_engine_done)
            else:
                self._status_var.set("停止超时：旧运行仍在收尾")
                self._header_metrics.set("引擎：旧运行仍在收尾    输入：已停发")
                self.root.after(250,self._poll_engine_done)
            return
        self._runner_thread=None
        self._reset_ui_stopped()

    def _reset_ui_stopped(self, message=None, terminal_messages=()):
        run_id=getattr(self,"_active_run_id",None)
        self._gui_log_pump.clear()
        if run_id is not None:
            if (run_id == self._engine_ready_run_id and
                    run_id != self._engine_closed_tone_run_id):
                announce_engine_status(False)
                self._engine_closed_tone_run_id=run_id
            self._append_log("[引擎] 已退出",run_id=run_id)
            for terminal in terminal_messages:
                self._append_log(terminal,run_id=run_id)
        self.running=False; self._running_loop=False
        self._active_run_id=None
        self._btn_var.set("启动引擎")
        self._status_var.set(message or "已停止")
        selected=self.config.get("manual_recoil",{}).get("profile_name","") or "默认"
        self._header_metrics.set(f"引擎：已停止    配置：{selected}    输入：等待")
        self._set_inputs(True)

    def _on_engine_ready(self, run_id, stop_event):
        """接收引擎线程的 ready 边沿；不在工作线程操作 Tk 控件。"""
        if (stop_event.is_set() or run_id != self._active_run_id or
                not self.running or run_id == self._engine_ready_run_id):
            return
        self._engine_ready_run_id=run_id
        announce_engine_status(True)

    # ── 启动 ──
    def _start(self):
        if self._runner_thread is not None and self._runner_thread.is_alive():
            self._status_var.set("旧运行仍在收尾…")
            self._append_log("[GUI] 旧控制线程尚未退出，拒绝重叠启动")
            return
        hotkeys=self._validated_ui_hotkeys()
        if hotkeys is None: return
        self._ui_to_config(); self.config["hotkeys"]=hotkeys; save_config(self.config)
        self._hk_generation+=1
        gen=self._hk_generation
        run_id=new_run_id()
        run_config=copy.deepcopy(self.config)
        run_stop_event=threading.Event()
        self.stop_event=run_stop_event
        self._active_run_id=run_id
        self._engine_ready_run_id=None
        self._engine_closed_tone_run_id=None
        self._gui_log_pump.clear()
        self.running=True; self._running_loop=True
        self._set_inputs(False); self._btn_var.set("■  停  止")
        self._status_var.set("运行中…")
        selected=self.config.get("manual_recoil",{}).get("profile_name","") or "默认0/5"
        self._header_metrics.set(f"引擎：运行中    配置：{selected}    输入：监听")
        self._append_log("="*40); self._append_log("[引擎] 正在启动…")
        hotkeys_cfg=hotkeys
        # 启动日志：显示当前绑定的所有热键，便于排查冲突
        self._append_log(f"[配置] {os.path.abspath(CONFIG_PATH)}")
        self._append_log(f"[热键] 瞄准={hotkeys_cfg.get('aim','?')}  压枪={hotkeys_cfg.get('recoil','?')}  "
                         f"截图={hotkeys_cfg.get('screenshot','?')}  退出={hotkeys_cfg.get('quit','?')}")
        if self._poller: self._poller.stop()
        self._poller=HotkeyPoller(hotkeys_cfg)
        self._poller.set_gui_callback(
            lambda event, rid=run_id: self._append_log(event, run_id=rid))
        self._poller.start()
        self._append_log("[热键] 轮询器已启动")
        self._append_log(f"[截图] 固定屏幕中心 {self.config.get('capture_size',640)}x{self.config.get('capture_size',640)}px")
        # 注：热键轮询循环常驻（__init__ 已启动），这里不再重复调度，避免双循环
        poller=self._poller
        def _runner(local_config=run_config,local_stop=run_stop_event,
                    local_poller=poller,local_run_id=run_id):
            error=None
            try:
                import main as engine
                engine.set_gui_callback(None)
                engine.set_recoil_state_callback(self._receive_recoil_state)
                engine._EXTERNAL_HK=local_poller
                engine.set_gui_generation(gen)
                engine.run_engine(local_config,local_stop,run_id=local_run_id,
                                  gui_submit=lambda event:self._append_log(
                                      event,run_id=local_run_id),
                                  ready_callback=lambda rid=local_run_id,
                                      ev=local_stop:self._on_engine_ready(rid,ev))
            except Exception as ex:
                error=(ex,traceback.format_exc())
            try:
                self.root.after(0,lambda rid=local_run_id,err=error:
                                self._on_engine_done(rid,err))
            except Exception:
                pass
        self._runner_thread=threading.Thread(target=_runner,
                                             name="Engine-%s"%run_id,daemon=True)
        self._runner_thread.start()

    def _on_engine_done(self,run_id,error=None):
        """Only the matching run may update the UI state."""
        if run_id!=getattr(self,"_active_run_id",None):
            return
        if error:
            ex,tb=error
            # Reset clears stale details, while terminal messages are appended
            # before the old run identity is detached so they remain visible.
            self._reset_ui_stopped(
                f"异常: {ex}",
                terminal_messages=(f"错误: {ex}", tb),
            )
        elif self.stop_event is None or self.stop_event.is_set():
            self._reset_ui_stopped()
        else:
            self._reset_ui_stopped()

    def _on_close(self):
        if self.running:
            self._do_stop()
        if self._poller: self._poller.stop()
        if self._hk_after_id is not None:
            try: self.root.after_cancel(self._hk_after_id)
            except Exception: pass
        if self._gui_log_after_id is not None:
            try: self.root.after_cancel(self._gui_log_after_id)
            except Exception: pass
        self._gui_log_pump.clear()
        try:
            import main as engine
            engine.set_recoil_state_callback(None)
        except Exception: pass
        self.root.destroy()

    def run(self): self.root.mainloop()

if __name__=="__main__": AimGUI().run()
