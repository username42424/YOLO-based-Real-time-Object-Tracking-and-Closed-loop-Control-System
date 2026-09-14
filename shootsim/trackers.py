# -*- coding: utf-8 -*-
"""追踪策略（模块化，便于替换与调优）。

输入：感知到的目标框（屏幕坐标 [x1,y1,x2,y2]，可能为 None = 看不到目标）。
输出：虚拟鼠标位移 (dx, dy)（px），直接作用于相机，与真实 main.py 的
"误差 → 鼠标移动"管道等价（sens=1 时 1px 鼠标 = 1px 世界位移）。

所有策略共享基础参数：gain(比例增益)、cap(单帧限幅)、deadzone(死区)、
aim_ratio(瞄准点高度比: 0=框顶, 0.166≈头部中心, 0.42≈胸口, 0.5=中心, 1=框底)。
"""
import math


class BaseTracker:
    name = "base"

    def __init__(self, cfg, sw, sh, rng=None):
        self.gain = float(cfg.get("gain", 0.35))
        self.cap = float(cfg.get("cap", 60.0))
        self.deadzone = float(cfg.get("deadzone", 6.0))
        self.aim_ratio = float(cfg.get("aim_ratio", 0.166))
        self.cx = sw / 2.0
        self.cy = sh / 2.0
        self.reset()

    def reset(self):
        pass

    def status(self):
        """公共控制状态；旧 Tracker 默认明确报告为未知/未锁定。"""
        return {
            "has_target": False,
            "locked": False,
            "waiting": False,
            "target_type": "NONE",
        }

    def set_reference(self, rx, ry):
        """设置瞄准参考点（红点/子弹落点，而非屏幕中心）。

        误差 = 目标瞄准点 - 参考点；命中判定也用红点。红点相对屏幕中心上方
        偏移约 red_dot_offset_px（默认 15px）+ 半径 red_dot_jitter_px 抖动。
        """
        self.cx = rx
        self.cy = ry

    def aim_point(self, box):
        """瞄准点：框水平中心 + aim_ratio 高度。"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y1 + (y2 - y1) * self.aim_ratio)

    def _p(self, err_x, err_y):
        """比例控制 + 限幅 + 死区。"""
        if abs(err_x) < self.deadzone and abs(err_y) < self.deadzone:
            return 0.0, 0.0
        dx, dy = err_x * self.gain, err_y * self.gain
        m = math.hypot(dx, dy)
        if m > self.cap:
            s = self.cap / m
            dx *= s
            dy *= s
        return dx, dy

    def update(self, box, dt):
        return 0.0, 0.0


class PTracker(BaseTracker):
    """纯比例控制：移动 = clamp(误差×gain, cap)，死区内不动。"""
    name = "p"

    def update(self, box, dt):
        if box is None:
            return 0.0, 0.0
        ax, ay = self.aim_point(box)
        return self._p(ax - self.cx, ay - self.cy)


class PIDTracker(BaseTracker):
    """PID 控制：+积分(消除稳态偏差，限幅防饱和) +微分(阻尼)。"""
    name = "pid"

    def __init__(self, cfg, sw, sh, rng=None):
        super().__init__(cfg, sw, sh, rng)
        self.ki = float(cfg.get("ki", 0.0))
        self.kd = float(cfg.get("kd", 0.0))
        self.reset()

    def reset(self):
        self.ix = self.iy = 0.0
        self.last_ex = self.last_ey = None

    def update(self, box, dt):
        if box is None:
            self.ix = self.iy = 0.0
            self.last_ex = self.last_ey = None
            return 0.0, 0.0
        ax, ay = self.aim_point(box)
        ex, ey = ax - self.cx, ay - self.cy
        if abs(ex) < self.deadzone and abs(ey) < self.deadzone:
            return 0.0, 0.0
        # 积分（限幅 ±cap，防饱和）
        self.ix = max(-self.cap, min(self.cap, self.ix + ex * dt))
        self.iy = max(-self.cap, min(self.cap, self.iy + ey * dt))
        # 微分
        dex = (ex - self.last_ex) / dt if self.last_ex is not None else 0.0
        dey = (ey - self.last_ey) / dt if self.last_ey is not None else 0.0
        self.last_ex, self.last_ey = ex, ey
        dx = self.gain * (ex + self.ki * self.ix + self.kd * dex)
        dy = self.gain * (ey + self.ki * self.iy + self.kd * dey)
        m = math.hypot(dx, dy)
        if m > self.cap:
            s = self.cap / m
            dx *= s
            dy *= s
        return dx, dy


class EmaTracker(BaseTracker):
    """EMA 平滑瞄准点 + 比例控制（对应旧版 main.py 的 smooth 模式）。"""
    name = "ema"

    def __init__(self, cfg, sw, sh, rng=None):
        super().__init__(cfg, sw, sh, rng)
        self.alpha = float(cfg.get("ema_alpha", 0.26))
        self.reset()

    def reset(self):
        self.ema = None

    def update(self, box, dt):
        if box is None:
            self.ema = None
            return 0.0, 0.0
        ax, ay = self.aim_point(box)
        if self.ema is None or math.hypot(ax - self.ema[0], ay - self.ema[1]) > 120.0:
            self.ema = (ax, ay)  # 目标跳变/换目标 → 重锁，不爬行
        else:
            a = max(0.0, min(1.0, self.alpha))
            self.ema = (self.ema[0] * a + ax * (1 - a), self.ema[1] * a + ay * (1 - a))
        return self._p(self.ema[0] - self.cx, self.ema[1] - self.cy)


class PredictTracker(BaseTracker):
    """速度外推预测 + 比例控制：用 EMA 估计目标屏幕速度，
    瞄准点 = 当前点 + 速度 × lead_frames 帧（补偿回路延迟）。"""
    name = "predict"

    def __init__(self, cfg, sw, sh, rng=None):
        super().__init__(cfg, sw, sh, rng)
        self.lead = float(cfg.get("lead_frames", 0))
        self.vel_alpha = float(cfg.get("vel_ema", 0.6))
        self.reset()

    def reset(self):
        self.vel = None
        self.last_pt = None

    def update(self, box, dt):
        if box is None:
            self.vel = None
            self.last_pt = None
            return 0.0, 0.0
        ax, ay = self.aim_point(box)
        if self.last_pt is not None:
            vx, vy = (ax - self.last_pt[0]) / dt, (ay - self.last_pt[1]) / dt
            if self.vel is None:
                self.vel = (vx, vy)
            else:
                self.vel = (self.vel[0] * self.vel_alpha + vx * (1 - self.vel_alpha),
                            self.vel[1] * self.vel_alpha + vy * (1 - self.vel_alpha))
        self.last_pt = (ax, ay)
        px, py = ax, ay
        if self.vel is not None:
            px += self.vel[0] * self.lead * dt
            py += self.vel[1] * self.lead * dt
        return self._p(px - self.cx, py - self.cy)


# ============================================================
# main 策略：完整复用根目录 main.py 的瞄准管线 + 参数
# ============================================================
# 这是 shootsim 作为"main 调试台"的核心：
#   追踪/瞄准逻辑、死区、限幅、灵敏度、人手仿真、分步发送、换向阻尼
#   全部来自 main.py 的平滑模式；参数从根目录 config.json 读取。
#   aim_move 的 send_fn 重定向到模拟环境 apply_mouse，实现"调参→直接对应真实 main"。
import importlib.util
import os
import math as _math
from other_test_tracker import OtherTestTracker
from obs_auto_aim_tracker import ObsAutoAimTracker


def _load_aim_main():
    """加载根目录 main.py（复用 MainEngine / aim_move 等; 按 sys.modules 缓存）。"""
    import sys
    cached = sys.modules.get("aim_main_root")
    if cached is not None:
        return cached
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    main_path = os.path.join(parent, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"找不到根目录 main.py: {main_path}")
    spec = importlib.util.spec_from_file_location("aim_main_root", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aim_main_root"] = mod
    spec.loader.exec_module(mod)
    return mod


class MainTracker(BaseTracker):
    """复刻 main.py 平滑瞄准管线，参数来自根目录 config.json。

    send_fn(dx, dy)：把分步后的位移交给它发送（模拟器注入 env.apply_mouse；
    真实环境留空则用 main 的 SendInput）。返回 (net_dx, net_dy)。
    """
    name = "main"

    def __init__(self, cfg, sw, sh, rng=None, send_fn=None, main_cfg_path=None):
        self.scfg = cfg  # shootsim tracker 配置（lost 搜索参数），须在 super().reset() 前设置
        super().__init__(cfg, sw, sh, rng)
        self.send_fn = send_fn
        # 加载根目录 main.py 模块与参数
        self.aim = _load_aim_main()
        here = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(here)
        main_cfg_path = main_cfg_path or os.path.join(parent, "config.json")
        import json
        with open(main_cfg_path, "r", encoding="utf-8") as f:
            self.mcfg = json.load(f)
        # 从 main config 提取 humanize / aim_control 参数
        self.hu = dict(self.aim.DEFAULT["humanize"])
        self.hu.update(self.mcfg.get("humanize", {}))
        ac = self.mcfg.get("aim_control", {})
        self.hu["smoothing"] = float(ac.get("smoothing", self.hu.get("smoothing", 0.5)))
        self.target_ema_alpha = float(ac.get("target_ema", 0.26))
        self.box_scale_enabled = bool(self.mcfg.get("box_scale_gain_enabled", False))
        bsg = self.mcfg.get("box_scale_gain", {}) or {}
        self.bsg_min_size = float(bsg.get("min_size", 30))
        self.bsg_max_size = float(bsg.get("max_size", 320))
        self.bsg_min_gain = float(bsg.get("min_gain", 1.15))
        self.bsg_max_gain = float(bsg.get("max_gain", 0.6))
        # EMA 跳变限速 + 有效总增益钳制（复刻 main.py 实战修正）
        self.ema_max_step = float(ac.get("ema_max_step", 60.0))       # 0=关闭(旧snap)
        self.max_total_gain = float(ac.get("max_total_gain", 0.0))    # 0=关闭
        # 速度前导（复刻 main.py）：连续检出建立速度模型，外推 lead 帧，
        # 抵消回路延迟与纯比例控制的稳态滞后。0=关闭。
        self.lead_frames = max(0.0, float(ac.get("lead_frames", 0.0)))
        self.vel_alpha = 0.45
        self.vel_confirm = 3
        self.lead_max = 45.0
        # 速度死区（Change3）：死区 = max(基础, 框高×比例)，大框放宽、小框保持基础值
        self.vel_deadzone_base = max(0.0, float(ac.get("vel_deadzone", 2.0)))
        self.vel_deadzone_frac = max(0.0, float(ac.get("vel_deadzone_frac", 0.05)))
        # 延迟补偿（复刻 main.py）：扣除在途未生效发送，防重复发送过冲。
        # 模拟器回路延迟 ≈ 观测延迟 1 帧(16.7ms)，窗口取 min(主配置, 观测延迟+2ms)
        # 避免按真实机器 42ms 过度扣除。
        _comp_ms = float(ac.get("delay_comp_ms", 0.0))
        if _comp_ms > 0:
            _obs_ms = int(self.scfg.get("obs_delay_frames", 1)) * (1000.0 / 60.0)
            self.delay_comp_ms = min(_comp_ms, _obs_ms + 2.0)
        else:
            self.delay_comp_ms = 0.0
        self.inflight = []   # [(t, dx, dy)] 在途未生效发送
        self.reversal_damp = 0.6
        # 屏幕px/鼠标px 换算系数：复刻 main.py 的 view_scale。模拟器正交投影且
        # view.sensitivity=1 → 1px 鼠标 = 1px 屏幕位移 → view_scale=1.0。
        # ⚠️ 不能读根目录 config.json 的 view_scale(0.3011)——那是真实游戏的
        # 360° 标定值，模拟器用了会欠抵消自身移动 ~3.3 倍。sim 的 view_scale
        # 应等于 sim 自身 view.sensitivity（改 sens 时同步改这里）。
        self.view_scale = float(self.scfg.get("view_scale", 1.0))
        # 目标选择参数（镜像 main.py find_target）：
        #   lock_dist_tol：同目标判定距离（main 的 target_lock_distance）
        #   body_grace：身框丢失多少帧后释放身框保持（main 的 lock_grace_frames）
        self.lock_dist_tol = float(self.mcfg.get("target_lock_distance", 100.0))
        self.body_grace = max(2, int(float(self.mcfg.get("lock_grace_frames", 30.0))))
        # 发送节流（ms），复刻 main 的 step_min_gap/基础间隔
        self.step_min_gap_ms = float(self.mcfg.get("step_min_gap_ms", 18.0))
        self.step_base_ms = float(self.mcfg.get("mouse_step_ms", 24.0))
        # 首步大位移上限（复刻 main.py max_first_step），0=关闭
        self.first_cap = max(0.0, float(self.mcfg.get("max_first_step", 0.0)))
        # mouse_step_ms=0=自动：复刻 main.py 的"实测回路延迟"基础间隔
        # （端到端~14ms + EMA滞后13ms + 相机响应15ms ≈ 42ms）。
        # 实测实战：固定30ms+收缩到20ms 低于回路延迟 → 同一误差发两次过冲。
        if self.step_base_ms <= 0:
            self.step_base_ms = 42.0
        # 约束：最小发送间隔不能大于基础间隔（否则 gap 会反向/异常）
        if self.step_min_gap_ms > self.step_base_ms:
            self.step_min_gap_ms = self.step_base_ms
        # 同步 main 全局位移上限，确保模拟与真实 main 一致
        self.aim._MAX_TARGET_STEP = max(0.0, float(self.mcfg.get("max_target_step", 80.0)))
        self.reset()

    def reset(self):
        self.ema = None
        self.ema_init = False
        self.last_net = [0.0, 0.0]
        self.last_send_t = None
        self.t = 0.0
        # 速度模型（前导预测）
        self.vel = [0.0, 0.0]
        self.vel_valid = False
        self.vel_frames = 0
        self.prev_target = None
        self.prev_cls = None   # 上帧类别（Change2：类别切换即重置速度模型）
        # base 类首轮 reset() 早于 MainTracker 配置注入，用 getattr 兜底默认值
        self.vel_deadzone = getattr(self, "vel_deadzone_base", 2.0)
        # 延迟补偿在途发送
        self.inflight = []
        self.comp_active = False   # 补偿启用段（阈值+迟滞+时限，复刻 main.py）
        self.comp_on = 50.0        # 未补偿误差超过该值启用
        self.comp_off = 30.0       # 低于该值退出
        self.comp_frames = 0       # 本次启用持续帧数
        self.comp_cooldown = False # 超时强制退出后的冷却
        self.comp_max_frames = 8   # 单次启用最长帧数（追击时限，防增益减半）
        # 自身视角速度 EMA（前导摄像头耦合修正）
        self.own_vel = [0.0, 0.0]
        self.frame_net = [0.0, 0.0]
        # 首步大位移：锁定/大跳变后的第一次发送放宽上限
        self.first_send_pending = True
        # 身框保持状态（镜像 main.py find_target 的 locked_box/locked_cls）
        self.locked_body = None   # 最近一次身框 [x1,y1,x2,y2]
        self.body_miss = 0        # 连续无身框帧数
        # lost 搜索状态
        self.lost_count = 0          # 连续无目标帧数
        self.searching = False       # 是否在随机搜索
        self.search_dir = [0.0, 0.0]  # 当前搜索方向
        self.search_timer = 0.0       # 搜索方向保持计时
        self.search_hold_frames = 0
        # 消失容忍帧数：目标消失(10帧)内保持等待，超过才搜索
        self.lost_search_delay = int(self.scfg.get("lost_search_delay", 14))
        self.search_interval_s = float(self.scfg.get("search_interval_s", 0.12))
        self.search_speed = float(self.scfg.get("search_speed", 120.0))

    def aim_point(self, box, cls=0):
        """复刻 MainEngine._aim_point：头框和身框使用各自的高度比例。"""
        x1, y1, x2, y2 = box
        chest = float(self.mcfg.get("chest_ratio", 0.10))
        if cls == 1:
            head = float(self.mcfg.get("head_ratio", 0.70))
            return ((x1 + x2) / 2.0, y1 + (y2 - y1) * head)
        return ((x1 + x2) / 2.0, y1 + (y2 - y1) * chest)

    def _box_gain(self, box):
        if not self.box_scale_enabled:
            return 1.0
        x1, y1, x2, y2 = box
        size = max(y2 - y1, x2 - x1)
        size = max(self.bsg_min_size, min(self.bsg_max_size, size))
        t = (size - self.bsg_min_size) / max(1.0, self.bsg_max_size - self.bsg_min_size)
        return self.bsg_min_gain + (self.bsg_max_gain - self.bsg_min_gain) * t

    def update(self, box, dt):
        """每帧：EMA → 死区 → smoothing×框增益 → 换向阻尼 → aim_move(send_fn)。

        box 可为单个框 [x1,y1,x2,y2] 或框列表（多目标）。空/None（目标丢失）：
         - 短暂丢失(≤lost_search_delay帧，含目标消失10帧) → 保持冻结等待重现。
         - 长时间丢失(>lost_search_delay帧，目标跑出yolo视野) → 随机搜索移动，
           直到重新检测到目标。
        """
        self.t += dt
        # 自身视角速度 EMA（px/帧）：前导摄像头耦合修正
        self.own_vel[0] = self.own_vel[0] * self.vel_alpha + self.frame_net[0] * (1.0 - self.vel_alpha)
        self.own_vel[1] = self.own_vel[1] * self.vel_alpha + self.frame_net[1] * (1.0 - self.vel_alpha)
        self.frame_net = [0.0, 0.0]
        # ── 目标选择（镜像 main.py find_target）──
        # 感知框为 [x1,y1,x2,y2,cls]（gt 模式为 4 元素，视作身框 cls=0）。
        # 规则：
        #   1. 身框优先：有身框就锁最近身框（能锁身子就锁身子）
        #   2. 身框保持(body_hold)：身框闪烁丢失、只剩头框且头框离锁定身框
        #      < lock_dist_tol 时，继续用上次身框瞄准（镜像 main.py 的头/身框横跳修复）
        if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
            boxes = [list(b) + [0] if len(b) < 5 else list(b) for b in box]
        elif box is not None:
            boxes = [list(box) + [0] if len(box) < 5 else list(box)]
        else:
            boxes = []
        if boxes:
            def _nearest(bs):
                return min(bs, key=lambda x: ((x[0] + x[2]) / 2.0 - self.cx) ** 2
                           + ((x[1] + x[3]) / 2.0 - self.cy) ** 2)
            bodies = [b for b in boxes if b[4] != 1]
            heads = [b for b in boxes if b[4] == 1]
            if bodies:
                # 身框优先：锁最近身框（对齐真实 main.py find_target）
                b = _nearest(bodies)
                self.locked_body = list(b[:4])
                self.body_miss = 0
            elif heads and self.locked_body is not None:
                h = _nearest(heads)
                lcx = (self.locked_body[0] + self.locked_body[2]) / 2.0
                lcy = (self.locked_body[1] + self.locked_body[3]) / 2.0
                hcx = (h[0] + h[2]) / 2.0
                hcy = (h[1] + h[3]) / 2.0
                if _math.hypot(hcx - lcx, hcy - lcy) < self.lock_dist_tol:
                    # 身框保持：只剩头框且与锁定身框同目标 → 继续用身框
                    b = list(self.locked_body) + [0]
                else:
                    b = list(h)
                self.body_miss += 1
            elif heads:
                b = list(_nearest(heads))
                self.body_miss += 1
            else:
                b = None
            if self.body_miss >= self.body_grace:
                self.locked_body = None
                self.body_miss = 0
            if b is not None:
                box = b[:4]
                box_cls = b[4]
            else:
                box = None
                box_cls = 0
        else:
            box = None
            box_cls = 0

        if box is None or (isinstance(box, (list, tuple)) and not box):
            # 无目标：进入丢失计数
            self.lost_count += 1
            # 速度模型作废（断档期间目标可能任意移动）
            self.vel_valid = False
            self.vel_frames = 0
            self.vel = [0.0, 0.0]
            self.prev_target = None
            self.prev_cls = None
            # 不重置 EMA/ema_init（复刻 main.py 实战修正）：重新检测到目标时走
            # ema_max_step 限速分帧逼近，而不是"首次锁定"全量 snap——
            # 重置 EMA 等于每次重检出都送一次满幅甩动（实战乱晃主因）。
            if self.lost_count > self.lost_search_delay:
                # 长时间丢失 → 随机搜索移动
                return self._lost_search(dt)
            return 0.0, 0.0

        # 检测到目标：重置丢失/搜索状态
        self.lost_count = 0
        self.searching = False
        ax, ay = self.aim_point(box, box_cls)
        # EMA 更新（复刻 main.py：首次直接采用；跳变超过 ema_max_step 限速分帧逼近，
        # 不再 snap 全量重锁——snap 会把检测框横跳放大成满幅甩动；否则轻量平滑）
        if not self.ema_init:
            self.ema = (ax, ay)
            self.ema_init = True
            self.first_send_pending = True
        else:
            _edx = ax - self.ema[0]
            _edy = ay - self.ema[1]
            _ed = _math.hypot(_edx, _edy)
            if self.ema_max_step > 0 and _ed > self.ema_max_step:
                self.ema = (self.ema[0] + _edx / _ed * self.ema_max_step,
                            self.ema[1] + _edy / _ed * self.ema_max_step)
                self.first_send_pending = True  # 大跳变≈切换目标，重给首步额度
            else:
                a = max(0.0, min(1.0, self.target_ema_alpha))
                self.ema = (self.ema[0] * a + ax * (1 - a), self.ema[1] * a + ay * (1 - a))

        # 速度模型更新（复刻 main.py 前导，含摄像头耦合修正与死区）：
        # 目标速度 = 框速度 + 自身视角速度；静态目标两者抵消 → 前导=0。
        # Change1: 速度锚点用「框中心」而非「瞄点」——瞄点 y = 框顶+chest×框高，且
        # 身/头类别瞄不同相对位置；大框 under 检测抖动被 chest×Δh 当成假速度灌进前导。
        # 框中心对 chest 与类别切换免疫，速度估计更稳。
        bcx = (box[0] + box[2]) / 2.0
        bcy = (box[1] + box[3]) / 2.0
        bh = box[3] - box[1]
        # Change2: 类别切换（身↔头/换目标）帧，框垂直位跳变，速度模型不可信 → 立即重置
        if self.prev_cls is not None and box_cls != self.prev_cls:
            self.vel_valid = False
            self.vel_frames = 0
            self.vel = [0.0, 0.0]
        self.prev_cls = box_cls
        # Change3: 速度死区随框高放大——大框检测抖动大，固定死区挡不住
        vel_dz = max(self.vel_deadzone, bh * self.vel_deadzone_frac)
        if self.prev_target is not None:
            vx = (bcx - self.prev_target[0]) + self.own_vel[0] * self.view_scale
            vy = (bcy - self.prev_target[1]) + self.own_vel[1] * self.view_scale
            if _math.hypot(vx, vy) < vel_dz:
                vx = vy = 0.0
            if _math.hypot(vx, vy) <= max(self.ema_max_step, 1.0):
                if self.vel_valid:
                    self.vel[0] += (vx - self.vel[0]) * (1.0 - self.vel_alpha)
                    self.vel[1] += (vy - self.vel[1]) * (1.0 - self.vel_alpha)
                else:
                    self.vel_frames += 1
                    self.vel[0] += (vx - self.vel[0]) / self.vel_frames
                    self.vel[1] += (vy - self.vel[1]) / self.vel_frames
                    if self.vel_frames >= self.vel_confirm:
                        self.vel_valid = True
            else:
                self.vel_valid = False
                self.vel_frames = 0
                self.vel = [0.0, 0.0]
        self.prev_target = (bcx, bcy)

        # 误差 = EMA + 速度前导 - 准星 - 在途未生效发送（前导量限幅）
        lx = self.vel[0] * self.lead_frames if (self.vel_valid and self.lead_frames > 0) else 0.0
        ly = self.vel[1] * self.lead_frames if (self.vel_valid and self.lead_frames > 0) else 0.0
        ld = _math.hypot(lx, ly)
        if ld > self.lead_max:
            s = self.lead_max / ld
            lx *= s
            ly *= s
        raw_x = self.ema[0] + lx - self.cx
        raw_y = self.ema[1] + ly - self.cy
        comp_x = 0.0
        comp_y = 0.0
        # 延迟补偿（复刻 main.py 阈值门控+时限版）：仅大误差追赶期启用，补偿量≤误差
        if self.delay_comp_ms > 0 and self.inflight:
            mag = _math.hypot(raw_x, raw_y)
            if self.comp_cooldown:
                if mag < self.comp_off:
                    self.comp_cooldown = False
            elif self.comp_active:
                if mag < self.comp_off:
                    self.comp_active = False
                    self.comp_frames = 0
                else:
                    self.comp_frames += 1
                    if self.comp_frames > self.comp_max_frames:
                        self.comp_active = False
                        self.comp_frames = 0
                        self.comp_cooldown = True
            elif mag > self.comp_on:
                self.comp_active = True
                self.comp_frames = 1
            if self.comp_active:
                self.inflight = [e for e in self.inflight if e[0] >= self.t]
                for _t, ix, iy in self.inflight:
                    # 在途发送是鼠标px，误差是屏幕px：×view_scale 换算（sim=1.0 等效直减）
                    comp_x += ix * self.view_scale
                    comp_y += iy * self.view_scale
                cmag = _math.hypot(comp_x, comp_y)
                if cmag > mag > 0:
                    s = mag / cmag
                    comp_x *= s
                    comp_y *= s
        raw_dx = raw_x - comp_x
        raw_dy = raw_y - comp_y
        # 死区：基于原始误差
        dz_base = float(self.hu.get("deadzone", 3.0))
        box_h = max(0.0, min(box[3] - box[1], 640.0))
        deadzone_px = dz_base * (0.5 + box_h / 640.0)
        if abs(raw_dx) < deadzone_px and abs(raw_dy) < deadzone_px:
            return 0.0, 0.0

        # smoothing × 框增益
        smoothing = float(self.hu.get("smoothing", 0.35))
        gain = self._box_gain(box)
        dx = raw_dx * smoothing * gain
        dy = raw_dy * smoothing * gain

        # 有效总增益钳制（复刻 main.py）：smoothing×框增益×sensitivity 总乘数
        # 超过上限时同比例缩小，保证实际位移 ≤ 误差×上限，不自激振荡。
        _total_gain = smoothing * gain * float(self.hu.get("sensitivity", 1.0))
        if self.max_total_gain > 0 and _total_gain > self.max_total_gain:
            _g = self.max_total_gain / _total_gain
            dx *= _g
            dy *= _g

        # 换向阻尼（复刻 main：与上次净移动反向时打折）
        if (self.last_net[0] > 0 and dx < 0) or (self.last_net[0] < 0 and dx > 0):
            dx *= self.reversal_damp
        if (self.last_net[1] > 0 and dy < 0) or (self.last_net[1] < 0 and dy > 0):
            dy *= self.reversal_damp

        # 发送节流（ms）
        cur_err = _math.hypot(raw_dx, raw_dy)
        span = max(1.0, self.step_base_ms * 2.5)
        frac = min(1.0, max(0.0, cur_err / span))
        gap_ms = self.step_base_ms - (self.step_base_ms - self.step_min_gap_ms) * frac
        now_ms = self.t * 1000.0
        if self.last_send_t is not None and (now_ms - self.last_send_t) < gap_ms:
            return 0.0, 0.0
        self.last_send_t = now_ms

        # 调用 main.aim_move，send_fn 重定向到模拟器
        hu_move = dict(self.hu)
        hu_move["deadzone"] = 0.0  # 死区已在上层判定
        _cap = self.first_cap if (self.first_send_pending and self.first_cap > 0) else None
        self.first_send_pending = False
        result = self.aim.aim_move(dx, dy, hu_move, send_fn=self.send_fn, cap=_cap)
        if result["sent"]:
            self.last_net = [result.get("net_dx", 0), result.get("net_dy", 0)]
            self.frame_net = [result.get("net_dx", 0), result.get("net_dy", 0)]
            if self.delay_comp_ms > 0:
                ndx = result.get("net_dx", 0)
                ndy = result.get("net_dy", 0)
                wm = max(self.delay_comp_ms, _math.hypot(ndx, ndy))
                self.inflight.append((self.t + wm / 1000.0, ndx, ndy))
        # 位移已通过 send_fn(env.apply_mouse) 应用到模拟环境，
        # 返回 (0,0) 避免 run_sim 再用返回值叠加导致位移翻倍。
        return 0.0, 0.0

    def _lost_search(self, dt):
        """目标长时间丢失（跑出 yolo 视野）→ 快速扫描搜索。

        搜索速度必须 ≥ 目标速度（目标 240px/s，这里用更高搜索速度），
        且每次朝一个方向多走一段再换向，避免来回摆动找不到目标。
        搜索位移直接通过 send_fn 应用，返回 (0,0) 避免 run_sim 双重应用。
        """
        import random as _r
        self.searching = True
        self.search_timer += dt
        if self.search_timer >= self.search_interval_s or self.search_hold_frames == 0:
            # 换方向：随机角度，但倾向横向扫描（覆盖更大屏幕区域）
            ang = _r.uniform(0, 2 * _math.pi)
            # 搜索速度 = max(目标速度, 240)，确保追得上
            spd = max(self.search_speed, 240.0)
            self.search_dir = [_math.cos(ang) * spd,
                               _math.sin(ang) * spd]
            self.search_timer = 0.0
        self.search_hold_frames += 1
        # 每帧移动 search_speed*dt px
        dx = self.search_dir[0] * dt
        dy = self.search_dir[1] * dt
        if self.send_fn is not None:
            self.send_fn(dx, dy)
        return 0.0, 0.0


class MainEngineTracker:
    """策略 main_real:直接跑 main.py 的真实瞄准核心(MainEngine)+ main.aim_move。

    感知框 [x1,y1,x2,y2,(cls),(conf)] 为模拟器屏幕坐标 → 换算成 main.py 的
    320 截图坐标(减 (sw-capture)/2 偏移)后喂引擎; 红点参考同样换算。
    这样引擎内部的目标锁定/瞄准点全部与实战同坐标系,
    [鼠标] 日志可与实战日志逐字段对比。
    view_scale 取自 tracker 配置(=模拟器 view.sensitivity, 实机标定 0.3011/0.2684),
    保证 notify_net 速度耦合(自身移动 counts→px)的单位与模拟器视角物理一致。
    mouse_log 注入时, 每次实际发送按 main.py 的 [鼠标] 行格式写日志。
    """

    def __init__(self, tracker_cfg, sw, sh, rng=None, send_fn=None, main_cfg_path=None,
                 mouse_log=None, command_hook=None):
        import json as _json
        _mod = _load_aim_main()
        MainEngine = _mod.MainEngine
        aim_move = _mod.aim_move
        self._ObservationFreshness = _mod.ObservationFreshness
        self._bounded_rate_integration_dt = _mod.bounded_rate_integration_dt
        self.aim_move = aim_move
        self.motion_arbiter = _mod.MotionArbiter()
        # main_cfg_path 可传 dict(参数扫描用, 免写临时文件)或配置文件路径
        if isinstance(main_cfg_path, dict):
            cfg = main_cfg_path
        else:
            if main_cfg_path:
                with open(main_cfg_path, encoding="utf-8") as cfg_file:
                    cfg = _json.load(cfg_file)
            else:
                cfg = {}
        self.engine = MainEngine(cfg)
        # 视角换算与模拟器一致(速度耦合用; 延迟补偿关闭时仅影响 lead 速度模型)
        self.engine.view_scale = float(tracker_cfg.get("view_scale", 1.0))
        self.engine.view_scale_y = float(tracker_cfg.get("view_scale_y",
                                                          self.engine.view_scale))
        self.send_fn = send_fn
        self.command_hook = command_hook
        # 钩子兼容: 接受 (dx,dy) 或 (dx,dy,out) 两种签名; 三参钩子可拿到引擎
        # compute 结果(含 target/dx/dy), 用于统计丢失目标后的错误移动与过冲。
        self._hook_wants_out = False
        if command_hook is not None:
            try:
                import inspect as _inspect
                self._hook_wants_out = len(
                    _inspect.signature(command_hook).parameters.values()) >= 3 and not any(
                    p.kind == _inspect.Parameter.VAR_POSITIONAL
                    for p in _inspect.signature(command_hook).parameters.values())
            except (TypeError, ValueError):
                self._hook_wants_out = False
        self.sw, self.sh = sw, sh
        self._cap = int((cfg or {}).get("capture_size", 320))
        self._off = sw / 2.0 - self._cap / 2.0   # 屏幕坐标 → 截图坐标 偏移
        # 实机 main.py 经 set_max_target_step(根级 max_target_step) 设置发送 counts 上限;
        # 模拟器不走该全局量, 在此等价传入 aim_move 的 cap(默认上限), cap_hint(首步)优先。
        self._max_step = float((cfg or {}).get("max_target_step", 80.0))
        # unit(满额纠错)模式的保真转换参数:
        # - 发送与 main.py aim_frame 的 unit 分支一致(sens=1、无死区/人手仿真, 仅大位移上限)
        self._unit_cap = float((cfg.get("aim_control", {}) or {}).get("unit_max_counts", 800.0))
        self.unit_mode = getattr(self.engine, "aim_mode", "") == "unit"
        # - 固定节拍(与 main.py 新架构一致): 每 unit.frame_ms(默认75ms)识别一次,
        #   每张识别图只移动一次; 节拍 > 游戏输入→画面延迟, 天然串行, 无需在途验证。
        #   frame_schedule=asap(跟随推理速度): 无 frame_ms 人工等待, 观测节奏由
        #   回放日志真实时间戳或 tracker 配置的 asap_e2e_ms 端到端序列决定;
        #   输出语义切换为"最新控制速率(counts/s)×真实 tick dt 积分"。
        unit_cfg = cfg.get("unit", {}) or {}
        self._frame_schedule = str(unit_cfg.get("frame_schedule", "fixed")).lower()
        if self._frame_schedule not in ("fixed", "asap"):
            self._frame_schedule = "fixed"
        self._reference_frame_s = max(
            0.002, float(unit_cfg.get("reference_frame_ms",
                                       unit_cfg.get("frame_ms", 22.0)))) / 1000.0
        self._frame_ms = max(0.02, float(unit_cfg.get("frame_ms", 75.0))) / 1000.0
        self._unit_steps = max(1, int(unit_cfg.get("move_steps", 3)))
        self._unit_profile = "ease" if str(unit_cfg.get("step_profile", "ease")) == "ease" else "linear"
        # asap 观测间隔(端到端时间序列, ms): None=跟随 env 步长; 数值=固定间隔;
        # 列表=按回合循环使用的可变间隔序列(不允许把 asap 简化成单一固定值时使用)。
        self._asap_e2e_ms = tracker_cfg.get("asap_e2e_ms")
        self._asap_e2e_idx = 0
        recoil_cfg = cfg.get("recoil", {}) or {}
        self._output_period = max(
            0.010, float(recoil_cfg.get("output_period_ms", 30.0)) / 1000.0)
        self._processing_latency = max(
            0.0, float(tracker_cfg.get("processing_latency_ms", 12.0)) / 1000.0)
        self._t = 0.0
        self._last_control_t = -1e9
        self._last_observation_t = None
        self._next_output_t = self._output_period
        self._last_out = None
        self._last_center = None
        self._time_synced_for_update = False
        # replay fidelity hooks: per-frame real processing latency from the log
        # (None → fall back to tracker_cfg constant) and the global 15ms output
        # grid phase fitted from the log's 输出t stamps
        self._live_latency = None
        # replay mode: the env already steps at the live capture cadence, so
        # control must fire on every update instead of re-imposing frame_ms
        self._control_every_update = bool(tracker_cfg.get("control_every_update", False))
        self._last_output_t = None
        self.active_plan = None
        self.pending_plan = None
        self.pending_available_at = None
        self._active_plan_published_at = None
        self._freshness = self._ObservationFreshness(self._reference_frame_s)
        self.ttl_clear_count = 0
        self.tick_stall_count = 0
        self.discarded_dt_s = 0.0
        self.max_actual_tick_dt_s = 0.0
        self._asap_e2e_idx = 0
        self.ref = [sw / 2.0, sh / 2.0]   # 瞄准基准(默认十字中心; run_sim 注入红点)
        self.dt = 1.0 / 120.0
        self.mouse_log = mouse_log
        self._frame = 0

    def set_live_latency(self, latency_s):
        self._live_latency = (
            max(0.0, float(latency_s)) if latency_s is not None else None)

    def _asap_gate_s(self):
        """asap 模式下一次观测前应等待的端到端时长(秒); None=不设闸门。

        tracker.asap_e2e_ms: None → 跟随 env 步长; 数值 → 固定端到端间隔;
        列表 → 按观测循环使用的可变端到端序列(避免把 asap 简化成单一固定值)。
        """
        if self._asap_e2e_ms is None:
            return None
        if isinstance(self._asap_e2e_ms, (list, tuple)):
            if not self._asap_e2e_ms:
                return None
            ms = float(self._asap_e2e_ms[self._asap_e2e_idx % len(self._asap_e2e_ms)])
        else:
            ms = float(self._asap_e2e_ms)
        return max(0.001, ms) / 1000.0

    def set_output_phase(self, phase_s):
        """Align the 15ms output grid to the live log's global tick grid.

        phase_s: episode-relative time of the first tick due after t=0
        (already reduced into [0, period))."""
        period = self._output_period
        phase = float(phase_s) % period
        self._next_output_t = phase if phase > 1e-9 else period
        self._t = 0.0
        self._last_control_t = -1e9
        self._last_observation_t = None
        self.motion_arbiter.reset()
        self.active_plan = None
        self.pending_plan = None
        self.pending_available_at = None
        self._active_plan_published_at = None
        self._freshness.reset()

    def reset(self):
        self.engine.reset()
        self.motion_arbiter.reset()
        self._frame = 0
        self._t = 0.0
        self._last_control_t = -1e9
        self._last_observation_t = None
        self._next_output_t = self._output_period
        self._last_out = None
        self._last_center = None
        self._time_synced_for_update = False
        self._last_output_t = None
        self.active_plan = None
        self.pending_plan = None
        self.pending_available_at = None
        self._active_plan_published_at = None
        self._freshness.reset()
        self.ttl_clear_count = 0
        self.tick_stall_count = 0
        self.discarded_dt_s = 0.0
        self.max_actual_tick_dt_s = 0.0
        self._asap_e2e_idx = 0

    def set_reference(self, rx, ry):
        """镜像基类 PTracker.set_reference:把瞄准基准设为红点(子弹落点,含后坐力),
        使引擎与镜像用同一参考,子弹才能落在目标上而不被后坐力顶偏。"""
        self.ref[0] = rx
        self.ref[1] = ry

    def advance_to(self, absolute_time):
        """Advance the independent output clock before the next capture."""
        target_time = max(self._t, float(absolute_time))
        self._t = target_time
        if self.unit_mode:
            self._drain_unit_output()
        self._time_synced_for_update = True

    def _submit_unit_plan(self, plan, available_at, out=None, center=None):
        """Queue a compute result; it becomes active only at available_at."""
        self.pending_plan = plan
        self.pending_available_at = float(available_at)
        if out is not None:
            self._pending_out = dict(out)
        if center is not None:
            self._pending_center = tuple(center)

    def _activate_pending_plan(self, tick_t):
        if (self.pending_plan is None or self.pending_available_at is None or
                tick_t + 1e-9 < self.pending_available_at):
            return False
        plan = self.pending_plan
        if plan[0] == "rate":
            self.motion_arbiter.publish_aim_rate(*plan[1])
            self._active_plan_published_at = tick_t
        elif plan[0] == "chunks":
            self.motion_arbiter.publish_aim(*plan[1])
            self._active_plan_published_at = tick_t
        else:
            self.motion_arbiter.clear_aim()
            self._active_plan_published_at = None
        self.active_plan = plan
        self.pending_plan = None
        self.pending_available_at = None
        if hasattr(self, "_pending_out"):
            self._last_out = self._pending_out
        if hasattr(self, "_pending_center"):
            self._last_center = self._pending_center
        return True

    def _drain_unit_output(self):
        while self._next_output_t <= self._t + 1e-9:
            tick_t = self._next_output_t
            # 输出线程按真实 tick dt 积分(asap 速率语义), 而非假设固定周期。
            if self._last_output_t is None:
                actual_tick_dt = self._output_period
            else:
                actual_tick_dt = max(0.0, tick_t - self._last_output_t)
            self._last_output_t = tick_t
            self.max_actual_tick_dt_s = max(
                self.max_actual_tick_dt_s, actual_tick_dt)
            tick_dt, discarded_dt, stalled = self._bounded_rate_integration_dt(
                actual_tick_dt, self._output_period)
            if stalled:
                self.tick_stall_count += 1
                self.discarded_dt_s += discarded_dt
            self._activate_pending_plan(tick_t)
            if (self._frame_schedule == "asap" and
                    self._active_plan_published_at is not None and
                    tick_t - self._active_plan_published_at > self._freshness.ttl_s):
                self.motion_arbiter.clear_aim()
                self.active_plan = ("clear", None)
                self._active_plan_published_at = None
                self.ttl_clear_count += 1
            if self._frame_schedule == "asap":
                aim_dx, aim_dy = self.motion_arbiter.next_aim_rate_motion(tick_dt)
            else:
                aim_dx, aim_dy = self.motion_arbiter.next_aim_chunk()
            send_x, send_y = self.motion_arbiter.quantize_components(
                aim_dx, aim_dy, 0.0, 0.0)
            if send_x or send_y:
                try:
                    if self.send_fn is not None:
                        self.send_fn(send_x, send_y)
                    self.motion_arbiter.mark_send_succeeded()
                    if self.command_hook is not None:
                        if self._hook_wants_out:
                            self.command_hook(send_x, send_y, self._last_out)
                        else:
                            self.command_hook(send_x, send_y)
                    if self.mouse_log is not None and self._last_out is not None:
                        self._mouse_line(
                            self._last_out,
                            {"sent": True, "net_dx": send_x, "net_dy": send_y,
                             "steps": 1},
                            self._last_center[0], self._last_center[1],
                        )
                except Exception:
                    self.motion_arbiter.mark_send_failed(send_x, send_y)
            else:
                self.motion_arbiter.mark_send_succeeded()
            self._next_output_t += self._output_period

    def update(self, box, dt):
        self.dt = dt if dt and dt > 0 else self.dt
        self._frame += 1
        if self._time_synced_for_update:
            self._time_synced_for_update = False
        else:
            self._t += self.dt
            if self.unit_mode:
                self._drain_unit_output()
        # ── 识别帧调度(与 main.py 一致)──
        # fixed: 每 frame_ms 识别+移动一次。replay 模式跳过该闸门: 回放步长已是
        # 实机真实节拍, 再叠加会跳帧。
        # asap: 无 frame_ms 人工等待; 观测节奏 = env 步长(replay=日志真实时间戳)
        # 或 tracker.asap_e2e_ms 指定的端到端序列(ms, 数值或列表)。
        if self.unit_mode and not self._control_every_update:
            if self._frame_schedule == "asap":
                _gate = self._asap_gate_s()
                if _gate is not None and (self._t - self._last_control_t) < _gate:
                    return 0.0, 0.0
                self._asap_e2e_idx += 1
            elif (self._t - self._last_control_t) < self._frame_ms:
                return 0.0, 0.0
        items = []
        if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
            items = box
        elif box is not None:
            items = [box]
        dets = []
        for b in items:
            b = list(b)
            conf = float(b[5]) if len(b) > 5 else 1.0
            cls = int(b[4]) if len(b) > 4 else 0
            dets.append({"cls": cls, "conf": conf,
                         "bbox": [b[0] - self._off, b[1] - self._off,
                                  b[2] - self._off, b[3] - self._off]})
        cx, cy = self.ref[0] - self._off, self.ref[1] - self._off
        observation_dt = (
            self.dt if self._last_observation_t is None
            else max(1e-4, self._t - self._last_observation_t)
        )
        self._last_observation_t = self._t
        if self._frame_schedule == "asap":
            self._freshness.observe(observation_dt)
        if self.unit_mode:
            capture_snapshot = self.motion_arbiter.snapshot_components()
            observed = self.motion_arbiter.drain_components_through(capture_snapshot)
            if observed.total != (0, 0):
                self.engine.notify_net(*observed.total)
        out = self.engine.compute(
            dets, (cx, cy), observation_dt,
            processing_latency_s=(
                self._live_latency if self._live_latency is not None
                else self._processing_latency) if self.unit_mode else 0.0,
        )
        if self.unit_mode:
            self._last_control_t = self._t
        if out["can_send"] and not out["in_deadzone"] and (abs(out["dx"]) > 0.01 or abs(out["dy"]) > 0.01):
            if self.unit_mode:
                udx, udy = out["dx"], out["dy"]
                mag = _math.hypot(udx, udy)
                if mag > self._unit_cap:
                    scale = self._unit_cap / mag
                    udx *= scale
                    udy *= scale
                if self._frame_schedule == "asap":
                    plan = ("rate", (udx / self._reference_frame_s,
                                      udy / self._reference_frame_s))
                else:
                    plan = ("chunks", (udx, udy,
                                        self._unit_steps, self._unit_profile))
                _lat = (self._live_latency if self._live_latency is not None
                        else self._processing_latency)
                self._submit_unit_plan(plan, self._t + _lat, out, (cx, cy))
                res = {"sent": False}
            else:
                hu_move = dict(self.engine.hu)
                hu_move["deadzone"] = 0.0
                res = (self.aim_move(out["dx"], out["dy"], hu_move,
                                     cap=(out["cap_hint"] if out.get("cap_hint") else self._max_step),
                                     send_fn=self.send_fn)
                       if self.send_fn is not None else {"sent": False})
                if res.get("sent") and self.command_hook is not None:
                    if self._hook_wants_out:
                        self.command_hook(res.get("net_dx", 0.0), res.get("net_dy", 0.0), out)
                    else:
                        self.command_hook(res.get("net_dx", 0.0), res.get("net_dy", 0.0))
            if res.get("sent") and not self.unit_mode:
                self.engine.notify_net(res.get("net_dx", 0.0), res.get("net_dy", 0.0))
                if self.mouse_log is not None:
                    self._mouse_line(out, res, cx, cy)
        elif self.unit_mode:
            _lat = (self._live_latency if self._live_latency is not None
                    else self._processing_latency)
            self._submit_unit_plan(("clear", None), self._t + _lat,
                                   out, (cx, cy))
        self.engine.end_frame()
        # 位移已通过 send_fn(env.apply_mouse) 应用到模拟环境，
        # 返回 (0,0) 避免 run_sim 再用返回值叠加导致位移翻倍。
        return 0.0, 0.0

    def _mouse_line(self, out, res, cx, cy):
        """按 main.py run_engine 的 [鼠标] 行逐字段复刻(坐标系=320截图坐标)。

        帧=策略帧号 目标=瞄准点 EMA=平滑目标 准星=参考点(红点) 误差=EMA+前导-准星
        前导=速度外推 补偿=延迟补偿(恒0) 发送=请求px 增益/sens=引擎诊断
        锁定=锁定框 cls/conf=锁定类别与置信度 净移=实际发送counts 步数=分步数
        """
        try:
            eng = self.engine
            target = out.get("target") or (0.0, 0.0)
            lb = eng.locked_box[0]
            lb_s = ("[" + ",".join(repr(float(v)) for v in lb) + "]") if lb else "无"
            self.mouse_log.line(
                f"[鼠标] 帧={self._frame} 目标=({target[0]:.1f},{target[1]:.1f}) "
                f"EMA=({eng._target_ema[0]:.1f},{eng._target_ema[1]:.1f}) "
                f"准星=({cx:.1f},{cy:.1f}) "
                f"误差=({out['raw_dx']:+.1f},{out['raw_dy']:+.1f}) 前导=({out['lead'][0]:+.1f},{out['lead'][1]:+.1f}) "
                f"补偿=(+0.0,+0.0) "
                f"发送=({out['dx']:+.1f},{out['dy']:+.1f}) "
                f"增益={out['diag_gain']:.2f} sens={out['sens']:.2f} "
                f"锁定={lb_s} cls={eng.locked_cls[0]} conf={eng.locked_conf[0]:.2f} "
                f"净移=({int(res.get('net_dx', 0)):+d},{int(res.get('net_dy', 0)):+d}) "
                f"步数={res.get('steps', 0)}")
        except Exception:
            pass


TRACKERS = {
    "other_test": OtherTestTracker,
    "obs_auto_aim": ObsAutoAimTracker,
    "p": PTracker,
    "pid": PIDTracker,
    "ema": EmaTracker,
    "predict": PredictTracker,
    "main": MainTracker,
    "main_real": MainEngineTracker,
}


def create_tracker(name, tracker_cfg, sw, sh, rng=None, send_fn=None, main_cfg_path=None,
                   mouse_log=None, command_hook=None):
    if name == "main_real":
        return TRACKERS[name](tracker_cfg, sw, sh, rng, send_fn, main_cfg_path,
                              mouse_log, command_hook)
    if name == "main":
        return TRACKERS[name](tracker_cfg, sw, sh, rng, send_fn, main_cfg_path)
    cls = TRACKERS.get(name, PTracker)
    return cls(tracker_cfg, sw, sh, rng)
