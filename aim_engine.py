# -*- coding: utf-8 -*-
"""
aim_engine — main.py 的瞄准"脑"层（MainEngine，headless 可复用）
=============================================================
本模块是从 main.py 拆分出来的全部"决定该往哪移、移多少"的逻辑：

  MainEngine
    find_target(dets)   目标选择：类别过滤 → 身框优先排序
                        → 新锁定/保持锁定/滞留/切换/释放 状态机
    _aim_point(d)       根据框类别计算瞄准点：头框=框中心，身框=chest_ratio 处
    _box_scale_gain(d)  按框大小缩放增益（小框升增益追得快，大框降增益防甩）
    compute(dets, crosshair, dt)
                        每帧瞄准计算，输出应发送的 dx/dy 及节流判定：
                        EMA 目标平滑(限速) → 速度模型+前导(长窗/短窗) →
                        摄像头耦合修正(_own_vel) → 延迟补偿(在途扣除) → 死区 →
                        有效增益钳制 → 换向阻尼(可调) → 首步上限

本类不碰任何 I/O（截图/SendInput/热键/日志/GUI），main.py(真实程序)与
shootsim 模拟器共用同一份代码，保证"模拟里跑的就是 main.py 的逻辑"。

对外接口（供 main.py 与模拟器调用）：
  set_crosshair(cx, cy)  —— 设准星参考点(整帧预先调用)
  reset()                —— 新回合/新会话清零状态
  compute(dets, crosshair, dt) -> dict
                    dets = [{cls, conf, bbox}]  (main.py 检测输出格式)
                    crosshair = (cx, cy) 在截图像素里
                    dt = 距上一帧秒数(据此驱动时间节流)
                    返回 {"can_send","in_deadzone","dx","dy","cap_hint",
                          "raw_dx","raw_dy","deadzone_px","target","diag"}
  notify_net(dx, dy)     —— 回填本帧实际发出的净移(供 own_vel 摄像头耦合)
  end_frame()            —— 用 _frame_net 更新 _own_vel 并清零(对齐 main.py)

常量与参数解析逐字对齐 main.py run_engine。
"""

import math

# ── 模块级常量(对齐 main.py)──
_LEAD_MAX = 45.0
_VEL_ALPHA = 0.45
_VEL_CONFIRM = 3
_COMP_ON = 50.0
_COMP_OFF = 30.0
_COMP_MAX_FRAMES = 8


class MainEngine:
    def __init__(self, config):
        # humanize(人手仿真) —— 偏好从配置读
        hu = dict(config.get("humanize", {}) or {})
        ac = config.get("aim_control", {}) or {}
        hu["smoothing"] = float(ac.get("smoothing", 0.35))
        self.hu = hu
        # EMA / 速度前导 / 增益钳制(aim_control 段)
        self.target_ema_alpha = float(ac.get("target_ema", 0.4))
        self.ema_max_step = max(0.0, float(ac.get("ema_max_step", 60.0)))
        self.max_total_gain = max(0.0, float(ac.get("max_total_gain", 0.95)))
        # 换向阻尼系数(0.5-2.0):与上一帧移动方向相反时,本帧位移乘该系数;
        # <1=反向打折(抑制振荡),>1=反向放大(激进补正)。GUI 在 aim_control.reversal_damp 调整。
        self.reversal_damp = min(2.0, max(0.5, float(ac.get("reversal_damp", 0.6))))
        self.lock_box_smoothing_alpha = min(
            1.0, max(0.05, float(ac.get("lock_box_smoothing_alpha", 0.20))))
        self.lock_box_filter_mode = str(
            ac.get("lock_box_filter_mode", "fixed")).lower()
        if self.lock_box_filter_mode not in ("adaptive", "fixed"):
            self.lock_box_filter_mode = "fixed"
        self.lock_box_alpha_min = min(
            1.0, max(0.05, float(ac.get("lock_box_alpha_min", 0.35))))
        self.lock_box_alpha_max = min(
            1.0, max(self.lock_box_alpha_min,
                     float(ac.get("lock_box_alpha_max", 0.85))))
        self.lock_box_speed_low = max(
            0.0, float(ac.get("lock_box_speed_low", 80.0)))
        self.lock_box_speed_high = max(
            self.lock_box_speed_low + 1.0,
            float(ac.get("lock_box_speed_high", 600.0)))
        # 发送周期(帧)：1=识别与移动 1:1(每帧直发)；2=识别每帧做、移动每 2 帧发一次。
        # 等效于把回路周期翻倍——一次移动被下一次识别完整看到后再决定下一步，
        # 抵消"增益×回路延迟"主导的过冲振荡(对应旧版 mouse_step_period 方案)。
        self.send_every_n_frames = max(1, int(ac.get("send_every_n_frames", 1)))
        # 远距追赶增益(分段P):误差在 boost_lo~boost_hi 之间线性从 1→boost_max。
        # 破"低增益追不上/高增益振荡"的困境:振荡源是零误差附近的抖动换向(那里永远
        # 用基础增益),而追赶只发生在远离目标处(那里用高增益),两者互不干扰。0=关闭。
        self.boost_lo_px = max(0.0, float(ac.get("boost_lo_px", 0.0)))
        self.boost_hi_px = max(0.0, float(ac.get("boost_hi_px", 0.0)))
        self.boost_max = max(1.0, float(ac.get("boost_max", 1.0)))
        self.lead_frames = max(0.0, float(ac.get("lead_frames", 0.0)))
        self.vel_deadzone = max(0.0, float(ac.get("vel_deadzone", 2.0)))
        self.vel_deadzone_frac = max(0.0, float(ac.get("vel_deadzone_frac", 0.05)))
        self.delay_comp_ms = max(0.0, float(ac.get("delay_comp_ms", 0.0)))
        self.view_scale = max(0.001, float(ac.get("view_scale", 1.0)))
        # 垂直轴独立换算(垂直灵敏度可能与水平不同;实测 29025px/360° 时垂直 180°≈13300px,
        # 垂/水比≈1.09,故 view_scale_y 需单独标定;缺省回退水平值保持旧行为)
        self.view_scale_y = max(0.001, float(ac.get("view_scale_y", self.view_scale)))
        # 长窗速度估计(lead 用):窗口时间常数 / 激活噪声地板(px/帧) / 框高缩放系数。
        # 抖动零均值→长窗平均抵消;小幅但持续的漂移能累计成速度 → lead 恢复预测。
        self.vel_lw_t = max(0.02, float(ac.get("vel_lw_t", 0.12)))
        self.vel_lw_floor = max(0.5, float(ac.get("vel_lw_floor", 2.0)))
        self.vel_lw_floor_frac = max(0.0, float(ac.get("vel_lw_floor_frac", 0.03)))
        # unit 模式使用按秒速度和实测处理延迟做前导，不再依赖固定“帧数”。
        self.unit_prediction_enabled = bool(ac.get("unit_prediction_enabled", True))
        self.prediction_mode = str(ac.get(
            "prediction_mode",
            "arrival" if self.unit_prediction_enabled else "current",
        )).lower()
        if self.prediction_mode not in ("current", "arrival"):
            self.prediction_mode = "arrival"
        self.prediction_min_s = max(0.0, float(ac.get("prediction_min_ms", 40.0)) / 1000.0)
        self.prediction_max_s = max(self.prediction_min_s,
                                    float(ac.get("prediction_max_ms", 120.0)) / 1000.0)
        self.prediction_box_ratio = max(0.0, float(ac.get("prediction_box_ratio", 0.75)))
        self.prediction_cap_px = max(0.0, float(ac.get("prediction_cap_px", 45.0)))
        self.prediction_lock_frames = max(2, int(ac.get("prediction_lock_frames", 2)))
        self.prediction_vel_tau = max(0.02, float(ac.get("prediction_vel_tau", 0.08)))
        self.recoil_prediction_box_ratio = max(
            0.0, float(ac.get("recoil_prediction_box_ratio", 0.35)))
        self.recoil_prediction_cap_px = max(
            0.0, float(ac.get("recoil_prediction_cap_px", 20.0)))
        # 目标选择
        self.target_cls = {
            int(cls) for cls in config.get("target_classes", [0, 1])
            if str(cls).lstrip("-").isdigit() and int(cls) in (0, 1)
        }
        # 只影响首次候选排序；已锁定目标仍沿用原有同类连续匹配/切换规则。
        self.target_priority = str(config.get("target_priority", "body")).lower()
        if self.target_priority not in ("body", "head"):
            self.target_priority = "body"
        self.chest = float(config.get("chest_ratio", 0.10))
        self.head_ratio = float(config.get("head_ratio", 0.70))
        self.lock_grace = max(2, int(config.get("lock_grace_frames", 20)))
        self.switch_grace = max(2, int(config.get("switch_grace_frames", 12)))
        self.lock_dist_tolerance = max(1.0, float(config.get("target_lock_distance", 100)))
        self.lock_iou_thresh = min(1.0, max(0.0, float(config.get("target_lock_iou", 0.3))))
        self.lock_zero_iou_max_distance = max(
            0.0, float(config.get("lock_zero_iou_max_distance", 24.0)))
        self.lock_ambiguity_margin = max(0.0, float(config.get("lock_ambiguity_margin", 0.15)))
        self.lock_prediction_max_step = max(
            0.0, float(config.get("lock_prediction_max_step", 25.0)))
        self.capture_size = max(1.0, float(config.get("capture_size", 640)))
        self.edge_new_lock_margin = max(0.0, float(config.get("edge_new_lock_margin", 1.0)))
        self._max_target_step = max(0.0, float(config.get("max_target_step", 80.0)))
        self.first_step_cap = max(0.0, float(config.get("max_first_step", 0.0)))
        # ── 粘滞(近区阻尼): 误差进入"框尺度近区"时按比例衰减位移, 减少"过框来回振荡"。
        #    sticky_gain: 近区中心的位移保留比例(0.55=中心处只发55%位移), 越小越"粘";
        #    sticky_radius_ratio: 近区半径 = 该比例 × 锁定框高(随目标远近自适应)。
        self.sticky_enabled = bool(ac.get("sticky_enabled", False))
        self.sticky_gain = min(1.0, max(0.0, float(ac.get("sticky_gain", 0.55))))
        self.sticky_radius_ratio = max(0.0, float(ac.get("sticky_radius_ratio", 0.55)))
        # ── 缓动起步: 新锁定后前 ease_frames 帧位移增益从 ease_min_gain 线性升到 1.0,
        #    避免刚锁上就满增益甩出(与 max_first_step 互补: 那是硬上限, 这是软坡道)。
        self.ease_frames = max(0, int(ac.get("ease_frames", 0)))
        self.ease_min_gain = min(1.0, max(0.0, float(ac.get("ease_min_gain", 0.35))))
        self._ease_left = 0
        # 框增益
        self.box_scale_enabled = bool(config.get("box_scale_gain_enabled", False))
        bsg = config.get("box_scale_gain", {}) or {}
        self.bsg_min_size = max(1.0, float(bsg.get("min_size", 60)))
        self.bsg_max_size = max(self.bsg_min_size + 1.0, float(bsg.get("max_size", 320)))
        self.bsg_min_gain = max(0.1, float(bsg.get("min_gain", 1.3)))
        self.bsg_max_gain = max(0.1, float(bsg.get("max_gain", 0.6)))
        # 直瞄
        self.aim_mode = str(config.get("aim_mode", "smooth")).lower()
        dm = config.get("direct", {}) or {}
        self.dm_max_delta = max(20.0, float(dm.get("max_delta", 400)))
        self.dm_interval_ms = max(8.0, float(dm.get("interval_ms", 24)))
        self.dm_deadzone = max(0.0, float(dm.get("deadzone", 4)))
        self.dm_sens = max(0.1, float(dm.get("sensitivity", 0.85)))
        self.dm_radius = max(50.0, float(dm.get("radius", 300)))
        self.dm_min_box_h = max(5.0, float(dm.get("min_box_h", 25)))
        # 满额纠错(aim_mode=unit): counts = err(px) / k(px per count)
        # k 初值直接用实机 360° 标定的 view_scale(=0.3011, 纵向 0.2684, 见 AGENTS.md),
        # 在线最小二乘只做开镜FOV/灵敏度变化的微调, 不从裸 1.0 起步猜。
        _u = config.get("unit", {}) or {}
        _u_k0 = max(0.05, min(3.0, float(_u.get("px_per_count", self.view_scale))))
        _u_k0y = max(0.05, min(3.0, float(_u.get("px_per_count_y", self.view_scale_y))))
        self._unit_k = [_u_k0, _u_k0y]      # [kx, ky]
        self._unit_cap = max(20.0, float(ac.get("unit_max_counts", 800.0)))
        _legacy_gain = float(_u.get("gain", 1.0))
        self._unit_far_gain = max(0.05, min(2.5, float(_u.get("far_gain", _legacy_gain))))
        self._unit_near_gain = max(0.05, min(1.0, float(_u.get("near_gain", 0.4))))
        self._unit_near_radius = max(1.0, float(_u.get("near_radius_px", 22.0)))
        self._unit_continuous_gain = bool(_u.get("continuous_gain", False))
        self._unit_near_gain_xy = [
            max(0.05, min(1.0, float(_u.get("near_gain_x", self._unit_near_gain)))),
            max(0.05, min(1.0, float(_u.get("near_gain_y", self._unit_near_gain)))),
        ]
        self._unit_far_gain_xy = [
            max(0.05, min(2.5, float(_u.get("far_gain_x", self._unit_far_gain)))),
            max(0.05, min(2.5, float(_u.get("far_gain_y", self._unit_far_gain)))),
        ]
        self._unit_gain_softness = [
            max(0.1, float(_u.get("gain_softness_px_x", self._unit_near_radius))),
            max(0.1, float(_u.get("gain_softness_px_y", self._unit_near_radius))),
        ]
        self._unit_damped = bool(_u.get("damped", False))  # 默认关=满额一次到位;
        # 开启后叠加平滑系阻尼链(平滑度/框增益/总增益钳/远距增益/粘滞/缓动)作防冲调节
        self._unit_ema = max(0.0, min(0.95, float(_u.get("ema", 0.40))))    # 轻EMA(旧权重)
        self._unit_target_filter_alpha = max(
            0.05, min(1.0, float(_u.get("target_filter_alpha", 0.35))))
        self._unit_target_filter_mode = str(
            _u.get("target_filter_mode", "legacy")).lower()
        if self._unit_target_filter_mode not in ("legacy", "one_euro"):
            self._unit_target_filter_mode = "legacy"
        self._unit_filter_min_cutoff = max(
            0.01, float(_u.get("target_filter_min_cutoff", 2.0)))
        self._unit_filter_beta = max(
            0.0, float(_u.get("target_filter_beta", 0.05)))
        self._unit_filter_d_cutoff = max(
            0.01, float(_u.get("target_filter_d_cutoff", 1.0)))
        self._unit_lead_f = max(0.0, float(_u.get("lead_frames", 1.0)))     # 速度前导(帧)
        # ── 帧调度与时间归一化(unit.frame_schedule) ──
        # fixed: 旧"每帧参数"语义原样生效(与历史版本逐位一致, 所有 _norm_*/时间门控
        #        直接退化为原值/帧计数)。
        # asap:  观测周期由真实端到端耗时决定, 把按帧整定的参数换算到真实时间:
        #        alpha_dt = 1 - (1 - alpha_ref) ** (actual_dt / reference_dt)
        #        每帧位移阈值 × (actual_dt / reference_dt); 帧数阈值 × reference_dt。
        self.frame_schedule = str(_u.get("frame_schedule", "fixed")).lower()
        if self.frame_schedule not in ("fixed", "asap"):
            self.frame_schedule = "fixed"
        self._time_norm = self.frame_schedule == "asap"
        _ref_ms = float(_u.get("reference_frame_ms", _u.get("frame_ms", 22.0)))
        self.reference_dt = max(0.002, _ref_ms / 1000.0)
        self._unit_auto_cal = bool(_u.get("auto_cal", False))  # 在线 k 自标定(默认关):
        # 实机对账显示自标定被"目标自身运动"污染的配对反复带偏(sens 3.32→7.68),
        # 直接标定的 view_scale(0.3011/0.2684) 比在线回归可靠; 需要再开。
        self._unit_dz = max(0.5, float(_u.get("deadzone", 8.0)))  # unit 停发死区(px):
        self._unit_stop_dz = max(
            0.0, min(self._unit_dz, float(_u.get("stop_deadzone", 0.75))))
        self._missing_decay = max(
            0.0, min(1.0, float(ac.get("missing_decay", 0.45))))
        self._missing_max_counts = max(
            0.0, float(ac.get("missing_max_counts", 24.0)))
        # 抖动相位里误差常超 3.96(人机化死区), 导致到达目标后仍连续小发、准星发抖;
        # 用更大的停发死区 + 目标点滤波(见 compute)让准星到位即停。
        self._unit_tp = [0.0, 0.0]      # 轻EMA目标点
        self._unit_tp_ok = False
        self._unit_tp_raw = [0.0, 0.0]
        self._unit_tp_derivative = [0.0, 0.0]
        self._send_verified = True  # 上一发是否"已验证生效"(main 串行验证结果; 模拟器恒 True)
        self._prev_sent = [0.0, 0.0]        # 上一发 counts
        self._prev_err = [0.0, 0.0]         # 上一发送前误差(px)
        # 逐轴最小二乘累加器(EWMA 滚动窗): k = Σ(C·Δe)/Σ(C²)
        self._u_ssx = self._u_ssy = 1e-9
        self._u_syx = self._u_syy = 0.0
        # 时间节流已按需求移除(smooth 模式逐帧直发,direct 模式保留固定间隔,
        # 见 compute;发送节奏交给实际灵敏度匹配)。
        self.reset()

    # ── 状态 ──
    def reset(self):
        self._unit_tp_ok = False   # unit 目标点吸附状态随会话重置(新开局首帧重新吸附)
        self._unit_tp[0] = self._unit_tp[1] = 0.0
        self._unit_tp_raw[0] = self._unit_tp_raw[1] = 0.0
        self._unit_tp_derivative[0] = self._unit_tp_derivative[1] = 0.0
        self._img_center = [0.0, 0.0]
        self.locked_box = [None]
        self.locked_cls = [None]
        self.locked_conf = [0.0]
        self.locked_frame_count = [0]
        self._locked_time_s = 0.0    # asap: 锁定持续真实时间(帧数门控的时间等价)
        self.lock_miss_count = [0]
        self._miss_time_s = 0.0      # asap: 丢失持续真实时间
        self._switch_miss = [0]
        self._switch_time_s = 0.0    # asap: 切换容错累计时间
        self._just_locked = [False]
        self._cand_gap = [0]
        self._locked_box_smooth = [None]
        self._raw_motion_point = None
        self._target_ema = [0.0, 0.0]
        self._target_ema_init = [False]
        self._no_target_frames = 0   # 连续无目标帧数(首枪快吸附:断档后重识别 snap EMA)
        self._no_target_time_s = 0.0  # asap: 同上的时间等价(2帧≈44ms)
        self._target_vel = [0.0, 0.0]
        self._vel_valid = [False]
        self._vel_frames = [0]
        self._vel_age_s = 0.0        # asap: 速度确认累计时间(_VEL_CONFIRM帧≈66ms)
        self._vel_lw = [0.0, 0.0]       # 长窗速度(屏幕px/帧)
        self._vel_lw_valid = False      # 长窗速度激活标志
        self._prev_target = [None]
        self._prev_cls = [None]
        self._own_vel = [0.0, 0.0]
        self._frame_net = [0.0, 0.0]
        self._lock_pending_shift = [0.0, 0.0]
        self._inflight = []
        self._comp_active = [False]
        self._comp_frames = [0]
        self._comp_cooldown = [False]
        self._last_net_move = [0.0, 0.0]
        self._unit_velocity = [0.0, 0.0]  # 物理目标速度(px/s)，已扣除自身鼠标运动
        self._unit_velocity_valid = False
        self._unit_velocity_frames = 0
        self._unit_velocity_valid_axes = [False, False]
        self._unit_velocity_frames_axes = [0, 0]
        self._unit_velocity_age_s_axes = [0.0, 0.0]  # asap: 速度轴确认时间(2帧≈44ms)
        self._unit_raw_velocity = [0.0, 0.0]
        self._net_since_observation = [0.0, 0.0]
        self._now = 0.0
        self._last_send_time = 0.0
        self._frame_no = 0   # 帧计数器(发送周期用,识别/EMA/速度每帧照常更新,只节流发送)
        self._frame_dt = 0.04
        self.lock_filter_alpha_current = 1.0
        # 诊断
        self.diag_reason = None
        self.diag_candidates = 0
        self.diag_target = None

    def set_crosshair(self, cx, cy):
        self._img_center[0] = cx
        self._img_center[1] = cy

    # ── 时间归一化(fixed 下全部退化为原值, 保证逐位兼容) ──
    def _frame_scale(self):
        """asap: 实际观测周期 / 标称整定周期(reference_frame_ms); fixed: 恒 1。"""
        if not self._time_norm:
            return 1.0
        return max(1e-4, self._frame_dt / self.reference_dt)

    def _norm_alpha(self, alpha_ref):
        """把"每参考帧"的修正系数换算到实际 dt: a_dt = 1-(1-a_ref)^(dt/ref)。"""
        if not self._time_norm:
            return alpha_ref
        k = max(1e-4, self._frame_dt / self.reference_dt)
        return 1.0 - (1.0 - alpha_ref) ** k

    def _norm_retention(self, retention_ref):
        """把"每参考帧"的保留权重换算到实际 dt: a_dt = a_ref^(dt/ref)。

        用于 "ema = ema*ret + x*(1-ret)" 形态的 EMA: 保留权重与修正权重
        必须同时归一(和恒为 1), 只归一边会导致稳态偏移。
        """
        if not self._time_norm:
            return retention_ref
        k = max(1e-4, self._frame_dt / self.reference_dt)
        return retention_ref ** k

    def _norm_disp(self, px):
        """把"每参考帧位移"阈值换算到实际 dt(×dt/ref), 保持 px/s 语义。"""
        if not self._time_norm:
            return px
        return px * max(1e-4, self._frame_dt / self.reference_dt)

    def _frames_gate(self, frame_count, elapsed_s, frames,
                     initial_counted=False):
        """Apply a frame threshold without changing fixed-mode semantics."""
        if not self._time_norm:
            return frame_count >= frames
        required_frames = max(0, int(frames) - (1 if initial_counted else 0))
        return elapsed_s >= required_frames * self.reference_dt

    def set_last_verified(self, ok):
        """main 串行验证结果: 上一发是否确认已生效(误差按发送方向缩小)。
        False(未验证到/超时)时, 该发送不参与 k 标定 —— 防止"效应还没落地"被误读
        成 k≈0, 导致 sens 一帧崩到上限(实战日志: sens 3.32→20 后永远顶满)。"""
        self._send_verified = bool(ok)

    def set_prediction_mode(self, mode):
        """Switch between current-box pursuit and arrival-time prediction."""
        mode = str(mode).lower()
        if mode not in ("current", "arrival"):
            raise ValueError("prediction mode must be 'current' or 'arrival'")
        self.prediction_mode = mode
        self.unit_prediction_enabled = mode == "arrival"

    @staticmethod
    def _one_euro_alpha(cutoff, dt):
        dt = max(1e-6, float(dt))
        tau = 1.0 / (2.0 * math.pi * max(0.01, float(cutoff)))
        return 1.0 / (1.0 + tau / dt)

    def _filter_unit_target(self, target, dt, reset=False):
        """Filter the unit-mode target without a displacement threshold."""
        if reset or not self._unit_tp_ok:
            self._unit_tp[0], self._unit_tp[1] = target[0], target[1]
            self._unit_tp_raw[0], self._unit_tp_raw[1] = target[0], target[1]
            self._unit_tp_derivative[0] = self._unit_tp_derivative[1] = 0.0
            self._unit_tp_ok = True
            return tuple(self._unit_tp)

        if self._unit_target_filter_mode == "legacy":
            # 6/40px 是"每参考帧位移"阈值; asap 下按 dt/ref 缩放保持 px/s 语义。
            _snap_px = self._norm_disp(40.0)
            _follow_px = self._norm_disp(6.0)
            _filt_alpha = self._norm_alpha(self._unit_target_filter_alpha)
            pd = math.hypot(target[0] - self._unit_tp[0],
                            target[1] - self._unit_tp[1])
            if pd > _snap_px:
                self._unit_tp[0], self._unit_tp[1] = target[0], target[1]
            elif pd > _follow_px:
                self._unit_tp[0], self._unit_tp[1] = target[0], target[1]
            else:
                for axis in (0, 1):
                    self._unit_tp[axis] += (
                        target[axis] - self._unit_tp[axis]
                    ) * _filt_alpha
            self._unit_tp_raw[0], self._unit_tp_raw[1] = target[0], target[1]
            return tuple(self._unit_tp)

        alpha_d = self._one_euro_alpha(self._unit_filter_d_cutoff, dt)
        for axis in (0, 1):
            raw_derivative = (target[axis] - self._unit_tp_raw[axis]) / max(1e-6, dt)
            self._unit_tp_derivative[axis] += alpha_d * (
                raw_derivative - self._unit_tp_derivative[axis])
            cutoff = (self._unit_filter_min_cutoff
                      + self._unit_filter_beta * abs(self._unit_tp_derivative[axis]))
            alpha = self._one_euro_alpha(cutoff, dt)
            self._unit_tp[axis] += alpha * (target[axis] - self._unit_tp[axis])
            self._unit_tp_raw[axis] = target[axis]
        return tuple(self._unit_tp)

    def notify_net(self, dx, dy):
        # 共享输出器在两次YOLO观测之间可能发送多次，必须累计而非覆盖。
        self._frame_net[0] += dx
        self._frame_net[1] += dy
        self._net_since_observation[0] += dx
        self._net_since_observation[1] += dy
        # 下一张截图里，同一物理目标会因本次自身视角移动反向平移。
        self._lock_pending_shift[0] -= float(dx) * self.view_scale
        self._lock_pending_shift[1] -= float(dy) * self.view_scale_y

    def end_frame(self):
        # 对齐 main.py 2019-2021:用本帧净发送更新"自身视角速度" EMA
        # (asap 下保留权重按 (dt/ref) 取幂归一, fixed 下为原 _VEL_ALPHA)
        _own_ret = self._norm_retention(_VEL_ALPHA)
        self._own_vel[0] = self._own_vel[0] * _own_ret + self._frame_net[0] * (1.0 - _own_ret)
        self._own_vel[1] = self._own_vel[1] * _own_ret + self._frame_net[1] * (1.0 - _own_ret)
        self._frame_net[0] = 0.0
        self._frame_net[1] = 0.0

    # ── 瞄准点(对齐 main.py _aim_point)──
    def _aim_point(self, d):
        bx1, by1, bx2, by2 = d["bbox"]
        cx = (bx1 + bx2) * 0.5
        if d["cls"] == 1:
            return cx, by1 + (by2 - by1) * self.head_ratio
        return cx, by1 + (by2 - by1) * self.chest

    @staticmethod
    def _motion_point(d):
        """Geometry-only velocity anchor, unaffected by box height/chest aim ratio."""
        bx1, by1, bx2, by2 = d["bbox"]
        return (bx1 + bx2) * 0.5, (by1 + by2) * 0.5

    # ── 框增益(对齐 main.py _box_scale_gain)──
    def _box_scale_gain(self, d):
        bx1, by1, bx2, by2 = d["bbox"]
        size = max(by2 - by1, bx2 - bx1)
        size = max(self.bsg_min_size, min(self.bsg_max_size, size))
        t = (size - self.bsg_min_size) / (self.bsg_max_size - self.bsg_min_size)
        return self.bsg_min_gain + (self.bsg_max_gain - self.bsg_min_gain) * t

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union

    def _is_edge_box(self, box):
        m = self.edge_new_lock_margin
        return (box[0] <= m or box[1] <= m or
                box[2] >= self.capture_size - m or
                box[3] >= self.capture_size - m)

    def _lock_filter_alpha(self, detection):
        """按扣除自身视角运动后的原始框速度选择位置滤波强度。"""
        if self.lock_box_filter_mode == "fixed":
            return self.lock_box_smoothing_alpha
        if self._raw_motion_point is None:
            return self.lock_box_alpha_min
        raw_x, raw_y = self._motion_point(detection)
        net_x, net_y = self._net_since_observation
        dx = raw_x - self._raw_motion_point[0] + net_x * self.view_scale
        dy = raw_y - self._raw_motion_point[1] + net_y * self.view_scale_y
        speed = math.hypot(dx, dy) / max(1e-4, self._frame_dt)
        ratio = ((speed - self.lock_box_speed_low) /
                 (self.lock_box_speed_high - self.lock_box_speed_low))
        ratio = max(0.0, min(1.0, ratio))
        return (self.lock_box_alpha_min
                + (self.lock_box_alpha_max - self.lock_box_alpha_min) * ratio)

    def _reset_unit_prediction(self):
        self._unit_velocity[0] = self._unit_velocity[1] = 0.0
        self._unit_velocity_valid_axes[:] = [False, False]
        self._unit_velocity_frames_axes[:] = [0, 0]
        self._unit_velocity_age_s_axes[:] = [0.0, 0.0]
        self._unit_velocity_valid = False
        self._unit_velocity_frames = 0

    def _clear_lock(self):
        self.locked_box[0] = None
        self._locked_box_smooth[0] = None
        self._raw_motion_point = None
        self.locked_cls[0] = None
        self.locked_conf[0] = 0.0
        self.locked_frame_count[0] = 0
        self._locked_time_s = 0.0
        self.lock_miss_count[0] = 0
        self._miss_time_s = 0.0
        self._switch_miss[0] = 0
        self._switch_time_s = 0.0
        self._unit_tp_ok = False
        self._vel_valid[0] = False
        self._vel_frames[0] = 0
        self._target_vel[0] = self._target_vel[1] = 0.0
        self._reset_unit_prediction()

    def _start_lock(self, d):
        self._just_locked[0] = True
        # 换向阻尼只属于同一目标的连续控制；新锁不能继承旧目标方向。
        self._last_net_move[0] = self._last_net_move[1] = 0.0
        self._ease_left = self.ease_frames
        self.locked_box[0] = list(d["bbox"])
        self._locked_box_smooth[0] = list(d["bbox"])
        self.locked_cls[0] = d["cls"]
        self.locked_conf[0] = d["conf"]
        self._raw_motion_point = self._motion_point(d)
        self.locked_frame_count[0] = 1
        self._locked_time_s = 0.0
        self.lock_miss_count[0] = 0
        self._miss_time_s = 0.0
        self._switch_miss[0] = 0
        self._switch_time_s = 0.0
        self._unit_tp_ok = False
        self._reset_unit_prediction()
        self._prev_target[0] = None
        return self._aim_point(d)

    def _apply_lock_motion_compensation(self):
        sx, sy = self._lock_pending_shift
        self._lock_pending_shift[0] = self._lock_pending_shift[1] = 0.0
        if self.locked_box[0] is None or (sx == 0.0 and sy == 0.0):
            return
        for box_ref in (self.locked_box, self._locked_box_smooth):
            if box_ref[0] is not None:
                box_ref[0] = [box_ref[0][0] + sx, box_ref[0][1] + sy,
                              box_ref[0][2] + sx, box_ref[0][3] + sy]
        if self._unit_tp_ok:
            self._unit_tp[0] += sx
            self._unit_tp[1] += sy
            self._unit_tp_raw[0] += sx
            self._unit_tp_raw[1] += sy
        if self._target_ema_init[0]:
            self._target_ema[0] += sx
            self._target_ema[1] += sy

    def _advance_lock_prediction(self):
        """歧义/短失检时只推进锁的预测位置，不产生鼠标指令。"""
        if self.locked_box[0] is None:
            return
        if self._vel_valid[0]:
            vx, vy = self._target_vel
        elif self._vel_lw_valid:
            vx, vy = self._vel_lw
        else:
            return
        mag = math.hypot(vx, vy)
        if mag <= 0.0:
            return
        _max_step = self._norm_disp(self.lock_prediction_max_step)
        if _max_step > 0.0 and mag > _max_step:
            scale = _max_step / mag
            vx *= scale
            vy *= scale
        for box_ref in (self.locked_box, self._locked_box_smooth):
            if box_ref[0] is not None:
                box_ref[0] = [box_ref[0][0] + vx, box_ref[0][1] + vy,
                              box_ref[0][2] + vx, box_ref[0][3] + vy]
        if self._unit_tp_ok:
            self._unit_tp[0] += vx
            self._unit_tp[1] += vy
            self._unit_tp_raw[0] += vx
            self._unit_tp_raw[1] += vy
        if self._target_ema_init[0]:
            self._target_ema[0] += vx
            self._target_ema[1] += vy
        if self._prev_target[0] is not None:
            self._prev_target[0] = (self._prev_target[0][0] + vx,
                                    self._prev_target[0][1] + vy)

    # ── 目标选择：首次目标粘锁，自身鼠标位移补偿后再做连续性匹配 ──
    def find_target(self, dets):
        self._apply_lock_motion_compensation()
        cands = [d for d in dets if d["cls"] in self.target_cls] if dets else []
        num_cands = len(cands)
        cx0, cy0 = self._img_center

        target_dists = []
        for d in cands:
            bx = d["bbox"]
            cx = (bx[0] + bx[2]) * 0.5
            cy = (bx[1] + bx[3]) * 0.5
            dist = math.hypot(cx - cx0, cy - cy0)
            target_dists.append((dist, d, cx, cy))
        # 头/身优先级可配置；同类内仍离准星最近，身框内部保持 0 优先于 5。
        # cls=1 是现有内部约定的头框，其他候选保持原有身框语义。
        head_first = self.target_priority == "head"
        target_dists.sort(key=lambda x: (
                                         0 if ((x[1]["cls"] == 1) == head_first) else 1,
                                         1 if x[1]["cls"] == 5 else 0,
                                         x[0]))

        if not target_dists:
            self._cand_gap[0] += 1
            self._switch_miss[0] = 0
            self._switch_time_s = 0.0
            if self.locked_box[0] is not None:
                self.lock_miss_count[0] += 1
                self._miss_time_s += self._frame_dt
                # 未达丢失容错阈值前继续预测锁(_frames_gate=已达标)。
                if not self._frames_gate(self.lock_miss_count[0], self._miss_time_s,
                                         self.lock_grace):
                    self._advance_lock_prediction()
                    target = self._aim_point({
                        "bbox": self.locked_box[0],
                        "cls": self.locked_cls[0],
                    })
                    self.diag_reason = (
                        f"predict_missing({self.lock_miss_count[0]}/{self.lock_grace})")
                    self.diag_candidates = num_cands
                    self.diag_target = target
                    return target
                self._clear_lock()
            self.diag_reason = "no_detections"
            self.diag_candidates = num_cands
            self.diag_target = None
            return None

        self._cand_gap[0] = 0
        switch_reason = "selected"
        if self.locked_box[0] is not None:
            lx1, ly1, lx2, ly2 = self.locked_box[0]
            lcx, lcy = (lx1 + lx2) * 0.5, (ly1 + ly2) * 0.5
            _lw = max(1.0, float(lx2 - lx1))
            _lh = max(1.0, float(ly2 - ly1))
            _lock_tol = min(self.lock_dist_tolerance, max(45.0, max(_lw, _lh) * 1.5))
            old_head = self.locked_cls[0] == 1
            matches = []
            for _, d, cx, cy in target_dists:
                # 头框与身框不互相接管锁；0/5 都视为身框组。
                if (d["cls"] == 1) != old_head:
                    continue
                center_dist = math.hypot(cx - lcx, cy - lcy)
                iou = self._iou(self.locked_box[0], d["bbox"])
                if center_dist > _lock_tol and iou < self.lock_iou_thresh:
                    continue
                if (iou < self.lock_iou_thresh and
                        center_dist > self.lock_zero_iou_max_distance):
                    continue
                nw = max(1.0, d["bbox"][2] - d["bbox"][0])
                nh = max(1.0, d["bbox"][3] - d["bbox"][1])
                size_penalty = 0.25 * (abs(math.log(nw / _lw)) + abs(math.log(nh / _lh)))
                cls_penalty = 0.1 if d["cls"] != self.locked_cls[0] else 0.0
                score = center_dist / max(1.0, _lock_tol) + (1.0 - iou) + size_penalty + cls_penalty
                matches.append((score, center_dist, iou, d))
            matches.sort(key=lambda row: row[0])

            ambiguous = (len(matches) > 1 and
                         matches[1][0] - matches[0][0] <= self.lock_ambiguity_margin)
            if matches and not ambiguous:
                _, center_dist, iou, d = matches[0]
                smooth = self._lock_filter_alpha(d)
                self.lock_filter_alpha_current = smooth
                # asap 下把"每参考帧"alpha 换算到实际 dt(公式见 _norm_alpha)。
                smooth_eff = self._norm_alpha(smooth)
                if self._locked_box_smooth[0] is None:
                    self._locked_box_smooth[0] = list(d["bbox"])
                else:
                    self._locked_box_smooth[0] = [
                        self._locked_box_smooth[0][i] * (1.0 - smooth_eff) + d["bbox"][i] * smooth_eff
                        for i in range(4)
                    ]
                self.locked_box[0] = list(self._locked_box_smooth[0])
                self.locked_cls[0] = d["cls"]
                self.locked_conf[0] = d["conf"]
                self._raw_motion_point = self._motion_point(d)
                self.locked_frame_count[0] += 1
                self._locked_time_s += self._frame_dt
                self.lock_miss_count[0] = 0
                self._miss_time_s = 0.0
                self._switch_miss[0] = 0
                self._switch_time_s = 0.0
                target = self._aim_point({"bbox": self.locked_box[0], "cls": d["cls"]})
                self.diag_reason = f"locked(dist={center_dist:.0f},iou={iou:.2f})"
                self.diag_candidates = num_cands
                self.diag_target = target
                return target

            self._switch_miss[0] += 1
            self._switch_time_s += self._frame_dt
            if not self._frames_gate(self._switch_miss[0], self._switch_time_s,
                                     self.switch_grace):
                self._advance_lock_prediction()
                kind = "ambiguous" if ambiguous else "switch"
                self.diag_reason = f"hold_{kind}({self._switch_miss[0]}/{self.switch_grace})"
                self.diag_candidates = num_cands
                self.diag_target = None
                return None
            self._clear_lock()
            switch_reason = "switched_after_grace"

        # 截图边缘残框不能建立新锁；若此前已匹配到原锁，则上面已经允许续锁。
        selectable = [row for row in target_dists if not self._is_edge_box(row[1]["bbox"])]
        if not selectable:
            self.diag_reason = "edge_only_no_new_lock"
            self.diag_candidates = num_cands
            self.diag_target = None
            return None
        dist, d, _, _ = selectable[0]
        target = self._start_lock(d)
        self.diag_reason = f"{switch_reason}(dist={dist:.0f})"
        self.diag_candidates = num_cands
        self.diag_target = target
        return target

    # ── 每帧瞄准计算(对齐 main.py aim_frame 的核心;返回给外部发鼠标)──
    def compute(self, dets, crosshair, dt, recoil_recovery=False,
                processing_latency_s=0.0,
                recoil_feedforward_active=False):
        self._img_center[0], self._img_center[1] = crosshair
        self._now += dt
        self._frame_dt = max(1e-4, float(dt))

        target = self.find_target(dets)
        _obs_net_x, _obs_net_y = self._net_since_observation
        self._net_since_observation[0] = self._net_since_observation[1] = 0.0

        # EMA 更新
        if target is not None:
            # 首枪快吸附:断档足够久(固定2帧/asap≈44ms)后重新识别,EMA 直接 snap 到新
            # 目标(并清速度),避免从旧 EMA 位置龟速逼近导致前几发准星不动。
            if self._frames_gate(self._no_target_frames, self._no_target_time_s, 2):
                self._target_ema[0] = target[0]
                self._target_ema[1] = target[1]
                self._target_ema_init[0] = True
                self._vel_valid[0] = False
                self._vel_frames[0] = 0
                self._vel_age_s = 0.0
                self._target_vel[0] = 0.0
                self._target_vel[1] = 0.0
            self._no_target_frames = 0
            self._no_target_time_s = 0.0
            _e_dx = target[0] - self._target_ema[0]
            _e_dy = target[1] - self._target_ema[1]
            _e_d = math.hypot(_e_dx, _e_dy)
            _ema_step = self._norm_disp(self.ema_max_step)
            if not self._target_ema_init[0]:
                self._target_ema[0] = target[0]
                self._target_ema[1] = target[1]
                self._target_ema_init[0] = True
            elif self.ema_max_step > 0 and _e_d > _ema_step:
                self._target_ema[0] += _e_dx / _e_d * _ema_step
                self._target_ema[1] += _e_dy / _e_d * _ema_step
            else:
                a = max(0.0, min(1.0, self.target_ema_alpha))
                if self._time_norm:
                    # 归一化"保留权重": a_dt = a_ref^(dt/ref), 修正权重=1-a_dt
                    # (等价于任务公式 alpha_dt = 1-(1-alpha_ref)^(dt/ref),
                    #  其中 alpha_ref = 1-a 为参考帧修正系数)。
                    a = a ** max(1e-4, self._frame_dt / self.reference_dt)
                self._target_ema[0] = self._target_ema[0] * a + target[0] * (1.0 - a)
                self._target_ema[1] = self._target_ema[1] * a + target[1] * (1.0 - a)

            # 速度模型(摄像头耦合 + Change1/2/3;A/D 侧移视差补偿已按需求移除)
            _new_cls = self.locked_cls[0]
            if self._prev_cls[0] is not None and _new_cls != self._prev_cls[0]:
                self._vel_valid[0] = False
                self._vel_frames[0] = 0
                self._vel_age_s = 0.0
                self._target_vel[0] = 0.0
                self._target_vel[1] = 0.0
                self._reset_unit_prediction()
            self._prev_cls[0] = _new_cls

            if self._raw_motion_point is not None:
                _bcn = self._raw_motion_point
            elif self.locked_box[0] is not None:
                _bcn = ((self.locked_box[0][0] + self.locked_box[0][2]) * 0.5,
                        (self.locked_box[0][1] + self.locked_box[0][3]) * 0.5)
            else:
                _bcn = (target[0], target[1])
            _bbH = max(0.0, self.locked_box[0][3] - self.locked_box[0][1]) if self.locked_box[0] is not None else 0.0
            # 速度死区与地板是"每参考帧位移"量, asap 下按 dt/ref 缩放。
            _vel_dz = self._norm_disp(max(self.vel_deadzone, _bbH * self.vel_deadzone_frac))
            if self._prev_target[0] is not None:
                # 新 unit 预测器：检测框位移 + 两次观测间自身鼠标造成的反向画面位移。
                _physical_dx = (_bcn[0] - self._prev_target[0][0]) + _obs_net_x * self.view_scale
                _physical_dy = (_bcn[1] - self._prev_target[0][1]) + _obs_net_y * self.view_scale_y
                _raw_vx = _physical_dx / max(1e-4, dt)
                _raw_vy = _physical_dy / max(1e-4, dt)
                self._unit_raw_velocity[0] = _raw_vx
                self._unit_raw_velocity[1] = _raw_vy
                # During replay, vertical box motion contains unknown weapon
                # kick.  Isolate axes before reversal/outlier gating so a Y
                # jump cannot erase a valid horizontal crab-walk estimate.
                _model_vx = _raw_vx
                _model_vy = 0.0 if recoil_feedforward_active else _raw_vy
                for axis, model_velocity in enumerate((_model_vx, _model_vy)):
                    suppressed = axis == 1 and recoil_feedforward_active
                    reversed_axis = bool(
                        self._unit_velocity_valid_axes[axis]
                        and model_velocity * self._unit_velocity[axis] < 0.0
                        and abs(model_velocity) > 80.0
                        and abs(self._unit_velocity[axis]) > 80.0)
                    if suppressed or abs(model_velocity) > 2400.0:
                        # 只让发生换向/污染的轴重新确认，另一轴继续提供前导。
                        self._unit_velocity[axis] = 0.0
                        if axis != 0 or not self._unit_velocity_valid_axes[axis]:
                            self._unit_velocity_valid_axes[axis] = False
                            self._unit_velocity_frames_axes[axis] = 0
                            self._unit_velocity_age_s_axes[axis] = 0.0
                        continue
                    if reversed_axis:
                        self._unit_velocity[axis] = 0.0
                        if axis == 0:
                            # 横向单帧反号先空一帧前导，但保留速度估计器就绪态；
                            # 下一次观测可直接按真实方向恢复，避免重新等待两帧。
                            continue
                        self._unit_velocity_valid_axes[axis] = False
                        self._unit_velocity_frames_axes[axis] = 0
                        self._unit_velocity_age_s_axes[axis] = 0.0
                        continue
                    _av = 1.0 - math.exp(-dt / self.prediction_vel_tau)
                    self._unit_velocity[axis] += (
                        model_velocity - self._unit_velocity[axis]) * _av
                    self._unit_velocity_frames_axes[axis] += 1
                    self._unit_velocity_age_s_axes[axis] += dt
                    self._unit_velocity_valid_axes[axis] = self._frames_gate(
                        self._unit_velocity_frames_axes[axis],
                        self._unit_velocity_age_s_axes[axis], 2)
                self._unit_velocity_valid = any(self._unit_velocity_valid_axes)
                self._unit_velocity_frames = max(self._unit_velocity_frames_axes)
                _vx = (_bcn[0] - self._prev_target[0][0]) + self._own_vel[0] * self.view_scale
                _vy = (_bcn[1] - self._prev_target[0][1]) + self._own_vel[1] * self.view_scale_y
                if math.hypot(_vx, _vy) < _vel_dz:
                    _vx = _vy = 0.0
                if math.hypot(_vx, _vy) <= max(_ema_step, self._norm_disp(1.0)):
                    if self._vel_valid[0]:
                        _vel_corr = self._norm_alpha(1.0 - _VEL_ALPHA)
                        self._target_vel[0] += (_vx - self._target_vel[0]) * _vel_corr
                        self._target_vel[1] += (_vy - self._target_vel[1]) * _vel_corr
                    else:
                        self._vel_frames[0] += 1
                        self._vel_age_s += dt
                        self._target_vel[0] += (_vx - self._target_vel[0]) / self._vel_frames[0]
                        self._target_vel[1] += (_vy - self._target_vel[1]) / self._vel_frames[0]
                        if self._frames_gate(self._vel_frames[0], self._vel_age_s,
                                             _VEL_CONFIRM):
                            self._vel_valid[0] = True
                else:
                    self._vel_valid[0] = False
                    self._vel_frames[0] = 0
                    self._vel_age_s = 0.0
                    self._target_vel[0] = 0.0
                    self._target_vel[1] = 0.0
            # ── 长窗速度估计(lead 用,不受 per-frame 死区门控)──
            # 目标小幅但持续漂移(实机晃动)时,per-frame 死区把单帧漂移直接判静止,
            # 旧速度模型永不激活 → lead 恒 0 → 只剩被动追赶的稳态滞后。
            # 这里用更长窗口(vel_lw_t~0.12s)EMA 平均漂移,抖动(零均值)自然抵消,
            # 只对"持续运动"激活;单帧跳变(锁切换/尖峰)不喂窗口,换向反向也被窗口磨平。
            if self._prev_target[0] is not None:
                _lw_x = (_bcn[0] - self._prev_target[0][0]) + self._own_vel[0] * self.view_scale
                _lw_y = (_bcn[1] - self._prev_target[0][1]) + self._own_vel[1] * self.view_scale_y
                _lw_limit = self._norm_disp(self.ema_max_step + 1.0)
                if math.hypot(_lw_x, _lw_y) <= _lw_limit:
                    _alw = 1.0 - math.exp(-dt / max(1e-3, self.vel_lw_t))
                    self._vel_lw[0] += (_lw_x - self._vel_lw[0]) * _alw
                    self._vel_lw[1] += (_lw_y - self._vel_lw[1]) * _alw
                    _lw_floor = self._norm_disp(
                        max(self.vel_lw_floor, _bbH * self.vel_lw_floor_frac))
                    if math.hypot(self._vel_lw[0], self._vel_lw[1]) >= _lw_floor:
                        self._vel_lw_valid = True
                    else:
                        self._vel_lw_valid = False
                        self._vel_lw[0] = 0.0
                        self._vel_lw[1] = 0.0
            else:
                self._vel_lw_valid = False
                self._vel_lw[0] = 0.0
                self._vel_lw[1] = 0.0
            self._prev_target[0] = _bcn
        else:
            self._no_target_frames += 1
            self._no_target_time_s += self._frame_dt
            # 短时失检/歧义仍保留最后可信速度，只推进预测锁且暂停鼠标；
            # 达到宽限并真正释放后才清空速度状态。
            if self.locked_box[0] is None:
                self._vel_valid[0] = False
                self._vel_frames[0] = 0
                self._vel_age_s = 0.0
                self._target_vel[0] = 0.0
                self._target_vel[1] = 0.0
                self._vel_lw_valid = False
                self._vel_lw[0] = 0.0
                self._vel_lw[1] = 0.0
                self._prev_target[0] = None
                self._prev_cls[0] = None

        # 时间节流(合成时钟)
        out = {
            "can_send": False, "in_deadzone": False, "dx": 0.0, "dy": 0.0,
            "cap_hint": None, "raw_dx": 0.0, "raw_dy": 0.0, "deadzone_px": 0.0,
            "target": target, "lead": (0.0, 0.0), "diag_gain": 1.0,
            "sens": 0.0, "gain_clamped": False,
            "recoil_recovery": False,
            "prediction_suppressed": False,
            "observed_error": (0.0, 0.0),
            "velocity_raw": tuple(self._unit_raw_velocity),
            "velocity_filtered": tuple(self._unit_velocity),
            "prediction_horizon_ms": 0.0,
            "prediction_reason": "current_mode" if self.prediction_mode == "current" else "velocity_unready",
            "prediction_mode": self.prediction_mode,
            "lock_filter_alpha": self.lock_filter_alpha_current,
        }

        # ── 满额纠错(aim_mode=unit): 直移目标框位置, 不做平滑/前导/增益 ──
        # 串行管线(发送后验证生效)保证本误差是"上一发已落地"后的新鲜值, 满额发送不积压。
        # 目标点三级处理防抖(与 k 自标定正交——auto_cal 关时纯滤波):
        #   >40px 跳变吸附(重置标定对账, 防目标切换);
        #   6~40px 真实运动硬跟(不拖慢追击);
        #   ≤6px 框抖动只极慢微调(滤抖, 准星到位即稳)。
        # 停发死区用 unit.deadzone(默认8px): 抖动相位误差常超旧 3.96px 导致到达后
        # 仍连续小发、准星发抖 —— 滤波+更大死区让到位即停。
        if self.aim_mode == "unit":
            out["lead"] = (0.0, 0.0)
            if target is None:
                self._prev_sent[0] = self._prev_sent[1] = 0.0
                self._unit_tp_ok = False   # 丢失后重置, 下次直接吸附
                self._ease_left = self.ease_frames
                out["can_send"] = False
                return out
            # ── 目标点: 自标定开启时用原始框中心(保证 Δe=k·C 模型成立);
            #    关闭时用三级滤波(去框抖, 与标定正交, 实机默认路径) ──
            if self._unit_auto_cal:
                _ue_x = target[0] - self._img_center[0]
                _ue_y = target[1] - self._img_center[1]
            else:
                _reset_filter = not self._unit_tp_ok
                if _reset_filter:
                    self._ease_left = self.ease_frames   # 新吸附重新缓动起步
                _previous = tuple(self._unit_tp)
                self._filter_unit_target(target, dt, reset=_reset_filter)
                if (not _reset_filter and self._unit_target_filter_mode == "legacy"
                        and math.hypot(target[0] - _previous[0],
                                       target[1] - _previous[1]) > self._norm_disp(40.0)):
                    self._ease_left = self.ease_frames
                _ue_x = self._unit_tp[0] - self._img_center[0]
                _ue_y = self._unit_tp[1] - self._img_center[1]
            out["observed_error"] = (_ue_x, _ue_y)
            if (self.unit_prediction_enabled and self.prediction_mode == "arrival"
                    and any(self._unit_velocity_valid_axes)
                    and self._frames_gate(self.locked_frame_count[0],
                                          self._locked_time_s,
                                          self.prediction_lock_frames,
                                          initial_counted=True)
                    and self.diag_reason is not None
                    and not self.diag_reason.startswith(
                        ("hold_", "switched_", "predict_missing"))):
                _horizon = max(self.prediction_min_s, min(
                    # 半帧采样相位 + 鼠标输出排队相位；上层传入实测处理延迟。
                    self.prediction_max_s, float(processing_latency_s) + dt * 0.75))
                _lead_x = (self._unit_velocity[0] * _horizon
                           if self._unit_velocity_valid_axes[0] else 0.0)
                _lead_y = (self._unit_velocity[1] * _horizon
                           if self._unit_velocity_valid_axes[1] else 0.0)
                out["prediction_horizon_ms"] = _horizon * 1000.0
                _box_w = (max(0.0, self.locked_box[0][2] - self.locked_box[0][0])
                          if self.locked_box[0] is not None else 0.0)
                _lead_cap = self.prediction_cap_px
                if _box_w > 0.0 and self.prediction_box_ratio > 0.0:
                    _lead_cap = min(_lead_cap, _box_w * self.prediction_box_ratio)
                if recoil_feedforward_active:
                    # Program-output attribution removes our known mouse motion,
                    # but the weapon's remaining camera kick is not directly
                    # observable.  Preserve crab-walk help on X while rejecting
                    # the recoil-contaminated Y estimate and using a tighter cap.
                    _lead_y = 0.0
                    _lead_cap = min(_lead_cap, self.recoil_prediction_cap_px)
                    if _box_w > 0.0 and self.recoil_prediction_box_ratio > 0.0:
                        _lead_cap = min(
                            _lead_cap, _box_w * self.recoil_prediction_box_ratio)
                    out["prediction_suppressed"] = True
                    out["prediction_mode"] = "recoil_horizontal"
                    out["prediction_reason"] = "recoil_horizontal_limited"
                else:
                    if all(self._unit_velocity_valid_axes):
                        out["prediction_reason"] = "arrival_active"
                    elif self._unit_velocity_valid_axes[0]:
                        out["prediction_reason"] = "arrival_partial_x"
                    else:
                        out["prediction_reason"] = "arrival_partial_y"
                _lead_mag = math.hypot(_lead_x, _lead_y)
                if _lead_cap > 0.0 and _lead_mag > _lead_cap:
                    _ls = _lead_cap / _lead_mag
                    _lead_x *= _ls
                    _lead_y *= _ls
                _ue_x += _lead_x
                _ue_y += _lead_y
                out["lead"] = (_lead_x, _lead_y)
            elif self.prediction_mode == "current" or not self.unit_prediction_enabled:
                out["prediction_reason"] = "current_mode"
            elif self.diag_reason is not None and self.diag_reason.startswith(("hold_", "switched_")):
                out["prediction_reason"] = "lock_state_blocked"
            elif not self._frames_gate(self.locked_frame_count[0],
                                       self._locked_time_s,
                                       self.prediction_lock_frames,
                                       initial_counted=True):
                out["prediction_reason"] = "lock_warmup"
            out["raw_dx"], out["raw_dy"] = _ue_x, _ue_y
            _dz = self._unit_dz
            _stop_dz = self._unit_stop_dz
            out["deadzone_px"] = _stop_dz
            # 上一发"已验证生效"后才标定(默认关闭, unit.auto_cal): Δerr ≈ k*C + 目标随机漂移
            # —— 实机对账证实漂移污染太大, 在线回归反而不如直接标定, 见 __init__ 注释。
            if self._unit_auto_cal and self._send_verified and \
                    (self._prev_sent[0] or self._prev_sent[1]) and \
                    max(abs(self._prev_err[0]), abs(self._prev_err[1])) > 3.0:
                _d_ex = self._prev_err[0] - _ue_x
                _d_ey = self._prev_err[1] - _ue_y
                # 过冲迹象(k 偏小): 上一发后某轴误差越过中轴翻号 → 快速忘记旧标定重新学习
                _decay = 0.90 if (
                    (self._prev_err[0] * _ue_x < -(_dz * _dz)) or
                    (self._prev_err[1] * _ue_y < -(_dz * _dz))) else 0.98
                self._u_ssx *= _decay; self._u_syx *= _decay
                self._u_ssy *= _decay; self._u_syy *= _decay
                if abs(self._prev_sent[0]) > 2.0:
                    self._u_ssx += self._prev_sent[0] * self._prev_sent[0]
                    self._u_syx += self._prev_sent[0] * _d_ex
                if abs(self._prev_sent[1]) > 2.0:
                    self._u_ssy += self._prev_sent[1] * self._prev_sent[1]
                    self._u_syy += self._prev_sent[1] * _d_ey
                # 单次标定限速 ±50%~100%: 任何一对坏数据最多把 k 折半/翻倍, 不瞬间崩塌
                if self._u_ssx > 0:
                    _nk = max(0.05, min(3.0, self._u_syx / self._u_ssx))
                    self._unit_k[0] = max(0.5 * self._unit_k[0],
                                          min(2.0 * self._unit_k[0], _nk))
                if self._u_ssy > 0:
                    _nk = max(0.05, min(3.0, self._u_syy / self._u_ssy))
                    self._unit_k[1] = max(0.5 * self._unit_k[1],
                                          min(2.0 * self._unit_k[1], _nk))
            _em = math.hypot(_ue_x, _ue_y)
            if _em <= _stop_dz:
                out["in_deadzone"] = True
                self._last_send_time = self._now
                self._prev_sent[0] = self._prev_sent[1] = 0.0
                out["can_send"] = True
                return out
            # 两段式 unit：远区接近满额一次到位，近区只对残差做低增益修正。
            # 这样保留标定换算的直接性，又不对红点/检测噪声满额反向纠正。
            if self._unit_continuous_gain:
                _stage_gains = []
                for axis, error in enumerate((_ue_x, _ue_y)):
                    _near = self._unit_near_gain_xy[axis]
                    _far = self._unit_far_gain_xy[axis]
                    _ratio = abs(error) / (abs(error) + self._unit_gain_softness[axis])
                    _stage_gains.append(_near + (_far - _near) * _ratio)
            else:
                _stage_gain = (self._unit_near_gain if _em <= self._unit_near_radius
                               else self._unit_far_gain)
                _stage_gains = [_stage_gain, _stage_gain]
            _soft_scale = 1.0
            if _stop_dz < _em < _dz:
                _soft_t = (_em - _stop_dz) / max(1e-6, _dz - _stop_dz)
                _soft_scale = _soft_t * _soft_t * (3.0 - 2.0 * _soft_t)
            if self._unit_damped:
                box_h = 0.0
                if self.locked_box[0] is not None:
                    box_h = max(0.0, min(640.0, self.locked_box[0][3] - self.locked_box[0][1]))
                _g_mult = float(self.hu.get("smoothing", 0.35))
                if self.box_scale_enabled and self.locked_box[0] is not None:
                    _g_mult *= self._box_scale_gain({"bbox": self.locked_box[0]})
                if self.boost_max > 1.0 and 0 < self.boost_lo_px < self.boost_hi_px \
                        and _em > self.boost_lo_px:
                    _t = min(1.0, (_em - self.boost_lo_px) / (self.boost_hi_px - self.boost_lo_px))
                    _g_mult *= 1.0 + (self.boost_max - 1.0) * _t
                _mult = 1.0
                if self.ease_frames > 0 and self._ease_left > 0:
                    _mult *= self.ease_min_gain + (1.0 - self.ease_min_gain) * (
                        1.0 - self._ease_left / float(max(1, self.ease_frames)))
                    self._ease_left -= 1
                if self.sticky_enabled and box_h > 0.0:
                    _sr = self.sticky_radius_ratio * box_h
                    if 0.0 < _em < _sr:
                        _mult *= self.sticky_gain + (1.0 - self.sticky_gain) * (_em / _sr)
                _g_mult *= _mult
                _gains = [max(0.05, min(2.0, gain * _g_mult))
                          for gain in _stage_gains]
                if self.max_total_gain > 0:
                    _gains = [min(self.max_total_gain, gain) for gain in _gains]
            else:
                _gains = list(_stage_gains)
            _gains[0] *= _soft_scale
            _gains[1] *= _soft_scale
            out["dx"] = _ue_x / self._unit_k[0] * _gains[0]
            # 首发红点已被可信地观测为向上跳时，Y 轴是外部后坐力扰动，
            # 不是控制器自身越界；这一帧直接按标定量全额追回。
            _recoil_y = bool(recoil_recovery and _ue_y > 0.0)
            _y_gain = 1.0 if _recoil_y else _gains[1]
            out["dy"] = _ue_y / self._unit_k[1] * _y_gain
            # 换向阻尼(与平滑 tail 一致): 与上一发方向相反时按系数缩放
            _damp_x = _damp_y = 1.0
            if (self._last_net_move[0] > 0 and out["dx"] < 0) or \
                    (self._last_net_move[0] < 0 and out["dx"] > 0):
                _damp_x = self.reversal_damp
            if not _recoil_y and ((self._last_net_move[1] > 0 and out["dy"] < 0) or \
                    (self._last_net_move[1] < 0 and out["dy"] > 0)):
                _damp_y = self.reversal_damp
            out["dx"] *= _damp_x
            out["dy"] *= _damp_y
            if self.diag_reason is not None and self.diag_reason.startswith("predict_missing"):
                # 丢失衰减按"丢失时长/参考帧"取幂: fixed 下等价于原帧数指数,
                # asap 下持续时间不受实际 YOLO FPS 影响。
                if self._time_norm:
                    _miss_exponent = max(1.0, self._miss_time_s / self.reference_dt)
                else:
                    _miss_exponent = max(1, self.lock_miss_count[0])
                _miss_scale = self._missing_decay ** _miss_exponent
                out["dx"] *= _miss_scale
                out["dy"] *= _miss_scale
                _miss_mag = math.hypot(out["dx"], out["dy"])
                if self._missing_max_counts > 0.0 and _miss_mag > self._missing_max_counts:
                    _miss_cap = self._missing_max_counts / _miss_mag
                    out["dx"] *= _miss_cap
                    out["dy"] *= _miss_cap
                # Keep the categorical reason stable for parity/replay; the
                # actual time basis is an explicit numeric diagnostic field,
                # never an ambiguous frame-count suffix.
                out["prediction_reason"] = "missing_decay"
                out["missing_duration_ms"] = self._miss_time_s * 1000.0
            self._last_net_move[0], self._last_net_move[1] = out["dx"], out["dy"]
            out["sens"] = 1.0 / self._unit_k[0] * _gains[0]
            out["diag_gain"] = max(_gains)
            out["recoil_recovery"] = _recoil_y
            out["gain_clamped"] = bool(
                self.max_total_gain > 0 and max(_stage_gains) > self.max_total_gain)
            out["can_send"] = True
            self._prev_sent[0], self._prev_sent[1] = out["dx"], out["dy"]
            self._prev_err[0], self._prev_err[1] = _ue_x, _ue_y
            return out
        # 发送周期(帧)：识别每帧都更新 EMA/速度/前导，只节流"发送"这一步——
        # 每 send_every_n_frames 帧发一次，给一次移动留出完整生效周期(用户 2:1 方案)。
        # direct 模式保留固定帧间隔(dm_interval_ms)防甩。
        if self.aim_mode == "direct":
            can_send = (self._now - self._last_send_time) * 1000.0 >= self.dm_interval_ms
        else:
            self._frame_no += 1
            # n=1:每帧直发; n=2:第1,3,5…帧发(会话首帧立即发,之后隔帧)
            can_send = (self.send_every_n_frames <= 1) or (self._frame_no % self.send_every_n_frames) == 1
        out["can_send"] = can_send if target is not None else False

        if out["can_send"]:
            _lead_x = 0.0
            _lead_y = 0.0
            _comp_x = 0.0
            _comp_y = 0.0
            if self.aim_mode == "direct":
                err_x = target[0] - self._img_center[0]
                err_y = target[1] - self._img_center[1]
                sens = self.dm_sens
                dx = err_x * sens
                dy = err_y * sens
                if max(abs(dx), abs(dy)) > self.dm_max_delta:
                    s = self.dm_max_delta / max(abs(dx), abs(dy))
                    dx *= s
                    dy *= s
                diag_gain = 1.0
                raw_dx, raw_dy = err_x, err_y
                deadzone_px = self.dm_deadzone
            else:
                _lead_x = self._vel_lw[0] * self.lead_frames if self._vel_lw_valid else (
                    self._target_vel[0] * self.lead_frames if self._vel_valid[0] else 0.0)
                _lead_y = self._vel_lw[1] * self.lead_frames if self._vel_lw_valid else (
                    self._target_vel[1] * self.lead_frames if self._vel_valid[0] else 0.0)
                _ld = math.hypot(_lead_x, _lead_y)
                if _ld > _LEAD_MAX:
                    _s = _LEAD_MAX / _ld
                    _lead_x *= _s
                    _lead_y *= _s
                # ── 前导安全限幅:前导 ≤ 比例误差(EMA-准星)的 0.5 倍 ──
                # 目标静止/急停时比例误差≈0 → 前导自动归零,根治"目标已停、前导仍在推"
                # 的反噬(lead_frames 调大时的失败模式);匀速追赶时前导最多贡献一半误差量。
                _plim = 0.5 * math.hypot(self._target_ema[0] - self._img_center[0],
                                         self._target_ema[1] - self._img_center[1])
                if _ld > _plim > 0.0:
                    _s = _plim / _ld
                    _lead_x *= _s
                    _lead_y *= _s
                _raw_dx = self._target_ema[0] + _lead_x - self._img_center[0]
                _raw_dy = self._target_ema[1] + _lead_y - self._img_center[1]
                if self.delay_comp_ms > 0 and self._inflight:
                    _raw_mag = math.hypot(_raw_dx, _raw_dy)
                    if self._comp_cooldown[0]:
                        if _raw_mag < _COMP_OFF:
                            self._comp_cooldown[0] = False
                    elif self._comp_active[0]:
                        if _raw_mag < _COMP_OFF:
                            self._comp_active[0] = False
                            self._comp_frames[0] = 0
                        else:
                            self._comp_frames[0] += 1
                            if self._comp_frames[0] > _COMP_MAX_FRAMES:
                                self._comp_active[0] = False
                                self._comp_frames[0] = 0
                                self._comp_cooldown[0] = True
                    elif _raw_mag > _COMP_ON:
                        self._comp_active[0] = True
                        self._comp_frames[0] = 1
                    if self._comp_active[0]:
                        # 在途发送按发送时的"响度"衰减(简化:按剩余误差比例扣)
                        _cmag = math.hypot(_comp_x, _comp_y)
                        if _cmag > _raw_mag > 0:
                            _s = _raw_mag / _cmag
                            _comp_x *= _s
                            _comp_y *= _s
                raw_dx = _raw_dx - _comp_x
                raw_dy = _raw_dy - _comp_y
                smoothing = float(self.hu.get("smoothing", 0.35))
                _hu_sens = float(self.hu.get("sensitivity", 1.0))
                dz_base = float(self.hu.get("deadzone", 3.0))
                box_h = 0.0
                if self.locked_box[0] is not None:
                    box_h = self.locked_box[0][3] - self.locked_box[0][1]
                box_h = max(0.0, min(box_h, 640.0))
                deadzone_px = dz_base * (0.5 + box_h / 640.0)
                dx = raw_dx * smoothing
                dy = raw_dy * smoothing
                diag_gain = 1.0
                if self.box_scale_enabled and self.locked_box[0] is not None:
                    _fake = {"bbox": self.locked_box[0]}
                    diag_gain = round(self._box_scale_gain(_fake), 2)
                    sens = smoothing * diag_gain * _hu_sens
                    dx *= diag_gain
                    dy *= diag_gain
                else:
                    sens = smoothing * _hu_sens
                gain_clamped = False
                if self.max_total_gain > 0 and sens > self.max_total_gain:
                    _g = self.max_total_gain / sens
                    dx *= _g
                    dy *= _g
                    sens = self.max_total_gain
                    gain_clamped = True
                out["gain_clamped"] = gain_clamped
                # ── 远距追赶增益(分段P) ──
                # 误差小 → 基础增益(稳定区);误差大 → 线性放大到 boost_max(追赶区)。
                if self.boost_max > 1.0 and 0 < self.boost_lo_px < self.boost_hi_px:
                    _emag = math.hypot(raw_dx, raw_dy)
                    if _emag > self.boost_lo_px:
                        _t = min(1.0, (_emag - self.boost_lo_px) / (self.boost_hi_px - self.boost_lo_px))
                        _k = 1.0 + (self.boost_max - 1.0) * _t
                        dx *= _k
                        dy *= _k

                # ── 粘滞(近区阻尼) + 缓动起步 ──
                _mult = 1.0
                if self.ease_frames > 0 and self._ease_left > 0:
                    _mult *= self.ease_min_gain + (1.0 - self.ease_min_gain) * (
                        1.0 - self._ease_left / float(max(1, self.ease_frames)))
                    self._ease_left -= 1
                if self.sticky_enabled and box_h > 0.0:
                    _smag = math.hypot(raw_dx, raw_dy)
                    _sr = self.sticky_radius_ratio * box_h
                    if 0.0 < _smag < _sr:
                        # 越接近框中心阻尼越强(线性过渡到 sticky_gain), 出区即恢复满增益
                        _mult *= self.sticky_gain + (1.0 - self.sticky_gain) * (_smag / _sr)
                if _mult != 1.0:
                    dx *= _mult
                    dy *= _mult

            # 死区判定(原始误差)
            out["raw_dx"] = raw_dx
            out["raw_dy"] = raw_dy
            out["deadzone_px"] = deadzone_px
            out["sens"] = sens
            out["diag_gain"] = diag_gain
            out["lead"] = (_lead_x, _lead_y)
            if abs(raw_dx) < deadzone_px and abs(raw_dy) < deadzone_px:
                out["in_deadzone"] = True
                self._last_send_time = self._now  # 死区内也算一个节流周期
            elif abs(dx) > 0.01 or abs(dy) > 0.01:
                # 换向阻尼 + 首步上限(对齐 main.py 1935-1946)
                _damp_x = _damp_y = 1.0
                if (self._last_net_move[0] > 0 and dx < 0) or (self._last_net_move[0] < 0 and dx > 0):
                    _damp_x = self.reversal_damp
                if (self._last_net_move[1] > 0 and dy < 0) or (self._last_net_move[1] < 0 and dy > 0):
                    _damp_y = self.reversal_damp
                dx *= _damp_x
                dy *= _damp_y
                cap_hint = self.first_step_cap if (self._just_locked[0] and self.first_step_cap > 0) else None
                self._just_locked[0] = False
                out["dx"] = dx
                out["dy"] = dy
                out["cap_hint"] = cap_hint
            else:
                out["in_deadzone"] = True
                self._last_send_time = self._now
        return out
