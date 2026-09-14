# -*- coding: utf-8 -*-
"""目标移动策略（模块化，便于替换）。

每个策略实现 BaseMover：reset(tx, ty) 重置内部状态；update(dt, tx, ty) 返回新世界坐标。
默认 random4：上下左右随机，方向变化间隔随机，边界反弹。
"""
import math
import random


class BaseMover:
    name = "base"

    def __init__(self, cfg, world_w, world_h, rng):
        self.speed = float(cfg.get("speed", 120.0))
        self.world_w = world_w
        self.world_h = world_h
        self.bounce = bool(cfg.get("bounce", True))
        self.rng = rng
        self.vx = self.vy = 0.0

    def reset(self, tx, ty):
        self.vx = self.vy = 0.0

    def _bounce(self, tx, ty):
        """越界反弹（可选）。"""
        if not self.bounce:
            return tx, ty
        tx = max(0.0, min(self.world_w, tx))
        ty = max(0.0, min(self.world_h, ty))
        return tx, ty

    def update(self, dt, tx, ty):
        return tx, ty


class Random4Mover(BaseMover):
    """上下左右四方向随机移动；每 [dir_change_min, dir_change_max] 秒随机换方向。"""
    name = "random4"
    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.t_min = float(cfg.get("dir_change_min", 0.4))
        self.t_max = float(cfg.get("dir_change_max", 1.2))
        self._next_change = 0.0

    def reset(self, tx, ty):
        super().reset(tx, ty)
        self._pick_dir()
        self._next_change = self.rng.uniform(self.t_min, self.t_max)

    def _pick_dir(self):
        dx, dy = self.rng.choice(self.DIRS)
        self.vx, self.vy = dx * self.speed, dy * self.speed

    def update(self, dt, tx, ty):
        self._next_change -= dt
        if self._next_change <= 0:
            self._pick_dir()
            self._next_change = self.rng.uniform(self.t_min, self.t_max)
        tx += self.vx * dt
        ty += self.vy * dt
        if self.bounce and (tx <= 0 or tx >= self.world_w or ty <= 0 or ty >= self.world_h):
            self._pick_dir()
            self._next_change = self.rng.uniform(self.t_min, self.t_max)
        return self._bounce(tx, ty)


class Random8Mover(Random4Mover):
    """八方向随机移动（含斜向）。"""
    name = "random8"

    def _pick_dir(self):
        ang = self.rng.uniform(0, 2 * math.pi)
        self.vx, self.vy = math.cos(ang) * self.speed, math.sin(ang) * self.speed


class SineMover(BaseMover):
    """正弦横移 + 正弦纵向（平滑可预测轨迹）。"""
    name = "sine"

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.phase_x = self.rng.uniform(0, 2 * math.pi)
        self.phase_y = self.rng.uniform(0, 2 * math.pi)
        self.wx = self.rng.uniform(0.5, 2.0)
        self.wy = self.rng.uniform(0.5, 2.0)
        self.cx0 = self.cy0 = None

    def reset(self, tx, ty):
        self.cx0, self.cy0 = tx, ty

    def update(self, dt, tx, ty):
        self.phase_x += self.wx * dt
        self.phase_y += self.wy * dt
        tx = self.cx0 + math.sin(self.phase_x) * self.speed / max(self.wx, 1e-6) * 0.6
        ty = self.cy0 + math.sin(self.phase_y) * self.speed / max(self.wy, 1e-6) * 0.6
        return self._bounce(tx, ty)


class JitterMover(BaseMover):
    """平滑随机游走（速度缓慢随机漂移）。"""
    name = "jitter"

    def update(self, dt, tx, ty):
        self.vx += self.rng.gauss(0, 1) * self.speed * 0.5 * dt * 8
        self.vy += self.rng.gauss(0, 1) * self.speed * 0.5 * dt * 8
        maxv = self.speed
        self.vx = max(-maxv, min(maxv, self.vx))
        self.vy = max(-maxv, min(maxv, self.vy))
        tx += self.vx * dt
        ty += self.vy * dt
        if self.bounce and (tx <= 0 or tx >= self.world_w):
            self.vx = -self.vx
        if self.bounce and (ty <= 0 or ty >= self.world_h):
            self.vy = -self.vy
        return self._bounce(tx, ty)


class TeleportMover(BaseMover):
    """随机瞬移（最难追踪，模拟短时目标）。"""
    name = "teleport"

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.t_min = float(cfg.get("dir_change_min", 0.8))
        self.t_max = float(cfg.get("dir_change_max", 2.0))
        self._next = 0.0

    def reset(self, tx, ty):
        self._next = self.rng.uniform(self.t_min, self.t_max)

    def update(self, dt, tx, ty):
        self._next -= dt
        if self._next <= 0:
            tx = self.rng.uniform(100, self.world_w - 100)
            ty = self.rng.uniform(100, self.world_h - 100)
            self._next = self.rng.uniform(self.t_min, self.t_max)
        return tx, ty


class ChaseMover(BaseMover):
    """朝相机冲（模拟敌人冲脸）。"""
    name = "chase"

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.cam_ref = None

    def set_camera(self, cam_x, cam_y):
        self.cam_ref = (cam_x, cam_y)

    def update(self, dt, tx, ty):
        if self.cam_ref is not None:
            cx, cy = self.cam_ref
            d = math.hypot(cx - tx, cy - ty)
            if d > 40:
                tx += (cx - tx) / d * self.speed * dt
                ty += (cy - ty) / d * self.speed * dt
        return tx, ty


class FixedDirMover(BaseMover):
    """固定方向持续移动：随机朝一个方向匀速前进，不频繁换向。

    用户要求：目标随机朝某一方向移动，而不是短时间内上下左右乱跑
    （乱跑会导致目标基本原地不动）。本策略只在碰世界边界时反弹换向，
    否则一直沿初始随机方向移动——模拟真实玩家/敌人的直线运动。
    """
    name = "fixeddir"

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.ang = 0.0

    def reset(self, tx, ty):
        super().reset(tx, ty)
        # 随机朝一个方向（任意角度）
        self.ang = self.rng.uniform(0, 2 * math.pi)
        # 随机速度 0~max_speed（config 里 speed 作为上限）
        max_speed = float(self.speed)
        self.cur_speed = self.rng.uniform(0, max_speed)
        self.vx = math.cos(self.ang) * self.cur_speed
        self.vy = math.sin(self.ang) * self.cur_speed

    def update(self, dt, tx, ty):
        tx += self.vx * dt
        ty += self.vy * dt
        if self.bounce:
            # 碰边界反弹（只翻转越界轴）
            if tx <= 0 or tx >= self.world_w:
                self.vx = -self.vx
                tx = max(0.0, min(self.world_w, tx))
            if ty <= 0 or ty >= self.world_h:
                self.vy = -self.vy
                ty = max(0.0, min(self.world_h, ty))
        return tx, ty


class StrafeMover(BaseMover):
    """左右横移（螃蟹步）：水平方向来回走，随机 0.25~0.8s 反向一次，
    叠加小幅纵向漂移。模拟真人左右晃动规避准星。
    """
    name = "strafe"

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.t_min = float(cfg.get("dir_change_min", 0.25))
        self.t_max = float(cfg.get("dir_change_max", 0.8))
        self.vy_amp = float(self.speed) * 0.25   # 纵向漂移上限
        self._next = 0.0
        self.dir = 1.0
        self.vy = 0.0

    def reset(self, tx, ty):
        super().reset(tx, ty)
        self.dir = self.rng.choice([-1.0, 1.0])
        self.vx = self.dir * self.speed
        self.vy = self.rng.uniform(-self.vy_amp, self.vy_amp)
        self._next = self.rng.uniform(self.t_min, self.t_max)

    def update(self, dt, tx, ty):
        self._next -= dt
        if self._next <= 0:
            self.dir = -self.dir
            self.vx = self.dir * self.speed
            self.vy = self.rng.uniform(-self.vy_amp, self.vy_amp)
            self._next = self.rng.uniform(self.t_min, self.t_max)
        tx += self.vx * dt
        ty += self.vy * dt
        if self.bounce:
            if tx <= 0 or tx >= self.world_w:
                self.dir = -self.dir
                self.vx = self.dir * self.speed
                tx = max(0.0, min(self.world_w, tx))
            if ty <= 0 or ty >= self.world_h:
                self.vy = -self.vy
                ty = max(0.0, min(self.world_h, ty))
        return self._bounce(tx, ty)


class JumpStrafeMover(StrafeMover):
    """跳跃横移（真人最常用规避）：左右螃蟹步 + 周期性垂直跳跃。

    跳跃用抛物线：起跳速度 jump_speed 向上（ty 减小），重力 gravity 拉回，
    落到起跳基准高度后落地。叠加在水平横移之上，屏幕上看是"左右晃 + 上下跳"。
    """
    name = "jumpstrafe"

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        self.jump_min = float(cfg.get("jump_interval_min", 0.8))
        self.jump_max = float(cfg.get("jump_interval_max", 2.2))
        self.jump_speed = float(self.speed) * 1.3    # 起跳垂直速度
        self.gravity = float(self.speed) * 3.2       # 重力（把跳拉回）
        self.vjump = 0.0
        self.ground_ty = None
        self._jump_next = 0.0

    def reset(self, tx, ty):
        super().reset(tx, ty)
        self.vjump = 0.0
        self.ground_ty = ty
        self._jump_next = self.rng.uniform(self.jump_min, self.jump_max)

    def update(self, dt, tx, ty):
        # 水平：复用 strafe 的横移逻辑（不调用 super，避免重复改 ty）
        self._next -= dt
        if self._next <= 0:
            self.dir = -self.dir
            self.vx = self.dir * self.speed
            self.vy = self.rng.uniform(-self.vy_amp, self.vy_amp)
            self._next = self.rng.uniform(self.t_min, self.t_max)
        tx += self.vx * dt
        # 垂直：跳跃抛物线 or 地面小幅漂移
        self._jump_next -= dt
        if self._jump_next <= 0 and abs(self.vjump) < 1e-6:
            self.vjump = -self.jump_speed   # 向上起跳（ty 减小）
            self.ground_ty = ty
            self._jump_next = self.rng.uniform(self.jump_min, self.jump_max)
        if abs(self.vjump) > 1e-6:
            ty += self.vjump * dt
            self.vjump += self.gravity * dt  # 重力下拉（ty 增大）
            if self.ground_ty is not None and ty >= self.ground_ty:
                ty = self.ground_ty
                self.vjump = 0.0
        else:
            ty += self.vy * dt
            self.ground_ty = ty
        if self.bounce:
            if tx <= 0 or tx >= self.world_w:
                self.dir = -self.dir
                self.vx = self.dir * self.speed
                tx = max(0.0, min(self.world_w, tx))
            ty = max(0.0, min(self.world_h, ty))
            self.ground_ty = max(0.0, min(self.world_h, self.ground_ty))
        return tx, ty


class RealFitMover(BaseMover):
    """实战日志拟合 mover（"realfit"）：从 fitted_motion.json 的经验分布采样。

    模型（分布全部来自 logs/*.log 全量实战日志的世界轨迹拟合，见 fit_real_motion.py）：
      - 状态机：移动段(时长~strafe_hold_s, 速度|vx|~leg_speed_px_s, 方向随机翻转,
        纵向漂移~leg_vy_px_s) ↔ 静止段(时长~idle_run_s, 概率按实战静止占比 21.7% 折算)；
      - vy 连续化：换段时纵向速度按 tau≈0.08s 指数趋近新目标（真实目标无速度瞬移）；
      - 跳跃：间隔~jump_interval_s(峰间口径, 落地后倒计时扣除滞空时间)，
        起跳速度~jump_peak_vy，重力~jump_gravity(逐跳采样, 实战中位≈3000 px/s²)；
      - 越界反弹。
    """

    name = "realfit"
    _FIT_CACHE = {}

    def __init__(self, cfg, world_w, world_h, rng):
        super().__init__(cfg, world_w, world_h, rng)
        import json as _json
        import os as _os
        path = cfg.get("realfit_motion", _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "fitted_motion.json"))
        if path not in RealFitMover._FIT_CACHE:
            with open(path, encoding="utf-8") as f:
                RealFitMover._FIT_CACHE[path] = _json.load(f)
        fitdoc = RealFitMover._FIT_CACHE[path]
        self.fit = fitdoc["samples"]
        self.idle_frac = float(fitdoc.get("idle_frac_below30", 0.2))
        self.jump_rate = float(fitdoc.get("jump_rate_unbiased", 0.6))
        self.speed_scale = float(cfg.get("realfit_speed_scale", 1.0))
        self.vy_tau = float(cfg.get("realfit_vy_tau", 0.08))
        self.vy_clamp = float(cfg.get("realfit_vy_clamp", 200.0))
        self.p_reverse = float(cfg.get("realfit_p_reverse", 0.9))
        self._idle_prob = self.idle_frac / max(1e-6, 1.0 - self.idle_frac)
        self._hold = 0.0
        self._idle = False
        self._dir = 1.0
        self._vx = 0.0
        self._vy = 0.0
        self._vy_tgt = 0.0
        self._jump_next = 0.0
        self._vjump = 0.0
        self._grav = 3000.0
        self._ground_ty = None

    @staticmethod
    def _sample(rng, xs):
        return float(xs[rng.randrange(len(xs))]) if len(xs) else 0.0

    def _new_state(self):
        """切到下一段：按概率进入静止段或新移动段。"""
        if self.rng.random() < self._idle_prob:
            self._idle = True
            self._vx = 0.0
            self._vy_tgt = 0.0
            self._hold = max(0.03, self._sample(
                self.rng, self.fit.get("idle_run_s", [0.3])))
        else:
            self._idle = False
            self._dir = -self._dir if self.rng.random() < self.p_reverse else self._dir
            speed = self._sample(self.rng, self.fit.get("leg_speed_px_s", [self.speed])) \
                * self.speed_scale
            self._vx = self._dir * speed
            vy_t = self._sample(self.rng, self.fit.get("leg_vy_px_s", [0.0])) \
                * self.speed_scale
            self._vy_tgt = max(-self.vy_clamp, min(self.vy_clamp, vy_t))
            self._hold = max(0.03, self._sample(
                self.rng, self.fit.get("strafe_hold_s", [0.3])))

    def reset(self, tx, ty):
        super().reset(tx, ty)
        self._dir = 1.0 if self.rng.random() < 0.5 else -1.0
        self._vy = 0.0
        self._new_state()
        self._vjump = 0.0
        self._ground_ty = ty
        self._jump_next = max(0.05, -math.log(
            1.0 - min(0.999, self.rng.random())) / max(1e-3, self.jump_rate))

    def update(self, dt, tx, ty):
        # 段推进（移动段/静止段）
        self._hold -= dt
        if self._hold <= 0:
            self._new_state()
        tx += self._vx * dt
        # vy 连续趋近目标（跳跃期间由抛物线接管）
        if abs(self._vjump) < 1e-6:
            a = min(1.0, dt / max(1e-3, self.vy_tau))
            self._vy += (self._vy_tgt - self._vy) * a
            ty += self._vy * dt
            self._ground_ty = ty
        # 跳跃：倒计时用无偏峰值频率的指数分布(均值 1/rate)，落地后再计时
        self._jump_next -= dt
        if self._jump_next <= 0 and abs(self._vjump) < 1e-6:
            peak = self._sample(self.rng, self.fit.get("jump_peak_vy", [350.0])) \
                * self.speed_scale
            grav = self._sample(self.rng, self.fit.get("jump_gravity", [3000.0]))
            if grav <= 0:
                grav = 3000.0
            self._vjump = -peak
            self._grav = grav
            self._ground_ty = ty
            airtime = 2.0 * peak / grav
            gap = -math.log(1.0 - min(0.999, self.rng.random())) / max(1e-3, self.jump_rate)
            self._jump_next = max(0.05, gap + airtime)
        if abs(self._vjump) > 1e-6:
            ty += self._vjump * dt
            self._vjump += self._grav * dt
            if self._ground_ty is not None and ty >= self._ground_ty:
                ty = self._ground_ty
                self._vjump = 0.0
                self._vy = 0.0
        if self.bounce:
            if tx <= 0 or tx >= self.world_w:
                self._dir = -self._dir
                self._vx = -self._vx
                tx = max(0.0, min(self.world_w, tx))
            ty = max(0.0, min(self.world_h, ty))
            self._ground_ty = max(0.0, min(self.world_h, self._ground_ty))
        return tx, ty


MOVERS = {
    "random4": Random4Mover,
    "random8": Random8Mover,
    "sine": SineMover,
    "jitter": JitterMover,
    "teleport": TeleportMover,
    "chase": ChaseMover,
    "fixeddir": FixedDirMover,
    "strafe": StrafeMover,
    "jumpstrafe": JumpStrafeMover,
    "realfit": RealFitMover,
}


def create_mover(name, target_cfg, world_w, world_h, rng):
    cls = MOVERS.get(name, Random4Mover)
    return cls(target_cfg, world_w, world_h, rng)
