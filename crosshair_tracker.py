# -*- coding: utf-8 -*-
"""红点瞄准参考的时间连续性状态机，不负责图像分割。"""

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class CrosshairResult:
    x: float
    y: float
    status: str
    trusted: bool
    jump_x: float = 0.0
    jump_y: float = 0.0


class CrosshairTracker:
    """验证红点候选，并在短时误检/丢失期间保持稳定参考点。

    帧调度语义(unit.frame_schedule):
      fixed — confirm/hold/return 全部按"帧数"计数，行为与历史版本逐位一致；
      asap  — 同样的状态机按真实时间运行，帧数阈值 × reference_frame_ms 换算成
              秒(候选首帧已计入，因此 confirm_frames=2 还需约22ms)，EMA alpha 按
              alpha_dt = 1-(1-alpha_ref)^(dt/ref) 归一，
              使状态持续时间基本不受实际 YOLO FPS 影响。
    """

    def __init__(self, max_center_offset=45.0, max_jump=12.0,
                 confirm_frames=2, hold_frames=3, ema_alpha=0.35,
                 fallback_decay=0.10, shot_vertical_jump=60.0,
                 shot_horizontal_jump=15.0, shot_center_offset=85.0,
                 shot_ema_alpha=1.0, return_frames=4,
                 return_max_step=24.0, frame_schedule="fixed",
                 reference_frame_ms=22.0):
        self.max_center_offset = max(1.0, float(max_center_offset))
        self.max_jump = max(1.0, float(max_jump))
        self.confirm_frames = max(1, int(confirm_frames))
        self.hold_frames = max(0, int(hold_frames))
        self.return_frames = max(1, int(return_frames))
        self.return_max_step = max(1.0, float(return_max_step))
        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self.fallback_decay = min(1.0, max(0.0, float(fallback_decay)))
        self.shot_vertical_jump = max(self.max_jump, float(shot_vertical_jump))
        self.shot_horizontal_jump = max(self.max_jump, float(shot_horizontal_jump))
        self.shot_center_offset = max(self.max_center_offset, float(shot_center_offset))
        self.shot_ema_alpha = min(1.0, max(0.0, float(shot_ema_alpha)))
        self.frame_schedule = str(frame_schedule).lower()
        if self.frame_schedule not in ("fixed", "asap"):
            self.frame_schedule = "fixed"
        self.reference_dt = max(0.002, float(reference_frame_ms) / 1000.0)
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.ever = False
        self.trusted = False
        self.lost = 0
        self._candidate = None
        self._candidate_count = 0
        self._reacquiring = False
        self._last_trusted_at = None
        # asap 时间状态
        self._lost_s = 0.0            # 连续丢失累计秒数(hold/return 门控)
        self._candidate_since = None  # 当前候选连续稳定起点
        self._last_now_s = None       # 上次 update 时间戳(推 dt 用)

    @property
    def preferred_position(self):
        """上一可信红点位置(供 _detect_crosshair_dot 的时间先验); 从未可信时 None。"""
        return (self.x, self.y) if self.ever else None

    # ── 时间归一化 helpers(fixed 下全部退化为帧语义) ──
    def _dt(self, now_s):
        """本次与上次观测的真实间隔; 首帧用参考周期。"""
        if self._last_now_s is None:
            return self.reference_dt
        return max(1e-4, min(0.5, float(now_s) - self._last_now_s))

    def _k(self, dt):
        """asap: 实际周期/参考周期; fixed: 1(不参与时间换算)。"""
        if self.frame_schedule != "asap":
            return 1.0
        return max(1e-4, dt / self.reference_dt)

    def _norm_alpha(self, alpha_ref, dt):
        if self.frame_schedule != "asap":
            return alpha_ref
        k = self._k(dt)
        if k == 1.0:
            return alpha_ref
        return 1.0 - (1.0 - alpha_ref) ** k

    def _fallback(self, center, status, mode, dt):
        if not self.ever:
            return CrosshairResult(center[0], center[1], status, False)
        if mode == "dot_fallback":
            decay = self._norm_alpha(self.fallback_decay, dt)
            self.x += (center[0] - self.x) * decay
            self.y += (center[1] - self.y) * decay
        return CrosshairResult(self.x, self.y, status, False)

    def _bounded_return(self, center, reason, dt=0.0, now_s=None):
        """短暂保持旧红点后，在有限帧(或有限时长)内受限回到屏幕中心。"""
        if self.frame_schedule == "asap":
            return self._bounded_return_time(center, reason, dt)
        self.lost += 1
        if self.lost <= self.hold_frames:
            status = (reason if reason.startswith("confirming(")
                      else f"hold_{reason}({self.lost}/{self.hold_frames})")
            return CrosshairResult(
                self.x, self.y, status, False,
            )

        self.trusted = False
        self._reacquiring = True
        return_index = self.lost - self.hold_frames
        if return_index <= self.return_frames:
            remaining = self.return_frames - return_index + 1
            dx = (center[0] - self.x) / remaining
            dy = (center[1] - self.y) / remaining
            distance = math.hypot(dx, dy)
            if distance > self.return_max_step:
                scale = self.return_max_step / distance
                dx *= scale
                dy *= scale
            self.x += dx
            self.y += dy
            if return_index == self.return_frames:
                # 正常候选受中心偏移限制，四步足以返回；末步消除浮点残差。
                self.x, self.y = center
            status = (reason if reason.startswith("confirming(")
                      else f"return_{reason}({return_index}/{self.return_frames})")
            return CrosshairResult(self.x, self.y, status, False)

        self.x, self.y = center
        return CrosshairResult(
            self.x, self.y, f"fallback_center_{reason}", False,
        )

    def _bounded_return_time(self, center, reason, dt):
        """asap 版 _bounded_return：hold=2帧≈44ms 保持，return=4帧≈88ms 受限回归。

        每次观测按 dt/剩余时长 的比例向中心推进(单步限幅同步按 dt/ref 缩放)，
        时长耗尽时精确吸附中心 —— 与 fixed 版"逐步回归、末步吸附"语义对齐。
        """
        hold_s = self.hold_frames * self.reference_dt
        return_s = self.return_frames * self.reference_dt
        self.lost += 1
        self._lost_s += dt
        if self._lost_s <= hold_s + 1e-9:
            index = min(self.lost, self.hold_frames)
            status = (reason if reason.startswith("confirming(")
                      else f"hold_{reason}({index}/{self.hold_frames})")
            return CrosshairResult(self.x, self.y, status, False)

        self.trusted = False
        self._reacquiring = True
        elapsed = self._lost_s - hold_s
        if elapsed <= return_s + 1e-9:
            remaining = max(dt, return_s - (elapsed - dt))
            frac = min(1.0, dt / remaining)
            dx = (center[0] - self.x) * frac
            dy = (center[1] - self.y) * frac
            step_cap = self.return_max_step * self._k(dt)
            distance = math.hypot(dx, dy)
            if distance > step_cap:
                scale = step_cap / distance
                dx *= scale
                dy *= scale
            self.x += dx
            self.y += dy
            if elapsed >= return_s - 1e-9:
                # 时长耗尽：末步消除浮点残差(对齐 fixed 版末步吸附)。
                self.x, self.y = center
            index = min(self.return_frames,
                        max(1, int(math.ceil(elapsed / self.reference_dt))))
            status = (reason if reason.startswith("confirming(")
                      else f"return_{reason}({index}/{self.return_frames})")
            return CrosshairResult(self.x, self.y, status, False)

        self.x, self.y = center
        return CrosshairResult(
            self.x, self.y, f"fallback_center_{reason}", False,
        )

    def update(self, candidate, center, mode="dot_fallback", shot_active=False,
               now_s=None):
        now_s = time.perf_counter() if now_s is None else float(now_s)
        center = (float(center[0]), float(center[1]))
        dt = self._dt(now_s)
        prev_now = self._last_now_s
        self._last_now_s = now_s
        if mode == "center":
            return CrosshairResult(center[0], center[1], "center", False)

        valid = candidate is not None
        reject = "missing"
        if valid:
            candidate = (float(candidate[0]), float(candidate[1]))
            center_limit = self.shot_center_offset if shot_active else self.max_center_offset
            if math.hypot(candidate[0] - center[0], candidate[1] - center[1]) \
                    > center_limit:
                valid = False
                reject = "center"

        # asap: 候选确认按"连续稳定时长 ≥ confirm_frames×参考周期"判定。
        confirm_ok = True
        if self.frame_schedule == "asap" and valid:
            if self._candidate is not None and math.hypot(
                    candidate[0] - self._candidate[0],
                    candidate[1] - self._candidate[1]) <= self.max_jump:
                if self._candidate_since is None:
                    self._candidate_since = prev_now
            else:
                self._candidate_since = now_s
            confirm_s = max(0, self.confirm_frames - 1) * self.reference_dt
            confirm_ok = (prev_now is not None and
                          (now_s - self._candidate_since) >=
                          confirm_s - 1e-9)

        if self.trusted:
            if not valid and reject == "missing":
                return self._bounded_return(center, "missing", dt)
            jump_x = candidate[0] - self.x if valid else 0.0
            jump_y = candidate[1] - self.y if valid else 0.0
            # 实战日志中的首发后坐力表现为红点近乎竖直向上跳 25~60px。
            # 只在开火窗口放行此单向模式；横向大跳仍按误检处理。
            shot_jump = bool(
                shot_active and valid
                and math.hypot(jump_x, jump_y) > self.max_jump
                and jump_y <= -3.0
                and -jump_y <= self.shot_vertical_jump
                and abs(jump_x) <= self.shot_horizontal_jump
            )
            if shot_jump:
                shot_alpha = self._norm_alpha(self.shot_ema_alpha, dt)
                self.x += jump_x * shot_alpha
                self.y += jump_y * shot_alpha
                self.lost = 0
                self._lost_s = 0.0
                self._last_trusted_at = now_s
                return CrosshairResult(
                    self.x, self.y, "shot_trusted", True, jump_x, jump_y,
                )
            if valid and math.hypot(candidate[0] - self.x, candidate[1] - self.y) \
                    <= self.max_jump:
                ema = self._norm_alpha(self.ema_alpha, dt)
                self.x += (candidate[0] - self.x) * ema
                self.y += (candidate[1] - self.y) * ema
                self.lost = 0
                self._lost_s = 0.0
                self._last_trusted_at = now_s
                return CrosshairResult(self.x, self.y, "trusted", True)
            if valid:
                reject = "jump"
            self.trusted = False
            self._reacquiring = True
            self._candidate = None
            self._candidate_count = 0
            self._candidate_since = None
            return self._bounded_return(center, f"{reject}_reacquire", dt)

        if valid:
            if self._candidate is not None and math.hypot(
                    candidate[0] - self._candidate[0],
                    candidate[1] - self._candidate[1]) <= self.max_jump:
                self._candidate_count += 1
            else:
                self._candidate_count = 1
                if self.frame_schedule == "asap":
                    self._candidate_since = now_s
            self._candidate = candidate
            confirmed = (confirm_ok if self.frame_schedule == "asap"
                         else self._candidate_count >= self.confirm_frames)
            if confirmed:
                ema = self._norm_alpha(self.ema_alpha, dt)
                if self.ever and not self._reacquiring:
                    self.x += (candidate[0] - self.x) * ema
                    self.y += (candidate[1] - self.y) * ema
                else:
                    self.x, self.y = candidate
                self.ever = True
                self.trusted = True
                self._reacquiring = False
                self.lost = 0
                self._lost_s = 0.0
                self._last_trusted_at = now_s
                self._candidate = None
                self._candidate_count = 0
                self._candidate_since = None
                return CrosshairResult(self.x, self.y, "trusted", True)
            status = (f"confirming({self._candidate_count}/{self.confirm_frames})"
                      if self.frame_schedule == "fixed" else
                      f"confirming({max(0.0, (now_s - (self._candidate_since or now_s))) * 1000.0:.0f}ms/"
                      f"{self.confirm_frames * self.reference_dt * 1000.0:.0f}ms)")
        else:
            self._candidate = None
            self._candidate_count = 0
            self._candidate_since = None
            if self.ever:
                return self._bounded_return(center, f"{reject}_reacquire", dt)
            status = f"reject_{reject}" if reject != "missing" else "missing"

        if self.ever:
            return self._bounded_return(center, status, dt)
        return self._fallback(center, status, mode, dt)
