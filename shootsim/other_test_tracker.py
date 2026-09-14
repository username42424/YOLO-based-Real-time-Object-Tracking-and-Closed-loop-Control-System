# -*- coding: utf-8 -*-
"""Simulation adapter for the control logic in ``other test/main.py``.

The original project combines YOLO, mss, Win32 input, and its control loop in
one module. This adapter keeps its target-selection, sticky-lock, and PID
semantics, while accepting simulator detections and returning mouse counts.
It deliberately does not import the original executable module or touch the
real screen/mouse.
"""

import math


class OtherTestTracker:
    """Run the other-test target selection and PID controller in sim."""

    name = "other_test"

    def __init__(self, cfg, sw, sh, rng=None):
        self.sw = float(sw)
        self.sh = float(sh)
        self.center = (self.sw / 2.0, self.sh / 2.0)
        self.aim_enabled = bool(cfg.get("aim_enabled", True))
        self.aim_head = bool(cfg.get("aim_head", True))
        self.aim_fov = max(1.0, float(cfg.get("aim_fov", 200.0)))
        self.kp = float(cfg.get("pid_kp", 0.60))
        self.ki = float(cfg.get("pid_ki", 0.0))
        self.kd = float(cfg.get("pid_kd", 0.005))
        self.lock_stickiness = max(
            0.0, float(cfg.get("lock_stickiness", 40.0)))
        self.lock_tolerance_frames = max(
            0, int(cfg.get("lock_tolerance_frames", 10)))
        self.deadzone = max(0.0, float(cfg.get("deadzone", 2.0)))
        self.reset()

    def reset(self):
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_lock_pos = None
        self.last_lock_frame_idx = 0
        self.frame_count = 0
        self.last_target_type = "NONE"
        self.last_class_missing = False
        self.class_missing_count = 0

    def status(self):
        """返回供 Shooter 使用的稳定公共状态，不暴露 Shooter 所需的私有字段。"""
        target_type = self.last_target_type
        return {
            "has_target": target_type in ("HEAD", "BODY", "LOCKED"),
            "locked": target_type in ("HEAD", "BODY", "LOCKED"),
            "waiting": target_type == "WAITING",
            "target_type": target_type,
            "class_missing": self.last_class_missing,
        }

    @staticmethod
    def _point(box):
        x1, y1, x2, y2 = map(float, box[:4])
        cls = int(box[4]) if len(box) > 4 else 0
        box_h = y2 - y1
        cy = y1 + box_h * (0.6 if cls == 1 else 0.2)
        return (int((x1 + x2) * 0.5), int(cy), cls)

    def _in_fov(self, point):
        return math.hypot(point[0] - self.center[0],
                          point[1] - self.center[1]) < self.aim_fov

    def _reset_pid(self):
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0

    def _pid(self, target_x, target_y, dt):
        dt = max(1e-6, float(dt))
        error_x = target_x - self.center[0]
        error_y = target_y - self.center[1]
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        output_x = (self.kp * error_x + self.ki * self.integral_x
                    + self.kd * derivative_x)
        output_y = (self.kp * error_y + self.ki * self.integral_y
                    + self.kd * derivative_y)
        if abs(error_x) < self.deadzone and abs(error_y) < self.deadzone:
            return 0.0, 0.0
        return output_x, output_y

    def update(self, boxes, dt):
        """Return simulator mouse counts for the current detection list."""
        self.frame_count += 1
        self.last_class_missing = False
        candidates = []
        for box in boxes or []:
            if len(box) <= 4:
                self.last_class_missing = True
                self.class_missing_count += 1
            point = self._point(box)
            if self._in_fov(point):
                candidates.append(point)

        target = None
        target_type = "NONE"
        frames_since_lock = self.frame_count - self.last_lock_frame_idx
        use_sticky = (
            self.aim_enabled and self.last_lock_pos is not None
            and frames_since_lock < self.lock_tolerance_frames)

        if use_sticky:
            sticky = [p for p in candidates
                      if math.hypot(p[0] - self.last_lock_pos[0],
                                   p[1] - self.last_lock_pos[1])
                      < self.lock_stickiness]
            if sticky:
                target = min(
                    sticky,
                    key=lambda p: math.hypot(
                        p[0] - self.last_lock_pos[0],
                        p[1] - self.last_lock_pos[1]))
                target_type = "LOCKED"
            else:
                target_type = "WAITING"
        else:
            heads = [p for p in candidates if p[2] == 1 and self.aim_head]
            bodies = [p for p in candidates if p[2] != 1]
            if heads:
                target = min(heads, key=lambda p: math.hypot(
                    p[0] - self.center[0], p[1] - self.center[1]))
                target_type = "HEAD"
            elif bodies:
                target = min(bodies, key=lambda p: math.hypot(
                    p[0] - self.center[0], p[1] - self.center[1]))
                target_type = "BODY"

        self.last_target_type = target_type
        if target is not None and self.aim_enabled:
            self.last_lock_pos = (target[0], target[1])
            self.last_lock_frame_idx = self.frame_count
            return self._pid(target[0], target[1], dt)
        if target_type == "WAITING" and self.aim_enabled:
            self.prev_error_x = 0.0
            self.prev_error_y = 0.0
            return 0.0, 0.0

        self._reset_pid()
        self.last_lock_pos = None
        self.last_lock_frame_idx = 0
        return 0.0, 0.0
