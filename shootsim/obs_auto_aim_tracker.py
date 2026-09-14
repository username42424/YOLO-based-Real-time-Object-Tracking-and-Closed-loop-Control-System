"""Deterministic port of the local ``main_aim.py`` aiming logic.

The repository also contains a cloud/humanized controller.  This adapter uses
the local controller's target score and distance-dependent movement only, so
the simulator comparison remains deterministic and accuracy-oriented.
"""

from __future__ import annotations

import math


class ObsAutoAimTracker:
    """Port of ``160037lk/obs-auto-aim``'s local FastAimController path."""

    name = "obs_auto_aim"

    def __init__(self, cfg, sw, sh, rng=None):
        self.sw = float(sw)
        self.sh = float(sh)
        self.cx = self.sw / 2.0
        self.cy = self.sh / 2.0
        target_type = cfg.get("target_type", "head")
        self.target_type = 1 if str(target_type).lower() in ("head", "1") else 0
        self.fov = max(1.0, float(cfg.get("aim_fov", 300.0)))
        self.base_sensitivity = float(cfg.get("base_sensitivity", 1.2))
        self.acceleration_curve = max(0.01, float(cfg.get("acceleration_curve", 1.8)))
        self.smooth_factor = max(0.0, min(1.0, float(cfg.get("smooth_factor", 0.2))))
        self.output_cap = max(0.0, float(cfg.get("output_cap", 0.0)))
        self.reset()

    def reset(self):
        self.last_move_x = 0.0
        self.last_move_y = 0.0
        self.is_locking = False
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_target_type = "NONE"
        self.last_confidence = 0.0

    def set_reference(self, rx, ry):
        self.cx = float(rx)
        self.cy = float(ry)

    def status(self):
        return {
            "has_target": self.last_target_type in ("HEAD", "BODY"),
            "locked": self.is_locking,
            "waiting": False,
            "target_type": self.last_target_type,
        }

    def _select_target(self, boxes):
        candidates = []
        for box in boxes or []:
            if len(box) < 5:
                continue
            x1, y1, x2, y2 = map(float, box[:4])
            cls = int(box[4])
            conf = float(box[5]) if len(box) > 5 else 1.0
            if cls != self.target_type:
                continue
            tx = (x1 + x2) / 2.0
            ty = (y1 + y2) / 2.0
            dx, dy = tx - self.cx, ty - self.cy
            dist = math.hypot(dx, dy)
            if dist > self.fov:
                continue
            # Same score as local main_aim.py: center proximity dominates,
            # confidence is a secondary tie-breaker.
            score = dist * 0.6 + (1.0 - conf) * 100.0
            if self.is_locking:
                last_dist = math.hypot(dx - self.last_dx, dy - self.last_dy)
                score = score * 0.4 + last_dist * 0.6
            candidates.append((score, dx, dy, conf))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])

    def _movement(self, error_x, error_y):
        distance = math.hypot(error_x, error_y)
        normalized = min(1.0, distance / 150.0)
        sensitivity = self.base_sensitivity * (0.7 + 1.3 * normalized)
        acceleration = min(1.0, distance / 120.0) ** self.acceleration_curve
        raw_x = error_x * sensitivity * acceleration
        raw_y = error_y * sensitivity * acceleration

        dynamic_smooth = self.smooth_factor * (1.0 - 0.5 * min(1.0, distance / 100.0))
        move_x = self.last_move_x * dynamic_smooth + raw_x * (1.0 - dynamic_smooth)
        move_y = self.last_move_y * dynamic_smooth + raw_y * (1.0 - dynamic_smooth)

        if self.output_cap > 0.0:
            magnitude = math.hypot(move_x, move_y)
            if magnitude > self.output_cap:
                scale = self.output_cap / magnitude
                move_x *= scale
                move_y *= scale

        self.last_move_x = move_x
        self.last_move_y = move_y
        return move_x, move_y

    def update(self, boxes, dt):
        selected = self._select_target(boxes)
        if selected is None:
            self.is_locking = False
            self.last_target_type = "NONE"
            self.last_confidence = 0.0
            self.reset()
            return 0.0, 0.0

        _, dx, dy, conf = selected
        self.is_locking = True
        self.last_dx = dx
        self.last_dy = dy
        self.last_confidence = conf
        self.last_target_type = "HEAD" if self.target_type == 1 else "BODY"
        return self._movement(dx, dy)
