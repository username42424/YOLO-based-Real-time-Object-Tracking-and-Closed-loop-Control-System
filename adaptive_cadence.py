# -*- coding: utf-8 -*-
"""红点ROI采样周期的纯状态降级器。"""


class AdaptiveCadence:
    def __init__(self, periods_ms=(30.0, 45.0, 60.0), pause_s=2.0,
                 skip_limit=0.50, yolo_slow_ratio=1.15):
        self.periods_s = tuple(float(v) / 1000.0 for v in periods_ms)
        self.pause_s = float(pause_s)
        self.skip_limit = float(skip_limit)
        self.yolo_slow_ratio = float(yolo_slow_ratio)
        self.index = 0
        self.disabled_until = 0.0
        self.last_bad = 0.0

    @property
    def period_s(self):
        return self.periods_s[self.index]

    def paused(self, now):
        return float(now) < self.disabled_until

    def observe(self, now, skip_rate, roi_p95_ms, idle_yolo_p95_ms=0.0,
                fire_yolo_p95_ms=0.0, yolo_comparison_ready=False):
        now = float(now)
        yolo_slow = bool(
            yolo_comparison_ready and idle_yolo_p95_ms > 0.0
            and fire_yolo_p95_ms > idle_yolo_p95_ms * self.yolo_slow_ratio)
        bad = (float(skip_rate) > self.skip_limit
               or float(roi_p95_ms) > self.period_s * 1000.0 or yolo_slow)
        if bad:
            self.last_bad = now
            if self.index < len(self.periods_s) - 1:
                self.index += 1
            else:
                self.disabled_until = now + self.pause_s
        elif self.index > 0 and now - self.last_bad >= 2.0:
            self.index -= 1
            self.last_bad = now
        return self.period_s, self.paused(now)
