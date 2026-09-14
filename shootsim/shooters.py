# -*- coding: utf-8 -*-
"""射击策略。

``always``：按射速连续射击。
``lock_delay``：只依据 Tracker 公共状态，在锁定稳定后开火。
``oracle_on_aim``：依据真实命中框开火，仅用于演示；旧的 ``on_aim`` 是兼容别名。
"""


class Shooter:
    def __init__(self, cfg, rng=None):
        raw_policy = str(cfg.get("policy", "always")).lower()
        self.is_oracle = raw_policy in ("on_aim", "oracle_on_aim")
        self.policy = "oracle_on_aim" if self.is_oracle else raw_policy
        self.auto_fire = bool(cfg.get("auto_fire", True))
        self.reaction_s = max(0.0, float(cfg.get("reaction_ms", 120.0))) / 1000.0
        self.lost_grace_s = max(0.0, float(cfg.get("lost_grace_ms", 50.0))) / 1000.0
        self.accuracy_valid = not self.is_oracle
        self.accuracy_note = (
            "oracle策略：开火条件读取真实命中框，accuracy不参与算法比较"
            if self.is_oracle else ""
        )
        self.reset()

    def reset(self):
        self.stable_s = 0.0
        self.lost_s = 0.0
        self.lock_delay_ready = False
        self.last_tracker_status = None

    def set_tracker_status(self, status, interval_s):
        """更新 lock_delay 状态；只接受 Tracker 的公共状态字典。"""
        self.last_tracker_status = dict(status) if isinstance(status, dict) else None
        if self.policy != "lock_delay":
            return self.lock_delay_ready
        interval_s = max(0.0, float(interval_s))
        valid = bool(
            isinstance(status, dict)
            and status.get("has_target")
            and not status.get("waiting")
        )
        if valid:
            self.stable_s += interval_s
            self.lost_s = 0.0
            if self.stable_s >= self.reaction_s:
                self.lock_delay_ready = True
        else:
            self.stable_s = 0.0
            self.lost_s += interval_s
            if self.lost_s > self.lost_grace_s:
                self.lock_delay_ready = False
        return self.lock_delay_ready

    def should_fire(self, crosshair_in_box=None):
        """每个射击节拍询问是否开火。"""
        if not self.auto_fire:
            return False
        if self.policy == "always":
            return True
        if self.policy == "oracle_on_aim":
            return bool(crosshair_in_box)
        if self.policy == "lock_delay":
            return self.lock_delay_ready
        return True


POLICIES = {
    "always": "连续射击",
    "lock_delay": "锁定延迟后射击",
    "oracle_on_aim": "真实命中框触发（oracle，仅演示）",
    "on_aim": "真实命中框触发（oracle兼容别名）",
}
