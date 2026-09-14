# -*- coding: utf-8 -*-
"""单写入鼠标输出器使用的纯状态组件。"""

import threading
from dataclasses import dataclass
from collections import deque


@dataclass(frozen=True)
class MotionComponents:
    """Cumulative or interval motion split by producer."""

    aim: tuple
    recoil: tuple

    @property
    def total(self):
        return (
            int(self.aim[0]) + int(self.recoil[0]),
            int(self.aim[1]) + int(self.recoil[1]),
        )


@dataclass(frozen=True)
class MotionEvent:
    """One committed physical packet with a monotonic arbiter event id."""

    event_id: int
    committed_at_s: float
    aim: tuple
    recoil: tuple
    namespace: str = "default"


@dataclass(frozen=True)
class MotionLedgerSnapshot:
    """Immutable motion ledger captured at image acquisition time.

    The event list is deliberately a value copy.  A visual observation can
    therefore be handed to the output worker without sharing mutable ledger
    state or accidentally attributing motion produced during inference to the
    image that preceded it.
    """

    captured_at_s: float
    aim_total: tuple
    recoil_total: tuple
    successful_events: tuple
    namespace: str = "default"

    @property
    def total(self):
        return (
            int(self.aim_total[0]) + int(self.recoil_total[0]),
            int(self.aim_total[1]) + int(self.recoil_total[1]),
        )


def mix_motion_components(aim_dx, aim_dy, recoil_dx, recoil_dy, max_counts):
    """共同限幅瞄准与后坐力，同时保留可独立学习的后坐力分量。"""
    aim_dx, aim_dy = float(aim_dx), float(aim_dy)
    recoil_dx, recoil_dy = float(recoil_dx), float(recoil_dy)
    total_x = aim_dx + recoil_dx
    total_y = aim_dy + recoil_dy
    limit = max(0.0, float(max_counts))
    magnitude = (total_x * total_x + total_y * total_y) ** 0.5
    scale = limit / magnitude if limit > 0.0 and magnitude > limit else 1.0
    return (
        total_x * scale,
        total_y * scale,
        recoil_dx * scale,
        recoil_dy * scale,
    )


class MotionArbiter:
    """保存最新瞄准计划，并用累计量差分做无漂移整数化。"""

    def __init__(self, namespace="default"):
        self._lock = threading.RLock()
        self.namespace = str(namespace or "default")
        self._rate_x = 0.0
        self._rate_y = 0.0
        self.reset()

    def reset(self):
        with self._lock:
            self._rate_x = 0.0
            self._rate_y = 0.0
            self._chunks = []
            self._desired_x = 0.0
            self._desired_y = 0.0
            self._aim_desired_x = 0.0
            self._aim_desired_y = 0.0
            self._recoil_desired_x = 0.0
            self._recoil_desired_y = 0.0
            self._aim_target_x = 0
            self._aim_target_y = 0
            self._recoil_target_x = 0
            self._recoil_target_y = 0
            self._last_aim_step = (0, 0)
            self._last_recoil_step = (0, 0)
            self._actual_aim_x = 0
            self._actual_aim_y = 0
            self._actual_recoil_x = 0
            self._actual_recoil_y = 0
            self._pending_component_commit = False
            self._sent_x = 0
            self._sent_y = 0
            self._drained_x = 0
            self._drained_y = 0
            self._drained_aim_x = 0
            self._drained_aim_y = 0
            self._drained_recoil_x = 0
            self._drained_recoil_y = 0
            self._pending_kind = None
            self._pending_previous = None
            self._successful_events = deque(maxlen=512)
            self._next_event_id = 0
            self._last_event_id = 0

    def reset_control_scope(self):
        """Drop target plans without erasing committed physical history.

        Session/target filters may be reset independently from the camera
        motion ledger.  Keeping cumulative committed counts and events lets a
        following capture account for input that is still entering the image.
        """
        with self._lock:
            self._rate_x = 0.0
            self._rate_y = 0.0
            self._chunks = []
            self._last_aim_step = (0, 0)
            self._last_recoil_step = (0, 0)
            self._pending_component_commit = False
            self._pending_kind = None
            self._pending_previous = None

    @property
    def sent_total(self):
        with self._lock:
            return self._sent_x, self._sent_y

    @property
    def last_event_id(self):
        with self._lock:
            return self._last_event_id

    @property
    def component_sent_totals(self):
        """Rounded aim/recoil targets used by the independent ledgers."""
        with self._lock:
            return (
                (self._aim_target_x, self._aim_target_y),
                (self._recoil_target_x, self._recoil_target_y),
            )

    @property
    def last_component_steps(self):
        with self._lock:
            return self._last_aim_step, self._last_recoil_step

    def publish_aim(self, dx, dy, steps=1, profile="linear"):
        """用新观测替换尚未发送的旧计划，避免补发过期移动。"""
        steps = max(1, int(steps))
        if profile == "ease" and steps > 1:
            raw = list(range(steps, 0, -1))
            total = float(sum(raw))
            weights = [value / total for value in raw]
        else:
            weights = [1.0 / steps] * steps
        chunks = [(float(dx) * weight, float(dy) * weight) for weight in weights]
        # 浮点尾数也严格补到最后一块。
        chunks[-1] = (
            chunks[-1][0] + float(dx) - sum(x for x, _ in chunks),
            chunks[-1][1] + float(dy) - sum(y for _, y in chunks),
        )
        with self._lock:
            self._rate_x = 0.0
            self._rate_y = 0.0
            self._chunks = chunks

    def publish_aim_rate(self, rate_x, rate_y):
        """asap 调度：发布"最新控制速率"（counts/second），替换旧速率。

        速率语义下没有"尚未发完的尾段"——输出线程按真实 tick dt 积分，
        每次发布即刻整体替换上一速率；换向时旧方向不会残留任何待发量。
        """
        with self._lock:
            self._rate_x = float(rate_x)
            self._rate_y = float(rate_y)
            self._chunks = []

    def aim_rate(self):
        """当前生效的控制速率（counts/s）；无速率计划时为 (0, 0)。"""
        with self._lock:
            return self._rate_x, self._rate_y

    def has_pending_aim(self):
        """是否还有未消费的瞄准计划（chunk 或非零速率），供 stale 清除计数。"""
        with self._lock:
            return bool(self._chunks) or self._rate_x != 0.0 or self._rate_y != 0.0

    def clear_aim(self):
        with self._lock:
            self._rate_x = 0.0
            self._rate_y = 0.0
            self._chunks = []

    def next_aim_chunk(self):
        with self._lock:
            if not self._chunks:
                return 0.0, 0.0
            return self._chunks.pop(0)

    def next_aim_rate_motion(self, dt_s):
        """asap 调度：把当前最新速率按本次真实 tick dt 积分成瞄准位移。

        不消耗、不排队——每次调用都用同一个最新速率；新观测通过
        publish_aim_rate 整体替换，绝不补发旧计划的尾段。
        """
        dt_s = min(0.5, max(0.0, float(dt_s)))
        with self._lock:
            return self._rate_x * dt_s, self._rate_y * dt_s

    def quantize(self, dx, dy):
        """累计期望量差分：累计发送恒等于累计期望的四舍五入。"""
        with self._lock:
            self._pending_kind = "simple"
            self._pending_previous = (
                self._desired_x, self._desired_y, self._sent_x, self._sent_y)
            self._desired_x += float(dx)
            self._desired_y += float(dy)
            target_x = round(self._desired_x)
            target_y = round(self._desired_y)
            send_x = target_x - self._sent_x
            send_y = target_y - self._sent_y
            self._sent_x = target_x
            self._sent_y = target_y
            return send_x, send_y

    def quantize_components(self, aim_dx, aim_dy, recoil_dx, recoil_dy):
        """Accumulate aim and recoil fractions independently, then combine.

        This prevents a small opposing YOLO correction from swallowing the
        recoil fraction before it reaches a whole Windows mouse count.
        """
        with self._lock:
            self._pending_kind = "components"
            self._pending_previous = (
                self._aim_desired_x, self._aim_desired_y,
                self._recoil_desired_x, self._recoil_desired_y,
                self._aim_target_x, self._aim_target_y,
                self._recoil_target_x, self._recoil_target_y,
                self._sent_x, self._sent_y)
            previous_aim = (self._aim_target_x, self._aim_target_y)
            previous_recoil = (self._recoil_target_x, self._recoil_target_y)
            self._aim_desired_x += float(aim_dx)
            self._aim_desired_y += float(aim_dy)
            self._recoil_desired_x += float(recoil_dx)
            self._recoil_desired_y += float(recoil_dy)
            self._aim_target_x = round(self._aim_desired_x)
            self._aim_target_y = round(self._aim_desired_y)
            self._recoil_target_x = round(self._recoil_desired_x)
            self._recoil_target_y = round(self._recoil_desired_y)
            self._last_aim_step = (
                self._aim_target_x - previous_aim[0],
                self._aim_target_y - previous_aim[1],
            )
            self._last_recoil_step = (
                self._recoil_target_x - previous_recoil[0],
                self._recoil_target_y - previous_recoil[1],
            )
            self._pending_component_commit = True
            target_x = self._aim_target_x + self._recoil_target_x
            target_y = self._aim_target_y + self._recoil_target_y
            send_x = target_x - self._sent_x
            send_y = target_y - self._sent_y
            self._sent_x = target_x
            self._sent_y = target_y
            return send_x, send_y

    def drain_sent(self):
        """返回上次读取后实际整数发送的净量，供下一次YOLO观测做自身运动补偿。"""
        with self._lock:
            dx = self._sent_x - self._drained_x
            dy = self._sent_y - self._drained_y
            self._drained_x = self._sent_x
            self._drained_y = self._sent_y
            return dx, dy

    def snapshot_components(self):
        """Snapshot the cumulative split at the instant an image is captured."""
        with self._lock:
            return MotionComponents(
                (self._actual_aim_x, self._actual_aim_y),
                (self._actual_recoil_x, self._actual_recoil_y),
            )

    def ledger_snapshot(self, captured_at_s=0.0):
        """Copy committed motion and successful output events atomically."""
        with self._lock:
            return MotionLedgerSnapshot(
                float(captured_at_s),
                (self._actual_aim_x, self._actual_aim_y),
                (self._actual_recoil_x, self._actual_recoil_y),
                tuple(self._successful_events),
                self.namespace,
            )

    def mark_send_succeeded(self, sent_at_s=None, record_event=True):
        """Commit the most recent component step after SendInput succeeds.

        ``record_event=False`` is used for a zero-net component commit or a
        fractional wait.  The split accumulators still advance and pending
        state is cleared, but no physical movement event is invented.
        """
        with self._lock:
            if not self._pending_component_commit:
                return
            self._actual_aim_x += self._last_aim_step[0]
            self._actual_aim_y += self._last_aim_step[1]
            self._actual_recoil_x += self._last_recoil_step[0]
            self._actual_recoil_y += self._last_recoil_step[1]
            if sent_at_s is not None and record_event:
                self._next_event_id += 1
                self._last_event_id = self._next_event_id
                self._successful_events.append(MotionEvent(
                    self._next_event_id,
                    float(sent_at_s),
                    (int(self._last_aim_step[0]), int(self._last_aim_step[1])),
                    (int(self._last_recoil_step[0]), int(self._last_recoil_step[1])),
                    self.namespace,
                ))
            self._pending_component_commit = False
            self._pending_kind = None
            self._pending_previous = None

    def drain_components_through(self, snapshot):
        """Drain only motion already present at a capture-time snapshot.

        Output produced while inference is running remains undrained and is
        attributed to the following image instead of the stale current image.
        """
        if not isinstance(snapshot, MotionComponents):
            raise TypeError("snapshot must be MotionComponents")
        with self._lock:
            aim = (
                int(snapshot.aim[0]) - self._drained_aim_x,
                int(snapshot.aim[1]) - self._drained_aim_y,
            )
            recoil = (
                int(snapshot.recoil[0]) - self._drained_recoil_x,
                int(snapshot.recoil[1]) - self._drained_recoil_y,
            )
            self._drained_aim_x, self._drained_aim_y = map(int, snapshot.aim)
            self._drained_recoil_x, self._drained_recoil_y = map(
                int, snapshot.recoil)
            self._drained_x, self._drained_y = snapshot.total
            return MotionComponents(aim, recoil)

    def mark_send_failed(self, dx, dy):
        """Drop a failed packet so stale motion is never replayed later."""
        with self._lock:
            if self._pending_kind == "components" and self._pending_previous is not None:
                (
                    self._aim_desired_x, self._aim_desired_y,
                    self._recoil_desired_x, self._recoil_desired_y,
                    self._aim_target_x, self._aim_target_y,
                    self._recoil_target_x, self._recoil_target_y,
                    self._sent_x, self._sent_y,
                ) = self._pending_previous
            elif self._pending_kind == "simple" and self._pending_previous is not None:
                self._desired_x, self._desired_y, self._sent_x, self._sent_y = (
                    self._pending_previous)
            self._last_aim_step = (0, 0)
            self._last_recoil_step = (0, 0)
            self._pending_component_commit = False
            self._pending_kind = None
            self._pending_previous = None
