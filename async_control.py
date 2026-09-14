# -*- coding: utf-8 -*-
"""Asynchronous observation snapshots and remaining-error control.

The visual worker owns observation creation; the mouse worker owns the
controller below.  No detector, screenshot, or MainEngine state is touched
here.  Units are explicit: errors are image px, velocities are px/s, and
commands are mouse counts/counts per second.
"""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class ObservationSnapshot:
    session_id: int
    target_id: int | None
    observation_id: int
    source_frame_id: int | None
    image_time_s: float
    image_time_kind: str
    capture_completed_s: float
    published_s: float
    target_point_px: tuple
    crosshair_px: tuple
    raw_error_px: tuple
    filtered_error_px: tuple
    box_size_px: tuple
    confidence: float
    state: str
    last_real_observation_s: float | None
    motion_ledger: object
    target_velocity_px_s: tuple = (0.0, 0.0)
    target_velocity_confidence: object = 0.0
    # ``filtered_error_px`` is the evaluation/control-state position error;
    # ``control_error_px`` may include the explicitly selected finite
    # prediction.  Keeping both prevents observed truth from being confused
    # with the controller's predicted command input.
    control_error_px: tuple | None = None
    prediction_mode: str = "current"
    prediction_reason: str = "current_mode"
    prediction_horizon_ms: float = 0.0


class LatestObservation:
    """One-lock latest-value mailbox; old frames are replaced, never queued."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._value = None

    def publish(self, snapshot):
        if not isinstance(snapshot, ObservationSnapshot):
            raise TypeError("snapshot must be ObservationSnapshot")
        with self._lock:
            self._value = snapshot

    def read(self):
        with self._lock:
            return self._value

    def clear(self):
        with self._lock:
            self._value = None


@dataclass(frozen=True)
class ControlToken:
    """Immutable authority used by capture/publish/send boundaries."""

    session_id: int
    cancel_generation: int


class AimControlEpoch:
    """Serialize cancellation with the final input-submission boundary.

    A token is taken before capture.  Cancellation advances the generation and
    waits for a currently executing send to leave this lock.  Consequently a
    successful cancellation acknowledgement means no later old-token send can
    enter the OS submission call.  It cannot undo an input already accepted by
    the OS.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._session_id = 0
        self._cancel_generation = 0
        self._pressed = False

    def start_session(self, session_id):
        with self._lock:
            self._session_id = int(session_id)
            self._cancel_generation += 1
            self._pressed = True
            return ControlToken(self._session_id, self._cancel_generation)

    def cancel(self):
        with self._lock:
            self._cancel_generation += 1
            self._pressed = False
            return ControlToken(self._session_id, self._cancel_generation)

    def token(self):
        with self._lock:
            return ControlToken(self._session_id, self._cancel_generation)

    def is_current(self, token):
        with self._lock:
            return bool(
                self._pressed and token is not None
                and token.session_id == self._session_id
                and token.cancel_generation == self._cancel_generation)

    def run_send(self, token, send_fn, dx, dy):
        """Run one final validity check and the sole send under one lock.

        Returns ``(ok, start_s, end_s, reason)``.  ``start_s`` and ``end_s``
        are measured around the actual send function, not the output tick.
        """
        with self._lock:
            if not self._pressed or token is None or \
                    token.session_id != self._session_id or \
                    token.cancel_generation != self._cancel_generation:
                now = time.perf_counter()
                return False, now, now, "cancelled_before_send"
            started = time.perf_counter()
            try:
                ok = bool(send_fn(dx, dy))
            except Exception:
                ok = False
            ended = time.perf_counter()
            return ok, started, ended, "submitted" if ok else "send_failed"


class RecoilControlEpoch(AimControlEpoch):
    """Independent authorization channel for recoil-only output.

    Releasing aim must not revoke an otherwise active recoil chord.  The
    output arbiter still combines both components into one OS packet, but it
    chooses this channel when the packet has no active aim plan.
    """

    def set_pressed(self, pressed):
        with self._lock:
            self._cancel_generation += 1
            self._pressed = bool(pressed)
            return ControlToken(self._session_id, self._cancel_generation)


class RemainingErrorController:
    """Proportional residual controller with successful-send accounting.

    A snapshot establishes one finite correction budget.  Every successful aim
    component sent after it reduces that budget; failed sends do not.  For the
    small, bounded first version, sends in the configured feedback window before
    image acquisition are conservatively treated as not yet visible in that
    image.  This is an estimate, never proof that the display has updated.
    """

    def __init__(self, px_per_count=(1.0, 1.0), tau_s=0.080,
                 max_speed_counts_s=(4000.0, 4000.0),
                 max_accel_counts_s2=(0.0, 0.0), deadzone_px=3.0,
                 stop_deadzone_px=1.0, feedback_delay_s=0.020,
                 max_observation_age_s=0.120, history=512):
        self.kx = max(1e-6, float(px_per_count[0]))
        self.ky = max(1e-6, float(px_per_count[1]))
        self.tau_s = max(1e-4, float(tau_s))
        self.max_speed = (max(0.0, float(max_speed_counts_s[0])),
                          max(0.0, float(max_speed_counts_s[1])))
        self.max_accel = (max(0.0, float(max_accel_counts_s2[0])),
                          max(0.0, float(max_accel_counts_s2[1])))
        self.deadzone_px = max(0.0, float(deadzone_px))
        self.stop_deadzone_px = max(0.0, float(stop_deadzone_px))
        self.feedback_delay_s = max(0.0, float(feedback_delay_s))
        self.max_observation_age_s = max(0.001, float(max_observation_age_s))
        self._events = deque(maxlen=max(8, int(history)))
        self._next_local_event_id = 0
        self.reset()

    def reset(self):
        self._snapshot = None
        self._snapshot_key = None
        self._session_id = None
        self._target_id = None
        self._committed_since_observation = [0.0, 0.0]
        self._initial_remaining = [0.0, 0.0]
        self._last_command = [0.0, 0.0]
        self._last_velocity = [0.0, 0.0]
        self._planned_total = [0.0, 0.0]
        self._successful_total = [0.0, 0.0]
        self._pre_capture_pending = [0.0, 0.0]
        self._stopped_latched = False
        self._last_observation_id = None
        self._last_capture_s = None
        self._last_reason = "reset"

    def _same_scope(self, session_id, target_id):
        return self._session_id == session_id and self._target_id == target_id

    def _pending_before_capture(self, snapshot, include_ledger=True):
        cutoff = float(snapshot.capture_completed_s) - self.feedback_delay_s
        pending = [0.0, 0.0]
        # The immutable ledger is kept on the snapshot for audit/replay.  The
        # output-owned event history additionally contains sends after it was
        # captured, so both sides of the image boundary are represented.
        events = list(self._events)
        ledger = getattr(snapshot, "motion_ledger", None)
        if include_ledger and ledger is not None and hasattr(ledger, "successful_events"):
            events.extend(ledger.successful_events)
        seen = set()
        for event in events:
            event_id = None
            if hasattr(event, "event_id"):
                event_id = int(event.event_id)
                t, aim = event.committed_at_s, event.aim
            elif isinstance(event, dict):
                event_id = event.get("event_id")
                t, aim = event.get("committed_at_s"), event.get("aim", (0, 0))
            elif isinstance(event, (tuple, list)) and len(event) == 3:
                t, aim, _recoil = event
            elif isinstance(event, (tuple, list)) and len(event) == 4:
                event_id, t, aim, _recoil = event
            else:
                continue
            namespace = getattr(event, "namespace", None)
            if isinstance(event, dict):
                namespace = event.get("namespace", namespace)
            key = (str(namespace or "default"), event_id, float(t), tuple(aim))
            if key in seen:
                continue
            seen.add(key)
            if cutoff - 1e-9 <= float(t) <= snapshot.capture_completed_s + 1e-9:
                pending[0] += float(aim[0])
                pending[1] += float(aim[1])
        return pending

    def observe(self, snapshot, now_s):
        """Adopt a newer real snapshot, or invalidate on bad state/session."""
        if snapshot is None or snapshot.state != "real" or snapshot.target_id is None:
            self.invalidate("invalid_observation")
            return False
        if self._snapshot_key == (snapshot.session_id, snapshot.observation_id):
            return True
        if self._session_id is not None:
            if int(snapshot.session_id) < int(self._session_id):
                self._last_reason = "out_of_order_session"
                return False
            if (int(snapshot.session_id) == int(self._session_id)
                    and self._last_observation_id is not None
                    and int(snapshot.observation_id) <= int(self._last_observation_id)):
                self._last_reason = "out_of_order_observation"
                return False
            if (int(snapshot.session_id) == int(self._session_id)
                    and self._last_capture_s is not None
                    and float(snapshot.capture_completed_s) + 1e-9 < self._last_capture_s):
                self._last_reason = "out_of_order_capture"
                return False
        scope_changed = (self._session_id != snapshot.session_id or
                         self._target_id != snapshot.target_id)
        if scope_changed:
            # Control state is reset, but physical successful events remain in
            # the audit window.  They can still have moved the current image.
            self._committed_since_observation = [0.0, 0.0]
            self._last_velocity = [0.0, 0.0]
            self._stopped_latched = False
        self._session_id = snapshot.session_id
        self._target_id = snapshot.target_id
        self._snapshot = snapshot
        self._snapshot_key = (snapshot.session_id, snapshot.observation_id)
        self._last_observation_id = int(snapshot.observation_id)
        self._last_capture_s = float(snapshot.capture_completed_s)
        err = (snapshot.control_error_px
               if snapshot.control_error_px is not None
               else snapshot.filtered_error_px)
        base = [float(err[0]) / self.kx, float(err[1]) / self.ky]
        # A scope reset changes target-control history, not physical history.
        # The capture-time ledger is therefore always considered, including
        # the first snapshot of a new target/session or a re-acquired target.
        self._pre_capture_pending = self._pending_before_capture(
            snapshot, include_ledger=True)
        self._initial_remaining = [base[0] - self._pre_capture_pending[0],
                                   base[1] - self._pre_capture_pending[1]]
        # Outputs after capture but before this result was published are absent
        # from the image and reduce the new budget even across a scope change.
        self._committed_since_observation = [0.0, 0.0]
        for event in self._events:
            if hasattr(event, "committed_at_s"):
                event_t, event_aim = event.committed_at_s, event.aim
            elif isinstance(event, dict):
                event_t, event_aim = event.get("committed_at_s"), event.get("aim", (0, 0))
            elif isinstance(event, (tuple, list)) and len(event) == 3:
                event_t, event_aim = event[0], event[1]
            elif isinstance(event, (tuple, list)) and len(event) == 4:
                event_t, event_aim = event[1], event[2]
            else:
                continue
            if float(snapshot.capture_completed_s) < float(event_t) <= float(now_s) + 1e-9:
                self._committed_since_observation[0] += float(event_aim[0])
                self._committed_since_observation[1] += float(event_aim[1])
        self._last_reason = "new_observation"
        return True

    def invalidate(self, reason="invalid"):
        self._snapshot = None
        self._snapshot_key = None
        self._session_id = None
        self._target_id = None
        self._committed_since_observation = [0.0, 0.0]
        self._initial_remaining = [0.0, 0.0]
        self._last_command = [0.0, 0.0]
        self._last_velocity = [0.0, 0.0]
        self._stopped_latched = False
        self._last_reason = str(reason)

    def on_send_succeeded(self, aim_dx, aim_dy, sent_at_s, event_id=None):
        if self._snapshot is None:
            return
        ax, ay = float(aim_dx), float(aim_dy)
        self._committed_since_observation[0] += ax
        self._committed_since_observation[1] += ay
        self._successful_total[0] += ax
        self._successful_total[1] += ay
        local_id = int(event_id) if event_id is not None else None
        if local_id is None:
            self._next_local_event_id -= 1
            local_id = self._next_local_event_id
        self._events.append((local_id,
                             float(sent_at_s), (ax, ay), (0, 0)))

    def next_motion(self, dt_s, now_s, valid=True):
        if not valid or self._snapshot is None:
            self._last_command = [0.0, 0.0]
            return 0.0, 0.0
        age = max(0.0, float(now_s) - float(self._snapshot.last_real_observation_s
                                            if self._snapshot.last_real_observation_s is not None
                                            else self._snapshot.capture_completed_s))
        if age > self.max_observation_age_s:
            self.invalidate("observation_expired")
            return 0.0, 0.0
        dt = max(0.0, min(0.5, float(dt_s)))
        remaining = [self._initial_remaining[0] - self._committed_since_observation[0],
                     self._initial_remaining[1] - self._committed_since_observation[1]]
        rem_px = (remaining[0] * self.kx, remaining[1] * self.ky)
        rem_mag = math.hypot(*rem_px)
        if self._stopped_latched:
            # ``stop_deadzone_px`` is the smaller stop threshold and
            # ``deadzone_px`` is the larger restart threshold.  This is real
            # hysteresis: a stopped controller does not chatter on a 2px
            # observation when it stopped at 1px.
            if rem_mag < self.deadzone_px:
                self._last_velocity = [0.0, 0.0]
                self._last_command = [0.0, 0.0]
                self._last_reason = "stop_hysteresis"
                return 0.0, 0.0
            self._stopped_latched = False
        if rem_mag <= self.stop_deadzone_px:
            self._stopped_latched = True
            self._last_velocity = [0.0, 0.0]
            self._last_command = [0.0, 0.0]
            self._last_reason = "stop_deadzone"
            return 0.0, 0.0
        gain = 1.0 - math.exp(-dt / self.tau_s) if dt > 0.0 else 0.0
        # Acceleration is a velocity-domain constraint.  The exponential
        # response produces a displacement for this tick; divide by dt before
        # applying counts/s and counts/s² limits, then integrate again.
        desired_velocity = [
            remaining[0] * gain / dt if dt > 0.0 else 0.0,
            remaining[1] * gain / dt if dt > 0.0 else 0.0,
        ]
        velocity = desired_velocity[:]
        for axis in (0, 1):
            velocity[axis] = max(-self.max_speed[axis],
                                 min(self.max_speed[axis], velocity[axis]))
            accel = self.max_accel[axis]
            if accel > 0.0:
                limit = accel * dt
                delta = velocity[axis] - self._last_velocity[axis]
                velocity[axis] = self._last_velocity[axis] + max(-limit, min(limit, delta))
        command = [velocity[0] * dt, velocity[1] * dt]
        # Never plan past the finite remaining budget in the current direction.
        for axis in (0, 1):
            if remaining[axis] * command[axis] > 0.0:
                command[axis] = math.copysign(
                    min(abs(command[axis]), abs(remaining[axis])), command[axis])
        self._last_velocity = [
            command[0] / dt if dt > 0.0 else 0.0,
            command[1] / dt if dt > 0.0 else 0.0,
        ]
        self._last_command = command[:]
        self._planned_total[0] += command[0]
        self._planned_total[1] += command[1]
        self._last_reason = "tracking"
        return command[0], command[1]

    def diagnostics(self):
        remaining = (self._initial_remaining[0] - self._committed_since_observation[0],
                     self._initial_remaining[1] - self._committed_since_observation[1])
        return {
            "planned_counts": tuple(self._planned_total),
            "successful_counts": tuple(self._successful_total),
            "pre_capture_pending_counts": tuple(self._pre_capture_pending),
            "committed_since_observation_counts": tuple(self._committed_since_observation),
            "remaining_counts": remaining,
            "last_command_counts": tuple(self._last_command),
            "reason": self._last_reason,
        }
