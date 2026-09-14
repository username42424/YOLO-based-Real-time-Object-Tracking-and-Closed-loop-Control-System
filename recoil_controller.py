# -*- coding: utf-8 -*-
"""五轮靶场校准、按时间回放和可持久化的二维后坐力控制器。"""

from copy import deepcopy
from dataclasses import dataclass
import math
import statistics
import threading


PROFILE_LABELS = {
    "unknown": "未知 / 自动学习", "low": "低后坐力", "medium": "中后坐力",
    "high": "高后坐力", "custom1": "自定义 1", "custom2": "自定义 2",
}


def _profile(first):
    return {"first_pulse_counts": float(first)}


DEFAULT_RECOIL_PROFILES = {
    "unknown": _profile(4.0), "low": _profile(3.0),
    "medium": _profile(5.0), "high": _profile(7.0),
    "custom1": _profile(4.0), "custom2": _profile(4.0),
}


@dataclass(frozen=True)
class RecoilOutput:
    phase: str
    feedforward_x: float = 0.0
    feedforward_y: float = 0.0
    sustain_x: float = 0.0
    sustain_y: float = 0.0
    recovery_x: float = 0.0
    recovery_y: float = 0.0
    shot_active: bool = False
    curve_bin: int = 0
    calibration_runs: int = 0
    ready: bool = False

    @property
    def total_x(self):
        return self.feedforward_x + self.sustain_x + self.recovery_x

    @property
    def total_y(self):
        return self.feedforward_y + self.sustain_y + self.recovery_y


@dataclass(frozen=True)
class RecoilGateTransition:
    start: bool = False
    stop: bool = False


class RecoilChordGate:
    """严格的 RMB→LMB 门控；右键松开会结束当前 burst。"""

    def __init__(self):
        self._lock = threading.RLock()
        self.right_held = False
        self.left_held = False
        self.active = False

    def reset(self):
        with self._lock:
            self.right_held = self.left_held = self.active = False

    def on_right(self, pressed):
        pressed = bool(pressed)
        with self._lock:
            if pressed == self.right_held:
                return RecoilGateTransition()
            self.right_held = pressed
            if not pressed and self.active:
                self.active = False
                return RecoilGateTransition(stop=True)
            # 右键后按不会接管一个已经按住的左键；必须重新产生 LMB down。
            return RecoilGateTransition()

    def on_left(self, pressed):
        pressed = bool(pressed)
        with self._lock:
            if pressed == self.left_held:
                return RecoilGateTransition()
            self.left_held = pressed
            if pressed:
                if self.right_held and not self.active:
                    self.active = True
                    return RecoilGateTransition(start=True)
                return RecoilGateTransition()
            if self.active:
                self.active = False
                return RecoilGateTransition(stop=True)
            return RecoilGateTransition()


@dataclass(frozen=True)
class ManualRecoilOutput:
    """One unscaled output slice from a learned physical-mouse trajectory."""

    phase: str
    x: float = 0.0
    y: float = 0.0
    curve_bin: int = 0
    calibration_runs: int = 0
    ready: bool = False
    missed_interval: bool = False


def apply_playback_blend(output, percent):
    """Scale learned playback only; 0% is an unconditional output kill switch."""

    blend = min(100.0, max(0.0, float(percent))) / 100.0
    return float(output.x) * blend, float(output.y) * blend


class ManualRecoilController:
    """Learn five physical mouse bursts and replay their per-bin median.

    Timestamps are integer monotonic nanoseconds. Raw mouse deltas are event data,
    so an otherwise healthy capture stream with no event in a bin means a real
    zero movement rather than a scheduler hole.
    """

    NS_PER_MS = 1_000_000

    def __init__(self, bin_ms=60.0, calibration_runs=5,
                 min_burst_ms=300.0, tail_vertical_ratio=0.30,
                 tail_max_ms=1500.0):
        self.bin_ns = max(20, int(round(float(bin_ms)))) * self.NS_PER_MS
        self.required_runs = max(1, int(calibration_runs))
        self.min_burst_ns = max(1, int(round(float(min_burst_ms) * self.NS_PER_MS)))
        self.tail_vertical_ratio = min(1.0, max(0.0, float(tail_vertical_ratio)))
        self.tail_max_ns = max(0, int(round(float(tail_max_ms) * self.NS_PER_MS)))
        self._lock = threading.RLock()
        self._runs = []
        self._curve = []
        self._last_accepted_run = []
        self._bound_device = None
        self._loaded_profile_name = None
        self._last_result = "not_started"
        self.reset_fire_state()

    def reset_fire_state(self):
        with self._lock:
            self.pressed = False
            self.phase = "IDLE"
            self._fire_started_ns = None
            self._last_step_ns = None
            self._collecting = False
            self._capture_healthy = True
            self._current_bins = []
            self._current_event_count = 0

    @property
    def calibration_count(self):
        with self._lock:
            return len(self._runs)

    @property
    def ready(self):
        with self._lock:
            return len(self._runs) >= self.required_runs

    @property
    def loaded_profile(self):
        with self._lock:
            return self._loaded_profile_name is not None

    @property
    def loaded_profile_name(self):
        with self._lock:
            return self._loaded_profile_name

    @property
    def curve_bins(self):
        with self._lock:
            return len(self._curve)

    @property
    def profile_duration_ms(self):
        with self._lock:
            return len(self._curve) * self.bin_ns / self.NS_PER_MS

    @property
    def last_calibration_result(self):
        with self._lock:
            return self._last_result

    @property
    def bound_device(self):
        with self._lock:
            return self._bound_device

    @property
    def calibrating(self):
        with self._lock:
            return bool(self.pressed and self._collecting)

    def curve_point(self, index):
        with self._lock:
            x, y = self._curve[int(index)]
            return x, y

    @property
    def curve_points(self):
        with self._lock:
            return tuple(self._curve)

    @property
    def last_accepted_run(self):
        with self._lock:
            return tuple(self._last_accepted_run)

    @property
    def source_runs(self):
        with self._lock:
            return tuple(tuple(run) for run in self._runs)

    def snapshot(self):
        """Thread-safe data for live GUI previews, including partial sessions."""
        with self._lock:
            return {
                "bin_ms": self.bin_ns / self.NS_PER_MS,
                "duration_ms": len(self._curve) * self.bin_ns / self.NS_PER_MS,
                "required_runs": self.required_runs,
                "source_runs": [
                    [[float(x), float(y)] for x, y in run]
                    for run in self._runs
                ],
                "curve": [[float(x), float(y)] for x, y in self._curve],
                "calibration_count": len(self._runs),
                "ready": self.ready,
                "loaded_profile_name": self._loaded_profile_name,
                "phase": self.phase,
                "last_result": self._last_result,
            }

    def export_profile(self):
        """Return the five raw runs and their median curve for persistence."""
        with self._lock:
            if not self.ready:
                raise ValueError("需要完成5次有效手动记录后才能保存")
            return self.snapshot()

    def load_profile(self, profile, name=None):
        """Load a validated saved profile and become READY without calibration."""
        from recoil_profiles import normalise_profile

        clean = normalise_profile(profile)
        with self._lock:
            self.bin_ns = int(round(clean["bin_ms"] * self.NS_PER_MS))
            self.required_runs = int(clean["required_runs"])
            self._runs = [
                [tuple(point) for point in run]
                for run in clean["source_runs"]
            ]
            self._curve = [tuple(point) for point in clean["curve"]]
            self._last_accepted_run = list(self._runs[-1])
            self._bound_device = None
            self._loaded_profile_name = str(name) if name else "已保存曲线"
            self._last_result = "loaded_profile"
            self.reset_fire_state()
            self.phase = "READY"
            return True

    def set_tail_vertical_ratio(self, ratio):
        with self._lock:
            self.tail_vertical_ratio = min(1.0, max(0.0, float(ratio)))
            return self.tail_vertical_ratio

    def reset_learning(self):
        with self._lock:
            self._runs = []
            self._curve = []
            self._last_accepted_run = []
            self._bound_device = None
            self._loaded_profile_name = None
            self._last_result = "reset"
            self.reset_fire_state()
            return True

    def mark_capture_unhealthy(self):
        with self._lock:
            if self._collecting:
                self._capture_healthy = False

    def _rebuild_curve(self):
        common = min((len(run) for run in self._runs), default=0)
        self._curve = [
            (
                statistics.median(run[index][0] for run in self._runs),
                statistics.median(run[index][1] for run in self._runs),
            )
            for index in range(common)
        ]

    def on_fire(self, pressed, now_ns, capture_healthy=True):
        pressed, now_ns = bool(pressed), int(now_ns)
        with self._lock:
            if pressed == self.pressed:
                return False
            if pressed:
                self.pressed = True
                self._fire_started_ns = now_ns
                self._last_step_ns = now_ns
                self._current_bins = []
                self._current_event_count = 0
                self._capture_healthy = bool(capture_healthy)
                self._collecting = not self.ready
                self.phase = "CALIBRATING" if self._collecting else "REPLAY"
                return False

            accepted = False
            if self._collecting and self._fire_started_ns is not None:
                healthy = bool(capture_healthy) and self._capture_healthy
                duration_ns = max(0, now_ns - self._fire_started_ns)
                full_bins = duration_ns // self.bin_ns
                if not healthy:
                    self._last_result = "discarded_capture_unhealthy"
                elif duration_ns < self.min_burst_ns or full_bins <= 0:
                    self._last_result = "discarded_too_short"
                elif self._current_event_count <= 0:
                    self._last_result = "discarded_no_mouse_data"
                else:
                    run = [[0.0, 0.0] for _ in range(full_bins)]
                    for index, point in enumerate(self._current_bins[:full_bins]):
                        run[index][0], run[index][1] = point
                    self._runs.append([tuple(point) for point in run])
                    self._last_accepted_run = [tuple(point) for point in run]
                    if len(self._runs) > self.required_runs:
                        self._runs = self._runs[:self.required_runs]
                    self._rebuild_curve()
                    count = len(self._runs)
                    self._last_result = (
                        "ready" if count >= self.required_runs
                        else f"accepted_{count}_of_{self.required_runs}"
                    )
                    accepted = True

            self.pressed = False
            self._collecting = False
            self._fire_started_ns = None
            self._last_step_ns = None
            self._current_bins = []
            self._current_event_count = 0
            self.phase = "READY" if self.ready else "IDLE"
            return accepted

    def record_raw_delta(self, dx, dy, now_ns, device):
        dx, dy, now_ns = float(dx), float(dy), int(now_ns)
        with self._lock:
            if (not self.pressed or not self._collecting
                    or self._fire_started_ns is None):
                return False
            if not math.isfinite(dx) or not math.isfinite(dy):
                self._capture_healthy = False
                return False
            if self._bound_device is None and (dx != 0.0 or dy != 0.0):
                self._bound_device = device
            if self._bound_device is not None and device != self._bound_device:
                return False
            elapsed_ns = now_ns - self._fire_started_ns
            if elapsed_ns < 0:
                return False
            index = elapsed_ns // self.bin_ns
            while len(self._current_bins) <= index:
                self._current_bins.append([0.0, 0.0])
            self._current_bins[index][0] += dx
            self._current_bins[index][1] += dy
            self._current_event_count += 1
            return True

    def _curve_integral_ns(self, start_ns, end_ns):
        if not self._curve or end_ns <= start_ns:
            return 0.0, 0.0
        duration_ns = len(self._curve) * self.bin_ns
        start_ns = max(0, int(start_ns))
        end_ns = min(int(end_ns), duration_ns)
        if end_ns <= start_ns:
            return 0.0, 0.0
        first = start_ns // self.bin_ns
        last = min(len(self._curve) - 1, (end_ns - 1) // self.bin_ns)
        out_x = out_y = 0.0
        for index in range(first, last + 1):
            bin_start = index * self.bin_ns
            bin_end = bin_start + self.bin_ns
            overlap = max(0, min(end_ns, bin_end) - max(start_ns, bin_start))
            fraction = overlap / self.bin_ns
            out_x += self._curve[index][0] * fraction
            out_y += self._curve[index][1] * fraction
        return out_x, out_y

    def _tail_integral_ns(self, start_ns, end_ns):
        curve_end = len(self._curve) * self.bin_ns
        tail_end = curve_end + self.tail_max_ns
        start_ns = max(int(start_ns), curve_end)
        end_ns = min(int(end_ns), tail_end)
        if end_ns <= start_ns or not self._curve or self.tail_max_ns <= 0:
            return 0.0
        tail_source = [point[1] for point in self._curve[-3:]]
        per_bin_y = statistics.median(tail_source) * self.tail_vertical_ratio
        return per_bin_y * ((end_ns - start_ns) / self.bin_ns)

    def step(self, now_ns, enabled):
        now_ns = int(now_ns)
        with self._lock:
            count = len(self._runs)
            if (not enabled or not self.pressed or not self.ready
                    or self._fire_started_ns is None):
                return ManualRecoilOutput(
                    self.phase, calibration_runs=count, ready=self.ready)

            previous = self._last_step_ns if self._last_step_ns is not None else now_ns
            self._last_step_ns = now_ns
            start_ns = max(0, previous - self._fire_started_ns)
            end_ns = max(start_ns, now_ns - self._fire_started_ns)
            missed = end_ns - start_ns > self.bin_ns
            if missed:
                start_ns = end_ns - self.bin_ns

            out_x, out_y = self._curve_integral_ns(start_ns, end_ns)
            out_y += self._tail_integral_ns(start_ns, end_ns)
            curve_end = len(self._curve) * self.bin_ns
            if end_ns <= curve_end:
                phase = "REPLAY"
            elif end_ns <= curve_end + self.tail_max_ns:
                phase = "TAIL"
            else:
                phase = "EXHAUSTED"
                out_x = out_y = 0.0
            self.phase = phase
            return ManualRecoilOutput(
                phase, out_x, out_y,
                curve_bin=int(end_ns // self.bin_ns),
                calibration_runs=count, ready=True,
                missed_interval=missed,
            )


class RecoilController:
    """记录五次目标辅助射击的后坐力分量，并回放逐格二维中位曲线。"""

    def __init__(self, profile="unknown", profiles=None, bin_ms=60.0,
                 calibration_runs=5, min_calibration_bins=10,
                 max_target_gap_ms=120.0, recovery_ms=100.0):
        self._profiles = deepcopy(profiles if profiles is not None else DEFAULT_RECOIL_PROFILES)
        if not self._profiles:
            raise ValueError("recoil profiles cannot be empty")
        self.bin_s = max(0.020, float(bin_ms) / 1000.0)
        self.required_runs = max(1, int(calibration_runs))
        self.min_calibration_bins = max(1, int(min_calibration_bins))
        self.max_target_gap_s = max(0.0, float(max_target_gap_ms) / 1000.0)
        self.recovery_s = max(0.0, float(recovery_ms) / 1000.0)
        self._lock = threading.RLock()
        self.profile = profile if profile in self._profiles else next(iter(self._profiles))
        self._runs = {key: [] for key in self._profiles}
        self._curves = {key: [] for key in self._profiles}
        self._observation_totals = {key: 0 for key in self._profiles}
        self._last_results = {key: "not_started" for key in self._profiles}
        self.reset_fire_state()

    def reset_fire_state(self):
        with self._lock:
            self.phase, self.pressed = "IDLE", False
            self._fire_started_at = self._released_at = self._last_step_at = None
            self._pending_recovery = [0.0, 0.0]
            self._recovery_until = None
            self._baseline_pending = False
            self._reset_run_candidate()

    def _reset_run_candidate(self):
        self._collecting = False
        self._run_bins = []
        self._run_seen_bins = set()
        self._run_last_target_at = None
        self._run_invalid_reason = None
        self._run_reliable_bins = 0
        self._run_closed = False
        self._run_truncated = False

    def set_profile(self, profile):
        with self._lock:
            if profile not in self._profiles:
                profile = "unknown" if "unknown" in self._profiles else next(iter(self._profiles))
            if profile != self.profile:
                self.profile = profile
                self.reset_fire_state()

    @property
    def calibration_count(self):
        with self._lock:
            return len(self._runs[self.profile])

    @property
    def learning_sample_count(self):
        return self.calibration_count

    @property
    def learning_observation_count(self):
        with self._lock:
            return self._observation_totals[self.profile]

    @property
    def ready(self):
        with self._lock:
            return len(self._runs[self.profile]) >= self.required_runs

    @property
    def can_control_without_target(self):
        return self.ready

    @property
    def curve_bins(self):
        with self._lock:
            return len(self._curves[self.profile])

    @property
    def profile_duration_ms(self):
        return self.curve_bins * self.bin_s * 1000.0

    @property
    def last_calibration_result(self):
        with self._lock:
            return self._last_results[self.profile]

    @property
    def learned_first_counts(self):
        with self._lock:
            curve = self._curves[self.profile]
            return curve[0][1] if curve else float(
                self._profiles[self.profile].get("first_pulse_counts", 0.0))

    @property
    def learned_first_x(self):
        with self._lock:
            curve = self._curves[self.profile]
            return curve[0][0] if curve else 0.0

    def curve_point(self, index):
        with self._lock:
            point = self._curves[self.profile][int(index)]
            return point[0], point[1]

    def reset_learning(self, profile=None):
        with self._lock:
            key = self.profile if profile is None else profile
            if key not in self._profiles:
                return False
            self._runs[key] = []
            self._curves[key] = []
            self._observation_totals[key] = 0
            self._last_results[key] = "reset"
            if key == self.profile:
                self.reset_fire_state()
            return True

    def _rebuild_curve(self):
        runs = self._runs[self.profile]
        if not runs:
            self._curves[self.profile] = []
            return
        common = min(len(run) for run in runs)
        self._curves[self.profile] = [
            (
                statistics.median(run[index][0] for run in runs),
                statistics.median(run[index][1] for run in runs),
            )
            for index in range(common)
        ]

    def _finalize_run(self, now):
        if not self._collecting:
            return False
        if (self._run_last_target_at is None
                or float(now) - self._run_last_target_at > self.max_target_gap_s):
            self._run_closed = True
            self._run_truncated = True

        contiguous = 0
        while contiguous in self._run_seen_bins:
            contiguous += 1
        usable = min(contiguous, self._run_reliable_bins)
        if usable < self.min_calibration_bins:
            reason = "target_gap" if self._run_closed else "too_short"
            self._run_invalid_reason = self._run_invalid_reason or reason

        if self._run_invalid_reason is not None:
            self._last_results[self.profile] = f"discarded_{self._run_invalid_reason}"
            self._reset_run_candidate()
            return False

        run = [tuple(point) for point in self._run_bins[:usable]]
        self._runs[self.profile].append(run)
        if len(self._runs[self.profile]) > self.required_runs:
            self._runs[self.profile] = self._runs[self.profile][:self.required_runs]
        self._rebuild_curve()
        count = len(self._runs[self.profile])
        if count >= self.required_runs:
            result = "ready_truncated" if self._run_truncated else "ready"
        else:
            prefix = "accepted_truncated" if self._run_truncated else "accepted"
            result = f"{prefix}_{count}_of_{self.required_runs}"
        self._last_results[self.profile] = result
        self._reset_run_candidate()
        return True

    def on_fire(self, pressed, now, calibration_eligible=False):
        pressed, now = bool(pressed), float(now)
        with self._lock:
            if pressed == self.pressed:
                return False
            if pressed:
                self.pressed = True
                self._fire_started_at = self._last_step_at = now
                self._released_at = None
                self._pending_recovery = [0.0, 0.0]
                self._recovery_until = None
                self._baseline_pending = True
                self._reset_run_candidate()
                self._collecting = bool(calibration_eligible and not self.ready)
                if self._collecting:
                    self._run_last_target_at = now
                    self.phase = "CALIBRATING"
                elif self.ready:
                    self.phase = "REPLAY"
                else:
                    self.phase = "WAITING"
                return self._collecting

            accepted = self._finalize_run(now)
            self.pressed, self.phase, self._released_at = False, "RECOVER", now
            self._last_step_at = now
            self._pending_recovery = [0.0, 0.0]
            return accepted

    def record_output(self, dx, dy, now, target_valid, sent_ok=True):
        """记录后坐力分量；目标丢失后保留此前连续可靠前缀。"""
        now = float(now)
        with self._lock:
            if not self.pressed or not self._collecting or self._fire_started_at is None:
                return False
            if self._run_closed:
                return False
            if target_valid:
                self._run_last_target_at = now
            elif (self._run_last_target_at is None
                  or now - self._run_last_target_at > self.max_target_gap_s):
                self._run_closed = True
                self._run_truncated = True
                return False
            if not sent_ok:
                self._run_invalid_reason = self._run_invalid_reason or "send_failed"

            elapsed = max(0.0, now - self._fire_started_at)
            index = int(math.floor((elapsed + 1e-9) / self.bin_s))
            while len(self._run_bins) <= index:
                self._run_bins.append([0.0, 0.0])
                # 调度空洞代表该格没有输出；按真实零位移纳入连续前缀。
                self._run_seen_bins.add(len(self._run_bins) - 1)
            if sent_ok:
                self._run_bins[index][0] += float(dx)
                self._run_bins[index][1] += float(dy)
            self._run_seen_bins.add(index)
            self._run_reliable_bins = max(self._run_reliable_bins, index + 1)
            return True

    def _curve_integral(self, start, end):
        curve = self._curves[self.profile]
        if not curve or end <= start:
            return 0.0, 0.0
        duration = len(curve) * self.bin_s
        start, end = max(0.0, start), min(float(end), duration)
        if end <= start:
            return 0.0, 0.0
        out_x = out_y = 0.0
        # 用有限的时间格范围计算重叠量，避免浮点游标停在边界上。
        # 例如 11 * 0.06 会得到 0.659999...；旧循环会把它仍算作
        # 第 10 格，随后得到 boundary == cursor，因而永久无法前进。
        first_index = max(0, int(math.floor(start / self.bin_s)))
        last_index = min(
            len(curve) - 1,
            int(math.floor((end - 1e-12) / self.bin_s)),
        )
        for index in range(first_index, last_index + 1):
            bin_start = index * self.bin_s
            bin_end = (index + 1) * self.bin_s
            overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
            if overlap <= 0.0:
                continue
            fraction = overlap / self.bin_s
            out_x += curve[index][0] * fraction
            out_y += curve[index][1] * fraction
        return out_x, out_y

    def step(self, now, enabled):
        now = float(now)
        with self._lock:
            last, self._last_step_at = self._last_step_at, now
            if not enabled or not self.pressed or self._fire_started_at is None:
                return RecoilOutput(
                    self.phase, calibration_runs=self.calibration_count, ready=self.ready)

            start = max(0.0, (last if last is not None else now) - self._fire_started_at)
            end = max(start, now - self._fire_started_at)
            # 调度停顿不补发多个旧时间格，避免恢复后一次性大幅移动。
            start = max(start, end - self.bin_s)
            feed_x, feed_y = self._curve_integral(start, end)
            if not self._curves[self.profile] and self._baseline_pending and self._collecting:
                feed_y = float(self._profiles[self.profile].get("first_pulse_counts", 0.0))
            self._baseline_pending = False

            recovery_x, recovery_y = self._pending_recovery
            self._pending_recovery = [0.0, 0.0]
            curve_bin = int(end / self.bin_s) if self.bin_s > 0.0 else 0
            if self._collecting:
                self.phase = "CALIBRATING"
            elif self.ready and end < len(self._curves[self.profile]) * self.bin_s:
                self.phase = "REPLAY"
            else:
                self.phase = "TRACK"
            return RecoilOutput(
                self.phase, feed_x, feed_y, 0.0, 0.0,
                recovery_x, recovery_y, True, curve_bin,
                self.calibration_count, self.ready,
            )

    def is_shot_window(self, now):
        del now
        with self._lock:
            return self.pressed

    def recovery_active(self, now):
        with self._lock:
            return bool(self.pressed and self._recovery_until is not None
                        and float(now) <= self._recovery_until)

    def observe_recoil(self, jump_x_px, jump_y_px, px_per_count_x,
                       px_per_count_y, now, sample_interval_ms=60.0):
        del sample_interval_ms
        now = float(now)
        with self._lock:
            if not self.pressed:
                return False
            jump_x_px, jump_y_px = float(jump_x_px), float(jump_y_px)
            kx, ky = float(px_per_count_x), float(px_per_count_y)
            if jump_y_px >= 0.0 or kx <= 0.0 or ky <= 0.0:
                return False
            comp_x, comp_y = -jump_x_px / kx, -jump_y_px / ky
            self._pending_recovery[0] += comp_x
            self._pending_recovery[1] += comp_y
            self._recovery_until = now + self.recovery_s
            self._observation_totals[self.profile] += 1
            return True
