# -*- coding: utf-8 -*-
"""Run-scoped, non-blocking logging primitives.

The control path only creates an immutable event and performs bounded
``put_nowait`` operations.  File, stdout and GUI delivery have separate
consumers and lifetimes, so a slow display cannot replace or stall a later
run's file sink.
"""
import json
import os
import queue
import threading
import time
import uuid
from collections import deque


def new_run_id():
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def is_minimal_log_message(message):
    """Return whether a lifecycle line survives the quiet debug-log mode."""
    text = str(message).strip()
    if text == "=" * 40:
        return True
    if text.startswith((
            "[引擎] 正在启动", "[引擎] 已退出",
            "[热键] 瞄准=", "[热键] 轮询器已启动",
            "[截图] 固定屏幕中心", "[日志] 本次运行日志 →")):
        return True
    if "▶ 瞄准开始" in text:
        return True
    # Keep the short GUI config-path line, but hide main.py's verbose
    # resolved-parameter line which also starts with [配置].
    return text.startswith("[配置] ") and not text.startswith("[配置] 路径=")


class GuiLogPump:
    """Bounded cross-thread GUI message queue; Tk is only touched by caller."""

    def __init__(self, maxsize=512):
        self._queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self._dropped = 0
        self._stopped_discarded = 0
        self._submitted = 0
        self._displayed = 0
        self._display_samples = deque(maxlen=2048)
        self._render_samples = deque(maxlen=2048)
        self._max_batch_chars = 0
        self._lock = threading.Lock()

    @property
    def depth(self):
        return self._queue.qsize()

    @property
    def dropped(self):
        with self._lock:
            return self._dropped

    @property
    def submitted(self):
        with self._lock:
            return self._submitted

    def submit(self, message):
        try:
            self._queue.put_nowait(message)
            with self._lock:
                self._submitted += 1
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def drain(self, max_items=80, max_ms=4.0, max_chars=12000):
        deadline = time.perf_counter() + max(0.0, float(max_ms)) / 1000.0
        rows = []
        chars = 0
        char_limit = max(1, int(max_chars))
        while len(rows) < max(0, int(max_items)) and time.perf_counter() <= deadline:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            text = str(item.get("message", item)) if isinstance(item, dict) else str(item)
            if rows and chars + len(text) > char_limit:
                self._queue.put_nowait(item)
                break
            rows.append(item)
            chars += len(text)
        with self._lock:
            self._max_batch_chars = max(self._max_batch_chars, chars)
        return rows

    def record_display(self, rows, render_ms):
        """Record Tk insertion/trim/scroll time after the UI work completes."""
        displayed_at = time.perf_counter()
        with self._lock:
            self._displayed += len(rows)
            self._render_samples.append(max(0.0, float(render_ms)) / 1000.0)
            for item in rows:
                if isinstance(item, dict) and item.get("enqueue_time_mono") is not None:
                    self._display_samples.append(max(
                        0.0, displayed_at - float(item["enqueue_time_mono"])))

    def stats(self):
        with self._lock:
            ages = sorted(self._display_samples)
            renders = sorted(self._render_samples)
            p95 = lambda values: values[min(len(values) - 1,
                                            round(0.95 * (len(values) - 1)))] if values else 0.0
            return {
                "submitted": self._submitted,
                "displayed": self._displayed,
                "dropped": self._dropped,
                "stopped_discarded": self._stopped_discarded,
                "depth": self._queue.qsize(),
                "queue_age_p95_s": p95(ages),
                "render_p95_s": p95(renders),
                "max_render_s": max(renders) if renders else 0.0,
                "max_batch_chars": self._max_batch_chars,
            }

    def clear(self):
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                with self._lock:
                    self._stopped_discarded += cleared
                return


class RunLogSink:
    """A fixed-lifetime logging sink owned by one engine invocation."""

    def __init__(self, base_dir, run_id=None, gui_submit=None,
                 file_queue_size=2048, critical_queue_size=128,
                 stdout_queue_size=512, file_write=None, stdout_write=None,
                 clock=None, wall_clock=None, stdout_detail=False,
                 gui_detail=False, stdout_enabled=True,
                 stdout_minimal=False, gui_minimal=False):
        self.run_id = str(run_id or new_run_id())
        self._base_dir = os.path.abspath(base_dir)
        self._gui_submit = gui_submit
        self._file_queue = queue.Queue(maxsize=max(1, int(file_queue_size)))
        self._critical_queue = queue.Queue(maxsize=max(1, int(critical_queue_size)))
        self._stdout_queue = queue.Queue(maxsize=max(1, int(stdout_queue_size)))
        self._file_write = file_write
        self._stdout_write = stdout_write or self._default_stdout_write
        self._stdout_detail = bool(stdout_detail)
        self._stdout_enabled = bool(stdout_enabled)
        self._stdout_minimal = bool(stdout_minimal)
        self._gui_detail = bool(gui_detail)
        self._gui_minimal = bool(gui_minimal)
        self._clock = clock or time.perf_counter
        self._wall_clock = wall_clock or time.time
        self._lock = threading.Lock()
        self._accepting = False
        self._started = False
        self._stop = threading.Event()
        self._file_thread = None
        self._stdout_thread = None
        self._file_handle = None
        self._event_seq = 0
        self._stats = {
            "produced": 0, "file_written": 0, "stdout_written": 0,
            "file_dropped": 0, "stdout_dropped": 0,
            "gui_dropped": 0, "stdout_filtered": 0, "gui_filtered": 0,
            "gui_submit_errors": 0,
            "file_errors": 0, "stdout_errors": 0,
            "file_high_water": 0, "critical_high_water": 0,
            "stdout_high_water": 0, "file_gap_count": 0,
            "max_queue_age_s": 0.0, "close_reason": "open",
            "truncated_count": 0,
            "stdout_truncated_count": 0, "file_consume_p95_s": 0.0,
            "stdout_consume_p95_s": 0.0,
        }
        self._file_queue_ages = deque(maxlen=2048)
        self._stdout_queue_ages = deque(maxlen=2048)
        self._file_consume_times = deque(maxlen=2048)
        self._stdout_consume_times = deque(maxlen=2048)
        self._started_mono = None
        self.path = None

    @staticmethod
    def _default_stdout_write(line):
        print(line, flush=True)

    def start(self):
        if self._started:
            return self
        os.makedirs(os.path.join(self._base_dir, "logs"), exist_ok=True)
        stamp = time.strftime("aim_%Y%m%d_%H%M%S")
        self.path = os.path.join(self._base_dir, "logs",
                                 "%s_%s.log" % (stamp, self.run_id[-8:]))
        self._file_handle = open(self.path, "a", encoding="utf-8", newline="")
        self._accepting = True
        self._started = True
        self._started_mono = self._clock()
        self._file_thread = threading.Thread(
            target=self._file_worker, name="RunFileLog-%s" % self.run_id,
            daemon=True)
        if self._stdout_enabled or self._stdout_minimal:
            self._stdout_thread = threading.Thread(
                target=self._stdout_worker, name="RunStdoutLog-%s" % self.run_id,
                daemon=True)
        self._file_thread.start()
        if self._stdout_thread is not None:
            self._stdout_thread.start()
        return self

    def _event_type(self, message, event_type=None):
        if event_type:
            return str(event_type)
        text = str(message)
        if text.startswith("错误") or text.startswith("[异常"):
            return "error"
        if text.startswith("[配置]"):
            return "config"
        if text.startswith("[瞄准观测]"):
            return "observation"
        if text.startswith("[发送提交]"):
            return "send_commit"
        if text.startswith("[发送]"):
            return "send"
        if text.startswith(("[日志统计]", "[输出器]")):
            return "summary"
        if text.startswith(("▶", "■", "清理", "已退出", "[日志]")):
            return "lifecycle"
        return "detail"

    def emit(self, message, event_type=None, fields=None, session_id=None,
             observation_id=None, send_id=None, event_time_mono=None):
        """Enqueue one event without waiting for any consumer."""
        produced = self._clock() if event_time_mono is None else float(event_time_mono)
        enqueue = self._clock()
        with self._lock:
            if not self._accepting:
                return False
            self._event_seq += 1
            raw_message = str(message)
            auto_fields = dict(fields or {})
            kind = self._event_type(raw_message, event_type)
            if not auto_fields and kind == "observation":
                try:
                    auto_fields = json.loads(
                        raw_message.split("[瞄准观测]", 1)[1].strip())
                except (ValueError, TypeError):
                    auto_fields = {}
            if kind == "observation" and auto_fields:
                message = ("[瞄准观测] "
                           f"observation_id={auto_fields.get('observation_id', '-') } "
                           f"target_id={auto_fields.get('target_id', '-') } "
                           f"state={auto_fields.get('state', '-')}")
            event = {
                "run_id": self.run_id,
                "event_seq": self._event_seq,
                "event_type": kind,
                "event_time_mono": float(produced),
                "enqueue_time_mono": float(enqueue),
                "event_time_wall": float(self._wall_clock()),
                "message": str(message),
            }
            if session_id is not None:
                event["session_id"] = session_id
            if observation_id is not None:
                event["observation_id"] = observation_id
            if send_id is not None:
                event["send_id"] = send_id
            if auto_fields:
                event["fields"] = auto_fields
                for key in ("session_id", "observation_id", "send_id"):
                    if key in auto_fields and key not in event:
                        event[key] = auto_fields[key]
            self._stats["produced"] += 1
        critical = event["event_type"] in {
            "error", "lifecycle", "cancel", "send_rejected", "send_failed"
        }
        target = self._critical_queue if critical else self._file_queue
        try:
            target.put_nowait(event)
            with self._lock:
                key = "critical_high_water" if critical else "file_high_water"
                self._stats[key] = max(self._stats[key], target.qsize())
        except queue.Full:
            with self._lock:
                self._stats["file_dropped"] += 1
                self._stats["file_gap_count"] += 1
        stdout_visible = (
            self._stdout_enabled and (
                self._stdout_detail or event["event_type"] in {
            "error", "lifecycle", "cancel", "send_rejected", "send_failed",
            "summary", "config", "warning"
                })
            ) or (
                self._stdout_minimal and
                is_minimal_log_message(event["message"])
            )
        if stdout_visible:
            try:
                self._stdout_queue.put_nowait(event)
                with self._lock:
                    self._stats["stdout_high_water"] = max(
                        self._stats["stdout_high_water"], self._stdout_queue.qsize())
            except queue.Full:
                with self._lock:
                    self._stats["stdout_dropped"] += 1
        else:
            with self._lock:
                self._stats["stdout_filtered"] += 1
        gui_visible = (
            self._gui_detail or event["event_type"] in {
                "error", "lifecycle", "cancel", "send_rejected", "send_failed",
                "summary", "config", "warning"
            } or (
                self._gui_minimal and
                is_minimal_log_message(event["message"])
            )
        )
        if self._gui_submit is not None and gui_visible:
            try:
                if self._gui_submit(event) is False:
                    with self._lock:
                        self._stats["gui_dropped"] += 1
            except Exception:
                with self._lock:
                    self._stats["gui_submit_errors"] += 1
        elif self._gui_submit is not None:
            with self._lock:
                self._stats["gui_filtered"] += 1
        return True

    def _format_event(self, event, written, channel):
        row = dict(event)
        row["write_time_mono"] = float(written)
        if row.get("enqueue_time_mono") is not None:
            age = max(0.0, float(written) - float(row["enqueue_time_mono"]))
            with self._lock:
                self._stats["max_queue_age_s"] = max(self._stats["max_queue_age_s"], age)
                ages = self._file_queue_ages if channel == "file" else self._stdout_queue_ages
                ages.append(age)
        return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"

    def _get_file_event(self, timeout=0.05):
        try:
            return self._critical_queue.get_nowait(), self._critical_queue
        except queue.Empty:
            pass
        try:
            return self._file_queue.get(timeout=timeout), self._file_queue
        except queue.Empty:
            return None, None

    def _file_worker(self):
        try:
            while not self._stop.is_set() or not self._critical_queue.empty() or not self._file_queue.empty():
                event, source = self._get_file_event()
                if event is None:
                    continue
                try:
                    consumed = self._clock()
                    written = self._clock()
                    line = self._format_event(event, written, "file")
                    if self._file_write is not None:
                        self._file_write(line)
                    else:
                        handle = self._file_handle
                        if handle is None:
                            raise RuntimeError("file sink closed before writer stopped")
                        handle.write(line)
                        handle.flush()
                    with self._lock:
                        self._stats["file_written"] += 1
                        self._file_consume_times.append(max(0.0, self._clock() - consumed))
                except Exception:
                    with self._lock:
                        self._stats["file_errors"] += 1
                        self._stats["file_gap_count"] += 1
                finally:
                    source.task_done()
        finally:
            # A writer that outlives the caller's bounded close window closes
            # its own handle when it really exits; it never touches a later
            # run's file handle.
            handle = self._file_handle
            if handle is not None and self._file_write is None:
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    pass
                with self._lock:
                    if self._file_handle is handle:
                        self._file_handle = None

    def _stdout_worker(self):
        while not self._stop.is_set() or not self._stdout_queue.empty():
            try:
                event = self._stdout_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                consumed = self._clock()
                self._stdout_write(self._format_event(
                    event, self._clock(), "stdout").rstrip("\n"))
                with self._lock:
                    self._stats["stdout_written"] += 1
                    self._stdout_consume_times.append(max(0.0, self._clock() - consumed))
            except Exception:
                with self._lock:
                    self._stats["stdout_errors"] += 1
            finally:
                self._stdout_queue.task_done()

    def close(self, timeout_s=1.0):
        """Stop admission, detach GUI, and close only this run's resources."""
        with self._lock:
            if not self._started:
                return dict(self._stats, closed=True)
            self._accepting = False
            self._gui_submit = None
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while (not self._critical_queue.empty() or not self._file_queue.empty()) and \
                time.perf_counter() < deadline:
            time.sleep(0.001)
        self._stop.set()
        remaining = max(0.0, deadline - time.perf_counter())
        if self._file_thread is not None:
            self._file_thread.join(timeout=remaining)
        remaining = max(0.0, deadline - time.perf_counter())
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=remaining)
        with self._lock:
            if not self._critical_queue.empty() or not self._file_queue.empty():
                self._stats["close_reason"] = "truncated_timeout"
                remaining_file = (self._critical_queue.qsize() +
                                  self._file_queue.qsize())
                self._stats["truncated_count"] += remaining_file
                self._stats["file_gap_count"] += remaining_file
            else:
                self._stats["close_reason"] = "closed"
            remaining_stdout = self._stdout_queue.qsize()
            if remaining_stdout:
                self._stats["stdout_truncated_count"] += remaining_stdout
                self._stats["stdout_dropped"] += remaining_stdout
        while True:
            try:
                self._stdout_queue.get_nowait()
                self._stdout_queue.task_done()
            except queue.Empty:
                break
        # Do not close a handle while a bounded/slow writer still owns it.
        # Its finally block closes it when the old run actually terminates.
        if self._file_handle is not None and \
                (self._file_thread is None or not self._file_thread.is_alive()):
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None
        result = self.stats()
        result["close_elapsed_s"] = max(0.0, time.perf_counter() - (deadline - max(0.0, float(timeout_s))))
        result["closed"] = (
            (self._file_thread is None or not self._file_thread.is_alive()) and
            (self._stdout_thread is None or not self._stdout_thread.is_alive()))
        return result

    def stats(self):
        with self._lock:
            result = dict(self._stats)
            file_ages = sorted(self._file_queue_ages)
            stdout_ages = sorted(self._stdout_queue_ages)
            file_consumes = sorted(self._file_consume_times)
            stdout_consumes = sorted(self._stdout_consume_times)
            p95 = lambda values: values[min(len(values) - 1,
                                            round(0.95 * (len(values) - 1)))] if values else 0.0
            result["file_queue_age_p95_s"] = p95(file_ages)
            result["stdout_queue_age_p95_s"] = p95(stdout_ages)
            result["queue_age_p95_s"] = result["file_queue_age_p95_s"]
            result["file_consume_p95_s"] = p95(file_consumes)
            result["stdout_consume_p95_s"] = p95(stdout_consumes)
            result["file_consume_p95_s"] = result["file_consume_p95_s"]
            if self._started_mono is not None:
                result["elapsed_s"] = max(0.0, self._clock() - self._started_mono)
                elapsed = max(1e-9, result["elapsed_s"])
                result["production_rate_hz"] = result["produced"] / elapsed
                result["file_write_rate_hz"] = result["file_written"] / elapsed
                result["stdout_write_rate_hz"] = result["stdout_written"] / elapsed
        result.update({
            "file_queue_depth": self._file_queue.qsize(),
            "critical_queue_depth": self._critical_queue.qsize(),
            "stdout_queue_depth": self._stdout_queue.qsize(),
        })
        return result
