# -*- coding: utf-8 -*-
"""Parse structured live aim logs for the 20260906 v2 tracking sweep.

Extracts per-log config snapshots (from header lines, not comments), the
[瞄准观测] JSON rows grouped into episodes, and the [鼠标] per-tick integer
output stream used for count-level fidelity checks.
"""
import io
import json
import math
import os
import re
import statistics

MARKER = "[瞄准观测] "
MOUSE_RE = re.compile(
    r"\[鼠标\] 帧=(?P<frame>\d+) "
    r"目标=\((?P<tx>[-\d.]+),(?P<ty>[-\d.]+)\) "
    r"EMA=\((?P<ex>[-\d.]+),(?P<ey>[-\d.]+)\) "
    r"准星=\((?P<sx>[-\d.]+),(?P<sy>[-\d.]+)\) "
    r"误差=\((?P<ex2>[-+\d.]+),(?P<ey2>[-+\d.]+)\) "
    r"前导=\((?P<lx>[-+\d.]+),(?P<ly>[-+\d.]+)\) "
    r".*?速度原始=\((?P<rvx>[-+\d.]+),(?P<rvy>[-+\d.]+)\) "
    r"速度滤波=\((?P<fvx>[-+\d.]+),(?P<fvy>[-+\d.]+)\) "
    r"预测时域=(?P<hor>[-\d.]+)ms 预测模式=(?P<pmode>\S+) 原因=(?P<reason>\S+) "
    r"截图快照=\((?P<csx>[-+\d]+),(?P<csy>[-+\d]+)\) "
    r"观测瞄准=\((?P<oax>[-+\d]+),(?P<oay>[-+\d]+)\) "
    r"观测压枪=\((?P<orx>[-+\d]+),(?P<ory>[-+\d]+)\) "
    r"前馈=(?P<ff>\d) 前导抑制=(?P<sup>\d) "
    r"瞄准分量=\((?P<acx>[-+\d.]+),(?P<acy>[-+\d.]+)\) "
    r".*?瞄准整数=\((?P<aix>[-+\d]+),(?P<aiy>[-+\d]+)\) "
    r".*?净移=\((?P<nx>[-+\d]+),(?P<ny>[-+\d]+)\) "
    r"选择=(?P<sel>[^ ]+) 红点=(?P<dot>[^ ]+)"
)

DEFAULTS = {
    "aim_20260906_003353.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260906_001842.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260906_000134.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260906_000124.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260905_234511.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260905_233827.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260905_232312.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260905_231714.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260905_230935.log": dict(view_scale=0.44, view_scale_y=0.51),
    "aim_20260905_230244.log": dict(view_scale=0.44, view_scale_y=0.51),
}


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def config_snapshot(path):
    """Read the per-log parameter snapshot from header/diagnostic lines."""
    text = read_text(path)
    snap = {"src": os.path.basename(path)}
    patterns = {
        "chest_ratio": r"\[配置\].*?chest_ratio=([0-9.]+)",
        "head_ratio": r"\[配置\].*?head_ratio=([0-9.]+)",
        "frame_ms": r"\[配置\].*?frame_ms=([0-9.]+)",
        "move_steps": r"\[配置\].*?move_steps=(\d+)",
        "model": r"正在加载 (\S+\.onnx)",
        "capture_size": r"固定屏幕中心 (\d+)x\d+px",
        "output_period_ms": r"共享周期=(\d+)ms",
        "aim_cadence_ms": r"YOLO识别节拍=(\d+)ms",
        "move_chunks": r"缓出=(\d+)周期",
        "lock_filter_line": r"目标框滤波: (.+)$",
        "prediction_mode": r"移动目标策略=(\S+)",
        "recoil_line": r"\[默认垂直压枪\] (\S+)",
        "legacy_recoil_line": r"\[旧视觉后坐力\] (\S+)",
        "dot_line": r"红点闭环: (.+)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.M)
        if m:
            snap[key] = m.group(1)
    # New headers use explicit strategy fields.  Legacy headers may only have
    # the output/runtime diagnostics; those are parsed below.  Missing values
    # remain unknown instead of being assigned to the new controller.
    for key, pat in {
        "frame_schedule": r"frame_schedule=(\w+)",
        "control_strategy": r"control_strategy=(\w+)",
        "effective_prediction_mode": r"(?:prediction_mode|移动目标策略)=(\w+)",
        "output_period_ms": r"(?:output_period_ms|共享周期)=(\d+(?:\.\d+)?)ms?",
        "px_per_count": r"px_per_count=\(([-\d.]+),([-\d.]+)\)",
        "tau_ms": r"tau_ms=([-\d.]+)",
        "feedback_delay_ms": r"feedback_delay_ms=([-\d.]+)",
    }.items():
        m = re.search(pat, text, re.M)
        if m:
            if key == "px_per_count":
                snap[key] = (float(m.group(1)), float(m.group(2)))
            else:
                snap[key] = m.group(1)
    # Runtime lines are authoritative for legacy files and also make a mode
    # switch visible to the evaluator.
    for line in text.splitlines():
        _update_runtime_meta(snap, line)
    if "frame_schedule" not in snap:
        # The old fixed logs predate the explicit field.  This is a bounded
        # inference from their fixed cadence header, not a default to ASAP.
        if "frame_ms" in snap and "输出语义" not in text:
            snap["frame_schedule"] = "fixed"
            snap["frame_schedule_source"] = "legacy_frame_ms_inferred"
    for key in ("frame_schedule", "control_strategy", "effective_prediction_mode"):
        snap.setdefault(key, "unknown")
    snap.setdefault("strategy_source", "header/runtime/observation")
    m = re.search(r"固定 alpha=([0-9.]+)", snap.get("lock_filter_line", ""))
    if m:
        snap["lock_box_filter_mode"] = "fixed"
        snap["lock_box_smoothing_alpha"] = float(m.group(1))
    elif "自适应" in snap.get("lock_filter_line", ""):
        snap["lock_box_filter_mode"] = "adaptive"
    return snap


def _update_runtime_meta(meta, line):
    """Update only fields that are explicitly present on one log line."""
    m = re.search(r"frame_schedule=(\w+)", line)
    if m:
        meta["frame_schedule"] = m.group(1).lower()
    m = re.search(r"control_strategy=(\w+)", line)
    if m:
        meta["control_strategy"] = m.group(1).lower()
    m = re.search(r"(?:prediction_mode|移动目标策略)=(\w+)", line)
    if m:
        meta["effective_prediction_mode"] = m.group(1).lower()
    if "输出语义=" in line:
        if "剩余误差" in line:
            meta["control_strategy"] = "remaining_error"
        elif "最新控制速率" in line:
            meta["control_strategy"] = "rate_hold"
    m = re.search(r"(?:output_period_ms|共享周期)=(\d+(?:\.\d+)?)ms?", line)
    if m:
        meta["output_period_ms"] = float(m.group(1))
    m = re.search(r"px_per_count=\(([-\d.]+),([-\d.]+)\)", line)
    if m:
        meta["px_per_count"] = (float(m.group(1)), float(m.group(2)))
    for key, pat in (("tau_ms", r"tau_ms=([-\d.]+)"),
                     ("feedback_delay_ms", r"feedback_delay_ms=([-\d.]+)")):
        m = re.search(pat, line)
        if m:
            meta[key] = float(m.group(1))


def strategy_segments(path):
    """Return line-ranged, explicitly observed runtime strategy metadata."""
    text = read_text(path)
    base = config_snapshot(path)
    segments = []
    current = dict(base)
    # config_snapshot may contain the final runtime values; recover the
    # initial header values so a later switch can be represented correctly.
    header = text.splitlines()[0:8]
    initial = {"src": os.path.basename(path)}
    for line in header:
        _update_runtime_meta(initial, line)
        for key, pat in (("frame_ms", r"frame_ms=([0-9.]+)"),
                         ("reference_frame_ms", r"reference_frame_ms=([0-9.]+)")):
            m = re.search(pat, line)
            if m:
                initial[key] = float(m.group(1))
    if "frame_schedule" not in initial and "frame_ms" in initial:
        initial["frame_schedule"] = "fixed"
        initial["frame_schedule_source"] = "legacy_frame_ms_inferred"
    for key in ("frame_schedule", "control_strategy", "effective_prediction_mode"):
        initial.setdefault(key, "unknown")
    current = initial
    segments.append((1, dict(current)))
    for line_no, line in enumerate(text.splitlines(), 1):
        before = dict(current)
        _update_runtime_meta(current, line)
        changed = any(current.get(k) != before.get(k) for k in (
            "frame_schedule", "control_strategy", "effective_prediction_mode",
            "output_period_ms", "px_per_count", "tau_ms", "feedback_delay_ms"))
        if changed:
            segments.append((line_no, dict(current)))
    # Preserve values only available from the full header scan on the first
    # segment, without assigning an unknown strategy.
    for key, value in base.items():
        if key not in segments[0][1] and key not in ("strategy_source",):
            segments[0][1][key] = value
    return segments


def _meta_at_line(segments, line_no):
    selected = segments[0][1] if segments else {}
    for start, meta in segments:
        if start <= line_no:
            selected = meta
        else:
            break
    return selected


def parse_rows(path):
    rows = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            envelope = None
            message = line
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict) and "message" in candidate:
                    envelope = candidate
                    message = str(candidate.get("message", ""))
            except (ValueError, TypeError):
                pass
            if MARKER not in message:
                row = envelope.get("fields") if (
                    envelope is not None and
                    envelope.get("event_type") == "observation" and
                    isinstance(envelope.get("fields"), dict)
                ) else None
            else:
                try:
                    row = json.loads(message.split(MARKER, 1)[1])
                except (ValueError, TypeError):
                    row = envelope.get("fields") if (
                        envelope is not None and
                        envelope.get("event_type") == "observation" and
                        isinstance(envelope.get("fields"), dict)
                    ) else None
            if isinstance(row, dict) and "frame" in row and "capture_t" in row:
                row["_log_line"] = line_no
                if envelope is not None:
                    for key in ("run_id", "event_seq", "event_type",
                                "event_time_mono", "enqueue_time_mono",
                                "write_time_mono", "session_id",
                                "observation_id", "send_id"):
                        if key in envelope:
                            row["_log_" + key] = envelope[key]
                rows.append(row)
    return rows


def inspect_log_integrity(path, rows):
    """Identify provenance problems without guessing a missing run identity.

    New JSON-envelope rows are safe to group by ``run_id``.  A legacy file
    without that field remains usable only when no concrete contamination
    signal is found; it is explicitly marked unverified in the result.
    """
    run_ids = sorted({str(row.get("_log_run_id")) for row in rows
                      if row.get("_log_run_id") not in (None, "")})
    with_ids = sum(1 for row in rows if row.get("_log_run_id") not in (None, ""))
    reasons = []
    if len(run_ids) > 1:
        reasons.append("multiple_run_id")
    if run_ids and with_ids != len(rows):
        reasons.append("mixed_run_identity")
    text = read_text(path)
    observation_lines = [i for i, line in enumerate(text.splitlines(), 1)
                         if MARKER in line]
    if observation_lines:
        first_obs = observation_lines[0]
        # A model/startup marker after observations is a concrete sign that a
        # new process was appended to an old file, not a strategy inference.
        if any(i > first_obs and ("正在加载" in line or "模型就绪" in line)
               for i, line in enumerate(text.splitlines(), 1)):
            reasons.append("startup_after_observation")
    times = [float(row.get("capture_t")) for row in rows
             if row.get("capture_t") is not None]
    if any(b + 1e-9 < a for a, b in zip(times, times[1:])):
        reasons.append("observation_time_reversed")
    sessions = [int(row["session_id"]) for row in rows
                if str(row.get("session_id", "")).lstrip("-").isdigit()]
    if any(b < a for a, b in zip(sessions, sessions[1:])):
        reasons.append("session_id_reversed")
    return {
        "run_ids": run_ids,
        "legacy_unverified": not bool(run_ids),
        "polluted": bool(reasons),
        "pollution_reasons": sorted(set(reasons)),
        "excluded_observation_count": len(rows) if reasons else 0,
    }


def parse_mouse_lines(path):
    out = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[鼠标] 帧=" not in line:
                continue
            m = MOUSE_RE.search(line)
            if not m:
                continue
            g = m.groupdict()
            out.append({
                "frame": int(g["frame"]),
                "err": (float(g["ex2"]), float(g["ey2"])),
                "lead": (float(g["lx"]), float(g["ly"])),
            "vel_raw": (float(g["rvx"]), float(g["rvy"])),
                "vel_f": (float(g["fvx"]), float(g["fvy"])),
                "horizon_ms": float(g["hor"]),
                "reason": g["reason"],
                "aim_chunk": (float(g["acx"]), float(g["acy"])),
                "aim_int": (int(g["aix"]), int(g["aiy"])),
                "net": (int(g["nx"]), int(g["ny"])),
                "obs_aim": (int(g["oax"]), int(g["oay"])),
                "sel": g["sel"], "dot": g["dot"],
            })
    return out


def group_episodes(rows):
    """Group observation rows into episodes by frame/capture reset (one aim
    session per group; the engine is reset at session start in main.py)."""
    groups = []
    current = []
    prev_f = prev_t = None
    prev_strategy = None
    for row in rows:
        f = int(row.get("frame", 0))
        t = float(row.get("capture_t", 0.0))
        strategy = row.get("_strategy_key")
        if current and (f <= prev_f or t <= prev_t or
                        (strategy is not None and prev_strategy is not None and
                         strategy != prev_strategy)):
            groups.append(current)
            current = []
        current.append(row)
        prev_f, prev_t = f, t
        prev_strategy = strategy
    if current:
        groups.append(current)
    return groups


def load_log(path, view_scale=0.44, view_scale_y=0.51):
    snap = config_snapshot(path)
    snap["view_scale"] = view_scale
    snap["view_scale_y"] = view_scale_y
    rows = parse_rows(path)
    integrity = inspect_log_integrity(path, rows)
    # File consumers may prioritize critical events, so physical file order
    # is not occurrence order.  New envelopes carry an event sequence for
    # deterministic offline reconstruction.
    if rows and all(row.get("_log_event_seq") is not None for row in rows):
        rows = sorted(rows, key=lambda row: int(row["_log_event_seq"]))
    segments = strategy_segments(path)
    running_meta = dict(segments[0][1] if segments else snap)
    for row in rows:
        meta = dict(_meta_at_line(segments, row.get("_log_line", 0)))
        # Observation JSON is the most precise source for the effective
        # prediction mode when the legacy header omitted it.
        if row.get("prediction_mode") in ("current", "arrival"):
            running_meta["effective_prediction_mode"] = row["prediction_mode"]
        meta.update({k: v for k, v in running_meta.items()
                     if k in ("effective_prediction_mode",)
                     and v != "unknown"})
        for key in ("frame_schedule", "control_strategy", "effective_prediction_mode"):
            meta.setdefault(key, "unknown")
        row["_strategy_meta"] = meta
        row["_strategy_key"] = tuple(meta.get(k, "unknown") for k in (
            "frame_schedule", "control_strategy", "effective_prediction_mode"))
    ticks = parse_mouse_lines(path)
    episodes = []
    for gi, group in enumerate(group_episodes(rows)):
        t0 = float(group[0].get("capture_t", 0.0))
        group_meta = dict(group[0].get("_strategy_meta") or snap)
        pts = []
        cum_x = cum_y = 0.0  # camera scroll in px at capture time
        for row in group:
            aim = row.get("observed_aim") or (0, 0)
            rec = row.get("observed_recoil") or (0, 0)
            dets = list(row.get("detections") or [])
            # camera scroll applies BEFORE this capture's aim error is measured
            pts.append({
                "frame": int(row["frame"]),
                "t": float(row["capture_t"]) - t0,
                "dt": float(row.get("dt", 0.0)),
                "lat_s": float(row.get("processing_latency_ms", 0.0)) / 1000.0,
                "cx_ref": tuple(float(v) for v in
                                (row.get("crosshair") or (160.0, 160.0))),
                "dets": [{"cls": int(d.get("cls", 0)),
                          "conf": float(d.get("conf", 0.0)),
                          "bbox": [float(v) for v in d["bbox"][:4]]} for d in dets],
                "detected": bool(dets),
                "raw_target_point": None,
                "target": (row.get("target") or None),
                "lock_reason": str(row.get("lock_reason", "")),
                "obs_err": tuple(map(float, row.get("observed_error") or (0, 0))),
                "filtered_error": tuple(map(float, row.get("observed_error") or (0, 0))),
                "raw_error": (
                    (float(row["target"][0]) - float(row.get("crosshair", (160, 160))[0]),
                     float(row["target"][1]) - float(row.get("crosshair", (160, 160))[1]))
                    if row.get("target") is not None else None),
                "observed_aim": tuple(map(int, row.get("observed_aim") or (0, 0))),
                "observed_recoil": tuple(map(int, row.get("observed_recoil") or (0, 0))),
                "vel_raw": tuple(map(float, row.get("velocity_raw") or (0, 0))),
                "vel_f": tuple(map(float, row.get("velocity_filtered") or (0, 0))),
                "reason": str(row.get("prediction_reason", "")),
                "horizon_ms": float(row.get("prediction_horizon_ms", 0.0)),
                "lead": tuple(map(float, row.get("lead") or (0, 0))),
                "cmd": tuple(map(float, row.get("command") or (0, 0))),
                "can_send": bool(row.get("can_send")),
                "net": (int(aim[0]) + int(rec[0]), int(aim[1]) + int(rec[1])),
                "dot_status": str(row.get("dot_status", "unknown")),
                "strategy_meta": dict(row.get("_strategy_meta") or group_meta),
            })
        episodes.append({
            "src": path, "ep": gi,
            "snap": dict(snap, **group_meta), "points": pts,
        })
    return {"snap": snap, "episodes": episodes, "ticks": ticks,
            "n_rows": len(rows), "integrity": integrity}


def fill_world(episode):
    """Compute physical (camera-free) positions and speeds per point.

    World position of capture k = camera-frame position + scroll(k), where
    scroll(k) = sum(net_j * scale for j<=k) — net_k is the count batch sent
    between capture k-1 and capture k, already applied to capture k's image.
    """
    vs_x = episode["snap"]["view_scale"]
    vs_y = episode["snap"]["view_scale_y"]
    pts = episode["points"]
    cum_x = cum_y = 0.0
    for i, p in enumerate(pts):
        cum_x += p["net"][0] * vs_x
        cum_y += p["net"][1] * vs_y
        p["wx"] = cum_x
        p["wy"] = cum_y
    # Per-point raw detection center of the chosen target.  ``box`` retains
    # the old carried-forward value for compatibility, while ``raw_box`` is
    # never carried and is the only source for velocity samples.
    last_box = None
    for i, p in enumerate(pts):
        if p["dets"]:
            tgt = p["target"]
            if tgt is not None:
                chest = episode["snap"].get("chest_ratio")
                chest = float(chest) if chest is not None else 0.2

                def aim_dist(d):
                    x1, y1, x2, y2 = d["bbox"]
                    ax = (x1 + x2) * 0.5
                    ay = (y1 + y2) * 0.5 if d["cls"] == 1 else y1 + (y2 - y1) * chest
                    return (ax - tgt[0]) ** 2 + (ay - tgt[1]) ** 2
                chosen = min(p["dets"], key=aim_dist)
            else:
                chosen = p["dets"][0]
            p["box"] = chosen["bbox"]
            p["raw_box"] = list(chosen["bbox"])
            p["cls"] = chosen["cls"]
            p["conf"] = chosen["conf"]
            last_box = chosen["bbox"]
        else:
            p["box"] = list(last_box) if last_box else None
            p["raw_box"] = None
            p["cls"] = None
            p["conf"] = None
    for i, p in enumerate(pts):
        if p["box"] is not None:
            p["bcx"] = (p["box"][0] + p["box"][2]) * 0.5
            p["bcy"] = (p["box"][1] + p["box"][3]) * 0.5
        else:
            p["bcx"] = p["bcy"] = None
        if p.get("raw_box") is not None:
            p["raw_bcx"] = (p["raw_box"][0] + p["raw_box"][2]) * 0.5
            p["raw_bcy"] = (p["raw_box"][1] + p["raw_box"][3]) * 0.5
            p["raw_target_point"] = (p["raw_bcx"], p["raw_bcy"])
        else:
            p["raw_bcx"] = p["raw_bcy"] = None
    reconstruct_velocity(episode)
    return episode


def reconstruct_velocity(episode, window_s=0.050, max_jump_px=80.0,
                          max_speed_px_s=5000.0):
    """Estimate target velocity on common windows with explicit confidence.

    The estimate uses raw detections from the same target class, compensates
    only known aim input, and rejects lock switches, gaps, carried boxes and
    implausible jumps.  Recoil/unknown camera motion is retained as a low
    confidence annotation rather than treated as a normal speed sample.
    """
    pts = episode["points"]
    sx = float(episode["snap"].get("view_scale", 0.44))
    sy = float(episode["snap"].get("view_scale_y", 0.51))
    history = []
    for p in pts:
        p["velocity_confidence"] = "none"
        p["velocity_kind"] = "unavailable"
        p["velocity_valid"] = False
        p["vx"] = p["vy"] = None
        if p.get("raw_bcx") is None:
            continue
        reason = str(p.get("lock_reason", ""))
        if (reason.startswith(("switched_", "hold_", "predict_missing"))
                or not reason.startswith(("locked", "selected"))):
            history.clear()
            continue
        cls = p.get("cls")
        prior = None
        for candidate in reversed(history):
            gap = p["t"] - candidate["t"]
            if gap >= window_s * 0.45:
                prior = candidate
                break
            if gap > window_s * 1.5:
                break
        if prior is None:
            history.append(p)
            history = history[-8:]
            continue
        dt = p["t"] - prior["t"]
        dx = p["raw_bcx"] - prior["raw_bcx"]
        dy = p["raw_bcy"] - prior["raw_bcy"]
        jump = math.hypot(dx, dy)
        if dt <= 1e-4 or dt > window_s * 1.5 or jump > max_jump_px:
            history.clear()
            history.append(p)
            continue
        # `observed_aim` is known program input; recoil remains an explicit
        # unknown disturbance and lowers confidence for normal scoring.
        aim_a = prior.get("observed_aim", (0, 0))
        aim_b = p.get("observed_aim", (0, 0))
        comp_x = (aim_b[0] - aim_a[0]) * sx
        comp_y = (aim_b[1] - aim_a[1]) * sy
        vx = (dx + comp_x) / dt
        vy = (dy + comp_y) / dt
        if math.hypot(vx, vy) > max_speed_px_s:
            history.clear()
            history.append(p)
            continue
        recoil = p.get("observed_recoil", (0, 0))
        if recoil != (0, 0):
            confidence, kind = "low", "estimated_recoil_contaminated"
        elif cls != prior.get("cls"):
            confidence, kind = "low", "estimated_class_transition"
        else:
            confidence, kind = "medium", "estimated_common_window"
            p["velocity_valid"] = True
        p["vx"], p["vy"] = vx, vy
        p["velocity_confidence"], p["velocity_kind"] = confidence, kind
        history.append(p)
        history = history[-8:]
