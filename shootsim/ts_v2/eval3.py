# -*- coding: utf-8 -*-
"""v4 evaluator: corrected metrics for the tracking_logic_v4 sweep.

Fixes over v3 (each verified by tests/test_tracking_v4.py):
  1. err_p75 is now the overall movement error: all locked observations whose
     target speed exceeds the stationary threshold (100 px/s), regardless of
     direction.  strafe_err_p75 remains the high-lateral-speed subset.  The
     two pools are different by construction.
  2. Entry timing is measured from the first valid detection:
     entry_elapsed_s = first_inner60_time - first_detection_t.  Negative or
     missing values are counted as anomalies (never as silent successes).
  3. started_inside episodes are excluded from entry success/timeout AND from
     the first_inner60 median/P75 pool; they are only counted in
     started_inside@T.
  4. Stable-phase tick attribution is interval based: a tick belongs to the
     observation whose interval [obs_t, next_obs_t) contains it; ticks more
     than MAX_STABLE_GAP after their owning observation are dropped; warmup,
     reversal and post-entry criteria unchanged.
  5. Plan/latency/reversal diagnostics unchanged from v3.
"""
import hashlib
import math
import statistics

from harness import run_episode, percentile, STRAFE_VX

STABLE_SPEED_MAX = 100.0     # px/s; speeds above this count as "moving"
STABLE_ERR_MAX = 10.0        # px; stable-phase rows must also be this close
MOVE_SPEED_MIN = STABLE_SPEED_MAX
ENTRY_THRESHOLDS_S = (0.3, 0.5, 0.8)
MIN_STREAK_FOR_ENTRY_STATS_S = 0.8
REVERSAL_MIN_V = 80.0        # px/s on both sides for a genuine reversal
LEAD_DIR_TOL_PX = 1.0
INSUFFICIENT_RATIO = 0.8
DWELL_H_RATIO = 0.6
PHASE_OFFSETS_MS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
PHASE_PERIOD_S = 0.015
MAX_STABLE_GAP_S = 0.12      # 3x the ~40ms capture cadence


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ── candidate-independent episode eligibility (from the log itself) ──────

def episode_eligibility(ep_points):
    """max consecutive-detected duration, whether the logged red dot started
    outside inner60, and the first detection time.  All are properties of the
    episode's own log rows, never of a candidate."""
    dts = [max(1e-3, b["t"] - a["t"]) for a, b in zip(ep_points, ep_points[1:])]
    median_dt = statistics.median(dts) if dts else 0.04
    best_streak = 0.0
    run_start = None
    first_det = None
    started_outside = None
    for p in ep_points:
        if p["detected"] and p["bcx"] is not None:
            if run_start is None:
                run_start = p["t"]
            best_streak = max(best_streak, p["t"] - run_start + median_dt)
            if first_det is None:
                first_det = p["t"]
                bx = p["box"]
                rx, ry = p["cx_ref"]
                if bx is not None:
                    hspan = (bx[3] - bx[1]) * DWELL_H_RATIO * 0.5
                    bcy = (bx[1] + bx[3]) * 0.5
                    started_outside = not (
                        bx[0] <= rx <= bx[2] and bcy - hspan <= ry <= bcy + hspan)
                else:
                    started_outside = True
        else:
            run_start = None
    return {
        "max_detected_streak_s": round(best_streak, 4),
        "started_outside_inner60": bool(started_outside),
        "first_detection_t": first_det,
    }


def _locked(row):
    return (row["target"] is not None
            and str(row["lock_reason"] or "").startswith("locked"))


# ── one-phase aggregation straight from episode records ─────────────────

def aggregate_one_phase(recs, ep_elig, cap_counts):
    out = {"episodes": len(recs)}
    firsts = []           # (first_inner60_time, elig) per episode
    errs = []             # overall movement error (speed > MOVE_SPEED_MIN)
    strafe_lags, strafe_errs, near_lags = [], [], []
    reasons = {}
    horizons = []
    flip_num = flip_den = 0
    micro_sq = micro_pairs = 0
    lead_wrong_n = lead_wrong_d = lead_ins_n = lead_ins_d = 0
    xzero = warmup = 0
    first_out_ms = []
    vel_ready_ms = []
    rev_ms = []
    lead_mags = []
    cap_sent = cap_sat = 0
    stable_ticks = []
    total_locked = 0
    entry_anomalies = 0
    for r, elig in zip(recs, ep_elig):
        s = r.summary
        fi = s.get("first_inner60_time")
        firsts.append((fi, elig))
        obs = r.obs
        first_det_t = None
        first_ready_t = None
        pending_rev = None
        prev_vfx = None
        # per-row stable classification (interval attribution below)
        row_is_stable = [False] * len(obs)
        row_is_reversal = [False] * len(obs)
        row_times = []
        for ri, row in enumerate(obs):
            reason = str(row["reason"])
            cmd = row["cmd"]
            locked = _locked(row)
            row_times.append(row["t"])
            if row["target"] is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
            if locked:
                total_locked += 1
                if first_det_t is None:
                    first_det_t = row["t"]
                if reason.startswith("arrival_") and first_ready_t is None:
                    first_ready_t = row["t"]
                if reason == "lock_warmup":
                    warmup += 1
            if cmd != (0.0, 0.0):
                cap_sent += 1
                if math.hypot(*cmd) >= cap_counts - 0.5:
                    cap_sat += 1
            if row["horizon_ms"] > 0:
                horizons.append(row["horizon_ms"])
            vx, vy = row["vel_raw"]
            sp = math.hypot(vx, vy)
            ex, ey = row["obs_err"]
            err = math.hypot(ex, ey)
            lead = row["lead"]
            if locked:
                moving = sp > MOVE_SPEED_MIN
                is_strafe = (abs(vx) >= STRAFE_VX and abs(vx) >= 1.5 * abs(vy)
                             and sp > 1.0)
                if moving:
                    errs.append(err)          # overall movement error: all axes
                if is_strafe:
                    lag = (ex * vx + ey * vy) / sp
                    strafe_lags.append(lag)
                    strafe_errs.append(err)
                    lead_mags.append(abs(lead[0]))
                    if lead[0] != 0.0:
                        lead_wrong_d += 1
                        lead_ins_d += 1
                        lp = (lead[0] * vx + lead[1] * vy) / sp
                        if lp < -LEAD_DIR_TOL_PX:
                            lead_wrong_n += 1
                        elif (lp > LEAD_DIR_TOL_PX and lag > 0
                                and lp < INSUFFICIENT_RATIO * lag):
                            lead_ins_n += 1
                    elif abs(vx) > 100.0:
                        xzero += 1
                elif sp > 1.0:
                    near_lags.append((ex * vx + ey * vy) / sp)
                # stable-phase rows: post-entry, warmup-free, non-reversal,
                # slow target, small error
                if (fi is not None and row["t"] >= fi
                        and reason != "lock_warmup"
                        and sp < STABLE_SPEED_MAX and err <= STABLE_ERR_MAX):
                    row_is_stable[ri] = True
            # reversal detection + recovery (x axis)
            if locked:
                vfx = row["vel_f"][0]
                if (pending_rev is None and prev_vfx is not None
                        and abs(prev_vfx) > REVERSAL_MIN_V and abs(vfx) <= 1e-9):
                    rvx = row["vel_raw"][0]
                    if rvx * prev_vfx < 0.0 and abs(rvx) > REVERSAL_MIN_V:
                        pending_rev = row["t"]
                        row_is_reversal[ri] = True
                elif (pending_rev is not None and row["t"] > pending_rev
                        and abs(vfx) > REVERSAL_MIN_V
                        and vfx * row["vel_raw"][0] > 0.0
                        and lead[0] * vfx > LEAD_DIR_TOL_PX):
                    rev_ms.append((row["t"] - pending_rev) * 1000.0)
                    pending_rev = None
                prev_vfx = vfx
        # reversal frames must never be stable samples
        for ri, is_rev in enumerate(row_is_reversal):
            if is_rev:
                row_is_stable[ri] = False
        # detection -> first nonzero output
        if first_det_t is not None:
            for tick in r.ticks:
                if tick["t"] >= first_det_t and math.hypot(*tick["counts"]) > 0:
                    first_out_ms.append((tick["t"] - first_det_t) * 1000.0)
                    break
        if first_ready_t is not None and first_det_t is not None:
            vel_ready_ms.append((first_ready_t - first_det_t) * 1000.0)
        # tick-stream metrics + interval-based stable attribution
        # (stable_ticks accumulates across records; per-record state resets here)
        last_move = last_any = None
        ri = 0
        n_obs = len(obs)
        for tick in r.ticks:
            c = tick["counts"]
            mag = math.hypot(*c)
            tt = tick["t"]
            if mag <= 0:
                continue
            if last_any is not None and mag < 20 and math.hypot(*last_any) < 20:
                micro_pairs += 1
                micro_sq += (c[0] - last_any[0]) ** 2 + (c[1] - last_any[1]) ** 2
            last_any = c
            if mag >= 20:
                if last_move is not None:
                    flip_den += 1
                    if c[0] * last_move[0] + c[1] * last_move[1] < 0:
                        flip_num += 1
                last_move = c
            # interval attribution: owning row k satisfies
            #   row_times[k] <= tt < row_times[k+1]  (never the future row)
            while ri + 1 < n_obs and row_times[ri + 1] <= tt:
                ri += 1
            if row_times[ri] <= tt < (row_times[ri + 1]
                                      if ri + 1 < n_obs else float("inf")):
                gap = tt - row_times[ri]
                if gap <= MAX_STABLE_GAP_S and row_is_stable[ri]:
                    stable_ticks.append(c)
    # entry metrics per threshold; timing measured from first detection
    for T in ENTRY_THRESHOLDS_S:
        elig_n = succ = timeout = started_in = censored = anomalies = 0
        elapsed_pool = []
        for (fi, e) in firsts:
            if e["max_detected_streak_s"] < T:
                censored += 1
                continue
            if not e["started_outside_inner60"]:
                started_in += 1
                continue
            fdt = e.get("first_detection_t")
            if fi is None or fdt is None:
                elig_n += 1
                timeout += 1        # visible long enough, never entered
                continue
            elapsed = fi - fdt
            if elapsed < 0:
                # timestamp inversion: never a success; counted as anomaly
                # and conservatively as timeout
                anomalies += 1
                elig_n += 1
                timeout += 1
                continue
            elig_n += 1
            elapsed_pool.append(elapsed)
            if elapsed <= T:
                succ += 1
            else:
                timeout += 1
        out[f"eligible@{T}"] = elig_n
        out[f"entry_ok@{T}"] = succ
        out[f"timeout@{T}"] = timeout
        out[f"timeout_rate@{T}"] = timeout / elig_n if elig_n else 0.0
        out[f"started_inside@{T}"] = started_in
        out[f"censored@{T}"] = censored
        out[f"entry_time_anomalies@{T}"] = anomalies
    # first-entry stats: only started-outside episodes, corrected timing
    inner = []
    for (fi, e) in firsts:
        if e["max_detected_streak_s"] < MIN_STREAK_FOR_ENTRY_STATS_S:
            continue
        if not e["started_outside_inner60"]:
            continue
        fdt = e.get("first_detection_t")
        if fi is None or fdt is None:
            continue
        elapsed = fi - fdt
        if elapsed < 0:
            continue
        inner.append(elapsed)
    out["first_inner60_median"] = statistics.median(inner) if inner else None
    out["first_inner60_p75"] = percentile(inner, .75) if inner else None
    out["first_inner60_n"] = len(inner)
    dwell_num = dwell_den = 0.0
    for r, _ in zip(recs, ep_elig):
        s = r.summary
        dwell_num += s.get("post_entry_inner60_seconds", 0.0)
        dwell_den += s.get("post_entry_observed_seconds", 0.0)
    out["dwell_inner60"] = dwell_num / dwell_den if dwell_den else 0.0
    out.update({
        "err_p50": percentile(errs, .5), "err_p75": percentile(errs, .75),
        "err_p90": percentile(errs, .9),
        "movement_samples": len(errs),
        "strafe_xlag_p50": percentile(strafe_lags, .5),
        "strafe_xlag_p75": percentile(strafe_lags, .75),
        "strafe_xlag_p90": percentile(strafe_lags, .9),
        "strafe_err_p75": percentile(strafe_errs, .75),
        "strafe_samples": len(strafe_lags),
        "lead_mag_p50": percentile(lead_mags, .5) if lead_mags else None,
        "lead_mag_p75": percentile(lead_mags, .75) if lead_mags else None,
        "near_xlag_p75": percentile(near_lags, .75),
        "total_locked_obs": total_locked,
        "flip_rate": flip_num / flip_den if flip_den else 0.0,
        "flip_pairs": flip_den,
        "reason_counts": reasons,
        "horizon_p50": percentile(horizons, .5),
        "cap_saturation_rate": cap_sat / cap_sent if cap_sent else 0.0,
        "lead_dir_wrong_rate": lead_wrong_n / max(1, lead_wrong_d),
        "lead_insufficient_rate": lead_ins_n / max(1, lead_ins_d),
        "xlead_zero_on_motion": xzero,
        "lock_warmup_frames": warmup,
        "first_output_latency_ms_median": (
            statistics.median(first_out_ms) if first_out_ms else None),
        "velocity_ready_s_median": (
            statistics.median(vel_ready_ms) if vel_ready_ms else None),
        "reversal_recovery_ms_median": (
            statistics.median(rev_ms) if rev_ms else None),
        "reversal_events": len(rev_ms),
        "micro_jitter_rms_median": (
            statistics.sqrt(micro_sq / micro_pairs) if micro_pairs else 0.0),
        "stable_cmd_p75": percentile([math.hypot(*c) for c in stable_ticks], .75)
        if stable_ticks else None,
        "stable_cmd_p95": percentile([math.hypot(*c) for c in stable_ticks], .95)
        if stable_ticks else None,
        "stable_tick_diff_p95": percentile(
            [math.hypot(b[0] - a[0], b[1] - a[1])
             for a, b in zip(stable_ticks, stable_ticks[1:])], .95)
        if len(stable_ticks) > 1 else None,
        "stable_flip_rate": (
            sum(1 for a, b in zip(stable_ticks, stable_ticks[1:])
                if abs(a[0]) >= 1 and abs(b[0]) >= 1 and a[0] * b[0] < 0)
            / max(1, sum(1 for a, b in zip(stable_ticks, stable_ticks[1:])
                         if abs(a[0]) >= 1 and abs(b[0]) >= 1))),
        "stable_jump_count": sum(
            1 for a, b in zip(stable_ticks, stable_ticks[1:])
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 40.0),
        "stable_samples": len(stable_ticks),
        "stable_reliable": len(stable_ticks) >= 50,
    })
    return out


def integrate_phases(per_phase, offsets_ms):
    keys = ("strafe_xlag_p75", "err_p75", "strafe_err_p75",
            "first_inner60_median", "dwell_inner60", "timeout_rate@0.5",
            "flip_rate", "stable_cmd_p95", "lead_dir_wrong_rate",
            "cap_saturation_rate", "reversal_recovery_ms_median",
            "movement_samples", "strafe_samples", "stable_samples",
            "first_inner60_n")
    integrated = {"phases_ms": list(offsets_ms), "n_phases": len(per_phase),
                  "per_phase": per_phase}
    for k in keys:
        vals = [p.get(k) for p in per_phase]
        if all(v is None for v in vals):
            integrated[k] = None
            continue
        vv = [v for v in vals if v is not None]
        if k == "dwell_inner60":
            worst = min(vv)
            worst_i = vals.index(min(vv))
        else:
            worst = max(vv)
            worst_i = vals.index(max(vv))
        integrated[k] = {
            "mean": sum(vv) / len(vv),
            "p25": percentile(vv, .25),
            "p75": percentile(vv, .75),
            "min": min(vv), "max": max(vv),
            "worst_phase_ms": offsets_ms[worst_i],
            "per_phase_values": vals,
        }
    return integrated


def evaluate_v3(sim_cfg, baseline_cfg, episodes, ep_elig, phases=None,
                collect_dets=False, tracker_hook=None,
                phase_offsets_ms=PHASE_OFFSETS_MS, engine_factory=None):
    """Run every episode at each phase offset; return the phase-integrated
    metrics dict.  ep_elig[i] = logged eligibility of episodes[i].
    tracker_hook(episode_index, phase_offset_ms) may install instrumentation;
    engine_factory(main_cfg) may swap the engine per run."""
    cap_counts = float((baseline_cfg.get("aim_control") or {}).get(
        "unit_max_counts", 280.3026))
    per_phase = []
    for off in phase_offsets_ms:
        recs = []
        for i, ep in enumerate(episodes):
            ph = None if phases is None else phases[i]
            if ph is not None and off:
                ph = (ph + off / 1000.0) % PHASE_PERIOD_S
            recs.append(run_episode(sim_cfg, baseline_cfg, ep, ep_id=i,
                                    output_phase_s=ph,
                                    collect_dets=collect_dets,
                                    on_tracker=tracker_hook(i, off) if tracker_hook else None,
                                    engine_factory=engine_factory))
        per_phase.append(aggregate_one_phase(recs, ep_elig, cap_counts))
    return integrate_phases(per_phase, phase_offsets_ms)
