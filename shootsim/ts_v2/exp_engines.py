# -*- coding: utf-8 -*-
"""Experimental v3 tracking engines (sim-only ablation bed).

V3Engine subclasses the real MainEngine; its unit-mode branch is a faithful
copy of aim_engine.MainEngine.compute with ONLY the per-axis velocity update
replaced by a pluggable V3Predictor.  A parity test
(tests/test_tracking_v3.py::test_v3engine_legacy_mode_matches_main_engine)
asserts bit-equal commands against the parent engine in legacy mode, so the
copy cannot drift silently.

Known copy limitations (all outside the swept configuration space):
  - `_unit_damped` damped-gain branch is not reproduced (production runs
    damped=false; guarded below);
  - `unit.auto_cal` online calibration block is not reproduced (production
    runs auto_cal=false).

Predictor capabilities (task §六):
  A  feedforward/feedback separation  (ff_gain, dedicated latency, tau)
  B  alpha-beta state estimation      (position+velocity, innovation gate)
  C  confidence-gated prediction      (speed band, rise frames, blank frames)
  D  reversal state machine           (confirm frames, keep fraction,
                                       horizon scale, post-reversal tau boost)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from aim_engine import MainEngine  # noqa: E402

_VEL_ALPHA = 0.45
_VEL_CONFIRM = 3
OUTLIER_PX_S = 2400.0
REVERSAL_MIN_V = 80.0

LEGACY_PARAMS = {
    "vel_tau": None,          # None → copy engine prediction_vel_tau
    "ff_gain": 1.0,
    "latency_s": None,        # None → engine horizon formula
    "reversal_confirm_frames": 0,
    "reversal_keep_frac": 0.0,
    "post_reval_horizon_scale": 1.0,
    "post_reval_frames": 0,
    "post_reval_tau_scale": 1.0,
    "post_reval_boost_frames": 0,
    "conf_enable": False,
    "ab_enable": False,
}


class V3Predictor:
    """Per-axis velocity/prediction state with the §六 variants."""

    def __init__(self, params):
        p = dict(params or {})
        self.vel_tau = p.get("vel_tau")
        self.ff_gain = float(p.get("ff_gain", 1.0))
        self.latency_s = p.get("latency_s")            # None → engine horizon
        self.reversal_confirm = int(p.get("reversal_confirm_frames", 0))
        self.reversal_keep = float(p.get("reversal_keep_frac", 0.0))
        self.post_rev_horizon_scale = float(p.get("post_reval_horizon_scale", 1.0))
        self.post_rev_frames = int(p.get("post_reval_frames", 0))
        self.post_rev_tau_scale = float(p.get("post_reval_tau_scale", 1.0))
        self.post_rev_boost_frames = int(p.get("post_reval_boost_frames", 0))
        self.conf_enable = bool(p.get("conf_enable", False))
        self.conf_speed_low = float(p.get("conf_speed_low", 150.0))
        self.conf_speed_high = float(p.get("conf_speed_high", 700.0))
        self.conf_rise = max(1, int(p.get("conf_rise_frames", 3)))
        self.conf_blank = int(p.get("conf_blank_frames", 1))
        self.ab_enable = bool(p.get("ab_enable", False))
        self.ab_alpha = float(p.get("ab_alpha", 0.5))
        self.ab_beta = float(p.get("ab_beta", 0.25))
        self.ab_gate = float(p.get("ab_innov_gate_px", 30.0))
        self.axes = [self._fresh_axis() for _ in range(2)]

    @staticmethod
    def _fresh_axis():
        return {"v": 0.0, "valid": False, "frames": 0, "opposite_frames": 0,
                "post_rev_left": 0, "same_dir": 0, "conf": 0.0,
                "blank_left": 0, "pos": 0.0, "vel_ab": 0.0, "ab_init": False,
                "horizon_scale": 1.0, "lead_scale": 1.0}

    def reset(self):
        self.axes = [self._fresh_axis() for _ in range(2)]

    def update_axis(self, axis, raw_v, dt, engine_tau, p_obs=None, own_shift=0.0):
        """Feed one raw physical velocity sample (plus, for the alpha-beta
        variant, the actual camera-frame position observation p_obs and the
        camera scroll own_shift applied over this interval).

        Returns (v_used, valid).  The per-axis horizon/lead scales are left
        in self.axes[axis] for the engine to read."""
        st = self.axes[axis]
        st["horizon_scale"] = 1.0
        st["lead_scale"] = self.ff_gain
        tau = self.vel_tau if self.vel_tau is not None else engine_tau
        # outlier: absurd velocity → reconfirm this axis (legacy semantics)
        if abs(raw_v) > OUTLIER_PX_S:
            st["v"] = 0.0
            if axis != 0 or not st["valid"]:
                st["valid"] = False
                st["frames"] = 0
            st["same_dir"] = 0
            st["conf"] = 0.0
            return st["v"], st["valid"]
        legacy_reversal = (st["valid"] and raw_v * st["v"] < 0.0
                           and abs(raw_v) > REVERSAL_MIN_V
                           and abs(st["v"]) > REVERSAL_MIN_V)
        if self.reversal_confirm > 0 and legacy_reversal:
            # D-variant: require N consecutive opposite samples before
            # disturbing the estimate; a single opposite frame is noise.
            st["opposite_frames"] += 1
            if st["opposite_frames"] < self.reversal_confirm:
                return st["v"], st["valid"]
        if legacy_reversal or (self.reversal_confirm > 0
                               and st["opposite_frames"] >= self.reversal_confirm):
            st["opposite_frames"] = 0
            if self.reversal_keep > 0.0 and abs(raw_v) > 1e-9:
                # D: keep part of the new raw velocity instead of zeroing
                st["v"] = raw_v * self.reversal_keep
                st["valid"] = True
                st["frames"] = max(st["frames"], 2)
            else:
                # legacy: zero the estimate; X keeps its ready state
                st["v"] = 0.0
                if axis != 0:
                    st["valid"] = False
                    st["frames"] = 0
            st["same_dir"] = 0
            st["conf"] = 0.0
            # conf blanking: the confirmation output and (blank-1) further
            # outputs carry no feedforward.  blank=0 → no blanking at all.
            st["blank_left"] = self.conf_blank if self.conf_enable else 0
            if st["blank_left"] > 0:
                st["lead_scale"] = 0.0
                st["blank_left"] -= 1
            st["post_rev_left"] = self.post_rev_frames
            st["horizon_scale"] = self.post_rev_horizon_scale
            return st["v"], st["valid"]
        st["opposite_frames"] = 0
        # adaptive tau right after a confirmed reversal (trust new obs more)
        if st["post_rev_left"] > 0:
            if self.post_rev_boost_frames > 0:
                tau = max(0.02, tau * self.post_rev_tau_scale)
            st["post_rev_left"] -= 1
        if self.ab_enable:
            # position-referenced alpha-beta: the measurement z is the actual
            # camera-frame detection centre (p_obs); the prediction accounts
            # for the camera scroll (own_shift) applied over this interval.
            if p_obs is None:
                # no observation this interval: coast on the current state
                st["pos"] += st["vel_ab"] * dt
                st["v"] = st["vel_ab"]
                return st["v"], st["valid"]
            if not st["ab_init"]:
                st["pos"] = p_obs
                st["vel_ab"] = raw_v
                st["ab_init"] = True
                st["valid"] = True
                st["frames"] = max(st["frames"], 2)
                st["v"] = st["vel_ab"]
                return st["v"], True
            p_pred = st["pos"] + st["vel_ab"] * dt - own_shift
            innov = p_obs - p_pred
            if abs(innov) > self.ab_gate:
                # innovation gate: snap to the measurement, re-estimate the
                # velocity from the raw physical sample
                st["pos"] = p_obs
                st["vel_ab"] = raw_v
            else:
                st["pos"] = p_pred + self.ab_alpha * innov
                st["vel_ab"] = st["vel_ab"] + self.ab_beta * innov / max(1e-4, dt)
            st["v"] = st["vel_ab"]
            st["valid"] = True
            st["frames"] += 1
            return st["v"], True
        # EMA velocity (legacy form)
        av = 1.0 - math.exp(-dt / max(0.02, tau))
        st["v"] += (raw_v - st["v"]) * av
        st["frames"] += 1
        st["valid"] = st["frames"] >= 2
        if self.conf_enable:
            if raw_v * st["v"] >= 0.0 and abs(raw_v) > self.conf_speed_low:
                st["same_dir"] += 1
            elif raw_v * st["v"] < 0.0 and abs(raw_v) > REVERSAL_MIN_V:
                st["same_dir"] = 0
            conf = min(1.0, st["same_dir"] / float(self.conf_rise))
            speed = abs(raw_v)
            if speed <= self.conf_speed_low:
                band = 0.0
            elif speed >= self.conf_speed_high:
                band = 1.0
            else:
                band = ((speed - self.conf_speed_low)
                        / max(1.0, self.conf_speed_high - self.conf_speed_low))
            if st["blank_left"] > 0:
                # blank window: suppress feedforward only; position loop,
                # lock state and velocity state are untouched
                st["blank_left"] -= 1
                st["conf"] = conf * band
                st["lead_scale"] = 0.0
            else:
                st["conf"] = conf * band
                st["lead_scale"] = self.ff_gain * st["conf"]
        return st["v"], st["valid"]


class V3Engine(MainEngine):
    """MainEngine with a pluggable v3 predictor (unit mode only)."""

    def __init__(self, config, predictor_params=None):
        self.v3 = None  # placeholder before the parent's __init__ reset
        super().__init__(config)
        params = dict(predictor_params or {})
        if params.get("vel_tau") is None:
            params["vel_tau"] = self.prediction_vel_tau
        self.v3 = V3Predictor(params)
        self.plan_stats = {"publish_calls": 0, "replaced_with_pending": 0}
        if self._unit_damped:
            raise ValueError("V3Engine requires unit.damped=false (production value)")

    def reset(self):
        super().reset()
        if self.v3 is not None:
            self.v3.reset()

    def _reset_unit_prediction(self):
        super()._reset_unit_prediction()
        if self.v3 is not None:
            self.v3.reset()

    def compute(self, dets, crosshair, dt, recoil_recovery=False,
                processing_latency_s=0.0, recoil_feedforward_active=False):
        if self.aim_mode != "unit":
            return super().compute(dets, crosshair, dt,
                                   recoil_recovery=recoil_recovery,
                                   processing_latency_s=processing_latency_s,
                                   recoil_feedforward_active=recoil_feedforward_active)
        self._img_center[0], self._img_center[1] = crosshair
        self._now += dt
        self._frame_dt = max(1e-4, float(dt))

        target = self.find_target(dets)
        _obs_net_x, _obs_net_y = self._net_since_observation
        self._net_since_observation[0] = self._net_since_observation[1] = 0.0

        if target is not None:
            if self._no_target_frames >= 2:
                self._target_ema[0] = target[0]
                self._target_ema[1] = target[1]
                self._target_ema_init[0] = True
                self._vel_valid[0] = False
                self._vel_frames[0] = 0
                self._target_vel[0] = 0.0
                self._target_vel[1] = 0.0
            self._no_target_frames = 0
            _e_dx = target[0] - self._target_ema[0]
            _e_dy = target[1] - self._target_ema[1]
            _e_d = math.hypot(_e_dx, _e_dy)
            if not self._target_ema_init[0]:
                self._target_ema[0] = target[0]
                self._target_ema[1] = target[1]
                self._target_ema_init[0] = True
            elif self.ema_max_step > 0 and _e_d > self.ema_max_step:
                self._target_ema[0] += _e_dx / _e_d * self.ema_max_step
                self._target_ema[1] += _e_dy / _e_d * self.ema_max_step
            else:
                a = max(0.0, min(1.0, self.target_ema_alpha))
                self._target_ema[0] = self._target_ema[0] * a + target[0] * (1.0 - a)
                self._target_ema[1] = self._target_ema[1] * a + target[1] * (1.0 - a)

            _new_cls = self.locked_cls[0]
            if self._prev_cls[0] is not None and _new_cls != self._prev_cls[0]:
                self._vel_valid[0] = False
                self._vel_frames[0] = 0
                self._target_vel[0] = 0.0
                self._target_vel[1] = 0.0
                self._reset_unit_prediction()
                self.v3.reset()
            self._prev_cls[0] = _new_cls

            if self._raw_motion_point is not None:
                _bcn = self._raw_motion_point
            elif self.locked_box[0] is not None:
                _bcn = ((self.locked_box[0][0] + self.locked_box[0][2]) * 0.5,
                        (self.locked_box[0][1] + self.locked_box[0][3]) * 0.5)
            else:
                _bcn = (target[0], target[1])
            _bbH = max(0.0, self.locked_box[0][3] - self.locked_box[0][1]) if self.locked_box[0] is not None else 0.0
            _vel_dz = max(self.vel_deadzone, _bbH * self.vel_deadzone_frac)
            if self._prev_target[0] is not None:
                # v3: raw physical velocity feeds the pluggable predictor
                _raw_vx = ((_bcn[0] - self._prev_target[0][0])
                           + _obs_net_x * self.view_scale) / max(1e-4, dt)
                _raw_vy = ((_bcn[1] - self._prev_target[0][1])
                           + _obs_net_y * self.view_scale_y) / max(1e-4, dt)
                self._unit_raw_velocity[0] = _raw_vx
                self._unit_raw_velocity[1] = _raw_vy
                own_shifts = (_obs_net_x * self.view_scale,
                              _obs_net_y * self.view_scale_y)
                for axis, raw_v in enumerate((_raw_vx, _raw_vy)):
                    v_used, valid = self.v3.update_axis(
                        axis, raw_v, dt, self.prediction_vel_tau,
                        p_obs=_bcn[axis], own_shift=own_shifts[axis])
                    self._unit_velocity[axis] = self.v3.axes[axis]["v"]
                    self._unit_velocity_valid_axes[axis] = valid
                    self._unit_velocity_frames_axes[axis] = self.v3.axes[axis]["frames"]
                self._unit_velocity_valid = any(self._unit_velocity_valid_axes)
                self._unit_velocity_frames = max(self._unit_velocity_frames_axes)
                # legacy velocity models kept for lock-advance / diagnostics
                _vx = (_bcn[0] - self._prev_target[0][0]) + self._own_vel[0] * self.view_scale
                _vy = (_bcn[1] - self._prev_target[0][1]) + self._own_vel[1] * self.view_scale_y
                if math.hypot(_vx, _vy) < _vel_dz:
                    _vx = _vy = 0.0
                if math.hypot(_vx, _vy) <= max(self.ema_max_step, 1.0):
                    if self._vel_valid[0]:
                        self._target_vel[0] += (_vx - self._target_vel[0]) * (1.0 - _VEL_ALPHA)
                        self._target_vel[1] += (_vy - self._target_vel[1]) * (1.0 - _VEL_ALPHA)
                    else:
                        self._vel_frames[0] += 1
                        self._target_vel[0] += (_vx - self._target_vel[0]) / self._vel_frames[0]
                        self._target_vel[1] += (_vy - self._target_vel[1]) / self._vel_frames[0]
                        if self._vel_frames[0] >= _VEL_CONFIRM:
                            self._vel_valid[0] = True
                else:
                    self._vel_valid[0] = False
                    self._vel_frames[0] = 0
                    self._target_vel[0] = 0.0
                    self._target_vel[1] = 0.0
            if self._prev_target[0] is not None:
                _lw_x = (_bcn[0] - self._prev_target[0][0]) + self._own_vel[0] * self.view_scale
                _lw_y = (_bcn[1] - self._prev_target[0][1]) + self._own_vel[1] * self.view_scale_y
                if math.hypot(_lw_x, _lw_y) > self.ema_max_step + 1.0:
                    _lw_x = _lw_y = 0.0
                _alw = 1.0 - math.exp(-dt / max(1e-3, self.vel_lw_t))
                self._vel_lw[0] += (_lw_x - self._vel_lw[0]) * _alw
                self._vel_lw[1] += (_lw_y - self._vel_lw[1]) * _alw
                _lw_floor = max(self.vel_lw_floor, _bbH * self.vel_lw_floor_frac)
                if math.hypot(self._vel_lw[0], self._vel_lw[1]) >= _lw_floor:
                    self._vel_lw_valid = True
                else:
                    self._vel_lw_valid = False
                    self._vel_lw[0] = 0.0
                    self._vel_lw[1] = 0.0
            else:
                self._vel_lw_valid = False
                self._vel_lw[0] = 0.0
                self._vel_lw[1] = 0.0
            self._prev_target[0] = _bcn
        else:
            self._no_target_frames += 1
            if self.locked_box[0] is None:
                self._vel_valid[0] = False
                self._vel_frames[0] = 0
                self._target_vel[0] = 0.0
                self._target_vel[1] = 0.0
                self._vel_lw_valid = False
                self._vel_lw[0] = 0.0
                self._vel_lw[1] = 0.0
                self._prev_target[0] = None
                self._prev_cls[0] = None

        out = {
            "can_send": False, "in_deadzone": False, "dx": 0.0, "dy": 0.0,
            "cap_hint": None, "raw_dx": 0.0, "raw_dy": 0.0, "deadzone_px": 0.0,
            "target": target, "lead": (0.0, 0.0), "diag_gain": 1.0,
            "sens": 0.0, "gain_clamped": False,
            "recoil_recovery": False,
            "prediction_suppressed": False,
            "observed_error": (0.0, 0.0),
            "velocity_raw": tuple(self._unit_raw_velocity),
            "velocity_filtered": tuple(self._unit_velocity),
            "prediction_horizon_ms": 0.0,
            "prediction_reason": "current_mode" if self.prediction_mode == "current" else "velocity_unready",
            "prediction_mode": self.prediction_mode,
            "lock_filter_alpha": self.lock_filter_alpha_current,
        }

        # ── unit mode: full-control branch (copy of MainEngine.compute) ──
        out["lead"] = (0.0, 0.0)
        if target is None:
            self._prev_sent[0] = self._prev_sent[1] = 0.0
            self._unit_tp_ok = False
            self._ease_left = self.ease_frames
            out["can_send"] = False
            return out
        if self._unit_auto_cal:
            _ue_x = target[0] - self._img_center[0]
            _ue_y = target[1] - self._img_center[1]
        else:
            _reset_filter = not self._unit_tp_ok
            if _reset_filter:
                self._ease_left = self.ease_frames
            _previous = tuple(self._unit_tp)
            self._filter_unit_target(target, dt, reset=_reset_filter)
            if (not _reset_filter and self._unit_target_filter_mode == "legacy"
                    and math.hypot(target[0] - _previous[0],
                                   target[1] - _previous[1]) > 40.0):
                self._ease_left = self.ease_frames
            _ue_x = self._unit_tp[0] - self._img_center[0]
            _ue_y = self._unit_tp[1] - self._img_center[1]
        out["observed_error"] = (_ue_x, _ue_y)
        if (self.unit_prediction_enabled and self.prediction_mode == "arrival"
                and any(self._unit_velocity_valid_axes)
                and self.locked_frame_count[0] >= self.prediction_lock_frames
                and self.diag_reason is not None
                and not self.diag_reason.startswith(
                    ("hold_", "switched_", "predict_missing"))):
            _horizon = self._v3_horizon(processing_latency_s, dt)
            hscale_x = self.v3.axes[0]["horizon_scale"]
            hscale_y = self.v3.axes[1]["horizon_scale"]
            lead_scale_x = self.v3.axes[0]["lead_scale"]
            lead_scale_y = self.v3.axes[1]["lead_scale"]
            _lead_x = (self._unit_velocity[0] * _horizon * hscale_x * lead_scale_x
                       if self._unit_velocity_valid_axes[0] else 0.0)
            _lead_y = (self._unit_velocity[1] * _horizon * hscale_y * lead_scale_y
                       if self._unit_velocity_valid_axes[1] else 0.0)
            out["prediction_horizon_ms"] = _horizon * 1000.0
            _box_w = (max(0.0, self.locked_box[0][2] - self.locked_box[0][0])
                      if self.locked_box[0] is not None else 0.0)
            _lead_cap = self.prediction_cap_px
            if _box_w > 0.0 and self.prediction_box_ratio > 0.0:
                _lead_cap = min(_lead_cap, _box_w * self.prediction_box_ratio)
            if recoil_feedforward_active:
                _lead_y = 0.0
                _lead_cap = min(_lead_cap, self.recoil_prediction_cap_px)
                if _box_w > 0.0 and self.recoil_prediction_box_ratio > 0.0:
                    _lead_cap = min(
                        _lead_cap, _box_w * self.recoil_prediction_box_ratio)
                out["prediction_suppressed"] = True
                out["prediction_mode"] = "recoil_horizontal"
                out["prediction_reason"] = "recoil_horizontal_limited"
            else:
                if all(self._unit_velocity_valid_axes):
                    out["prediction_reason"] = "arrival_active"
                elif self._unit_velocity_valid_axes[0]:
                    out["prediction_reason"] = "arrival_partial_x"
                else:
                    out["prediction_reason"] = "arrival_partial_y"
            _lead_mag = math.hypot(_lead_x, _lead_y)
            if _lead_cap > 0.0 and _lead_mag > _lead_cap:
                _ls = _lead_cap / _lead_mag
                _lead_x *= _ls
                _lead_y *= _ls
            _ue_x += _lead_x
            _ue_y += _lead_y
            out["lead"] = (_lead_x, _lead_y)
        elif self.prediction_mode == "current" or not self.unit_prediction_enabled:
            out["prediction_reason"] = "current_mode"
        elif self.diag_reason is not None and self.diag_reason.startswith(("hold_", "switched_")):
            out["prediction_reason"] = "lock_state_blocked"
        elif self.locked_frame_count[0] < self.prediction_lock_frames:
            out["prediction_reason"] = "lock_warmup"
        out["raw_dx"], out["raw_dy"] = _ue_x, _ue_y
        _dz = self._unit_dz
        _stop_dz = self._unit_stop_dz
        out["deadzone_px"] = _stop_dz
        _em = math.hypot(_ue_x, _ue_y)
        if _em <= _stop_dz:
            out["in_deadzone"] = True
            self._last_send_time = self._now
            self._prev_sent[0] = self._prev_sent[1] = 0.0
            out["can_send"] = True
            return out
        if self._unit_continuous_gain:
            _stage_gains = []
            for axis, error in enumerate((_ue_x, _ue_y)):
                _near = self._unit_near_gain_xy[axis]
                _far = self._unit_far_gain_xy[axis]
                _ratio = abs(error) / (abs(error) + self._unit_gain_softness[axis])
                _stage_gains.append(_near + (_far - _near) * _ratio)
        else:
            _stage_gain = (self._unit_near_gain if _em <= self._unit_near_radius
                           else self._unit_far_gain)
            _stage_gains = [_stage_gain, _stage_gain]
        _soft_scale = 1.0
        if _stop_dz < _em < _dz:
            _soft_t = (_em - _stop_dz) / max(1e-6, _dz - _stop_dz)
            _soft_scale = _soft_t * _soft_t * (3.0 - 2.0 * _soft_t)
        _gains = list(_stage_gains)
        _gains[0] *= _soft_scale
        _gains[1] *= _soft_scale
        out["dx"] = _ue_x / self._unit_k[0] * _gains[0]
        _recoil_y = bool(recoil_recovery and _ue_y > 0.0)
        _y_gain = 1.0 if _recoil_y else _gains[1]
        out["dy"] = _ue_y / self._unit_k[1] * _y_gain
        _damp_x = _damp_y = 1.0
        if (self._last_net_move[0] > 0 and out["dx"] < 0) or \
                (self._last_net_move[0] < 0 and out["dx"] > 0):
            _damp_x = self.reversal_damp
        if not _recoil_y and ((self._last_net_move[1] > 0 and out["dy"] < 0) or \
                (self._last_net_move[1] < 0 and out["dy"] > 0)):
            _damp_y = self.reversal_damp
        out["dx"] *= _damp_x
        out["dy"] *= _damp_y
        if self.diag_reason is not None and self.diag_reason.startswith("predict_missing"):
            _miss_scale = self._missing_decay ** max(1, self.lock_miss_count[0])
            out["dx"] *= _miss_scale
            out["dy"] *= _miss_scale
            _miss_mag = math.hypot(out["dx"], out["dy"])
            if self._missing_max_counts > 0.0 and _miss_mag > self._missing_max_counts:
                _miss_cap = self._missing_max_counts / _miss_mag
                out["dx"] *= _miss_cap
                out["dy"] *= _miss_cap
            out["prediction_reason"] = "missing_decay"
            out["missing_duration_ms"] = getattr(self, "_miss_time_s", 0.0) * 1000.0
        self._last_net_move[0], self._last_net_move[1] = out["dx"], out["dy"]
        out["sens"] = 1.0 / self._unit_k[0] * _gains[0]
        out["diag_gain"] = max(_gains)
        out["recoil_recovery"] = _recoil_y
        out["gain_clamped"] = bool(
            self.max_total_gain > 0 and max(_stage_gains) > self.max_total_gain)
        out["can_send"] = True
        self._prev_sent[0], self._prev_sent[1] = out["dx"], out["dy"]
        self._prev_err[0], self._prev_err[1] = _ue_x, _ue_y
        return out

    # ── v3 helpers ────────────────────────────────────────────────────────
    def _v3_horizon(self, processing_latency_s, dt):
        if self.v3.latency_s is not None:
            return max(0.005, min(0.20, float(self.v3.latency_s)))
        return max(self.prediction_min_s, min(
            self.prediction_max_s, float(processing_latency_s) + dt * 0.75))


def make_v3_tracker_factory(predictor_params):
    """Return a tracker factory for harness.run_episode that swaps the
    engine for a V3Engine with the given predictor params."""
    def factory(tracker):
        old = tracker.engine
        v3 = V3Engine(tracker.engine_config_ref, predictor_params)
        tracker.engine = v3
        _ = old
        return v3
    return factory
