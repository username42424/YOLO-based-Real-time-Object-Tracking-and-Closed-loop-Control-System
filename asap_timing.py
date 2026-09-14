# -*- coding: utf-8 -*-
"""Small, side-effect-free timing helpers shared by the live and sim paths."""

from collections import deque
from copy import deepcopy
from statistics import median


def clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def bounded_rate_integration_dt(actual_dt_s, output_period_s, max_factor=1.5):
    """Return (dt used for rate integration, discarded dt, is_stall).

    A delayed output worker must not repay missed wall-clock time in one mouse
    packet.  The caller still records the real interval separately for
    diagnostics.
    """
    actual = max(0.0, float(actual_dt_s))
    period = max(1e-6, float(output_period_s))
    limit = period * max(1.0, float(max_factor))
    integrated = min(actual, limit)
    discarded = max(0.0, actual - integrated)
    return integrated, discarded, actual > limit + 1e-6


class ObservationFreshness:
    """Track recent observation cadence and derive a bounded ASAP TTL."""

    def __init__(self, reference_dt, minimum_s=0.040, maximum_s=0.060,
                 multiplier=2.5, history=9, ewma_alpha=0.35):
        self.reference_dt = max(1e-4, float(reference_dt))
        self.minimum_s = max(0.0, float(minimum_s))
        self.maximum_s = max(self.minimum_s, float(maximum_s))
        self.multiplier = max(0.0, float(multiplier))
        self.ewma_alpha = clamp(ewma_alpha, 0.0, 1.0)
        self._dts = deque(maxlen=max(1, int(history)))
        self.reset()

    def reset(self):
        self._dts.clear()
        self._ewma = None

    def observe(self, dt_s):
        dt = max(1e-4, min(0.5, float(dt_s)))
        self._dts.append(dt)
        if self._ewma is None:
            self._ewma = dt
        else:
            self._ewma += (dt - self._ewma) * self.ewma_alpha
        return self.recent_dt

    @property
    def recent_dt(self):
        if not self._dts:
            return self.reference_dt
        # Median rejects an occasional capture/inference spike better than a
        # raw latest sample while the EWMA remains available for diagnostics.
        return float(median(self._dts))

    @property
    def ewma_dt(self):
        return self.reference_dt if self._ewma is None else float(self._ewma)

    @property
    def ttl_s(self):
        return clamp(self.recent_dt * self.multiplier,
                     self.minimum_s, self.maximum_s)


def merge_config_defaults(config, default):
    """Deep-merge defaults while preserving the legacy reference-period rule.

    ``reference_frame_ms`` was introduced after ``frame_ms``.  If it is absent
    in the source config, its compatibility value must come from that source's
    actual ``frame_ms`` rather than from a default inserted during merging.
    """
    source = config if isinstance(config, dict) else {}
    result = deepcopy(source)
    defaults = default if isinstance(default, dict) else {}
    source_unit = source.get("unit") if isinstance(source.get("unit"), dict) else {}
    default_unit = defaults.get("unit") if isinstance(defaults.get("unit"), dict) else {}

    for key, value in defaults.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(result[key], dict):
            for subkey, subvalue in value.items():
                if subkey not in result[key]:
                    result[key][subkey] = deepcopy(subvalue)

    unit = result.setdefault("unit", deepcopy(default_unit))
    if "frame_schedule" not in source_unit:
        unit["frame_schedule"] = "fixed"
    if "reference_frame_ms" not in source_unit:
        unit["reference_frame_ms"] = source_unit.get(
            "frame_ms", default_unit.get("reference_frame_ms", 22.0))
    return result
