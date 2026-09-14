# -*- coding: utf-8 -*-
"""Persistent named profiles for physical-mouse recoil trajectories."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
import re
import statistics
import threading


DEFAULT_PROFILE_LABEL = "默认｜重新校准5次"
PROFILE_FILE_VERSION = 1
_NAME_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9 _\-()（）]{1,40}\Z")


class ProfileStoreError(RuntimeError):
    """The profile file or one of its entries is invalid."""


def validate_profile_name(name):
    value = str(name or "").strip()
    if value in {"默认", DEFAULT_PROFILE_LABEL}:
        raise ValueError("“默认”是系统保留名称")
    if not value:
        raise ValueError("曲线名称不能为空")
    if len(value) > 40:
        raise ValueError("曲线名称不能超过40个字符")
    if _NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("名称只能包含中英文、数字、空格、-、_和括号")
    return value


def _finite_number(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileStoreError(f"{field} 不是数字") from exc
    if not math.isfinite(number):
        raise ProfileStoreError(f"{field} 不是有限数值")
    return number


def _normalise_points(points, field):
    if not isinstance(points, list) or not points:
        raise ProfileStoreError(f"{field} 不能为空")
    normalised = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ProfileStoreError(f"{field}[{index}] 必须是二维坐标")
        normalised.append([
            _finite_number(point[0], f"{field}[{index}].x"),
            _finite_number(point[1], f"{field}[{index}].y"),
        ])
    return normalised


def normalise_profile(snapshot):
    if not isinstance(snapshot, dict):
        raise ProfileStoreError("曲线配置必须是对象")
    bin_ms = _finite_number(snapshot.get("bin_ms", 60.0), "bin_ms")
    if not 20.0 <= bin_ms <= 1000.0:
        raise ProfileStoreError("bin_ms 超出允许范围")
    required_runs = int(snapshot.get("required_runs", 5))
    if required_runs != 5:
        raise ProfileStoreError("当前格式要求恰好保留5次手动记录")
    source = snapshot.get("source_runs")
    if not isinstance(source, list) or len(source) != required_runs:
        raise ProfileStoreError("source_runs 必须包含5次手动记录")
    runs = [
        _normalise_points(run, f"source_runs[{index}]")
        for index, run in enumerate(source)
    ]
    common = min(len(run) for run in runs)
    if common <= 0:
        raise ProfileStoreError("五次记录没有共同有效时长")
    curve = _normalise_points(snapshot.get("curve"), "curve")
    if len(curve) != common:
        raise ProfileStoreError("中位数曲线长度必须等于五次记录的最短长度")
    for index in range(common):
        expected = (
            statistics.median(run[index][0] for run in runs),
            statistics.median(run[index][1] for run in runs),
        )
        if (abs(curve[index][0] - expected[0]) > 1e-6
                or abs(curve[index][1] - expected[1]) > 1e-6):
            raise ProfileStoreError("中位数曲线与五次原始记录不一致")
    duration_ms = common * bin_ms
    return {
        "bin_ms": bin_ms,
        "duration_ms": duration_ms,
        "required_runs": required_runs,
        "source_runs": runs,
        "curve": curve,
    }


def infer_recoil_path(curve):
    """Return the estimated uncontrolled trend: negative cumulative input."""
    total_x = total_y = 0.0
    path = []
    for point in curve:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("curve points must be two-dimensional")
        total_x -= float(point[0])
        total_y -= float(point[1])
        path.append((total_x, total_y))
    return path


class RecoilProfileStore:
    """Atomic JSON storage; playback settings remain outside this file."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._lock = threading.RLock()

    def _empty_document(self):
        return {"version": PROFILE_FILE_VERSION, "profiles": {}}

    def _read_document(self):
        if not os.path.exists(self.path):
            return self._empty_document()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileStoreError(f"无法读取曲线配置：{exc}") from exc
        if not isinstance(document, dict) or document.get("version") != PROFILE_FILE_VERSION:
            raise ProfileStoreError("曲线配置文件版本无效")
        profiles = document.get("profiles")
        if not isinstance(profiles, dict):
            raise ProfileStoreError("profiles 字段无效")
        clean = {}
        for raw_name, raw_profile in profiles.items():
            try:
                name = validate_profile_name(raw_name)
                profile = normalise_profile(raw_profile)
            except (ValueError, ProfileStoreError) as exc:
                raise ProfileStoreError(f"曲线“{raw_name}”损坏：{exc}") from exc
            profile["created_at"] = str(raw_profile.get("created_at", ""))
            profile["updated_at"] = str(raw_profile.get("updated_at", ""))
            clean[name] = profile
        return {"version": PROFILE_FILE_VERSION, "profiles": clean}

    def _atomic_write(self, document):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        temp_path = self.path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except OSError as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise ProfileStoreError(f"保存曲线配置失败：{exc}") from exc

    def names(self):
        with self._lock:
            return sorted(self._read_document()["profiles"], key=str.casefold)

    def exists(self, name):
        name = validate_profile_name(name)
        with self._lock:
            return name in self._read_document()["profiles"]

    def load(self, name):
        name = validate_profile_name(name)
        with self._lock:
            profiles = self._read_document()["profiles"]
            if name not in profiles:
                raise ProfileStoreError(f"找不到曲线“{name}”")
            return deepcopy(profiles[name])

    def save(self, name, snapshot):
        name = validate_profile_name(name)
        profile = normalise_profile(snapshot)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            document = self._read_document()
            previous = document["profiles"].get(name, {})
            profile["created_at"] = previous.get("created_at") or now
            profile["updated_at"] = now
            document["profiles"][name] = profile
            self._atomic_write(document)
        return deepcopy(profile)

    def delete(self, name):
        name = validate_profile_name(name)
        with self._lock:
            document = self._read_document()
            if name not in document["profiles"]:
                return False
            del document["profiles"][name]
            self._atomic_write(document)
            return True
