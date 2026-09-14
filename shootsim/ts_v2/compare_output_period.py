# -*- coding: utf-8 -*-
"""输出周期对比: asap+arrival 下 共享输出周期 10/8/5ms 实战日志对比.

新旧日志格式(JSON-envelope / 旧文本)统一走 analyze_log(load_log+fill_world)
→ group_metrics, 全部组同一管线、同一速度估计、同一有效性门。
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_cadence_sweep import analyze_log, group_metrics  # noqa

ROOT = os.path.dirname(os.path.dirname(HERE))
LOGS = os.path.join(ROOT, "logs")


def scan_meta(path):
    """扫描 config/summary 行: 输出周期、策略模式、tick 健康、发送统计."""
    meta = {"output_ms": None, "mode": None, "tick_p50": None,
            "tick_p95": None, "send_att": 0, "send_ok": 0, "send_fail": 0,
            "format": "legacy"}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"event_type"' in line and '"config"' in line \
                    and "output_period_ms=" in line:
                m = re.search(r"output_period_ms=([0-9.]+)", line)
                if m:
                    meta["output_ms"] = float(m.group(1))
                m = re.search(r"prediction_mode=(\w+)", line)
                if m:
                    meta["mode"] = m.group(1)
                meta["format"] = "structured"
            elif "[输出器]" in line:
                m = re.search(r"tick_dt P50/P95=([0-9.]+)/([0-9.]+)ms", line)
                if m:
                    meta["tick_p50"] = float(m.group(1))
                    meta["tick_p95"] = float(m.group(2))
                for key in ("send_attempt", "send_success", "send_failed"):
                    mm = re.search(key + r"=(\d+)", line)
                    if mm:
                        meta[{"send_attempt": "send_att",
                              "send_success": "send_ok",
                              "send_failed": "send_fail"}[key]] = \
                            int(mm.group(1))
    return meta


GROUPS = {
    "新asap+arrival@10ms": ["10ms_v1.log", "10ms_v2.log"],
    "新asap+arrival@8ms": ["8ms.log"],
    "新asap+arrival@5ms": ["5ms.log"],
    "新asap+current@10ms(昨日)": [
        "aim_20260907_120855.log", "aim_20260907_125456.log",
        "aim_20260907_142216.log", "aim_20260907_142639.log",
        "aim_20260907_143132.log", "aim_20260907_230030.log",
        "aim_20260907_230737.log"],
    "旧30ms+arrival": ["aim_20260906_135713.log", "aim_20260906_140550.log",
                       "aim_20260906_141313.log", "aim_20260906_143059.log",
                       "aim_20260906_222229.log"],
    "旧40ms+arrival": ["aim_20260906_125713.log", "aim_20260906_130338.log",
                       "aim_20260906_131131.log"],
}


def main():
    results = {}

    def fmt(v, nd=2):
        return "null" if v is None else ("%.*f" % (nd, v))

    print("%-26s %5s %4s %5s %8s %8s %8s %7s %7s %7s %6s" % (
        "组", "周期", "eps", "锁定帧", "xlag75", "err75", "tw_err",
        "dt_p50", "dt_p95", "lat_p95", "FPS"))
    for name, files in GROUPS.items():
        eps_all, meta, used = [], {}, []
        for fn in files:
            path = os.path.join(LOGS, fn)
            if not os.path.exists(path):
                print("missing:", fn)
                continue
            eps = analyze_log(path)
            if eps:
                eps_all.extend(eps)
                used.append(fn)
                m0 = scan_meta(path)
                if not meta:
                    meta = m0
                elif m0.get("output_ms") and meta.get("output_ms") is None:
                    meta = m0
        if not eps_all:
            continue
        m = group_metrics(eps_all)
        m["logs"] = used
        m["meta"] = meta
        results[name] = m
        out = ("%.0fms" % meta["output_ms"]) if meta.get("output_ms") else "  - "
        fps = 1000.0 / m["dt_p50_ms"] if m["dt_p50_ms"] else 0.0
        print("%-26s %5s %4d %5d %8s %8s %8s %7s %7s %7s %6.1f" % (
            name, out, m["episodes"], m["locked_obs"],
            fmt(m["strafe_xlag_p75"]), fmt(m["err_p75"]),
            fmt(m.get("time_weighted_err_mean_px"), 1),
            fmt(m["dt_p50_ms"], 1), fmt(m["dt_p95_ms"], 1),
            fmt(m["lat_p95_ms"], 1), fps))
        for band in sorted(m["xlag_by_speed_band"]):
            bl = m["xlag_by_speed_band"][band]
            be = m["err_by_speed_band"].get(band, {})
            print("      %10s px/s: xlag75=%6s (n=%4d)  err75=%6s" % (
                band, fmt(bl["p75"]), bl["n"], fmt(be.get("p75"))))
        print("      发送: attempt=%d ok=%d fail=%d  tick_p95=%sms" % (
            meta.get("send_att", 0), meta.get("send_ok", 0),
            meta.get("send_fail", 0), fmt(meta.get("tick_p95"), 2)))
    out_path = os.path.join(ROOT, "日志V4", "cadence",
                            "output_period_compare.json")
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print("saved", os.path.relpath(out_path, ROOT))


if __name__ == "__main__":
    main()
