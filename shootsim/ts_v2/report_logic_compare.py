# -*- coding: utf-8 -*-
"""Render a comparison report from computed groups without fixed claims."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def fmt(v, nd=2):
    return "缺失" if v is None else ("%.*f" % (nd, v))


def pct_delta(a, b):
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0


def main():
    src = os.path.join(ROOT, "日志V4", "cadence", "logic_compare.json")
    with open(src, encoding="utf-8") as handle:
        payload = json.load(handle)
    groups = payload.get("groups", payload)
    lines, add = [], lambda x="": lines.append(x)
    add("# 日志策略分组与闭环观测对比")
    add("")
    add("数据源: `日志V4/cadence/logic_compare.json`。分组来自日志头、运行时记录和观测行；未知字段保持为 `unknown`。")
    add("")
    add("## 分组与参数")
    add("")
    add("| 实际策略组 | 日志数 | 场次/段数 | 有效观测数 | 观测时长(s) | 观测率(Hz) | 间隔P50/P95(ms) | 输出P(ms) | 标定/滤波/响应 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, m in groups.items():
        params = m.get("parameters") or []
        unique = []
        for p in params:
            text = "k=%s tau=%s filter=%s" % (
                p.get("px_per_count", "缺失"), p.get("tau_ms", "缺失"),
                p.get("lock_box_smoothing_alpha", "缺失"))
            if text not in unique:
                unique.append(text)
        add("| %s | %d | %d | %d | %s | %s | %s/%s | %s | %s |" % (
            name, len(m.get("logs", [])), m.get("episodes", 0),
            m.get("observation_count", 0), fmt(m.get("observed_duration_s")),
            fmt(m.get("observation_rate_hz")), fmt(m.get("dt_p50_ms"), 1),
            fmt(m.get("dt_p95_ms"), 1), fmt(m.get("output_period_ms"), 1),
            "; ".join(unique[:3]) if unique else "缺失"))
    add("")
    add("## 一致口径指标")
    add("")
    add("`raw75` 使用同一图像中原始目标点-准星；`filtered75` 使用控制器滤波误差。速度仅纳入共同时间窗、连续目标、无异常跳变且中等可信样本。")
    add("")
    add("| 实际策略组 | 原始误差P75(px) | 滤波误差P75(px) | 横移滞后P75(px) | 有效速度样本 | 锁定时间占比 | 连续锁定P50(s) | 丢锁次数 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in groups.items():
        add("| %s | %s | %s | %s | %d | %s | %s | %s |" % (
            name, fmt(m.get("raw_err_p75")), fmt(m.get("err_p75")),
            fmt(m.get("strafe_xlag_p75")), m.get("movement_samples", 0),
            fmt(m.get("locked_time_ratio")), fmt(m.get("continuous_lock_p50_s")),
            m.get("lock_loss_count", "缺失")))
    add("")
    add("## 数据驱动观察")
    add("")
    names = list(groups)
    if len(names) >= 2:
        ref_name = names[0]
        ref = groups[ref_name]
        add("以下仅描述当前数据中的数值差异，基准为第一组 `%s`；不是因果结论。" % ref_name)
        add("")
        for name in names[1:]:
            m = groups[name]
            deltas = []
            for key, label in (("err_p75", "滤波误差P75"),
                               ("strafe_xlag_p75", "横移滞后P75"),
                               ("locked_time_ratio", "锁定时间占比")):
                d = pct_delta(m.get(key), ref.get(key))
                if d is not None:
                    deltas.append("%s %+.1f%%" % (label, d))
            add("- `%s` 相对 `%s`：%s。" % (name, ref_name, "；".join(deltas) or "可比指标缺失"))
    else:
        add("可比较策略组不足两个，未生成胜负描述。")
    add("")
    add("## 限制")
    add("")
    add("- 实战日志未记录游戏命中/击杀事件，因此命中率、击杀率、标准化超时率和红点60%停留率不从日志猜测，标为缺失。")
    add("- 这里的速度是共同窗口重建估计；发送输入到画面生效的时间和真实目标速度仍不能仅由这些日志证明。")
    add("- 仿真中的命中、击杀、超时和停留指标必须使用闭环仿真单独报告，不能与实战观测指标混列。")
    text = "\n".join(lines) + "\n"
    for path in (os.path.join(ROOT, "日志V4", "cadence", "LOGIC_COMPARE_REPORT.md"),
                 os.path.join(ROOT, "shootsim", "results", "tracking_logic_v4",
                              "cadence", "LOGIC_COMPARE_REPORT.md")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        print("saved", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
