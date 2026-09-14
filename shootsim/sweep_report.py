# -*- coding: utf-8 -*-
"""修正版分阶段回放寻优进度报告: 概要 + 当前 TopN。

用法: python sweep_report.py [topN=5]
"""
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def _latest_results_dir():
    """取 results/ 下版本号最大的 staged_v* 目录(找不到则退回 staged_v5)。"""
    base = os.path.join(HERE, "results")
    best, best_n = None, -1
    if os.path.isdir(base):
        for name in os.listdir(base):
            if not name.startswith("staged_v"):
                continue
            try:
                n = int(name[len("staged_v"):])
            except ValueError:
                continue
            if n > best_n and os.path.isfile(
                    os.path.join(base, name, "staged_sweep_progress.json")):
                best, best_n = name, n
    return os.path.join(base, best) if best else os.path.join(base, "staged_v5")


RESULTS = _latest_results_dir()
PROGRESS = os.path.join(RESULTS, "staged_sweep_progress.json")


def metric_text(m):
    return ("首入60%%=%s 失败=%.1f%% 首入后保持=%.1f%% 偏上=%.1f%% 超时=%.1f%%"
            % ("无" if m.get("first_inner60_median") is None
               else "%.3fs" % m["first_inner60_median"],
               m.get("inner60_entry_failure_rate", 0.0) * 100,
               m.get("target_dwell_rate", m.get("dwell_reddot", 0.0)) * 100,
               m.get("above_box_ratio", 0.0) * 100,
               m.get("timeout_rate", 0.0) * 100))


def main():
    topn = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if not os.path.exists(PROGRESS):
        print("尚无进度文件(扫描未启动?)")
        return
    with open(PROGRESS, encoding="utf-8") as f:
        p = json.load(f)
    print("阶段: %s  阶段组数: %s  总耗时: %smin"
          % (p.get("phase", "未知"), p.get("stage_groups", "?"),
             p.get("total_min", "进行中")))
    best = p.get("overall_best") or {}
    if best.get("params"):
        print("当前最优 score=%.4f  %s" %
              (best.get("score", float("nan")), metric_text(best.get("metrics") or {})))
    # TopN(按 score 升序, 汇总各阶段 jsonl)
    paths = [os.path.join(RESULTS, "%s_results.jsonl" % stage)
             for stage in ("stage1", "stage2", "stage3")]
    if any(os.path.exists(path) for path in paths):
        rows = []
        for path in paths:
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("score") is not None:
                            rows.append(r)
                    except Exception:
                        pass
        rows.sort(key=lambda r: r["score"])
        print("Top%d(按修正版控制质量评分):" % topn)
        for r in rows[:topn]:
            m, pr = r["metrics"], r["params"]
            print("  #%s %s score=%.4f  %s"
                  % (r.get("gid", "?"), r.get("stage", "?"), r["score"],
                     metric_text(m)))
    if p.get("phase") == "all_done":
        print("状态: 扫描已全部完成。")


if __name__ == "__main__":
    main()
