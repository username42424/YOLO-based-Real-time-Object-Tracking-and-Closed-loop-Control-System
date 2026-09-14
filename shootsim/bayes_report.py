# -*- coding: utf-8 -*-
"""三阶段寻优进度报告(供定时任务调用)。

用法: python bayes_report.py
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS = os.path.join(HERE, "results", "bayes_unit_v7_progress.json")
FINAL = os.path.join(HERE, "results", "bayes_unit_v7_final.json")


def metric_text(m):
    return ("首入60%%=%s 失败=%.1f%% 驻留=%.1f%% 偏上=%.1f%% 超时=%.1f%%"
            % ("无" if m.get("first_inner60_median") is None
               else "%.3fs" % m["first_inner60_median"],
               m.get("inner60_entry_failure_rate", 0.0) * 100,
               m.get("dwell_reddot", 0.0) * 100,
               m.get("above_box_ratio", 0.0) * 100,
               m.get("timeout_rate", 0.0) * 100))


def main():
    if not os.path.exists(PROGRESS):
        print("尚无进度文件(扫描未启动?)")
        return
    with open(PROGRESS, encoding="utf-8") as f:
        p = json.load(f)
    print("阶段: %s | 进度: %d/%d 组 | 已用时 %.1fmin | 速率 %.0f组/min | ETA %.1fmin"
          % (p.get("stage"), p.get("groups_done", 0), p.get("total", 0),
             p.get("elapsed_min", 0), p.get("groups_per_min", 0), p.get("eta_min") or 0))
    b = p.get("best") or {}
    if b.get("params"):
        m = b["metrics"]
        print("当前最优 score=%.4f  %s" % (b["score"], metric_text(m)))
        pr = b["params"]
        print("  chest=%.3f far=%.2f near=%.2f radius=%.1f dz=%.1f damp=%.2f cap=%.0f "
              "lock=%.0f iou=%.2f amb=%.2f"
              % (pr["chest_ratio"], pr["unit.far_gain"], pr["unit.near_gain"],
                 pr["unit.near_radius_px"], pr["unit.deadzone"],
                 pr["aim_control.reversal_damp"], pr["aim_control.unit_max_counts"],
                 pr["target_lock_distance"], pr["target_lock_iou"],
                 pr["lock_ambiguity_margin"]))
    if os.path.exists(FINAL):
        with open(FINAL, encoding="utf-8") as f:
            fin = json.load(f)
        rows = fin.get("ranking", []) if isinstance(fin, dict) else fin
        champion = fin.get("champion") if isinstance(fin, dict) else None
        print("== 独立验证集终评 Top%d ==" % len(rows))
        for r in rows[:5]:
            m, pr = r["metrics"], r["params"]
            print("  #%d score=%.4f %s | far=%.2f near=%.2f radius=%.1f damp=%.2f cap=%.0f"
                  % (r["rank"], r["score"], metric_text(m), pr["unit.far_gain"],
                     pr["unit.near_gain"], pr["unit.near_radius_px"],
                     pr["aim_control.reversal_damp"], pr["aim_control.unit_max_counts"]))
        if champion is None:
            print("结论: 没有候选通过全部硬门槛，未产生冠军。")
    if p.get("done"):
        print("状态: 全部完成, 可删除定时汇报。")


if __name__ == "__main__":
    main()
