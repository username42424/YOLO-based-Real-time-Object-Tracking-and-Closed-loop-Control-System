# -*- coding: utf-8 -*-
"""并行基准：同时起 K 个 bench_yolo 进程，测墙钟时间与总吞吐。"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
BENCH = os.path.join(HERE, "bench_yolo.py")
N = 200


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.perf_counter()
    procs = [subprocess.Popen([PY, BENCH], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True) for _ in range(K)]
    outs = [p.communicate()[0] for p in procs]
    wall = time.perf_counter() - t0
    print(f"\n=== K={K} 并发进程 ===")
    print(f"总墙钟时间={wall:.2f}s")
    for o in outs:
        print("  " + o.strip())
    thr = K * N / wall
    print(f"总吞吐={thr:.1f} 次/秒  (单进程基线约 {1/0.00962:.0f} 次/秒)")
    print(f"加速比 vs 单进程 = {thr / (1/0.00962):.2f}x")


if __name__ == "__main__":
    main()
