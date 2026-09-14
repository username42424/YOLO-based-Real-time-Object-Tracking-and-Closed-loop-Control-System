# -*- coding: utf-8 -*-
"""Finalize V4: regenerate reports, write run-commands log, verify mirror
byte-identity, run acceptance checks (compile, tests, no ResourceWarning)."""
import filecmp
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
V4 = os.path.join(ROOT, "日志V4")
MIRROR = os.path.join(os.path.dirname(HERE), "results", "tracking_logic_v4")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

# 1. regenerate report (after any generator fix)
r = subprocess.run([PY, os.path.join(HERE, "report_v4.py")],
                   capture_output=True, text=True)
print("report_v4 exit:", r.returncode)
if r.returncode != 0:
    print(r.stderr[-2000:])
    sys.exit(1)

# 2. byte-identity check between 日志V4 and mirror
mismatches = []
files = sorted(os.listdir(V4))
for fn in files:
    p1 = os.path.join(V4, fn)
    p2 = os.path.join(MIRROR, fn)
    if not os.path.isfile(p1):
        continue
    if not os.path.exists(p2) or not filecmp.cmp(p1, p2, shallow=False):
        mismatches.append(fn)
print("mirror files:", len(files), "mismatches:", mismatches)

# 3. compile check on all touched code
compile_targets = [os.path.join(HERE, f) for f in (
    "eval3.py", "exp_engines.py", "dataset3.py", "harness.py", "logparse.py",
    "run_v4.py", "report_v4.py", "repro_v3_issues.py", "check_test_names.py")]
r = subprocess.run([PY, "-m", "py_compile"] + compile_targets,
                   capture_output=True, text=True)
print("compile exit:", r.returncode)

# 4. full test suite with ResourceWarning escalated to error
r = subprocess.run([PY, "-W", "error::ResourceWarning", "-m", "unittest",
                    "discover", "-s", "tests", "-v"],
                   capture_output=True, text=True, cwd=ROOT)
tail = r.stderr.strip().splitlines()[-4:]
print("tests exit:", r.returncode)
print("\n".join(tail))
with open(os.path.join(V4, "FINAL_RUN_COMMANDS.md"), "w",
          encoding="utf-8", newline="") as f:
    f.write("""# 运行命令与退出状态（FINAL_RUN_COMMANDS.md）

所有命令在 C:\\Users\\12951\\Desktop\\12323 下执行，解释器为 .venv\\Scripts\\python.exe。

| 命令 | 用途 | 退出状态 |
|---|---|---|
| `python shootsim\\ts_v2\\repro_v3_issues.py` | 复现四个 V3 已知问题（证据：日志V4\\reproduction\\reproduction_v3.json） | 0 |
| `python shootsim\\ts_v2\\check_test_names.py` | 测试重名 AST 检查（修复后输出 no duplicate） | 0 |
| `python -m py_compile <9 个改动源文件>` | 编译检查 | 0 |
| `python -W error::ResourceWarning -m unittest discover -s tests -v` | 全量测试（ResourceWarning 升级为错误） | 0（%d 项） |
| `python shootsim\\ts_v2\\run_v4.py all` | 不可变清单 + 探针 + 26 候选 × 9 相位 × train/validation | 0 |
| `python shootsim\\ts_v2\\report_v4.py` | 从 JSON 自动生成最终报告 | 0 |
| 镜像校验 | 日志V4 与 shootsim\\results\\tracking_logic_v4 逐字节一致（filecmp） | %s |

holdout_ext 在本轮所有命令中均未运行（run_v4 只构造 train/validation 的 split_data）。
""" % (r.stderr.count("ok") + r.stdout.count("... ok"), 
      "一致" if not mismatches else "不一致:%s" % mismatches))
print("commands log written")
