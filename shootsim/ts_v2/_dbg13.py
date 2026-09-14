# -*- coding: utf-8 -*-
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final2
print("final2 module file:", final2.__file__)
print("ROBUST keys:", sorted(final2.ROBUST.keys()))
print("ROBUST:", json.dumps(final2.ROBUST, sort_keys=True))

from harness import load_production_cfg, patch_cfg
prod = load_production_cfg()
cfg = patch_cfg(prod, final2.ROBUST)
diffs = []
for sec in ("aim_control", "unit"):
    for k in set(list(prod[sec]) + list(cfg[sec])):
        if prod[sec].get(k) != cfg[sec].get(k):
            diffs.append(f"{sec}.{k}: {prod[sec].get(k)} -> {cfg[sec].get(k)}")
print("patched diffs (%d):" % len(diffs))
for d in diffs:
    print("  ", d)

# one-episode bit check
from search import prepare_dataset
from harness import build_sim_cfg, run_episode
metas, episodes, phases, idx = prepare_dataset()
sim_cfg = build_sim_cfg(["aim_20260906_003353.log", "aim_20260906_001842.log",
                         "aim_20260906_000134.log"])
i = idx["train"][0]
r1 = run_episode(sim_cfg, prod, episodes[i], output_phase_s=phases[i])
r2 = run_episode(sim_cfg, cfg, episodes[i], output_phase_s=phases[i])
c1 = [row["cmd"] for row in r1.obs]
c2 = [row["cmd"] for row in r2.obs]
print("episode", i, "identical commands:", c1 == c2,
      "n:", len(c1), len(c2))
