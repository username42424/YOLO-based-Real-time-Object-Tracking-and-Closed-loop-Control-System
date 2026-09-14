# -*- coding: utf-8 -*-
"""日志记录：JSONL 逐发子弹 + 回合汇总。

文件结构（每回合一个文件）：
    {"type":"episode_start", "seed":…, "config":{…}, "spawn":{…}}
    {"type":"shot", "frame":…, "t":…, "crosshair":[…], "box":[…], "hit":…,
     "region":"head|mid|low|null", "dmg":…, "cum_dmg":…, "hp":…}
    （可选）{"type":"frame", …每 frame_interval 帧记录准星/目标框/相机…}
    {"type":"episode_end", "summary":{kill_time, shots, hits, accuracy, …}}
"""
import itertools
import json
import os
import time


class EpisodeLogger:
    _seq = itertools.count(1)

    def __init__(self, log_dir="results", per_frame=False, frame_interval=10):
        self.log_dir = log_dir
        self.per_frame = per_frame
        self.frame_interval = max(1, frame_interval)
        self.fh = None
        self.path = None

    def start(self, seed, cfg, env):
        os.makedirs(self.log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        n = next(EpisodeLogger._seq)
        self.path = os.path.join(self.log_dir, f"episode_{ts}_{n}_{seed}.jsonl")
        self.fh = open(self.path, "w", encoding="utf-8")
        self._w({"type": "episode_start",
                 "seed": seed,
                 "screen": [env.sw, env.sh],
                 "config": cfg,
                 "spawn": {"cam": [round(env.cam_x, 1), round(env.cam_y, 1)],
                           "targets": [[round(t["tx"], 1), round(t["ty"], 1)]
                                       for t in env.targets]}})

    def shot(self, record):
        self._w({"type": "shot", **record})

    def frame(self, env, tracker=None, yolo_boxes=None):
        rect = env.gt_box(0)
        self._w({"type": "frame",
                 "frame": env.frame,
                 "t": round(env.t, 4),
                 "crosshair": [round(env.cx, 1), round(env.cy, 1)],
                 "boxes": [[round(v, 1) for v in b] for b in env.gt_boxes(0)],
                 "锁定": [[round(v, 1) for v in b] for b in (yolo_boxes or [])],
                 "cam": [round(env.cam_x, 1), round(env.cam_y, 1)],
                 "hp": sum(max(0, t["hp"]) for t in env.targets)})

    def end(self, env):
        self._w({"type": "episode_end", "summary": env.summary()})
        self.close()

    def _w(self, obj):
        if self.fh:
            self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


class SilentLogger(EpisodeLogger):
    """批量优化时用：不写文件，只保留路径计数。"""
    def __init__(self):
        super().__init__(log_dir=None)
    def start(self, *a, **k):
        pass
    def shot(self, record):
        pass
    def frame(self, *a, **k):
        pass
    def end(self, env):
        pass
