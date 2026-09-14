# -*- coding: utf-8 -*-
"""实战日志回放：从 main.py 的 [鼠标] 日志提取目标框轨迹，扣除自身鼠标移动，
生成模拟器的目标世界轨迹（ReplayPlayer），替代原"图片+yolo+随机移动"的场景生成。

字段语义（与 main.py run_engine 的日志行一一对应）：
  帧      main.py 循环帧号（回合内从 1 计；只有实际发送了鼠标移动的帧才写日志行，
          未发送帧无日志行，但帧号连续，可精确推算循环周期）
  锁定    锁定目标检测框，320 截图坐标（截图中心=准星=160,160，屏幕像素 1:1）
  发送    本帧请求位移(px, 灵敏度缩放前)
  净移    本帧实际发送的鼠标 counts（分步向量和）—— 即"自身鼠标移动"
  准星    瞄准参考点（红点 EMA，未检出时为截图中心），非自身移动量
  误差    EMA + 前导 - 准星

自身移动扣除：发送 counts 后视角旋转，同一目标在后续截图中的坐标反向偏移
  Δcapture = -Δcounts × view_scale（实机标定 0.3011 / 0.2684 px/counts）。
因此目标世界轨迹（相对回合首帧视角）= 截图框中心 - 160 + Σ(此前净移×scale)。
回放时模拟器视角灵敏度 = 同一 scale，模拟器自己发出的 counts 通过 env.apply_mouse
按相同比例转动视角 —— 与实战物理一致。
"""
import io
import json
import os
import re
import statistics

# ── 日志行解析 ──
_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
_START_RE = re.compile(r"▶ ")
_STOP_RE = re.compile(r"■ ")
_SOURCE_CHEST_RE = re.compile(r"\[配置\].*?\bchest_ratio=([0-9]+(?:\.[0-9]+)?)")
_MOUSE_RE = re.compile(
    r"\[鼠标\] 帧=(?P<frame>\d+) "
    r"目标=\((?P<tx>[-\d.]+),(?P<ty>[-\d.]+)\) "
    r"EMA=\((?P<ex>[-\d.]+),(?P<ey>[-\d.]+)\) "
    r"准星=\((?P<sx>[-\d.]+),(?P<sy>[-\d.]+)\) "
    r"误差=\((?P<ex2>[-+\d.]+),(?P<ey2>[-+\d.]+)\) "
    r"前导=\((?P<lx>[-+\d.]+),(?P<ly>[-+\d.]+)\) "
    r"补偿=\((?P<cx>[-+\d.]+),(?P<cy>[-+\d.]+)\) "
    r"发送=\((?P<dx>[-+\d.]+),(?P<dy>[-+\d.]+)\) "
    r"增益=(?P<gain>[\d.]+) sens=(?P<sens>[\d.]+) "
    r"锁定=\[(?P<b>[^\]]*)\] cls=(?P<cls>\d+) conf=(?P<conf>[\d.]+) "
    r"净移=\((?P<nx>[-+\d]+),(?P<ny>[-+\d]+)\) 步数=(?P<steps>\d+)"
)

_DEFAULT_ITER = 0.0225  # 兜底循环周期(s)（实战约 22~25ms/帧）


def _parse_ts_ms(s):
    h, m, sec, ms = s.split(":")[0], s.split(":")[1], s.split(":")[2].split(".")[0], s.split(":")[2].split(".")[1]
    return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(ms)


def _read_text(path):
    b = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def _structured_episodes(path, rows, source_chest_ratio):
    """Build sessions from capture order, immune to asynchronously flushed markers."""
    groups = []
    current = []
    previous_frame = previous_capture = None
    for row in rows:
        frame = int(row.get("frame", 0))
        capture_t = float(row.get("capture_t", 0.0))
        if current and (frame <= previous_frame or capture_t <= previous_capture):
            groups.append(current)
            current = []
        current.append(row)
        previous_frame, previous_capture = frame, capture_t
    if current:
        groups.append(current)

    episodes = []
    for group in groups:
        capture0 = float(group[0].get("capture_t", 0.0))
        ep = {
            "src": path,
            "t_start": capture0 * 1000.0,
            "t_end": float(group[-1].get("capture_t", capture0)) * 1000.0,
            "points": [],
            "motion_already_observed": True,
        }
        last_box = None
        last_cls = 0
        last_conf = 0.0
        for row in group:
            dets = list(row.get("detections") or [])
            target = row.get("target")
            chosen = None
            if dets:
                if target is None:
                    chosen = dets[0]
                else:
                    tx, ty = float(target[0]), float(target[1])

                    def _aim_distance(det):
                        x1, y1, x2, y2 = map(float, det["bbox"][:4])
                        ax = (x1 + x2) * 0.5
                        ay = ((y1 + y2) * 0.5
                              if int(det.get("cls", 0)) == 1
                              else y1 + (y2 - y1) * float(source_chest_ratio))
                        return (ax - tx) ** 2 + (ay - ty) ** 2

                    chosen = min(dets, key=_aim_distance)
                last_box = [float(v) for v in chosen["bbox"][:4]]
                last_cls = int(chosen.get("cls", 0))
                last_conf = float(chosen.get("conf", 0.0))
            if last_box is None:
                continue
            box = list(last_box)
            aim = row.get("observed_aim") or (0, 0)
            recoil = row.get("observed_recoil") or (0, 0)
            crosshair = row.get("crosshair") or (160.0, 160.0)
            observed_error = row.get("observed_error") or (0.0, 0.0)
            lead = row.get("lead") or (0.0, 0.0)
            command = row.get("command") or (0.0, 0.0)
            target_value = target or (
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            )
            ep["points"].append({
                "frame": int(row.get("frame", len(ep["points"]) + 1)),
                "t_rel": float(row.get("capture_t", capture0)) - capture0,
                # real per-frame processing latency drives the arrival-prediction
                # horizon exactly as the live engine saw it
                "lat_s": max(0.0, float(row.get("processing_latency_ms", 0.0))) / 1000.0,
                "box": box,
                "cx": (box[0] + box[2]) * 0.5,
                "cy": (box[1] + box[3]) * 0.5,
                "w": box[2] - box[0],
                "h": box[3] - box[1],
                "cls": last_cls,
                "conf": last_conf if chosen is None else float(chosen.get("conf", 0.0)),
                "net": (int(aim[0]) + int(recoil[0]),
                        int(aim[1]) + int(recoil[1])),
                "ref": (float(crosshair[0]), float(crosshair[1])),
                "err": (float(observed_error[0]), float(observed_error[1])),
                "lead": (float(lead[0]), float(lead[1])),
                "ema": (float(target_value[0]), float(target_value[1])),
                "target": (float(target_value[0]), float(target_value[1])),
                "sent": (float(command[0]), float(command[1])),
                "detected": bool(dets),
                "detections": [
                    {"cls": int(det.get("cls", 0)),
                     "conf": float(det.get("conf", 0.0)),
                     "bbox": [float(v) for v in det["bbox"][:4]]}
                    for det in dets
                ],
                "dot_status": str(row.get("dot_status", "unknown")),
            })
        if ep["points"]:
            episodes.append(ep)
    return episodes


def parse_log_file(path, source_chest_ratio=0.3):
    """解析一个实战日志 → 回合列表。

    回合 = ▶ 到 ■ 之间；点 = [鼠标] 行（仅含实际发送了移动的帧）。
    """
    raw_text = _read_text(path)
    logged_ratio = _SOURCE_CHEST_RE.search(raw_text)
    if logged_ratio:
        source_chest_ratio = float(logged_ratio.group(1))
    structured_rows = []
    marker = "[瞄准观测] "
    for raw in raw_text.splitlines():
        if marker not in raw:
            continue
        try:
            row = json.loads(raw.split(marker, 1)[1])
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and "frame" in row and "capture_t" in row:
            structured_rows.append(row)
    if structured_rows:
        return _structured_episodes(path, structured_rows, source_chest_ratio)

    episodes = []
    cur = None
    t0 = None
    for raw in raw_text.splitlines():
        m = _TS_RE.match(raw)
        ts = _parse_ts_ms(m.group(0)) if m else None
        if _START_RE.search(raw):
            cur = {
                "src": path, "t_start": ts, "t_end": None, "points": [],
                "motion_already_observed": False,
                "_capture_t0": None,
                "_last_box": None,
            }
            episodes.append(cur)
            continue
        if _STOP_RE.search(raw):
            if cur is not None:
                cur["t_end"] = ts
                cur = None
            continue
        if cur is None or ts is None:
            continue
        marker = "[瞄准观测] "
        if marker in raw:
            try:
                row = json.loads(raw.split(marker, 1)[1])
            except (ValueError, TypeError):
                continue
            if not cur["motion_already_observed"]:
                cur["points"].clear()
                cur["motion_already_observed"] = True
            dets = list(row.get("detections") or [])
            target = row.get("target")
            chosen = None
            if dets:
                if target is None:
                    chosen = dets[0]
                else:
                    tx, ty = float(target[0]), float(target[1])
                    def _aim_distance(d):
                        x1, y1, x2, y2 = map(float, d["bbox"][:4])
                        ax = (x1 + x2) * 0.5
                        ay = ((y1 + y2) * 0.5 if int(d.get("cls", 0)) == 1
                              else y1 + (y2 - y1) * float(source_chest_ratio))
                        return (ax - tx) ** 2 + (ay - ty) ** 2

                    chosen = min(dets, key=_aim_distance)
                box = [float(v) for v in chosen["bbox"][:4]]
                cur["_last_box"] = list(box)
            elif cur["_last_box"] is not None:
                box = list(cur["_last_box"])
            else:
                continue
            capture_t = float(row.get("capture_t", 0.0))
            if cur["_capture_t0"] is None:
                cur["_capture_t0"] = capture_t
            aim = row.get("observed_aim") or (0, 0)
            recoil = row.get("observed_recoil") or (0, 0)
            crosshair = row.get("crosshair") or (160.0, 160.0)
            observed_error = row.get("observed_error") or (0.0, 0.0)
            lead = row.get("lead") or (0.0, 0.0)
            command = row.get("command") or (0.0, 0.0)
            target = target or (
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            )
            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            cur["points"].append({
                "frame": int(row.get("frame", len(cur["points"]) + 1)),
                "t_rel": capture_t - cur["_capture_t0"],
                "lat_s": max(0.0, float(row.get("processing_latency_ms", 0.0))) / 1000.0,
                "box": box, "cx": cx, "cy": cy,
                "w": box[2] - box[0], "h": box[3] - box[1],
                "cls": int(chosen.get("cls", 0)) if chosen else 0,
                "conf": float(chosen.get("conf", 0.0)) if chosen else 0.0,
                "net": (int(aim[0]) + int(recoil[0]),
                        int(aim[1]) + int(recoil[1])),
                "ref": (float(crosshair[0]), float(crosshair[1])),
                "err": (float(observed_error[0]), float(observed_error[1])),
                "lead": (float(lead[0]), float(lead[1])),
                "ema": (float(target[0]), float(target[1])),
                "target": (float(target[0]), float(target[1])),
                "sent": (float(command[0]), float(command[1])),
                "detected": bool(dets),
                "detections": [
                    {"cls": int(d.get("cls", 0)),
                     "conf": float(d.get("conf", 0.0)),
                     "bbox": [float(v) for v in d["bbox"][:4]]}
                    for d in dets
                ],
                "dot_status": str(row.get("dot_status", "unknown")),
            })
            continue
        if cur.get("motion_already_observed"):
            continue
        mm = _MOUSE_RE.search(raw)
        if not mm:
            continue
        g = mm.groupdict()
        b = g["b"].strip()
        if b:
            parts = [float(v) for v in b.split(",")]
            box = parts[:4] if len(parts) >= 4 else None
        else:
            box = None
        cx = (box[0] + box[2]) * 0.5 if box else None
        cy = (box[1] + box[3]) * 0.5 if box else None
        cur["points"].append({
            "frame": int(g["frame"]),
            "t_rel": (ts - cur["t_start"]) / 1000.0,
            "lat_s": 0.0,
            "box": box, "cx": cx, "cy": cy,
            "w": (box[2] - box[0]) if box else 0.0,
            "h": (box[3] - box[1]) if box else 0.0,
            "cls": int(g["cls"]), "conf": float(g["conf"]),
            "net": (int(g["nx"]), int(g["ny"])),
            "ref": (float(g["sx"]), float(g["sy"])),
            "err": (float(g["ex2"]), float(g["ey2"])),
            "lead": (float(g["lx"]), float(g["ly"])),
            "ema": (float(g["ex"]), float(g["ey"])),
            "target": (float(g["tx"]), float(g["ty"])),
            "sent": (float(g["dx"]), float(g["dy"])),
            "detections": ([{"cls": int(g["cls"]), "conf": float(g["conf"]),
                              "bbox": list(box)}] if box else []),
            "dot_status": "legacy",
        })
    for episode in episodes:
        episode.pop("_capture_t0", None)
        episode.pop("_last_box", None)
    return [e for e in episodes if e["points"]]


class ReplayEpisode:
    """一个回合的世界轨迹（已扣除自身移动），并展开成逐帧序列。

    帧模型：回合从 ▶ 时刻起，每帧间隔≈循环周期。首个 [鼠标] 点之前为"未检出"帧；
    点之间的帧按线性插值（实战中这些帧大多只是死区未发送，检测仍在）；
    点之后（■ 之后）按 on_end 策略处理；实战回放默认 stop，避免凭空补造轨迹。
    """

    def __init__(self, ep, idx, scale=(0.3011, 0.2684), cap_c=160.0, min_points=3,
                 send_lag=2, apply_theta=1.0):
        self.src = ep["src"]
        self.idx = idx
        pts = sorted(ep["points"], key=lambda p: p["frame"])
        self.points = pts
        self.n_points = len(pts)
        self.send_lag = 0 if ep.get("motion_already_observed") else max(0, int(send_lag))
        # Camera-mixing model: fraction theta of the counts batch sent in
        # interval (k-1,k] that the game camera had already applied when
        # capture k was taken (theta=1 → no input latency).  1-theta is
        # applied one capture later.  Must match Env.apply_theta.
        self.apply_theta = min(1.0, max(0.0, float(apply_theta)))
        ok = [p for p in pts if p["box"] is not None]
        if len(ok) < min_points or len(pts) < 2:
            raise ValueError("点太少, 无法回放")
        # 循环周期: 相邻点 dt/Δ帧 的中位数
        ds = []
        for a, b in zip(pts, pts[1:]):
            df = b["frame"] - a["frame"]
            if df > 0:
                ds.append((b["t_rel"] - a["t_rel"]) / df)
        self.iter = statistics.median(ds) if ds else _DEFAULT_ITER
        if not (0.004 <= self.iter <= 0.08):
            self.iter = _DEFAULT_ITER
        # 自身移动累计（capture px）。结构化日志中的 observed_aim 已经是截图内
        # 实际观察到的位移，因此 send_lag=0；旧格式才允许显式配置滞后。
        cumx = cumy = 0.0
        prev_net_x = prev_net_y = 0
        self.cum = []
        for i, p in enumerate(pts):
            if self.send_lag:
                # legacy logs: attribute a batch to a later capture verbatim
                if i - self.send_lag >= 0:
                    applied = pts[i - self.send_lag]["net"]
                else:
                    applied = (0, 0)
            elif self.apply_theta >= 1.0:
                applied = p["net"]
            else:
                applied = (self.apply_theta * p["net"][0]
                           + (1.0 - self.apply_theta) * prev_net_x,
                           self.apply_theta * p["net"][1]
                           + (1.0 - self.apply_theta) * prev_net_y)
            prev_net_x, prev_net_y = p["net"]
            cumx += applied[0] * scale[0]
            cumy += applied[1] * scale[1]
            self.cum.append((cumx, cumy))
        # 点级世界轨迹（相对回合首帧视角）
        self.wx = [p["cx"] - cap_c + c[0] for p, c in zip(pts, self.cum)]
        self.wy = [p["cy"] - cap_c + c[1] for p, c in zip(pts, self.cum)]
        self.bw = [p["w"] for p in pts]
        self.bh = [p["h"] for p in pts]
        self.cls = pts[0]["cls"]
        self.confs = [p["conf"] for p in pts]
        self.detected = [bool(p.get("detected", True)) for p in pts]
        self.refs = [tuple(p.get("ref", (cap_c, cap_c))) for p in pts]
        self.refx = [float(v[0]) for v in self.refs]
        self.refy = [float(v[1]) for v in self.refs]
        self.dot_statuses = [str(p.get("dot_status", "unknown")) for p in pts]
        self.lats = [max(0.0, float(p.get("lat_s", 0.0))) for p in pts]
        self.world_detections = []
        for p, c in zip(pts, self.cum):
            items = []
            for det in p.get("detections", []):
                x1, y1, x2, y2 = map(float, det["bbox"][:4])
                items.append([
                    x1 - cap_c + c[0], y1 - cap_c + c[1],
                    x2 - cap_c + c[0], y2 - cap_c + c[1],
                    int(det.get("cls", 0)), float(det.get("conf", 0.0)),
                ])
            self.world_detections.append(items)
        self._interpolate_missing_target_tracks()
        self.f0, self.f1 = pts[0]["frame"], pts[-1]["frame"]
        self.net_by_frame = {p["frame"]: p["net"] for p in pts}
        self.t_first = pts[0]["t_rel"]
        self.t_last = pts[-1]["t_rel"]
        self.duration = ((ep["t_end"] - ep["t_start"]) / 1000.0
                         if ep["t_end"] else self.t_last)
        self.pre_frames = max(0, self.f0 - 1)   # 首个检出帧之前的未检出帧数
        # 逐帧时间（检出区间内线性插值）
        self._ft = []
        for k, p in enumerate(pts):
            self._ft.append(p["t_rel"])

    def _interpolate_missing_target_tracks(self):
        """Fill the hidden physical track without inventing detections."""
        n = len(self.detected)
        i = 0
        arrays = (self.wx, self.wy, self.bw, self.bh)
        while i < n:
            if self.detected[i]:
                i += 1
                continue
            start = i
            while i < n and not self.detected[i]:
                i += 1
            left = start - 1
            right = i if i < n else None
            if left >= 0 and right is not None:
                span = float(right - left)
                for k in range(start, right):
                    ratio = (k - left) / span
                    for arr in arrays:
                        arr[k] = arr[left] + (arr[right] - arr[left]) * ratio
            elif left >= 1:
                for k in range(start, n):
                    for arr in arrays:
                        arr[k] = arr[left] + (arr[left] - arr[left - 1]) * (k - left)

    # ── 帧插值工具 ──
    def _lerp(self, arr, a, b, fa, fb, f):
        if fb == fa:
            return arr[a]
        t = (f - fa) / float(fb - fa)
        return arr[a] + (arr[b] - arr[a]) * t

    def frame_at(self, f):
        """返回检出区间内插值后的目标、红点状态和该帧全部检测框。"""
        f = max(self.f0, min(self.f1, f))
        if f == self.f0:
            i = 0
        elif f == self.f1:
            i = len(self.points) - 1
        else:
            i = None
            for k in range(len(self.points) - 1):
                if self.points[k]["frame"] <= f <= self.points[k + 1]["frame"]:
                    i = k
                    break
            if i is None:
                i = len(self.points) - 1
        j = min(i + 1, len(self.points) - 1)
        fa, fb = self.points[i]["frame"], self.points[j]["frame"]
        t = self._lerp(self._ft, i, j, fa, fb, f)
        ref = (
            self._lerp(self.refx, i, j, fa, fb, f),
            self._lerp(self.refy, i, j, fa, fb, f),
        )
        return (t,
                self._lerp(self.wx, i, j, fa, fb, f),
                self._lerp(self.wy, i, j, fa, fb, f),
                self._lerp(self.bw, i, j, fa, fb, f),
                self._lerp(self.bh, i, j, fa, fb, f),
                self._lerp(self.confs, i, j, fa, fb, f),
                self.detected[i], ref, self.dot_statuses[i],
                self.world_detections[i], self.lats[i])

    def point_at(self, index):
        """Return one recorded capture without inventing intermediate observations."""
        i = max(0, min(len(self.points) - 1, int(index)))
        return (
            self._ft[i], self.wx[i], self.wy[i], self.bw[i], self.bh[i],
            self.confs[i], self.detected[i],
            (self.refx[i], self.refy[i]), self.dot_statuses[i],
            self.world_detections[i], self.lats[i],
        )

    @property
    def n_frames(self):
        return self.n_points


class ReplayPlayer:
    """逐帧推进一个回合：env.step 每帧调用一次 step()。

    返回 dict(dt, tx, ty, bw, bh, cls, conf, detected)。
    世界坐标约定：回合首帧相机位于 (0,0)，目标世界坐标 = 截图框中心-160+累计自移。
    """

    def __init__(self, episode, max_seconds=3.0, on_end="hold"):
        self.ep = episode
        self.max_seconds = max_seconds
        self.on_end = on_end
        self.reset()

    def reset(self):
        self._i = -self.ep.pre_frames - 1  # 先给 0 帧作为回合起点（未检出）
        self._t = 0.0
        first = self.ep.point_at(0)
        last = self.ep.point_at(self.ep.n_points - 1)
        self._first = (first[1], first[2], first[3], first[4], self.ep.cls,
                       first[5], first[7], first[8], first[9])
        self._last = (last[1], last[2], last[3], last[4], self.ep.cls,
                      last[5], last[7], last[8], last[9])

    def step(self):
        ep = self.ep
        self._i += 1
        i = self._i
        if i < 0:
            # 首个检出之前的未检出帧
            dt = ep.iter
            self._t += dt
            return {"dt": dt, "tx": self._first[0], "ty": self._first[1],
                    "bw": self._first[2], "bh": self._first[3],
                    "cls": self._first[4], "conf": self._first[5],
                    "detected": False, "ref": self._first[6],
                    "dot_status": self._first[7], "detections": [],
                    "latency": ep.lats[0],
                    "ended": False}
        if i < ep.n_points:
            t, wx, wy, bw, bh, cf, detected, ref, dot_status, detections, lat = ep.point_at(i)
            dt = max(1e-3, t - self._t)
            self._t = t
            return {"dt": dt, "tx": wx, "ty": wy, "bw": bw, "bh": bh,
                    "cls": ep.cls, "conf": cf, "detected": detected,
                    "ref": ref, "dot_status": dot_status,
                    "detections": detections, "latency": lat, "ended": False}
        # 日志结束。stop 是真实回放默认；hold 只用于显式的合成实验。
        if self.on_end in ("stop", "end", "truncate"):
            return {"dt": 0.0, "tx": self._last[0], "ty": self._last[1],
                    "bw": self._last[2], "bh": self._last[3],
                    "cls": self._last[4], "conf": self._last[5],
                    "detected": False, "ref": self._last[6],
                    "dot_status": self._last[7], "detections": [],
                    "latency": 0.0, "ended": True}
        dt = ep.iter
        self._t += dt
        return {"dt": dt, "tx": self._last[0], "ty": self._last[1],
                "bw": self._last[2], "bh": self._last[3],
                "cls": self._last[4], "conf": self._last[5], "detected": True,
                "ref": self._last[6], "dot_status": self._last[7],
                "detections": self._last[8], "latency": 0.0, "ended": False}

    @property
    def t(self):
        return self._t

    def peek_frame_no(self):
        """下一步 step() 将返回的日志帧号(未检出/保持段返回 None)。"""
        i = self._i + 1
        if i < 0 or i > self.ep.f1 - self.ep.f0:
            return None
        return self.ep.f0 + i

    def info(self):
        return {"src": os.path.basename(self.ep.src), "ep": self.ep.idx,
                "points": self.ep.n_points, "frames": self.ep.n_frames,
                "duration": round(self.ep.duration, 3), "iter_ms": round(self.ep.iter * 1000, 1)}


# ── 配置装载（带缓存） ──
_EP_CACHE = {}


def _source_chest_ratio_for_path(rp_cfg, path):
    """Return the aim ratio used when a particular log was recorded."""
    default = float(rp_cfg.get("source_chest_ratio", 0.3))
    overrides = rp_cfg.get("source_chest_ratio_by_log", {}) or {}
    by_name = {str(name).lower(): value for name, value in overrides.items()}
    value = by_name.get(os.path.basename(path).lower(), default)
    return float(value)


def load_episodes(rp_cfg):
    """按 shootsim config 的 replay 段装载全部有效回合（进程内缓存）。

    rp_cfg: {logs: [...], scale_x, scale_y, capture_center, min_points, on_end}
    """
    import json
    key = json.dumps(rp_cfg, sort_keys=True, ensure_ascii=False)
    if key in _EP_CACHE:
        return _EP_CACHE[key]
    scale = (float(rp_cfg.get("scale_x", 0.3011)), float(rp_cfg.get("scale_y", 0.2684)))
    cap_c = float(rp_cfg.get("capture_center", 160.0))
    min_points = int(rp_cfg.get("min_points", 3))
    send_lag = int(rp_cfg.get("send_lag", 2))
    apply_theta = float(rp_cfg.get("apply_theta", 1.0))
    here = os.path.dirname(os.path.abspath(__file__))
    episodes = []
    dropped = 0
    for lp in rp_cfg.get("logs", []):
        path = lp if os.path.isabs(lp) else os.path.normpath(os.path.join(here, lp))
        for e in parse_log_file(
                path, source_chest_ratio=_source_chest_ratio_for_path(rp_cfg, path)):
            try:
                episodes.append(ReplayEpisode(e, len(episodes), scale, cap_c, min_points,
                                              send_lag, apply_theta))
            except ValueError:
                dropped += 1
    episodes.sort(key=lambda e: (e.src, e.t_first))
    for i, e in enumerate(episodes):
        e.idx = i
    _EP_CACHE[key] = episodes
    load_episodes.last_dropped = dropped
    return episodes


# ── 模拟器侧 [鼠标] 日志（格式与 main.py 完全一致，供逐行对比/同一解析器互检） ──
class MouseLog:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = io.open(path, "w", encoding="utf-8")
        self._n = 0

    @staticmethod
    def _ts():
        import time as _t
        return _t.strftime("%H:%M:%S.") + ("%03d" % (int(_t.time() * 1000) % 1000))

    def episode(self, seed, info):
        self._f.write("%s [INFO] ▶ 回合开始 seed=%s %s\n" % (self._ts(), seed, info))
        self._f.flush()

    def line(self, s):
        self._n += 1
        self._f.write("%s [INFO] " % self._ts() + s + "\n")
        self._f.flush()

    def end(self, info):
        self._f.write("%s [INFO] ■ 回合结束 %s\n" % (self._ts(), info))
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    @property
    def lines(self):
        return self._n
