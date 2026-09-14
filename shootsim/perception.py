# -*- coding: utf-8 -*-
"""感知模块：向追踪器提供目标框（屏幕坐标 [x1,y1,x2,y2,cls,conf] 或 None）。

三种模式：
    gt       : 精确地面真值框（env.gt_box，可回放延迟帧）
    gt_noise : 地面真值 + 高斯噪声 + 概率丢帧（模拟真实检测的不确定性）
    yolo     : 复用根目录 main.py 的 YOLO 视觉识别（截取渲染窗口 → 检测）
               —— 仅在可视化模式可用；需要提供 grab_fn（返回 BGR 截图）。
"""
import math
import random

# 检测器缓存：避免 optimize 训练时每回合重复加载 main.py + ONNX 模型
_detector_cache = {}
_aim_main_cache = {}


def _get_aim_main_and_detector(model_path, conf, classes, iou=0.7):
    """加载根目录 main.py 检测器（带缓存，进程内只加载一次）。

    main.py 顶层 Windows API 已做跨平台兼容，非 Windows 也可加载检测器。
    """
    global _detector_cache, _aim_main_cache
    key = (model_path, conf)
    if key in _detector_cache:
        return _detector_cache[key]
    import importlib.util
    import inspect
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    model_path = os.path.normpath(
        model_path if os.path.isabs(model_path) else os.path.join(here, model_path))
    main_path = os.path.join(parent, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"找不到根目录 main.py: {main_path}")
    if main_path in _aim_main_cache:
        aim_main = _aim_main_cache[main_path]
    else:
        spec = importlib.util.spec_from_file_location("aim_main_root", main_path)
        aim_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aim_main)
        _aim_main_cache[main_path] = aim_main
    fn = getattr(aim_main, "_create_detector", None) or getattr(aim_main, "create_detector", None)
    if fn is None:
        raise RuntimeError(f"{main_path} 中没有检测器工厂函数")
    if len(inspect.signature(fn).parameters) >= 4:
        det = fn(model_path, conf, iou, "fp32")
    else:
        det = fn(model_path, conf, iou)
    _detector_cache[key] = det
    return det


class BasePerceiver:
    name = "base"

    def reset(self):
        pass

    def perceive(self, env, delay_frames=0):
        return env.gt_boxes(delay_frames)


class GTPerceiver(BasePerceiver):
    """精确地面真值。"""
    name = "gt"


class GTNoisePerceiver(BasePerceiver):
    """带可控噪声的类别真值感知器。

    默认保持原先的保守噪声；额外参数用于回放更复杂的 YOLO 失真，
    但不在本轮自动调参。
    """
    name = "gt_noise"

    def __init__(self, cfg, rng):
        self.std = float(cfg.get("noise_std_px", 4.0))
        self.miss_rate = float(cfg.get("miss_rate", 0.02))
        self.class_miss_rate = float(cfg.get(
            "class_miss_rate", cfg.get("cls_miss_rate", 0.0)))
        self.size_noise_px = float(cfg.get("size_noise_px", 0.0))
        self.size_noise_frac = float(cfg.get("size_noise_frac", 0.0))
        self.burst_miss_rate = float(cfg.get("burst_miss_rate", 0.0))
        self.burst_miss_min = max(1, int(cfg.get("burst_miss_min", 2)))
        self.burst_miss_max = max(
            self.burst_miss_min, int(cfg.get("burst_miss_max", 6)))
        self.phantom_rate = float(cfg.get("phantom_rate", 0.0))
        self.phantom_size_px = float(cfg.get("phantom_size_px", 40.0))
        self.class_swap_rate = float(cfg.get("class_swap_rate", 0.0))
        self.confidence = float(cfg.get("confidence", 1.0))
        self.confidence_jitter = float(cfg.get("confidence_jitter", 0.0))
        self.delay_jitter_frames = max(
            0, int(cfg.get("delay_jitter_frames", 0)))
        self.rng = rng
        self._burst_remaining = 0

    def reset(self):
        self._burst_remaining = 0

    def _effective_delay(self, delay_frames):
        if not self.delay_jitter_frames:
            return max(0, int(delay_frames))
        jitter = self.rng.randint(-self.delay_jitter_frames,
                                  self.delay_jitter_frames)
        return max(0, int(delay_frames) + jitter)

    def _confidence(self):
        value = self.confidence + self.rng.gauss(0, self.confidence_jitter)
        return max(0.0, min(1.0, value))

    def perceive(self, env, delay_frames=0):
        effective_delay = self._effective_delay(delay_frames)
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            return None
        if self.burst_miss_rate > 0 and self.rng.random() < self.burst_miss_rate:
            self._burst_remaining = self.rng.randint(
                self.burst_miss_min, self.burst_miss_max) - 1
            return None

        if hasattr(env, "gt_detection_boxes"):
            rects = env.gt_detection_boxes(effective_delay)
        else:
            # 兼容旧的轻量测试环境；完整 Env 会走带类别的接口。
            rects = [list(rect) + [0, 1.0]
                     for rect in (env.gt_boxes(effective_delay) or [])]
        if not rects:
            return None
        out = []
        # 同一目标的部位框共享主噪声，避免头/身独立漂移到不合理位置。
        shared_nx = self.rng.gauss(0, self.std)
        shared_ny = self.rng.gauss(0, self.std)
        for rect in rects:
            if self.rng.random() < self.miss_rate:
                continue
            x1, y1, x2, y2, cls, _conf = rect
            if self.rng.random() < self.class_miss_rate:
                continue
            local_std = self.std * 0.15
            nx = shared_nx + self.rng.gauss(0, local_std)
            ny = shared_ny + self.rng.gauss(0, local_std)
            cx = (x1 + x2) * 0.5 + nx
            cy = (y1 + y2) * 0.5 + ny
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            size_noise = self.size_noise_px + self.size_noise_frac * max(width, height)
            width = max(1.0, width + self.rng.gauss(0, size_noise))
            height = max(1.0, height + self.rng.gauss(0, size_noise))
            if self.class_swap_rate > 0 and self.rng.random() < self.class_swap_rate:
                cls = 1 - int(cls)
            out.append([
                cx - width * 0.5, cy - height * 0.5,
                cx + width * 0.5, cy + height * 0.5,
                int(cls), self._confidence(),
            ])
        if self.phantom_rate > 0 and self.rng.random() < self.phantom_rate:
            cx = env.sw * 0.5 + self.rng.gauss(0, self.phantom_size_px)
            cy = env.sh * 0.5 + self.rng.gauss(0, self.phantom_size_px)
            half = self.phantom_size_px * 0.5
            out.append([cx - half, cy - half, cx + half, cy + half,
                        self.rng.randint(0, 1), self._confidence()])
        return out if out else None


class YoloPerceiver(BasePerceiver):
    """复用根目录 main.py 的 YOLO 视觉识别。

    grab_fn() 返回当前渲染窗口的 BGR 截图（由 run_sim 注入）。
    取屏幕中心 capture_size×capture_size 区域送入检测器，
    检测框按"离准星最近"筛选，转换回完整屏幕坐标。

    与 gt 模式对齐的两个行为：
      - 感知延迟：perceive(env, delay_frames) 返回 delay_frames 帧前的检测结果
        （模拟真实回路：截图+推理+相机响应 ≈ 1~2 帧滞后）。
      - 身框优先：头框与某身框高度重叠（IoU>0.4）时丢弃头框，
        复刻 main.py IOU 匹配"身框(cls=0)最高优先"的锁定逻辑，
        防止头/身框交替导致瞄准点上下跳变。

    实战噪声（yolo_noise 配置块，默认全关）：从四段实战日志提取的目标出现规律——
      - drop_rate：每帧丢帧概率，成段丢失 drop_min~drop_max 帧
        （实战：42% 帧无目标点，即使目标在视野内；空窗成段出现）
      - jitter_px：框坐标高斯抖动（实战：锁定框逐帧 ±2~10px）
      - head_swap：身框→头框切换概率（实战：锁定框一半是小头框，与身框来回切换）
      - phantom_rate：误检/异目标假框出现概率，存活 phantom_min~max 帧
        （实战：目标点中位跳变 37px、12% 跳变 >120px——就是假框/异目标在抢锁定）
    """
    name = "yolo"

    def __init__(self, cfg, rng=None, grab_fn=None):
        self.grab_fn = grab_fn
        self.rng = rng if rng is not None else random.Random(0)
        self.capture_size = int(cfg.get("capture_size", 640))
        self.conf = float(cfg.get("yolo_conf", 0.35))
        self.classes = set(cfg.get("yolo_classes", [0, 1, 5]))
        self.head_classes = set(cfg.get("yolo_head_classes", [1]))  # 头框类别
        self.prefer_body = bool(cfg.get("yolo_prefer_body", True))
        # 实战噪声参数
        nz = cfg.get("yolo_noise", {}) or {}
        self.noise_enabled = bool(nz)
        self.drop_rate = float(nz.get("drop_rate", 0.0))
        self.drop_min = int(nz.get("drop_min_frames", 2))
        self.drop_max = int(nz.get("drop_max_frames", 6))
        self.jitter_px = float(nz.get("jitter_px", 0.0))
        self.size_fluct = float(nz.get("size_fluct", 0.0))
        self.head_swap = float(nz.get("head_swap", 0.0))
        # 部分消失（实战：头框/身框单独闪烁丢失，触发锁定类别横跳）
        self.body_drop_rate = float(nz.get("body_drop_rate", 0.0))
        self.head_drop_rate = float(nz.get("head_drop_rate", 0.0))
        self.phantom_rate = float(nz.get("phantom_rate", 0.0))
        self.phantom_min = int(nz.get("phantom_min_frames", 3))
        self.phantom_max = int(nz.get("phantom_max_frames", 8))
        self._hist = []       # 检测结果历史（感知延迟回放）
        self._drop_remain = 0  # 丢帧剩余帧数（成段闪烁）
        self._phantoms = []    # [(bbox, 剩余帧)] 异目标/误检假框
        self._cls_cycle = {}       # both-cls 图：per-cls 消失周期 {cls: {"phase","remain"}}
        self._last_env_frame = 0   # 上一次 perceive 的 env.frame（按实际帧推进周期）
        model_path = cfg.get("yolo_model", "")
        self.detector = None
        if model_path:
            self.detector = self._load_detector(model_path, cfg)

    def reset(self):
        self._hist = []
        self._drop_remain = 0
        self._phantoms = []
        self._cls_cycle = {}
        self._last_env_frame = 0

    def _load_detector(self, model_path, cfg):
        return _get_aim_main_and_detector(model_path, self.conf, self.classes, 0.7)

    @staticmethod
    def _iou(a, b):
        x1, y1, x2, y2 = a
        u1, v1, u2, v2 = b
        ix1, iy1 = max(x1, u1), max(y1, v1)
        ix2, iy2 = min(x2, u2), min(y2, v2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (u2 - u1) * (v2 - v1)
        return inter / max(1e-6, area_a + area_b - inter)

    def _apply_noise(self, boxes, crop_rect):
        """给真实检测框叠加实战噪声：抖动 / 头身切换 / 异目标假框。

        boxes 元素为 [x1,y1,x2,y2,cls]。
        """
        x0, y0, x1, y1 = crop_rect
        out = []
        for b in boxes:
            bx1, by1, bx2, by2, cls = b
            w, h = bx2 - bx1, by2 - by1
            if self.jitter_px > 0:
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                cx += self.rng.gauss(0, self.jitter_px)
                cy += self.rng.gauss(0, self.jitter_px)
                bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            # 尺寸伸缩：实战中同一目标的框宽高逐帧伸缩可达 ±40%
            # （人体部分进入/离开画面、姿态变化），框中心随之漂移 10~30px
            if self.size_fluct > 0:
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                w = max(4.0, w * self.rng.uniform(1 - self.size_fluct, 1 + self.size_fluct))
                h = max(4.0, h * self.rng.uniform(1 - self.size_fluct, 1 + self.size_fluct))
                bx1, by1, bx2, by2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            # 身框→头框切换（实战：锁定框一半是小头框，与身框来回切换）
            if self.head_swap > 0 and h > 30 and self.rng.random() < self.head_swap:
                hw = max(6.0, w * self.rng.uniform(0.5, 0.9))
                hh = max(6.0, h * self.rng.uniform(0.08, 0.18))
                cx, cy = (bx1 + bx2) / 2.0, by1 + h * self.rng.uniform(0.0, 0.06)
                bx1, by1, bx2, by2 = cx - hw / 2, cy, cx + hw / 2, cy + hh
                cls = 1  # 切换后是头框
            out.append([max(x0, bx1), max(y0, by1), min(x1, bx2), min(y1, by2), cls])
        # 异目标/误检假框（实战：目标点 12% 跳变 >120px，就是假框在抢锁定）
        if self.phantom_rate > 0:
            if self._phantoms:
                alive = []
                for pb, prem in self._phantoms:
                    prem -= 1
                    if prem > 0:
                        alive.append([pb, prem])
                        out.append(pb)
                self._phantoms = alive
            if self.rng.random() < self.phantom_rate:
                pw = self.rng.uniform(8.0, 55.0)
                ph = self.rng.uniform(8.0, 60.0)
                # 假框刷在截图区中心附近（模拟"另一个目标/误检出现在准星附近"），
                # 这样它往往比真实目标离准星更近，追踪器会真的切过去——
                # 复现实战中 25% 的 60~120px 和 12% 的 >120px 目标点跳变。
                cx0 = (x0 + x1) / 2.0
                cy0 = (y0 + y1) / 2.0
                ang = self.rng.uniform(0, 2 * math.pi)
                rad = self.rng.uniform(0, 150.0)
                px = cx0 + math.cos(ang) * rad - pw / 2.0
                py = cy0 + math.sin(ang) * rad - ph / 2.0
                px = max(x0 + 4, min(x1 - pw - 4, px))
                py = max(y0 + 4, min(y1 - ph - 4, py))
                life = self.rng.randint(self.phantom_min, self.phantom_max)
                pb = [px, py, px + pw, py + ph, 0]  # cls=0 身框（异目标）
                self._phantoms.append([pb, life])
                out.append(pb)
        return out

    def _cls_cycle_drop(self, cands, env):
        """both-cls 图：某一个 cls 会消失一瞬间（~50 帧显身 + ~10 帧消失），
        按实际帧数推进周期（perceive 每 control_interval 帧调用一次，须换算）。

        single-cls 图的整体消失由 env.visible 处理（目标不渲染 → yolo 检不出），
        这里只对 both-cls 图做 per-cls 消失。
        """
        if not env.targets or env.targets[0].get("cls_type", "both") != "both":
            return cands
        vmin = env.disappear_vis_min
        vmax = env.disappear_vis_max
        hmin = env.disappear_hid_min
        hmax = env.disappear_hid_max
        elapsed = max(1, env.frame - self._last_env_frame)
        self._last_env_frame = env.frame
        present_cls = {d["cls"] for d in cands}
        visible_cls = set()
        for cls in present_cls:
            st = self._cls_cycle.get(cls)
            if st is None:
                st = {"phase": "visible", "remain": self.rng.randint(vmin, vmax)}
                self._cls_cycle[cls] = st
            rem = st["remain"] - elapsed
            while rem <= 0:
                if st["phase"] == "visible":
                    st["phase"] = "hidden"
                    rem += self.rng.randint(hmin, hmax)
                else:
                    st["phase"] = "visible"
                    rem += self.rng.randint(vmin, vmax)
            st["remain"] = rem
            if st["phase"] == "visible":
                visible_cls.add(cls)
        return [d for d in cands if d["cls"] in visible_cls]

    def perceive(self, env, delay_frames=0):
        if self.detector is None or self.grab_fn is None:
            return None
        # 成段丢帧（实战：42% 帧无目标点，空窗成段出现 2~11 个采样点）
        if self._drop_remain > 0:
            self._drop_remain -= 1
            self._hist.append((env.frame, None))
            return None
        if self.drop_rate > 0 and self.rng.random() < self.drop_rate:
            self._drop_remain = self.rng.randint(self.drop_min, self.drop_max)
            self._hist.append((env.frame, None))
            return None
        img = self.grab_fn()
        if img is None:
            return None
        h, w = img.shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0
        # 中心裁剪
        cs = min(self.capture_size, w, h)
        x0, y0 = int(cx0 - cs / 2), int(cy0 - cs / 2)
        crop = img[y0:y0 + cs, x0:x0 + cs]
        try:
            dets = self.detector.detect(crop)
        except Exception:
            dets = []
        cands = [d for d in dets if d["cls"] in self.classes]
        # both-cls 图：per-cls 随机消失（~50帧显身 + ~10帧消失）
        cands = self._cls_cycle_drop(cands, env)
        # 部分消失：头框/身框单独闪烁丢失（实战触发锁定类别横跳）
        # body_drop_rate：身框本帧丢失（只剩头框）
        # head_drop_rate：头框本帧丢失（只剩身框）
        if self.body_drop_rate > 0 or self.head_drop_rate > 0:
            keep = []
            for d in cands:
                if d["cls"] in self.head_classes:
                    if self.rng.random() < self.head_drop_rate:
                        continue  # 头框丢失
                else:
                    if self.rng.random() < self.body_drop_rate:
                        continue  # 身框丢失
                keep.append(d)
            cands = keep
        # 身框优先：头框与某身框高度重叠时丢弃头框（复刻 main.py 身框优先锁定）
        if self.prefer_body and len(cands) > 1:
            bodies = [d for d in cands if d["cls"] not in self.head_classes]
            heads = [d for d in cands if d["cls"] in self.head_classes]
            if bodies and heads:
                cands = bodies + [hd for hd in heads
                                  if not any(self._iou(hd["bbox"], b["bbox"]) > 0.4
                                             for b in bodies)]
        # 转换回完整屏幕坐标（附带 cls，供追踪器做身框优先/身框保持）
        out = []
        for d in cands:
            x1, y1, x2, y2 = d["bbox"]
            out.append([x0 + x1, y0 + y1, x0 + x2, y0 + y2, d["cls"]])
        # 实战噪声（抖动/头身切换/假框）
        if self.noise_enabled and out:
            out = self._apply_noise(out, (x0, y0, x0 + cs, y0 + cs))
        result = out if out else None
        # 感知延迟：返回 delay_frames 实际帧前的结果（历史按帧号记录，
        # 避免 perceive 每 control_interval 帧调用一次导致延迟翻倍）
        self._hist.append((env.frame, result))
        if len(self._hist) > 64:
            self._hist = self._hist[-64:]
        if delay_frames > 0:
            target = env.frame - delay_frames
            found = None
            for fr, res in reversed(self._hist):
                if fr <= target:
                    found = res
                    break
            return found
        return result


class ReplayPerceiver(BasePerceiver):
    """实战日志回放感知：目标框由 env.replay 脚本给出（不跑 yolo、无随机）。

    框即实战系统当帧所见（截图坐标+160 映射到模拟器屏幕），
    延迟参数不适用——实战链路延迟已内含在日志轨迹里。
    """
    name = "replay"

    def perceive(self, env, delay_frames=0):
        return env.replay_detections(delay_frames)


PERCEIVERS = {
    "gt": GTPerceiver,
    "gt_noise": GTNoisePerceiver,
    "yolo": YoloPerceiver,
    "replay": ReplayPerceiver,
}


def create_perceiver(mode, cfg, rng=None, grab_fn=None):
    cls = PERCEIVERS.get(mode, GTNoisePerceiver)
    if mode == "gt_noise":
        return cls(cfg, rng)
    if mode == "yolo":
        return cls(cfg, rng, grab_fn)
    return cls()
