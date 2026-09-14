# -*- coding: utf-8 -*-
"""FPS 射击模拟核心环境（纯逻辑，不依赖 pygame，可 headless 批量运行）。

坐标模型（准星固定、场景移动）：
    世界坐标 (px)：w×h 的平面场景。
    相机 (cam_x, cam_y)：屏幕中心点对应的世界坐标，即"视角朝向"。
    目标世界坐标 (tx, ty)；其在屏幕上的位置 = (tx - cam_x + sw/2, ty - cam_y + sh/2)。
    鼠标右移 → 相机 x 增大 → 目标在屏幕上左移（FPS 视角行为）。

关键机制（本版本）：
    1. 单目标，7 张目标图随机选一张；随机速度 / 随机大小 / 随机位置。
       - 随机大小保持图片宽高比（aspect-preserving），框高 = box_h*scale，
         框宽 = 框高 * 图片宽高比（否则头部特写图会被压扁到无法检出）。
       - 随机位置出生在"环形区域"内（离准星 100~260px，且避开屏幕中下遮挡区），
         保证一开始就在锁定半径内、且能被 yolo 看到。
    2. 随机消失（~50 帧显身 + ~10 帧消失）：
       - both-cls 图：由 perception 层按 cls 单独消失（头框或身框闪断）。
       - single-cls 图：整个目标消失（本文件处理 visible 状态）。
    3. 屏幕遮挡：右键 ADS 后枪身盖住屏幕中下九分之一（renderer 负责画，本文件
       只提供 occlusion_rect 供出生位置避让）。
    4. 红点：子弹实际落点 = 屏幕中心上方 red_dot_offset_px（默认 15px）+ 半径
       red_dot_jitter_px（默认 5px）的抖动。命中判定用红点而不是屏幕中心。

命中判定：红点落在任一目标框内 → 按框垂直两分区判定 头部/身体（已取消脚部 low 区），
伤害 40/20；未命中无伤害。
"""
from collections import deque
import math

# 图片宽高比缓存：避免 optimize 每个回合重复读图
_ASPECT_CACHE = {}


def _image_aspect(path):
    """返回图片宽/高比（失败回退 0.5）。"""
    if path in _ASPECT_CACHE:
        return _ASPECT_CACHE[path]
    aspect = 0.5
    try:
        import cv2
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            if h > 0:
                aspect = w / float(h)
    except Exception:
        pass
    if aspect == 0.5:
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
                if h > 0:
                    aspect = w / float(h)
        except Exception:
            pass
    _ASPECT_CACHE[path] = aspect
    return aspect


# 预标注真值框缓存（cls0=身框 / cls1=头框，归一化 [0,1]），避免每个回合重复读 JSON
_GT_CLS_CACHE = None


def _load_gt_cls():
    """读取 label_targets.py 预标注的 targets_gt.json，返回 {文件名: {cls0:..., cls1:...}}。"""
    global _GT_CLS_CACHE
    if _GT_CLS_CACHE is not None:
        return _GT_CLS_CACHE
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets_gt.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _GT_CLS_CACHE = json.load(f)
    except Exception:
        _GT_CLS_CACHE = {}
    return _GT_CLS_CACHE


class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        self.world_w = float(cfg["world"]["width"])
        self.world_h = float(cfg["world"]["height"])
        self.spawn_margin = float(cfg["world"]["spawn_margin"])
        self.sw = int(cfg["screen"]["width"])
        self.sh = int(cfg["screen"]["height"])
        tgt = cfg["target"]
        self.bw_base = float(tgt.get("box_w", 80.0))
        self.bh_base = float(tgt.get("box_h", 160.0))
        self.size_h_min = float(tgt.get("size_h_min", 90.0))
        self.img_max_h = [float(v) for v in (tgt.get("img_max_h", []) or [])]
        self.fixed_bh = float(tgt.get("fixed_h", 0.0) or 0.0)
        if self.fixed_bh <= 0:
            self.fixed_bh = None
        self.scale = 1.0
        self.bw = self.bw_base
        self.bh = self.bh_base
        self.hp_max = int(tgt["hp"])
        self.sens = float(cfg["view"]["sensitivity"])
        self.sens_y = float(cfg["view"].get("sensitivity_y", self.sens))
        # Camera-mixing model for replay fidelity (must match ReplayEpisode.
        # apply_theta): fraction of each mouse batch the game camera had
        # already applied when the *current* capture was taken.  The rest is
        # queued and applied at the next capture boundary.  1.0 = no latency.
        self._apply_theta = min(1.0, max(0.0, float(cfg["view"].get("apply_theta", 1.0))))
        self._theta_queue = []
        # 发送生效延迟(帧): 实测游戏输入→画面延迟 ≈2 帧(replay_verify.py 开环注入测得)。
        # 模拟发送在本调用后 q 帧才体现在画面上, 与实战闭环延迟一致。0=立即生效(旧行为)。
        self._apply_delay = int(cfg["view"].get("apply_delay_frames", 0))
        self._send_queue = []
        self.fps = int(cfg["view"]["fps"])
        self.dt = 1.0 / self.fps
        self.fire_rate_hz = float(cfg["weapon"]["fire_rate_hz"])
        self.fire_every = max(1, round(self.fps / self.fire_rate_hz))
        self.damage = cfg["weapon"]["damage"]
        self.max_seconds = float(cfg["episode"]["max_seconds"])
        # 目标数量配置（当前只测单目标）
        self.max_count = int(tgt.get("max_count", 1))
        self.min_count = int(tgt.get("min_count", 1))
        self.images = tgt.get("images", []) or []
        self.img_cls_types = tgt.get("img_cls_types", []) or []
        self.gt_cls = _load_gt_cls()   # 预标注真值框（打中该框才算命中）
        # 随机消失（~50 帧显身 + ~10 帧消失）
        self.disappear_vis_min = int(tgt.get("disappear_visible_min", 40))
        self.disappear_vis_max = int(tgt.get("disappear_visible_max", 70))
        self.disappear_hid_min = int(tgt.get("disappear_hidden_min", 8))
        self.disappear_hid_max = int(tgt.get("disappear_hidden_max", 14))
        # yolo 感知：出生在中心截图区内（否则出生在视野外，随机搜索找不到）
        self.spawn_in_crop = bool(tgt.get("spawn_in_crop", True))
        self.crop_size = int((cfg.get("perception") or {}).get("capture_size", 640))
        # 环形出生半径（让目标偏向截图区边缘）
        self.spawn_radius_min = float(tgt.get("spawn_radius_min", 100.0))
        self.spawn_radius_max = float(tgt.get("spawn_radius_max", 260.0))
        # 红点（子弹落点）
        self.rd_offset = float(cfg["weapon"].get("red_dot_offset_px", 15.0))
        self.rd_jitter = float(cfg["weapon"].get("red_dot_jitter_px", 5.0))
        # 后坐力：连发时红点逐发上飘（最大 rd_recoil_max），停火后逐帧回落。
        # 实测实战中红点会飘到中心上方 ~40px（基础 15px + 后坐 ~25px），
        # 所以红点不能再固定 15px，需按连发状态动态偏移。
        self.rd_recoil_max = float(cfg["weapon"].get("red_dot_recoil_max_px", 30.0))
        self.rd_recoil_per_shot = float(cfg["weapon"].get("red_dot_recoil_per_shot_px", 6.0))
        self.rd_recoil_recover = float(cfg["weapon"].get("red_dot_recoil_recover_px", 1.0))
        self.recoil = 0.0
        # 屏幕遮挡(开镜遮挡延迟)功能已按需求移除: 保留字段仅供 renderer 兼容(恒 False)
        self.occl_enabled = False
        # 实战日志回放(replay 模式): 目标轨迹由 replay.ReplayPlayer 给出
        rp = cfg.get("replay", {}) or {}
        self.replay_head_ratio = float(rp.get("head_zone_ratio", 0.25))
        self.replay_dwell_h = float(rp.get("dwell_h_ratio", 0.6))
        self.replay_capture_size = int(rp.get("capture_size", 320))
        self.replay_use_logged_crosshair = bool(
            rp.get("use_logged_crosshair", True))
        self.replay_tracking_only = bool(rp.get("tracking_only", False))
        self.replay = None
        self.replay_det = False
        self.replay_meta = (0, 0.0)
        self.replay_world_detections = []
        self.replay_latency = 0.0
        self.replay_ref = (self.cx if hasattr(self, "cx") else self.sw / 2.0,
                           self.cy if hasattr(self, "cy") else self.sh / 2.0)
        self.replay_dot_status = "unknown"
        self.replay_exhausted = False
        self.replay_target_t = 0.0
        self.replay_above_box_t = 0.0
        self.first_inner60_time = None
        self.first_box_entry_time = None
        self.dwell_rd_t = 0.0
        self.dwell_ch_t = 0.0
        self.post_entry_observed_t = 0.0
        self.post_entry_inner60_t = 0.0
        self.replay_visible_streak_t = 0.0
        self.replay_max_visible_streak_t = 0.0
        # 准星：固定屏幕中心
        self.cx = self.sw / 2.0
        self.cy = self.sh / 2.0

        # 历史（帧级快照）：(targets_snapshot, cam_x, cam_y) — 供感知延迟回放
        self._hist = deque(maxlen=8)

        self.mover = None  # 由 run 循环注入（目标移动策略工厂用）
        self.shooter = None  # 射击策略，由 run 循环注入
        self.frame = 0
        self.t = 0.0
        self.dead = False
        self.kill_time = None
        self.rd = (self.cx, self.cy - self.rd_offset)

    # ── 工具 ──
    def occlusion_rect(self):
        """屏幕遮挡区（3x3 网格的中下格）[x1, y1, x2, y2]。"""
        return [self.sw / 3.0, 2.0 * self.sh / 3.0, 2.0 * self.sw / 3.0, self.sh]

    def _img_aspect(self, img_idx):
        if not self.images:
            return 0.5
        idx = img_idx if 0 <= img_idx < len(self.images) else 0
        return _image_aspect(self.images[idx])

    def _cls_type(self, img_idx):
        """目标 cls 类型：'both'（身+头）或 'single'（仅单类）。"""
        if not self.img_cls_types:
            return "both"
        idx = img_idx if 0 <= img_idx < len(self.img_cls_types) else 0
        t = self.img_cls_types[idx]
        return "single" if str(t).lower() in ("single", "1", "only") else "both"

    # ── 回合生命周期 ──
    def spawn(self, seed=None):
        import random
        self.rng = random.Random(seed if seed is not None else 0)
        # 视角初始朝向随机：相机随机落点
        self.cam_x = self.rng.uniform(0, self.world_w)
        self.cam_y = self.rng.uniform(0, self.world_h)
        # 单目标：随机选一张图
        n = max(1, self.rng.randint(self.min_count, self.max_count))
        self.targets = []
        occl = self.occlusion_rect() if self.occl_enabled else None
        cs = min(self.crop_size, self.sw, self.sh)
        for _ in range(n):
            img_idx = self.rng.randrange(len(self.images)) if self.images else 0
            # 保持宽高比：框高在 [size_h_min, 该图可检出最大高度] 内随机（远~贴脸）
            # fixed_h 用于演示/调试：强制固定框高
            aspect = self._img_aspect(img_idx)
            if self.fixed_bh is not None:
                self.bh = self.fixed_bh
            else:
                h_max = self.img_max_h[img_idx] if 0 <= img_idx < len(self.img_max_h) else self.bh_base
                h_max = max(self.size_h_min, float(h_max))
                self.bh = self.rng.uniform(self.size_h_min, h_max)
            self.bw = self.bh * aspect
            self.scale = self.bh / self.bh_base
            if self.spawn_in_crop:
                cx0 = (self.sw - cs) / 2.0 + cs / 2.0
                cy0 = (self.sh - cs) / 2.0 + cs / 2.0
                # 出生半径自适应：大目标必须更靠近中心，才能完整进入 640 截图区
                max_dim = max(self.bw, self.bh)
                rad_max = min(self.spawn_radius_max, max(0.0, (cs - max_dim) / 2.0 - 8.0))
                rad_min = min(self.spawn_radius_min, rad_max * 0.5) if rad_max > 0 else 0.0
                sx = sy = 0.0
                for _try in range(200):
                    ang = self.rng.uniform(0, 2 * math.pi)
                    rad = self.rng.uniform(rad_min, rad_max)
                    sx = cx0 + math.cos(ang) * rad
                    sy = cy0 + math.sin(ang) * rad
                    # 完全被枪挡住（框顶都在遮挡区之下）→ 重新采样
                    if occl is None or (sy - self.bh / 2.0) < occl[1]:
                        break
            else:
                sx = self.rng.uniform(self.spawn_margin, self.sw - self.spawn_margin)
                sy = self.rng.uniform(self.spawn_margin, self.sh - self.spawn_margin)
            self.targets.append({
                "tx": self.cam_x + (sx - self.cx),
                "ty": self.cam_y + (sy - self.cy),
                "hp": self.hp_max,
                "dead": False,
                "visible": True,
                "cls_type": self._cls_type(img_idx),
                "phase": "visible",   # single-cls 用：整体消失周期
                "phase_remain": self.rng.randint(self.disappear_vis_min, self.disappear_vis_max),
                "mover": None,
                "img_idx": img_idx,
            })

        self.t = 0.0
        self.frame = 0
        self.dead = False
        self.kill_time = None
        self.shots = 0
        self.hits = 0
        self.first_hit_time = None   # 首枪命中时刻(首中评估用)
        self.cum_dmg = 0
        self.region_hits = {"head": 0, "mid": 0}
        self.shot_log = []
        self.dwell_rd_t = 0.0
        self.dwell_ch_t = 0.0
        self.replay = None
        self.replay_det = False
        self.replay_world_detections = []
        self.replay_ref = (self.cx, self.cy)
        self.replay_dot_status = "unknown"
        self.replay_exhausted = False
        self.replay_target_t = 0.0
        self.replay_above_box_t = 0.0
        self.first_inner60_time = None
        self.first_box_entry_time = None
        self.post_entry_observed_t = 0.0
        self.post_entry_inner60_t = 0.0
        self.replay_visible_streak_t = 0.0
        self.replay_max_visible_streak_t = 0.0
        self._reset_normal_observation_metrics()
        self._send_queue = []
        self._theta_queue = []
        if self.mover is not None:
            for tg in self.targets:
                tg["mover"] = self.mover(tg["img_idx"])
                tg["mover"].reset(tg["tx"], tg["ty"])
        self._hist.clear()
        self._hist.append((self._snapshot(), self.cam_x, self.cam_y))
        self.rd = (self.cx, self.cy - self.rd_offset)
        self.recoil = 0.0
        self.update_red_dot()

    def spawn_replay(self, player, seed=None):
        """replay 模式回合初始化: 目标世界轨迹由 ReplayPlayer 逐帧给出。

        世界坐标约定: 回合首帧相机位于 (0,0)。屏幕框 = 世界框 - 相机 + 屏幕中心,
        与实战的"截图框(准星居中)"一一对应。
        """
        import random
        self.rng = random.Random(seed if seed is not None else 0)
        self.replay = player
        player.reset()
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.scale = 1.0
        self.bw = 40.0
        self.bh = 80.0
        self.targets = [{
            "tx": 0.0, "ty": 0.0, "hp": self.hp_max, "dead": False,
            "visible": True, "cls_type": "replay", "phase": "visible",
            "phase_remain": 0, "mover": None, "img_idx": 0,
        }]
        self.t = 0.0
        self.frame = 0
        self.dead = False
        self.kill_time = None
        self.shots = 0
        self.hits = 0
        self.first_hit_time = None
        self.cum_dmg = 0
        self.region_hits = {"head": 0, "mid": 0}
        self.shot_log = []
        self.dwell_rd_t = 0.0
        self.dwell_ch_t = 0.0
        self.replay_det = False
        self.replay_world_detections = []
        self.replay_ref = (self.cx, self.cy)
        self.replay_dot_status = "unknown"
        self.replay_latency = 0.0
        self.replay_exhausted = False
        self.replay_target_t = 0.0
        self.replay_above_box_t = 0.0
        self.first_inner60_time = None
        self.first_box_entry_time = None
        self.post_entry_observed_t = 0.0
        self.post_entry_inner60_t = 0.0
        self.replay_visible_streak_t = 0.0
        self.replay_max_visible_streak_t = 0.0
        self._reset_normal_observation_metrics()
        self._send_queue = []
        self._theta_queue = []
        self.rd = (self.cx, self.cy - self.rd_offset)
        self.recoil = 0.0
        # 射击节拍改为按真实时间驱动(回放帧 dt 来自实战日志, 不均匀)
        self._next_fire_t = 1.0 / self.fire_rate_hz
        # 控制器第一次 perceive 前必须已经看到日志首帧。旧实现等到第一次
        # env.step 才装载首帧，导致整条闭环固定落后一帧。
        first = player.step()
        if first.get("ended"):
            self.replay_exhausted = True
        else:
            self.dt = first["dt"]
            self._set_replay_state(first)
        self._hist.clear()
        self._hist.append((self._snapshot(), self.cam_x, self.cam_y))
        self.update_red_dot()

    def _set_replay_state(self, st):
        """Install one captured replay observation without advancing simulation time."""
        tg = self.targets[0]
        tg["tx"], tg["ty"] = st["tx"], st["ty"]
        self.bw, self.bh = st["bw"], st["bh"]
        self.replay_det = st["detected"]
        self.replay_meta = (st["cls"], st["conf"])
        self.replay_world_detections = list(st.get("detections") or [])
        self.replay_dot_status = str(st.get("dot_status", "unknown"))
        self.replay_latency = max(0.0, float(st.get("latency", 0.0) or 0.0))
        if self.replay_use_logged_crosshair and st.get("ref") is not None:
            off_x = (self.sw - self.replay_capture_size) * 0.5
            off_y = (self.sh - self.replay_capture_size) * 0.5
            self.replay_ref = (float(st["ref"][0]) + off_x,
                               float(st["ref"][1]) + off_y)
            self.rd = self.replay_ref

    def _score_replay_observation(self, duration):
        """Score the currently loaded capture for the time until the next capture."""
        tg = self.targets[0]
        box = self._box(tg)
        # 缺检区间的物理位置只用于保持轨迹连续，不能把外推出来的框
        # 当作真实可观测框参与首入/驻留/偏上评分。
        if box is None or not self.replay_det:
            self.replay_visible_streak_t = 0.0
            return
        self.replay_target_t += duration
        self.replay_visible_streak_t += duration
        self.replay_max_visible_streak_t = max(
            self.replay_max_visible_streak_t, self.replay_visible_streak_t)
        hspan = self.bh * self.replay_dwell_h * 0.5
        bcy = (box[1] + box[3]) * 0.5
        rx, ry = self.rd
        in_box = box[0] <= rx <= box[2] and box[1] <= ry <= box[3]
        in_inner60 = box[0] <= rx <= box[2] and bcy - hspan <= ry <= bcy + hspan
        if in_box and self.first_box_entry_time is None:
            self.first_box_entry_time = self.t
        if in_inner60:
            self.dwell_rd_t += duration
            if self.first_inner60_time is None:
                self.first_inner60_time = self.t
        if self.first_inner60_time is not None:
            self.post_entry_observed_t += duration
            if in_inner60:
                self.post_entry_inner60_t += duration
        if box[0] <= self.cx <= box[2] and bcy - hspan <= self.cy <= bcy + hspan:
            self.dwell_ch_t += duration
        if ry < box[1]:
            self.replay_above_box_t += duration

    def _reset_normal_observation_metrics(self):
        """Reset metrics accumulated at ordinary observation boundaries."""
        self.target_visible_t = 0.0
        self.observable_t = 0.0
        self.detected_observed_t = 0.0
        self.normal_above_box_t = 0.0
        self.normal_post_entry_visible_t = 0.0
        self.normal_error_samples = []
        self.normal_observation_count = 0

    def score_observation(self, detections=None, duration=None):
        """Score an ordinary observation before the new control takes effect.

        Detection boxes are used only for auxiliary detector-availability
        statistics. Aim and dwell metrics always use the current ground-truth
        geometry, and this method never calls ``crosshair_in_box``/``_hit_box``.
        """
        if self.replay is not None:
            return
        duration = self.dt if duration is None else max(0.0, float(duration))
        self.normal_observation_count += 1
        if detections:
            self.detected_observed_t += duration

        tg = next((t for t in self.targets if not t["dead"]), None)
        if tg is not None and tg.get("visible", False):
            self.target_visible_t += duration
        box = self._box(tg) if tg is not None else None
        if box is None:
            return

        self.observable_t += duration
        bcx = (box[0] + box[2]) * 0.5
        bcy = (box[1] + box[3]) * 0.5
        rx, ry = self.rd
        self.normal_error_samples.append(math.hypot(rx - bcx, ry - bcy))
        in_box = box[0] <= rx <= box[2] and box[1] <= ry <= box[3]
        hspan = (box[3] - box[1]) * 0.6 * 0.5
        in_inner60 = box[0] <= rx <= box[2] and bcy - hspan <= ry <= bcy + hspan
        if in_box and self.first_box_entry_time is None:
            self.first_box_entry_time = self.t
        if in_inner60:
            self.dwell_rd_t += duration
            if self.first_inner60_time is None:
                self.first_inner60_time = self.t
        if self.first_inner60_time is not None:
            self.normal_post_entry_visible_t += duration
            if in_inner60:
                self.post_entry_inner60_t += duration
        if box[0] <= self.cx <= box[2] and bcy - hspan <= self.cy <= bcy + hspan:
            self.dwell_ch_t += duration
        if ry < box[1]:
            self.normal_above_box_t += duration

    def _snapshot(self):
        """当前所有目标位置快照。"""
        return [(tg["tx"], tg["ty"], tg["visible"]) for tg in self.targets]

    def timed_out(self):
        return self.t >= self.max_seconds and not self.dead

    def all_dead(self):
        return all(tg["dead"] for tg in self.targets)

    # ── 鼠标/视角 ──
    def apply_mouse(self, dx, dy):
        """鼠标相对位移(counts) → 相机移动。右移 → 目标在屏幕上左移。
        垂直灵敏度独立(view.sensitivity_y, 实机 0.2684), 与回放扣除比例一致。
        _apply_theta<1 时按重建模型把 (1-theta) 份额推迟到下一次截图边界生效。"""
        if self._apply_theta >= 1.0 and self._apply_delay <= 0:
            self.cam_x += dx * self.sens
            self.cam_y += dy * self.sens_y
            return
        if self._apply_theta < 1.0:
            if dx or dy:
                self._theta_queue.append((dx * (1.0 - self._apply_theta),
                                          dy * (1.0 - self._apply_theta)))
            dx *= self._apply_theta
            dy *= self._apply_theta
        if self._apply_delay <= 0:
            self.cam_x += dx * self.sens
            self.cam_y += dy * self.sens_y
            return
        # 延迟单位是画面帧，不是 SendInput 调用次数。同一帧的多个子步必须
        # 具有相同的落地帧，不能因为连续调用而互相“推进时间”。
        while self._send_queue and self._send_queue[0][2] <= self.frame:
            ddx, ddy, _ = self._send_queue.pop(0)
            self.cam_x += ddx * self.sens
            self.cam_y += ddy * self.sens_y
        if dx or dy:
            self._send_queue.append((dx, dy, self.frame + self._apply_delay))

    def drain_theta_queue(self):
        """Apply the deferred (1-theta) camera fraction at a capture boundary."""
        if not self._theta_queue:
            return
        for ddx, ddy in self._theta_queue:
            self.cam_x += ddx * self.sens
            self.cam_y += ddy * self.sens_y
        self._theta_queue.clear()

    # ── 红点 ──
    def update_red_dot(self):
        """红点 = 屏幕中心上方 (rd_offset + recoil) px + 半径 rd_jitter 抖动。

        连发时 recoil 逐发累积（后坐力上飘，上限 rd_recoil_max），停火后逐帧回落。
        """
        if self.replay is not None and self.replay_use_logged_crosshair:
            self.rd = self.replay_ref
            return
        self.recoil = max(0.0, self.recoil - self.rd_recoil_recover)
        ang = self.rng.uniform(0, 2 * math.pi)
        rad = self.rng.uniform(0, self.rd_jitter)
        jx = math.cos(ang) * rad
        jy = math.sin(ang) * rad
        self.rd = (self.cx + jx, self.cy - self.rd_offset - self.recoil + jy)

    def red_dot(self):
        return self.rd

    # ── 屏幕坐标换算 ──
    def target_screen(self, tx, ty, cam_x=None, cam_y=None):
        cam_x = self.cam_x if cam_x is None else cam_x
        cam_y = self.cam_y if cam_y is None else cam_y
        return (tx - cam_x + self.cx, ty - cam_y + self.cy)

    def _box(self, tg, delay_frames=0):
        """单目标框（屏幕坐标），可回放延迟帧。消失/死亡返回 None。"""
        if tg["dead"]:
            return None
        if delay_frames > 0 and len(self._hist) > delay_frames:
            snap, cam_x, cam_y = self._hist[-1 - delay_frames]
            idx = next((i for i, t in enumerate(self.targets) if t is tg), None)
            if idx is None or idx >= len(snap):
                return None
            tx, ty, vis = snap[idx]
            if not vis:
                return None
        else:
            if not tg["visible"]:
                return None
            tx, ty = tg["tx"], tg["ty"]
            cam_x, cam_y = self.cam_x, self.cam_y
        sx, sy = self.target_screen(tx, ty, cam_x, cam_y)
        rect = [sx - self.bw / 2, sy - self.bh / 2, sx + self.bw / 2, sy + self.bh / 2]
        if rect[2] < 0 or rect[0] > self.sw or rect[3] < 0 or rect[1] > self.sh:
            return None
        return rect

    def gt_boxes(self, delay_frames=0):
        """所有可见目标框（屏幕坐标 [x1,y1,x2,y2]）列表。"""
        return [self._box(tg, delay_frames) for tg in self.targets if self._box(tg, delay_frames) is not None]

    def gt_detection_boxes(self, delay_frames=0):
        """返回带类别的真值检测框，供 GTNoisePerceiver 使用。"""
        out = []
        for tg in self.targets:
            if tg["dead"]:
                continue
            parts = self.gt_cls_boxes(tg, delay_frames=delay_frames)
            if not parts:
                box = self._box(tg, delay_frames)
                if box is not None:
                    out.append([*box, 0, 1.0])
                continue
            if self._cls_type(tg["img_idx"]) == "single":
                parts = {"cls0": parts.get("cls0") or parts.get("cls1")}
            for key, box in parts.items():
                if box is not None:
                    out.append([*box, 1 if key == "cls1" else 0, 1.0])
        return out

    def gt_box(self, delay_frames=0):
        """最近目标框（离红点最近，保持单目标兼容）。"""
        boxes = self.gt_boxes(delay_frames)
        if not boxes:
            return None
        rx, ry = self.red_dot()
        return min(boxes, key=lambda b: ((b[0] + b[2]) / 2 - rx) ** 2 + ((b[1] + b[3]) / 2 - ry) ** 2)

    def replay_detections(self, delay_frames=0):
        """replay 模式感知: 本帧"实战系统看到什么"。

        返回 [[x1,y1,x2,y2,cls,conf], ...]（模拟器屏幕坐标, =截图坐标+160 偏移）,
        无检出帧返回 None。延迟 obs_delay 不适用(实战日志的延迟已内含)。
        """
        if self.replay is None or not self.replay_det:
            return None
        crop = max(1.0, float(self.replay_capture_size))
        crop_x1 = self.cx - crop * 0.5
        crop_y1 = self.cy - crop * 0.5
        crop_x2 = self.cx + crop * 0.5
        crop_y2 = self.cy + crop * 0.5
        out = []
        for det in self.replay_world_detections:
            x1 = float(det[0]) - self.cam_x + self.cx
            y1 = float(det[1]) - self.cam_y + self.cy
            x2 = float(det[2]) - self.cam_x + self.cx
            y2 = float(det[3]) - self.cam_y + self.cy
            if x2 <= crop_x1 or x1 >= crop_x2 or y2 <= crop_y1 or y1 >= crop_y2:
                continue
            out.append([
                max(crop_x1, x1), max(crop_y1, y1),
                min(crop_x2, x2), min(crop_y2, y2),
                int(det[4]), float(det[5]),
            ])
        if out:
            return out
        # Legacy logs contain only the selected box.
        if self.replay_world_detections:
            return None
        tg = self.targets[0]
        b = self._box(tg, delay_frames)
        if b is None or b[2] <= crop_x1 or b[0] >= crop_x2 \
                or b[3] <= crop_y1 or b[1] >= crop_y2:
            return None
        cls, conf = self.replay_meta
        return [[max(crop_x1, b[0]), max(crop_y1, b[1]),
                 min(crop_x2, b[2]), min(crop_y2, b[3]), cls, conf]]

    def gt_cls_boxes(self, tg, delay_frames=0):
        """返回目标当前真实部位框（屏幕坐标）：
        {cls0: [x1,y1,x2,y2] 或 None, cls1: [...] 或 None}。
        这些是预先用 YOLO 标好的"真实目标"（打中才算击中），不随运行时检测抖动。
        replay 模式: 头框=顶部 head_zone_ratio 区, 身框=整框。
        """
        if self.replay is not None:
            b = self._box(tg)
            if b is None:
                return {}
            hy = b[1] + (b[3] - b[1]) * self.replay_head_ratio
            return {"cls0": list(b), "cls1": [b[0], b[1], b[2], hy]}
        idx = tg["img_idx"] if 0 <= tg["img_idx"] < len(self.images) else 0
        info = self.gt_cls.get(self.images[idx]) if idx < len(self.images) else None
        # 配置中的图片可以带相对目录（例如从项目根目录启动时的
        # ``shootsim/foo.png``），而预标注表按文件名保存；两者都支持。
        if info is None and idx < len(self.images):
            import os
            info = self.gt_cls.get(os.path.basename(self.images[idx]))
        if not info:
            return {}
        full_box = self._box(tg, delay_frames=delay_frames)
        if full_box is None:
            return {}
        fx1, fy1, fx2, fy2 = full_box
        out = {}
        for key in ("cls0", "cls1"):
            nb = info.get(key)
            if nb:
                out[key] = [
                    fx1 + nb[0] * (fx2 - fx1),
                    fy1 + nb[1] * (fy2 - fy1),
                    fx1 + nb[2] * (fx2 - fx1),
                    fy1 + nb[3] * (fy2 - fy1),
                ]
        return out

    def crosshair_in_box(self):
        """红点是否落在某个真实部位框（cls0/cls1）内。"""
        box, _ = self._hit_box()
        return box is not None

    def _hit_box(self):
        """返回红点命中的真实部位框与其 cls：(box, cls)；未命中 (None, None)。

        replay 模式: 用回放脚本的真实框, 头部=框顶部 head_zone_ratio 区(默认25%)。
        传统模式: 优先头框(cls=1)，再身框(cls=0)；脚部（框外）不算命中。
        """
        rx, ry = self.red_dot()
        if self.replay is not None:
            tg = self.targets[0]
            b = self._box(tg)
            if b is None:
                return None, None
            hy = b[1] + (b[3] - b[1]) * self.replay_head_ratio
            if b[0] <= rx <= b[2] and b[1] <= ry <= hy:
                return [b[0], b[1], b[2], hy], 1
            if b[0] <= rx <= b[2] and hy <= ry <= b[3]:
                return b, 0
            return None, None
        rx, ry = self.red_dot()
        for tg in self.targets:
            if tg["dead"] or not tg["visible"]:
                continue
            gb = self.gt_cls_boxes(tg)
            if not gb:
                continue
            hb = gb.get("cls1")
            if hb and hb[0] <= rx <= hb[2] and hb[1] <= ry <= hb[3]:
                return hb, 1
            bb = gb.get("cls0")
            if bb and bb[0] <= rx <= bb[2] and bb[1] <= ry <= bb[3]:
                return bb, 0
        return None, None

    def hit_region(self):
        """按预标注框判定命中区域：cls=1（头框）→ 头部；cls=0（身框）→ 胸部。"""
        box, cls = self._hit_box()
        if box is None:
            return None
        return "head" if cls == 1 else "mid"

    # ── 主循环 ──
    def step(self, mouse_dx=0.0, mouse_dy=0.0, advance_to=None):
        """推进一帧。返回本帧是否继续（未全死且未超时）。"""
        if self.replay is not None:
            # ── replay 模式: 目标世界位置来自实战日志脚本 ──
            self.drain_theta_queue()
            st = self.replay.step()
            if st.get("ended"):
                self.replay_exhausted = True
                return False
            self.dt = st["dt"]
            # 当前检测框、红点和控制器输入来自同一张截图。先为当前截图
            # 计分，再让输出在 dt 内落地并装入下一张截图。
            self._score_replay_observation(self.dt)
            if advance_to is not None:
                advance_to(self.t + self.dt)
            self._set_replay_state(st)
            self.apply_mouse(mouse_dx, mouse_dy)
            self._hist.append((self._snapshot(), self.cam_x, self.cam_y))
            self.frame += 1
            self.t += self.dt
            while self._next_fire_t <= self.t:
                if not self.replay_tracking_only:
                    self._fire_shot()
                self._next_fire_t += 1.0 / self.fire_rate_hz
            if self.all_dead():
                self.dead = True
            return not self.dead and not self.timed_out()
        # 1. 目标移动 + 消失/重现
        for tg in self.targets:
            if tg["dead"]:
                continue
            # 移动（消失期间也继续移动，更真实）
            if tg["mover"] is not None:
                tg["tx"], tg["ty"] = tg["mover"].update(self.dt, tg["tx"], tg["ty"])
            # single-cls 图：整体消失周期（~50 帧显身 + ~10 帧消失）
            if tg["cls_type"] == "single":
                tg["phase_remain"] -= 1
                if tg["phase_remain"] <= 0:
                    if tg["phase"] == "visible":
                        tg["phase"] = "hidden"
                        tg["phase_remain"] = self.rng.randint(self.disappear_hid_min, self.disappear_hid_max)
                    else:
                        tg["phase"] = "visible"
                        tg["phase_remain"] = self.rng.randint(self.disappear_vis_min, self.disappear_vis_max)
                tg["visible"] = (tg["phase"] == "visible")
            else:
                # both-cls 图：整体不消失，per-cls 消失由 perception 层处理
                tg["visible"] = True
        # 2. 应用鼠标位移
        self.apply_mouse(mouse_dx, mouse_dy)
        # 3. 历史快照
        self._hist.append((self._snapshot(), self.cam_x, self.cam_y))
        # 4. 射击节拍
        self.frame += 1
        self.t += self.dt
        if self.frame % self.fire_every == 0:
            self._fire_shot()
        # 全部目标死亡 → 结束
        if self.all_dead():
            self.dead = True
        return not self.dead and not self.timed_out()

    def _fire_shot(self):
        """hit-scan：从红点(子弹落点)发一枪。射击策略决定本拍是否开火。
        (开镜遮挡延迟功能已移除)"""
        if self.shooter is not None:
            # 只有 oracle 策略允许在开火决策阶段读取真实命中框。
            # lock_delay/always 与真实命中评估保持隔离。
            crosshair_in_box = (
                self.crosshair_in_box()
                if getattr(self.shooter, "is_oracle", False) else None
            )
            if not self.shooter.should_fire(crosshair_in_box):
                return None
        self.shots += 1
        # 后坐力：每发一枪红点向上飘一点，连发累积到 rd_recoil_max 上限
        self.recoil = min(self.rd_recoil_max, self.recoil + self.rd_recoil_per_shot)
        box, box_cls = self._hit_box()
        hit = box is not None
        region = None
        dmg = 0
        tg = self.targets[0] if self.targets else None
        if hit and tg is not None and not tg["dead"]:
            region = self.hit_region()
            dmg = self.damage[region]
            tg["hp"] -= dmg
            self.cum_dmg += dmg
            self.hits += 1
            if self.first_hit_time is None:
                self.first_hit_time = self.t
            self.region_hits[region] += 1
            if tg["hp"] <= 0:
                tg["dead"] = True
                if self.all_dead() and self.kill_time is None:
                    self.kill_time = self.t
        rd = self.red_dot()
        record = {
            "frame": self.frame,
            "t": round(self.t, 4),
            "crosshair": [round(rd[0], 1), round(rd[1], 1)],
            "box": [round(v, 1) for v in box[:4]] if box is not None else None,
            "hit": hit,
            "region": region,
            "dmg": dmg,
            "cum_dmg": self.cum_dmg,
            "hp": max(0, tg["hp"]) if tg else self.hp_max,
            "alive": sum(1 for x in self.targets if not x["dead"]),
        }
        self.shot_log.append(record)
        return record

    # ── 回合结果 ──
    def summary(self):
        import statistics
        if self.replay is not None:
            dwell_denominator = self.replay_target_t
            visible_seconds = self.replay_target_t
            observable_seconds = self.replay_target_t
            above_seconds = self.replay_above_box_t
        else:
            # 普通模式按真实可见时长计分，检测丢帧不会从分母消失。
            dwell_denominator = self.target_visible_t
            visible_seconds = self.target_visible_t
            observable_seconds = self.observable_t
            above_seconds = self.normal_above_box_t
        true_timeout = bool(
            self.first_inner60_time is None
            and (
                self.timed_out() if self.replay is None else
                self.replay_max_visible_streak_t >=
                self.max_seconds - max(1e-3, self.dt)
            )
        )
        errors = self.normal_error_samples
        shooter = self.shooter
        accuracy_valid = True if shooter is None else bool(
            getattr(shooter, "accuracy_valid", True))
        accuracy_note = "" if shooter is None else str(
            getattr(shooter, "accuracy_note", ""))
        return {
            "kill_time": round(self.kill_time, 4) if self.kill_time is not None else None,
            "timeout": self.timed_out() and not self.replay_exhausted,
            "frames": self.frame,
            "shots": self.shots,
            "hits": self.hits,
            "first_hit_time": round(self.first_hit_time, 4) if self.first_hit_time is not None else None,
            "first_box_entry_time": (round(self.first_box_entry_time, 4)
                                     if self.first_box_entry_time is not None else None),
            "accuracy": round(self.hits / self.shots, 4) if self.shots else 0.0,
            "accuracy_valid": accuracy_valid,
            "accuracy_note": accuracy_note,
            "shot_policy": (getattr(shooter, "policy", None)
                             if shooter is not None else None),
            "cum_dmg": self.cum_dmg,
            "region_hits": dict(self.region_hits),
            "targets": len(self.targets),
            "scale": round(self.scale, 3),
            "hp_left": sum(max(0, tg["hp"]) for tg in self.targets),
            # 60%区域停留占比(时间占比): 红点=准星参考口径 / 十字=几何中心口径
            "dwell_reddot": (round(self.dwell_rd_t / dwell_denominator, 4)
                              if dwell_denominator > 0 else 0.0),
            "dwell_cross": (round(self.dwell_ch_t / dwell_denominator, 4)
                             if dwell_denominator > 0 else 0.0),
            # 保留原始时长，供多回合统计按可观测时间加权。短日志若直接
            # 对百分比取平均，会与长日志拥有相同权重，容易扭曲寻优结果。
            "dwell_reddot_seconds": round(self.dwell_rd_t, 4),
            "dwell_cross_seconds": round(self.dwell_ch_t, 4),
            "first_inner60_time": (round(self.first_inner60_time, 4)
                                     if self.first_inner60_time is not None else None),
            "post_entry_inner60_dwell": (
                round(
                    self.post_entry_inner60_t /
                    (self.post_entry_observed_t if self.replay is not None
                     else self.normal_post_entry_visible_t), 4
                )
                if (self.post_entry_observed_t if self.replay is not None
                    else self.normal_post_entry_visible_t) > 0 else 0.0
            ),
            "post_entry_inner60_seconds": round(self.post_entry_inner60_t, 4),
            "post_entry_observed_seconds": round(
                self.post_entry_observed_t if self.replay is not None
                else self.normal_post_entry_visible_t, 4),
            "above_box_ratio": (
                round(self.replay_above_box_t / self.replay_target_t, 4)
                if self.replay is not None and self.replay_target_t > 0
                else round(above_seconds / visible_seconds, 4)
                if visible_seconds > 0 else 0.0
            ),
            "above_box_seconds": round(above_seconds, 4),
            "target_visible_seconds": round(visible_seconds, 4),
            "observable_seconds": round(observable_seconds, 4),
            "detected_observed_seconds": round(
                self.detected_observed_t if self.replay is None else 0.0, 4),
            "detected_visible_ratio": (
                round(self.detected_observed_t / visible_seconds, 4)
                if self.replay is None and visible_seconds > 0 else 0.0
            ),
            "observable_ratio": (round(observable_seconds / visible_seconds, 4)
                                  if visible_seconds > 0 else 0.0),
            "never_inner60_timeout": bool(
                self.first_inner60_time is None
                and (self.timed_out() if self.replay is None else true_timeout)
            ),
            "timeout_rate": 1.0 if self.timed_out() else 0.0,
            "peak_error_px": round(max(errors), 4) if errors else 0.0,
            "p50_error_px": round(statistics.median(errors), 4) if errors else None,
            "p75_error_px": (
                round(statistics.quantiles(errors, n=4, method="inclusive")[2], 4)
                if len(errors) >= 2 else (round(errors[0], 4) if errors else None)
            ),
            "replay_exhausted": self.replay_exhausted,
            "replay_observed_seconds": round(self.replay_target_t, 4),
            "max_visible_streak_seconds": round(self.replay_max_visible_streak_t, 4),
            "true_timeout": true_timeout,
            "entry_censored": bool(self.first_inner60_time is None and not true_timeout),
            "replay": None if self.replay is None else self.replay.info(),
        }
