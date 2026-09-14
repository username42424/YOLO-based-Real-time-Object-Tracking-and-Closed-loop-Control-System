# -*- coding: utf-8 -*-
"""pygame 可视化渲染。

FPS 视角特点：准星固定在屏幕中心；鼠标移动时整个场景（世界网格 + 目标）相对移动。
绘制内容：世界网格背景、目标(图/占位图)+框+三区域线、血量条、命中特效、
射击轨迹、准星、HUD(时间/血量/命中/击杀时间)。
"""
import os
import math

import pygame


def _generate_sprite(w, h):
    """生成占位目标图（头部/躯干/腿部三区域，颜色区分）。"""
    s = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
    hh = h / 3.0
    # 头（上部区域）
    pygame.draw.ellipse(s, (220, 70, 70), (w * 0.3, 0, w * 0.4, hh * 0.9))
    # 躯干（中部）
    pygame.draw.rect(s, (60, 90, 200), (w * 0.15, hh, w * 0.7, hh))
    # 腿（下部）
    pygame.draw.rect(s, (40, 140, 70), (w * 0.25, hh * 2, w * 0.18, hh * 0.9))
    pygame.draw.rect(s, (40, 140, 70), (w * 0.57, hh * 2, w * 0.18, hh * 0.9))
    return s


class Renderer:
    def __init__(self, cfg, headless=False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.sw = int(cfg["screen"]["width"])
        self.sh = int(cfg["screen"]["height"])
        self.bw = float(cfg["target"]["box_w"])
        self.bh = float(cfg["target"]["box_h"])
        self.image_path = cfg["target"].get("image", "") or None
        self.images = cfg["target"].get("images", []) or []
        self.headless = headless
        self.screen = None
        self.font = None
        self.font_big = None
        self.sprites = []   # 多目标：每张图片一个 sprite
        self.effects = []   # (age, x, y, region, dmg)
        self.tracers = []   # (age, x1, y1, x2, y2)
        self.clock = None
        self._sprite_h = None
        self.perception_surf = None   # 供 YOLO 感知用的"干净"帧（不含调试框/HUD）

    # ── 生命周期 ──
    def start(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.sw, self.sh))
        self.perception_surf = pygame.Surface((self.sw, self.sh))
        pygame.display.set_caption("ShootSim — FPS 射击追踪模拟")
        if not self.headless:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        self.font = pygame.font.Font(None, 22)
        self.font_big = pygame.font.Font(None, 30)
        self.clock = pygame.time.Clock()
        # 目标图：images 列表（多目标）或单 image 或生成的占位图。
        # 先加载原始图，尺寸按每回合 env.bw/env.bh 在 draw 时缩放（目标尺寸随机）。
        self._raw_images = []
        here = os.path.dirname(os.path.abspath(__file__))
        img_paths = list(self.images) if self.images else ([self.image_path] if self.image_path else [])
        for ip in img_paths:
            if not os.path.isabs(ip):
                ip = os.path.join(here, ip)  # 相对 shootsim 目录解析
            if ip and os.path.exists(ip):
                self._raw_images.append(pygame.image.load(ip).convert_alpha())
            else:
                self._raw_images.append(None)  # 缺失 → 占位图
        if not self._raw_images:
            self._raw_images.append(None)
        self.sprites = []
        self._sprite_h = None

    def _ensure_sprites(self, env):
        """目标尺寸每回合随机：保持各图宽高比，按框高 env.bh 缩放（避免压扁导致
        头部特写等图无法被 yolo 检出）。占位图仍用 env.bw/env.bh。
        回放模式(env.replay 非 None): 目标用中性占位人形(头/躯干/腿色块),
        不贴真实截图——识别与命中完全由日志回放框驱动, 贴图仅为示意。"""
        bh = max(1, int(round(env.bh)))
        if getattr(env, "replay", None) is not None:
            self.sprites = [_generate_sprite(max(1, int(round(env.bw))), bh)]
            self._sprite_h = bh
            return
        if self._sprite_h == bh:
            return
        sprites = []
        for raw in self._raw_images:
            if raw is None:
                sprites.append(_generate_sprite(max(1, int(round(env.bw))), bh))
            else:
                w, h = raw.get_width(), raw.get_height()
                aspect = w / max(1, h)
                bw = max(1, int(round(bh * aspect)))
                sprites.append(pygame.transform.smoothscale(raw, (bw, bh)))
        self.sprites = sprites
        self._sprite_h = bh

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass

    # ── 输入 ──
    def poll_events(self):
        """返回 (rel_x, rel_y, quit)。物理鼠标位移 = 视角移动。"""
        rel_x = rel_y = 0.0
        quit_flag = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_flag = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                quit_flag = True
        if not self.headless:
            dx, dy = pygame.mouse.get_rel()
            rel_x += dx
            rel_y += dy
        return rel_x, rel_y, quit_flag

    # ── 特效 ──
    def add_hit(self, region, dmg, sx, sy):
        self.effects.append([0.0, sx, sy, region, dmg])
        # 射击轨迹：准星 → 命中点
        self.tracers.append([0.0, self.sw / 2, self.sh / 2, sx, sy])

    # ── 绘制 ──
    def draw(self, env, hud=None, yolo_boxes=None):
        if self.screen is None:
            self.start()
        self._ensure_sprites(env)
        surf = self.screen
        # ── 背景：世界网格（随相机移动，体现"场景移动"）──
        surf.fill((32, 34, 40))
        step = 100
        ox = env.cam_x % step
        oy = env.cam_y % step
        x0 = self.sw / 2.0
        y0 = self.sh / 2.0
        for gx in range(int(-x0 - step), int(self.sw - x0 + step)):
            wx = gx - (self.sw / 2 - ox) % step + self.sw / 2
            pygame.draw.line(surf, (48, 52, 62), (wx, 0), (wx, self.sh), 1)
        for gy in range(int(-y0 - step), int(self.sh - y0 + step)):
            wy = gy - (self.sh / 2 - oy) % step + self.sh / 2
            pygame.draw.line(surf, (48, 52, 62), (0, wy), (self.sw, wy), 1)
        # 世界边界
        wx1 = 0 - env.cam_x + self.sw / 2
        wy1 = 0 - env.cam_y + self.sh / 2
        wx2 = env.world_w - env.cam_x + self.sw / 2
        wy2 = env.world_h - env.cam_y + self.sh / 2
        pygame.draw.rect(surf, (80, 60, 40), (wx1, wy1, wx2 - wx1, wy2 - wy1), 3)

        # ── 目标精灵（只画本体，不含调试框）──
        target_rects = []
        for tg in env.targets:
            rect = env._box(tg, 0)
            if rect is None:
                target_rects.append((tg, None))
                continue
            sx, sy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
            idx = tg["img_idx"] if 0 <= tg["img_idx"] < len(self.sprites) else 0
            sprite = self.sprites[idx]
            sprite_rect = sprite.get_rect(center=(sx, sy))
            surf.blit(sprite, sprite_rect)
            target_rects.append((tg, rect))

        # ── 屏幕遮挡（右键 ADS 后枪身盖住中下九分之一，从下方滑上来）──
        if env.occl_enabled:
            p = min(1.0, env.t * 1000.0 / max(1.0, env.occl_duration_ms))
            ox1, oy1, ox2, oy2 = env.occlusion_rect()
            top = oy2 - (oy2 - oy1) * p
            pygame.draw.rect(surf, (10, 10, 12), (ox1, top, ox2 - ox1, oy2 - top))

        # ── 准星（固定屏幕中心，蓝色十字）+ 红点（子弹落点，上方 offset px）──
        cx, cy = self.sw / 2, self.sh / 2
        pygame.draw.line(surf, (80, 160, 255), (cx - 14, cy), (cx - 4, cy), 2)
        pygame.draw.line(surf, (80, 160, 255), (cx + 4, cy), (cx + 14, cy), 2)
        pygame.draw.line(surf, (80, 160, 255), (cx, cy - 14), (cx, cy - 4), 2)
        pygame.draw.line(surf, (80, 160, 255), (cx, cy + 4), (cx, cy + 14), 2)
        pygame.draw.circle(surf, (80, 160, 255), (int(cx), int(cy)), 2)
        rx, ry = env.red_dot()
        pygame.draw.circle(surf, (255, 40, 40), (int(rx), int(ry)), 3)

        # ── 感知快照：YOLO 只看"游戏内画面"（目标+遮挡+准星+红点），
        #    不含调试绿框/HUD/特效——避免把真值框喂给 YOLO 造成作弊 ──
        if self.perception_surf is not None:
            self.perception_surf.blit(surf, (0, 0))

        # ── 射击轨迹 ──
        for tr in self.tracers:
            tr[0] += 1
            if tr[0] < 8:
                alpha = 220 - tr[0] * 25
                col = (255, 200, 80)
                pygame.draw.line(surf, col, (tr[1], tr[2]), (tr[3], tr[4]), 2)
        self.tracers = [t for t in self.tracers if t[0] < 8]

        # ── 命中特效 ──
        for ef in self.effects:
            ef[0] += 1
            if ef[0] < 20:
                region_colors = {"head": (255, 60, 60), "mid": (255, 170, 60)}
                col = region_colors.get(ef[3], (255, 255, 255))
                r = 8 + ef[0] * 2
                pygame.draw.circle(surf, col, (int(ef[1]), int(ef[2])), r, 2)
                if ef[4]:
                    txt = self.font.render(str(ef[4]), True, col)
                    surf.blit(txt, (ef[1] + 8, ef[2] - 18 - ef[0]))
        self.effects = [e for e in self.effects if e[0] < 20]

        # ── 调试：真实部位框（预标注 cls0=身/绿、cls1=头/青）+ 血量条 ──
        for tg, rect in target_rects:
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            gb = env.gt_cls_boxes(tg)
            bb = gb.get("cls0")
            hb = gb.get("cls1")
            if bb:
                pygame.draw.rect(surf, (0, 200, 0), (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]), 2)
            if hb:
                pygame.draw.rect(surf, (0, 220, 220), (hb[0], hb[1], hb[2] - hb[0], hb[3] - hb[1]), 2)
            hpw = x2 - x1
            pygame.draw.rect(surf, (60, 60, 60), (x1, y1 - 12, hpw, 7))
            hp_ratio = max(0.0, tg["hp"] / env.hp_max)
            pygame.draw.rect(surf, (255, 60, 60), (x1, y1 - 12, hpw * hp_ratio, 7))
            if not tg["visible"]:
                txt = self.font.render("HIDDEN", True, (255, 255, 0))
                surf.blit(txt, (x1, y1 - 26))

        # ── YOLO 实时识别框（黄色，可肉眼核对"真的是 yolo 在识别"）──
        if yolo_boxes:
            for b in yolo_boxes:
                if len(b) < 4:
                    continue
                bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
                cls = int(b[4]) if len(b) >= 5 else 0
                pygame.draw.rect(surf, (255, 220, 40), (bx1, by1, bx2 - bx1, by2 - by1), 2)
                txt = self.font.render(f"cls{cls}", True, (255, 220, 40))
                surf.blit(txt, (bx1, max(0, by1 - 14)))

        # ── HUD ──
        if hud:
            lines = [
                f"TIME {hud.get('t', 0):.2f}s   HP {max(0, hud.get('hp', 0))}",
                f"SHOTS {hud.get('shots', 0)}  HITS {hud.get('hits', 0)}  ACC {hud.get('acc', 0):.0%}",
                f"REGION head:{hud.get('rh', {}).get('head', 0)} mid:{hud.get('rh', {}).get('mid', 0)}",
            ]
            if hud.get("kill_time") is not None:
                lines.append(f"KILL {hud['kill_time']:.2f}s  <<<")
            elif hud.get("timeout"):
                lines.append("TIMEOUT")
            for i, ln in enumerate(lines):
                txt = self.font_big.render(ln, True, (230, 230, 230))
                surf.blit(txt, (12, 8 + i * 26))

        pygame.display.flip()

    def tick(self, fps):
        # headless（dummy SDL）不 throttle，训练更快
        if self.clock is not None and not self.headless:
            self.clock.tick(fps)

    # ── 工具 ──
    def save_frame(self, path):
        if self.screen is not None:
            pygame.image.save(self.screen, path)

    def grab_frame_bgr(self):
        """返回"干净帧"（不含调试框/HUD）的 BGR numpy 截图，供 yolo 感知使用。

        只包含：目标精灵 + 屏幕遮挡(枪) + 准星 + 红点，即游戏内画面。
        调试绿框/HUD/特效不在此帧内，避免把真值框喂给 YOLO。
        """
        import numpy as np
        if self.perception_surf is None:
            return None
        arr = pygame.surfarray.array3d(self.perception_surf)   # (w, h, 3) RGB
        arr = np.transpose(arr, (1, 0, 2))                     # (h, w, 3) RGB
        return arr[..., ::-1].copy()                           # → BGR
