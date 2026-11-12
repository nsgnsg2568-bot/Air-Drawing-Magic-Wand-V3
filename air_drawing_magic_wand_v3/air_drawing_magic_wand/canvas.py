"""Glowing ink + particles/trail effects."""
import math
import random

import cv2
import numpy as np

import config


class GlowCanvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ink = np.zeros((h, w, 3), np.uint8)    # coloured body of the strokes
        self.core = np.zeros((h, w, 3), np.uint8)   # white-hot centre line (separate so it is never painted over)
        self.fx = np.zeros((h, w, 3), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        vig = np.clip(1.15 - 0.45 * d ** 2, 0.45, 1.0)
        self.vig3 = np.ascontiguousarray(np.repeat((vig * 255).astype(np.uint8)[..., None], 3, axis=2))

    def clear(self):
        self.ink[:] = 0
        self.core[:] = 0

    @staticmethod
    def hot(color, k=0.65):
        return tuple(int(c + (255 - c) * k) for c in color)

    def dot(self, p, color):
        p = (int(p[0]), int(p[1]))
        r = config.BRUSH_THICKNESS // 2
        cv2.circle(self.ink, p, r, color, -1, cv2.LINE_AA)
        cv2.circle(self.core, p, max(1, r // 2), self.hot(color), -1, cv2.LINE_AA)

    def segment(self, p1, p2, color):
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        t = config.BRUSH_THICKNESS
        cv2.line(self.ink, a, b, color, t, cv2.LINE_AA)
        cv2.circle(self.ink, b, t // 2, color, -1, cv2.LINE_AA)           # round joint
        hc = self.hot(color)
        cv2.line(self.core, a, b, hc, max(2, t // 3), cv2.LINE_AA)

    def polyline(self, pts, color, closed=False):
        q = np.asarray(pts, np.float32).round().astype(np.int32).reshape(-1, 1, 2)
        t = config.BRUSH_THICKNESS
        cv2.polylines(self.ink, [q], closed, color, t, cv2.LINE_AA)
        cv2.polylines(self.core, [q], closed, self.hot(color), max(2, t // 3), cv2.LINE_AA)

    @staticmethod
    def _bezier(p0, c, p1):
        p0, c, p1 = (np.asarray(v, np.float32) for v in (p0, c, p1))
        length = np.linalg.norm(c - p0) + np.linalg.norm(p1 - c)
        n = int(min(28, max(2, length / 4)))
        t = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1

    def curve(self, p0, c, p1, color, fx=False, caps=False):
        """Smooth quadratic Bezier stroke piece from p0 to p1 with control point c."""
        layer = self.fx if fx else self.ink
        core = self.fx if fx else self.core
        pts = self._bezier(p0, c, p1)
        q = pts.round().astype(np.int32).reshape(-1, 1, 2)
        t = config.BRUSH_THICKNESS
        hc = self.hot(color)
        ends = [(int(e[0]), int(e[1])) for e in (q[0, 0], q[-1, 0])] if caps else []
        cv2.polylines(layer, [q], False, color, t, cv2.LINE_AA)
        for e in ends:
            cv2.circle(layer, e, t // 2, color, -1, cv2.LINE_AA)
        cv2.polylines(core, [q], False, hc, max(2, t // 3), cv2.LINE_AA)
        for e in ends:
            cv2.circle(core, e, max(1, t // 6) + 1, hc, -1, cv2.LINE_AA)

    def begin_fx(self):
        self.fx[:] = 0

    def compose(self, frame):
        base = cv2.multiply(frame, self.vig3, scale=config.CAMERA_DIM / 255.0)
        src = cv2.add(cv2.max(self.ink, self.core), self.fx)
        # bloom via a small image pyramid: wide glow is blurred at low resolution (cheap), then merged upward
        s4 = cv2.resize(src, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        s8 = cv2.resize(s4, (self.w // 8, self.h // 8), interpolation=cv2.INTER_AREA)
        s16 = cv2.resize(s8, (self.w // 16, self.h // 16), interpolation=cv2.INTER_AREA)
        g3 = cv2.GaussianBlur(s16, (0, 0), 3.5)
        g2 = cv2.GaussianBlur(s8, (0, 0), 3.0)
        g1 = cv2.GaussianBlur(s4, (0, 0), 2.0)
        glow8 = cv2.addWeighted(g2, 1.3, cv2.resize(g3, (g2.shape[1], g2.shape[0])), 1.5, 0)
        glow4 = cv2.addWeighted(g1, 0.9, cv2.resize(glow8, (g1.shape[1], g1.shape[0])), 1.0, 0)
        glow = cv2.resize(glow4, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = cv2.add(base, glow)
        return cv2.add(out, src)


class Particles:
    def __init__(self):
        self.p = []   # [x, y, vx, vy, life, max_life, color]

    def spawn(self, x, y, color, n=2):
        if not config.PARTICLES or len(self.p) > 260:
            return
        for _ in range(n):
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(25, 110)
            life = random.uniform(0.35, 0.9)
            self.p.append([x, y, math.cos(a) * s, math.sin(a) * s - 20, life, life, color])

    def update_draw(self, img, dt):
        keep = []
        for q in self.p:
            q[4] -= dt
            if q[4] <= 0:
                continue
            q[0] += q[2] * dt
            q[1] += q[3] * dt
            q[2] *= 0.96
            q[3] = q[3] * 0.96 + 18 * dt
            k = q[4] / q[5]
            col = tuple(int(c * k) for c in GlowCanvas.hot(q[6], 0.5))
            cv2.circle(img, (int(q[0]), int(q[1])), max(1, int(1 + 2.5 * k)), col, -1, cv2.LINE_AA)
            keep.append(q)
        self.p = keep
