"""The wand's brain: gestures -> strokes -> glowing canvas -> recognition -> UI.

Two entry points, so tracking and display can run at their own speeds:
  on_measure(hand_state, capture_time)  - a new fingertip measurement arrived (~30 Hz)
  render(frame, now)                    - draw a screen frame (~60 Hz), pen tip predicted to "now"
No webcam needed, so selftest.py can drive it with a fake hand.
"""
import math
from collections import deque

import cv2
import numpy as np

import config
import ui
from canvas import GlowCanvas, Particles
from gestures import DRAW, FIST, IDLE, NONE, PALM, SELECT
from smoother import FingerSmoother
from utils import drawing_bbox

COLLECTING, SHOWING = "collecting", "showing"
STATUS = {DRAW: "Drawing", SELECT: "Pick a colour", PALM: "Open palm", FIST: "Fist", IDLE: "Hand ready", NONE: "Show your hand"}
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


class WandApp:
    def __init__(self, w, h, recognizer):
        self.w, self.h, self.rec = w, h, recognizer
        self.canvas, self.particles = GlowCanvas(w, h), Particles()
        self.smoother = FingerSmoother()
        self.centers, self.pr = ui.palette_layout(w, len(config.PALETTE))
        bar_w = self.centers[-1][0] - self.centers[0][0] + 2 * self.pr + 56
        self.bar = ui.panel(bar_w, 2 * self.pr + 36, 36)
        self.bar_x = self.centers[0][0] - self.pr - 28
        self.bar_y = self.centers[0][1] - self.pr - 18
        self.help = ui.help_card()
        self.color_idx = 0
        # stroke state
        self.strokes, self.current, self.last_pt, self.last_mid = [], None, None, None
        self.pending = []                       # points seen while a DRAW gesture was still being confirmed
        self.state, self.guesses, self.card, self.result_t = COLLECTING, None, None, 0.0
        # hand state
        self.mode, self.cand, self.cand_t0, self.cand_n = NONE, NONE, 0.0, 0
        self.visible, self.last_seen, self.meas_t = False, -1e9, -1e9
        self.pos, self.anchor, self.hs = None, None, None
        self.last_activity = 0.0
        self.hold_mode, self.hold_t0, self.hold_done, self.hold_p = None, 0.0, False, 0.0
        # effects / UI
        self.trail = deque(maxlen=40)
        self.flash = None
        self.toast, self.toast_t = "", -9.0
        self.hover = None
        self.start_t, self.last_render = None, None
        self.debug, self.show_help = False, True
        self.stats = {}

    # ------------------------------------------------------------------ state
    @property
    def color(self):
        return config.PALETTE[self.color_idx][1]

    def set_color(self, i, now):
        if i != self.color_idx:
            self.color_idx = i
            self.flash = (*self.centers[i], now, self.color)

    def _end_stroke(self):
        if self.current is not None and self.last_mid is not None and self.last_pt is not None:
            self.canvas.curve(self.last_mid, mid(self.last_mid, self.last_pt), self.last_pt, self.color, caps=True)
        self.current, self.last_pt, self.last_mid = None, None, None

    def reset(self):
        self.canvas.clear()
        self.strokes, self.current, self.last_pt, self.last_mid, self.pending = [], None, None, None, []
        self.state, self.guesses, self.card = COLLECTING, None, None

    def say(self, msg, now):
        self.toast, self.toast_t = msg, now

    def _recognize(self, now):
        self._end_stroke()
        mn, mx = drawing_bbox(self.strokes)
        if max(mx - mn) < config.MIN_DRAWING_SIZE_PX:
            self.reset()
            self.say("Draw a bit bigger", now)
            return
        guesses, snap = self.rec.predict(self.strokes)
        self.guesses, self.state, self.result_t = guesses, SHOWING, now
        self.card = ui.result_card(guesses, accent=self.color)
        print("[wand] result:", guesses)
        if snap is not None and config.SNAP_SHAPES:
            self._apply_snap(snap, now)

    def _apply_snap(self, geom, now):
        if geom[0] == "ellipse":
            _, c, (a, b), ang = geom
            poly = cv2.ellipse2Poly((int(c[0]), int(c[1])), (max(1, int(a)), max(1, int(b))), int(ang), 0, 360, 5)
            pts, closed = poly.astype(np.float32), True
        else:
            _, pts, closed = geom
        self.canvas.clear()
        self.canvas.polyline(pts, self.color, closed)
        c = np.mean(pts, axis=0)
        self.flash = (float(c[0]), float(c[1]), now, self.color)

    # ------------------------------------------------------------ measurements
    def _add_point(self, p, t):
        """Commit one smoothed fingertip point to the current stroke (Bezier through midpoints)."""
        if self.current is not None and self.last_pt is not None and math.dist(p, self.last_pt) > config.JUMP_BREAK_PX:
            self._end_stroke()                      # hand teleported: don't draw a streak across the screen
        if self.current is None:
            self.current, self.last_pt, self.last_mid = [], None, None
            self.strokes.append(self.current)
        if self.last_pt is not None and math.dist(p, self.last_pt) < config.MIN_STEP_PX:
            return
        if self.last_pt is None:
            self.canvas.dot(p, self.color)
            self.last_mid = p
        else:
            m = mid(self.last_pt, p)
            self.canvas.curve(self.last_mid, self.last_pt, m, self.color)
            self.last_mid = m
        self.current.append(p)
        self.last_pt = p
        self.last_activity = t
        self.particles.spawn(p[0], p[1], self.color)

    def on_measure(self, hs, t):
        """hs: HandState or None; t: capture time (seconds) of the frame it came from."""
        if self.start_t is None:
            self.start_t = t
        visible = hs is not None and hs.tip is not None
        raw = hs.mode if hs is not None else NONE
        self.hs, self.visible = hs, visible
        if visible:
            if t - self.last_seen > 0.3:
                self.smoother.reset()
            self.last_seen = self.meas_t = t
            self.pos = self.smoother.update(hs.tip, t)
            self.anchor = hs.anchor

        # gesture debounce (time based): a gesture must hold STABLE_MS and 2 measurements
        if raw == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_t0, self.cand_n = raw, t, 1
        prev_mode = self.mode
        if self.cand_n >= 2 and (t - self.cand_t0) * 1000 >= config.STABLE_MS:
            self.mode = self.cand

        # points seen during DRAW confirmation are kept, so the start of a stroke is not lost
        if visible and self.cand == DRAW and self.mode != DRAW:
            self.pending.append(self.pos)
            self.pending = self.pending[-5:]
        elif self.cand != DRAW:
            self.pending = []

        drawing = self.mode == DRAW and raw == DRAW and visible
        if drawing:
            if self.state == SHOWING:
                self.reset()
            if prev_mode != DRAW:
                for p in self.pending:
                    self._add_point(p, t)
                self.pending = []
            self._add_point(self.pos, t)
        elif self.mode == DRAW:
            # a one-frame blip or a brief tracker loss: keep the stroke alive
            if not visible and t - self.last_seen >= config.LOST_GRACE:
                self._end_stroke()
        elif self.current is not None and (visible or t - self.last_seen >= config.LOST_GRACE):
            self._end_stroke()

        # colour picking
        self.hover = None
        if self.mode == SELECT and visible and raw == SELECT:
            for i, (cx, cy) in enumerate(self.centers):
                if math.hypot(self.pos[0] - cx, self.pos[1] - cy) < self.pr + 16:
                    self.hover = i
                    self.set_color(i, t)
                    self.last_activity = t
                    break

        # hold gestures: open palm = recognise now, fist = clear
        hm = self.mode if (self.mode in (PALM, FIST) and visible) else None
        if hm != self.hold_mode:
            self.hold_mode, self.hold_t0, self.hold_done = hm, t, False
        self.hold_p = 0.0
        if hm and not self.hold_done:
            self.hold_p = (t - self.hold_t0) / (config.PALM_HOLD if hm == PALM else config.FIST_HOLD)
            if self.hold_p >= 1:
                self.hold_done, self.hold_p = True, 0.0
                if hm == PALM and self.state == COLLECTING and self.strokes:
                    self._recognize(t)
                elif hm == FIST:
                    self.reset()

    # convenience for tests / simple callers
    def update(self, frame, hs, now):
        self.on_measure(hs, now)
        return self.render(frame, now)

    # ----------------------------------------------------------------- render
    def _pen_tip(self, now):
        """Where the fingertip is *right now* (measurement extrapolated by the pipeline latency)."""
        if not self.visible or now - self.meas_t > 0.25 or self.smoother.pos is None:
            return None
        return self.smoother.predict(now - self.meas_t)

    def render(self, frame, now):
        if self.start_t is None:
            self.start_t = now
        dt = 1 / 60 if self.last_render is None else min(max(now - self.last_render, 1e-3), 0.1)
        self.last_render = now

        # stopped for a while -> read the drawing
        if self.state == COLLECTING and self.strokes and now - self.last_activity > config.PAUSE_SECONDS:
            self._recognize(now)

        cv, col = self.canvas, self.color
        tip = self._pen_tip(now)
        cv.begin_fx()

        # predicted tail: ink from the last committed point up to where the finger is now (not permanent)
        if (tip is not None and self.mode == DRAW and self.current is not None and self.last_mid is not None
                and self.hs is not None and self.hs.mode == DRAW):
            cv.curve(self.last_mid, self.last_pt, tip, col, fx=True, caps=True)

        if tip is not None:
            self.trail.append((tip[0], tip[1], now))
        for x, y, t in self.trail:
            a = 1 - (now - t) / 0.28
            if a > 0:
                c = tuple(int(v * a) for v in (col if self.mode in (DRAW, SELECT) else (200, 200, 200)))
                cv2.circle(cv.fx, (int(x), int(y)), int(2 + 7 * a), c, -1, cv2.LINE_AA)
        self.particles.update_draw(cv.fx, dt)
        if self.flash:
            fx, fy, t0, fc = self.flash
            p = (now - t0) / 0.6
            if p >= 1:
                self.flash = None
            else:
                cv2.circle(cv.fx, (int(fx), int(fy)), int(20 + 90 * ease_out(p)),
                           tuple(int(v * (1 - p)) for v in fc), 3, cv2.LINE_AA)

        out = cv.compose(frame)

        # palette
        ui.blit(out, self.bar, self.bar_x, self.bar_y)
        for i, ((cx, cy), (_, c)) in enumerate(zip(self.centers, config.PALETTE)):
            r = self.pr + (6 if i == self.hover else 0)
            cv2.circle(out, (cx, cy), r, c, -1, cv2.LINE_AA)
            if i == self.color_idx:
                cv2.circle(out, (cx, cy), r + 6, (255, 255, 255), 3, cv2.LINE_AA)

        if self.debug and self.hs is not None and self.hs.points is not None:
            for a, b in SKELETON:
                cv2.line(out, tuple(int(v) for v in self.hs.points[a]), tuple(int(v) for v in self.hs.points[b]),
                         (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out, tuple(int(v) for v in self.hs.tip), 5, (0, 0, 255), -1, cv2.LINE_AA)  # raw tip

        # cursor (at the predicted pen tip)
        if tip is not None:
            p = (int(tip[0]), int(tip[1]))
            if self.mode == DRAW:
                cv2.circle(out, p, 9, col, -1, cv2.LINE_AA)
                cv2.circle(out, p, 17, (255, 255, 255), 2, cv2.LINE_AA)
            elif self.mode == SELECT:
                cv2.circle(out, p, 22, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(out, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(out, p, 9, (170, 170, 170), 2, cv2.LINE_AA)

        if self.hold_p > 0 and self.anchor is not None and tip is not None:
            c = (int(self.anchor[0]), int(self.anchor[1]))
            cv2.ellipse(out, c, (56, 56), -90, 0, 360, (90, 90, 90), 4, cv2.LINE_AA)
            cv2.ellipse(out, c, (56, 56), -90, 0, int(360 * min(self.hold_p, 1)), (255, 255, 255), 5, cv2.LINE_AA)
            s = ui.text_sprite("Recognise" if self.hold_mode == PALM else "Clear", 22, bold=True)
            ui.blit(out, s, c[0] - s.shape[1] // 2, c[1] - 100)

        if self.state == SHOWING and self.card is not None:
            p = ease_out((now - self.result_t) / 0.45)
            ch, cw = self.card.shape[:2]
            ui.blit(out, self.card, (self.w - cw) // 2, self.h - ch - 30 + (1 - p) * 70, p)

        if self.state == COLLECTING:
            s = ui.text_sprite(STATUS[self.mode], 24, bold=True)
            ui.blit(out, s, 24, self.h - 56)
        engine = f"AI: CNN, {len(self.rec.classes)} objects" if self.rec.has_cnn else "AI: shapes only  (run train.py)"
        s = ui.text_sprite(engine, 18, (170, 255, 190) if self.rec.has_cnn else (255, 210, 140))
        ui.blit(out, s, self.w - s.shape[1] - 20, 16)
        if now - self.toast_t < 1.6:
            s = ui.text_sprite(self.toast, 30, bold=True)
            ui.blit(out, s, (self.w - s.shape[1]) // 2, 150, 1 - max(0, (now - self.toast_t - 1.0) / 0.6))
        if self.show_help and not self.strokes and self.state == COLLECTING:
            ui.blit(out, self.help, 24, 150)
        if self.debug:
            st = self.stats
            lat = (now - self.meas_t) * 1000 if self.meas_t > -1e8 else 0
            txt = (f"display {st.get('display', 0):.0f} fps | camera {st.get('camera', 0):.0f} fps | "
                   f"tracker {st.get('tracker', 0):.0f} fps | latency {lat:.0f} ms | mode {self.mode}")
            cv2.putText(out, txt, (24, self.h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return out
