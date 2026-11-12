"""Camera-free self-test + smoothness benchmark.

    python selftest.py

Feeds a fake, noisy, delayed hand through the whole pipeline (30 Hz tracker, 60 Hz display),
saves PNGs to ./selftest_output/ and prints how accurately the pen follows the finger.
"""
import math
import os
import time

import cv2
import numpy as np

import config
from app import WandApp
from gestures import DRAW, NONE, SELECT, HandState
from recognizer import Recognizer

W, H = 1280, 720
TRACK_HZ, DISPLAY_HZ = 30, 60
LATENCY = 0.045          # camera + tracker delay (s)
NOISE_PX = 2.5           # fingertip measurement noise (px)
rng = np.random.default_rng(1)


def background():
    y = np.linspace(40, 90, H, dtype=np.uint8)[:, None, None]
    return np.repeat(np.repeat(y, W, 1), 3, 2).copy()


def circle_truth(t, dur=3.2):
    a = 2 * math.pi * 1.04 * min(t / dur, 1)
    return (640 + 150 * math.cos(a), 330 + 150 * math.sin(a))


def triangle_truth(t, dur=3.0):
    v = [(640, 170), (800, 470), (480, 470), (640, 170)]
    s = min(t / dur, 0.9999) * 3
    i = int(s)
    f = s - i
    return (v[i][0] + (v[i + 1][0] - v[i][0]) * f, v[i][1] + (v[i + 1][1] - v[i][1]) * f)


def run():
    os.makedirs("selftest_output", exist_ok=True)
    app, bg = WandApp(W, H, Recognizer()), background()
    clock = 100.0
    render_ms = []
    tip_err, raw_err = [], []
    queue = []     # (deliver_time, capture_time, HandState)

    def advance(dur, truth, mode, tag=None, snap_at=None):
        nonlocal clock
        t_end = clock + dur
        next_meas, t0 = clock, clock
        saved = False
        while clock < t_end:
            while next_meas <= clock:                    # tracker produces a measurement
                p = truth(next_meas - t0) if truth else None
                hs = None
                if p is not None and mode != NONE:
                    noisy = (p[0] + rng.normal(0, NOISE_PX), p[1] + rng.normal(0, NOISE_PX))
                    hs = HandState(mode, noisy, (p[0], p[1] + 90))
                queue.append((next_meas + LATENCY, next_meas, hs))
                next_meas += 1.0 / TRACK_HZ
            while queue and queue[0][0] <= clock:        # ...delivered after the pipeline delay
                _, tc, hs = queue.pop(0)
                app.on_measure(hs, tc)
            tr = time.perf_counter()
            out = app.render(bg.copy(), clock)
            render_ms.append((time.perf_counter() - tr) * 1000)
            if mode == DRAW and truth is not None and clock - t0 > 0.6:
                tip = app._pen_tip(clock)
                if tip is not None:
                    gt = truth(clock - t0)
                    tip_err.append(math.dist(tip, gt))
                    if app.hs is not None and app.hs.tip is not None:
                        raw_err.append(math.dist(app.hs.tip, gt))    # what you'd see with no filtering at all
            if tag and not saved and clock - t0 > dur * 0.55:
                cv2.imwrite(f"selftest_output/{tag}_1_drawing.png", out)
                saved = True
            clock += 1.0 / DISPLAY_HZ
        return out

    px, py = app.centers[2]
    advance(0.4, lambda t: (px + 1, py), SELECT)
    for name, truth, dur in (("circle", circle_truth, 3.2), ("triangle", triangle_truth, 3.0)):
        app.reset()
        advance(dur, truth, DRAW, tag=name)
        advance(config.PAUSE_SECONDS + 0.4, None, NONE)
        out = advance(0.6, None, NONE)
        cv2.imwrite(f"selftest_output/{name}_2_result.png", out)
        print(f"{name:9s}->", app.guesses)

    print(f"\nPen-tip error vs real finger position (display at {DISPLAY_HZ} Hz, tracker {TRACK_HZ} Hz, "
          f"{LATENCY*1000:.0f} ms delay, {NOISE_PX} px noise):")
    print(f"  v3 pen (Kalman + prediction): {np.sqrt(np.mean(np.square(tip_err))):5.1f} px RMS")
    print(f"  raw tracker output:           {np.sqrt(np.mean(np.square(raw_err))):5.1f} px RMS")
    print(f"Render time: {np.mean(render_ms):.1f} ms avg, {np.percentile(render_ms, 95):.1f} ms p95 "
          f"-> can sustain ~{1000/np.mean(render_ms):.0f} fps on this PC")
    print("Images saved to ./selftest_output/")


if __name__ == "__main__":
    run()
