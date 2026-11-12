"""Shared helpers: turn air-drawn strokes into the 28x28 image the CNN expects."""
import cv2
import numpy as np

CANVAS = 256   # intermediate render size
MARGIN = 6
THICKNESS = 16  # ~1.75px after shrinking to 28x28, close to QuickDraw's look


def render_strokes(strokes, size=28):
    """strokes: list of lists of (x, y). Returns float32 array (size, size) in [0, 1].

    Mirrors how Google QuickDraw bitmaps are made: keep aspect ratio, scale the
    drawing to fill the frame, centre it, white strokes on black.
    """
    pts = [np.asarray(s, dtype=np.float32) for s in strokes if len(s) > 0]
    if not pts:
        return np.zeros((size, size), np.float32)
    allp = np.vstack(pts)
    mn, mx = allp.min(0), allp.max(0)
    span = max(float((mx - mn).max()), 1.0)
    scale = (CANVAS - 2 * MARGIN) / span
    offset = (CANVAS - (mx - mn) * scale) / 2.0

    img = np.zeros((CANVAS, CANVAS), np.uint8)
    for p in pts:
        q = ((p - mn) * scale + offset).astype(np.int32)
        if len(q) == 1:
            cv2.circle(img, tuple(int(v) for v in q[0]), THICKNESS // 2, 255, -1)
        else:
            cv2.polylines(img, [q.reshape(-1, 1, 2)], False, 255, THICKNESS, cv2.LINE_AA)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def drawing_bbox(strokes):
    allp = np.vstack([np.asarray(s, np.float32) for s in strokes if len(s) > 0])
    mn, mx = allp.min(0), allp.max(0)
    return mn, mx
