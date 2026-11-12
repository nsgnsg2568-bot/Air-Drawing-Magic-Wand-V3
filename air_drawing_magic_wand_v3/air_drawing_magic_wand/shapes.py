"""Classic computer-vision shape recogniser (no training needed).

Returns (label, confidence, geometry) where geometry is an idealised version of the
shape, used for Apple-Notes-style "snapping". Works as a fallback when no CNN is
trained and as a sanity check for clean geometric shapes.

geometry is one of:
  ("ellipse", (cx, cy), (semi_a, semi_b), angle_deg)
  ("poly", Nx2 float array, closed_bool)
  None
"""
import math

import cv2
import numpy as np


def _resample(pts, n=120):
    pts = np.asarray(pts, np.float32)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return pts
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])].astype(np.float32)


def _smooth(pts, k=5):
    if len(pts) < k + 2:
        return pts
    pad = k // 2
    p = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.c_[np.convolve(p[:, 0], ker, "valid"), np.convolve(p[:, 1], ker, "valid")].astype(np.float32)


def classify_shape(strokes):
    strokes = [s for s in strokes if len(s) > 3]
    if len(strokes) != 1:
        return None
    raw = np.asarray(strokes[0], np.float32)
    length = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
    if length < 20:
        return None
    chord = float(np.linalg.norm(raw[-1] - raw[0]))
    pts = _smooth(_resample(raw))
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    diag = math.hypot(w, h)

    if chord / length > 0.93:   # straight line
        return "line", 0.85, ("poly", np.array([raw[0], raw[-1]], np.float32), False)

    if chord < 0.28 * length and chord < 0.2 * diag:   # closed shape
        contour = pts.reshape(-1, 1, 2)
        peri = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if area < 1:
            return None
        hull = cv2.convexHull(contour)
        solidity = area / max(cv2.contourArea(hull), 1.0)
        circularity = 4 * math.pi * area / (peri ** 2)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        n = len(approx)
        rect = cv2.minAreaRect(contour)
        (rcx, rcy), (rw, rh), rang = rect
        aspect = min(rw, rh) / max(rw, rh, 1e-6)
        poly = approx.reshape(-1, 2).astype(np.float32)

        if n >= 8 and solidity < 0.72:
            return "star", 0.7, None
        if circularity > 0.80 and n >= 6:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(contour)
            if aspect > 0.8:
                r = (MA + ma) / 4.0
                return "circle", 0.85, ("ellipse", (cx, cy), (r, r), 0.0)
            return "oval", 0.75, ("ellipse", (cx, cy), (MA / 2.0, ma / 2.0), ang)
        if n == 3:
            return "triangle", 0.8, ("poly", poly, True)
        if n == 4:
            if aspect > 0.8:
                side = (rw + rh) / 2.0
                box = cv2.boxPoints(((rcx, rcy), (side, side), rang))
                return "square", 0.8, ("poly", box.astype(np.float32), True)
            box = cv2.boxPoints(rect)
            return "rectangle", 0.8, ("poly", box.astype(np.float32), True)
        if n == 5 and solidity > 0.9:
            return "pentagon", 0.65, ("poly", poly, True)
        if n == 6 and solidity > 0.9:
            return "hexagon", 0.6, ("poly", poly, True)
        return None

    if diag > 0 and length / diag > 2.2 and chord / length < 0.6:
        return "squiggle", 0.4, None
    return None
