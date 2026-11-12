"""MediaPipe hand tracking + robust, rotation-independent gesture detection."""
import cv2
import numpy as np

import config

try:
    import mediapipe as mp
    _ = mp.solutions.hands
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "MediaPipe's hand solution is not available (%s).\n"
        "Fix: pip install \"mediapipe>=0.10.9,<0.10.22\" on Python 3.9-3.12." % e)

from gestures import DRAW, FIST, IDLE, PALM, SELECT, HandState

# (mcp, pip, tip) for index, middle, ring, pinky
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.ext = [False] * 4   # per-finger state with hysteresis (no flicker)

    def _update_fingers(self, p):
        wrist = p[0]
        for i, (mcp, pip, tip) in enumerate(FINGERS):
            v1, v2 = p[pip] - p[mcp], p[tip] - p[pip]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
            far = float(np.linalg.norm(p[tip] - wrist) / (np.linalg.norm(p[pip] - wrist) + 1e-6))
            if self.ext[i]:
                self.ext[i] = not (cos < 0.35 or far < 0.95)
            else:
                self.ext[i] = cos > 0.70 and far > 1.05

    def process(self, small_bgr, out_w, out_h):
        """small_bgr: downscaled mirrored frame. Returns HandState (pixels in out_w x out_h) or None."""
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.ext = [False] * 4
            return None
        lm = res.multi_hand_landmarks[0].landmark
        p = np.array([[l.x * out_w, l.y * out_h, l.z * out_w] for l in lm], np.float32)
        self._update_fingers(p)
        idx, mid, ring, pinky = self.ext
        if idx and not mid and not ring and not pinky:
            mode = DRAW
        elif idx and mid and not ring and not pinky:
            mode = SELECT
        elif idx and mid and ring and pinky:
            mode = PALM
        elif not (idx or mid or ring or pinky):
            mode = FIST
        else:
            mode = IDLE
        return HandState(mode=mode, tip=(float(p[8, 0]), float(p[8, 1])),
                         anchor=(float(p[9, 0]), float(p[9, 1])), points=p[:, :2].copy())

    def close(self):
        self.hands.close()
