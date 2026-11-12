"""Speed-adaptive Kalman filter for the fingertip, with latency-compensating prediction.

State = [x, y, vx, vy]. Process noise rises with speed:
  * finger (nearly) still  -> heavy smoothing, no wobble
  * finger moving fast     -> trusts the camera, almost no lag
predict() extrapolates the position forward by the pipeline latency (like Apple Pencil's
predicted touches), so the pen tip lands where your finger is *now*, not where it was.
"""
import math

import numpy as np

import config


class FingerSmoother:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.t = None

    @property
    def pos(self):
        return None if self.x is None else (float(self.x[0]), float(self.x[1]))

    @property
    def vel(self):
        return (0.0, 0.0) if self.x is None else (float(self.x[2]), float(self.x[3]))

    def update(self, z, t):
        z = np.asarray(z, np.float64)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([config.KF_MEAS_NOISE ** 2] * 2 + [1e5] * 2)
            self.t = t
            return self.pos
        dt = min(max(t - self.t, 1e-3), 0.2)
        self.t = t
        # a jump far beyond what velocity predicts (hand re-entered, tracker re-locked): snap, don't glide
        pred = self.x[:2] + self.x[2:] * dt
        if np.linalg.norm(z - pred) > config.KF_JUMP_PX:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([config.KF_MEAS_NOISE ** 2] * 2 + [1e5] * 2)
            return self.pos
        speed = math.hypot(self.x[2], self.x[3])
        w = 1.0 - math.exp(-(speed / config.KF_V0) ** 2)
        q = config.KF_Q_LOW + (config.KF_Q_HIGH - config.KF_Q_LOW) * w

        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        x = F @ self.x
        P = F @ self.P @ F.T + (G @ G.T) * q
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ P @ H.T + np.eye(2) * config.KF_MEAS_NOISE ** 2
        K = P @ H.T @ np.linalg.inv(S)
        self.x = x + K @ (z - H @ x)
        self.P = (np.eye(4) - K @ H) @ P
        return self.pos

    def predict(self, horizon):
        """Position `horizon` seconds ahead (capped so it can never fly off)."""
        if self.x is None:
            return None
        h = min(max(horizon, 0.0), config.PREDICT_MAX_MS / 1000.0) * config.PREDICT_GAIN
        dx, dy = self.x[2] * h, self.x[3] * h
        n = math.hypot(dx, dy)
        if n > config.PREDICT_MAX_PX:
            dx, dy = dx / n * config.PREDICT_MAX_PX, dy / n * config.PREDICT_MAX_PX
        return float(self.x[0] + dx), float(self.x[1] + dy)
