"""Air-Drawing Magic Wand.

  index finger        draw (glowing neon ink)
  two fingers         hover a colour to pick it   (or keys 1-6)
  pause ~1.5 s        the wand reads your drawing  (or hold an open palm)
  fist (hold)         clear                         (or press C)
Keys: C clear | S save screenshot | D debug | H help | F fullscreen | Q/ESC quit

Three things run in parallel so nothing waits for anything else:
  camera thread  -> grabs frames          tracker thread -> finds your fingertip (MediaPipe)
  main thread    -> draws the screen at ~60 fps, pen tip predicted to "now"
"""
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # fixes slow/failed webcam start on Windows

import sys
import threading
import time
from time import perf_counter

import cv2

import config
from app import WandApp
from hand_tracker import HandTracker
from recognizer import Recognizer

WINDOW = "Air-Drawing Magic Wand"


class Rate:
    """Smoothed events-per-second counter."""

    def __init__(self):
        self.t, self.v = perf_counter(), 0.0

    def tick(self):
        n = perf_counter()
        self.v = 0.9 * self.v + 0.1 / max(n - self.t, 1e-4)
        self.t = n


class Camera:
    """Grabs mirrored frames on a background thread and timestamps each one."""

    def __init__(self, index):
        self.cap = self._open(index)
        if self.cap is None:
            raise SystemExit("Could not get frames from any webcam (tried indexes 0-3).\n"
                             "Windows: Settings > Privacy > Camera > allow apps (incl. desktop apps), close Zoom/Teams/"
                             "Camera app, unplug/replug an external webcam, then retry. Run `python diagnose.py`.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock, self.frame, self.t, self.seq, self.stop = threading.Lock(), None, 0.0, 0, False
        self.rate = Rate()
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _open(preferred):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for idx in [preferred] + [i for i in range(4) if i != preferred]:
            for b in backends:
                cap = cv2.VideoCapture(idx, b)
                if not cap.isOpened():
                    cap.release()
                    continue
                t0 = perf_counter()
                while perf_counter() - t0 < 2.5:
                    ok, f = cap.read()
                    if ok and f is not None:
                        print(f"[wand] camera {idx} opened (backend {b})")
                        return cap
                cap.release()
        return None

    def _loop(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            t = perf_counter()
            f = cv2.flip(f, 1)          # mirror: drawings look right on screen
            with self.lock:
                self.frame, self.t, self.seq = f, t, self.seq + 1
            self.rate.tick()

    def get(self):
        with self.lock:
            return self.frame, self.t, self.seq

    def release(self):
        self.stop = True
        time.sleep(0.05)
        self.cap.release()


class TrackerThread(threading.Thread):
    """Runs MediaPipe on the newest frame as fast as it can, independent of the display."""

    def __init__(self, cam, tracker, w, h):
        super().__init__(daemon=True)
        self.cam, self.tracker, self.w, self.h = cam, tracker, w, h
        self.pw = config.PROC_WIDTH
        self.ph = int(h * self.pw / w)
        self.lock, self.result, self.n, self.stop = threading.Lock(), (0, None, 0.0), 0, False
        self.rate = Rate()

    def run(self):
        last = -1
        while not self.stop:
            frame, t, seq = self.cam.get()
            if frame is None or seq == last:
                time.sleep(0.002)
                continue
            last = seq
            small = cv2.resize(frame, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
            hs = self.tracker.process(small, self.w, self.h)
            with self.lock:
                self.n += 1
                self.result = (self.n, hs, t)
            self.rate.tick()

    def get(self):
        with self.lock:
            return self.result


def main():
    cam = Camera(config.CAMERA_INDEX)
    t0 = perf_counter()
    while cam.get()[0] is None:
        if perf_counter() - t0 > 8:
            raise SystemExit("The webcam opened but sent no frames. Close other apps using it and retry.")
        time.sleep(0.01)
    h, w = cam.get()[0].shape[:2]
    tracker, recognizer = HandTracker(), Recognizer()
    trk = TrackerThread(cam, tracker, w, h)
    trk.start()
    app = WandApp(w, h, recognizer)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w, 1280), min(h, 720))
    os.makedirs("captures", exist_ok=True)
    fullscreen, last_mseq, disp = False, 0, Rate()
    frame_budget = 1.0 / config.DISPLAY_FPS

    while True:
        t_start = perf_counter()
        frame, _, _ = cam.get()
        mseq, hs, t_cap = trk.get()
        if mseq != last_mseq:
            last_mseq = mseq
            app.on_measure(hs, t_cap)
        app.stats = {"display": disp.v, "camera": cam.rate.v, "tracker": trk.rate.v}
        out = app.render(frame, perf_counter())
        cv2.imshow(WINDOW, out)
        disp.tick()

        wait = max(1, int((frame_budget - (perf_counter() - t_start)) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            app.reset()
        elif key == ord("d"):
            app.debug = not app.debug
        elif key == ord("h"):
            app.show_help = not app.show_help
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("s"):
            path = os.path.join("captures", f"wand_{int(time.time())}.png")
            cv2.imwrite(path, out)
            print("[wand] saved", path)
        elif ord("1") <= key <= ord("0") + len(config.PALETTE):
            app.set_color(key - ord("1"), perf_counter())

    trk.stop = True
    trk.join(timeout=1.0)
    tracker.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
