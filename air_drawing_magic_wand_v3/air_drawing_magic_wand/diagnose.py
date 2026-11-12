"""Checks your setup and tells you exactly what's wrong.   python diagnose.py"""
import importlib
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
import platform
import sys
import time

print("Python", sys.version.split()[0], "|", platform.platform())
ok = True
for mod in ("cv2", "numpy", "PIL", "mediapipe", "torch"):
    try:
        m = importlib.import_module(mod)
        print(f"  [ok]   {mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        tag = "[warn]" if mod == "torch" else "[FAIL]"
        ok &= mod == "torch"
        print(f"  {tag} {mod}: {e}")
try:
    import mediapipe as mp
    mp.solutions.hands
    print("  [ok]   mediapipe hands solution available")
except Exception as e:
    ok = False
    print("  [FAIL] mediapipe hands solution missing:", e,
          '\n         fix: pip install "mediapipe>=0.10.9,<0.10.22" (Python 3.9-3.12)')
print("  [ok]   trained model found" if os.path.exists("models/quickdraw_cnn.pt")
      else "  [info] no trained model yet -> run: python train.py   (shapes-only mode works without it)")

import cv2
print("Testing cameras 0-3 (DirectShow first on Windows) ...")
found = False
backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)] if sys.platform == "win32" else [("ANY", cv2.CAP_ANY)]
for i in range(4):
    for name, b in backends:
        cap = cv2.VideoCapture(i, b)
        if not cap.isOpened():
            cap.release()
            continue
        t0, n, f = time.time(), 0, None
        while time.time() - t0 < 2.0:
            good, f = cap.read()
            n += bool(good)
        cap.release()
        print(f"  camera {i} via {name}: {n/2.0:.0f} fps" + (f", {f.shape[1]}x{f.shape[0]}" if n and f is not None else ""))
        if n:
            found = True
            break
if not found:
    ok = False
    print("  [FAIL] no camera delivers frames. Windows: Settings > Privacy > Camera > allow apps (and desktop apps); "
          "close Zoom/Teams/Camera app; unplug/replug an external webcam.")
print("\nAll good - run: python main.py" if ok else "\nFix the [FAIL] items above, then run again.")
