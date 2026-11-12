# Air-Drawing Magic Wand v3

Point your index finger at the webcam, draw glowing neon lines in the air, and the
wand tells you what you drew. Shapes snap to perfect circles, triangles and squares.

## Setup (Python 3.9 - 3.12)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python diagnose.py                # checks libraries + camera, tells you what's wrong
python selftest.py                # camera-free test, saves images to selftest_output/
python train.py --preset fast     # ML brain (needs internet once); use `python train.py` for 75 objects
python main.py
```
Without `train.py` the app still works with the built-in shape recogniser
(line, circle, oval, triangle, square, rectangle, pentagon, hexagon, star).
Recognising objects (cat, house, tree...) needs the trained CNN.

## Gestures

| Gesture | Action |
|---|---|
| Index finger only | Draw |
| Index + middle finger | Hover a colour bubble to pick it |
| Pause ~1.5 s | Wand reads your drawing |
| Open palm (hold 0.6 s) | Recognise immediately |
| Fist (hold 0.9 s) | Clear |
| Keys | 1-6 colour, C clear, S save, D debug view, H help, F fullscreen, Q quit |

Tips: good front lighting, hand about an arm's length away, palm facing the camera,
draw big and simple.

## What's new in v3 (smoothness overhaul)

- **Speed-adaptive Kalman filter** (`smoother.py`): rock steady when your finger is still, almost no lag when you move fast
- **Latency-compensated pen tip**: like Apple Pencil's predicted touches, the live pen tip is drawn where your finger is *now*; committed ink stays exact
- **Decoupled pipeline**: camera thread + tracker thread + a 60 fps display loop, so nothing waits for anything
- **Smooth Bezier strokes** instead of straight segments, with a continuous white-hot neon core
- Stroke starts are never lost (points seen while the gesture was being confirmed are replayed)
- Jump detection: a hand re-entering the frame never draws a streak across the screen
- 3x faster renderer (about 6 ms per frame), camera asked for 60 fps
- `selftest.py` now benchmarks pen accuracy (about 8 px vs 19 px for raw tracking in the simulation)

## Tuning (config.py)

- **`SMOOTHNESS`** is the main knob: 0 = snappiest, 1 = silkiest. Default 0.4. Lines wobble -> raise it. Pen lags -> lower it.
- `PREDICT_GAIN`: 1 = pen tip lands on the finger, 0 = no prediction, 1.3 = slightly leads
- `HAND_MODEL_COMPLEXITY = 0` and a lower `PROC_WIDTH` for slow PCs
- `PAUSE_SECONDS`, `SNAP_SHAPES`, `PARTICLES`, `CAMERA_INDEX`, `CAMERA_FPS`

Press **D** in the app for the debug view: display / camera / tracker fps, latency in ms and your hand skeleton
(red dot = raw fingertip). Good targets: display 55-60, tracker 25+, latency under 80 ms.
For the best feel use bright, even front lighting: webcams drop to 15 fps in dim rooms, which no filter can fix.

## How it works

1. **Tracking** (`hand_tracker.py`): MediaPipe gives 21 landmarks; a finger is "up" if its
   joint is straight (angle between knuckle->middle joint and middle joint->tip) and the tip
   is far from the wrist.
2. **Smoothing** (`smoother.py`): Kalman filter whose trust in the camera rises with speed, plus prediction.
3. **Rendering** (`canvas.py`, `ui.py`): ink layer blurred at 3 scales for bloom, added over a
   dimmed camera feed.
4. **Recognition** (`utils.py`, `model.py`, `recognizer.py`, `shapes.py`): strokes are re-rendered
   as a QuickDraw-style 28x28 image for the CNN; contour geometry handles clean shapes.

## Troubleshooting

- Those `oneDNN` / `absl` lines are harmless TensorFlow-style notices; v2 silences them.
- No camera: change `CAMERA_INDEX`, allow camera permission, close Zoom/Teams.
- Press `D` in the app to see the hand skeleton, fps and the detected gesture.
- `mediapipe` install fails: use Python 3.9-3.12.

## Ideas to level up your ML

Stroke-sequence model (LSTM/Transformer) on QuickDraw raw data, fine-tune on your own
air-drawn samples, confusion matrix, ONNX export, digit/letter recognition (EMNIST).
