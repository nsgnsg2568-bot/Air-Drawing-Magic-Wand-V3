"""All the knobs in one place. Tweak these to change how the wand feels."""

# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = 0            # try 1 if you have an external webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 60             # asked from the camera; it will use the best it supports (good light helps)
PROC_WIDTH = 640            # hand tracking runs on a downscaled frame (faster, same accuracy)
HAND_MODEL_COMPLEXITY = 1   # 1 = more accurate, 0 = faster (try 0 on a slow PC)
DISPLAY_FPS = 60            # screen refresh target; pen tip + cursor are redrawn at this rate

# --- Pen feel ---------------------------------------------------------------
BRUSH_THICKNESS = 10        # px
MIN_STEP_PX = 1.5             # ignore finger movement smaller than this
# Speed-adaptive Kalman filter (see smoother.py)
SMOOTHNESS = 1            # THE main knob, 0..1: 0 = snappiest (more wobble), 1 = silkiest (more lag)
KF_MEAS_NOISE = 2.5         # px, how noisy the raw fingertip is. Jittery lines -> raise it
KF_Q_LOW = 1500.0           # smoothing when the finger is still (lower = steadier)
KF_Q_HIGH = 3.0e6 * (0.03 ** SMOOTHNESS)   # responsiveness when moving fast, derived from SMOOTHNESS
KF_JUMP_PX = 120.0          # a sudden jump bigger than this snaps the pen instead of gliding
KF_V0 = 150.0               # px/s, speed at which it switches from "steady" to "responsive"
# Apple-Pencil-style prediction: the pen tip is drawn where your finger is *now*
PREDICT_GAIN = 1.0          # 0 = off, 1 = full latency compensation, >1 = lead the finger
PREDICT_MAX_MS = 90         # never extrapolate further than this
PREDICT_MAX_PX = 90
JUMP_BREAK_PX = 160         # a jump this big in one step starts a new stroke
STABLE_MS = 70              # a gesture must hold this long to count (stroke starts are replayed)
LOST_GRACE = 0.30           # seconds the hand may vanish without breaking the stroke

# --- Gestures / timing ------------------------------------------------------
PAUSE_SECONDS = 1.5         # stop moving this long -> the wand reads your drawing
PALM_HOLD = 0.6             # open palm held this long -> recognise immediately
FIST_HOLD = 0.9             # fist held this long -> clear the canvas
MIN_DRAWING_SIZE_PX = 60    # drawings smaller than this are ignored

# --- Look -------------------------------------------------------------------
PALETTE = [
    ("Cyan",    (255, 255, 0)),
    ("Magenta", (255, 0, 255)),
    ("Lime",    (60, 255, 80)),
    ("Yellow",  (0, 235, 255)),
    ("Orange",  (0, 140, 255)),
    ("Pink",    (140, 90, 255)),
]
CAMERA_DIM = 0.50           # how dark the camera feed is (lower = glow pops more)
SNAP_SHAPES = True          # tidy up circles, triangles, squares, lines like Apple Notes does
PARTICLES = True

# --- Recognition ------------------------------------------------------------
MODEL_PATH = "models/quickdraw_cnn.pt"
CNN_TRUST_THRESHOLD = 0.60  # below this, a clean geometric shape can override the CNN
UNSURE_BELOW = 0.45         # below this the card says "Maybe a ..."
