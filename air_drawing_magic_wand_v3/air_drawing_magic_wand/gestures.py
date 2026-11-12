"""Plain data types shared by the tracker and the app (no heavy imports)."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DRAW, SELECT, PALM, FIST, IDLE, NONE = "DRAW", "SELECT", "PALM", "FIST", "IDLE", "NONE"


@dataclass
class HandState:
    mode: str
    tip: Optional[Tuple[float, float]] = None      # index fingertip, full-res pixels
    anchor: Optional[Tuple[float, float]] = None   # palm centre, full-res pixels
    points: Optional[np.ndarray] = None            # (21, 2) landmarks, full-res pixels
