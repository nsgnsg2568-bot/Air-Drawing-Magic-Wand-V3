"""Combines the CNN (if trained) with the classic shape recogniser."""
import os

import numpy as np

import config
from shapes import classify_shape
from utils import render_strokes


class Recognizer:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model, self.classes, self.torch = None, [], None
        if os.path.exists(model_path):
            try:
                import torch
                from model import QuickDrawCNN
                ckpt = torch.load(model_path, map_location="cpu")
                self.classes = ckpt["classes"]
                self.model = QuickDrawCNN(len(self.classes))
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.eval()
                self.torch = torch
                print(f"[wand] CNN loaded with {len(self.classes)} classes.")
            except Exception as e:
                print(f"[wand] Could not load CNN ({e}). Using shape recogniser only.")
        else:
            print("[wand] No trained model found (run `python train.py`). Using the basic shape recogniser.")

    @property
    def has_cnn(self):
        return self.model is not None

    def _cnn(self, strokes, k=3):
        """Test-time augmentation: average predictions over tiny shifts / thicker strokes."""
        t = self.torch
        img = t.from_numpy(render_strokes(strokes, 28)).view(1, 1, 28, 28)
        views = [img, t.nn.functional.max_pool2d(img, 3, 1, 1)]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            views.append(t.roll(img, shifts=(dy, dx), dims=(2, 3)))
        with t.no_grad():
            probs = t.softmax(self.model(t.cat(views)), dim=1).mean(0).numpy()
        top = np.argsort(probs)[::-1][:k]
        return [(self.classes[i], float(probs[i])) for i in top]

    def predict(self, strokes):
        """Returns (guesses, snap): guesses = [(label, prob)] best first; snap = idealised geometry or None."""
        cnn = self._cnn(strokes) if self.has_cnn else []
        shape = classify_shape(strokes)
        merged = cnn
        if shape and (not cnn or cnn[0][1] < config.CNN_TRUST_THRESHOLD):
            merged = [(shape[0], shape[1])] + [c for c in cnn if c[0] != shape[0]]
        merged = merged[:3]
        snap = shape[2] if (shape and merged and merged[0][0] == shape[0]) else None
        return merged, snap
