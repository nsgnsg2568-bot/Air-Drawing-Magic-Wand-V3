"""Pretty UI pieces: anti-aliased text, glass panels, result card. Rendered once, cached as sprites."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONTS, _SPRITES = {}, {}

_REG = ["SFNS.ttf", "/System/Library/Fonts/SFNS.ttf", "segoeui.ttf", "Inter-Regular.ttf",
        "Poppins-Regular.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"]
_BOLD = ["segoeuib.ttf", "Inter-Bold.ttf", "Poppins-Bold.ttf", "DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        f = None
        for name in (_BOLD if bold else _REG):
            try:
                f = ImageFont.truetype(name, size)
                break
            except Exception:
                continue
        if f is None:
            try:
                f = ImageFont.load_default(size=size)
            except Exception:
                f = ImageFont.load_default()
        _FONTS[key] = f
    return _FONTS[key]


def _to_bgra(pil_rgba):
    a = np.array(pil_rgba)
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


def text_sprite(msg, size=24, color=(255, 255, 255), bold=False, shadow=True):
    key = ("t", msg, size, color, bold, shadow)
    if key in _SPRITES:
        return _SPRITES[key]
    f = font(size, bold)
    l, t, r, b = f.getbbox(msg)
    pad = 6
    img = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if shadow:
        d.text((pad - l + 2, pad - t + 2), msg, font=f, fill=(0, 0, 0, 170))
    d.text((pad - l, pad - t), msg, font=f, fill=color + (255,))
    _SPRITES[key] = _to_bgra(img)
    return _SPRITES[key]


def panel(w, h, radius=28, fill=(16, 16, 26, 200), border=(255, 255, 255, 38)):
    key = ("p", w, h, radius, fill, border)
    if key in _SPRITES:
        return _SPRITES[key]
    s = 2  # supersample for smooth corners
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius * s, fill=fill, outline=border, width=2 * s)
    img = img.resize((w, h), Image.LANCZOS)
    _SPRITES[key] = _to_bgra(img)
    return _SPRITES[key]


_PRE = {}


def _premult(spr):
    """Cache (premultiplied colour, inverse alpha) so a blit is two fast OpenCV ops."""
    k = id(spr)
    if len(_PRE) > 80:
        _PRE.clear()
    if k not in _PRE:
        a = spr[..., 3:4].astype(np.uint16)
        pm = ((spr[..., :3].astype(np.uint16) * a) // 255).astype(np.uint8)
        inv = np.ascontiguousarray(np.repeat((255 - spr[..., 3:4]), 3, axis=2))
        _PRE[k] = (spr, np.ascontiguousarray(pm), inv)   # keep spr alive so the id stays valid
    return _PRE[k]


def blit(dst, spr, x, y, opacity=1.0):
    x, y = int(x), int(y)
    h, w = spr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    _, pm, inv = _premult(spr)
    sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    pm_r, inv_r = pm[sl], inv[sl]
    roi = dst[y0:y1, x0:x1]
    if opacity < 0.995:
        pm_r = cv2.convertScaleAbs(pm_r, alpha=opacity)
        inv_r = cv2.subtract(np.full_like(inv_r, 255), cv2.convertScaleAbs(cv2.subtract(np.full_like(inv_r, 255), inv_r), alpha=opacity))
    roi[:] = cv2.add(cv2.multiply(roi, inv_r, scale=1.0 / 255.0), pm_r)


def result_card(guesses, accent=(120, 230, 255)):
    """guesses: list of (label, prob). Returns a BGRA sprite."""
    from config import UNSURE_BELOW
    W, H = 720, 230
    s = 2
    img = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W * s - 1, H * s - 1), 40 * s, fill=(14, 14, 24, 215),
                        outline=(255, 255, 255, 40), width=2 * s)
    acc = accent[::-1]  # BGR -> RGB
    if not guesses:
        d.text((44 * s, 60 * s), "Hmm, I can't tell what that is", font=font(38 * s, True), fill=(255, 255, 255, 255))
        d.text((44 * s, 130 * s), "Try again: draw it bigger and simpler.", font=font(24 * s), fill=(190, 190, 200, 255))
    else:
        label, p = guesses[0]
        unsure = p < UNSURE_BELOW
        cap = "MAYBE A..." if unsure else "YOU DREW"
        d.text((44 * s, 28 * s), cap, font=font(20 * s, True), fill=acc + (255,))
        d.text((44 * s, 56 * s), label.title(), font=font(72 * s, True), fill=(255, 255, 255, 255))
        # confidence bar
        bx, by, bw, bh = 44, 160, W - 88, 12
        d.rounded_rectangle((bx * s, by * s, (bx + bw) * s, (by + bh) * s), 6 * s, fill=(255, 255, 255, 40))
        d.rounded_rectangle((bx * s, by * s, (bx + max(12, int(bw * p))) * s, (by + bh) * s), 6 * s, fill=acc + (255,))
        d.text((bx * s, 184 * s), f"{p:.0%} sure", font=font(20 * s, True), fill=(235, 235, 240, 255))
        if len(guesses) > 1:
            alt = "   ".join(f"{l.title()} {q:.0%}" for l, q in guesses[1:])
            f = font(20 * s)
            tw = d.textlength(alt, font=f)
            d.text(((W - 44) * s - tw, 184 * s), alt, font=f, fill=(170, 170, 185, 255))
    img = img.resize((W, H), Image.LANCZOS)
    return _to_bgra(img)


def help_card():
    lines = [("Index finger", "draw"), ("Two fingers", "pick a colour"),
             ("Open palm (hold)", "recognise now"), ("Fist (hold)", "clear"),
             ("Pause", "auto-recognise")]
    W, H = 400, 60 + 34 * len(lines)
    s = 2
    img = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W * s - 1, H * s - 1), 28 * s, fill=(14, 14, 24, 205),
                        outline=(255, 255, 255, 38), width=2 * s)
    d.text((28 * s, 18 * s), "HOW TO", font=font(16 * s, True), fill=(120, 230, 255, 255))
    for i, (a, b) in enumerate(lines):
        y = (50 + 34 * i) * s
        d.text((28 * s, y), a, font=font(20 * s, True), fill=(255, 255, 255, 255))
        d.text((190 * s, y), b, font=font(20 * s), fill=(185, 185, 200, 255))
    img = img.resize((W, H), Image.LANCZOS)
    return _to_bgra(img)


def palette_layout(w, n, r=26, gap=22, y=62):
    total = n * 2 * r + (n - 1) * gap
    x0 = (w - total) // 2 + r
    return [(x0 + i * (2 * r + gap), y) for i in range(n)], r
