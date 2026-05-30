"""
Local monocular-depth free-space perception for the car (Depth Anything V2 small).

Replaces the brightness heuristic with a real ML depth estimate. Runs on Apple
Silicon GPU (MPS) via transformers/torch, preprocessing done manually (no
torchvision dependency).

API:
    load(model_id="models/da2-small", size=252) -> device str
    depth(pil_rgb) -> ndarray (H,W), LARGER = NEARER (Depth Anything convention)
    free_space(depth) -> {"left","center","right"} openness 0..1 (1 = most open)

Free-space signal = HOW MUCH THE FLOOR RECEDES into the distance:
  open floor -> disparity much larger (nearer) at the bottom of the look-ahead
               than the top (floor running away from camera) -> big gradient.
  wall/obstacle head-on -> vertical plane at ~constant distance -> gradient ~0.
This 'flatness vs recession' separates a wall from open floor far more reliably
than absolute nearness (a low camera makes everything look 'far' vs the floor
right under it). Scaled by the immediate-floor reference for frame stability.

Self-check (saves a grayscale depth viz; near = bright):
    .venv/bin/python tools/depth_perception.py <image.jpg>
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
from PIL import Image

_model = _device = None
_size = 252
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load(model_id="models/da2-small", size=252):
    global _model, _device, _size
    import torch
    from transformers import AutoModelForDepthEstimation
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _size = (size // 14) * 14   # ViT patch size = 14
    _model = AutoModelForDepthEstimation.from_pretrained(model_id).to(_device).eval()
    return _device


def depth(pil_rgb):
    import torch
    im = pil_rgb.convert("RGB").resize((_size, _size))
    a = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
    t = torch.from_numpy(a.transpose(2, 0, 1)[None]).to(_device)
    with torch.no_grad():
        d = _model(pixel_values=t).predicted_depth.squeeze().float().cpu().numpy()
    return d


def free_space(d, ref_rows=(0.82, 0.98), ref_cols=(0.33, 0.67),
               look_rows=(0.55, 0.78), pct_ahead=85, ratio_open=0.45, ratio_block=0.85):
    """Openness per zone via NEARNESS RATIO vs the immediate floor.

    ref = disparity of the floor right in front of the wheels (always ground at a
    known close distance). For each zone's look-ahead band we take how near its
    closest content is, and compare to ref:
        ratio = near_ahead / ref_floor
      ratio ~< 0.5  -> the view recedes past the floor = OPEN
      ratio ~>= 0.85 -> something as near as our own floor is right ahead = OBSTACLE
    A wall/furniture head-on pushes ratio toward/above 1; open floor stays low.
    This separates wall (ratio~1.0) from open floor (ratio~0.4-0.55) cleanly,
    unlike a gradient which collapses on both. Depth Anything: larger = nearer.
    """
    h, w = d.shape
    ref = d[int(h * ref_rows[0]):int(h * ref_rows[1]), int(w * ref_cols[0]):int(w * ref_cols[1])]
    ref_disp = float(np.percentile(ref, 90)) + 1e-6
    lo = d[int(h * look_rows[0]):int(h * look_rows[1]), :]
    out = []
    for a, b in ((0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)):
        near = float(np.percentile(lo[:, a:b], pct_ahead))      # nearness of closest stuff ahead
        ratio = near / ref_disp
        openness = float(np.clip((ratio_block - ratio) / (ratio_block - ratio_open), 0.0, 1.0))
        out.append(round(openness, 3))
    return {"left": out[0], "center": out[1], "right": out[2]}


def _viz(d, path):
    lo, hi = float(d.min()), float(d.max())
    g = ((d - lo) / (hi - lo + 1e-6) * 255).astype("uint8")   # near = bright
    Image.fromarray(g).save(path)


if __name__ == "__main__":
    import sys, time
    img = sys.argv[1] if len(sys.argv) > 1 else "open_floor.jpg"
    load()
    im = Image.open(img).convert("RGB")
    t = time.time(); d = depth(im); print(f"first {time.time()-t:.2f}s")
    ts = [(lambda: (time.time(), depth(im), time.time()))() for _ in range(5)]
    dt = sum(b - a for a, _, b in ts) / len(ts)
    print(f"steady {dt*1000:.0f} ms (~{1/dt:.1f} fps)")
    _viz(d, os.path.splitext(img)[0] + "_depth.png")
    print("free_space:", free_space(d))
