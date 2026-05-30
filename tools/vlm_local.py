"""
Local VLM brain for seek.py — Qwen2.5-VL-7B (4-bit) via MLX on Apple GPU.

Drop-in alternative to the cloud Gemini call in seek.py: same return dict
(target_visible / target_bearing / target_size / target_motion / turn_strength /
search_strategy / safe_forward / arrived / reason). No network, no per-call cost.

Local 4-bit models don't enforce a JSON schema like Gemini structured output, so
we (1) state the exact JSON shape in the prompt and (2) parse tolerantly.

API:
    load_local(model_dir="models/qwen2.5-vl-7b-4bit") -> None
    ask_seek_local(target, frames, hist_lines) -> dict   # frames: [(t, jpg_bytes), ...]
"""
import os, json, re, time, tempfile
from io import BytesIO
from PIL import Image

_model = _proc = _config = None
_DEF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "qwen2.5-vl-7b-4bit")

# valid enum values, used to clamp whatever the model returns
_BEARING = {"far_left", "left", "center", "right", "far_right", "none"}
_SIZE = {"none", "small", "medium", "large"}
_MOTION = {"approaching", "receding", "steady", "lost", "unknown"}
_STRENGTH = {"none", "slight", "medium", "hard"}
_STRAT = {"spin_left", "spin_right", "forward", "turn_around", "stop"}

_DEFAULT = {"target_visible": False, "target_bearing": "none", "target_size": "none",
            "target_motion": "unknown", "turn_strength": "medium",
            "search_strategy": "spin_right", "safe_forward": False, "reason": "parse-failed", "arrived": False}


def load_local(model_dir=None):
    global _model, _proc, _config
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    md = model_dir or _DEF_DIR
    _model, _proc = load(md)
    _config = load_config(md)
    return "mlx/Qwen2.5-VL-7B-4bit"


def _build_prompt(target, hist_lines):
    mem = "\n".join(hist_lines) if hist_lines else "none yet"
    return (
        "You are the navigation brain of a small ground robot car (forward camera) that rolls "
        "slowly forward. You are shown the most recent camera frame(s).\n"
        f'TASK: find this target and drive right up to it: "{target}".\n'
        "SAFETY: stay on the rug/carpet, keep to open floor, avoid furniture/walls; only the "
        "target itself may be approached closely.\n"
        f"Your recent decisions (newest first):\n{mem}\n\n"
        "Reply with ONLY a JSON object (no prose, no markdown) with EXACTLY these keys:\n"
        '{"target_visible": true|false, '
        '"target_bearing": "far_left|left|center|right|far_right|none", '
        '"target_size": "none|small|medium|large", '
        '"target_motion": "approaching|receding|steady|lost|unknown", '
        '"turn_strength": "none|slight|medium|hard", '
        '"search_strategy": "spin_left|spin_right|forward|turn_around|stop", '
        '"safe_forward": true|false, '
        '"arrived": true|false, '
        '"reason": "<short>"}\n'
        "Rules: target_bearing/size describe the CURRENT frame; target_motion compares frames; "
        "turn_strength = how hard to steer toward the target; search_strategy = where to look if "
        "not visible (use memory); arrived=true only if target_size is large and centered."
    )


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # try trimming trailing junk / single quotes
        s = m.group(0).replace("'", '"')
        try:
            return json.loads(re.search(r"\{.*\}", s, re.DOTALL).group(0))
        except Exception:
            return None


def _clamp(d):
    out = dict(_DEFAULT)
    if not isinstance(d, dict):
        return out
    out["target_visible"] = bool(d.get("target_visible", False))
    out["arrived"] = bool(d.get("arrived", False))
    out["safe_forward"] = bool(d.get("safe_forward", False))
    for key, allowed, default in (("target_bearing", _BEARING, "none"),
                                  ("target_size", _SIZE, "none"),
                                  ("target_motion", _MOTION, "unknown"),
                                  ("turn_strength", _STRENGTH, "medium"),
                                  ("search_strategy", _STRAT, "spin_right")):
        v = str(d.get(key, default)).strip().lower()
        out[key] = v if v in allowed else default
    out["reason"] = str(d.get("reason", ""))[:120]
    return out


def ask_seek_local(target, frames, hist_lines, max_frames=2, max_tokens=200):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    use = frames[-max_frames:] if frames else []
    tmp = []
    try:
        for _, jpg in use:
            f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            f.write(jpg); f.close(); tmp.append(f.name)
        prompt = apply_chat_template(_proc, _config, _build_prompt(target, hist_lines),
                                     num_images=max(1, len(tmp)))
        out = generate(_model, _proc, prompt, image=tmp, max_tokens=max_tokens,
                       temperature=0.2, verbose=False)
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        return _clamp(_extract_json(text))
    finally:
        for p in tmp:
            try:
                os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "cam_check.jpg"
    print("loading local VLM...")
    print("device:", load_local())
    jpg = open(img, "rb").read()
    t = time.time()
    d = ask_seek_local("yellow toy road roller / steamroller with a big cylindrical drum",
                       [(time.time(), jpg)], ["- target lost -> searched spin_right"])
    print(f"latency {time.time()-t:.1f}s")
    print(json.dumps(d, ensure_ascii=False, indent=2))
