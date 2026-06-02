from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..utils.io import ensure_dir, write_json


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_environment(project_root: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "env": {k: os.environ.get(k) for k in ["MUJOCO_GL", "DISPLAY", "PYOPENGL_PLATFORM"]},
        "modules": {m: module_available(m) for m in ["metaworld", "mani_skill2", "mani_skill", "mujoco", "gymnasium", "gym", "cv2", "imageio", "numpy"]},
        "rendering": {},
        "selected_generator": None,
        "blocking": [],
    }
    result["rendering"]["egl"] = _render_probe("egl")
    result["rendering"]["osmesa"] = _render_probe("osmesa")
    if result["modules"]["metaworld"]:
        result["selected_generator"] = "metaworld"
    elif result["modules"]["mani_skill2"] or result["modules"]["mani_skill"]:
        result["selected_generator"] = "maniskill"
    elif result["modules"]["mujoco"] and result["modules"]["cv2"] and result["rendering"]["egl"].get("ok"):
        result["selected_generator"] = "mujoco_scripted_fallback"
        result["blocking"].append("Meta-World and ManiSkill are unavailable; generated data must be reported as MuJoCo-scripted fallback, not Meta-World.")
    else:
        result["selected_generator"] = "blocked"
        result["blocking"].append("No supported simulator with headless rendering is available.")
    out = Path(project_root) / "outputs" / "e0_data_audit"
    ensure_dir(out)
    write_json(out / "simulator_env_check.json", result)
    _write_markdown(Path(project_root), result)
    return result


def _render_probe(gl_backend: str) -> dict[str, Any]:
    code = r"""
import json, os, traceback
out={"ok": False, "error": None, "pixel_sum": None, "shape": None}
try:
    import mujoco
    xml='<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="1 1 .1" rgba=".8 .8 .8 1"/><body pos="0 0 .1"><geom type="box" size=".05 .05 .05" rgba="1 0 0 1"/></body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer=mujoco.Renderer(model, height=64, width=64)
    renderer.update_scene(data)
    img=renderer.render()
    renderer.close()
    out["shape"]=list(img.shape)
    out["pixel_sum"]=int(img.sum())
    out["ok"]=bool(img.sum() > 0)
except Exception:
    out["error"]=traceback.format_exc()
print(json.dumps(out))
"""
    env = os.environ.copy()
    env["MUJOCO_GL"] = gl_backend
    proc = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, check=False, timeout=30)
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    except Exception:
        parsed = {"ok": False, "error": "Could not parse render probe output.", "stdout": proc.stdout, "stderr": proc.stderr}
    parsed["returncode"] = proc.returncode
    parsed["stderr_tail"] = proc.stderr[-1000:]
    return parsed


def _write_markdown(root: Path, result: dict[str, Any]) -> None:
    install = [
        "Meta-World installation option:",
        "`/path/to/envs/piper_torch/bin/python -m pip install git+https://github.com/Farama-Foundation/Metaworld.git`",
        "",
        "ManiSkill2 installation option:",
        "`/path/to/envs/piper_torch/bin/python -m pip install mani-skill2`",
        "",
        "After installation, re-run:",
        "`/path/to/envs/piper_torch/bin/python scripts/generate_metaworld_cf_prefbench.py --project-root /path/to/project --generator auto`",
    ]
    lines = [
        "# Simulator Environment Check",
        "",
        f"Python: `{result['python']}`",
        f"Selected generator: `{result['selected_generator']}`",
        "",
        "## Modules",
        "",
    ]
    lines.extend([f"- {k}: `{v}`" for k, v in sorted(result["modules"].items())])
    lines.extend(["", "## Rendering", ""])
    for backend, probe in result["rendering"].items():
        lines.append(f"- {backend}: ok=`{probe.get('ok')}`, pixel_sum=`{probe.get('pixel_sum')}`")
    lines.extend(["", "## Blocking Notes", ""])
    lines.extend([f"- {x}" for x in result.get("blocking", [])] or ["- None."])
    lines.extend(["", "## Install / Fix Instructions", ""])
    lines.extend(install)
    (root / "SIMULATOR_ENV_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

