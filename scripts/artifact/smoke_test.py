#!/usr/bin/env python3
"""Self-contained smoke test for the CF-PrefBench v4 / LIB v0 artifact.

Runs real checks against the released assets and emits
``smoke_test_results.json``. It does NOT fabricate a pass: each step
reports PASS / SKIP / FAIL with a reason. A SKIP (e.g. torch missing)
never counts as a PASS.

Steps
  1. Environment + package availability (numpy required, torch optional)
  2. Dataset metadata: count rows in the 8 core v4 splits; assert total 7,749
  3. Load the shipped smoke feature cache; report tensor shapes
  4. Load one released checkpoint; report tensor / parameter counts
  5. Dry-run forward: instantiate the released LIB v0 model sized to the
     smoke features and run a real forward pass producing preference scores
     (untrained weights -> a *dry-run eval*, not a paper-number reproduction)

Full paper-number reproduction needs the multi-GB frozen-CLIP feature
caches, which are excluded for size (regenerate via scripts/eval/extract_*).
"""
from __future__ import annotations
import json, os, sys, struct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

results = {"root": ROOT, "steps": [], "overall": "UNKNOWN"}

def step(name, status, detail):
    results["steps"].append({"name": name, "status": status, "detail": detail})
    print(f"[{status:4s}] {name}: {detail}")

# ---- Step 1: environment ----
py = sys.version.split()[0]
have = {}
for mod in ("numpy", "torch"):
    try:
        m = __import__(mod); have[mod] = getattr(m, "__version__", "?")
    except Exception as e:
        have[mod] = None
detail = f"python={py} numpy={have['numpy']} torch={have['torch']}"
if have["numpy"]:
    step("environment", "PASS", detail)
else:
    step("environment", "FAIL", detail + " (numpy is required)")

import numpy as np  # noqa  (numpy is a hard requirement)

# ---- Step 2: dataset ----
DATA = os.path.join(ROOT, "data", "cf_prefbench_v4")
CORE = ["train","val","test_seen","test_heldout_lexical","test_heldout_camera",
        "test_heldout_color","test_heldout_spatial","test_hard_negatives"]
try:
    total = 0; per = {}
    for nm in CORE:
        p = os.path.join(DATA, nm + ".jsonl")
        n = sum(1 for ln in open(p) if ln.strip())
        per[nm] = n; total += n
    ok = (total == 7749)
    step("dataset_rows", "PASS" if ok else "FAIL",
         f"core total={total} (expected 7749 -> {'OK' if ok else 'MISMATCH'}); per-split={per}")
except Exception as e:
    step("dataset_rows", "FAIL", f"{type(e).__name__}: {e}")

# ---- Step 3: feature cache ----
FEAT = os.path.join(ROOT, "features", "smoke_test_features.npz")
feat = None
try:
    feat = np.load(FEAT, allow_pickle=True)
    shapes = {k: tuple(getattr(feat[k], "shape", ())) for k in feat.files}
    step("feature_cache", "PASS", f"{os.path.basename(FEAT)} keys={feat.files} shapes={shapes}")
except Exception as e:
    step("feature_cache", "FAIL", f"{type(e).__name__}: {e}")

# ---- Step 4: checkpoint integrity ----
CKPT = os.path.join(ROOT, "checkpoints", "lib_b32", "lib_v0_b32_seed1_best.pt")
ckpt_state = None
try:
    if not os.path.exists(CKPT):
        step("checkpoint", "FAIL", f"missing {CKPT}")
    elif have["torch"]:
        import torch
        obj = torch.load(CKPT, map_location="cpu", weights_only=False)
        # find a state_dict-like mapping
        sd = obj
        if isinstance(obj, dict) and not all(hasattr(v, "shape") for v in obj.values()):
            for k in ("state_dict","model","lib","model_state"):
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]; break
        tensors = {k: v for k, v in (sd.items() if isinstance(sd, dict) else [])
                   if hasattr(v, "numel")}
        nparam = int(sum(v.numel() for v in tensors.values()))
        ckpt_state = obj
        step("checkpoint", "PASS",
             f"{os.path.basename(CKPT)} top-keys={list(obj.keys())[:6] if isinstance(obj,dict) else type(obj)} "
             f"tensors={len(tensors)} params={nparam}")
    else:
        # verify it is a valid PyTorch (zip) archive without torch
        magic = open(CKPT, "rb").read(2)
        ok = magic == b"PK"
        step("checkpoint", "PASS" if ok else "FAIL",
             f"torch absent; zip-magic={'OK' if ok else magic!r} size={os.path.getsize(CKPT)}")
except Exception as e:
    step("checkpoint", "FAIL", f"{type(e).__name__}: {e}")

# ---- Step 5: dry-run forward ----
if not have["torch"]:
    step("dry_run_forward", "SKIP",
         "torch not installed in this environment; install requirements.txt to run the forward pass")
elif feat is None or "video_patches" not in getattr(feat, "files", []):
    step("dry_run_forward", "SKIP", "smoke feature cache unavailable")
else:
    try:
        import torch
        from cf_pref_learning.models.lib import LIBModule, LIBPreferenceHead
        vp = np.asarray(feat["video_patches"]).astype("float32")   # [N,K,P,D]
        tf = np.asarray(feat["text_features"]).astype("float32")   # [M,Dt]
        N, K, P, D = vp.shape
        Dt = tf.shape[1]
        lib = LIBModule(clip_text_dim=Dt, clip_patch_dim=D).eval()
        head = LIBPreferenceHead(clip_text_dim=Dt).eval()
        with torch.no_grad():
            patches = torch.from_numpy(vp[: min(4, N)])
            text = torch.from_numpy(tf[: patches.shape[0]])
            out = lib(patches, text)
            # pair video i vs video (i+1) as a dry preference
            b = out["binding"]
            score = head(b, torch.roll(b, 1, 0), text)
        step("dry_run_forward", "PASS",
             f"LIB(text_dim={Dt},patch_dim={D}) forward OK; binding{tuple(b.shape)} "
             f"pref_scores={[round(float(x),4) for x in score]} (untrained weights = dry run)")
    except Exception as e:
        step("dry_run_forward", "FAIL", f"{type(e).__name__}: {e}")

# ---- overall ----
statuses = [s["status"] for s in results["steps"]]
core_steps = [s for s in results["steps"] if s["name"] in
              ("environment","dataset_rows","feature_cache","checkpoint")]
core_ok = all(s["status"] == "PASS" for s in core_steps)
results["overall"] = "PASS" if (core_ok and "FAIL" not in statuses) else (
    "PASS_CORE_WITH_SKIPS" if core_ok else "FAIL")
results["note"] = ("Core integrity checks (env/data/features/checkpoint) gate the result. "
                   "SKIP on dry_run_forward means torch was unavailable; it is not a pass. "
                   "Paper-number reproduction requires regenerating the excluded frozen-CLIP "
                   "feature caches (see docs/REPRODUCTION_GUIDE.md).")

out_path = os.path.join(ROOT, "smoke_test_results.json")
json.dump(results, open(out_path, "w"), indent=2)
print(f"\nOVERALL: {results['overall']}  ->  {out_path}")
sys.exit(0 if results["overall"].startswith("PASS") else 1)
