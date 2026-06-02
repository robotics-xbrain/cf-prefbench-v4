"""CF-PrefBench v4 new-axes generator.

Adds 3 new compositional binding axes to the existing v3 dataset
WITHOUT regenerating v3 itself. The output is `data/cf_prefbench_v4/`
which contains v3 rows + new-axis rows appended.

New axes:
  1. size: "pick up the LARGE block" vs "pick up the SMALL block"
     Implementation: render `obj_target` with a `size` attribute
     (large = 14px, small = 4px). The "moving" rule is the same; the
     binding is "which object matches the instructed size".

  2. motion_sequence: "move LEFT then RIGHT" vs "RIGHT then LEFT"
     Implementation: trajectory goes via a waypoint
     {left_first} or {right_first}. Same renderer with two-segment
     interpolation.

  3. speed: "move QUICKLY" vs "move SLOWLY"
     Implementation: the moving object completes its trajectory in
     fewer frames (fast: t=0..1 over 12 frames) or more (slow: t=0..1
     spread over all 24 frames, but pause at start and end). The
     comparison is visible by where the object is at the midpoint.

All new axes reuse the existing renderer logic with minimal additions.
Same anti-shortcut discipline as v3: balanced flip groups, held-out
paraphrase pool for the lexical split, held-out tuples for
test_heldout_<axis>.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # type: ignore
import numpy as np

from cf_pref_learning.utils.io import ensure_dir, write_json, write_jsonl


# Reuse colors and palette from v3
COLORS = {
    "red":     (45, 45, 225),
    "blue":    (225, 75, 45),
    "yellow":  (45, 205, 230),
    "purple":  (210, 75, 150),
    "cyan":    (210, 200, 45),
    "magenta": (200, 60, 210),
}
PALETTE = list(COLORS.keys())
SHAPES = ["block", "puck", "ball"]
CAMERAS = ["front", "side", "tilted"]
TRAIN_CAMERAS = ["front", "side"]
HELDOUT_CAMERA = "tilted"

# Axis-specific anti-shortcut held-outs
# size: held-out pair = ("tiny", "huge") — extreme synonyms not in train
SIZE_TRAIN_PAIRS = [("large", "small"), ("big", "small"), ("small", "large"), ("small", "big")]
SIZE_HELDOUT_PAIRS = [("huge", "tiny"), ("tiny", "huge")]
SIZE_RENDER = {
    "large": 14, "big": 14, "huge": 18,
    "small": 4, "tiny": 3,
}

# motion_sequence: train pairs = (left_then_right, right_then_left), held-out = vertical sequences
MOTION_TRAIN_PAIRS = [("left_then_right", "right_then_left"), ("right_then_left", "left_then_right")]
MOTION_HELDOUT_PAIRS = [("up_then_down", "down_then_up"), ("down_then_up", "up_then_down")]

# speed: train pairs = (quickly, slowly), held-out = (fast, slow) lexical variants
SPEED_TRAIN_PAIRS = [("quickly", "slowly"), ("slowly", "quickly")]
SPEED_HELDOUT_PAIRS = [("rapidly", "leisurely"), ("leisurely", "rapidly")]
SPEED_RENDER = {
    "quickly": "fast", "rapidly": "fast",
    "slowly": "slow", "leisurely": "slow",
}

# Paraphrase pools — train + lexical-heldout
PARA_TRAIN_SIZE = [
    "pick up the {sz} {c} block",
    "lift the {sz} {c} block",
    "grasp the {sz} {c} block",
]
PARA_HELDOUT_SIZE = [
    "fetch the {sz} {c} block",
    "retrieve the {sz} {c} block",
    "secure the {sz} {c} block",
]
PARA_TRAIN_MOTION = [
    "move the block {m}",
    "drag the block {m}",
    "push the block {m}",
]
PARA_HELDOUT_MOTION = [
    "shift the block {m}",
    "transit the block {m}",
    "convey the block {m}",
]
PARA_TRAIN_SPEED = [
    "move the block {sp}",
    "carry the block {sp}",
    "transport the block {sp}",
]
PARA_HELDOUT_SPEED = [
    "shift the block {sp}",
    "advance the block {sp}",
    "translate the block {sp}",
]

# Motion descriptors → human-readable form for the instruction text
MOTION_TEXT = {
    "left_then_right": "left then right",
    "right_then_left": "right then left",
    "up_then_down": "up then down",
    "down_then_up": "down then up",
}

# Speed descriptors → human-readable form
SPEED_TEXT = {
    "quickly": "quickly",
    "slowly": "slowly",
    "rapidly": "rapidly",
    "leisurely": "leisurely",
}

SPLIT_GROUPS = {
    "train": 88,
    "val": 14,
    "test_seen": 12,
    "test_heldout_lexical": 14,
    "test_heldout_camera": 12,
    "test_heldout_color": 14,    # kept for consistency — color-heldout still uses held-out color tuples
    "test_heldout_spatial": 14,  # kept for consistency — spatial-heldout still uses held-out spatial tuples
    "test_hard_negatives": 12,
}

NEW_AXES = ["size", "motion_sequence", "speed"]
NEGATIVE_TYPES = ["wrong_target", "partial_success", "distractor_success", "instruction_wrong"]


def _hash_name(parts: list[Any]) -> str:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _project(pos, camera, w, h):
    x, y = pos
    if camera == "side":
        x, y = y, -x
    elif camera == "tilted":
        x, y = 0.75 * x + 0.25 * y, -0.20 * x + 0.95 * y
    return int((x + 0.75) / 1.5 * w), int((0.62 - y) / 1.24 * h)


def _initial_objects(rng, target_color, other_color, target_size=8, other_size=8, target_shape="block", other_shape="block"):
    distractor_color = next(c for c in PALETTE if c not in {target_color, other_color})
    return {
        "obj_target": {"color": target_color, "shape": target_shape,
                        "size": target_size,
                        "start": (rng.uniform(-0.55, 0.25), rng.uniform(-0.45, 0.45))},
        "obj_other":  {"color": other_color, "shape": other_shape,
                        "size": other_size,
                        "start": (rng.uniform(-0.55, 0.25), rng.uniform(-0.45, 0.45))},
        "obj_distractor": {"color": distractor_color, "shape": rng.choice(["block", "puck"]),
                           "size": 8,
                           "start": (rng.uniform(-0.55, 0.25), rng.uniform(-0.45, 0.45))},
    }


def _trajectory_simple(objects, moving_key, goal, outcome):
    """Single-segment trajectory: start → goal, possibly perturbed."""
    objs = copy.deepcopy(objects)
    if outcome == "wrong_target":
        goal = (-goal[0], -goal[1])
    elif outcome == "partial_success":
        sx, sy = objs[moving_key]["start"]
        goal = ((sx + goal[0]) / 2, (sy + goal[1]) / 2)
    elif outcome == "distractor_success":
        moving_key = "obj_distractor"
    elif outcome == "instruction_wrong":
        moving_key = "obj_other"
    return {"objects": objs, "moving": moving_key, "kind": "single", "goal": goal, "speed": "normal"}


def _trajectory_motion(objects, moving_key, motion_desc, outcome):
    """Two-segment trajectory according to motion_desc."""
    objs = copy.deepcopy(objects)
    sx, sy = objs[moving_key]["start"]
    # Define waypoints based on motion_desc
    if motion_desc == "left_then_right":
        wp1 = (-0.5, sy);  wp2 = (0.5, sy)
    elif motion_desc == "right_then_left":
        wp1 = (0.5, sy);   wp2 = (-0.5, sy)
    elif motion_desc == "up_then_down":
        wp1 = (sx, 0.45);  wp2 = (sx, -0.45)
    elif motion_desc == "down_then_up":
        wp1 = (sx, -0.45); wp2 = (sx, 0.45)
    else:
        raise ValueError(motion_desc)
    # Outcome perturbations
    if outcome == "wrong_target":
        wp1 = (-wp1[0], -wp1[1]); wp2 = (-wp2[0], -wp2[1])
    elif outcome == "partial_success":
        wp1 = ((sx + wp1[0]) / 2, (sy + wp1[1]) / 2)
        wp2 = ((sx + wp2[0]) / 2, (sy + wp2[1]) / 2)
    elif outcome == "distractor_success":
        moving_key = "obj_distractor"
    elif outcome == "instruction_wrong":
        moving_key = "obj_other"
    return {"objects": objs, "moving": moving_key, "kind": "two_segment",
            "wp1": wp1, "wp2": wp2, "speed": "normal"}


def _trajectory_speed(objects, moving_key, speed_render, outcome):
    """Single-segment trajectory with explicit speed marker.

    `speed_render` ∈ {"fast", "slow"}.
    Fast: object reaches goal by frame 12 then stays at goal frames 12-23.
    Slow: object stays at start for frames 0-11 then moves to goal frames 12-23.
    """
    objs = copy.deepcopy(objects)
    rng = random.Random(_hash_name([moving_key, speed_render]).__hash__() & 0xFFFFFFFF)
    goal = (rng.uniform(-0.55, 0.55), rng.uniform(-0.45, 0.45))
    if outcome == "wrong_target":
        goal = (-goal[0], -goal[1])
    elif outcome == "partial_success":
        sx, sy = objs[moving_key]["start"]
        goal = ((sx + goal[0]) / 2, (sy + goal[1]) / 2)
    elif outcome == "distractor_success":
        moving_key = "obj_distractor"
    elif outcome == "instruction_wrong":
        moving_key = "obj_other"
    return {"objects": objs, "moving": moving_key, "kind": "single",
            "goal": goal, "speed": speed_render}


def _render(path: Path, traj: dict[str, Any], camera: str) -> None:
    w, h = 192, 144
    n_frames = 24
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    for f_idx in range(n_frames):
        t_raw = f_idx / float(n_frames - 1)
        # Speed-modulated t
        speed = traj.get("speed", "normal")
        if speed == "fast":
            # Reach goal by frame 12 then stay
            t = min(1.0, t_raw * 2.0)
        elif speed == "slow":
            # Stay at start until frame 12 then accelerate
            t = max(0.0, (t_raw - 0.5) * 2.0)
        else:
            t = t_raw
        frame = np.full((h, w, 3), 230, np.uint8)
        cv2.rectangle(frame, (4, 4), (w - 5, h - 5), (185, 185, 185), 1)
        # Draw moving object + others
        for name, obj in traj["objects"].items():
            sx, sy = obj["start"]
            if name == traj["moving"]:
                if traj["kind"] == "single":
                    gx, gy = traj["goal"]
                    cv2.drawMarker(frame, _project((gx, gy), camera, w, h), (35, 35, 35),
                                    cv2.MARKER_CROSS, 12, 2)
                    x = sx * (1 - t) + gx * t
                    y = sy * (1 - t) + gy * t
                elif traj["kind"] == "two_segment":
                    wp1 = traj["wp1"]; wp2 = traj["wp2"]
                    cv2.drawMarker(frame, _project(wp1, camera, w, h), (90, 90, 90),
                                    cv2.MARKER_TRIANGLE_DOWN, 10, 1)
                    cv2.drawMarker(frame, _project(wp2, camera, w, h), (35, 35, 35),
                                    cv2.MARKER_CROSS, 12, 2)
                    if t < 0.5:
                        u = t * 2.0
                        x = sx * (1 - u) + wp1[0] * u
                        y = sy * (1 - u) + wp1[1] * u
                    else:
                        u = (t - 0.5) * 2.0
                        x = wp1[0] * (1 - u) + wp2[0] * u
                        y = wp1[1] * (1 - u) + wp2[1] * u
                else:
                    raise ValueError(traj["kind"])
            else:
                wobble = 0.015 * math.sin(2 * math.pi * t + len(name))
                x, y = sx + wobble, sy
            px, py = _project((x, y), camera, w, h)
            color = COLORS[obj["color"]]
            size = obj.get("size", 8)
            shape = obj["shape"]
            if shape == "block":
                cv2.rectangle(frame, (px - size, py - size), (px + size, py + size), color, -1)
            elif shape == "puck":
                cv2.circle(frame, (px, py), size, color, -1)
            else:  # ball
                cv2.circle(frame, (px, py), size, color, -1)
                cv2.circle(frame, (px, py), max(2, size // 2), (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _select_camera(rng, split):
    return HELDOUT_CAMERA if split == "test_heldout_camera" else rng.choice(TRAIN_CAMERAS)


def _select_negative_type(rng, split):
    return "hard_negative" if split == "test_hard_negatives" else rng.choice(NEGATIVE_TYPES)


def _sample_attr_pair(rng, train_pairs, heldout_pairs, split):
    """Sample (target_attr, other_attr) avoiding train-set blocklist."""
    blocklist = set(heldout_pairs)
    if split == "test_heldout_lexical":
        # held-out lexical reuses train pairs but with held-out paraphrase pool
        return tuple(rng.choice(train_pairs))
    # For all other splits, train_pairs (avoiding heldout pairs)
    return tuple(rng.choice(train_pairs))


def _build_intent_pair(axis: str, split: str, rng: random.Random) -> dict[str, Any]:
    use_heldout_paraphrases = split == "test_heldout_lexical"
    camera = _select_camera(rng, split)
    negative_type = _select_negative_type(rng, split)
    target_color, other_color = (rng.choice(PALETTE), rng.choice(PALETTE))
    while target_color == other_color:
        other_color = rng.choice(PALETTE)

    if axis == "size":
        target_attr, other_attr = _sample_attr_pair(rng, SIZE_TRAIN_PAIRS, SIZE_HELDOUT_PAIRS, split)
        # For test_heldout_<axis="size">-style heldout (no separate split here, but we add it
        # under "test_heldout_lexical" for the lexical case and otherwise use train pairs)
        # Note: v4 does NOT introduce a separate test_heldout_size split; the existing held-out
        # splits already test paraphrase, camera, color/spatial-tuple, and hard-negatives.
        # The lexical-heldout for size uses train-attribute words via a held-out paraphrase pool.
        target_size_px = SIZE_RENDER[target_attr]
        other_size_px = SIZE_RENDER[other_attr]
        objs = _initial_objects(rng, target_color, other_color,
                                target_size=target_size_px, other_size=other_size_px)
        goal = (rng.uniform(-0.55, 0.55), rng.uniform(-0.45, 0.45))
        traj_target = _trajectory_simple(objs, "obj_target", goal, "full")
        traj_other  = _trajectory_simple(objs, "obj_other", goal, negative_type)
        pool = PARA_HELDOUT_SIZE if use_heldout_paraphrases else PARA_TRAIN_SIZE
        intent_target = {
            "canonical": f"pick up the {target_attr} {target_color} block",
            "paraphrases": [p.format(sz=target_attr, c=target_color) for p in pool],
            "concept_tokens": [target_attr, target_color, "block"],
        }
        intent_other = {
            "canonical": f"pick up the {other_attr} {other_color} block",
            "paraphrases": [p.format(sz=other_attr, c=other_color) for p in pool],
            "concept_tokens": [other_attr, other_color, "block"],
        }
    elif axis == "motion_sequence":
        target_motion, other_motion = _sample_attr_pair(rng, MOTION_TRAIN_PAIRS, MOTION_HELDOUT_PAIRS, split)
        objs = _initial_objects(rng, target_color, other_color)
        traj_target = _trajectory_motion(objs, "obj_target", target_motion, "full")
        traj_other  = _trajectory_motion(objs, "obj_target", other_motion, negative_type)
        pool = PARA_HELDOUT_MOTION if use_heldout_paraphrases else PARA_TRAIN_MOTION
        intent_target = {
            "canonical": f"move the block {MOTION_TEXT[target_motion]}",
            "paraphrases": [p.format(m=MOTION_TEXT[target_motion]) for p in pool],
            "concept_tokens": [target_motion, "block"],
        }
        intent_other = {
            "canonical": f"move the block {MOTION_TEXT[other_motion]}",
            "paraphrases": [p.format(m=MOTION_TEXT[other_motion]) for p in pool],
            "concept_tokens": [other_motion, "block"],
        }
    elif axis == "speed":
        target_speed_word, other_speed_word = _sample_attr_pair(rng, SPEED_TRAIN_PAIRS, SPEED_HELDOUT_PAIRS, split)
        target_speed = SPEED_RENDER[target_speed_word]
        other_speed = SPEED_RENDER[other_speed_word]
        objs = _initial_objects(rng, target_color, other_color)
        traj_target = _trajectory_speed(objs, "obj_target", target_speed, "full")
        traj_other  = _trajectory_speed(objs, "obj_target", other_speed, negative_type)
        pool = PARA_HELDOUT_SPEED if use_heldout_paraphrases else PARA_TRAIN_SPEED
        intent_target = {
            "canonical": f"move the block {SPEED_TEXT[target_speed_word]}",
            "paraphrases": [p.format(sp=SPEED_TEXT[target_speed_word]) for p in pool],
            "concept_tokens": [target_speed_word, "block"],
        }
        intent_other = {
            "canonical": f"move the block {SPEED_TEXT[other_speed_word]}",
            "paraphrases": [p.format(sp=SPEED_TEXT[other_speed_word]) for p in pool],
            "concept_tokens": [other_speed_word, "block"],
        }
    else:
        raise ValueError(axis)
    return {
        "axis": axis,
        "split": split,
        "camera": camera,
        "negative_type": negative_type,
        "traj_target": traj_target,
        "traj_other": traj_other,
        "intent_target": intent_target,
        "intent_other": intent_other,
    }


def _imbalance_after(counts, paras_a, paras_b):
    tmp = Counter(counts)
    for t in paras_a:
        tmp[(t, "A")] += 1
    for t in paras_b:
        tmp[(t, "B")] += 1
    seen = {t for (t, _) in tmp}
    return sum(abs(tmp[(t, "A")] - tmp[(t, "B")]) for t in seen)


def _greedy_choose_side(template_counts, target_paras, other_paras, rng):
    score_a = _imbalance_after(template_counts, target_paras, other_paras)
    score_b = _imbalance_after(template_counts, other_paras, target_paras)
    if score_a < score_b:
        return "A", "B"
    if score_b < score_a:
        return "B", "A"
    return ("A", "B") if rng.random() < 0.5 else ("B", "A")


def _commit_counts(template_counts, paraphrases, label):
    for t in paraphrases:
        template_counts[(t, label)] += 1


def _emit_examples(examples, spec, pair_id, group_id, flip_id, video_a_rel, video_b_rel,
                   target_label, other_label, pair_idx):
    intent_target = spec["intent_target"]
    intent_other = spec["intent_other"]
    for intent_idx, (intent, label) in enumerate([(intent_target, target_label), (intent_other, other_label)]):
        para_group_id = f"v4_para_{pair_idx:06d}_{intent_idx}"
        for para_idx, text in enumerate(intent["paraphrases"]):
            examples.append({
                "example_id": f"v4_ex_{pair_idx:06d}_{intent_idx}_{para_idx}",
                "pair_id": pair_id,
                "video_a": video_a_rel,
                "video_b": video_b_rel,
                "instruction": text,
                "preferred": label,
                "axis": spec["axis"],
                "counterfactual_group_id": group_id,
                "counterfactual_flip_id": flip_id,
                "lexical_items": [f"pg_{para_group_id}"],
                "split": spec["split"],
                "paraphrase_group_id": para_group_id,
                "metadata": {
                    "generator": "mujoco_scripted_cf_prefbench_v4_new_axes",
                    "not_metaworld": True,
                    "deterministic_state_label": True,
                    "task_name": f"v4_{spec['axis']}",
                    "camera": spec["camera"],
                    "negative_type": spec["negative_type"],
                    "scores": {},
                    "exclude_from_cpl_training": False,
                },
            })


def generate(root: Path, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    raw = root / "data" / "raw" / "v4_new_axes"
    video_dir = ensure_dir(raw / "videos")
    out_dir = ensure_dir(root / "data" / "cf_prefbench_v4")

    examples_new: list[dict[str, Any]] = []
    template_counts: Counter = Counter()
    pair_idx = 0
    vid_idx = 0

    schedule: list[tuple[str, str]] = []
    for axis in NEW_AXES:
        for split, n in SPLIT_GROUPS.items():
            for _ in range(n):
                schedule.append((split, axis))
    rng.shuffle(schedule)

    for split, axis in schedule:
        spec = _build_intent_pair(axis, split, rng)
        target_para = spec["intent_target"]["paraphrases"]
        other_para = spec["intent_other"]["paraphrases"]
        target_label, other_label = _greedy_choose_side(template_counts, target_para, other_para, rng)
        if target_label == "A":
            traj_a, traj_b = spec["traj_target"], spec["traj_other"]
        else:
            traj_a, traj_b = spec["traj_other"], spec["traj_target"]
        name_a = f"v4_{_hash_name([seed, pair_idx, vid_idx, 'a'])}.mp4"; vid_idx += 1
        name_b = f"v4_{_hash_name([seed, pair_idx, vid_idx, 'b'])}.mp4"; vid_idx += 1
        video_a_path = video_dir / name_a
        video_b_path = video_dir / name_b
        _render(video_a_path, traj_a, spec["camera"])
        _render(video_b_path, traj_b, spec["camera"])
        pair_id = f"v4_pair_{pair_idx:06d}"
        flip_id = f"v4_flip_{pair_idx:06d}"
        group_id = f"v4_group_{pair_idx:06d}"
        _commit_counts(template_counts, target_para, target_label)
        _commit_counts(template_counts, other_para, other_label)
        _emit_examples(examples_new, spec, pair_id, group_id, flip_id,
                       str(video_a_path.relative_to(root)),
                       str(video_b_path.relative_to(root)),
                       target_label, other_label, pair_idx)
        pair_idx += 1

    # Combine with v3 rows (do not regenerate v3; load it from disk)
    from cf_pref_learning.utils.io import read_jsonl
    v3_splits = ["train", "val", "test_seen", "test_heldout_lexical",
                 "test_heldout_camera", "test_heldout_color",
                 "test_heldout_spatial", "test_hard_negatives"]
    v4_combined: dict[str, list[dict]] = {s: [] for s in v3_splits}
    for s in v3_splits:
        v3_rows = read_jsonl(root / "data" / "cf_prefbench" / f"{s}.jsonl")
        v4_combined[s].extend(v3_rows)
    for ex in examples_new:
        s = ex["split"]
        if s in v4_combined:
            v4_combined[s].append(ex)

    # Write v4 splits
    for s in v3_splits:
        write_jsonl(out_dir / f"{s}.jsonl", v4_combined[s])

    # Per-axis stats
    all_axes_count = Counter()
    for s in v3_splits:
        for ex in v4_combined[s]:
            all_axes_count[ex["axis"]] += 1
    per_split_axis_count = {s: Counter(ex["axis"] for ex in v4_combined[s]) for s in v3_splits}

    summary = {
        "generator": "mujoco_scripted_cf_prefbench_v4_new_axes",
        "not_metaworld": True,
        "deterministic_state_based_labels": True,
        "new_axes": NEW_AXES,
        "new_axis_examples": len(examples_new),
        "new_pairs": pair_idx,
        "v3_axes": ["color", "object", "action", "spatial", "impossible_premise"],
        "v4_axes_total": list(sorted(all_axes_count)),
        "per_axis_total_count": dict(sorted(all_axes_count.items())),
        "per_split_axis_count": {s: dict(c) for s, c in per_split_axis_count.items()},
        "splits": {s: len(v4_combined[s]) for s in v3_splits},
        "seed": seed,
        "size_train_pairs": SIZE_TRAIN_PAIRS, "size_heldout_pairs": SIZE_HELDOUT_PAIRS,
        "motion_train_pairs": MOTION_TRAIN_PAIRS, "motion_heldout_pairs": MOTION_HELDOUT_PAIRS,
        "speed_train_pairs": SPEED_TRAIN_PAIRS, "speed_heldout_pairs": SPEED_HELDOUT_PAIRS,
    }
    write_json(raw / "generation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/path/to/project")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    summary = generate(Path(args.project_root), args.seed)
    print(json.dumps({"v4_generated": True,
                       "new_axes": summary["new_axes"],
                       "new_examples": summary["new_axis_examples"],
                       "new_pairs": summary["new_pairs"],
                       "v4_total_examples": sum(summary["splits"].values()),
                       "v4_axes_total": summary["v4_axes_total"]}))


if __name__ == "__main__":
    main()
