"""Generate Phase 5 Exp 4 axis-generalization test data.

Semantic-preserving replacement: for each original held-out lexical row,
swap the size adjective or speed adverb with a new one IN THE SAME
SEMANTIC CLASS (BIG ↔ BIG, SMALL ↔ SMALL; FAST ↔ FAST, SLOW ↔ SLOW).
This keeps the labels correct while perturbing the lexical surface.

We test multiple new tokens spanning the CLIP-B/32 cosine spectrum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cf_pref_learning.utils.io import read_jsonl, write_jsonl


# Semantic class maps for replacement
SIZE_BIG = {"large", "big", "huge"}
SIZE_SMALL = {"tiny", "small"}
SPEED_FAST = {"quickly", "rapidly"}
SPEED_SLOW = {"slowly", "leisurely"}

# New tokens grouped by semantic class
SIZE_NEW = {
    "colossal": "BIG",     # cos 0.898 (deep cliff)
    "miniature": "SMALL",  # cos 0.918 (cliff)
    "petite": "SMALL",     # cos 0.917 (cliff)
    "gigantic": "BIG",     # cos 0.918 (cliff, BIG)
}
SPEED_NEW = {
    "briskly": "FAST",       # cos 0.937
    "sluggishly": "SLOW",    # cos 0.932
    "speedily": "FAST",      # cos 0.942
    "gradually": "SLOW",     # cos 0.938
}


def main() -> None:
    root = Path("/path/to/project")
    src = read_jsonl(root / "data" / "cf_prefbench_v4" / "test_heldout_lexical.jsonl")

    size_rows = [r for r in src if r["axis"] == "size"]
    speed_rows = [r for r in src if r["axis"] == "speed"]
    print(f"source: size={len(size_rows)}  speed={len(speed_rows)}")

    # For SIZE: identify the size word and its semantic class
    def find_size_word(instr):
        for w in instr.split():
            if w in SIZE_BIG: return w, "BIG"
            if w in SIZE_SMALL: return w, "SMALL"
        return None, None

    def find_speed_word(instr):
        for w in instr.split():
            if w in SPEED_FAST: return w, "FAST"
            if w in SPEED_SLOW: return w, "SLOW"
        return None, None

    # For each new size token, generate test rows
    for new_token, sem_class in SIZE_NEW.items():
        out_rows = []
        seen = set()
        skipped = 0
        for r in size_rows:
            old_word, old_class = find_size_word(r["instruction"])
            if old_word is None or old_class != sem_class:
                skipped += 1
                continue
            new_instr = r["instruction"].replace(old_word, new_token, 1)
            key = (r["pair_id"], new_instr, r["preferred"])
            if key in seen: continue
            seen.add(key)
            nr = dict(r)
            nr["instruction"] = new_instr
            nr["example_id"] = f"v4size_{new_token}_{r['example_id']}"
            nr["paraphrase_group_id"] = f"v4size_{new_token}_{r['pair_id']}"
            nr["lexical_items"] = [f"pg_size_{new_token}_{r['pair_id']}"]
            nr["split"] = f"test_heldout_size_{new_token}"
            md = dict(nr.get("metadata", {}))
            md["generator"] = "phase5_exp4_size_axis_gen"
            md["semantic_class"] = sem_class
            nr["metadata"] = md
            out_rows.append(nr)
        out_path = root / "data" / "cf_prefbench_v4" / f"test_heldout_size_{new_token}.jsonl"
        write_jsonl(out_path, out_rows)
        c = Counter(r["preferred"] for r in out_rows)
        print(f"  size/{new_token:12s} ({sem_class}): {len(out_rows)} rows  A={c['A']}  B={c['B']}  (skipped {skipped})")

    for new_token, sem_class in SPEED_NEW.items():
        out_rows = []
        seen = set()
        skipped = 0
        for r in speed_rows:
            old_word, old_class = find_speed_word(r["instruction"])
            if old_word is None or old_class != sem_class:
                skipped += 1
                continue
            new_instr = r["instruction"].replace(old_word, new_token, 1)
            key = (r["pair_id"], new_instr, r["preferred"])
            if key in seen: continue
            seen.add(key)
            nr = dict(r)
            nr["instruction"] = new_instr
            nr["example_id"] = f"v4speed_{new_token}_{r['example_id']}"
            nr["paraphrase_group_id"] = f"v4speed_{new_token}_{r['pair_id']}"
            nr["lexical_items"] = [f"pg_speed_{new_token}_{r['pair_id']}"]
            nr["split"] = f"test_heldout_speed_{new_token}"
            md = dict(nr.get("metadata", {}))
            md["generator"] = "phase5_exp4_speed_axis_gen"
            md["semantic_class"] = sem_class
            nr["metadata"] = md
            out_rows.append(nr)
        out_path = root / "data" / "cf_prefbench_v4" / f"test_heldout_speed_{new_token}.jsonl"
        write_jsonl(out_path, out_rows)
        c = Counter(r["preferred"] for r in out_rows)
        print(f"  speed/{new_token:12s} ({sem_class}): {len(out_rows)} rows  A={c['A']}  B={c['B']}  (skipped {skipped})")


if __name__ == "__main__":
    main()
