"""Stage C-prep: build held-out JSONL files for the 8 new tokens.

Mirrors scripts/generate_v4_axis_gen_tests.py exactly, but for the new pool. Output goes
to data/cf_prefbench_v4/test_heldout_{size,speed}_{token}.jsonl so the existing
eval pipeline can pick them up by filename.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT))
from cf_pref_learning.utils.io import read_jsonl, write_jsonl  # noqa: E402

# Match the in-class member sets used by scripts/generate_v4_axis_gen_tests.py
SIZE_BIG = {"large", "big", "huge"}
SIZE_SMALL = {"tiny", "small"}
SPEED_FAST = {"quickly", "rapidly"}
SPEED_SLOW = {"slowly", "leisurely"}

NEW_TOKENS = {
    "size": [
        ("enormous", "BIG"),
        ("vast", "BIG"),
        ("tiny", "SMALL"),
        ("minute", "SMALL"),
    ],
    "speed": [
        ("swiftly", "FAST"),
        ("hastily", "FAST"),
        ("leisurely", "SLOW"),
        ("languidly", "SLOW"),
    ],
}


def find_word(instr: str, members: set[str]) -> str | None:
    for w in instr.split():
        if w in members:
            return w
    return None


def main() -> None:
    src = list(read_jsonl(ROOT / "data/cf_prefbench_v4/test_heldout_lexical.jsonl"))
    size_rows = [r for r in src if r["axis"] == "size"]
    speed_rows = [r for r in src if r["axis"] == "speed"]
    print(f"source: size={len(size_rows)}  speed={len(speed_rows)}")

    for token, sem_class in NEW_TOKENS["size"]:
        members = SIZE_BIG if sem_class == "BIG" else SIZE_SMALL
        out: list[dict] = []
        seen = set()
        for r in size_rows:
            old = find_word(r["instruction"], members)
            if old is None:
                continue
            new_instr = r["instruction"].replace(old, token, 1)
            key = (r["pair_id"], new_instr, r["preferred"])
            if key in seen:
                continue
            seen.add(key)
            nr = dict(r)
            nr["instruction"] = new_instr
            nr["example_id"] = f"v4size_{token}_{r['example_id']}"
            nr["paraphrase_group_id"] = f"v4size_{token}_{r['pair_id']}"
            nr["lexical_items"] = [f"pg_size_{token}_{r['pair_id']}"]
            nr["split"] = f"test_heldout_size_{token}"
            md = dict(nr.get("metadata", {}))
            md["generator"] = "expanded_tokens_2026_05_24"
            md["semantic_class"] = sem_class
            nr["metadata"] = md
            out.append(nr)
        out_path = ROOT / f"data/cf_prefbench_v4/test_heldout_size_{token}.jsonl"
        write_jsonl(out_path, out)
        c = Counter(r["preferred"] for r in out)
        print(f"  size/{token:12s} ({sem_class}): {len(out)} rows  A={c['A']}  B={c['B']}")

    for token, sem_class in NEW_TOKENS["speed"]:
        members = SPEED_FAST if sem_class == "FAST" else SPEED_SLOW
        out = []
        seen = set()
        for r in speed_rows:
            old = find_word(r["instruction"], members)
            if old is None:
                continue
            new_instr = r["instruction"].replace(old, token, 1)
            key = (r["pair_id"], new_instr, r["preferred"])
            if key in seen:
                continue
            seen.add(key)
            nr = dict(r)
            nr["instruction"] = new_instr
            nr["example_id"] = f"v4speed_{token}_{r['example_id']}"
            nr["paraphrase_group_id"] = f"v4speed_{token}_{r['pair_id']}"
            nr["lexical_items"] = [f"pg_speed_{token}_{r['pair_id']}"]
            nr["split"] = f"test_heldout_speed_{token}"
            md = dict(nr.get("metadata", {}))
            md["generator"] = "expanded_tokens_2026_05_24"
            md["semantic_class"] = sem_class
            nr["metadata"] = md
            out.append(nr)
        out_path = ROOT / f"data/cf_prefbench_v4/test_heldout_speed_{token}.jsonl"
        write_jsonl(out_path, out)
        c = Counter(r["preferred"] for r in out)
        print(f"  speed/{token:12s} ({sem_class}): {len(out)} rows  A={c['A']}  B={c['B']}")


if __name__ == "__main__":
    main()
