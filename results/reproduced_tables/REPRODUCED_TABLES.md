# Reproduced main tables (from released results)

Recomputed from `results/` per-seed JSONs by `scripts/artifact/reproduce_main_tables.py`.

## Table 1 — ViT-B/32 motion-verb cliff (3-seed)

| Verb | cos(B/32) | Accuracy (3-seed mean ± std) |
|---|---|---|
| shift | 0.968 | 0.917 ± 0.017 |
| convey | 0.938 | 0.929 ± 0.000 |
| scoot | 0.927 | 0.679 ± 0.058 |
| transit | 0.915 | 0.500 ± 0.000 |

## Table 2 — Motion-verb cliff under two backbones

| Verb | cos(B/32) | B/32 | cos(L/14) | L/14 |
|---|---|---|---|---|
| shift | 0.968 | 0.917 | 0.925 | 0.905 |
| convey | 0.938 | 0.929 | 0.918 | 0.929 |
| scoot | 0.927 | 0.679 | 0.875 | 0.905 |
| transit | 0.915 | 0.500 | 0.876 | 0.917 |

## Table 3 — 2x2 encoder factorial (mean cliff accuracy over cliff classes)

| Cell | cliff acc (mean over 6 cliff classes) | #seed files |
|---|---|---|
| B-B (baseline) | 0.633 | 3 |
| B-L | 0.827 | 3 |
| L-B | 0.611 | 3 |
| L-L (full upgrade) | 0.803 | 3 |

_Cross-check Table 3 (PDF): B-B 0.633, B-L 0.827, L-B 0.611, L-L 0.803 (n=3 cliff-probe-token protocol; small protocol differences expected)._

## Table 4 — Cross-family per-class cliff accuracy

| Class | B-B (OpenAI CLIP) | OC-OC (LAION-2B) | SG-SG (SigLIP) |
|---|---|---|---|
| motion_above | 0.887 | 0.679 | 0.661 |
| motion_below | 0.595 | 0.839 | 0.780 |
| size_BIG | 0.925 | 0.972 | 0.929 |
| size_SMALL | 0.444 | 0.929 | 0.817 |
| speed_FAST | 0.115 | 0.321 | 0.619 |
| speed_SLOW | 0.829 | 0.639 | 0.667 |

## Table 5 — Three-axis x two-architecture cliff matrix (expanded pool)

| Axis | Class | B/32 | L/14 | Δ |
|---|---|---|---|---|
| Size | BIG | 0.925 | 0.810 | -0.115 |
| Size | SMALL | 0.444 | 0.794 | +0.349 |
| Speed | FAST | 0.115 | 0.389 | +0.274 |
| Speed | SLOW | 0.829 | 0.996 | +0.167 |

_Note: this released file (`cliff_table.json`) holds the size/speed expanded pool; the motion above/below rows of Table 5 are in Tables 1-2 and `results/appendix/expanded_tokens/tables/`. Cross-check Table 5 (PDF): size SMALL 0.485, speed FAST 0.159 on B/32 (canonical n=8 x 4-token pool)._
