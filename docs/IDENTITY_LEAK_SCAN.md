# Identity leak scan

Run after anonymization over the full artifact tree (text + binary).

## Method
- **Text files** (`.py .md .json .csv .tex .txt .sh .yml`): ordered string + token
  replacement in place (see `anonymize_log.json` in the build dir; 108 files, 179
  replacements), then `grep -rInE` for residual high-risk patterns.
- **Binary files** (`.pt .npz .mp4 .pdf`): `strings | grep` for high-risk patterns;
  bytes never modified.

## High-risk patterns scanned
`Xinmiao`, `Du Xinmiao`, `duxinmiao`, `Zhuoyu`, `Xihong`, `wxh`, `Peking University`,
`北京大学`, `PKU`, `stu.pku`, `pku.edu`, `/data3/`, `/home/duxinmiao`, `descfly`,
`shrc`, `2001111389`, `2401112131`, personal email.

## Replacements applied
| From | To |
|---|---|
| `/data3/.../EMNLP` | `/path/to/project` |
| `/data3/.../cf-prefbench-v4-anonymous` | `/path/to/cf-prefbench-v4-anonymous` |
| `/data3/duxinmiao/conda` → `/data3/duxinmiao` | `/path/to/conda` → `/path/to` |
| `/home/duxinmiao` | `/path/to/home` |
| `Xinmiao` / `Du Xinmiao` / `Zhuoyu` / `Xihong` | `Anonymous Author` |
| `duxinmiao` | `anonymous_user` |
| `Peking University` / `北京大学` / `PKU` | `Anonymous Institution` / `AnonInst` |
| `*pku.edu.cn` emails / personal email | `anonymous@example.com` |
| `descfly` / `shrc` | `anonymous_local_user` / `anonymous_server` |
| student IDs | `[redacted-id]` |

## Results
| Surface | High-risk hits after scrub |
|---|---|
| Text files | **0** |
| `.pt` checkpoints (48) | **0** (state-dict layer-name keys only; no embedded paths) |
| `.npz` feature caches (2) | **0** (`video_paths` are *relative* `data/raw/...`, no absolute path / identity) |
| `.mp4` sample videos (6) | **0** |
| `docs/EMNLP_final.pdf` | **0** (author block already `Anonymous ACL submission`) |

## Items removed during scan
- `results/appendix/vlm/gpt4o/run.log` — stray log (server paths) → deleted.
- `cf_pref_learning/**/__pycache__/*.pyc` — caches → deleted.
- a `.mplconfig/` dir that rode along a copy → deleted.

## Low-risk tokens (recorded, NOT blindly replaced)
Generic words like `author`, `affiliation`, `institution` appear only in anonymized
templated text (e.g. "Anonymous Author") — no identity-revealing context. Bibliography
author surnames inside `docs/EMNLP_final.pdf` (e.g. cited papers) are third-party
citations, not the submitting authors, and were correctly left untouched.

## Verdict
**PASS — no high-confidence identity leak detected.** No `BLOCKING_ANONYMITY_ISSUES.md`
generated. Safe to proceed to packaging (subject to a human spot-check before upload).
