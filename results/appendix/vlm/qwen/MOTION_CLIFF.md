# qwen2-vl-2b Per-Binding-Token Cliff Analysis

Each row is the VLM's row-level accuracy on a single held-out binding token,
compared with LIB v0 (B/32 and L/14) reference numbers from Exp 3-4.

| Token | n | VLM | LIB B/32 | LIB L/14 | cos(B/32) | Class |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| convey | 24 | 0.458 | 0.929 | 0.929 | 0.9380 | motion above-cliff |
| shift | 9 | 0.556 | 0.917 | 0.905 | 0.9680 | motion above-cliff |
| scoot | 15 | 0.333 | 0.679 | 0.905 | 0.9270 | motion below-cliff |
| transit | 9 | 0.556 | 0.500 | 0.917 | 0.9150 | motion below-cliff |
| colossal | 41 | 0.439 | 0.921 | 0.810 | 0.8980 | size BIG |
| gigantic | 40 | 0.450 | 0.929 | 0.810 | 0.9180 | size BIG |
| miniature | 39 | 0.538 | 0.397 | 0.817 | 0.9180 | size SMALL |
| petite | 39 | 0.513 | 0.492 | 0.770 | 0.9170 | size SMALL |
| briskly | 31 | 0.355 | 0.048 | 0.317 | 0.9370 | speed FAST |
| speedily | 41 | 0.537 | 0.183 | 0.460 | 0.9420 | speed FAST |
| gradually | 28 | 0.429 | 0.698 | 1.000 | 0.9380 | speed SLOW |
| sluggishly | 31 | 0.484 | 0.960 | 0.992 | 0.9320 | speed SLOW |

## Cliff tokens (VLM − LIB B/32)

| Token | VLM | LIB B/32 | Δ |
| --- | ---: | ---: | ---: |
| transit | 0.556 | 0.500 | +0.056 |
| scoot | 0.333 | 0.679 | -0.346 |
| miniature | 0.538 | 0.397 | +0.141 |
| petite | 0.513 | 0.492 | +0.021 |
| briskly | 0.355 | 0.048 | +0.307 |
| speedily | 0.537 | 0.183 | +0.354 |