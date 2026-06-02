# gpt-4o-2024-11-20 Per-Binding-Token Cliff Analysis

Each row is the VLM's row-level accuracy on a single held-out binding token,
compared with LIB v0 (B/32 and L/14) reference numbers from Exp 3-4.

| Token | n | VLM | LIB B/32 | LIB L/14 | cos(B/32) | Class |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| convey | 26 | 0.500 | 0.929 | 0.929 | 0.9380 | motion above-cliff |
| shift | 26 | 0.577 | 0.917 | 0.905 | 0.9680 | motion above-cliff |
| scoot | 25 | 0.600 | 0.679 | 0.905 | 0.9270 | motion below-cliff |
| transit | 23 | 0.478 | 0.500 | 0.917 | 0.9150 | motion below-cliff |
| colossal | 38 | 0.895 | 0.921 | 0.810 | 0.8980 | size BIG |
| gigantic | 36 | 0.889 | 0.929 | 0.810 | 0.9180 | size BIG |
| miniature | 42 | 0.786 | 0.397 | 0.817 | 0.9180 | size SMALL |
| petite | 41 | 0.829 | 0.492 | 0.770 | 0.9170 | size SMALL |
| briskly | 42 | 0.643 | 0.048 | 0.317 | 0.9370 | speed FAST |
| speedily | 42 | 0.714 | 0.183 | 0.460 | 0.9420 | speed FAST |
| gradually | 42 | 0.571 | 0.698 | 1.000 | 0.9380 | speed SLOW |
| sluggishly | 42 | 0.524 | 0.960 | 0.992 | 0.9320 | speed SLOW |

## Cliff tokens (VLM − LIB B/32)

| Token | VLM | LIB B/32 | Δ |
| --- | ---: | ---: | ---: |
| transit | 0.478 | 0.500 | -0.022 |
| scoot | 0.600 | 0.679 | -0.079 |
| miniature | 0.786 | 0.397 | +0.389 |
| petite | 0.829 | 0.492 | +0.337 |
| briskly | 0.643 | 0.048 | +0.595 |
| speedily | 0.714 | 0.183 | +0.531 |