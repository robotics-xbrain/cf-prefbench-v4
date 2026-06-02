from __future__ import annotations

import random
from typing import Callable


def bootstrap_ci(values: list[float], statistic: Callable[[list[float]], float] | None = None, seed: int = 42, n_boot: int = 1000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        draws.append(stat(sample))
    draws.sort()
    return {"mean": stat(values), "ci_low": draws[int(0.025 * n_boot)], "ci_high": draws[int(0.975 * n_boot)]}

