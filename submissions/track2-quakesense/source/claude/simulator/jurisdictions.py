"""Equal-population jurisdictions for the leader tier.

Leaders were previously drawn as random agents, with every block assigned to
its nearest leader. Leader *density* came out right -- agents are distributed
in proportion to population, so a district with twice the people gets twice the
leaders -- but jurisdiction *size* did not:

    350 leaders    p5  14,040   p50  56,223   p100 298,113   ratio 9.9x
    1,000 leaders  p5       0   p50  19,571   p100 109,237   ratio 48,628x
                                              53 jurisdictions with nobody in them

Randomly scattered seeds compete for territory, and the losers end up empty.
At 1,000 leaders that wasted 53 LLM calls per round on jurisdictions with no
one to direct, and left one leader responsible for 109,237 people while another
had none. "One commander, one area" stops meaning anything.

So partition deliberately instead. Order the blocks along a Hilbert curve and
cut the sequence wherever cumulative population reaches the next 1/k share.
Because a Hilbert curve keeps points that are near each other in the plane near
each other in the ordering, every contiguous run of the sequence is a compact
patch of city -- so the cuts give jurisdictions that are both equal in
population and spatially coherent, with no iteration and no empty cells.

This is a partition of convenience, not Chengdu's real subdistrict boundaries.
Those would need the administrative boundary vectors, which this project does
not have; see docs/ROADMAP_FULL_SCALE_AGENTS.md. What it does honestly claim is
a defensible span of control per leader.
"""

from __future__ import annotations

import numpy as np

#: Hilbert grid resolution. 2^16 cells across Chengdu's ~200 km extent is
#: roughly 3 m -- far finer than a block, so the ordering never has to break a
#: tie between two blocks that share a cell.
_ORDER = 16


def _hilbert_index(x: np.ndarray, y: np.ndarray, order: int = _ORDER) -> np.ndarray:
    """Vectorised (x, y) -> Hilbert distance on a 2^order grid."""
    n = 1 << order
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    d = np.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # Rotate the quadrant so the curve stays continuous across it.
        flip = (ry == 0) & (rx == 1)
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return d


def equal_population_partition(bx, by, population, k: int):
    """Split blocks into ``k`` compact, equal-population jurisdictions.

    Returns ``(owner, seed_block)`` where ``owner[b]`` is the jurisdiction of
    block b and ``seed_block[j]`` is the block holding jurisdiction j's
    population centre -- a natural place to put its leader.
    """
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    pop = np.asarray(population, dtype=np.float64)
    n = len(bx)
    k = max(1, min(int(k), n))

    span_x = max(bx.max() - bx.min(), 1e-9)
    span_y = max(by.max() - by.min(), 1e-9)
    grid = (1 << _ORDER) - 1
    gx = np.clip(((bx - bx.min()) / span_x * grid).astype(np.int64), 0, grid)
    gy = np.clip(((by - by.min()) / span_y * grid).astype(np.int64), 0, grid)

    order = np.argsort(_hilbert_index(gx, gy), kind="stable")
    cum = np.cumsum(pop[order])
    total = cum[-1] if cum[-1] > 0 else 1.0

    # Cut at the k-1 interior quantiles of cumulative population. searchsorted
    # on the running total is what makes the split exactly equal-population
    # rather than equal-block-count, which is the whole point: a rural block
    # and a tower block are not the same amount of responsibility.
    cuts = np.searchsorted(cum, total * np.arange(1, k) / k)
    owner_sorted = np.zeros(n, dtype=np.int32)
    owner_sorted[np.clip(cuts, 0, n - 1)] = 1
    owner_sorted = np.cumsum(owner_sorted).astype(np.int32)
    owner_sorted = np.minimum(owner_sorted, k - 1)

    owner = np.empty(n, dtype=np.int32)
    owner[order] = owner_sorted

    # Seed = the block at each jurisdiction's population midpoint along the
    # curve, which sits near its centre rather than at either edge.
    seed = np.empty(k, dtype=np.int32)
    start = 0
    for j in range(k):
        end = int(cuts[j]) + 1 if j < k - 1 else n
        end = max(end, start + 1)
        seg = order[start:end]
        c = np.cumsum(pop[seg])
        mid = int(np.searchsorted(c, c[-1] / 2.0)) if c[-1] > 0 else len(seg) // 2
        seed[j] = seg[min(mid, len(seg) - 1)]
        start = end
    return owner, seed


def partition_report(owner, population, k: int) -> dict:
    """Numbers to put in the paper instead of asserting the split is balanced."""
    jp = np.bincount(owner, weights=np.asarray(population, dtype=np.float64),
                     minlength=k)
    p5, p95 = np.percentile(jp, 5), np.percentile(jp, 95)
    return {
        "jurisdictions": int(k),
        "mean_population": float(jp.mean()),
        "p5": float(p5), "p50": float(np.median(jp)), "p95": float(p95),
        "min": float(jp.min()), "max": float(jp.max()),
        "p95_over_p5": float(p95 / p5) if p5 > 0 else float("inf"),
        "empty_jurisdictions": int((jp == 0).sum()),
    }
