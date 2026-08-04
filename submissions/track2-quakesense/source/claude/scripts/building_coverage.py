"""Measured walkable area per block, from building footprints.

The crowd density in the step loop divides people by the whole block polygon.
People walk outdoors, so that denominator is too large and congestion is
under-represented -- a limitation the documentation states, with an admitted
guess that the effective share is "plausibly 5-10 %". This replaces the guess
with a measurement: subtract the building footprints inside each block.

    walkable = block_area - built_area

The source is a DWG-derived footprint layer (222,421 polygons, 125.2 km2,
dated 2019) covering the central districts only. Eight of the twenty districts
have no coverage at all and two more are only grazed at the edges, so the ratio
is written where it is measured and left NaN elsewhere. An outer county is
mostly farmland and does not share the centre's built ratio; extrapolating
would invent data.

    python -m scripts.building_coverage \\
        --shp "../成都/成都/Chengdu_Buildings_DWG-Polygon.shp" \\
        --blocks blocks.geojson --cache chengdu_city.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Blocks with fewer than this many footprints are treated as unmeasured. One
#: stray building on a county edge says nothing about that block's built ratio,
#: and letting it through would report a 0.1 % coverage that is really the
#: absence of data.
MIN_FOOTPRINTS = 3


def main():
    import geopandas as gpd
    import shapely

    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.cache)
    n_blocks = len(z["lon"])
    blocks = gpd.read_file(a.blocks)
    print(f"blocks in geojson {len(blocks):,}, in cache {n_blocks:,}")

    # Match on the lon/lat the geojson already carries as properties -- the
    # same numbers the cache was built from. Recomputing polygon centroids
    # gives a slightly different point, and made 66,418 of 66,546 keys miss.
    gx = blocks["lon"].to_numpy(float)
    gy = blocks["lat"].to_numpy(float)
    key = {(round(x, 7), round(y, 7)): i for i, (x, y) in enumerate(zip(gx, gy))}
    want = np.array([key.get((round(x, 7), round(y, 7)), -1)
                     for x, y in zip(z["lon"], z["lat"])])
    if (want < 0).any():
        raise SystemExit(f"{int((want < 0).sum()):,} cache blocks have no "
                         f"matching geojson feature -- wrong block layer?")
    blocks = blocks.iloc[want].reset_index(drop=True)

    bld = gpd.read_file(a.shp)
    print(f"footprints {len(bld):,}  crs {bld.crs.name if bld.crs else '?'}")
    # Measure in the footprint layer's own metric CRS: areas must come out in
    # square metres, and reprojecting to a geographic CRS to then measure area
    # would be both slower and wrong.
    blocks = blocks.to_crs(bld.crs)

    # A DWG export carries self-touching rings. GEOS raises a side-location
    # conflict on those rather than quietly returning a wrong answer, so
    # repair before intersecting anything.
    bad = ~bld.geometry.is_valid
    if bad.any():
        print(f"  repairing {int(bad.sum()):,} invalid footprints")
        bld.loc[bad, "geometry"] = bld.geometry[bad].make_valid()
    bbad = ~blocks.geometry.is_valid
    if bbad.any():
        print(f"  repairing {int(bbad.sum()):,} invalid blocks")
        blocks.loc[bbad, "geometry"] = blocks.geometry[bbad].make_valid()

    # Pairwise intersection, vectorised. A Python loop over 66,546 blocks and
    # 222,421 footprints is the slow way to ask the same question.
    pairs = gpd.sjoin(blocks[["geometry"]], bld[["geometry"]],
                      predicate="intersects", how="inner")
    print(f"  {len(pairs):,} block-building pairs")
    li = pairs.index.to_numpy()
    ri = pairs["index_right"].to_numpy()
    inter = shapely.intersection(blocks.geometry.values[li],
                                 bld.geometry.values[ri])
    # A building straddling a block boundary contributes only its share to
    # each side, which is what taking the intersection area gives.
    built = np.bincount(li, weights=shapely.area(inter), minlength=n_blocks)
    hits = np.bincount(li, minlength=n_blocks)

    block_area = blocks.geometry.area.values
    measured = hits >= MIN_FOOTPRINTS
    ratio = np.where(measured,
                     np.clip(built / np.maximum(block_area, 1.0), 0, 0.95),
                     np.nan)

    m = ratio[measured]
    print(f"\nmeasured on {measured.sum():,} of {n_blocks:,} blocks "
          f"({measured.mean() * 100:.1f} %)")
    print(f"  built ratio  p10 {np.percentile(m, 10):.3f}   "
          f"median {np.median(m):.3f}   p90 {np.percentile(m, 90):.3f}")
    print(f"  open share   median {1 - np.median(m):.3f}")
    print(f"\n  The documented guess put walkable space at 5-10 % of block area.")
    print(f"  Measured, the open share is {(1 - np.median(m)) * 100:.0f} %. The guess")
    print(f"  described the street network alone; a block's open ground also")
    print(f"  includes courtyards and the gaps between buildings, which people")
    print(f"  can stand in. Congestion is under-represented, but by less than")
    print(f"  the order of magnitude the guess implied.")

    np.savez(a.out, built_ratio=ratio, measured=measured,
             block_area=block_area, built_area=built, hits=hits)
    Path(a.out).with_suffix(".json").write_text(json.dumps({
        "blocks_measured": int(measured.sum()),
        "blocks_total": int(n_blocks),
        "built_ratio_median": float(np.median(m)),
        "built_ratio_p10": float(np.percentile(m, 10)),
        "built_ratio_p90": float(np.percentile(m, 90)),
        "open_share_median": float(1 - np.median(m)),
        "footprints": int(len(bld)),
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
