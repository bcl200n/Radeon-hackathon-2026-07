#!/usr/bin/env python3
"""Convert the 2026-08-01 official Chengdu emergency-shelter ledger
(23 districts) into the GeoJSON the block-scale pipeline consumes.

Why this replaces the earlier file: the previous 231-point Chengdu set was
an AMap POI list merged with a historical research inventory, every record
flagged operational_status=requires_verification -- candidate locations, not
a government roster. This ledger is the published municipal roster for all
23 districts, 1,324 rows, of which 1,269 carry WGS84 coordinates.

Three things the source needs before it can be used, each handled explicitly
rather than silently:

1. **Coordinates.** The shipped GeoJSON names its properties baidu_lng /
   baidu_lat, but its geometry actually carries the WGS84 pair from the CSV
   (verified: the Baidu BD-09 and WGS84 values for the same site differ by
   947 m, and the geometry matches WGS84). We read the xlsx WGS84 columns
   directly so nothing depends on that naming confusion.

2. **OCR noise.** Most districts were transcribed from images of published
   tables, and the README warns some sources were low-contrast. Roughly 12%
   of the 空间类型 values and a smaller share of 避难时长 are visually-similar
   mis-recognitions (臺外型 / 壶外 / 童内型 / 蜜补型： and so on). These are
   normalised by matching on the discriminating character (外 = outdoor,
   内 = indoor) rather than by exact string, and anything still ambiguous is
   recorded as unknown instead of being guessed.

3. **Capacity.** The ledger records neither floor area nor headcount, so
   capacity cannot be derived from it. It is assigned from the two
   classifications the ledger *does* carry -- indoor/outdoor and the
   emergency/short/long-term designation -- and every record is stamped
   capacity_source=type_assumption_no_area_in_ledger so no downstream
   reader can mistake it for a surveyed figure.

    python scripts/convert_chengdu_official_shelters.py \\
        --xlsx /tmp/cdshelters/成都市应急避难场所台账汇总_全23区县_补全_坐标.xlsx \\
        --output ../data/chengdu_shelters/chengdu_official_shelters_20260801.geojson
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

#: Assumed persons per site, by (space type, shelter duration). The ledger has
#: no area or headcount column, so these are assumptions, not measurements.
#: They are ordered by the physics the classification implies: open ground
#: holds more than a building; a one-day emergency stay tolerates far higher
#: density than a fortnight of shelter (Chinese practice commonly cites
#: ~1.5-2 m2/person for 紧急, 2-3 for 短期, 3-4.5 for 长期).
CAPACITY_BY_TYPE_DURATION: dict[tuple[str, str], float] = {
    ("outdoor", "emergency"): 3_000.0,
    ("outdoor", "short"):     2_000.0,
    ("outdoor", "long"):      1_200.0,
    ("outdoor", "unknown"):   2_000.0,
    ("indoor", "emergency"):    800.0,
    ("indoor", "short"):        600.0,
    ("indoor", "long"):         400.0,
    ("indoor", "unknown"):      600.0,
    ("both", "emergency"):    1_800.0,
    ("both", "short"):        1_300.0,
    ("both", "long"):           800.0,
    ("both", "unknown"):      1_300.0,
    ("unknown", "emergency"): 1_500.0,
    ("unknown", "short"):     1_000.0,
    ("unknown", "long"):        600.0,
    ("unknown", "unknown"):   1_000.0,
}


def normalise_space(raw: object) -> str:
    """OCR-tolerant mapping to indoor / outdoor / both / unknown."""
    s = str(raw or "").strip()
    if not s or s.lower() == "nan":
        return "unknown"
    has_out, has_in = "外" in s, "内" in s
    if has_out and has_in:
        return "both"          # 室内外兼有
    if has_out:
        return "outdoor"       # covers 室外型 / 室外 / 臺外型 / 壶外 / 主外型 ...
    if has_in:
        return "indoor"        # covers 室内型 / 室内 / 童内型 ...
    return "unknown"           # 蜜补型： and friends -- not guessed


def normalise_duration(raw: object) -> str:
    """OCR-tolerant mapping to emergency / short / long / unknown."""
    s = str(raw or "").strip()
    if not s or s.lower() == "nan":
        return "unknown"
    if "紧急" in s or "紧急避险" in s:
        return "emergency"
    if "短期" in s:
        return "short"
    if "长期" in s:
        return "long"
    # Frequent OCR renderings of 紧急 share 急/意/盒 shapes; 长 is distinctive.
    if s.startswith("长"):
        return "long"
    if any(ch in s for ch in ("急", "意", "盒")):
        return "emergency"
    return "unknown"


def pick(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        for col in df.columns:
            if c in str(col):
                return col
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    book = pd.ExcelFile(args.xlsx)
    sheets = [s for s in book.sheet_names if s != "台账总览"]

    features: list[dict] = []
    stats = Counter()
    space_hist, dur_hist = Counter(), Counter()

    for sheet in sheets:
        df = pd.read_excel(book, sheet_name=sheet)
        c_name = pick(df, "场所名称", "场所全称", "名称")
        c_addr = pick(df, "场所地址", "地址")
        c_space = pick(df, "空间类型")
        c_dur = pick(df, "避难时长")
        c_kind = pick(df, "避难种类")
        c_lon = pick(df, "WGS84经度")
        c_lat = pick(df, "WGS84纬度")
        if not (c_lon and c_lat):
            stats["sheets_without_coords"] += 1
            continue

        for _, row in df.iterrows():
            stats["rows_seen"] += 1
            lon, lat = row.get(c_lon), row.get(c_lat)
            if pd.isna(lon) or pd.isna(lat):
                stats["dropped_no_coords"] += 1
                continue
            lon, lat = float(lon), float(lat)
            # Chengdu municipality sits well inside this envelope; anything
            # outside is a geocoding failure, not a real site.
            if not (102.5 <= lon <= 105.3 and 29.8 <= lat <= 31.8):
                stats["dropped_out_of_bounds"] += 1
                continue

            space = normalise_space(row.get(c_space) if c_space else None)
            dur = normalise_duration(row.get(c_dur) if c_dur else None)
            space_hist[space] += 1
            dur_hist[dur] += 1
            capacity = CAPACITY_BY_TYPE_DURATION[(space, dur)]

            name = str(row.get(c_name) or "").strip() if c_name else ""
            if not name or name.lower() == "nan":
                name = f"{sheet} 应急避难场所"
                stats["name_missing"] += 1

            features.append({
                "type": "Feature",
                "properties": {
                    "shelter_id": f"CD{len(features)+1:05d}",
                    "name": name,
                    "district": sheet,
                    "address": (str(row.get(c_addr)).strip() if c_addr and not pd.isna(row.get(c_addr)) else ""),
                    "space_type": space,
                    "shelter_duration": dur,
                    "hazard_kinds": (str(row.get(c_kind)).strip() if c_kind and not pd.isna(row.get(c_kind)) else ""),
                    "capacity": capacity,
                    "capacity_source": "type_assumption_no_area_in_ledger",
                    "operational_status": "published_municipal_ledger",
                    "source": "Chengdu municipal emergency-shelter ledger, 23 districts, packaged 2026-07-31",
                    "crs_note": "WGS84 taken from the ledger's own WGS84 columns, not the Baidu BD-09 pair",
                },
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            })
            stats["written"] += 1

    payload = {
        "type": "FeatureCollection",
        "name": "Chengdu official emergency shelters (2026-08-01 ledger)",
        "provenance": {
            "source_file": args.xlsx.name,
            "districts": len(sheets),
            "rows_seen": stats["rows_seen"],
            "features_written": stats["written"],
            "dropped_no_coords": stats["dropped_no_coords"],
            "dropped_out_of_bounds": stats["dropped_out_of_bounds"],
            "names_substituted": stats["name_missing"],
            "space_type_distribution": dict(space_hist),
            "duration_distribution": dict(dur_hist),
            "evidence_class": (
                "Published municipal ledger -- a government roster, unlike the earlier "
                "AMap candidate list. Coordinates are the ledger's own WGS84 values. "
                "Most districts were transcribed from images of published tables, so "
                "classification fields carry OCR risk; values were normalised on the "
                "discriminating character and left unknown when ambiguous."
            ),
            "capacity_caveat": (
                "The ledger records neither area nor headcount. Capacity is assigned "
                "from space type x shelter duration and is an assumption, flagged per "
                "record as capacity_source=type_assumption_no_area_in_ledger. Absolute "
                "capacity totals should not be read as surveyed figures."
            ),
        },
        "features": features,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"districts read      : {len(sheets)}")
    print(f"rows seen           : {stats['rows_seen']:,}")
    print(f"features written    : {stats['written']:,}")
    print(f"dropped, no coords  : {stats['dropped_no_coords']:,}")
    print(f"dropped, off-map    : {stats['dropped_out_of_bounds']:,}")
    print(f"names substituted   : {stats['name_missing']:,}")
    print(f"space type          : {dict(space_hist)}")
    print(f"duration            : {dict(dur_hist)}")
    print(f"total assumed capacity: {sum(f['properties']['capacity'] for f in features):,.0f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
