#!/usr/bin/env python3
"""Build the integrated QuakeSense planning + simulation WebGIS."""

from __future__ import annotations

import base64
import copy
import json
import zlib
from pathlib import Path

import numpy as np

try:
    from extract_roads import extract as extract_roads
except ImportError:  # Support both `python scripts/...` and module imports.
    from scripts.extract_roads import extract as extract_roads


PROJECT = Path("/workspace/persistence/codex/project")
CODEX = PROJECT.parent
CACHE = CODEX / "cache"
PLANNING = PROJECT / "planning" / "results"
RUNS = PROJECT / "results_current"
OUT = PROJECT / "webgis"
CHENGDU_ROAD_SOURCE = Path(
    "/workspace/persistence/xichang-agentic-evacuation/data/external/"
    "chengdu_admin_roads.geojson"
)
XIAN_ROAD_SOURCE = Path("/workspace/persistence/xiaomi/xian_shelters/xian_roads.geojson")
NOTO_ROAD_SOURCE = Path(
    "/workspace/persistence/xichang-agentic-evacuation/data/external/noto_roads.geojson"
)
GLOBAL_SOURCE = Path("/workspace/persistence/xichang-agentic-evacuation")
GLOBAL_CITY_META = {
    "kathmandu": ("加德满都", 100.0),
    "l_aquila": ("拉奎拉", 70.0),
    "los_angeles": ("洛杉矶", 120.0),
    "mandalay": ("曼德勒", 100.0),
    "naples": ("那不勒斯", 70.0),
    "noto": ("能登", 80.0),
    "taipei": ("台北", 70.0),
    "wellington": ("惠灵顿", 80.0),
}
CELL_DEG = 0.025


def pack(values: np.ndarray) -> str:
    raw = np.ascontiguousarray(values, dtype=np.float32).tobytes()
    return base64.b64encode(zlib.compress(raw, 7)).decode("ascii")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_network(city: np.lib.npyio.NpzFile) -> dict:
    """Encode the exact block-adjacency edges traversed by the agents."""
    neigh = city["neigh"].astype(np.int64)
    source = np.repeat(np.arange(len(neigh), dtype=np.int64), neigh.shape[1])
    target = neigh.reshape(-1)
    keep = (target >= 0) & (target > source)
    source = source[keep]
    target = target[keep]
    lon = city["lon"].astype(np.float32)
    lat = city["lat"].astype(np.float32)
    edges = np.column_stack((lon[source], lat[source], lon[target], lat[target]))
    return {"count": int(len(edges)), "edges": pack(edges)}


def scenario(run_name: str, label: str) -> dict:
    run = RUNS / run_name
    frames = np.load(run / "frames.npz")
    summary = load_json(run / "summary.json")
    lon = frames["lon"].astype(np.float64)
    lat = frames["lat"].astype(np.float64)
    pop = frames["pop"].astype(np.float64)
    x0, y0 = float(lon.min()), float(lat.min())
    col = np.floor((lon - x0) / CELL_DEG).astype(np.int32)
    row = np.floor((lat - y0) / CELL_DEG).astype(np.int32)
    ncol = int(col.max()) + 1
    cell = row * ncol + col
    occupied = np.unique(cell[pop > 0])
    remap = np.full(int(cell.max()) + 1, -1, dtype=np.int32)
    remap[occupied] = np.arange(len(occupied), dtype=np.int32)
    compact = remap[cell]
    valid = compact >= 0
    nc = len(occupied)

    weights = np.bincount(compact[valid], weights=pop[valid], minlength=nc)
    cell_lon = np.bincount(compact[valid], weights=lon[valid] * pop[valid], minlength=nc)
    cell_lat = np.bincount(compact[valid], weights=lat[valid] * pop[valid], minlength=nc)
    cell_lon /= np.maximum(weights, 1)
    cell_lat /= np.maximum(weights, 1)

    arrays: dict[str, np.ndarray] = {}
    totals: dict[str, list[float]] = {}
    for key in ("transit", "sheltered", "gaveup"):
        out = np.stack([
            np.bincount(compact[valid], weights=frames[key][i, valid], minlength=nc)
            for i in range(len(frames["t_min"]))
        ]).astype(np.float32)
        arrays[key] = out
        totals[key] = out.sum(axis=1).astype(float).tolist()
    for key in ("known", "belief"):
        out = np.stack([
            np.bincount(
                compact[valid], weights=frames[key][i, valid] * pop[valid], minlength=nc
            ) / np.maximum(weights, 1)
            for i in range(len(frames["t_min"]))
        ]).astype(np.float32)
        arrays[key] = out

    t = summary["totals"]
    return {
        "label": label,
        "run": run_name,
        "nframes": len(frames["t_min"]),
        "ncells": nc,
        "time": frames["t_min"].astype(float).round(2).tolist(),
        "lon": pack(cell_lon),
        "lat": pack(cell_lat),
        "layers": {key: pack(value) for key, value in arrays.items()},
        "totals": totals,
        "summary": {
            "agents": int(summary["n_agents"]),
            "sheltered": int(t["sheltered"]),
            "safe15": float(t["share_safe_by_15min"]),
            "safe120": float(t["share_safe_by_120min"]),
            "gini": float(t.get("shelter_occupancy_gini", 0)),
            "rejections": int(t.get("rejections_at_full_gate", 0)),
        },
    }


def build_chengdu_payload() -> dict:
    city = np.load(CACHE / "chengdu_city.npz")
    candidates = np.load(CACHE / "poi_candidate_blocks.npz")
    owner = load_json(CACHE / "district_owner_city.json")
    names = [row["name"] for row in owner["districts"]]
    name_to_idx = {name: i for i, name in enumerate(names)}
    constraints = load_json(PLANNING / "two_constraints.json")
    by_name = {row["name"]: row for row in constraints}
    blocks = np.load(PLANNING / "poi_network_blocks.npz")
    nearer = np.isfinite(blocks["near_candidate"]) & (
        blocks["near_candidate"] < blocks["near_official"]
    )
    district = blocks["district"].astype(int)
    pop = blocks["pop"].astype(float)
    benefit = np.zeros(len(names))
    population = np.zeros(len(names))
    for i in range(len(names)):
        m = district == i
        population[i] = pop[m].sum()
        benefit[i] = pop[m & nearer].sum() / max(population[i], 1)

    geo = load_json(PROJECT / "chengdu_districts.geojson")
    for feature in geo["features"]:
        props = feature["properties"]
        name = props.get("name")
        row = by_name.get(name)
        idx = name_to_idx.get(name)
        if row is None or idx is None:
            continue
        props.update({
            "population": round(float(row["pop"])),
            "reach15": round(float(row["reach"]) * 100, 2),
            "stock": round(float(row["stock"]) * 100, 2),
            "benefit": round(float(benefit[idx]) * 100, 2),
            "constraint": "capacity" if row["stock"] < row["reach"] else "reach",
        })

    planning = load_json(PLANNING / "poi_network.json")
    kinds = [str(value) for value in candidates["kind"]]
    canonical = load_json(CODEX / "reports" / "canonical_metrics.json")
    road_bbox = [
        float(city["lon"].min()), float(city["lat"].min()),
        float(city["lon"].max()), float(city["lat"].max()),
    ]
    roads = extract_roads(CHENGDU_ROAD_SOURCE, road_bbox, tol_m=140.0)
    return {
        "cityKey": "chengdu",
        "cityName": "成都",
        "subtitle": "规划诊断 × 1,887 万个体疏散仿真",
        "districtCount": 20,
        "hasCandidates": True,
        "candidatePopulation": "1,574万",
        "callout": "17 个区县首先受容量约束。候选点把“够不够近”显著改善，但只有赋予可审计的容量，才能把可达性转化为真正安置。",
        "methodNote": "42.45% 是静态路网可达率；仿真中的 15 分钟安全率还受容量、信息和行为影响，两者不可直接等同。",
        "districts": geo,
        "official": [
            [round(float(lat), 5), round(float(lon), 5), int(cap)]
            for lon, lat, cap in zip(city["s_lon"], city["s_lat"], city["s_cap"])
        ],
        "candidates": [
            [round(float(lat), 5), round(float(lon), 5), kind]
            for lon, lat, kind in zip(candidates["lon"], candidates["lat"], kinds)
        ],
        "headline": {
            "population": int(round(canonical["planning"]["population_from_district_table"])),
            "shelters": int(len(city["s_lon"])),
            "candidates": int(len(candidates["lon"])),
            "reach15": round(planning["reach_official_network"]["15"] * 100, 1),
            "reachBoth15": round(planning["reach_both_network"]["15"] * 100, 1),
            "capacity": round(canonical["planning"]["capacity_share"] * 100, 1),
            "stockLimited": canonical["planning"]["stock_limited_districts"],
        },
        "roadBbox": road_bbox,
        "roads": roads,
        "modelNetwork": model_network(city),
        "scenarios": {
            "baseline": scenario("ab_city_k000", "零知识基线"),
            "llm": scenario("ab_city_llm000", "LLM 领导引导"),
        },
    }


def build_xian_payload() -> dict:
    city = np.load(CACHE / "xian_city.npz")
    planning = load_json(PROJECT / "cities" / "xian" / "results" / "planning.json")
    by_name = {row["name"]: row for row in planning["districts"]}
    geo = load_json(PROJECT / "portal" / "data" / "xian" / "xian_districts.geojson")
    for feature in geo["features"]:
        props = feature["properties"]
        row = by_name[props["name"]]
        props.update({
            "population": round(float(row["population"])),
            "reach15": round(float(row["reach15"]) * 100, 2),
            "stock": round(float(row["stock"]) * 100, 2),
            "benefit": 0.0,
            "constraint": row["constraint"],
        })
    road_bbox = [
        float(city["lon"].min()), float(city["lat"].min()),
        float(city["lon"].max()), float(city["lat"].max()),
    ]
    return {
        "cityKey": "xian",
        "cityName": "西安",
        "subtitle": "13 区县规划诊断 × 1,295 万个体疏散仿真",
        "districtCount": 13,
        "hasCandidates": False,
        "candidatePopulation": "—",
        "callout": "10 个区县首先受场所容量约束；周至、长安和高陵首先受 15 分钟路网距离约束。当前场所容量只能覆盖约四成人口。",
        "methodNote": "静态可达性采用街区邻接网络和质心距离；动态安全率还受容量、信息传播和行为响应影响，两者不可直接等同。",
        "districts": geo,
        "official": [
            [round(float(lat), 5), round(float(lon), 5), int(cap)]
            for lon, lat, cap in zip(city["s_lon"], city["s_lat"], city["s_cap"])
        ],
        "candidates": [],
        "headline": {
            "population": int(round(planning["population"])),
            "shelters": int(planning["shelters"]),
            "candidates": 0,
            "reach15": round(planning["reach"]["15"] * 100, 1),
            "reachBoth15": round(planning["network_reachable_share"] * 100, 1),
            "capacity": round(planning["capacity_share"] * 100, 1),
            "stockLimited": int(planning["stock_limited_districts"]),
        },
        "roadBbox": road_bbox,
        "roads": extract_roads(XIAN_ROAD_SOURCE, road_bbox, tol_m=120.0),
        "modelNetwork": model_network(city),
        "scenarios": {
            "baseline": scenario("xian_k000", "零知识基线"),
            "llm": scenario("xian_llm000", "LLM 领导引导"),
        },
    }


def build_noto_payload() -> dict:
    city = np.load(CACHE / "noto_city.npz")
    planning = load_json(PROJECT / "cities" / "noto" / "results" / "planning.json")
    row = planning["districts"][0]
    west, south, east, north = planning["bbox"]
    geo = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "能登研究区",
                "population": round(float(row["population"])),
                "reach15": round(float(row["reach15"]) * 100, 2),
                "stock": round(float(row["stock"]) * 100, 2),
                "benefit": 0.0,
                "constraint": row["constraint"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [west, south], [east, south], [east, north],
                    [west, north], [west, south],
                ]],
            },
        }],
    }


def build_global_payload(city_key: str) -> dict:
    city_name, tolerance = GLOBAL_CITY_META[city_key]
    city = np.load(CACHE / f"{city_key}_city.npz")
    planning = load_json(PROJECT / "cities" / city_key / "results" / "planning.json")
    row = planning["districts"][0]
    west, south, east, north = planning["bbox"]
    geo = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": row["name"],
                "population": round(float(row["population"])),
                "reach15": round(float(row["reach15"]) * 100, 2),
                "stock": round(float(row["stock"]) * 100, 2),
                "benefit": 0.0,
                "constraint": row["constraint"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [west, south], [east, south], [east, north],
                    [west, north], [west, south],
                ]],
            },
        }],
    }
    capacity_pct = planning["capacity_share"] * 100
    reach_pct = planning["reach"]["15"] * 100
    if row["constraint"] == "capacity":
        diagnosis = (
            f"情景容量仅相当于研究人口的 {capacity_pct:.1f}%，低于15分钟网络可达率 "
            f"{reach_pct:.1f}%，当前首先受容量约束。"
        )
    else:
        diagnosis = (
            f"情景容量相当于研究人口的 {capacity_pct:.1f}%，但15分钟网络可达率只有 "
            f"{reach_pct:.1f}%，当前首先受距离和地形阻力约束。"
        )
    road_bbox = [
        float(city["lon"].min()), float(city["lat"].min()),
        float(city["lon"].max()), float(city["lat"].max()),
    ]
    return {
        "cityKey": city_key,
        "cityName": city_name,
        "subtitle": f"道路街区与地形阻力 × {planning['population'] / 1e4:.1f} 万个体疏散仿真",
        "districtCount": 1,
        "hasCandidates": False,
        "candidatePopulation": "—",
        "officialTerm": "筛选后 OSM 场所候选",
        "reachLabel": "候选场所15分钟可达",
        "capacityLabel": "情景容量／人口",
        "limitedLabel": "首要受限研究区",
        "callout": "容量为按场所类型推定的情景值，并非当地政府核定容量。" + diagnosis,
        "methodNote": "人口来自 WorldPop 2020；道路来自 OSM；模型加入 SRTM 坡度与 ESA WorldCover 阻力。消防栓、除颤器等非避难设施已剔除；容量与候选名单仍不能替代官方应急规划。",
        "districts": geo,
        "official": [
            [round(float(lat), 5), round(float(lon), 5), int(cap)]
            for lon, lat, cap in zip(city["s_lon"], city["s_lat"], city["s_cap"])
        ],
        "candidates": [],
        "headline": {
            "population": int(round(planning["population"])),
            "shelters": int(planning["shelters"]),
            "candidates": 0,
            "reach15": round(reach_pct, 1),
            "reachBoth15": round(planning["network_reachable_share"] * 100, 1),
            "capacity": round(capacity_pct, 1),
            "stockLimited": 1,
        },
        "roadBbox": road_bbox,
        "roads": extract_roads(
            GLOBAL_SOURCE / "data" / "external" / f"{city_key}_roads.geojson",
            road_bbox,
            tol_m=tolerance,
        ),
        "modelNetwork": model_network(city),
        "scenarios": {
            "baseline": scenario(f"{city_key}_k000", "零知识基线"),
            "llm": scenario(f"{city_key}_llm000", "LLM 领导引导"),
        },
    }
    road_bbox = [
        float(city["lon"].min()), float(city["lat"].min()),
        float(city["lon"].max()), float(city["lat"].max()),
    ]
    return {
        "cityKey": "noto",
        "cityName": "能登",
        "subtitle": "道路街区与地形阻力 × 6.26 万个体疏散仿真",
        "districtCount": 1,
        "hasCandidates": False,
        "candidatePopulation": "—",
        "officialTerm": "OSM 场所候选",
        "reachLabel": "候选场所15分钟可达",
        "capacityLabel": "情景容量／人口",
        "limitedLabel": "距离受限研究区",
        "callout": "容量为按场所类型推定的情景值，并非日本政府核定容量。总量高于人口不等于能够及时到达；15 分钟网络可达率仅 42.5%，能登首先受距离与地形阻力约束。",
        "methodNote": "人口来自 WorldPop 2020；道路来自 OSM；模型加入 SRTM 坡度与 ESA WorldCover 阻力。容量和场所名单均属于情景假设，不能作为官方应急规划结论。",
        "districts": geo,
        "official": [
            [round(float(lat), 5), round(float(lon), 5), int(cap)]
            for lon, lat, cap in zip(city["s_lon"], city["s_lat"], city["s_cap"])
        ],
        "candidates": [],
        "headline": {
            "population": int(round(planning["population"])),
            "shelters": int(planning["shelters"]),
            "candidates": 0,
            "reach15": round(planning["reach"]["15"] * 100, 1),
            "reachBoth15": round(planning["network_reachable_share"] * 100, 1),
            "capacity": round(planning["capacity_share"] * 100, 1),
            "stockLimited": 1,
        },
        "roadBbox": road_bbox,
        "roads": extract_roads(NOTO_ROAD_SOURCE, road_bbox, tol_m=80.0),
        "modelNetwork": model_network(city),
        "scenarios": {
            "baseline": scenario("noto_k000", "零知识基线"),
            "llm": scenario("noto_llm000", "LLM 领导引导"),
        },
    }


ENGLISH_CITY_NAMES = {
    "chengdu": "Chengdu",
    "xian": "Xi'an",
    "kathmandu": "Kathmandu",
    "l_aquila": "L'Aquila",
    "los_angeles": "Los Angeles",
    "mandalay": "Mandalay",
    "naples": "Naples",
    "noto": "Noto",
    "taipei": "Taipei",
    "wellington": "Wellington",
}


def english_payload(source: dict) -> dict:
    """Return an English presentation layer without changing model data."""
    payload = copy.deepcopy(source)
    key = payload["cityKey"]
    payload["cityName"] = ENGLISH_CITY_NAMES[key]
    population_m = payload["headline"]["population"] / 1e6
    payload["candidatePopulation"] = "15.74M" if key == "chengdu" else "—"
    if key == "chengdu":
        payload.update({
            "subtitle": "Planning diagnosis × 18.87 million-agent evacuation simulation",
            "callout": "Capacity is the first constraint in 17 districts. Candidate sites greatly improve proximity, but accessibility becomes real placement only when capacity is auditable.",
            "methodNote": "42.45% is static road-network accessibility. Dynamic 15-minute safety also depends on capacity, information, and behavior; the two measures are not interchangeable.",
        })
    elif key == "xian":
        payload.update({
            "subtitle": "13-district diagnosis × 12.95 million-agent evacuation simulation",
            "callout": "Shelter capacity is the first constraint in 10 districts; Zhouzhi, Chang'an, and Gaoling are first constrained by 15-minute network distance. Current capacity covers only about 40% of the population.",
            "methodNote": "Static access uses the block-adjacency network and centroid distance. Dynamic safety also depends on capacity, information diffusion, and behavioral response.",
        })
    else:
        capacity = payload["headline"]["capacity"]
        reach = payload["headline"]["reach15"]
        constraint = payload["districts"]["features"][0]["properties"]["constraint"]
        if constraint == "capacity":
            diagnosis = f"Scenario capacity equals {capacity:.1f}% of the study population, below 15-minute network access of {reach:.1f}%; capacity is the first constraint."
        else:
            diagnosis = f"Scenario capacity equals {capacity:.1f}% of the study population, but 15-minute network access is only {reach:.1f}%; distance and terrain resistance are the first constraints."
        payload.update({
            "subtitle": f"Road blocks and terrain resistance × {population_m:.2f} million-agent simulation",
            "officialTerm": "Screened OSM candidate sites",
            "reachLabel": "Candidate sites within 15 minutes",
            "capacityLabel": "Scenario capacity / population",
            "limitedLabel": "Primary constrained study area",
            "callout": "Capacity is a site-type scenario estimate, not a government-certified value. " + diagnosis,
            "methodNote": "Population: WorldPop 2020; roads: OSM; resistance: SRTM slope and ESA WorldCover. Fire hydrants, defibrillators, and other non-shelter facilities were removed. Candidate lists and scenario capacity do not replace official emergency planning.",
        })
        payload["districts"]["features"][0]["properties"]["name"] = f"{payload['cityName']} study area"
    payload["scenarios"]["baseline"]["label"] = "Zero-knowledge baseline"
    payload["scenarios"]["llm"]["label"] = "LLM leadership guidance"
    return payload


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>QuakeSense 应急避难 WebGIS</title>
<link rel="stylesheet" href="leaflet.css">
<style>
:root{--bg:#081018;--panel:#0e1a25;--panel2:#132331;--ink:#edf5f7;--muted:#91a7b3;
--cyan:#42d3c8;--amber:#ffbb55;--red:#ef6a68;--blue:#5fa8ff;--line:#294050}
*{box-sizing:border-box}html,body,#app{height:100%;margin:0}body{font-family:Inter,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink);overflow:hidden}
#app{display:grid;grid-template-rows:72px 1fr}.top{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--line);background:rgba(8,16,24,.97);z-index:1200}
.brand{display:flex;align-items:center;gap:14px}.brandtext h1{font-size:19px;margin:0 0 4px;letter-spacing:.02em}.brandtext p{margin:0;color:var(--muted);font-size:12px}.citynav{width:110px;padding:7px}.langswitch{color:var(--cyan);font-size:12px;text-decoration:none;border:1px solid var(--line);border-radius:7px;padding:7px 9px}.headline{display:flex;gap:22px}.stat b{display:block;font-size:17px;color:var(--cyan)}.stat span{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.main{display:grid;grid-template-columns:340px 1fr;min-height:0}.side{background:var(--panel);border-right:1px solid var(--line);padding:16px;overflow:auto;z-index:1000}.mapwrap{position:relative;min-width:0}#map{height:100%;background:#0a141c}
.tabs{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:14px}.tab,.chip,button,select{border:1px solid var(--line);background:var(--panel2);color:var(--ink);border-radius:7px;padding:8px;cursor:pointer}.tab.on,.chip.on{background:var(--cyan);border-color:var(--cyan);color:#051315;font-weight:700}
.section{border-top:1px solid var(--line);padding-top:14px;margin-top:14px}.section h2{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.chip{font-size:12px}.row{display:flex;gap:8px;align-items:center;margin:8px 0}.row label{font-size:12px;color:var(--muted);flex:1}.row select{width:170px}.check{display:flex;gap:9px;align-items:center;font-size:13px;margin:9px 0}.check input{accent-color:var(--cyan)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px}.card{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px}.card b{font-size:18px;display:block}.card span{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.roadkey{display:inline-block;width:24px;height:3px;border-radius:3px;margin-left:auto}.roadmeta{font-size:10px;color:var(--muted);margin-left:5px}
.playrow{display:grid;grid-template-columns:42px 1fr 52px;gap:8px;align-items:center}.playrow button{font-size:15px}.playrow input{width:100%;accent-color:var(--cyan)}#clock{text-align:right;font-variant-numeric:tabular-nums;color:var(--cyan);font-size:12px}
.legend{height:9px;border-radius:9px;background:linear-gradient(90deg,#10242b,#1aa6a0,#ffe18a,#ef6a68);margin-top:10px}.legendlabels{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;margin-top:4px}.note{font-size:12px;color:var(--muted);line-height:1.55}.callout{border-left:3px solid var(--amber);padding:9px 10px;background:#182331;font-size:12px;line-height:1.5}.hidden{display:none!important}
.leaflet-control-attribution{background:rgba(8,16,24,.8)!important;color:#a9bac2!important}.leaflet-control-attribution a{color:#70cfc8!important}.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:#12222f;color:var(--ink)}
@media(max-width:820px){#app{grid-template-rows:104px 1fr}.top{align-items:flex-start}.headline{gap:9px;flex-wrap:wrap;justify-content:flex-end}.stat:nth-child(n+3){display:none}.main{grid-template-columns:1fr}.side{position:absolute;left:10px;top:114px;width:min(340px,calc(100vw - 20px));max-height:55vh;border:1px solid var(--line);border-radius:10px;z-index:1100;box-shadow:0 14px 40px #0009}.mapwrap{grid-area:1/1}.brand p{max-width:210px}}
</style></head><body><div id="app">
<header class="top"><div class="brand"><div class="brandtext"><h1 id="brandTitle">QuakeSense · 应急避难 WebGIS</h1><p id="brandSubtitle">规划诊断 × 个体疏散仿真</p></div><select id="cityNav" class="citynav" aria-label="切换城市"><option value="index.html">成都</option><option value="xian.html">西安</option><option value="noto.html">能登</option><option value="l_aquila.html">拉奎拉</option><option value="wellington.html">惠灵顿</option><option value="naples.html">那不勒斯</option><option value="mandalay.html">曼德勒</option><option value="kathmandu.html">加德满都</option><option value="taipei.html">台北</option><option value="los_angeles.html">洛杉矶</option></select><a id="langSwitch" class="langswitch" href="#">EN</a></div><div class="headline">
<div class="stat"><b id="hReach">—</b><span id="hReachLabel">官方15分钟可达</span></div><div class="stat"><b id="hBoth">—</b><span id="hBothLabel">候选点加入后</span></div><div class="stat"><b id="hCap">—</b><span id="hCapLabel">总容量覆盖</span></div><div class="stat"><b id="hLimited">—</b><span id="hLimitedLabel">容量受限区县</span></div></div></header>
<main class="main"><aside class="side">
<div class="tabs"><button class="tab on" data-mode="planning">规划诊断</button><button class="tab" data-mode="simulation">动态仿真</button></div>
<div id="planningPanel">
<div class="section" style="margin-top:0;border:0;padding:0"><h2>区县着色指标</h2><div class="grid" id="metricButtons">
<button class="chip on" data-metric="reach15">15分钟可达</button><button class="chip" data-metric="stock">容量覆盖</button><button class="chip" data-metric="constraint">首要瓶颈</button><button class="chip candidateOnly" data-metric="benefit">候选点受益</button></div></div>
<div class="section"><h2>设施图层</h2><label class="check"><input id="officialToggle" type="checkbox" checked><span id="officialLabel">官方避难场所</span></label><label class="check candidateOnly"><input id="candidateToggle" type="checkbox"><span id="candidateLabel">候选 POI</span></label></div>
<div class="section"><div class="cards"><div class="card"><b id="pPopulation">—</b><span>研究人口</span></div><div class="card candidateOnly"><b id="pCandidatePop">—</b><span>候选点更近人口</span></div></div></div>
<div class="section"><div class="callout" id="planningCallout"></div></div>
</div>
<div id="simulationPanel" class="hidden">
<div class="section" style="margin-top:0;border:0;padding:0"><h2>仿真情景</h2><div class="row"><label for="scenario">情景</label><select id="scenario"><option value="baseline">零知识基线</option><option value="llm">LLM 领导引导</option></select></div><div class="row"><label for="simLayer">显示状态</label><select id="simLayer"><option value="transit">在途人群</option><option value="sheltered">已安置</option><option value="gaveup">已放弃</option><option value="known">平均知晓场所数</option><option value="belief">拥挤信念（秒）</option></select></div></div>
<div class="section"><h2>时间回放</h2><div class="playrow"><button id="play" aria-label="播放">▶</button><input id="scrub" type="range" min="0" value="0"><span id="clock">0 min</span></div><div class="legend"></div><div class="legendlabels"><span>低</span><span>高</span></div></div>
<div class="section"><div class="cards"><div class="card"><b id="sTransit">—</b><span>当前在途</span></div><div class="card"><b id="sSheltered">—</b><span>累计安置</span></div><div class="card"><b id="sGaveup">—</b><span>累计放弃</span></div><div class="card"><b id="sSafe15">—</b><span>15分钟安全率</span></div></div></div>
<div class="section"><p class="note" id="scenarioNote"></p></div>
</div>
<div class="section"><h2>道路与计算网络</h2>
<label class="check"><input class="roadToggle" data-road="0" type="checkbox" checked>高速／快速路 <i class="roadkey" style="background:#ffcf66"></i></label>
<label class="check"><input class="roadToggle" data-road="1" type="checkbox" checked>主干路 <i class="roadkey" style="background:#70d8d0"></i><span class="roadmeta">≥9级</span></label>
<label class="check"><input class="roadToggle" data-road="2" type="checkbox" checked>次干路 <i class="roadkey" style="background:#70a9ca"></i><span class="roadmeta">≥10级</span></label>
<label class="check"><input class="roadToggle" data-road="3" type="checkbox">支路／三级路 <i class="roadkey" style="background:#687f90"></i><span class="roadmeta">≥11级</span></label>
<label class="check"><input id="modelNetworkToggle" type="checkbox">模型计算网络 <i class="roadkey" style="background:#e58cc8"></i><span class="roadmeta">≥11级</span></label>
<p class="note" id="roadStatus">道路中心线用于解释城市结构；紫色网络是 agent 实际遍历的街区邻接图。</p></div>
<div class="section"><p class="note" id="methodNote"></p></div>
</aside><div class="mapwrap"><div id="map"></div></div></main></div>
<script src="leaflet.js"></script><script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(async()=>{const P=JSON.parse(document.getElementById('payload').textContent),H=P.headline;
const fmt=n=>Math.round(n).toLocaleString('zh-CN'),pct=n=>(100*n).toFixed(2)+'%';
document.title=`QuakeSense ${P.cityName}应急避难 WebGIS`;document.getElementById('brandTitle').textContent=`QuakeSense · ${P.cityName}应急避难 WebGIS`;document.getElementById('brandSubtitle').textContent=P.subtitle;const cityNav=document.getElementById('cityNav');const cityFile=P.cityKey==='chengdu'?'index.html':P.cityKey+'.html';cityNav.value=cityFile;cityNav.onchange=e=>location.href=e.target.value;document.getElementById('langSwitch').href='../'+cityFile;
document.getElementById('hReach').textContent=H.reach15+'%';document.getElementById('hReachLabel').textContent=P.reachLabel||'官方15分钟可达';document.getElementById('hBoth').textContent=H.reachBoth15+'%';document.getElementById('hBothLabel').textContent=P.hasCandidates?'候选点加入后':'路网覆盖人口';document.getElementById('hCap').textContent=H.capacity+'%';document.getElementById('hCapLabel').textContent=P.capacityLabel||'总容量覆盖';document.getElementById('hLimited').textContent=H.stockLimited+'/'+P.districtCount;document.getElementById('hLimitedLabel').textContent=P.limitedLabel||'容量受限区县';document.getElementById('pPopulation').textContent=(H.population/1e4).toFixed(0)+'万';document.getElementById('pCandidatePop').textContent=P.candidatePopulation;document.getElementById('officialLabel').textContent=`${P.officialTerm||'官方避难场所'}（${fmt(H.shelters)}）`;document.getElementById('candidateLabel').textContent=`候选 POI（${fmt(H.candidates)}）`;document.getElementById('planningCallout').textContent=P.callout;document.getElementById('methodNote').textContent=P.methodNote;if(!P.hasCandidates)document.querySelectorAll('.candidateOnly').forEach(x=>x.classList.add('hidden'));
const map=L.map('map',{preferCanvas:true,zoomControl:false}).setView([30.67,104.07],9);L.control.zoom({position:'bottomright'}).addTo(map);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:18,attribution:'&copy; OpenStreetMap &copy; CARTO'}).addTo(map);
const ramp=v=>{v=Math.max(0,Math.min(100,+v||0))/100;return `hsl(${180-180*v} 72% ${32+18*v}%)`};let metric='reach15';
function districtStyle(f){const p=f.properties;if(metric==='constraint')return{color:'#7090a0',weight:1,fillOpacity:.56,fillColor:p.constraint==='capacity'?'#ef6a68':'#5fa8ff'};return{color:'#66818f',weight:1,fillOpacity:.58,fillColor:ramp(p[metric])}}
const districts=L.geoJSON(P.districts,{style:districtStyle,onEachFeature:(f,l)=>{const p=f.properties,benefit=P.hasCandidates?`<br>候选点受益：${p.benefit}%`:'';l.bindPopup(`<b>${p.name}</b><br>人口：${fmt(p.population)}<br>15分钟可达：${p.reach15}%<br>容量覆盖：${p.stock}%${benefit}<br>首要瓶颈：${p.constraint==='capacity'?'容量':'距离'}`)}}).addTo(map);map.fitBounds(districts.getBounds(),{padding:[12,12]});
const canvas=L.canvas({padding:.5});const official=L.layerGroup(P.official.map(x=>L.circleMarker([x[0],x[1]],{renderer:canvas,radius:2.4,weight:0,fillOpacity:.8,fillColor:'#ffbb55'}).bindTooltip(`${P.officialTerm||'官方场所'} · 容量 ${fmt(x[2])}`))).addTo(map);
const kindName={park:'公园',square:'广场',school:'学校',sports:'体育场馆'};const candidates=L.layerGroup(P.candidates.map(x=>L.circleMarker([x[0],x[1]],{renderer:canvas,radius:2,weight:0,fillOpacity:.72,fillColor:'#42d3c8'}).bindTooltip(`候选点 · ${kindName[x[2]]||x[2]}`)));
document.getElementById('officialToggle').onchange=e=>e.target.checked?official.addTo(map):map.removeLayer(official);document.getElementById('candidateToggle').onchange=e=>e.target.checked?candidates.addTo(map):map.removeLayer(candidates);
document.querySelectorAll('[data-metric]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-metric]').forEach(x=>x.classList.remove('on'));b.classList.add('on');metric=b.dataset.metric;districts.setStyle(districtStyle)});
const b64=s=>{const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a};
async function inflateRaw(s){const ds=new DecompressionStream('deflate');return await new Response(new Blob([b64(s)]).stream().pipeThrough(ds)).arrayBuffer()}
async function unpack(s){return new Float32Array(await inflateRaw(s))}
async function decodeRoadTier(encoded){if(!encoded)return[];const a=new Int16Array(await inflateRaw(encoded)),[loLon,loLat,hiLon,hiLat]=P.roadBbox,lines=[];let i=0;while(i<a.length){const n=a[i++];if(n<2||i+2>a.length)break;let u=a[i++]+32768,v=a[i++]+32768;const out=new Float32Array(n*2);let minLon=999,minLat=999,maxLon=-999,maxLat=-999;for(let k=0;k<n;k++){if(k){u+=a[i++];v+=a[i++]}const lon=loLon+u/65535*(hiLon-loLon),lat=loLat+v/65535*(hiLat-loLat);out[k*2]=lon;out[k*2+1]=lat;minLon=Math.min(minLon,lon);maxLon=Math.max(maxLon,lon);minLat=Math.min(minLat,lat);maxLat=Math.max(maxLat,lat)}lines.push({p:out,b:[minLon,minLat,maxLon,maxLat]})}return lines}
const roadTiers={};for(const key of Object.keys(P.roads))roadTiers[key]=await decodeRoadTier(P.roads[key]);const modelEdges=await unpack(P.modelNetwork.edges);
const roadCanvas=document.createElement('canvas');roadCanvas.setAttribute('aria-hidden','true');Object.assign(roadCanvas.style,{position:'absolute',inset:'0',zIndex:'410',pointerEvents:'none'});map.getContainer().appendChild(roadCanvas);
const roadState={0:true,1:true,2:true,3:false,model:false};const roadStyle={0:['#ffcf66',1.8,.82,0],1:['#70d8d0',1.25,.68,9],2:['#70a9ca',.9,.55,10],3:['#687f90',.65,.42,11]};
function drawRoads(){const size=map.getSize(),dpr=Math.min(window.devicePixelRatio||1,2);roadCanvas.width=Math.round(size.x*dpr);roadCanvas.height=Math.round(size.y*dpr);roadCanvas.style.width=size.x+'px';roadCanvas.style.height=size.y+'px';const ctx=roadCanvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,size.x,size.y);const z=map.getZoom(),bb=map.getBounds().pad(.08),west=bb.getWest(),east=bb.getEast(),south=bb.getSouth(),north=bb.getNorth();ctx.lineCap='round';ctx.lineJoin='round';for(const key of ['3','2','1','0']){const st=roadStyle[key];if(!roadState[key]||z<st[3])continue;ctx.beginPath();for(const line of roadTiers[key]){const b=line.b;if(b[2]<west||b[0]>east||b[3]<south||b[1]>north)continue;const p=line.p;for(let i=0;i<p.length;i+=2){const q=map.latLngToContainerPoint([p[i+1],p[i]]);if(i===0)ctx.moveTo(q.x,q.y);else ctx.lineTo(q.x,q.y)}}ctx.strokeStyle=st[0];ctx.globalAlpha=st[2];ctx.lineWidth=st[1];ctx.stroke()}if(roadState.model&&z>=11){ctx.beginPath();const step=z<12?4:z<13?2:1;for(let i=0;i<modelEdges.length;i+=4*step){const lon1=modelEdges[i],lat1=modelEdges[i+1],lon2=modelEdges[i+2],lat2=modelEdges[i+3];if((lon1<west&&lon2<west)||(lon1>east&&lon2>east)||(lat1<south&&lat2<south)||(lat1>north&&lat2>north))continue;const a=map.latLngToContainerPoint([lat1,lon1]),b=map.latLngToContainerPoint([lat2,lon2]);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y)}ctx.strokeStyle='#e58cc8';ctx.globalAlpha=.52;ctx.lineWidth=.65;ctx.stroke()}ctx.globalAlpha=1;const hidden=[];if(z<9&&roadState[1])hidden.push('主干路');if(z<10&&roadState[2])hidden.push('次干路');if(z<11&&(roadState[3]||roadState.model))hidden.push('支路/计算网络');document.getElementById('roadStatus').textContent=hidden.length?`当前缩放 ${z}：${hidden.join('、')}将在继续放大后显示。`:`当前缩放 ${z}：已显示所选道路层级；紫色为 ${P.modelNetwork.count.toLocaleString()} 条模型邻接边。`}
document.querySelectorAll('.roadToggle').forEach(x=>x.onchange=()=>{roadState[x.dataset.road]=x.checked;drawRoads()});document.getElementById('modelNetworkToggle').onchange=e=>{roadState.model=e.target.checked;drawRoads()};map.on('moveend zoomend resize',drawRoads);drawRoads();
const decoded={};async function loadScenario(key){if(decoded[key])return decoded[key];const s=P.scenarios[key],d={...s,lon:await unpack(s.lon),lat:await unpack(s.lat),layers:{}};for(const k of Object.keys(s.layers))d.layers[k]=await unpack(s.layers[k]);decoded[key]=d;return d}
let simGroup=L.layerGroup(),markers=[],scenarioKey='baseline',layerKey='transit',frame=0,timer=null,current=null;
const palette=v=>{const x=Math.max(0,Math.min(1,v));return `hsl(${180-180*x} 78% ${38+15*x}%)`};
async function resetScenario(){current=await loadScenario(scenarioKey);frame=0;document.getElementById('scrub').max=current.nframes-1;document.getElementById('scrub').value=0;simGroup.clearLayers();markers=[];for(let i=0;i<current.ncells;i++){const m=L.circleMarker([current.lat[i],current.lon[i]],{renderer:canvas,radius:2,weight:0,fillOpacity:.65,fillColor:'#42d3c8'});markers.push(m);simGroup.addLayer(m)}drawFrame()}
function drawFrame(){if(!current)return;const a=current.layers[layerKey],off=frame*current.ncells;let max=0;for(let i=0;i<current.ncells;i++)max=Math.max(max,a[off+i]);const denom=Math.max(max,1);for(let i=0;i<markers.length;i++){const v=a[off+i],q=Math.sqrt(Math.max(v,0)/denom);markers[i].setStyle({fillColor:palette(q),fillOpacity:v>0?.18+.72*q:0});markers[i].setRadius(v>0?1.6+5*q:1)}const t=current.time[frame];document.getElementById('clock').textContent=Math.round(t)+' min';document.getElementById('sTransit').textContent=fmt(current.totals.transit[frame]);document.getElementById('sSheltered').textContent=fmt(current.totals.sheltered[frame]);document.getElementById('sGaveup').textContent=fmt(current.totals.gaveup[frame]);document.getElementById('sSafe15').textContent=pct(current.summary.safe15);document.getElementById('scenarioNote').textContent=`${current.label}：最终安置 ${fmt(current.summary.sheltered)} 人，120 分钟安全率 ${pct(current.summary.safe120)}，场所占用基尼 ${current.summary.gini.toFixed(3)}。`}
document.getElementById('scenario').onchange=async e=>{scenarioKey=e.target.value;await resetScenario()};document.getElementById('simLayer').onchange=e=>{layerKey=e.target.value;drawFrame()};document.getElementById('scrub').oninput=e=>{frame=+e.target.value;drawFrame()};document.getElementById('play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='▶'}else{e.target.textContent='❚❚';timer=setInterval(()=>{frame=(frame+1)%current.nframes;document.getElementById('scrub').value=frame;drawFrame()},420)}};
async function setMode(mode){const planning=mode==='planning';document.getElementById('planningPanel').classList.toggle('hidden',!planning);document.getElementById('simulationPanel').classList.toggle('hidden',planning);document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.mode===mode));if(planning){map.removeLayer(simGroup);districts.addTo(map);if(document.getElementById('officialToggle').checked)official.addTo(map);if(document.getElementById('candidateToggle').checked)candidates.addTo(map)}else{map.removeLayer(districts);map.removeLayer(candidates);official.addTo(map);simGroup.addTo(map);if(!current)await resetScenario()}setTimeout(()=>map.invalidateSize(),20)}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));window.addEventListener('resize',()=>map.invalidateSize());})();
</script></body></html>'''


ENGLISH_HTML_REPLACEMENTS = [
    ('lang="zh-CN"', 'lang="en"'),
    ('<title>QuakeSense 应急避难 WebGIS</title>', '<title>QuakeSense Emergency Evacuation WebGIS</title>'),
    ('.citynav{width:110px', '.citynav{width:155px'),
    ('QuakeSense · 应急避难 WebGIS', 'QuakeSense · Emergency Evacuation WebGIS'),
    ('规划诊断 × 个体疏散仿真', 'Planning diagnosis × agent-based evacuation simulation'),
    ('aria-label="切换城市"', 'aria-label="Switch city"'),
    ('<option value="index.html">成都</option>', '<option value="index.html">Chengdu</option>'),
    ('<option value="xian.html">西安</option>', '<option value="xian.html">Xi\'an</option>'),
    ('<option value="noto.html">能登</option>', '<option value="noto.html">Noto</option>'),
    ('<option value="l_aquila.html">拉奎拉</option>', '<option value="l_aquila.html">L\'Aquila</option>'),
    ('<option value="wellington.html">惠灵顿</option>', '<option value="wellington.html">Wellington</option>'),
    ('<option value="naples.html">那不勒斯</option>', '<option value="naples.html">Naples</option>'),
    ('<option value="mandalay.html">曼德勒</option>', '<option value="mandalay.html">Mandalay</option>'),
    ('<option value="kathmandu.html">加德满都</option>', '<option value="kathmandu.html">Kathmandu</option>'),
    ('<option value="taipei.html">台北</option>', '<option value="taipei.html">Taipei</option>'),
    ('<option value="los_angeles.html">洛杉矶</option>', '<option value="los_angeles.html">Los Angeles</option>'),
    ('>EN</a>', '>中文</a>'),
    ('官方15分钟可达', 'Official sites within 15 min'),
    ('候选点加入后', 'With candidate sites'),
    ('总容量覆盖', 'Total capacity coverage'),
    ('容量受限区县', 'Capacity-limited districts'),
    ('规划诊断', 'Planning diagnosis'),
    ('动态仿真', 'Dynamic simulation'),
    ('区县着色指标', 'District color metric'),
    ('15分钟可达', '15-minute access'),
    ('容量覆盖', 'Capacity coverage'),
    ('首要瓶颈', 'Primary bottleneck'),
    ('候选点受益', 'Candidate-site benefit'),
    ('设施图层', 'Facility layers'),
    ('官方避难场所', 'Official shelters'),
    ('候选 POI', 'Candidate POIs'),
    ('研究人口', 'Study population'),
    ('候选点更近人口', 'Population closer to candidates'),
    ('仿真情景', 'Simulation scenario'),
    ('<label for="scenario">情景</label>', '<label for="scenario">Scenario</label>'),
    ('零知识基线', 'Zero-knowledge baseline'),
    ('LLM 领导引导', 'LLM leadership guidance'),
    ('显示状态', 'Displayed state'),
    ('在途人群', 'In transit'),
    ('已安置', 'Placed'),
    ('已放弃', 'Abandoned'),
    ('平均知晓场所数', 'Mean known sites'),
    ('拥挤信念（秒）', 'Perceived congestion (s)'),
    ('时间回放', 'Timeline playback'),
    ('aria-label="播放"', 'aria-label="Play"'),
    ('>低</span><span>高<', '>Low</span><span>High<'),
    ('当前在途', 'Currently in transit'),
    ('累计安置', 'Cumulative placement'),
    ('累计放弃', 'Cumulative abandonment'),
    ('15分钟安全率', '15-minute safety'),
    ('道路与计算网络', 'Roads and computational network'),
    ('高速／快速路', 'Motorways / expressways'),
    ('主干路', 'Primary roads'),
    ('次干路', 'Secondary roads'),
    ('支路／三级路', 'Local / tertiary roads'),
    ('模型计算网络', 'Model computational network'),
    ('≥9级', 'zoom ≥9'),
    ('≥10级', 'zoom ≥10'),
    ('≥11级', 'zoom ≥11'),
    ('道路中心线用于解释城市结构；紫色网络是 agent 实际遍历的街区邻接图。', 'Road centerlines explain urban structure; the purple network is the block-adjacency graph traversed by agents.'),
    ("toLocaleString('zh-CN')", "toLocaleString('en-US')"),
    ('${P.cityName}应急避难 WebGIS', '${P.cityName} Emergency Evacuation WebGIS'),
    ("`QuakeSense ${P.cityName}应急避难 WebGIS`", "`QuakeSense ${P.cityName} Emergency Evacuation WebGIS`"),
    ("href='../'+cityFile", "href='zh/'+cityFile"),
    ("(H.population/1e4).toFixed(0)+'万'", "(H.population/1e6).toFixed(2)+'M'"),
    ("P.reachLabel||'Official 15-minute access'", "P.reachLabel||'Official sites within 15 min'"),
    ("P.hasCandidates?'With candidate sites':'路网覆盖人口'", "P.hasCandidates?'With candidate sites':'Population covered by network'"),
    ("P.capacityLabel||'Total capacity coverage'", "P.capacityLabel||'Total capacity coverage'"),
    ("P.limitedLabel||'Capacity-limited districts'", "P.limitedLabel||'Capacity-limited districts'"),
    ("P.officialTerm||'Official shelters'", "P.officialTerm||'Official shelters'"),
    ('候选点 · ', 'Candidate site · '),
    ('公园', 'Park'),
    ('广场', 'Square'),
    ('学校', 'School'),
    ('体育场馆', 'Sports facility'),
    ('人口：', 'Population: '),
    ('候选点受益：', 'Candidate-site benefit: '),
    ('首要瓶颈：', 'Primary bottleneck: '),
    ("?'容量':'距离'", "?'Capacity':'Distance'"),
    ('官方场所', 'Official site'),
    ('容量 ', 'Capacity '),
    ("hidden.push('Primary roads')", "hidden.push('primary roads')"),
    ("hidden.push('Secondary roads')", "hidden.push('secondary roads')"),
    ("hidden.push('支路/计算网络')", "hidden.push('local / model network')"),
    ('当前缩放 ${z}：${hidden.join(\'、\')}将在继续放大后显示。', 'Zoom ${z}: ${hidden.join(\', \')} appear after zooming in.'),
    ('当前缩放 ${z}：已显示所选道路层级；紫色为 ${P.modelNetwork.count.toLocaleString()} 条模型邻接边。', 'Zoom ${z}: selected road tiers are visible; purple shows ${P.modelNetwork.count.toLocaleString()} model adjacency edges.'),
    ('${current.label}：最终安置 ${fmt(current.summary.sheltered)} 人，120 分钟安全率 ${pct(current.summary.safe120)}，场所占用基尼 ${current.summary.gini.toFixed(3)}。', '${current.label}: ${fmt(current.summary.sheltered)} people ultimately placed; 120-minute safety ${pct(current.summary.safe120)}; shelter occupancy Gini ${current.summary.gini.toFixed(3)}.'),
    ('（', ' ('),
    ('）', ')'),
]


def english_html() -> str:
    html = HTML
    for source, target in ENGLISH_HTML_REPLACEMENTS:
        html = html.replace(source, target)
    return html


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zh_out = OUT / "zh"
    zh_out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "index.html": build_chengdu_payload(),
        "xian.html": build_xian_payload(),
        **{
            f"{city_key}.html": build_global_payload(city_key)
            for city_key in GLOBAL_CITY_META
        },
    }
    for filename, payload in payloads.items():
        html_zh = HTML.replace(
            "__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        zh_target = zh_out / filename
        zh_target.write_text(html_zh, encoding="utf-8")
        en_payload = english_payload(payload)
        html_en = english_html().replace(
            "__PAYLOAD__", json.dumps(en_payload, ensure_ascii=False, separators=(",", ":"))
        )
        target = OUT / filename
        target.write_text(html_en, encoding="utf-8")
        print(f"wrote {target} [EN] ({target.stat().st_size / 1024**2:.2f} MiB)")
        print(f"wrote {zh_target} [ZH] ({zh_target.stat().st_size / 1024**2:.2f} MiB)")
        print(
            f"  districts={len(payload['districts']['features'])} "
            f"official={len(payload['official'])} candidates={len(payload['candidates'])}"
        )
        print("  road tiers=" + ", ".join(
            f"{key}:{len(value) // 1024}KiB" for key, value in payload["roads"].items()
        ))
        print(f"  model adjacency edges={payload['modelNetwork']['count']:,}")
        for key, value in payload["scenarios"].items():
            print(f"  {key}: frames={value['nframes']} cells={value['ncells']} run={value['run']}")
    for asset in ("leaflet.js", "leaflet.css"):
        (OUT / asset).write_bytes((PROJECT / "portal" / "lib" / asset).read_bytes())
        (zh_out / asset).write_bytes((PROJECT / "portal" / "lib" / asset).read_bytes())


if __name__ == "__main__":
    main()
