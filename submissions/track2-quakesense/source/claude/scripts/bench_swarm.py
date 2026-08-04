"""P0-P4: measure the full-population swarm on real Chengdu data.

Replaces every "估算" in docs/ROADMAP_FULL_SCALE_AGENTS.md with a measurement:
per-kernel time, peak VRAM, conservation, shelter overfill, event volume, and
the wall clock of the LLM leader tier.

    python -m scripts.bench_swarm --steps 1080 --llm

Block adjacency and the parsed layer are cached: parsing 82,766 GeoJSON
features and hashing their shared edges costs more than the simulation it
feeds, and is identical every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import orjson as _fastjson

    def _load(p):
        return _fastjson.loads(Path(p).read_bytes())
except ImportError:
    def _load(p):
        with open(p, "rb") as fh:
            return json.load(fh)

import torch

from simulator.agent_swarm import AgentSwarm, SwarmConfig
from simulator.block_graph import build_adjacency
from simulator.block_routing import cache_routing, locate_shelter_blocks
from simulator.swarm_step import SwarmStepper, K_BELIEF, TRAJ_DEPTH


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _Layer:
    pass


class _Shelter:
    __slots__ = ("lon", "lat", "capacity")

    def __init__(self, lon, lat, cap):
        self.lon, self.lat, self.capacity = lon, lat, cap


def load_city(base: Path, cache: Path, max_degree: int):
    # Every city lays its files out as <name>_blocks.geojson alongside
    # <name>_block_evacuation.json, so the stem comes from the directory
    # rather than being hard-coded to Chengdu.
    stem = base.name
    if cache.exists():
        z = np.load(cache)
        log(f"cache hit {cache.name}")
    else:
        t0 = time.perf_counter()
        geo = _load(base / f"{stem}_blocks.geojson")
        feats = geo["features"]
        log(f"parsed {len(feats):,} features in {time.perf_counter()-t0:.1f} s")

        props = [f["properties"] for f in feats]
        g = lambda key, dflt=None: np.fromiter(
            ((p.get(key, dflt) if dflt is not None else p[key]) for p in props),
            dtype=np.float64, count=len(props))
        lon, lat, pop = g("lon"), g("lat"), g("population")
        area = np.fromiter(
            (float(p.get("area_m2") or p.get("area") or 40_000.0) for p in props),
            dtype=np.float64, count=len(props))

        t0 = time.perf_counter()
        neigh, deg = build_adjacency(feats, max_degree=max_degree)
        log(f"built adjacency in {time.perf_counter()-t0:.1f} s  "
            f"mean degree {deg.mean():.2f}  isolated {int((deg==0).sum()):,}")

        sim = _load(base / f"{stem}_block_evacuation.json")
        sh = sim["shelters"]
        s_lon = np.fromiter((s["lon"] for s in sh), dtype=np.float64, count=len(sh))
        s_lat = np.fromiter((s["lat"] for s in sh), dtype=np.float64, count=len(sh))
        s_cap = np.fromiter((float(s.get("capacity", 0.0)) for s in sh),
                            dtype=np.float64, count=len(sh))

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lon=lon, lat=lat, pop=pop, area=area, neigh=neigh, deg=deg,
                 s_lon=s_lon, s_lat=s_lat, s_cap=s_cap)
        z = np.load(cache)
        log(f"cached to {cache}")

    layer = _Layer()
    layer.lon, layer.lat = z["lon"], z["lat"]
    layer.population, layer.area_m2 = z["pop"], z["area"]
    # Measured open-ground share, when the cache carries one. Absent it the
    # density denominator stays the whole block polygon.
    layer.open_share = z["open_share"] if "open_share" in z.files else None
    shelters = [_Shelter(a, b, c) for a, b, c in zip(z["s_lon"], z["s_lat"], z["s_cap"])]
    return layer, shelters, z["neigh"], z["deg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/xichang-agentic-evacuation/"
                                      "claude/portal/data/chengdu_admin")
    ap.add_argument("--cache", default="/data/quakesense/cache/chengdu_p2.npz")
    ap.add_argument("--run-dir", default="/data/quakesense/runs/p2")
    ap.add_argument("--route-cache",
                    default="/data/quakesense/cache/chengdu_routing.npz")
    ap.add_argument("--route-workers", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1080)
    ap.add_argument("--profile-every", type=int, default=90)
    ap.add_argument("--profile-after", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-degree", type=int, default=12)
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--verify-at", default="",
                    help="Comma-separated steps at which to dump the live state "
                         "vector, for scripts/replay_verify.py.")
    ap.add_argument("--frames-every", type=int, default=0,
                    help="Dump per-block aggregates every N steps for the map.")
    ap.add_argument("--llm", action="store_true", help="Run the LLM leader tier.")
    ap.add_argument("--llm-slots", type=int, default=32)
    ap.add_argument("--n-subdistrict", type=int, default=0,
                    help="Override the LLM-driven subdistrict count (0 = config).")
    ap.add_argument("--n-district", type=int, default=0)
    ap.add_argument("--n-llm-community", type=int, default=0,
                    help="LLM-driven community leaders on top of the "
                         "350 district+subdistrict ones.")
    ap.add_argument("--tag", default="", help="Label recorded in summary.json.")
    # The communication topology, one knob per channel. These are the
    # experiment: how far information travels decides whether a leader tier is
    # worth having at all.
    ap.add_argument("--frac-knows-one", type=float, default=-1.0,
                    help="Share who know their nearest shelter at t=0.")
    ap.add_argument("--word-of-mouth", type=float, default=-1.0,
                    help="Agent-to-agent: chance per step of picking up what "
                         "the block knows.")
    ap.add_argument("--knowledge-spread", type=float, default=-1.0,
                    help="Block-to-block spread of what is common knowledge.")
    ap.add_argument("--broadcast-reach", type=float, default=-1.0,
                    help="Leader-to-public: share who act on a broadcast.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="Random seed. Every configuration so far has been run "
                         "once, so no result has an error bar; replicates are "
                         "what turn a difference into a finding.")
    ap.add_argument("--road-damage", type=float, default=0.0,
                    help="Fraction of block-to-block links severed by the quake. "
                         "Routing is rebuilt from the damaged graph, so detours "
                         "and cut-off blocks are computed, not assumed.")
    ap.add_argument("--trail-decay", type=float, default=-1.0,
                    help="Per-step decay of the trail the lost follow. 0 makes "
                         "it an instantaneous headcount, which leaves nothing "
                         "behind for anyone to follow.")
    ap.add_argument("--bellwether", type=float, default=-1.0,
                    help="How many walkers a moving leader is worth in the "
                         "field lost people follow. 1 = no bellwether effect.")
    ap.add_argument("--leader-comm", type=float, default=0.5,
                    help="Leader-to-leader: share of peers reached per round.")
    ap.add_argument("--llm-interval-min", type=float, default=10.0)
    ap.add_argument("--vram-fraction", type=float, default=0.0,
                    help="Cap this process's share of VRAM (0 = uncapped). The "
                         "simulation and llama.cpp share one device; leaving "
                         "torch uncapped lets it take 43 of 48 GB.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(os.cpu_count() or 32)
    if args.vram_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, 0)
        cap = torch.cuda.get_device_properties(0).total_memory * args.vram_fraction
        log(f"VRAM capped at {args.vram_fraction:.0%} = {cap/1024**3:.1f} GiB "
            f"so llama.cpp keeps room for its compute buffers")
    log(f"torch {torch.__version__}  cpu_threads={torch.get_num_threads()}  "
        f"cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"device: {p.name}  total {p.total_memory/1024**3:.1f} GiB")

    layer, shelters, neigh, deg = load_city(Path(args.base), Path(args.cache),
                                            args.max_degree)
    log(f"blocks={len(layer.population):,}  pop={layer.population.sum():,.0f}  "
        f"shelters={len(shelters):,}  capacity={sum(s.capacity for s in shelters):,.0f}")
    log(f"adjacency: mean degree {deg.mean():.2f}  isolated blocks {int((deg==0).sum()):,}")

    t0 = time.perf_counter()
    scfg = SwarmConfig(device="cpu", n_candidates=32)
    if args.n_subdistrict:
        scfg.n_subdistrict = args.n_subdistrict
    if args.n_district:
        scfg.n_district = args.n_district
    if args.seed >= 0:
        scfg.seed = args.seed
    if args.n_llm_community:
        scfg.n_llm_community = args.n_llm_community
    for name in ("frac_knows_one", "word_of_mouth", "knowledge_spread",
                 "broadcast_reach", "bellwether_weight", "trail_decay"):
        v = getattr(args, "bellwether" if name == "bellwether_weight" else name)
        if v >= 0.0:
            setattr(scfg, name, v)
    log(f"communication: knows_one={scfg.frac_knows_one} "
        f"word_of_mouth={scfg.word_of_mouth} spread={scfg.knowledge_spread} "
        f"broadcast={scfg.broadcast_reach} leader_comm={args.leader_comm}")
    swarm = AgentSwarm(layer, shelters, routing=None, config=scfg)
    log(f"materialised {swarm.n_agents:,} agents (host) in {time.perf_counter()-t0:.1f} s")

    import math as _m
    kx = 111_320.0 * _m.cos(_m.radians(float(np.mean(layer.lat))))
    bx = ((layer.lon - layer.lon.mean()) * kx).astype(np.float64)
    by = ((layer.lat - layer.lat.mean()) * 110_540.0).astype(np.float64)
    if args.road_damage > 0:
        from simulator.block_graph import damage_edges
        neigh, n_cut = damage_edges(neigh, args.road_damage, seed=scfg.seed,
                                    max_degree=neigh.shape[1], log=log)
        # A damaged network has different shortest paths, so the cached routing
        # tables are wrong by construction. Rebuild rather than reuse.
        args.route_cache = (f"/data/quakesense/cache/route_dmg"
                            f"{args.road_damage:.2f}.npz")
    s_blocks = locate_shelter_blocks([s.lon for s in shelters],
                                     [s.lat for s in shelters],
                                     layer.lon, layer.lat)
    route_dist, route_pred = cache_routing(Path(args.route_cache), neigh, bx, by,
                                           s_blocks, device=args.device, log=log)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    st = SwarmStepper(swarm, device=args.device, neigh=neigh,
                      route_dist=route_dist, route_pred=route_pred,
                      event_dir=None if args.no_events else run_dir / "events")
    torch.cuda.synchronize()
    alloc_s = time.perf_counter() - t0
    sb = st.state_bytes()
    log(f"device state allocated in {alloc_s:.1f} s  "
        f"{sb['total_bytes_per_agent']} B/agent = {sb['total_gb']} GB")
    log(f"  torch reserved {torch.cuda.memory_reserved()/1024**3:.2f} GiB")

    tier = None
    if args.llm:
        from simulator.llm_leaders import LeaderTier
        tier = LeaderTier(st, swarm, n_slots=args.llm_slots,
                          interval_min=args.llm_interval_min,
                          leader_comm=args.leader_comm,
                          log_dir=run_dir / "llm_context")
        log(f"LLM tier: {tier.n_leaders} leaders, {args.llm_slots} slots, "
            f"every {args.llm_interval_min} sim-min")

    for _ in range(args.warmup):
        st.step()
    torch.cuda.synchronize()
    st.timings.clear()
    log(f"warmed {args.warmup} steps; clock starts at t={st.t/60:.1f} min")

    verify_at = {int(x) for x in args.verify_at.split(",") if x.strip()}
    run_t0 = time.perf_counter()
    prof_steps, trace, llm_rounds, frames = 0, [], [], []
    for i in range(args.steps):
        profile = (args.profile_every and i >= args.profile_after
                   and i % args.profile_every == 0)
        st.step(profile=profile)
        prof_steps += int(bool(profile))
        if args.frames_every and i % args.frames_every == 0:
            frames.append(st.frame())
        if i in verify_at:
            # Flush first: an event still sitting in the device buffer has not
            # reached the stream, so replaying without it would fail for a
            # reason that has nothing to do with the stream's completeness.
            st._flush_events()
            import numpy as _np
            _np.save(run_dir / f"state_{i:05d}.npy", st.state.cpu().numpy())
            log(f"  verification checkpoint at step {i}")
        if tier is not None and tier.due():
            r = tier.run_round()
            llm_rounds.append(r)
            log(f"  LLM round {r['round']:>2}  {r.get('calls',0):>4} calls  "
                f"{r.get('seconds',0):>7.2f} s  {r.get('per_call_s',0):>6.3f} s/call")
        if profile and i and i % (args.profile_every * 4) == 0:
            tt = st.totals()
            log(f"  t={tt['t_minutes']:>6.1f} min  indoors={tt['indoors']:>12,}  "
                f"transit={tt['in_transit']:>11,}  sheltered={tt['sheltered']:>10,}")
            trace.append(tt)
    torch.cuda.synchronize()
    run_s = time.perf_counter() - run_t0
    st.finish()

    tot = st.totals()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    llm_s = tier.llm_seconds if tier else 0.0

    log("")
    log("=" * 74)
    log(f"{args.steps} steps in {run_s:.2f} s  =  {run_s/args.steps*1000:.2f} ms/step")
    if tier:
        log(f"  of which LLM tier {llm_s:.1f} s ({llm_s/max(run_s,1e-9)*100:.0f}%)  "
            f"-> GPU-only {(run_s-llm_s)/args.steps*1000:.2f} ms/step")
    log(f"peak VRAM {peak:.2f} GiB")
    log("=" * 74)
    kern = {}
    for k, v in sorted(st.timings.items()):
        ms = v / max(prof_steps, 1) * 1000
        kern[k] = round(ms, 3)
        log(f"  {k:<14}{ms:>9.3f} ms")
    log("")
    for k, v in tot.items():
        log(f"  {k:<26}{v}")
    if tier:
        log("")
        for k, v in tier.report().items():
            log(f"  {k:<26}{v}")

    if tot["conservation_error"]:
        log(f"!! CONSERVATION VIOLATED by {tot['conservation_error']}")
    if tot["overfilled_shelters"]:
        log(f"!! {tot['overfilled_shelters']} shelters over capacity")

    ev_dir = run_dir / "events"
    ev_bytes = sum(f.stat().st_size for f in ev_dir.glob("*.npy")) if ev_dir.exists() else 0
    out = run_dir / "summary.json"
    out.write_text(json.dumps({
        "n_agents": st.n, "n_blocks": st.nb, "n_shelters": st.ns,
        "adjacency_mean_degree": float(deg.mean()),
        "adjacency_isolated": int((deg == 0).sum()),
        "steps": args.steps, "wall_s": round(run_s, 3),
        "ms_per_step": round(run_s / args.steps * 1000, 3),
        "llm_wall_s": round(llm_s, 2),
        "gpu_only_ms_per_step": round((run_s - llm_s) / args.steps * 1000, 3),
        "alloc_s": round(alloc_s, 2), "peak_vram_gib": round(peak, 3),
        "state_bytes": sb, "kernels_ms": kern,
        "K_belief": K_BELIEF, "traj_depth": TRAJ_DEPTH,
        "event_bytes": ev_bytes, "event_mb": round(ev_bytes / 1024 ** 2, 1),
        "totals": tot, "trace": trace,
        "llm": tier.report() if tier else None, "llm_rounds": llm_rounds,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "tag": args.tag,
        "llm_enabled": bool(args.llm),
        "n_llm_leaders_configured": int(len(swarm.llm_ids)),
        "broadcast_reach": scfg.broadcast_reach,
        "frac_knows_one": scfg.frac_knows_one,
        "word_of_mouth": scfg.word_of_mouth,
        "knowledge_spread": scfg.knowledge_spread,
        "leader_comm": args.leader_comm,
        "bellwether_weight": scfg.bellwether_weight,
        "trail_decay": scfg.trail_decay,
        "road_damage": args.road_damage,
        "seed": scfg.seed,
        "search_speed_factor": scfg.search_speed_factor,
    }, indent=2), encoding="utf-8")
    if frames:
        import numpy as _np
        fp = run_dir / "frames.npz"
        _np.savez_compressed(
            fp,
            t_min=_np.array([f["t_min"] for f in frames], dtype=_np.float32),
            transit=_np.stack([f["transit"] for f in frames]),
            sheltered=_np.stack([f["sheltered"] for f in frames]),
            gaveup=_np.stack([f["gaveup"] for f in frames]),
            belief=_np.stack([f["belief"] for f in frames]),
            # The layer the whole argument is about. frame() has emitted it
            # since the knowledge model went in, but this list is explicit, so
            # it was silently dropped on the way to disk.
            known=_np.stack([f["known"] for f in frames]),
            lon=layer.lon.astype(_np.float32), lat=layer.lat.astype(_np.float32),
            pop=layer.population.astype(_np.float32),
            s_lon=_np.array([s.lon for s in shelters], dtype=_np.float32),
            s_lat=_np.array([s.lat for s in shelters], dtype=_np.float32),
            s_cap=_np.array([s.capacity for s in shelters], dtype=_np.float32))
        log(f"wrote {len(frames)} frames to {fp} "
            f"({fp.stat().st_size/1024**2:.1f} MB)")
    log(f"events on disk: {ev_bytes/1024**2:.1f} MB")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
