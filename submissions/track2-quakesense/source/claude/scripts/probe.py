"""Mid-flight probe: judge a run from its first LLM round, not its last.

An LLM arm costs about twenty minutes. Three of them were wasted today on
faults that were already visible in round 0 -- every reply a hold, every reply
rejected as a hallucination, every reply an HTTP error. This reads the newest
context log of whatever is running and says whether it is worth finishing.

    python -m scripts.probe                      # one look
    python -m scripts.probe --watch 900          # every 15 minutes

Exit code is 2 when a running job looks dead on arrival, so a supervisor can
kill it rather than wait.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

RUNS = Path("/data/quakesense/runs")

#: A round is suspicious when almost every decision is the same one. Both
#: extremes are failures with different causes: all-hold means the brief has
#: nothing actionable in it, all-error means the server is refusing.
ALL_SAME = 0.98


def probe_round(f: Path):
    recs = json.loads(f.read_text(encoding="utf-8"))
    n = len(recs)
    if not n:
        return None
    hold = err = unparse = 0
    ids = []
    for r in recs:
        rep = (r.get("reply") or "").strip()
        if rep.startswith("__error__"):
            err += 1
            continue
        m = re.search(r'"send_to"\s*:\s*(-?\d+)', rep)
        if not m:
            unparse += 1
            continue
        sid = int(m.group(1))
        (ids.append(sid) if sid >= 0 else None)
        hold += sid < 0
    return {"file": f.name, "calls": n, "hold": hold, "err": err,
            "unparseable": unparse, "directed": len(ids),
            "distinct_targets": len(set(ids))}


def look(verbose=True):
    alerts = []
    live = sorted(RUNS.glob("*/llm_context"), key=lambda p: p.stat().st_mtime)
    for ctx in live[-3:]:
        rounds = sorted(ctx.glob("round_*.json"))
        if not rounds:
            continue
        s = probe_round(rounds[-1])
        if not s:
            continue
        run = ctx.parent.name
        finished = (ctx.parent / "summary.json").exists()
        n = s["calls"]
        line = (f"{run:<16} {s['file']:<16} calls {n:>4}  "
                f"directed {s['directed']:>4}  hold {s['hold']:>4}  "
                f"err {s['err']:>4}  unparse {s['unparseable']:>4}  "
                f"targets {s['distinct_targets']:>4}"
                f"{'  [done]' if finished else '  [running]'}")
        if verbose:
            print(line)
        if finished:
            continue
        if s["err"] / n > 0.5:
            alerts.append(f"{run}: {s['err']}/{n} replies are transport errors "
                          f"-- the model server is refusing, kill and restart it")
        elif s["unparseable"] / n > 0.5:
            alerts.append(f"{run}: {s['unparseable']}/{n} unparseable -- check "
                          f"the grammar is actually being sent")
        elif s["hold"] / n > ALL_SAME:
            alerts.append(f"{run}: {s['hold']}/{n} decisions are hold -- the "
                          f"brief has nothing actionable in it; this arm will "
                          f"come out identical to LLM-off")
        elif s["directed"] and s["distinct_targets"] <= 2:
            alerts.append(f"{run}: {s['directed']} directions but only "
                          f"{s['distinct_targets']} distinct targets -- every "
                          f"leader is naming the same shelter")
    return alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="Seconds between looks. 0 = look once and exit.")
    ap.add_argument("--log", default="/tmp/probe.log")
    a = ap.parse_args()
    log = Path(a.log)
    worst = 0
    while True:
        stamp = time.strftime("%H:%M:%S")
        alerts = look()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"--- {stamp}\n")
            for m in alerts:
                fh.write(f"[ALERT] {m}\n")
            if not alerts:
                fh.write("[ok] nothing looks dead on arrival\n")
        for m in alerts:
            print(f"[ALERT] {m}")
        worst = max(worst, 2 if alerts else 0)
        if not a.watch:
            break
        time.sleep(a.watch)
    raise SystemExit(worst)


if __name__ == "__main__":
    main()
