#!/usr/bin/env python3
"""Cycle, latency and throughput analysis of the streamed SIENNA pipeline from TB_sienna_top's PERF trace."""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import regression as reg  # noqa: E402

REPORT = os.path.join(ROOT, "testbenches", "results", "perf", "pipeline_performance_report.log")
MESH = {1: "RESET_SEQ", 2: "BROADCAST", 3: "FIRE", 4: "WAIT_TILES", 5: "REDUCE", 6: "WAIT_REDUCE", 7: "DONE"}
CONFIGS = ["matmul_random_sigm", "matmul_random_tanh", "matmul_ident_selu", "conv_basic_selu",
           "matmul_random_tanh_train", "matmul_large_selu", "matmul_large_sigm", "matmul_large_tanh"]


def run(name: str, num_sets: int, build_dir: str) -> str:
    cfg = next(t for t in reg.PIPELINE_TESTS if t["name"] == name)
    reg.generate_vectors({"n": 16, "tile_size": 4, **cfg, "num_sets": num_sets})
    cmd = ["make", "verilator", "TRACE=0", "EXTRA_FLAGS=-DPERF", f"VERILATOR_DIR={build_dir}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.stdout + r.stderr


def events(raw: str) -> dict:
    """PERF lines of the first plain stream pass, as {kind: [(cycle, value), ...]}."""
    ev, on = {}, False
    for m in re.finditer(r"^PERF (\d+) (\w+)(?: (-?\d+))?(?: (-?\d+))?", raw, re.M):
        c, kind = int(m.group(1)), m.group(2)
        a = int(m.group(3)) if m.group(3) is not None else None
        b = int(m.group(4)) if m.group(4) is not None else None
        if kind == "PASS":
            if on:
                break  # only the first stream pass, which has no overrun and no reset
            on = b == 0
            continue
        if on:
            ev.setdefault(kind, []).append((c, a, b))
    return ev


def entries(ev: dict, kind: str, value: int) -> list:
    return [c for c, a, _ in ev.get(kind, []) if a == value]


def exits(ev: dict, kind: str, value: int) -> list:
    out, prev = [], None
    for c, a, _ in ev.get(kind, []):
        if prev == value and a != value:
            out.append(c)
        prev = a
    return out


def analyse(ev: dict, k_sets: int) -> dict:
    load, start = [c for c, _, _ in ev["HOST_LOAD"]], [c for c, _, _ in ev["HOST_START"]]
    mesh = {v: entries(ev, "MESH", v) for v in MESH}
    rd_up, rd_dn = entries(ev, "MREAD", 1), entries(ev, "MREAD", 0)
    g_feed, g_round, g_done = entries(ev, "G", 1), entries(ev, "G", 3), exits(ev, "G", 3)
    p_start, p_wait, p_done = entries(ev, "P", 1), entries(ev, "P", 2), exits(ev, "P", 2)
    done = [c for c, _, _ in ev["DONE"]]
    lanes = [(a, b) for _, a, b in ev.get("LANES", [])]
    n = min(k_sets, len(done), len(mesh[7]), len(g_done), len(p_done), len(start))
    sets = []
    for k in range(n):
        s = dict(
            load=start[k] - load[k],
            wait_mesh=mesh[1][k] - start[k],
            broadcast=mesh[3][k] - mesh[2][k],
            tiles=mesh[5][k] - mesh[4][k],
            reduce=mesh[7][k] - mesh[5][k],
            mesh=mesh[7][k] - mesh[1][k],
            wait_act=g_feed[k] - mesh[7][k],
            read=rd_dn[k] - rd_up[k] if k < min(len(rd_up), len(rd_dn)) else 0,
            act=g_done[k] - g_feed[k],
            wait_pool=p_start[k] - g_done[k],
            dispatch=p_wait[k] - p_start[k],
            drain=p_done[k] - p_wait[k],
            pool=p_done[k] - p_start[k],
            latency=done[k] - start[k],
            lane_util=(lanes[k][0] / (16 * lanes[k][1])) if k < len(lanes) and lanes[k][1] else 0.0,
        )
        sets.append(s)
    gaps = [done[k] - done[k - 1] for k in range(1, n)]
    steady = gaps[len(gaps) // 2:] if gaps else []
    span = done[n - 1] - start[0] if n else 0
    busy = {name: sum(s[name] for s in sets) for name in ("load", "mesh", "act", "pool")}
    return dict(sets=sets, gaps=gaps, steady=steady, span=span, busy=busy, n=n)


def mesh_block() -> list:
    """Per-matmul mesh cycles by tile size, from the mesh regression's own logs if present."""
    d = os.path.join(ROOT, "SystolicMesh", "testbenches", "results", "readiness")
    rows = []
    for t in (2, 4, 8, 16):
        f = os.path.join(d, f"mm_random_N16_T{t}.log")
        if not os.path.exists(f):
            continue
        raw = open(f, errors="ignore").read()
        one = re.search(r"Set 0 : (\d+) cycles", raw)
        ser = re.search(r"\[Serial\] (\d+) sets in (\d+) cycles", raw)
        stm = re.search(r"\[Stream\] (\d+) sets in (\d+) cycles", raw)
        if one and ser and stm:
            rows.append([t, (16 // t) ** 3, one.group(1), f"{int(ser.group(2)) / int(ser.group(1)):.0f}",
                         f"{int(stm.group(2)) / int(stm.group(1)):.0f}"])
    return rows


def fmt_table(rows: list, cols: list) -> list:
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    line = "  " + "  ".join(str(c).rjust(w[i]) for i, c in enumerate(cols))
    out = [line, "  " + "  ".join("-" * x for x in w)]
    out += ["  " + "  ".join(str(r[i]).rjust(w[i]) for i in range(len(cols))) for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", type=int, default=12)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--build-dir", default=os.environ.get("PERF_BUILD_DIR", os.path.join(ROOT, "Verilator_perf")))
    args = ap.parse_args()
    L = ["=" * 100, " SIENNA PIPELINE PERFORMANCE (measured in simulation, cycles)", "=" * 100,
         f" Generated {time.strftime('%Y-%m-%d %H:%M')}  N=16  TILE=4  lanes=16  {args.sets} streamed sets per config",
         " Every number below is measured from TB_sienna_top's PERF trace unless marked MODEL.",
         " Stage times are occupancy: from the stage taking a set to releasing it.", ""]
    summary, models = [], []
    for name in args.configs:
        raw = run(name, args.sets, args.build_dir)
        if "RESULT: PASSED" not in raw:
            L += [f"--- {name}: SIMULATION DID NOT PASS; numbers omitted", ""]
            summary.append([name, "-", "-", "-", "-", "FAIL"])
            continue
        a = analyse(events(raw), args.sets)
        s = a["sets"]
        med = lambda key: int(statistics.median(x[key] for x in s))
        steady = statistics.mean(a["steady"]) if a["steady"] else 0
        stage = {"host load": med("load"), "mesh": med("mesh"), "activation": med("act"), "pooling": med("pool")}
        bott = max(stage, key=stage.get)
        L += [f"--- {name}", ""]
        L += fmt_table([[k, x["load"], x["wait_mesh"], x["mesh"], x["wait_act"], x["act"], x["wait_pool"],
                         x["pool"], x["latency"], f"{100*x['lane_util']:.0f}%"] for k, x in enumerate(s)],
                       ["set", "load", "wait", "mesh", "wait", "activ", "wait", "pool", "latency", "lanes"])
        L += ["",
              f"  Mesh inside  : broadcast {med('broadcast')}, tiles {med('tiles')}, reduce {med('reduce')} (median cycles)",
              f"  Activation   : mesh read {med('read')} of {med('act')} cycles; lanes busy {100*statistics.median(x['lane_util'] for x in s):.0f}% of the round",
              f"  Pooling      : dispatch {med('dispatch')}, drain {med('drain')} cycles",
              f"  Completion gaps: {a['gaps']}",
              f"  Steady state : {steady:.0f} cycles per set (mean of the last {len(a['steady'])} gaps); "
              f"single-set latency {s[0]['latency']} cycles",
              f"  Bottleneck   : {bott} ({stage[bott]} cycles per set); stage occupancy over the run: "
              + ", ".join(f"{k} {100*v/a['span']:.0f}%" for k, v in a['busy'].items()),
              f"  Matmul rate  : 8192 FLOP per set -> {8192/steady:.1f} FLOP/cycle at steady state" if steady else "",
              ""]
        summary.append([name, s[0]["latency"], f"{steady:.0f}", bott, stage[bott], "pass"])
        # MODEL: lanes fill one word per cycle, so lane 15 starts about 240 cycles after lane 0.
        if bott == "activation" and med("read") >= 240:
            par = med("act") - 240
            est = max(med("load"), med("mesh"), par, med("pool"))
            models.append([name, med("act"), par, f"{steady:.0f}", est, f"{steady / est:.1f}x"])
    L += ["=" * 100, " SUMMARY", "=" * 100]
    L += fmt_table(summary, ["config", "latency", "cycles/set", "bottleneck", "its cycles", "sim"])
    mb = mesh_block()
    if mb:
        L += ["", " MESH BLOCK (measured, SystolicMesh regression, mm_random, 5 sets)"]
        L += fmt_table(mb, ["tile", "tile-matmuls", "one set", "serial per set", "streamed per set"])
    if models:
        L += ["", " MODEL (estimate, not measured): the activation stage fills its 16 lanes one word per cycle,",
              " so lane 15 cannot start until about 240 cycles after lane 0. Filling every lane at once from a",
              " 16-wide mesh result read would remove that, and the next limit is the 257-cycle host load:"]
        L += fmt_table(models, ["config", "activ now", "activ est", "cycles/set now", "cycles/set est", "gain"])
    L += ["", " MODEL (not measured): the host writes A and B one word per cycle each, so a set cannot enter",
          " faster than 256 cycles plus the start handshake; the activation stage reads the mesh result",
          " one word per cycle, so it cannot finish a set in fewer than 256 cycles either."]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
