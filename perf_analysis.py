#!/usr/bin/env python3
"""Cycle, latency and throughput analysis of the streamed SIENNA pipeline from TB_sienna_top's PERF trace."""
import argparse
import json
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
CONFIGS = [t["name"] for t in reg.PIPELINE_TESTS]


GEOM = {"n": 16, "tile_size": 4, "lanes": 32}  # set from the command line in main()


def run(name: str, num_sets: int, build_dir: str) -> str:
    cfg = next(t for t in reg.PIPELINE_TESTS if t["name"] == name)
    reg.generate_vectors({**GEOM, **cfg, "num_sets": num_sets})
    cmd = ["make", "verilator", "TRACE=0", "EXTRA_FLAGS=-DPERF", f"VERILATOR_DIR={build_dir}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    raw_dir = os.path.join(os.path.dirname(REPORT), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    open(os.path.join(raw_dir, f"{name}_N{GEOM['n']}.log"), "w").write(r.stdout + r.stderr)  # to re-parse without a rerun
    return r.stdout + r.stderr


def events(raw: str) -> dict:
    """PERF lines of the first plain stream pass, as {kind: [(cycle, value, value2), ...]}."""
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


def leaves(ev: dict, kind: str, idle: int) -> tuple:
    """Cycles a state machine leaves and re-enters its idle state: one pair per set it takes."""
    up, down, prev = [], [], idle
    for c, a, _ in ev.get(kind, []):
        if prev == idle and a != idle:
            up.append(c)
        if prev != idle and a == idle:
            down.append(c)
        prev = a
    return up, down


def analyse(ev: dict, k_sets: int, passes: int = 1) -> dict:
    """Per-set stage times; with passes > 1 only every passes-th set outputs (the others are partial sums)."""
    load, start = [c for c, _, _ in ev["HOST_LOAD"]], [c for c, _, _ in ev["HOST_START"]]
    launch = [c for c, a, _ in ev.get("MESH", []) if a == 1]
    written = [c for c, a, _ in ev.get("MESH", []) if a == 7]
    reduce = [c for c, a, _ in ev.get("MESH", []) if a == 5]
    g_up, g_dn = leaves(ev, "G", 0)
    p_up, p_dn = leaves(ev, "P", 0)
    done = [c for c, _, _ in ev["DONE"]]
    n = min(k_sets, len(done), len(start), len(launch), passes * min(len(written), len(g_dn)))
    n -= n % passes  # whole outputs only
    sets = []
    for k in range(n):
        j, f = k // passes, k - k % passes  # output index (reduce, write and activation are per output); its first set
        s = dict(load=start[k] - load[k], wait_mesh=launch[k] - start[k], mesh=None, wait_act=None, act=None, out=None,
                 latency=None)
        if (k + 1) % passes == 0:  # the set that completes an output
            s.update(mesh=written[j] - launch[f], wait_act=g_up[j] - written[j], act=g_dn[j] - g_up[j],
                     out=done[k] - g_dn[j], latency=done[k] - start[f])
        sets.append(s)
    last = lambda k: (k + 1) % passes == 0
    gap = lambda xs: [xs[k] - xs[k - 1] for k in range(1, min(n, len(xs)))]
    gaps = gap(done)
    full = [done[k] for k in range(n) if last(k)]
    per_out = [b - a for a, b in zip(full, full[1:])]
    steady = [g / passes for g in per_out[len(per_out) // 2:]]  # cycles per set: an output's interval over its passes
    lanes = [a / (GEOM["lanes"] * b) for _, a, b in ev.get("LANES", []) if b]
    pool = [d - u for u, d in zip(p_up, p_dn)]  # sets that output; partial sums skip pooling
    return dict(sets=sets, gaps=gaps, steady=steady, n=n, launch_gaps=gap(launch), host_gaps=gap(start),
                first_latency=next((x["latency"] for x in sets if x["latency"] is not None), 0),
                act_gaps=gap(g_up), pool=pool, lanes=lanes, reduce_to_written=[w - r for r, w in zip(reduce, written)])


DEGREE = {"selu": 8, "sigmoid": 6, "tanh": 8}  # gpnae_poly coefficient table in fp32 and bf16; int8 takes gpnae_model.SETS_INT8's
MUL_LAT, ADD_LAT = 8, 5  # valid in to done out: fp32Multiplier and fp32Adder; main() sets the build's format's values
UNIT_LAT = {"fp32": (8, 5), "bf16": (3, 5), "int8": (1, 1)}  # sienna_fmt_pkg::mul_lat, add_lat
MAC_LAT = {"fp32": 13, "bf16": 8, "int8": 3}  # barrel_mac's Horner loop: multiplier then adder, or fxMac behind a register stage
REQ_LAT = {"fp32": 0, "bf16": 0, "int8": 3}  # tfliteRequant at the lane feed (sienna_fmt_pkg::req_lat()), int8 only
FMT = "fp32"  # the build's format; main() sets it


def degree(act: str) -> int:
    """The polynomial degree of act's coefficient set in the build's format: Task 10's SETS_INT8 in int8."""
    if FMT == "int8":
        return reg.gpnae_model.SETS_INT8[reg.activation_to_code(act)][1]
    return DEGREE[act]


def pkg() -> dict:
    """The generated test_config_pkg's integer parameters: the geometry this run was built with."""
    raw = open(os.path.join(ROOT, "testbenches", "test_config_pkg.sv")).read()
    return {k: int(v) for k, v in re.findall(r"localparam int (\w+) = (-?\d+);", raw)}


def clog2(x: int) -> int:
    return max(0, (x - 1).bit_length())


def model(cfg: dict, collapse: bool = True) -> dict:
    """Cycles each stage should take per set, from the RTL's structure."""
    P = pkg()
    N, T, lanes = P["N"], P["TILE_SIZE"], P["NUM_LANES"]
    per_lane = P["SRAM_DEPTH"] // lanes
    K = N if collapse else T  # depth each array multiplies
    U = min(K, ADD_LAT + 1)  # partial sums per PE pixel: the adder loop plus one (6 in fp32 and bf16, 2 in int8)
    RP = 1 if collapse else N // T  # depth slices summed per output tile
    LAT = 1 + ADD_LAT * clog2(RP * U + 1)  # reducer read to write: tree over the partials and the bias
    words = len(open(os.path.join(reg.TB_DIR, "matrix_west_0.mem")).read().split())
    rows = -(-words // P["HOST_WORDS"])
    m = dict(
        # TB_sienna_top: one row per cycle, enable drops for a cycle, start, one cycle for the credit to land.
        host=rows + 3,
        # Broadcast T rows plus commit, the array feed of K products, the reducer's T^2 reads: whichever is longest.
        mesh=max(T + 2, K, T * T),
        # Launch to written: broadcast T+2, feed registers 2, skew 2(T-1), depth K-1, multiply, add, final flag 2,
        # then T^2 reads and the tree.
        mesh_lat=(T + 2) + 2 + 2 * (T - 1) + (K - 1) + MUL_LAT + ADD_LAT + 2 + T * T + LAT,
        per_lane=per_lane, LAT=LAT)
    act = cfg.get("act")
    windows = ((P["IN_ROWS"] + 2 * P["PADDING"] - P["POOL_H"]) // P["STRIDE_ROWS"] + 1) * \
              ((P["IN_COLS"] + 2 * P["PADDING"] - P["POOL_W"]) // P["STRIDE_COLS"] + 1)
    m["pool_dispatch"] = -(-windows // lanes) * P["POOL_H"] * P["POOL_W"]  # every lane takes a window element per cycle
    if act in ("relu", "linear") and not cfg.get("mixed_acts") and cfg.get("accum_passes", 1) == 1:
        m["act"] = per_lane + 4 + REQ_LAT[FMT]  # FEED, LATCH, one wide beat per cycle plus a cycle of read latency, the requantize stage in int8, then done
    elif act in DEGREE:
        m["act_rounds"] = (degree(act) + 1) * max(per_lane, MAC_LAT[FMT] + 1)  # barrel MAC round: max(n, Horner loop + 1)
    return m


def fmt_table(rows: list, cols: list) -> list:
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    line = "  " + "  ".join(str(c).rjust(w[i]) for i, c in enumerate(cols))
    out = [line, "  " + "  ".join("-" * x for x in w)]
    out += ["  " + "  ".join(str(r[i]).rjust(w[i]) for i in range(len(cols))) for r in rows]
    return out


def med(xs: list) -> int:
    return int(statistics.median(xs)) if xs else 0


def summary_cols() -> list:
    """The summary table's columns; int8 counts integer operations and TOPS where the floats count FLOP and GFLOPS."""
    op, rate = ("OP/cyc", "TOPS*") if FMT == "int8" else ("FLOP/cyc", "GFLOPS*")
    return ["config", "latency", "cycles/set", op, rate, "PE use", "limit", "its cycles",
            "host model", "mesh model", "mesh lat model", "mesh lat", "sim"]


def one_config(name: str, args) -> tuple:
    """(detail lines, summary row) for one regression config."""
    flop = 2 * args.n ** 3
    op = "OP" if FMT == "int8" else "FLOP"
    rate = lambda s: flop / s * args.clock_mhz / (1e6 if FMT == "int8" else 1000)  # TOPS in int8, GFLOPS otherwise
    rate_s = lambda s: f"{rate(s):.3f} TOPS" if FMT == "int8" else f"{rate(s):.1f} GFLOPS"
    cfg = next(t for t in reg.PIPELINE_TESTS if t["name"] == name)
    if args.reparse:  # the saved trace; only the stimulus is regenerated, for the model's geometry
        reg.generate_vectors({**GEOM, **cfg, "num_sets": args.sets})
        raw = open(os.path.join(args.reparse, f"{name}_N{args.n}.log"), errors="ignore").read()
    else:
        raw = run(name, args.sets, args.build_dir)
    mdl = model(cfg, not args.slices)
    if "RESULT: PASSED" not in raw or "Assertion failed" in raw:
        return [f"--- {name}: SIMULATION DID NOT PASS; numbers omitted", ""], [name] + ["-"] * 11 + ["FAIL"]
    a = analyse(events(raw), args.sets, cfg.get("accum_passes", 1))
    s = a["sets"]
    P = cfg.get("accum_passes", 1)
    first = next(x for x in s if x["mesh"] is not None)  # the first output's last set
    steady = statistics.mean(a["steady"]) if a["steady"] else 0
    # Each stage's own cost per set: the host's start interval; activation runs one set at a time; the mesh is
    # pipelined, so its cost is its tightest launch interval, not the time a set spends inside it. Activation and
    # pooling run once per output, so an accumulate config spreads them over its P sets.
    stage = {"host": min(a["host_gaps"] or [0]), "mesh": min(a["launch_gaps"] or [0]),
             "activation": round(med([x["act"] for x in s if x["act"] is not None]) / P), "pooling": round(med(a["pool"]) / P)}
    lim = max(stage, key=stage.get)
    L = [f"--- {name}", ""]
    dash = lambda v: "-" if v is None else v  # a partial-sum set has no output of its own
    L += fmt_table([[k, x["load"], x["wait_mesh"], x["mesh"], x["wait_act"], x["act"], dash(x["out"]), dash(x["latency"])]
                    for k, x in enumerate(s)],
                   ["set", "load", "wait", "mesh", "wait", "activ", "pool+out", "latency"])
    L += ["",
          f"  Stage cost per set: host load {stage['host']}, mesh launch interval min {stage['mesh']} median "
          f"{med(a['launch_gaps'])}, activation {stage['activation']}, pooling {stage['pooling']} (cycles)",
          f"  Mesh inside  : reduce start to result written {med(a['reduce_to_written'])} cycles (median)"
          + (f"; GPNAE lanes busy {100 * statistics.median(a['lanes']):.0f}% of a round" if a["lanes"] else ""),
          f"  Completion gaps: {a['gaps']}",
          f"  Steady state : {steady:.1f} cycles per set (mean of the last {len(a['steady'])} output intervals"
          + (f", each over its {cfg['accum_passes']} depth passes" if cfg.get("accum_passes", 1) > 1 else "")
          + f"); first-output latency {a['first_latency']} cycles",
          f"  Limit        : {lim} ({stage[lim]} cycles per set)",
          f"  Design model : host {mdl['host']} (measured {stage['host']}), mesh interval {mdl['mesh']} (measured min "
          f"{stage['mesh']}, host-bound), mesh latency {mdl['mesh_lat']} (measured {first['mesh']} on the first output"
          + (f", over its {P} depth passes" if P > 1 else "") + "), "
          + (f"activation {mdl['act']} (measured {stage['activation']})" if "act" in mdl else
             f"activation: {mdl.get('act_rounds', '-')} cycles of polynomial rounds in the measured {stage['activation']}")
          + f", pooling dispatch {mdl['pool_dispatch']} of the measured {stage['pooling']}",
          f"  Matmul rate  : {flop} {op} per set -> {flop / steady:.1f} {op}/cycle, {rate_s(steady)}"
          f" at an ASSUMED {args.clock_mhz:.0f} MHz, {100 * flop / 2 / steady / args.n ** 2:.1f}% of the mesh's"
          f" {args.n ** 2} MAC/cycle" if steady else "", ""]
    row = [name, a["first_latency"], f"{steady:.1f}", f"{flop / steady:.1f}", f"{rate(steady):.3f}" if FMT == "int8" else f"{rate(steady):.1f}",
           f"{100 * flop / 2 / steady / args.n ** 2:.0f}%", lim, stage[lim], mdl["host"], mdl["mesh"], mdl["mesh_lat"],
           first["mesh"], "pass"]
    return L, row


def header(args) -> list:
    return ["=" * 100, " SIENNA PIPELINE PERFORMANCE (measured in simulation, cycles)", "=" * 100,
            f" Generated {time.strftime('%Y-%m-%d %H:%M')}  N={args.n}  TILE={args.tile_size}  lanes={args.lanes}  "
            f"{args.sets} streamed sets per config, streaming host (one row of N words per operand per cycle)",
            " Every number is measured from TB_sienna_top's PERF trace. Per set: load = host rows, wait = to mesh launch,",
            " mesh = launch to result written (sets overlap inside it), wait = to the activation stage, activ = activation",
            " stage occupancy (a partial sum, summed in the PEs, passes as a null), pool+out = to the set's completion pulse.", ""]


def footer(args, summary: list) -> list:
    L = ["=" * 100, " SUMMARY", "=" * 100]
    L += fmt_table(summary, summary_cols())
    L += [" latency = host start of the first output's first set to its completion pulse, on an idle pipeline; cycles/set",
          " counts each depth pass of an accumulate config as a set. host/mesh model: cycles per set the",
          " design needs (host N+3 is TB_sienna_top's handshake; the mesh alone needs max(T+2, K, T^2)).",
          " mesh lat model: launch to result written, 3T+K+T^2+LAT+16; mesh lat: the same, measured on set 0."]
    unit = "TOPS (2 x MACs per second)" if FMT == "int8" else "GFLOPS"
    L += [f" * {unit} at an ASSUMED {args.clock_mhz:.0f} MHz clock, not a timing result; they count the set's"
          f" {2 * args.n ** 3}-{'OP' if FMT == 'int8' else 'FLOP'} matmul only.",
          " PE use = multiply-accumulates per cycle over the mesh's N^2. limit = the stage with the largest cost per set."]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", type=int, default=24, help="streamed sets; 24 fits every accumulate and mixed pattern")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--clock-mhz", type=float, default=950.0,
                    help="assumed clock for GFLOPS; no timing run has demonstrated one")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--build-dir", default=os.environ.get("PERF_BUILD_DIR", os.path.join(ROOT, "Verilator_perf")))
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--reparse", help="directory of saved traces (<config>_N<n>.log) to analyse instead of simulating")
    ap.add_argument("--slices", action="store_true", help="the build has COLLAPSE_K=0 (depth slices); for the model only")
    ap.add_argument("--merge", nargs="*", help="combine the .json parts of earlier runs into --report, in order")
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(UNIT_LAT), help="number format of the build")
    args = ap.parse_args()
    global MUL_LAT, ADD_LAT, FMT
    MUL_LAT, ADD_LAT = UNIT_LAT[args.fmt_name]
    FMT = args.fmt_name
    fmts = {t["name"]: t.get("formats", tuple(UNIT_LAT)) for t in reg.PIPELINE_TESTS}
    args.configs = [c for c in args.configs if args.fmt_name in fmts[c]]  # int8-only tests run only in int8
    GEOM.update(n=args.n, tile_size=args.tile_size, lanes=args.lanes, fmt_name=args.fmt_name)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)  # the .json part lands there first
    if args.merge:
        parts = [json.load(open(f)) for f in args.merge]
        L = header(args) + [x for p in parts for x in p["lines"]] + footer(args, [r for p in parts for r in p["rows"]])
    else:
        lines, rows = [], []
        for name in args.configs:
            d, r = one_config(name, args)
            lines += d
            rows.append(r)
            print("\n".join(d), flush=True)
        json.dump({"lines": lines, "rows": rows}, open(os.path.splitext(args.report)[0] + ".json", "w"))
        L = header(args) + lines + footer(args, rows)
    open(args.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-len(rows if not args.merge else parts) - 8:]))
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
