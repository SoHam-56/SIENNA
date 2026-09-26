#!/usr/bin/env python3
"""Model-agnostic benchmark: C = A (M x K) @ B (K x N) on the RTL over a grid of shapes, plus transformer-sized shapes."""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_runner as mr  # noqa: E402

GRID_M = [1, 16, 64, 256, 1024]
GRID_K = [16, 64, 256, 1024]
GRID_N = [16, 64, 256]
# A BERT-base / small-LLM layer with 128 tokens: projections, MLP, attention scores and values, and one decode step.
TRANSFORMER = [
    ("qkv_proj_128tok", 128, 768, 768),
    ("mlp_up_128tok", 128, 768, 3072),
    ("mlp_down_128tok", 128, 3072, 768),
    ("attn_scores", 128, 64, 128),
    ("attn_values", 128, 128, 64),
    ("decode_proj_1tok", 1, 768, 768),
    ("decode_mlp_up_1tok", 1, 768, 3072),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "gemm"))
    ap.add_argument("--emulate", action="store_true", help="numpy stand-in for the RTL")
    ap.add_argument("--quick", action="store_true", help="a few small shapes only")
    ap.add_argument("--engine", choices=("layer", "sets"), default="layer",
                    help="layer: sienna_layer schedules everything; sets: the host drives sienna_top set by set")
    ap.add_argument("--host-gaps", action="store_true", help="host idles a cycle after each load and waits for the credit")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.emulate:
        sim = mr.EmuSim(a.n, a.lanes, a.work)
    elif a.engine == "layer":
        sim = mr.LayerSim(a.n, a.lanes, a.work)
    else:
        sim = mr.Sim(a.n, a.lanes, a.work, a.host_gaps)
    sim.build()
    shapes = [(f"grid_{m}x{k}x{n}", m, k, n) for m in GRID_M for k in GRID_K for n in GRID_N] + TRANSFORMER
    if a.quick:
        shapes = [s for s in shapes if s[1] * s[2] * s[3] <= 64 * 64 * 64][:6]
    rep = open(os.path.join(a.work, f"gemm_sweep_N{a.n}.log"), "w")
    peak = a.n * a.n  # collapse-k mesh: N^2 PEs, one product per PE per cycle at best
    head = f"{'shape':<22} {'M':>5} {'K':>5} {'N':>5} {'sets':>7} {'cycles':>10} {'MAC/cycle':>9} {'PE use':>7} {'slot use':>8} {'max err':>8} {'wall s':>6}"
    for line in (f"GEMM sweep on the RTL, mesh N={a.n}, {a.lanes} lanes, linear activation, fp32; peak {peak} MAC/cycle", head):
        print(line, flush=True)
        rep.write(line + "\n")
    rows = []
    for name, m, k, n in shapes:
        rng = np.random.RandomState(m * 7 + k * 13 + n)
        A = rng.uniform(-1, 1, (m, k)).astype(np.float32)
        B = rng.uniform(-1, 1, (k, n)).astype(np.float32)
        job = {"terms": [(A, B)], "bias": None, "act": "linear", "shape": (m, n)}
        t0 = time.time()
        y, sets, cyc = sim.run_job(job, name) if isinstance(sim, mr.LayerSim) else mr.run_job_hw(job, sim, name)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = float(np.max(np.abs(y - ref)) / (np.max(np.abs(ref)) or 1.0))
        macs = m * k * n
        r = {"shape": name, "M": m, "K": k, "N": n, "sets": sets, "cycles": cyc, "macs": macs,
             "mac_per_cycle": macs / cyc if cyc else 0, "pe_use": macs / (cyc * peak) if cyc else 0,
             "slot_use": macs / (sets * a.n ** 3), "err": err, "wall": time.time() - t0}
        rows.append(r)
        line = (f"{name:<22} {m:>5} {k:>5} {n:>5} {sets:>7} {cyc:>10} {r['mac_per_cycle']:>9.1f} {100 * r['pe_use']:>6.1f}% "
                f"{100 * r['slot_use']:>7.1f}% {err:>8.1e} {r['wall']:>6.0f}")
        print(line, flush=True)
        rep.write(line + "\n")
        rep.flush()
        json.dump(rows, open(os.path.join(a.work, f"gemm_sweep_N{a.n}.json"), "w"), indent=1)
        if err > 1e-4:
            print(f"FAIL {name}: error {err:.2e} above 1e-4", flush=True)
            sys.exit(1)
    if isinstance(sim, mr.LayerSim):
        # The polynomial activations through the layer engine, with a bias; GPNAE approximates within about 2%.
        import regression
        rng = np.random.RandomState(5)
        A = rng.uniform(-1, 1, (64, 48)).astype(np.float32)
        B = rng.uniform(-0.3, 0.3, (48, 40)).astype(np.float32)
        b = rng.uniform(-0.5, 0.5, 40).astype(np.float32)
        for act in ("tanh", "sigmoid", "selu"):
            y, sets, cyc = sim.run_job({"terms": [(A, B)], "bias": b, "act": act, "shape": (64, 40)}, f"act_{act}")
            ref = regression.apply_activation((A.astype(np.float64) @ B + b).astype(np.float32), act)
            err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
            line = f"layer_{act}_bias{'':<9} {64:>5} {48:>5} {40:>5} {sets:>7} {cyc:>10}  max err {err:.1e} of the output range"
            print(line, flush=True)
            rep.write(line + "\n")
            if err > 3e-2:
                print(f"FAIL layer_{act}: error {err:.2e} above 3e-2", flush=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
