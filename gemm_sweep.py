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


def exact_layer(A, B, bias, act, N, fmt):
    """Bit-exact output of sienna_layer for one product in a narrow format: per output tile, the depth blocks as passes in
    order (format_layer's order), the bias with the first, then the lane; returns the output's bit patterns."""
    import regression as reg
    from mesh_model import fpu
    import mesh_model
    import gpnae_model
    f = fpu.FORMATS[fmt]
    M, K = A.shape
    C = B.shape[1]
    rt, ct, dt = -(-M // N), -(-C // N), -(-K // N)
    Ap = np.zeros((rt * N, dt * N), np.float32)
    Ap[:M, :K] = A
    Bp = np.zeros((dt * N, ct * N), np.float32)
    Bp[:K, :C] = B
    bp = np.zeros(ct * N, np.float32)
    if bias is not None:
        bp[: bias.size] = bias
    rom = gpnae_model.read_rom(os.path.join(reg.ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f)))
    lane = gpnae_model.Lane(f, rom)
    Y = np.zeros((rt * N, ct * N), np.int64)
    for c in range(ct):
        for r in range(rt):
            passes = [(reg.fmt_bits(Ap[r * N:(r + 1) * N, t * N:(t + 1) * N], fmt), reg.fmt_bits(Bp[t * N:(t + 1) * N, c * N:(c + 1) * N], fmt))
                      for t in range(dt)]
            b = reg.fmt_bits(bp[c * N:(c + 1) * N], fmt) if bias is not None else None
            Ct = mesh_model.matmul(f, passes, N, 4, 1, b)
            Y[r * N:(r + 1) * N, c * N:(c + 1) * N] = lane.run(Ct, reg.activation_to_code(act))
    return Y[:M, :C]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "gemm"))
    ap.add_argument("--emulate", action="store_true", help="numpy stand-in for the RTL")
    ap.add_argument("--quick", action="store_true", help="a few small shapes only")
    ap.add_argument("--engine", choices=("layer", "sets"), default="layer",
                    help="layer: sienna_layer schedules everything; sets: the host drives sienna_top set by set")
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(mr.regression.FORMATS),
                    help="format of A and B on the layer engine; sums stay fp32, and the error is judged on the rounded inputs")
    ap.add_argument("--host-gaps", action="store_true", help="host idles a cycle after each load and waits for the credit")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.emulate:
        sim = mr.EmuSim(a.n, a.lanes, a.work)
    elif a.engine == "layer":
        sim = mr.LayerSim(a.n, a.lanes, a.work, a.fmt_name)
    else:
        sim = mr.Sim(a.n, a.lanes, a.work, a.host_gaps)
    sim.build()
    shapes = [(f"grid_{m}x{k}x{n}", m, k, n) for m in GRID_M for k in GRID_K for n in GRID_N] + TRANSFORMER
    if a.quick:
        shapes = [s for s in shapes if s[1] * s[2] * s[3] <= 64 * 64 * 64][:6]
    rep = open(os.path.join(a.work, f"gemm_sweep_N{a.n}.log"), "w")
    peak = a.n * a.n  # collapse-k mesh: N^2 PEs, one product per PE per cycle at best
    head = f"{'shape':<22} {'M':>5} {'K':>5} {'N':>5} {'sets':>7} {'cycles':>10} {'MAC/cycle':>9} {'PE use':>7} {'slot use':>8} {'max err':>8} {'wall s':>6}"
    for line in (f"GEMM sweep on the RTL, mesh N={a.n}, {a.lanes} lanes, linear activation, {a.fmt_name} operands and sums; "
                 f"peak {peak} MAC/cycle", head):
        print(line, flush=True)
        rep.write(line + "\n")
    rows = []
    for name, m, k, n in shapes:
        rng = np.random.RandomState(m * 7 + k * 13 + n)
        A = mr.regression.op_round(rng.uniform(-1, 1, (m, k)), a.fmt_name)
        B = mr.regression.op_round(rng.uniform(-1, 1, (k, n)), a.fmt_name)
        job = {"terms": [(A, B)], "bias": None, "act": "linear", "shape": (m, n)}
        t0 = time.time()
        y, sets, cyc = sim.run_job(job, name) if isinstance(sim, mr.LayerSim) else mr.run_job_hw(job, sim, name)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = float(np.max(np.abs(y - ref)) / (np.max(np.abs(ref)) or 1.0))
        mism = 0
        if a.fmt_name != "fp32" and isinstance(sim, mr.LayerSim):  # narrow formats: every output bit-exact
            mism = int(np.sum(mr.regression.fmt_bits(y, a.fmt_name) != exact_layer(A, B, None, "linear", a.n, a.fmt_name)))
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
        if a.fmt_name == "fp32" and err > 1e-4:
            print(f"FAIL {name}: error {err:.2e} above 1e-4", flush=True)
            sys.exit(1)
        if a.fmt_name != "fp32" and mism:
            print(f"FAIL {name}: {mism} outputs differ from the bit-exact model", flush=True)
            sys.exit(1)
    if isinstance(sim, mr.LayerSim):
        # The polynomial activations through the layer engine, with a bias; GPNAE approximates within about 2%.
        import regression
        rng = np.random.RandomState(5)
        A = regression.op_round(rng.uniform(-1, 1, (64, 48)), a.fmt_name)
        B = regression.op_round(rng.uniform(-0.3, 0.3, (48, 40)), a.fmt_name)
        b = regression.op_round(rng.uniform(-0.5, 0.5, 40), a.fmt_name)
        for act in ("tanh", "sigmoid", "selu"):
            y, sets, cyc = sim.run_job({"terms": [(A, B)], "bias": b, "act": act, "shape": (64, 40)}, f"act_{act}")
            ref = regression.apply_activation((A.astype(np.float64) @ B + b).astype(np.float32), act)
            err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
            line = f"layer_{act}_bias{'':<9} {64:>5} {48:>5} {40:>5} {sets:>7} {cyc:>10}  max err {err:.1e} of the output range"
            print(line, flush=True)
            rep.write(line + "\n")
            if a.fmt_name == "fp32" and err > 3e-2:
                print(f"FAIL layer_{act}: error {err:.2e} above 3e-2", flush=True)
                sys.exit(1)
            if a.fmt_name != "fp32":  # narrow formats: bit-exact against the model; the error above is reported, not gated
                mism = int(np.sum(regression.fmt_bits(y, a.fmt_name) != exact_layer(A, B, b, act, a.n, a.fmt_name)))
                print(f"  layer_{act}: {mism} outputs differ from the bit-exact model", flush=True)
                rep.write(f"  layer_{act}: {mism} outputs differ from the bit-exact model\n")
                if mism:
                    sys.exit(1)


if __name__ == "__main__":
    main()
