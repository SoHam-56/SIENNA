#!/usr/bin/env python3
"""Packed layers on sienna_layer: each job's block equals the job alone on the RTL and its golden, bit for bit; cycles packed vs alone."""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import gemm_sweep as gs  # noqa: E402
import model_runner as mr  # noqa: E402

reg = mr
ACTS = ["linear", "tanh", "relu", "selu", "sigmoid", "linear", "relu", "tanh"]


def layer(N, sh, R, rng, fmt, zero_rows=False):
    """A packed layer of R row tiles: block c's job is A[:, block] @ W_c + bias, through ACTS[(c + 1) % 8]."""
    b = N >> sh
    A = rng.uniform(-1, 1, (R * N, N))
    B = np.zeros((N, N))
    for c in range(N // b):
        B[c * b:(c + 1) * b, c * b:(c + 1) * b] = rng.uniform(-1, 1, (b, b))
    bias = rng.uniform(-0.5, 0.5, N)
    mp = [((c + 1) % 8 if c < N // b else 0) for c in range(N // 2)]
    if zero_rows:  # exact zero sums, no bias beat: rows 0, 1 are -0 / +0 inputs, rows 2 mod 4 cancel x * w against -x * w, block 0's first column has zero weights
        A[0::4, :], A[1::4, :] = -0.0, 0.0
        A[2::4, :] = 0.0
        for c in range(N // b):
            B[c * b + 1, c * b:(c + 1) * b] = B[c * b, c * b:(c + 1) * b]
            A[2::4, c * b] = rng.uniform(-1, 1, A[2::4, c * b].shape)
            A[2::4, c * b + 1] = -A[2::4, c * b]
        B[0:b, 0] = 0.0
        bias = np.zeros(N)
        mp = [(0, 1, 5, 7)[c % 4] if c < N // b else 0 for c in range(N // 2)]  # linear and tanh entries only
    return A, B, bias, mp, b


def run_case(a, sim, N, sh, R, seed, log, zero_rows=False):
    rng = np.random.RandomState(seed)
    A, B, bias, mp, b = layer(N, sh, R, rng, a.fmt, zero_rows)
    col_ent = [mp[j // b] for j in range(N)]
    ents = [(a.acts[e], None) for e in range(8)]
    if a.fmt == "int8":
        A_q, s_a, z_a = reg.quant_act(A)
        B_q, s_w = reg.quant_weights(B)
        hw = reg.fold_bias(bias, s_a, s_w, z_a, B_q)
        acc = reg.wrap32(reg.imatmul(A_q, B_q) + hw[None, :])
        mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
        for e in sorted(set(col_ent)):
            cols = np.array(col_ent) == e
            act = "linear" if a.acts[e] == "selu" else a.acts[e]  # SELU's int8 saturation is pack_jobs' to refuse (Task 7)
            rq = reg.requant_params(acc[:, cols], s_a, s_w[cols], act)
            mult[cols], shift[cols] = rq["mult"], rq["shift"]
            ents[e] = (act, rq)
        head = ents[0][1] or reg.requant_params(acc, s_a, s_w, "linear")
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw, "act": ents[0][0], "shape": (R * N, N),
               "req": dict(head, mult=mult, shift=shift), "pack": {"shift": sh, "map": mp, "ents": ents}}
    else:
        A, B, bias = (reg.op_round(v, a.fmt) for v in (A, B, bias))
        job = {"terms": [(A, B)], "bias": bias, "act": a.acts[0], "shape": (R * N, N), "pack": {"shift": sh, "map": mp, "ents": ents}}
    t0 = time.time()
    Yp, sets_p, cyc_p = sim.run_job(job, f"pk_s{sh}")
    bad = cyc_a = 0
    for c in range(N // b):
        cols = slice(c * b, (c + 1) * b)
        e = mp[c]
        if a.fmt == "int8":
            rq = dict(ents[e][1], mult=mult[cols], shift=shift[cols])
            alone = {"terms": [(A_q[:, cols].astype(np.float32), B_q[cols, cols].astype(np.float32))], "bias": hw[cols],
                     "act": ents[e][0], "shape": (R * N, b), "req": rq}
            gold = reg.int8_layer_exact(A_q[:, cols], B_q[cols, cols], hw[cols], rq, ents[e][0])
        else:
            alone = {"terms": [(A[:, cols], B[cols, cols])], "bias": bias[cols], "act": a.acts[e], "shape": (R * N, b)}
            gold = gs.exact_layer(A[:, cols], B[cols, cols], bias[cols] if np.any(bias[cols]) else None, a.acts[e], N, a.fmt, a.tile) if a.fmt == "bf16" else None  # fp32: RTL against RTL (F-GP1)
        Ya, _, cyc = sim.run_job(alone, f"al_s{sh}_{c}")
        cyc_a += cyc
        bits = (lambda y: y) if a.fmt == "int8" else (lambda y: reg.fmt_bits(y, a.fmt))
        bad += int(np.sum(bits(Yp[:, cols]) != bits(Ya)))
        if gold is not None:
            bad += int(np.sum(bits(Yp[:, cols]) != gold))
    line = (f"{a.fmt} N={N} T={a.tile} b={b:<3} rows={R * N:<4} {'zero rows ' if zero_rows else ''}packed: {sets_p} sets "
            f"{cyc_p} cycles | {N // b} jobs alone: {cyc_a} cycles | speedup {cyc_a / cyc_p:5.2f}x | mismatches {bad} | "
            f"wall {time.time() - t0:.0f}s")
    print(line, flush=True)
    log.write(line + "\n")
    return bad


def run_models(a, sim, log):
    """Review Focus 2: heterogeneous small jobs through pack_jobs, fewer models than blocks; each against itself alone."""
    rng = np.random.RandomState(77)
    N = a.n
    shapes = [(3, 2, [5, 9]), (2, 2, [N]), (2, 1, [3])]  # blocks of 4: N = 8 holds two models, 16 and 32 leave blocks empty
    acts = ["tanh", "relu", "linear"]
    models = [{"W": rng.uniform(-1, 1, (K, C)), "bias": rng.uniform(-0.5, 0.5, C), "act": act, "req": None,
               "inputs": [rng.uniform(-1, 1, (m, K)) for m in ms]} for (K, C, ms), act in zip(shapes, acts)][:N // 4]
    if a.fmt != "fp32":
        for m in models:
            m["W"], m["bias"] = reg.op_round(m["W"], a.fmt), reg.op_round(m["bias"], a.fmt)
            m["inputs"] = [reg.op_round(x, a.fmt) for x in m["inputs"]]
    if a.fmt == "int8":
        log.write("int8 pack_jobs case: covered by model_runner.py --action tflite --pack (Task 8)\n")
        return 0
    job, recipe = mr.pack_jobs(models, N, int8=False)
    Y, _, _ = sim.run_job(job, "pj")
    bad = 0
    for i, (m, outs) in enumerate(zip(models, mr.unpack(Y, recipe))):
        X = np.vstack(m["inputs"]).astype(np.float32)
        alone = {"terms": [(X, m["W"].astype(np.float32))], "bias": m["bias"].astype(np.float32), "act": m["act"],
                 "shape": (X.shape[0], m["W"].shape[1])}
        Ya, _, _ = sim.run_job(alone, f"pj_al{i}")
        got = np.vstack(outs)
        bad += int(np.sum(reg.fmt_bits(got, a.fmt) != reg.fmt_bits(Ya, a.fmt)))
    line = f"{a.fmt} N={N} T={a.tile} pack_jobs: {len(models)} models in {N // (N >> job['pack']['shift'])} blocks, mismatches {bad}"
    print(line, flush=True)
    log.write(line + "\n")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--rows", type=int, nargs="+", default=[2], help="row tiles per packed layer; several run in turn")
    ap.add_argument("--format", dest="fmt", default="int8", choices=sorted(reg.FORMATS))
    ap.add_argument("--act", choices=sorted(set(ACTS)), help="every entry holds this activation (a same-activation cycle sweep); default the mixed table")
    a = ap.parse_args()
    a.acts = [a.act] * 8 if a.act else ACTS
    work = os.path.join(ROOT, "testbenches", "results", "pack")
    os.makedirs(work, exist_ok=True)
    sim = mr.RtlLayer(a.n, a.lanes, work, a.fmt, a.tile)
    sim.build()
    log = open(os.path.join(work, f"pack_regression_{a.fmt}_N{a.n}_T{a.tile}{'_' + a.act if a.act else ''}.log"), "w")
    bad = 0
    for R in a.rows:
        for sh in range(1, a.n.bit_length() - 1):
            bad += run_case(a, sim, a.n, sh, R, 900 + sh, log)
    if a.act is None:  # the correctness cases below choose their own activations
        if a.fmt != "int8":
            bad += run_case(a, sim, a.n, 2, a.rows[0], 990, log, zero_rows=True)  # Review Focus 3: signed zeros
        bad += run_models(a, sim, log)
    tail = f"PACK REGRESSION {'PASS' if bad == 0 else 'FAIL'}: {bad} mismatching outputs"
    print(tail, flush=True)
    log.write(tail + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
