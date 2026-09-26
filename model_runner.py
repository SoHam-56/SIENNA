#!/usr/bin/env python3
"""Runs tflite float models end to end on the SIENNA RTL: every multiply-accumulate on the pipeline, host only reshapes and softmax."""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import regression  # noqa: E402

ACT_CODE = {"relu": 4, "linear": 5}  # the bypass modes of gpnae_poly
HW_BIAS = True  # the mesh adds the bias; False lowers it as a ones column and an extra depth row


# =============================================================================
# tflite reader
# =============================================================================


def load_tflite(path: str) -> dict:
    import tflite
    from tflite.ActivationFunctionType import ActivationFunctionType as AF
    from tflite.BuiltinOperator import BuiltinOperator as BO

    names = {v: k for k, v in BO.__dict__.items() if not k.startswith("_")}
    acts = {AF.NONE: "linear", AF.RELU: "relu"}
    m = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
    g = m.Subgraphs(0)
    consts = {}
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        buf = m.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(buf, int) or buf is None or len(buf) == 0:
            continue
        shape = tuple(t.ShapeAsNumpy()) if t.ShapeLength() else ()
        if t.Type() == 0:  # FLOAT32
            consts[i] = np.frombuffer(buf.tobytes(), dtype=np.float32).reshape(shape)
        elif t.Type() == 9:  # INT8 weights of a hybrid model, dequantized here once
            q = t.Quantization()
            scale = q.ScaleAsNumpy().astype(np.float32)
            zp = q.ZeroPointAsNumpy() if q.ZeroPointLength() else np.zeros(1)
            w = np.frombuffer(buf.tobytes(), dtype=np.int8).reshape(shape).astype(np.float32)
            ax = q.QuantizedDimension()
            bshape = [1] * len(shape)
            if scale.size > 1:
                bshape[ax] = scale.size
            consts[i] = ((w - zp.reshape(bshape).astype(np.float32)) * scale.reshape(bshape)).astype(np.float32)
        elif t.Type() == 2:  # INT32, reshape targets
            consts[i] = np.frombuffer(buf.tobytes(), dtype=np.int32).reshape(shape)
    opts = {
        "CONV_2D": tflite.Conv2DOptions,
        "DEPTHWISE_CONV_2D": tflite.DepthwiseConv2DOptions,
        "ADD": tflite.AddOptions,
        "FULLY_CONNECTED": tflite.FullyConnectedOptions,
        "AVERAGE_POOL_2D": tflite.Pool2DOptions,
    }
    ops = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        c = m.OperatorCodes(op.OpcodeIndex())
        kind = names[max(c.BuiltinCode(), c.DeprecatedBuiltinCode())]
        d = {
            "kind": kind,
            "inputs": [op.Inputs(j) for j in range(op.InputsLength())],
            "outputs": [op.Outputs(j) for j in range(op.OutputsLength())],
            "act": "linear",
        }
        if kind in opts:
            o = opts[kind]()
            t = op.BuiltinOptions()
            o.Init(t.Bytes, t.Pos)
            fa = o.FusedActivationFunction()
            if fa not in acts:
                raise ValueError(f"op {i} {kind}: fused activation {fa} is not supported")
            d["act"] = acts[fa]
            if hasattr(o, "Padding"):
                d["same"] = o.Padding() == 0
                d["stride"] = (o.StrideH(), o.StrideW())
            if kind == "AVERAGE_POOL_2D":
                d["filter"] = (o.FilterHeight(), o.FilterWidth())
        ops.append(d)
    shapes = {i: tuple(g.Tensors(i).ShapeAsNumpy()) for i in range(g.TensorsLength()) if g.Tensors(i).ShapeLength()}
    return {"ops": ops, "consts": consts, "shapes": shapes, "input": g.Inputs(0), "output": g.Outputs(0)}


# =============================================================================
# Lowering: every compute op becomes Y = sum_i X_i @ W_i + b, then an activation
# =============================================================================


def _same_pad(n, k, s):
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2, total - total // 2


def im2col(x, kh, kw, stride, same, channel_major=False):
    """x is H x W x C; rows are output pixels, depth is (ky, kx, c), or (c, ky, kx) when channel_major."""
    H, W, C = x.shape
    sh, sw = stride
    if same:
        oh, pt, pb = _same_pad(H, kh, sh)
        ow, pl, pr = _same_pad(W, kw, sw)
    else:
        oh, ow, pt, pb, pl, pr = (H - kh) // sh + 1, (W - kw) // sw + 1, 0, 0, 0, 0
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    cols = np.empty((oh, ow, kh, kw, C), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, ky, kx, :] = xp[ky : ky + sh * oh : sh, kx : kx + sw * ow : sw, :]
    if channel_major:
        cols = cols.transpose(0, 1, 4, 2, 3)
    return cols.reshape(oh * ow, -1), (oh, ow)


def lower_op(op, t, consts):
    """Returns a job {terms: [(X, W)], bias, act, shape} for a compute op, given its input tensors t."""
    kind = op["kind"]
    x = t[op["inputs"][0]]
    if kind == "CONV_2D":
        w = consts[op["inputs"][1]]  # Cout x kh x kw x Cin
        b = consts[op["inputs"][2]]
        X, (oh, ow) = im2col(x[0], w.shape[1], w.shape[2], op["stride"], op["same"])
        return {"terms": [(X, w.reshape(w.shape[0], -1).T.copy())], "bias": b, "act": op["act"], "shape": (1, oh, ow, w.shape[0])}
    if kind == "DEPTHWISE_CONV_2D":
        w = consts[op["inputs"][1]]  # 1 x kh x kw x C
        b = consts[op["inputs"][2]]
        C, kk = w.shape[3], w.shape[1] * w.shape[2]
        if x.shape[3] != C:
            raise ValueError("depth multiplier other than 1 is not supported")
        X, (oh, ow) = im2col(x[0], w.shape[1], w.shape[2], op["stride"], op["same"], channel_major=True)
        Wd = np.zeros((C * kk, C), dtype=np.float32)  # block diagonal: channel c's taps feed only output c
        taps = w.reshape(kk, C)
        for c in range(C):
            Wd[c * kk : (c + 1) * kk, c] = taps[:, c]
        return {"terms": [(X, Wd)], "bias": b, "act": op["act"], "shape": (1, oh, ow, C), "kk": kk}
    if kind == "FULLY_CONNECTED":
        w = consts[op["inputs"][1]]  # Dout x Din
        b = consts[op["inputs"][2]] if len(op["inputs"]) > 2 and op["inputs"][2] >= 0 else np.zeros(w.shape[0], np.float32)
        X = x.reshape(-1, w.shape[1])
        return {"terms": [(X, w.T.copy())], "bias": b, "act": op["act"], "shape": (X.shape[0], w.shape[0])}
    if kind == "AVERAGE_POOL_2D":
        _, H, W, C = x.shape
        if op["filter"] != (H, W):
            raise ValueError("only global average pooling is supported")
        ones = np.full((1, H * W), 1.0 / (H * W), dtype=np.float32)  # the mean as a matmul: (1 x P) @ (P x C)
        return {"terms": [(ones, x[0].reshape(H * W, C))], "bias": None, "act": op["act"], "shape": (1, 1, 1, C), "pool": True}
    raise ValueError(kind)


def fuse_add(add_op, producers, t, consts, consumers):
    """An ADD of conv outputs and tensors becomes one accumulate group: conv terms, identity terms for plain tensors, one activation."""
    terms, bias, shape = [], None, None
    for ti in add_op["inputs"]:
        p = producers.get(ti)
        if p is not None and p["kind"] in ("CONV_2D", "DEPTHWISE_CONV_2D") and p["act"] == "linear" and consumers[ti] == 1:
            job = lower_op(p, t, consts)
            terms += job["terms"]
            bias = job["bias"] if bias is None else bias + job["bias"]
            shape = job["shape"]
        else:
            v = t[ti]
            C = v.shape[-1]
            terms.append((v.reshape(-1, C), np.eye(C, dtype=np.float32)))
            shape = v.shape
    return {"terms": terms, "bias": bias, "act": add_op["act"], "shape": shape}


def job_reference(job):
    """Float64 value of a job before its activation is applied, and after."""
    y = sum(X.astype(np.float64) @ W.astype(np.float64) for X, W in job["terms"])
    if job["bias"] is not None:
        y = y + job["bias"].astype(np.float64)
    return y, (np.maximum(y, 0) if job["act"] == "relu" else y)


# =============================================================================
# Tiling into N x N sets
# =============================================================================


def tile_job(job, N):
    """Accumulate groups, one per N x N output tile, in order: (tile, passes, bias); all-zero weight tiles are skipped."""
    terms = [(X.astype(np.float32), W.astype(np.float32)) for X, W in job["terms"]]
    has_bias = job["bias"] is not None and np.any(job["bias"])
    if has_bias and not HW_BIAS:
        X0, W0 = terms[0]
        terms[0] = (np.hstack([X0, np.ones((X0.shape[0], 1), np.float32)]), np.vstack([W0, job["bias"][None, :]]))
    P, C = terms[0][0].shape[0], terms[0][1].shape[1]
    rt, ct = -(-P // N), -(-C // N)
    padded = []
    for X, W in terms:
        D = X.shape[1]
        dt = -(-D // N)
        Xp = np.zeros((rt * N, dt * N), np.float32)
        Xp[:P, :D] = X
        Wp = np.zeros((dt * N, ct * N), np.float32)
        Wp[:D, :C] = W
        padded.append((Xp, Wp, dt))
    groups = []
    for r in range(rt):
        for c in range(ct):
            passes = []
            for Xp, Wp, dt in padded:
                for d in range(dt):
                    B = Wp[d * N : (d + 1) * N, c * N : (c + 1) * N]
                    if not B.any():
                        continue
                    passes.append((Xp[r * N : (r + 1) * N, d * N : (d + 1) * N], B))
            if not passes:
                passes.append((np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)))
            bias = None
            if has_bias and HW_BIAS:
                bias = np.zeros(N, np.float32)
                cols = job["bias"][c * N : (c + 1) * N]
                bias[: cols.size] = cols
            groups.append(((r, c), passes, bias))
    return groups, (P, C, rt, ct)


def write_sets(path, groups, act):
    code = ACT_CODE[act]
    lines = []
    n = 0
    for _, passes, bias in groups:
        for k, (A, B) in enumerate(passes):
            partial = int(k < len(passes) - 1)
            with_bias = int(bias is not None and k == 0)  # the first pass carries the group's bias
            lines.append(f"{partial} {code} 0 {with_bias}")
            parts = [A.ravel(), B.ravel()] + ([bias] if with_bias else [])
            words = np.concatenate(parts).astype(np.float32).view(np.uint32)
            lines.append("\n".join(f"{v:08x}" for v in words.tolist()))
            n += 1
    with open(path, "w") as f:
        f.write(f"{n}\n" + "\n".join(lines) + "\n")
    return n


def read_outputs(path):
    sets = []
    cur = None
    for line in open(path):
        if line.startswith("S "):
            cur = []
            sets.append(cur)
        else:
            cur.append(int(line, 16))
    return [np.array(s, dtype=np.uint32).view(np.float32) for s in sets]


# =============================================================================
# Execution
# =============================================================================


class Sim:
    """The TB_sienna_model binary, built once per configuration and run once per layer."""

    def __init__(self, N, lanes, work, host_gaps=False):
        self.N, self.lanes, self.work = N, lanes, work
        self.host_gaps = host_gaps  # TB_sienna_top's handshake instead of a streaming host
        self.bin = os.path.join(ROOT, "Verilator", "TB_sienna_model_sim")
        self.cycles = 0
        self.sets = 0

    def build(self):
        t = next(x for x in regression.PIPELINE_TESTS if x["name"] == "matmul_relu_nopool")
        regression.generate_vectors({"n": self.N, "tile_size": 4, "lanes": self.lanes, "host_words": self.N, **t})
        r = subprocess.run(["make", "verilator", "TOP_MODULE=TB_sienna_model", "TESTBENCH=TB_sienna_model.sv", "TRACE=0"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(self.bin):
            sys.stdout.write(r.stdout[-4000:] + r.stderr[-4000:])
            raise RuntimeError("TB_sienna_model build failed")

    def run(self, groups, act, tag):
        sets_f = os.path.join(self.work, f"{tag}.sets")
        out_f = os.path.join(self.work, f"{tag}.out")
        n = write_sets(sets_f, groups, act)
        r = subprocess.run([self.bin, f"+sets={sets_f}", f"+out={out_f}"] + (["+host_gaps"] if self.host_gaps else []),
                           cwd=os.path.dirname(self.bin),
                           capture_output=True, text=True)
        m = re.search(r"\[MODEL\] sets=(\d+) outputs=(\d+) cycles=(\d+) mesh_busy=(\d+) act_busy=(\d+) order_errors=(\d+)", r.stdout)
        if not m or int(m.group(6)) != 0:
            sys.stdout.write(r.stdout[-3000:])
            raise RuntimeError(f"{tag}: simulation failed")
        outs = read_outputs(out_f)
        os.remove(sets_f)
        os.remove(out_f)
        cyc = int(m.group(3))
        self.cycles += cyc
        self.sets += n
        return outs, n, cyc


class EmuSim(Sim):
    """Numpy stand-in for the RTL with the same set stream: checks tiling and reassembly, not the hardware."""

    def build(self):
        pass

    def run(self, groups, act, tag):
        outs, n = [], 0
        for _, passes, bias in groups:
            acc = None
            for k, (A, B) in enumerate(passes):
                p = A.astype(np.float32) @ B.astype(np.float32)
                if bias is not None and k == 0:
                    p = (p + bias[None, :]).astype(np.float32)
                acc = p if acc is None else (acc + p).astype(np.float32)
                n += 1
                outs.append(np.zeros(0, np.float32))
            outs[-1] = (np.maximum(acc, 0) if act == "relu" else acc).ravel()
        self.sets += n
        return outs, n, 0


def run_job_hw(job, sim, tag):
    N = sim.N
    groups, (P, C, rt, ct) = tile_job(job, N)
    outs, n, cyc = sim.run(groups, job["act"], tag)
    Y = np.zeros((rt * N, ct * N), np.float32)
    k = 0
    for (r, c), passes, _ in groups:
        k += len(passes) - 1  # partial sets complete with no output
        o = outs[k]
        k += 1
        if o.size != N * N:
            raise RuntimeError(f"{tag}: tile {r},{c} returned {o.size} words")
        Y[r * N : (r + 1) * N, c * N : (c + 1) * N] = o.reshape(N, N)
    return Y[:P, :C], n, cyc


WC_TILES = 128  # sienna_layer's weight cache; a column block with more than half of it is streamed with its sets
ACT_CODES = {"linear": 5, "relu": 4, "selu": 1, "sigmoid": 2, "tanh": 3}


def format_layer(job, N):
    """The configuration and the two input streams sienna_layer expects for a job, in its fixed order.

    Terms whose weight is an identity become the residual input; the rest are one product with their depths side by side.
    A depthwise layer gives each column block only its own channels' depth."""
    res = [X for X, W in job["terms"] if W.shape[0] == W.shape[1] and np.array_equal(W, np.eye(W.shape[0], dtype=W.dtype))]
    dense = [(X, W) for X, W in job["terms"] if not any(X is r for r in res)]
    if len(res) > 1 or not dense:
        raise ValueError("a layer takes one product and at most one residual input")
    X = np.hstack([x for x, _ in dense]).astype(np.float32)
    W = np.vstack([w for _, w in dense]).astype(np.float32)
    M, C = X.shape[0], W.shape[1]
    kk = job.get("kk")
    kb = kk * min(N, C) if kk else X.shape[1]  # depthwise: the taps of one block's channels, fewer when C < N
    rt, ct, dt = -(-M // N), -(-C // N), -(-kb // N)
    bias = job["bias"] if job["bias"] is not None and np.any(job["bias"]) else None
    cached = rt > 1 and dt <= WC_TILES // 2

    def pad(a, rows, cols):
        out = np.zeros((rows, cols), np.float32)
        out[: a.shape[0], : a.shape[1]] = a
        return out

    Xp = pad(X, rt * N, max(X.shape[1], ct * N * (kk or 0)) if kk else dt * N)
    Wp = pad(W, max(W.shape[0], ct * N * (kk or 0)) if kk else dt * N, ct * N)
    Rp = pad(res[0], rt * N, ct * N) if res else None
    bp = np.zeros(ct * N, np.float32)
    if bias is not None:
        bp[: bias.size] = bias
    a_rows, w_rows = [], []
    for c in range(ct):
        k0 = c * N * kk if kk else 0  # depthwise: this block's channels only
        A_c = pad(Xp[:, k0 : k0 + kb], rt * N, dt * N)
        W_c = pad(Wp[k0 : k0 + kb, c * N : (c + 1) * N], dt * N, N)
        if bias is not None:
            w_rows.append(bp[c * N : (c + 1) * N][None, :])
        w_rows += [W_c] * (1 if cached else rt)
        for r in range(rt):
            for t in range(dt):  # the depth tiles of this row tile, N rows of N words each
                a_rows.append(A_c[r * N : (r + 1) * N, t * N : (t + 1) * N])
            if Rp is not None:
                a_rows.append(Rp[r * N : (r + 1) * N, c * N : (c + 1) * N])
    cfg = dict(m=M, kb=kb, n=C, residual=int(Rp is not None), bias=int(bias is not None), act=ACT_CODES[job["act"]])
    return cfg, np.vstack(a_rows), np.vstack(w_rows), (M, C, rt, ct)


class LayerSim:
    """TB_sienna_layer: one layer per run; software writes the configuration and the streams, then reads the results."""

    def __init__(self, N, lanes, work):
        self.N, self.lanes, self.work = N, lanes, work
        self.bin = os.path.join(ROOT, "Verilator", "TB_sienna_layer_sim")
        self.cycles = self.sets = self.words = 0

    def build(self):
        t = next(x for x in regression.PIPELINE_TESTS if x["name"] == "matmul_relu_nopool")
        regression.generate_vectors({"n": self.N, "tile_size": 4, "lanes": self.lanes, "host_words": self.N, **t})
        r = subprocess.run(["make", "verilator", "TOP_MODULE=TB_sienna_layer", "TESTBENCH=TB_sienna_layer.sv", "TRACE=0"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(self.bin):
            sys.stdout.write(r.stdout[-4000:] + r.stderr[-4000:])
            raise RuntimeError("TB_sienna_layer build failed")

    def run_job(self, job, tag):
        N = self.N
        cfg, a, w, (M, C, rt, ct) = format_layer(job, N)
        lf = os.path.join(self.work, f"{tag}.layer")
        of = os.path.join(self.work, f"{tag}.out")
        with open(lf, "w") as f:
            f.write(f"L {cfg['m']} {cfg['kb']} {cfg['n']} {cfg['residual']} {cfg['bias']} {cfg['act']} 0 0 {len(a)} {len(w)}\n")
            f.write("\n".join(f"{v:08x}" for v in np.concatenate([a.ravel(), w.ravel()]).astype(np.float32).view(np.uint32).tolist()))
            f.write("\n")
        r = subprocess.run([self.bin, f"+layer={lf}", f"+out={of}"], cwd=os.path.dirname(self.bin), capture_output=True, text=True)
        m = re.search(r"\[LAYER\] sets=(\d+) outputs=(\d+) cycles=(\d+) a_rows=(\d+)/(\d+) w_rows=(\d+)/(\d+)", r.stdout)
        if not m or m.group(4) != m.group(5) or m.group(6) != m.group(7):
            sys.stdout.write(r.stdout[-3000:])
            raise RuntimeError(f"{tag}: layer simulation failed")
        outs = read_outputs(of)
        os.remove(lf)
        os.remove(of)
        if len(outs) != rt * ct:
            raise RuntimeError(f"{tag}: {len(outs)} output tiles, expected {rt * ct}")
        Y = np.zeros((rt * N, ct * N), np.float32)
        k = 0
        for c in range(ct):  # the spec's output order: column blocks outer, row tiles inner
            for r_ in range(rt):
                Y[r_ * N : (r_ + 1) * N, c * N : (c + 1) * N] = outs[k].reshape(N, N)
                k += 1
        n, cyc = int(m.group(1)), int(m.group(3))
        self.sets += n
        self.cycles += cyc
        self.words += (len(a) + len(w)) * N
        return Y[:M, :C], n, cyc


def macs_of(job):
    """Multiply-accumulates the model defines: structural zeros and identity (residual) passes not counted."""
    if job.get("pool"):
        return int(job["terms"][0][1].size)  # one add per input element
    n = 0
    for X, W in job["terms"]:
        if W.shape[0] == W.shape[1] and np.array_equal(W, np.eye(W.shape[0], dtype=W.dtype)):
            continue
        n += X.shape[0] * np.count_nonzero(W)
    return int(n)


def execute(model, x, sim=None, log=None):
    """Runs the graph. With sim, compute ops run on the RTL and outputs feed the next layer; otherwise float64 reference."""
    ops, consts = model["ops"], model["consts"]
    t = dict(consts)
    t[model["input"]] = x.astype(np.float32)
    producers = {o: op for op in ops for o in op["outputs"]}
    consumers = {}
    for op in ops:
        for i in op["inputs"]:
            consumers[i] = consumers.get(i, 0) + 1
    fused = set()
    for op in ops:
        if op["kind"] == "ADD":
            for ti in op["inputs"]:
                p = producers.get(ti)
                if p is not None and p["kind"] in ("CONV_2D", "DEPTHWISE_CONV_2D") and p["act"] == "linear" and consumers[ti] == 1:
                    fused.add(id(p))
    stats = []
    for li, op in enumerate(ops):
        kind = op["kind"]
        out = op["outputs"][0]
        if id(op) in fused:
            continue  # computed inside the ADD that consumes it
        if kind == "RESHAPE":
            t[out] = t[op["inputs"][0]].reshape(model["shapes"][out])
            continue
        if kind == "SOFTMAX":
            v = t[op["inputs"][0]].astype(np.float64)
            e = np.exp(v - v.max(axis=-1, keepdims=True))
            t[out] = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)  # host
            continue
        job = fuse_add(op, producers, t, consts, consumers) if kind == "ADD" else lower_op(op, t, consts)
        _, ref = job_reference(job)
        if sim is None:
            y = ref.astype(np.float32)
            n = cyc = 0
        elif isinstance(sim, LayerSim):
            y, n, cyc = sim.run_job(job, f"L{li:02d}")
        else:
            y, n, cyc = run_job_hw(job, sim, f"L{li:02d}")
        scale = float(np.max(np.abs(ref))) or 1.0
        err = float(np.max(np.abs(y.astype(np.float64) - ref))) / scale
        stats.append({"layer": li, "kind": kind, "sets": n, "cycles": cyc, "macs": macs_of(job),
                      "passes": len(job["terms"]), "shape": list(job["shape"]), "err": err})
        if log:
            log(f"    L{li:02d} {kind:<18} {str(job['shape']):<18} sets {n:6d}  cycles {cyc:8d}  MACs {stats[-1]['macs']:9d}  max err/max|ref| {err:.2e}")
        t[out] = y.reshape(job["shape"]).astype(np.float32)
    return t[model["output"]], stats


# =============================================================================
# Models and inputs
# =============================================================================


def cifar_test(data_dir):
    import pickle
    d = pickle.load(open(os.path.join(data_dir, "cifar-10-batches-py", "test_batch"), "rb"), encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)  # raw 0..255, as train.py feeds it
    return x, np.array(d[b"labels"])


def kws_mfcc(wav_path):
    """49 x 10 MFCCs as the MLPerf Tiny KWS get_dataset.py computes them with tf.signal, redone in numpy."""
    import wave
    w = wave.open(wav_path)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())[:, 0].astype(np.float32)
    a = np.pad(a / a.max(), (0, max(0, 16000 - a.size)))[:16000]  # scaled by the max, as reduce_max does
    frame, step, nfft = 480, 320, 512  # 30 ms window, 20 ms stride, next power of two
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)  # periodic Hann
    frames = np.stack([a[i : i + frame] * win for i in range(0, a.size - frame + 1, step)])
    spec = np.abs(np.fft.rfft(frames, nfft))
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    bins = mel(np.linspace(0, 8000, nfft // 2 + 1)[1:])[:, None]
    edges = np.linspace(mel(20.0), mel(4000.0), 42)
    lo, ce, hi = edges[:-2], edges[1:-1], edges[2:]
    wts = np.maximum(0, np.minimum((bins - lo) / (ce - lo), (hi - bins) / (hi - ce)))
    wts = np.vstack([np.zeros((1, 40)), wts])  # the DC bin carries no weight
    logmel = np.log(spec @ wts + 1e-6)
    n = np.arange(40)
    dct = 2 * np.cos(np.pi * np.outer(2 * n + 1, np.arange(40)) / 80)  # unnormalized DCT-II
    mfcc = (logmel @ dct) / np.sqrt(80.0)
    return mfcc[:, :10].reshape(1, 49, 10, 1).astype(np.float32)


def model_inputs(name, mdir, count, seed):
    """Real inputs where MLPerf Tiny ships them; otherwise seeded synthetic inputs, flagged as such."""
    rng = np.random.RandomState(seed)
    if name == "resnet8":
        x, y = cifar_test(os.path.join(mdir, "data"))
        idx = np.load(os.path.join(mdir, "perf_samples_idxs.npy"))[:count]
        return [(x[i : i + 1], int(y[i]), f"cifar10_test[{i}]") for i in idx], "CIFAR-10 test images (MLPerf Tiny perf-sample indices)"
    if name == "ad01":
        v = np.fromfile(os.path.join(mdir, "normal_id_01_00000000_hist_librosa.bin"), dtype="<f4").reshape(-1, 640)
        runs = [(v[i : i + 1], None, f"dcase01 slice {i}") for i in range(min(count, len(v)))]
        runs.append((v, None, f"dcase01 all {len(v)} slices as one batch"))  # rows fill the mesh tiles
        return runs, "ToyCar normal_id_01 spectrogram slices shipped with MLPerf Tiny"
    if name == "kws":
        runs = [(kws_mfcc(os.path.join(mdir, "marvin_617de221_0.wav")), 11, "MLPerf runner clip 'marvin' (not a keyword: label Unknown)")]
        runs += [(rng.normal(0, 10, (1, 49, 10, 1)).astype(np.float32), None, f"synthetic seed {seed} #{i}") for i in range(count - 1)]
        return runs, "one real speech clip with numpy MFCCs, the rest synthetic (no Speech Commands set here)"
    if name == "vww":
        return [(rng.uniform(0, 1, (1, 96, 96, 3)).astype(np.float32), None, f"synthetic seed {seed} #{i}") for i in range(count)], \
            "synthetic images in [0, 1] (no COCO person set available here)"
    raise ValueError(name)


MODELS = {
    "resnet8": "pretrainedResnet.tflite",
    "ad01": "ad01_fp32.tflite",
    "kws": "kws_ref_model_float32.tflite",
    "vww": "vww_96_float.tflite",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="resnet8,ad01,kws,vww")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--count", type=int, default=1, help="inferences per model on the RTL")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "models"))
    ap.add_argument("--ref-accuracy", type=int, default=0, help="CIFAR-10 test images for the float reference accuracy")
    ap.add_argument("--no-sim", action="store_true", help="float reference only")
    ap.add_argument("--emulate", action="store_true", help="numpy stand-in for the RTL, to check the lowering")
    ap.add_argument("--engine", choices=("layer", "sets"), default="layer",
                    help="layer: sienna_layer schedules everything; sets: the host drives sienna_top set by set")
    ap.add_argument("--host-gaps", action="store_true", help="host idles a cycle after each load and waits for the credit")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    report = os.path.join(a.work, f"model_report_N{a.n}.log")
    js = os.path.join(a.work, f"model_results_N{a.n}.json")
    rep = open(report, "w")

    def log(s):
        print(s, flush=True)
        rep.write(s + "\n")
        rep.flush()

    sim = None
    if not a.no_sim:
        if a.emulate:
            sim = EmuSim(a.n, a.lanes, a.work, a.host_gaps)
        elif a.engine == "layer":
            sim = LayerSim(a.n, a.lanes, a.work)
        else:
            sim = Sim(a.n, a.lanes, a.work, a.host_gaps)
        t0 = time.time()
        sim.build()
        log(f"built {os.path.basename(sim.bin)[:-4] if hasattr(sim, 'bin') else 'emulator'} N={a.n} lanes={a.lanes} in {time.time() - t0:.0f} s")
    results = {}
    for name in a.models.split(","):
        model = load_tflite(os.path.join(a.model_dir, MODELS[name]))
        if name == "resnet8" and a.ref_accuracy:
            x, y = cifar_test(os.path.join(a.model_dir, "data"))
            hits = sum(int(np.argmax(execute(model, x[i : i + 1])[0]) == y[i]) for i in range(a.ref_accuracy))
            log(f"{name}: float64 reference top-1 on the first {a.ref_accuracy} CIFAR-10 test images: {hits}/{a.ref_accuracy} = {100 * hits / a.ref_accuracy:.1f}%")
            results.setdefault(name, {})["ref_top1"] = [hits, a.ref_accuracy]
        inputs, source = model_inputs(name, a.model_dir, a.count, 7)
        log(f"\n== {name} ({MODELS[name]}), inputs: {source}")
        runs = []
        for x, label, desc in inputs:
            ref, _ = execute(model, x)
            if sim is None:
                runs.append({"input": desc, "label": label, "ref_top": int(np.argmax(ref))})
                continue
            c0, s0, w0, t0 = sim.cycles, sim.sets, getattr(sim, "words", 0), time.time()
            log(f"  inference on {desc}")
            hw, stats = execute(model, x, sim, log)
            cyc, sets, hwords = sim.cycles - c0, sim.sets - s0, getattr(sim, "words", 0) - w0
            macs = sum(s["macs"] for s in stats)
            diff = float(np.max(np.abs(hw.astype(np.float64) - ref)))
            r = {"input": desc, "label": label, "ref_top": int(np.argmax(ref)), "hw_top": int(np.argmax(hw)), "cycles": cyc,
                 "host_words": hwords,
                 "sets": sets, "macs": macs, "max_abs_out_diff": diff, "wall_s": time.time() - t0, "layers": stats,
                 "ref_out": ref.ravel().tolist()[:16], "hw_out": hw.ravel().tolist()[:16]}
            if name == "ad01":  # the anomaly score is the reconstruction error of each slice
                r["score_ref"] = np.mean((x.astype(np.float64) - ref) ** 2, axis=1).tolist()
                r["score_hw"] = np.mean((x.astype(np.float64) - hw) ** 2, axis=1).tolist()
                log(f"  anomaly score (MSE) ref {np.mean(r['score_ref']):.6f} hw {np.mean(r['score_hw']):.6f}")
            runs.append(r)
            util = macs / (sets * a.n ** 3) if sets else 0
            log(f"  result: hw top {r['hw_top']} ref top {r['ref_top']} label {label}  max |hw-ref| on outputs {diff:.2e}  "
                f"{cyc} cycles = {cyc / 950e3:.3f} ms @950 MHz (assumed)  {sets} sets  {hwords} host words  {macs} MACs  MAC-slot use {100 * util:.1f}%  wall {r['wall_s']:.0f} s")
        results.setdefault(name, {})["runs"] = runs
        results[name]["source"] = source
        json.dump(results, open(js, "w"), indent=1)
    log(f"\nreport {report}\nresults {js}")


if __name__ == "__main__":
    main()
