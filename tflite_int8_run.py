#!/usr/bin/env python3
"""Runs testbenches/tflite_int8/'s single-layer int8 models through sienna_layer in int8, bit for bit against the interpreter's saved outputs."""
import argparse
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_runner as mr  # noqa: E402

reg = mr
MODEL_DIR = os.path.join(ROOT, "testbenches", "tflite_int8")


def load_layer(path: str) -> dict:
    """The one CONV_2D or FULLY_CONNECTED operator of an int8 .tflite, with its tensors' codes and quantization."""
    import tflite
    from tflite.ActivationFunctionType import ActivationFunctionType as AF
    from tflite.BuiltinOperator import BuiltinOperator as BO

    names = {v: k for k, v in BO.__dict__.items() if not k.startswith("_")}
    m = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
    g = m.Subgraphs(0)
    code = lambda op: m.OperatorCodes(op.OpcodeIndex())
    kinds = [names[max(code(g.Operators(i)).BuiltinCode(), code(g.Operators(i)).DeprecatedBuiltinCode())]
             for i in range(g.OperatorsLength())]
    if kinds not in (["CONV_2D"], ["FULLY_CONNECTED"]):
        raise ValueError(f"{path}: operators {kinds}; expected one CONV_2D or FULLY_CONNECTED with int8 input and output")
    op = g.Operators(0)

    def tensor(i):
        if i < 0:
            return None
        t = g.Tensors(i)
        q = t.Quantization()
        buf = m.Buffers(t.Buffer()).DataAsNumpy()
        data = None
        if not isinstance(buf, int) and buf is not None and len(buf):
            data = np.frombuffer(buf.tobytes(), dtype={9: np.int8, 2: np.int32}[t.Type()]).reshape(tuple(t.ShapeAsNumpy()))
        return dict(type=t.Type(), shape=tuple(t.ShapeAsNumpy()), data=data,
                    scale=np.atleast_1d(q.ScaleAsNumpy()).astype(np.float32),
                    zp=np.atleast_1d(q.ZeroPointAsNumpy()).astype(np.int64))

    ins = [op.Inputs(j) for j in range(op.InputsLength())]
    inp, flt = tensor(ins[0]), tensor(ins[1])
    bias = tensor(ins[2]) if len(ins) > 2 else None
    out = tensor(op.Outputs(0))
    if inp["type"] != 9 or flt["type"] != 9 or out["type"] != 9:  # 9 is INT8
        raise ValueError(f"{path}: input, filter and output must be int8")
    if np.any(flt["zp"] != 0):
        raise ValueError(f"{path}: TFLite's int8 filters are symmetric; a non-zero filter zero point is not supported")
    opt = (tflite.Conv2DOptions if kinds[0] == "CONV_2D" else tflite.FullyConnectedOptions)()
    t = op.BuiltinOptions()
    opt.Init(t.Bytes, t.Pos)
    d = dict(kind=kinds[0], input=inp, filter=flt, bias=bias, output=out)
    if kinds[0] == "CONV_2D":
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
            raise ValueError(f"{path}: dilation is not supported")
        d.update(same=opt.Padding() == 0, stride=(opt.StrideH(), opt.StrideW()))
    fa = opt.FusedActivationFunction()
    acts = {AF.NONE: "none", AF.RELU: "relu", AF.RELU6: "relu6"}  # the activations mr.activation_range defines
    if fa not in acts:
        raise ValueError(f"{path}: fused activation {fa} is not supported")
    d["act_range"] = mr.activation_range(acts[fa], out["scale"][0], int(out["zp"][0]))
    return d


def job_of(layer: dict, x: np.ndarray, saved) -> tuple:
    """(int8 job for LayerSim, output shape) of one layer on the interpreter's input codes x; saved is the layer's G0 npz."""
    inp, flt, b, out = layer["input"], layer["filter"], layer["bias"], layer["output"]
    z_in, z_out = int(inp["zp"][0]), int(out["zp"][0])
    w = flt["data"].astype(np.int64)
    cout = w.shape[0]
    if str(saved["rounding"]) != mr.ROUNDING:
        raise ValueError(f"the npz was written for {saved['rounding']}, the pinned rounding is {mr.ROUNDING}")
    mult, shift = np.asarray(saved["mults"], np.int64), np.asarray(saved["shifts"], np.int64)  # G0's words; the recompute below only checks them
    kind = "conv" if layer["kind"] == "CONV_2D" else "fc"
    rm, rs = mr.layer_multipliers(kind, flt["scale"], inp["scale"][0], out["scale"][0], cout, mr.ROUNDING)
    if not (np.array_equal(rm, mult) and np.array_equal(rs, shift)):
        raise ValueError("the .tflite's scales give other multipliers than its npz holds: the model and the npz do not belong together")
    if tuple(layer["act_range"]) != (int(saved["act_min"]), int(saved["act_max"])):
        raise ValueError(f"clamp {layer['act_range']} differs from the npz's ({int(saved['act_min'])}, {int(saved['act_max'])})")
    if layer["kind"] == "CONV_2D":
        cols = [mr.im2col(np.asarray(xi, np.float32), w.shape[1], w.shape[2], layer["stride"], layer["same"],
                          pad_value=float(z_in)) for xi in x]  # every saved input image; rows in (image, y, x) order
        X, (oh, ow) = np.vstack([c for c, _ in cols]), cols[0][1]
        W = w.reshape(cout, -1).T  # OHWI filters: depth in (ky, kx, c) order, as im2col lays it out
        shape = (len(x), oh, ow, cout)
    else:
        W = w.T
        X = np.asarray(x, np.float32).reshape(-1, W.shape[0])
        shape = (X.shape[0], cout)
    bq = np.zeros(cout, np.int64) if b is None or b["data"] is None else b["data"].astype(np.int64)
    hw_bias = reg.wrap32(bq - z_in * W.sum(axis=0))
    amin, amax = layer["act_range"]
    req = dict(mult=mult, shift=shift, zp=z_out, amin=amin, amax=amax, mx=0, shx=0, mout=0, shout=0, zout=0)
    job = {"terms": [(X.astype(np.float32), W.astype(np.float32))], "bias": hw_bias, "act": "linear", "shape": shape,
           "req": req}
    return job, shape


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--models", default=MODEL_DIR, help="<name>.tflite files, each with <name>.npz holding x_test and y_test")
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "int8"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    rep = open(os.path.join(a.work, f"tflite_int8_N{a.n}_T{a.tile_size}.log"), "w")

    def log(s):
        print(s, flush=True)
        rep.write(s + "\n")
        rep.flush()

    paths = sorted(glob.glob(os.path.join(a.models, "*.tflite")))
    if not paths:
        log(f"no .tflite models in {a.models}")
        sys.exit(1)
    sim = mr.LayerSim(a.n, a.lanes, a.work, "int8", a.tile_size)
    t0 = time.time()
    sim.build()
    log(f"TB_model_run built in int8, N={a.n} T={a.tile_size} lanes={a.lanes}, in {time.time() - t0:.0f} s")
    bad, covered = 0, False
    for path in paths:
        name = os.path.basename(path)[:-len(".tflite")]
        ref = np.load(path[:-len(".tflite")] + ".npz")
        x, y = ref["x_test"], ref["y_test"].astype(np.int64)
        layer = load_layer(path)
        job, shape = job_of(layer, x, ref)
        X, W = job["terms"][0]
        low = reg.int8_layer_exact(X.astype(np.int64), W.astype(np.int64), job["bias"], job["req"], "linear").reshape(shape)
        got, sets, cyc = sim.run_job(job, name)
        got = got.reshape(shape)
        want = y.reshape(shape)
        per_channel = layer["filter"]["scale"].size > 1
        z_in = int(layer["input"]["zp"][0])
        padded = layer["kind"] == "CONV_2D" and layer["same"]
        covered |= padded and per_channel and z_in != 0
        m_rtl, m_low = int(np.sum(got != want)), int(np.sum(low != want))
        bad += int(m_rtl != 0)
        log(f"MODEL {name}: {layer['kind']} out {shape} z_in {z_in} {'per-channel' if per_channel else 'per-tensor'} "
            f"{'SAME-padded' if padded else 'unpadded'} clamp {layer['act_range']}: RTL {m_rtl}/{want.size} differ from the "
            f"interpreter, host lowering {m_low}/{want.size}; {sets} sets, {cyc} cycles")
    if not covered:
        log("no SAME-padded per-channel conv with a non-zero input zero point among the models")
        bad += 1
    log(f"TFLITE_INT8: {len(paths)} models, {bad} failing")
    log("RESULT: PASSED" if bad == 0 else "RESULT: FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
