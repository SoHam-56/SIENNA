#!/usr/bin/env python3
"""Gate G0: single-layer int8 TFLite models on the interpreter's BUILTIN_REF kernels against tflite_ref in both roundings; pins the rounding."""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_ref as ref  # noqa: E402

FC_IN, FC_OUT = 64, 16
CONV_HW, CONV_CIN, CONV_COUT = 8, 16, 16
MODELS = {  # name: (layer, fused activation); seed 0 of each is saved for G4
    "fc64x16_linear": ("fc", "none"),
    "fc64x16_relu": ("fc", "relu"),
    "conv3x3_8x8x16_linear": ("conv", "none"),
    "conv3x3_8x8x16_relu6": ("conv", "relu6"),
}
OP_NAME = {"fc": "FULLY_CONNECTED", "conv": "CONV_2D"}
ACT_FN = {"none": tf.identity, "relu": tf.nn.relu, "relu6": tf.nn.relu6}
ROUNDINGS = ("DOUBLE", "SINGLE")
REF = tf.lite.experimental.OpResolverType.BUILTIN_REF
MIN_DISCRIMINATING = 100  # outputs where the two roundings differ: the pick must rest on evidence
MIN_UNCLAMPED = 0.25  # per model: the comparison must not be dominated by saturated outputs
N_SAVED = 64  # test inputs and interpreter outputs kept per npz for G4


def build_model(layer, act, seed):
    """One layer: per-channel weight magnitudes over two decades; input range with max >= 1.5 |min|, so the zero point is not 0."""
    rng = np.random.default_rng(seed)
    lo = -rng.uniform(0.2, 2.0)
    hi = -lo * rng.uniform(1.5, 4.0)
    cout = FC_OUT if layer == "fc" else CONV_COUT
    sig = np.exp(rng.uniform(np.log(0.01), np.log(1.0), cout))
    if layer == "fc":
        in_shape = (1, FC_IN)
        w = rng.standard_normal((FC_IN, cout)) * sig
    else:
        in_shape = (1, CONV_HW, CONV_HW, CONV_CIN)
        w = rng.standard_normal((3, 3, CONV_CIN, cout)) * sig
    wc = tf.constant(w.astype(np.float32))
    bc = tf.constant((rng.standard_normal(cout) * 0.5).astype(np.float32))
    act_fn = ACT_FN[act]

    @tf.function(input_signature=[tf.TensorSpec(in_shape, tf.float32)])
    def layer_fn(x):
        y = tf.matmul(x, wc) if layer == "fc" else tf.nn.conv2d(x, wc, strides=1, padding="SAME")
        return act_fn(tf.nn.bias_add(y, bc))

    rep_rng = np.random.default_rng(seed + 1000)

    def representative():
        for _ in range(200):
            yield [rep_rng.uniform(lo, hi, in_shape).astype(np.float32)]

    holder = tf.Module()
    holder.layer_fn = layer_fn
    conv = tf.lite.TFLiteConverter.from_concrete_functions([layer_fn.get_concrete_function()], holder)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert(), lo, hi


def interpreter(model, resolver):
    it = tf.lite.Interpreter(model_content=model, experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    return it


def invoke_all(it, xs):
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    ys = []
    for x in xs:
        it.set_tensor(i, x[None].astype(np.int8))
        it.invoke()
        ys.append(it.get_tensor(o)[0].astype(np.int64))
    return np.stack(ys)


def extract(it, layer, act):
    """The op's tensors and quantization; stops unless the graph is one int8 FULLY_CONNECTED or CONV_2D with fused activation."""
    ops = it._get_ops_details()  # private in TF 2.x: op_name, inputs, outputs per op
    names = [o["op_name"] for o in ops]
    if names != [OP_NAME[layer]]:
        sys.exit(f"G0: FAIL, expected one {OP_NAME[layer]}, the converter produced {names}")
    xi, wi, bi = (int(i) for i in ops[0]["inputs"][:3])
    oi = int(ops[0]["outputs"][0])
    td = {t["index"]: t for t in it.get_tensor_details()}
    qp = {i: td[i]["quantization_parameters"] for i in (xi, wi, bi, oi)}
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    if (inp["index"], out["index"]) != (xi, oi) or td[xi]["dtype"] != np.int8 or td[oi]["dtype"] != np.int8:
        sys.exit("G0: FAIL, the layer's input and output are not the graph's int8 input and output")
    if td[wi]["dtype"] != np.int8 or td[bi]["dtype"] != np.int32 or np.any(qp[wi]["zero_points"] != 0):
        sys.exit("G0: FAIL, weights are not symmetric int8 or the bias is not int32")
    p = {"layer": layer, "activation": act, "in_shape": tuple(int(d) for d in inp["shape"]),
         "w_q": it.get_tensor(wi).astype(np.int64), "b_q": it.get_tensor(bi).astype(np.int64),
         "w_scales": qp[wi]["scales"].astype(np.float32),
         "in_scale": np.float32(qp[xi]["scales"][0]), "in_zp": int(qp[xi]["zero_points"][0]),
         "out_scale": np.float32(qp[oi]["scales"][0]), "out_zp": int(qp[oi]["zero_points"][0])}
    p["amin"], p["amax"] = ref.activation_range(act, p["out_scale"], p["out_zp"])
    return p


def reference(p, xs, rounding, folded=False, scale_product=None):
    args = (xs, p["w_q"], p["b_q"], p["in_zp"], p["w_scales"], p["in_scale"], p["out_scale"], p["out_zp"], p["amin"],
            p["amax"], rounding)
    if p["layer"] == "fc":
        return ref.fc_int8(*args, folded=folded, scale_product=scale_product).astype(np.int64)
    return ref.conv2d_int8(*args, folded=folded).astype(np.int64)


def test_inputs(p, lo, hi, n, rng):
    """Mostly the calibration distribution quantized with the model's input parameters; a tenth uniform over all of int8."""
    shape = (n,) + p["in_shape"][1:]
    xq = np.clip(np.round(rng.uniform(lo, hi, shape) / p["in_scale"]) + p["in_zp"], -128, 127).astype(np.int64)
    k = n // 10
    xq[:k] = rng.integers(-128, 128, (k,) + shape[1:])
    return xq


def save(out, name, model, p, xs, ys, rounding):
    with open(os.path.join(out, name + ".tflite"), "wb") as f:
        f.write(model)
    mults, shifts = ref.layer_multipliers(p["layer"], p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], rounding)
    np.savez(os.path.join(out, name + ".npz"), layer=p["layer"], activation=p["activation"], in_shape=np.array(p["in_shape"]),
             w_q=p["w_q"].astype(np.int8), b_q=p["b_q"].astype(np.int32), w_scales=p["w_scales"], in_scale=p["in_scale"],
             in_zp=p["in_zp"], out_scale=p["out_scale"], out_zp=p["out_zp"], act_min=p["amin"], act_max=p["amax"],
             mults=mults, shifts=shifts, x_test=xs.astype(np.int8), y_test=ys.astype(np.int8), rounding=rounding,
             tf_version=tf.__version__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="directory for the G4 models, npz files and rounding files")
    ap.add_argument("--report", required=True, help="G0 report (.log)")
    ap.add_argument("--seeds", type=int, default=8, help="models per kind; seed 0 is saved")
    ap.add_argument("--inputs", type=int, default=1000, help="random test inputs per model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    lines = [f"G0 TFLite oracle: TensorFlow {tf.__version__}, numpy {np.__version__}, op resolver BUILTIN_REF, "
             f"{a.seeds} seeds x {len(MODELS)} models, {a.inputs} inputs each"]
    tot = {"outputs": 0, "DOUBLE": 0, "SINGLE": 0, "disc": 0, "folded": 0}
    problems, keep = [], {}
    for name, (layer, act) in MODELS.items():
        for seed in range(a.seeds):
            model, lo, hi = build_model(layer, act, seed)
            it = interpreter(model, REF)
            p = extract(it, layer, act)
            xs = test_inputs(p, lo, hi, a.inputs, np.random.default_rng(seed + 2000))
            ys = invoke_all(it, xs)
            ys_default = invoke_all(interpreter(model, tf.lite.experimental.OpResolverType.AUTO), xs)
            want = {r: reference(p, xs, r) for r in ROUNDINGS}
            mism = {r: int(np.sum(ys != want[r])) for r in ROUNDINGS}
            dflt = {r: int(np.sum(ys_default != want[r])) for r in ROUNDINGS}
            disc = int(np.sum(want["DOUBLE"] != want["SINGLE"]))
            fold = sum(int(np.sum(reference(p, xs, r, folded=True) != want[r])) for r in ROUNDINGS)
            unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
            _, shifts = ref.layer_multipliers(layer, p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], "DOUBLE")
            per_ch = p["w_scales"].size > 1
            lines.append(f"{name} seed {seed}: in scale {p['in_scale']:.6g} zp {p['in_zp']}, out scale {p['out_scale']:.6g} "
                         f"zp {p['out_zp']}, act [{p['amin']}, {p['amax']}], weights "
                         f"{'per-channel' if per_ch else 'per-tensor'} ({p['w_scales'].size}), shifts [{shifts.min()}, "
                         f"{shifts.max()}], outputs {ys.size}, mismatches DOUBLE {mism['DOUBLE']} SINGLE {mism['SINGLE']}, "
                         f"discriminating {disc}, folded {fold}, unclamped {100 * unclamped:.1f}%, "
                         f"default resolver (info) DOUBLE {dflt['DOUBLE']} SINGLE {dflt['SINGLE']}")
            if layer == "fc" and not per_ch and min(mism.values()) > 0:
                alt = {r: int(np.sum(ys != reference(p, xs, r, scale_product="double"))) for r in ROUNDINGS}
                lines.append(f"  diagnostic: per-tensor FC with a double scale product: DOUBLE {alt['DOUBLE']} SINGLE {alt['SINGLE']}")
            if p["in_zp"] == 0:
                problems.append(f"{name} seed {seed}: input zero point is 0")
            if layer == "conv" and (not per_ch or np.unique(p["w_scales"]).size < 2):
                problems.append(f"{name} seed {seed}: conv weights are not per-channel")
            if unclamped < MIN_UNCLAMPED:
                problems.append(f"{name} seed {seed}: only {100 * unclamped:.1f}% of outputs unclamped")
            tot["outputs"] += ys.size
            tot["disc"] += disc
            tot["folded"] += fold
            for r in ROUNDINGS:
                tot[r] += mism[r]
            if seed == 0:
                keep[name] = (model, p, xs[:N_SAVED], ys[:N_SAVED])
    lines.append(f"totals: outputs {tot['outputs']}, mismatches DOUBLE {tot['DOUBLE']} SINGLE {tot['SINGLE']}, "
                 f"discriminating {tot['disc']}, folded {tot['folded']}")
    winners = [r for r in ROUNDINGS if tot[r] == 0]
    if len(winners) != 1:
        problems.append(f"{len(winners)} roundings match every output; exactly one must")
    if tot["disc"] < MIN_DISCRIMINATING:
        problems.append(f"only {tot['disc']} discriminating outputs (need {MIN_DISCRIMINATING})")
    if tot["folded"]:
        problems.append(f"SIENNA's folded zero-point algebra differs from TFLite's on {tot['folded']} outputs")
    if problems:
        lines += [f"problem: {m}" for m in problems] + ["G0: FAIL"]
    else:
        rounding = winners[0]
        for name, (model, p, xs, ys) in keep.items():
            save(a.out, name, model, p, xs, ys, rounding)
        with open(os.path.join(a.out, "rounding.txt"), "w") as f:
            f.write(rounding + "\n")
        lines += [f"ROUNDING: {rounding}", "G0: PASS"]
    with open(a.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
