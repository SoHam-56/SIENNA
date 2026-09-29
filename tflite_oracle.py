#!/usr/bin/env python3
"""Gate G0: single-layer int8 TFLite models on the interpreter's BUILTIN_REF kernels against tflite_ref in both roundings; pins the rounding."""
import argparse
import itertools
import os
import sys

import numpy as np
import tensorflow as tf
import tflite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tflite_ref as ref  # noqa: E402

FC_IN, FC_OUT = 64, 16
CONV_HW, CONV_CIN, CONV_COUT = 8, 16, 16
MODELS = {  # name: (layer, fused activation); seed 0 of each is saved for G4
    "fc64x16_linear": ("fc", "none"),
    "fc64x16_relu": ("fc", "relu"),
    "conv3x3_8x8x16_linear": ("conv", "none"),
    "conv3x3_8x8x16_relu6": ("conv", "relu6"),
    "conv3x3_8x8x16_relu6_wide": ("conv", "relu6"),
}
OUT_RANGE = {"conv3x3_8x8x16_relu6_wide": (-1.0, 8.0)}  # output fake-quantized wider than relu6, so the fused clamp is not int8's
FUSED = {"none": tflite.ActivationFunctionType.NONE, "relu": tflite.ActivationFunctionType.RELU,
         "relu6": tflite.ActivationFunctionType.RELU6}
OP_NAME = {"fc": "FULLY_CONNECTED", "conv": "CONV_2D"}
ACT_FN = {"none": tf.identity, "relu": tf.nn.relu, "relu6": tf.nn.relu6}
ROUNDINGS = ("DOUBLE", "SINGLE")
REF = tf.lite.experimental.OpResolverType.BUILTIN_REF
MIN_DISCRIMINATING = 100  # outputs where the two roundings differ: the pick must rest on evidence
MIN_UNCLAMPED = 0.25  # per model: the comparison must not be dominated by saturated outputs
N_SAVED = 64  # test inputs and interpreter outputs kept per npz for G4
MIN_SAVED_DISC = {"fc": 1, "conv": 50}  # discriminating outputs each saved npz must hold, so G4 sees a wrong rounding


def build_model(layer, act, seed, out_range=None):
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
        y = act_fn(tf.nn.bias_add(y, bc))
        return y if out_range is None else tf.quantization.fake_quant_with_min_max_args(y, *out_range, num_bits=8)

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


def fused_activation(model):
    """The fused activation of the model's only operator, read from the flatbuffer."""
    op = tflite.Model.GetRootAsModel(model, 0).Subgraphs(0).Operators(0)
    opt = (tflite.Conv2DOptions if op.BuiltinOptionsType() == tflite.BuiltinOptions.Conv2DOptions else tflite.FullyConnectedOptions)()
    t = op.BuiltinOptions()
    opt.Init(t.Bytes, t.Pos)
    return opt.FusedActivationFunction()


def extract(it, model, layer, act):
    """The op's tensors and quantization; stops unless the graph is one int8 FULLY_CONNECTED or CONV_2D with fused activation."""
    ops = it._get_ops_details()  # private in TF 2.x: op_name, inputs, outputs per op
    names = [o["op_name"] for o in ops]
    if names != [OP_NAME[layer]]:
        sys.exit(f"G0: FAIL, expected one {OP_NAME[layer]}, the converter produced {names}")
    if fused_activation(model) != FUSED[act]:
        sys.exit(f"G0: FAIL, fused activation {fused_activation(model)}, expected {act} ({FUSED[act]})")
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


def pick_saved(disc_in, k, n):
    """Indices of the saved inputs: the most discriminating first (up to n), then alternately calibration and uniform ones."""
    top = [int(i) for i in np.argsort(-disc_in, kind="stable") if disc_in[i] > 0][:n]
    rest = [i for i in range(disc_in.size) if i not in set(top)]
    fill = [i for pair in itertools.zip_longest([i for i in rest if i >= k], [i for i in rest if i < k]) for i in pair if i is not None]
    return np.array(sorted(top + fill[:n - len(top)]))


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
    problems, keep, clamp_models = [], {}, []
    for name, (layer, act) in MODELS.items():
        for seed in range(a.seeds):
            model, lo, hi = build_model(layer, act, seed, OUT_RANGE.get(name))
            it = interpreter(model, REF)
            p = extract(it, model, layer, act)
            xs = test_inputs(p, lo, hi, a.inputs, np.random.default_rng(seed + 2000))
            ys = invoke_all(it, xs)
            ys_default = invoke_all(interpreter(model, tf.lite.experimental.OpResolverType.AUTO), xs)
            want = {r: reference(p, xs, r) for r in ROUNDINGS}
            mism = {r: int(np.sum(ys != want[r])) for r in ROUNDINGS}
            dflt = {r: int(np.sum(ys_default != want[r])) for r in ROUNDINGS}
            disc_in = (want["DOUBLE"] != want["SINGLE"]).reshape(len(xs), -1).sum(axis=1)
            disc = int(disc_in.sum())
            nontrivial = p["amin"] > -128 or p["amax"] < 127
            full = dict(p, amin=-128, amax=127)
            act_hits = {r: int(np.sum(reference(full, xs, r) != want[r])) for r in ROUNDINGS} if nontrivial else None
            fold = sum(int(np.sum(reference(p, xs, r, folded=True) != want[r])) for r in ROUNDINGS)
            unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
            _, shifts = ref.layer_multipliers(layer, p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], "DOUBLE")
            per_ch = p["w_scales"].size > 1
            lines.append(f"{name} seed {seed}: in scale {p['in_scale']:.6g} zp {p['in_zp']}, out scale {p['out_scale']:.6g} "
                         f"zp {p['out_zp']}, act [{p['amin']}, {p['amax']}], weights "
                         f"{'per-channel' if per_ch else 'per-tensor'} ({p['w_scales'].size}), shifts [{shifts.min()}, "
                         f"{shifts.max()}], outputs {ys.size}, mismatches DOUBLE {mism['DOUBLE']} SINGLE {mism['SINGLE']}, "
                         f"discriminating {disc}, folded {fold}, unclamped {100 * unclamped:.1f}%, "
                         f"default resolver (info) DOUBLE {dflt['DOUBLE']} SINGLE {dflt['SINGLE']}"
                         + (f", outputs the fused clamp changes DOUBLE {act_hits['DOUBLE']} SINGLE {act_hits['SINGLE']}"
                            if nontrivial else ""))
            if nontrivial:
                clamp_models.append((f"{name} seed {seed}", act_hits))
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
                idx = pick_saved(disc_in, a.inputs // 10, N_SAVED)
                sd, need = int(disc_in[idx].sum()), MIN_SAVED_DISC[layer]
                lines.append(f"  saved {name}: {idx.size} inputs ({int(np.sum(idx < a.inputs // 10))} uniform int8, "
                             f"{int(np.sum(idx >= a.inputs // 10))} calibration), discriminating outputs {sd} (need {need}, "
                             f"the {a.inputs} test inputs hold {disc})")
                if sd < need:
                    problems.append(f"saved {name}: {sd} discriminating outputs, need {need} (at most {disc} available)")
                keep[name] = (model, p, xs[idx], ys[idx])
    lines.append(f"totals: outputs {tot['outputs']}, mismatches DOUBLE {tot['DOUBLE']} SINGLE {tot['SINGLE']}, "
                 f"discriminating {tot['disc']}, folded {tot['folded']}")
    winners = [r for r in ROUNDINGS if tot[r] == 0]
    if len(winners) != 1:
        problems.append(f"{len(winners)} roundings match every output; exactly one must")
    elif not any(h[winners[0]] > 0 for _, h in clamp_models):
        problems.append(f"no model has a non-trivial fused clamp that changes an output ({len(clamp_models)} non-trivial)")
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


def activation_int8(op, in_scale, in_zp):
    """TFLite's int8 TANH or LOGISTIC (reference kernels) on every int8 input; returns (outputs, input scale, input zero point)."""
    lo, hi = in_scale * (-128 - in_zp), in_scale * (127 - in_zp)
    grid = np.linspace(lo, hi, 256, dtype=np.float32).reshape(1, 256)
    model = tf.keras.Sequential([tf.keras.Input(shape=(256,)),
                                 tf.keras.layers.Activation({"tanh": "tanh", "logistic": "sigmoid"}[op])])
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: iter([[grid]])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    interp = tf.lite.Interpreter(model_content=conv.convert(),
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    ops = {d["op_name"] for d in interp._get_ops_details()}
    assert ops == {op.upper()}, f"expected one int8 {op.upper()} op, got {ops}"
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    want = {"tanh": (1 / 128, 0), "logistic": (1 / 256, -128)}[op]
    assert abs(o["quantization"][0] - want[0]) < 1e-12 and o["quantization"][1] == want[1], o["quantization"]
    interp.set_tensor(i["index"], np.arange(-128, 128, dtype=np.int8).reshape(1, 256))
    interp.invoke()
    s, z = i["quantization"]
    return interp.get_tensor(o["index"]).reshape(256).astype(np.int64), float(s), int(z)


if __name__ == "__main__":
    main()
