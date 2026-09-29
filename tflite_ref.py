"""numpy-only int8 FC and conv2d as TFLite's reference kernels compute them, on ipu.requant; TFLite layouts (FC [out, in], OHWI, NHWC)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh", "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402
from ipu import quantize_multiplier, round_half_away  # noqa: E402,F401  re-exported: TFLite's QuantizeMultiplier and TfLiteRound, one copy in AriL

ROUNDING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbenches", "tflite_int8", "rounding.txt")
ROUNDING = open(ROUNDING_FILE).read().strip() if os.path.exists(ROUNDING_FILE) else None  # G0's pinned variant; None before G0


def effective_scale(in_scale, w_scale, out_scale, product: str) -> float:
    """The real multiplier of TFLite's Prepare: 'double' is the per-channel path, 'float32' the per-tensor FC path."""
    i, w, o = np.float32(in_scale), np.float32(w_scale), np.float32(out_scale)
    if product == "double":
        return float(i) * float(w) / float(o)
    if product == "float32":
        return float(i * w) / float(o)
    raise ValueError(product)


def layer_multipliers(layer, w_scales, in_scale, out_scale, cout, rounding, scale_product=None):
    """Per output channel (mult, shift); a single weight scale is per tensor and broadcast (conv still takes the double path)."""
    ws = np.atleast_1d(np.asarray(w_scales, dtype=np.float32))
    product = scale_product or ("double" if layer == "conv" or ws.size > 1 else "float32")
    pairs = [quantize_multiplier(effective_scale(in_scale, ws[c if ws.size > 1 else 0], out_scale, product), rounding)
             for c in range(cout)]
    return np.array([m for m, _ in pairs], dtype=np.int64), np.array([s for _, s in pairs], dtype=np.int64)


def activation_range(activation: str, out_scale, out_zp: int):
    """TFLite CalculateActivationRangeQuantized for int8: bounds quantized as zp + round(f / scale), f / scale in float32."""
    def quant(f):
        return int(out_zp) + round_half_away(float(np.float32(f) / np.float32(out_scale)))
    if activation == "none":
        return -128, 127
    if activation == "relu":
        return max(-128, quant(0.0)), 127
    if activation == "relu6":
        return max(-128, quant(0.0)), min(127, quant(6.0))
    raise ValueError(activation)


def fold_input_zp(b_q, w_q, in_zp):
    """SIENNA's bias: b - in_zp * sum(w) per output channel, wrapped to int32, so the mesh sees no zero point."""
    w = np.asarray(w_q, dtype=np.int64).reshape(np.shape(w_q)[0], -1)
    return ipu.sx(np.asarray(b_q, dtype=np.int64) - int(in_zp) * w.sum(axis=1), 32)


def im2col_same(x_q, kh, kw, pad_value):
    """NHWC patches of a stride-1 SAME convolution padded with pad_value; rows (b, y, x), columns (kh, kw, cin)."""
    x = np.asarray(x_q, dtype=np.int64)
    b, h, w, c = x.shape
    ph, pw = (kh - 1) // 2, (kw - 1) // 2
    xp = np.pad(x, ((0, 0), (ph, kh - 1 - ph), (pw, kw - 1 - pw), (0, 0)), constant_values=pad_value)
    cols = [xp[:, dy:dy + h, dx:dx + w, :] for dy in range(kh) for dx in range(kw)]
    return np.stack(cols, axis=3).reshape(b * h * w, kh * kw * c)


def fc_int8(x_q, w_q, b_q, in_zp, w_scales, in_scale, out_scale, out_zp, amin, amax, rounding, folded=False,
            scale_product=None):
    """reference_integer_ops::FullyConnected(PerChannel), requantized per channel; folded=True is SIENNA's algebra (fold_input_zp bias)."""
    x = np.asarray(x_q, dtype=np.int64)
    w = np.asarray(w_q, dtype=np.int64)
    b = np.zeros(w.shape[0], dtype=np.int64) if b_q is None else np.asarray(b_q, dtype=np.int64)
    acc = x @ w.T + fold_input_zp(b, w, in_zp) if folded else (x - int(in_zp)) @ w.T + b
    mult, shift = layer_multipliers("fc", w_scales, in_scale, out_scale, w.shape[0], rounding, scale_product)
    return ipu.requant(ipu.sx(acc, 32), mult, shift, out_zp, amin, amax, rounding).astype(np.int8)


def conv2d_int8(x_q, w_q, b_q, in_zp, w_scales, in_scale, out_scale, out_zp, amin, amax, rounding, folded=False):
    """reference_integer_ops::ConvPerChannel, stride 1, SAME (outside taps skipped); folded=True pads with in_zp and folds the bias."""
    x = np.asarray(x_q, dtype=np.int64)
    w = np.asarray(w_q, dtype=np.int64)
    bsz, h, wd, _ = x.shape
    cout, kh, kw, _ = w.shape
    b = np.zeros(cout, dtype=np.int64) if b_q is None else np.asarray(b_q, dtype=np.int64)
    wm = w.reshape(cout, -1)
    if folded:
        acc = im2col_same(x, kh, kw, int(in_zp)) @ wm.T + fold_input_zp(b, w, in_zp)
    else:
        acc = im2col_same(x - int(in_zp), kh, kw, 0) @ wm.T + b
    mult, shift = layer_multipliers("conv", w_scales, in_scale, out_scale, cout, rounding)
    acc = ipu.sx(acc, 32).reshape(bsz, h, wd, cout)
    return ipu.requant(acc, mult, shift, out_zp, amin, amax, rounding).astype(np.int8)
