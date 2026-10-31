#!/usr/bin/env python3
"""SIENNA's host software stack: lowers TFLite models to the RTL's layer and set protocol and runs them on a backend (--action model, the default: float models end to end, every multiply-accumulate on the pipeline, host only reshapes and softmax; --action tflite: the int8 TFLite layers)."""

# Contents, in the order a driver / compiler is layered (sections call each other through these names; 1-4 are what a driver needs):
#   1. Numerics         op_round, fmt_bits, quant_act, quant_weights, fold_bias, requant_params, requantize, encode_scales, int8_layer_exact, add_words, qadd_codes, exact_sum, exact_layer; layer_scales, fc_acc, conv2d_acc
#   2. Frontend         load_tflite, lower_op, fuse_add, im2col, job_reference; int8 TFLite layers: load_int8_graph, load_layer, job_of, add_record_of_tflite
#   3. Middle end       tile_job (sets), format_layer and layer_epilogue (a layer's streams), pack_jobs, unpack, pack_precheck
#   4. Device protocol  write_layer and read_outputs (TB_sienna (layer)'s layer and result files), write_sets (TB_sienna (sets)'s set file)
#   5. Device build     write_build_pkg (test_config_pkg.sv), write_sv_package, _config_items, SETS_IN_FLIGHT, COLLAPSE_K
#   6. Backends         RtlLayer (TB_sienna (layer)), RtlSets (TB_sienna (sets)), Emulator (numpy)
#   7. Runtime          execute, macs_of; the model gate: check_layer, judge, gate_verdict, the bounds (LAYER_BOUND, SCORE_BOUND, REPORTED), test-only faults
#   8. CLI              model_main, tflite_main, tflite_pack_main, main

import argparse
from dataclasses import dataclass
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the SIENNA repository
SIM = os.environ.get("SIENNA_SIM", "verilator")  # the simulator: make SIM=vcs exports SIENNA_SIM=vcs to every flow it starts
SIM_DIR = os.path.join(ROOT, "VCS" if SIM == "vcs" else "Verilator")  # make vcs builds <top>_sim there, as make verilator does in Verilator/
SIM_TARGET = "vcs" if SIM == "vcs" else "verilator"
FIRED = re.compile(r'Assertion failed|%Error|%Fatal|\[FATAL\]|\[FAIL\]|^(?:Error|Fatal): "|^Error-\[|: started at .* failed at', re.M)  # Verilator's and VCS's forms
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "GPNAE", "scripts"))  # appended: the submodules have their own regression.py, which must not shadow ours
sys.path.append(os.path.join(ROOT, "SystolicMesh", "scripts"))
import gpnae_model  # noqa: E402
import mesh_model  # noqa: E402
from mesh_model import fpu  # noqa: E402
sys.path.append(os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "models"))
import ipu  # noqa: E402
import truth  # noqa: E402
import fourstate  # noqa: E402  4-state safe parsing: an x or z word or count fails its test by name


# ── 1. Numerics: the reference arithmetic the host must reproduce ────────────

# Activation control words: 001/010/011 are the GPNAE polynomial modes, 100/101 bypass the polynomial.
# No entry for 0 on purpose: a code the RTL does not implement must not be reachable from a test.
ACTIVATION_CODES = {"selu": 1, "sigmoid": 2, "tanh": 3, "relu": 4, "linear": 5}
# Number formats a build may use, as (exponent bits, mantissa bits); every word of the pipeline is in the build's format.
FORMATS = {"fp32": (8, 23), "bf16": (8, 7), "int8": (0, 7)}  # int8: EXP_W = 0, 8-bit two's-complement codes


def op_round(x: np.ndarray, fmt: str) -> np.ndarray:
    """x rounded to the operand format (nearest, ties to even), as float32; subnormals flush to zero as the RTL reads them."""
    x = np.asarray(x, dtype=np.float32)
    if fmt == "fp32":
        return x
    if fmt == "int8":  # int8 codes, as float32 values holding integers
        return np.clip(np.rint(x), -128, 127).astype(np.float32)
    if fmt == "bf16":
        u = x.view(np.uint32).astype(np.uint64)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
        y = u.astype(np.uint32).view(np.float32)
        return np.where((u.astype(np.uint32) & 0x7F800000) == 0, np.copysign(np.float32(0), y), y).astype(np.float32)
    h = x.astype(np.float16)
    h = np.where((h.view(np.uint16) & 0x7C00) == 0, np.copysign(np.float16(0), h), h)
    return h.astype(np.float32)


def op_hex(data, fmt: str) -> list:
    """Values as hex words of the operand format, after rounding: 8 digits for fp32, 4 for bf16 and fp16."""
    v = op_round(np.asarray(data, dtype=np.float32).flatten(), fmt)
    if fmt == "int8":
        return [f"{int(b) & 0xFF:02x}" for b in v.astype(np.int64)]
    if fmt == "fp32":
        return [f"{int(b):08x}" for b in v.view(np.uint32)]
    bits = (v.view(np.uint32) >> 16) if fmt == "bf16" else v.astype(np.float16).view(np.uint16)
    return [f"{int(b):04x}" for b in bits]


# The published Taylor lane's terms per activation; no RTL reads them (the lanes' degrees are coefficient-memory data), kept for the untracked run_real_model.py.
ACTIVATION_TERMS = {"selu": 14, "sigmoid": 15, "tanh": 30, "relu": 0, "linear": 0}


def activation_to_code(act: str) -> int:
    key = act.lower()
    if key not in ACTIVATION_CODES:
        raise ValueError(
            f"unsupported activation {act!r}: GPNAE implements only "
            f"{sorted(ACTIVATION_CODES)}. Control word 0 is not a pass-through "
            f"mode -- selecting it stalls the pipeline until timeout."
        )
    return ACTIVATION_CODES[key]


def get_polynomial_terms(act: str) -> int:
    key = act.lower()
    if key not in ACTIVATION_TERMS:
        raise ValueError(f"unsupported activation {act!r}")
    return ACTIVATION_TERMS[key]


def apply_activation(x: np.ndarray, act: str) -> np.ndarray:
    act = act.lower()
    if act == "selu":
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(
            x > 0, x, alpha * (np.exp(np.clip(x, -50, 50)) - 1)
        ).astype(np.float32)
    if act == "sigmoid":
        return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)
    if act == "tanh":
        return np.tanh(x).astype(np.float32)
    if act == "relu":
        return np.where(x > 0, x, np.float32(0.0)).astype(np.float32)
    return x.copy()  # linear


def fmt_bits(x, fmt: str) -> np.ndarray:
    """Values already rounded to the format (op_round), as its bit patterns."""
    x = np.asarray(x, dtype=np.float32)
    return np.array([int(h, 16) for h in op_hex(x, fmt)], dtype=np.int64).reshape(x.shape)


def bits_float(b, fmt: str) -> np.ndarray:
    f = fpu.FORMATS[fmt]
    return (np.asarray(b, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def wrap32(x) -> np.ndarray:
    """int64 values wrapped to int32, as the mesh's two's-complement accumulate does."""
    return ((np.asarray(x, dtype=np.int64) + (1 << 31)) % (1 << 32)) - (1 << 31)


def imatmul(a, b) -> np.ndarray:
    """Exact integer product: int8 x int8 terms and their sums stay below 2^53, so float64 holds every partial sum."""
    return np.rint(np.asarray(a, np.float64) @ np.asarray(b, np.float64)).astype(np.int64)


def quant_act(x) -> tuple:
    """(q, scale, zero point) of an activation tensor as TFLite PTQ picks them: asymmetric int8 over its range widened to hold 0."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = min(0.0, float(x.min())), max(0.0, float(x.max()))
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    zp = int(np.clip(np.rint(-128.0 - lo / scale), -128, 127))
    return np.clip(np.rint(x / scale) + zp, -128, 127).astype(np.int64), scale, zp


def quant_weights(w) -> tuple:
    """(q, per-column scales) of a weight matrix as TFLite quantizes weights: symmetric per output channel, zero point 0, codes -127..127."""
    w = np.asarray(w, dtype=np.float64)
    s = np.max(np.abs(w), axis=0) / 127.0
    s = np.where(s > 0, s, 1.0)
    return np.clip(np.rint(w / s[None, :]), -127, 127).astype(np.int64), s


def fold_bias(bias, s_a: float, s_w, z_a: int, B_q) -> np.ndarray:
    """The mesh's int32 bias: TFLite's bias minus z_a * sum_k B_q[k, c], so the mesh's sum of a * w is TFLite's sum of (a - z_a) * w."""
    B_q = np.asarray(B_q, np.int64)
    b_q = np.zeros(B_q.shape[1], np.int64) if bias is None else \
        np.rint(np.asarray(bias, np.float64) / (s_a * np.asarray(s_w, np.float64))).astype(np.int64)
    return wrap32(b_q - z_a * B_q.sum(axis=0))


def requantize(acc, rq: dict) -> np.ndarray:
    """ipu.requantize (int8Requantizer, bit-exact) of int32 sums (rows x channels) with channel c's mantissa and shift on column c."""
    acc = np.asarray(acc, np.int64)
    m = np.broadcast_to(np.asarray(rq["mult"], np.int64)[None, :acc.shape[1]], acc.shape)
    s = np.broadcast_to(np.asarray(rq["shift"], np.int64)[None, :acc.shape[1]], acc.shape)
    return np.asarray(ipu.requantize(acc, m, s, rq["zp"], rq["amin"], rq["amax"]), np.int64)


def encode_scales(scales):
    """(mult, shift) arrays of ipu.encode_scale per real scale."""
    pairs = [ipu.encode_scale(float(x)) for x in np.atleast_1d(scales)]
    return np.array([m for m, _ in pairs], np.int64), np.array([e for _, e in pairs], np.int64)


def code_scales(mult, shift) -> np.ndarray:
    """The real scales mult 2^-shift that requantize words hold exactly."""
    return np.ldexp(np.asarray(mult, np.float64), -np.asarray(shift, np.int64))


def requant_params(acc, s_a: float, s_w, act: str, rng=None, zq=None) -> dict:
    """Requantize and GPNAE parameters for int32 sums acc (rows x channels), asymmetric min/max calibration; with rng, random words per channel."""
    acc = np.asarray(acc, np.int64)
    s_w = np.asarray(s_w, np.float64)
    real = acc * (s_a * s_w)[None, :]
    if act == "relu":
        real = np.maximum(real, 0.0)
    _, s_out, z_out = quant_act(real)
    ratio = np.array([s_a * float(s) / s_out for s in s_w], np.float64)  # each channel's real scale s_a s_w / s_out
    mult, shift = encode_scales(ratio)
    if rng is not None:  # random words: every normalized mantissa, shifts 24..36 (scales in [2^-13, 1)), each its own exact scale
        mult = rng.randint(1 << 23, 1 << 24, s_w.size).astype(np.int64)
        shift = rng.randint(24, 37, s_w.size).astype(np.int64)
        ratio = code_scales(mult, shift)
    mx, shx = gpnae_model.rescale_params(s_out)
    rq = dict(mult=mult, shift=shift, zp=z_out, amin=max(-128, z_out) if act == "relu" else -128, amax=127,
              mx=mx, shx=shx, s_out=s_out, ratio=[float(x) for x in ratio])  # the real scales the requantize gate judges against
    if zq is not None:  # zp_random: the given (zero point, min, max) in place of the calibrated ones
        rq.update(zp=zq[0], amin=zq[1], amax=zq[2])
    x = (requantize(acc, rq) - rq["zp"]) * s_out  # the lane's real inputs
    _, s_selu, z_selu = quant_act(apply_activation(np.maximum(x, -16.0).astype(np.float32), "selu"))  # Q4.11's floor; SELU is flat there
    mout, shout = gpnae_model.selu_out_params(s_out, mx, s_selu)  # the lane's SELU accumulator, 2^-(shx + 18), to its int8 code (D-4)
    rq.update(mout=int(mout), shout=int(shout), zout=z_selu, s_selu=s_selu)
    return rq


_INT8_LANES = {}


def int8_lane(mem=None):
    """gpnae_model's int8 lane on the coefficient memory mem (default: the lanes' reset image)."""
    if mem is not None:
        return gpnae_model.Lane(gpnae_model.FORMATS["int8"], mem)
    if "int8" not in _INT8_LANES:
        _INT8_LANES["int8"] = gpnae_model.Lane(gpnae_model.FORMATS["int8"])
    return _INT8_LANES["int8"]


def activate_int8(R, act: str, rq: dict, tier: int = 0, mem=None) -> np.ndarray:
    """The int8 lane stage: ReLU and linear pass the requantized value, the others run the fixed-point lane with the set's parameters,
    at a tier of a coefficient memory (default the reset image)."""
    code = activation_to_code(act)
    R = np.asarray(R, np.int64)
    if code in (4, 5):
        return R.copy()
    par = gpnae_model.Int8Params(mx=rq["mx"], shx=rq["shx"], zin=rq["zp"], mout=rq["mout"], shout=rq["shout"], zout=rq["zout"])
    return np.asarray(int8_lane(mem).run(R, code, par, tier), np.int64)


# ── the lanes' coefficient memory: the host's loader and its tier choice ──

def coef_writes(mem) -> list:
    """The load-path writes (address, data) that put a coefficient memory image (gpnae_model.Mem) into every lane."""
    return mem.writes()


def write_coef(path: str, mem) -> int:
    """A +coef= load file ("aaa dddddddd" per write) and beside it <path>.img, the image the bench checks the lanes against; returns the count."""
    w = coef_writes(mem)
    with open(path, "w") as fh:
        fh.write("".join(f"{a:03x} {d & 0xFFFFFFFF:08x}\n" for a, d in w))
    mem.write_contents(path + ".img")
    return len(w)


COEF_IMAGE = re.compile(r"COEF IMAGE : checks (\d+)\s+differ (\d+)")  # TB_sienna's coefficient-image check line, every mode


def coef_image_ran(text: str, tag: str, loads: int) -> None:
    """The bench's coefficient-image check ran, after reset and each of loads loads at least, and found no difference; else RuntimeError."""
    m = COEF_IMAGE.search(text)
    if not m or int(m.group(1)) < 1 + loads or int(m.group(2)) != 0:
        raise RuntimeError(f"{tag}: the coefficient-image check " + (f"ran {m.group(1)} times, {m.group(2)} differing" if m else "did not report") +
                           f" (at least {1 + loads} checks, none differing, expected)")


def busy_writes(mem, act: str, tier: int = 0) -> list:
    """Four load-path writes that would change act's own words at tier if one landed: its descriptor's first two coefficient words
    inverted, its piece set pointed at another activation's words, and its first bound inverted (+coef_busy's corrupting writes)."""
    g = gpnae_model
    d = {"selu": g.D_SELU, "sigmoid": g.D_SIGP, "tanh": g.D_TANH}[act]
    ds, other = mem.desc[tier][d], mem.desc[tier][g.D_TANH if d != g.D_TANH else g.D_SELU]
    m = (1 << mem.act_w) - 1
    b = g.A_DESC + g.DESC_STRIDE * (tier * g.NUM_DESC + d)
    moved = g.Desc(**{k: v for k, v in vars(ds).items()})
    moved.base = other.base if other.base != ds.base else (ds.base + 1) % g.COEF_WORDS
    return [(ds.base, ~mem.words[ds.base] & m), (ds.base + 1, ~mem.words[ds.base + 1] & m), (b + g.F_SET, moved.set_word()),
            (b, ~ds.bounds[0] & m)]


def write_writes(path: str, w) -> int:
    """(address, data) writes as a +coef= or +coef_busy= file; returns the count."""
    with open(path, "w") as fh:
        fh.write("".join(f"{a:03x} {d & 0xFFFFFFFF:08x}\n" for a, d in w))
    return len(w)


def pick_tier(tol: float) -> int:
    """The cheapest tier whose tolerance is within tol: a layer that tolerates 10% runs tier 2, 5% tier 1, anything tighter tier 0."""
    ok = [t for t, v in enumerate(gpnae_model.TIER_TOL) if v <= tol + 1e-12]
    return max(ok) if ok else 0


def drop_zp(act: str, rq: dict) -> int:
    """D-5: dropout's drop value, the output zero point of the set's activation (SELU gp_zout, sigmoid -128, ReLU / linear zp, else 0)."""
    return {1: rq["zout"], 2: -128, 4: rq["zp"], 5: rq["zp"]}.get(activation_to_code(act), 0)


ADD_FIELDS = ("zp1", "zp2", "m1", "m2", "shift", "zpo", "amin", "amax")  # a layer's ADD words, add_words_t's order (add_<k>.mem)


def add_words(s1, z1, s2, z2, s_out, z_out, act: str, name: str = "") -> dict:
    """A layer's ADD words from its inputs' and output's real scales and zero points and the fused activation's clamp: the AddRecord of
    the generator's numbers through add_record_words."""
    lo, hi = ipu.activation_clamp(act if act in ("relu", "relu6") else "none", s_out, int(z_out))
    return add_record_words(AddRecord(float(s1), int(z1), float(s2), int(z2), float(s_out), int(z_out), int(lo), int(hi), name))


def add_record_words(rec: "AddRecord") -> dict:
    """int8QuantizedAdder's words for an ADD record (ipu.encode_add over s1 / s_out and s2 / s_out); ratios keeps those two for the
    true-function gate. A ratio outside the adder's domain raises ValueError naming the layer."""
    r1, r2 = rec.s1 / rec.s_out, rec.s2 / rec.s_out
    try:
        m1, m2, e = ipu.encode_add(r1, r2)
    except ValueError as x:
        raise ValueError(f"ADD layer {rec.name or '(unnamed)'}: {x}") from x
    return dict(zp1=rec.z1, zp2=rec.z2, m1=int(m1), m2=int(m2), shift=int(e), zpo=rec.z_out, amin=rec.lo, amax=rec.hi, ratios=(r1, r2))


def qadd_codes(q1, q2, add: dict) -> np.ndarray:
    """ipu.qadd (int8QuantizedAdder, bit-exact) of two int8 code arrays with a layer's ADD words."""
    return np.asarray(ipu.qadd(q1, q2, *[add[k] for k in ADD_FIELDS]), np.int64)


def add_lane_params(add: dict, s_out: float, y) -> dict:
    """The lane's words after an ADD: input zero point the ADD's zp_o, the rescale of s_out, SELU's output calibrated on the ADD's outputs y."""
    mx, shx = gpnae_model.rescale_params(s_out)
    x = (np.asarray(y, np.int64) - add["zpo"]) * s_out  # the lane's real inputs
    _, s_selu, z_selu = quant_act(apply_activation(np.maximum(x, -16.0).astype(np.float32), "selu"))  # as requant_params calibrates SELU
    mout, shout = gpnae_model.selu_out_params(s_out, mx, s_selu)
    return dict(zp=add["zpo"], amin=add["amin"], amax=add["amax"], mx=mx, shx=shx, mout=int(mout), shout=int(shout), zout=z_selu,
                s_out=s_out, s_selu=s_selu)


def int8_residual_exact(A_q, B_q, hw_bias, rq: dict, R_q, add: dict, ra: dict, act: str, tier: int = 0, mem=None) -> np.ndarray:
    """the layer contract's int8 residual output for one product, bit-exact: the conv's requantize (q1), the ADD with the residual codes R_q, the lane."""
    acc = wrap32(imatmul(A_q, B_q) + np.asarray(hw_bias, np.int64)[None, :])
    return activate_int8(qadd_codes(requantize(acc, rq), R_q, add), act, ra, tier, mem)


def int8_layer_exact(A_q, B_q, hw_bias, rq: dict, act: str, tier: int = 0, mem=None) -> np.ndarray:
    """the layer contract's int8 output for one product: int32 sums, per-column requantize, the lane; the layer engine neither pools nor drops out."""
    assert all(np.asarray(m).dtype == np.int64 and np.asarray(m).min() >= -128 and np.asarray(m).max() <= 127
               for m in (A_q, B_q)), "int8_layer_exact: operands must be int64 arrays of int8 values"  # as _golden_int8
    acc = wrap32(imatmul(A_q, B_q) + np.asarray(hw_bias, np.int64)[None, :])
    return activate_int8(requantize(acc, rq), act, rq, tier, mem)


def word_bits(x, fmt: str) -> np.ndarray:
    """Float32 values holding a float format's words (as read_outputs widens them) back to the words, unrounded."""
    return np.asarray(x, np.float32).view(np.uint32).astype(np.int64) >> (23 - FORMATS[fmt][1])


_LANES = {}


def exact_sum(passes, bias, act: str, N: int, fmt: str, T: int, collapse_k=None, tier: int = 0, mem=None) -> np.ndarray:
    """Bit-exact N x N result of one sum in a float format: mesh_model over its passes (A, B words) in order, the bias words in the reducer, then the
    GPNAE lane at tier of the coefficient memory mem (default the reset image)."""
    f = fpu.FORMATS[fmt]
    if mem is not None:
        lane = gpnae_model.Lane(f, mem)
    else:
        if fmt not in _LANES:
            _LANES[fmt] = gpnae_model.Lane(f)
        lane = _LANES[fmt]
    C = mesh_model.matmul(f, passes, N, T, COLLAPSE_K if collapse_k is None else collapse_k, bias)
    return lane.run(C, activation_to_code(act), tier)


def exact_layer(A, B, bias, act, N, fmt, T=4):
    """Bit-exact output of the layer contract for one product in a float format: per output tile, the depth blocks as passes in
    order (format_layer's order), the bias with the first, then the lane; T is the build's tile size."""
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
    Y = np.zeros((rt * N, ct * N), np.int64)
    for c in range(ct):
        for r in range(rt):
            passes = [(fmt_bits(Ap[r * N:(r + 1) * N, t * N:(t + 1) * N], fmt), fmt_bits(Bp[t * N:(t + 1) * N, c * N:(c + 1) * N], fmt))
                      for t in range(dt)]
            b = fmt_bits(bp[c * N:(c + 1) * N], fmt) if bias is not None else None
            Y[r * N:(r + 1) * N, c * N:(c + 1) * N] = exact_sum(passes, b, act, N, fmt, T)
    return Y[:M, :C]


def layer_scales(w_scales, in_scale, out_scale, cout) -> np.ndarray:
    """Per output channel the real requantize scale s_in s_w[c] / s_out in float64 from the tensors' stored scales; one weight scale is per tensor."""
    ws = np.atleast_1d(np.asarray(w_scales, dtype=np.float64))
    return np.array([float(in_scale) * float(ws[c if ws.size > 1 else 0]) / float(out_scale) for c in range(cout)], np.float64)


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


def fc_acc(x_q, w_q, b_q, in_zp, folded=False) -> np.ndarray:
    """A fully connected layer's int32 sums sum_k (x - in_zp) w + b; folded=True is SIENNA's algebra (fold_input_zp bias), equal exactly."""
    x = np.asarray(x_q, dtype=np.int64)
    w = np.asarray(w_q, dtype=np.int64)
    b = np.zeros(w.shape[0], dtype=np.int64) if b_q is None else np.asarray(b_q, dtype=np.int64)
    return ipu.sx(x @ w.T + fold_input_zp(b, w, in_zp) if folded else (x - int(in_zp)) @ w.T + b, 32)


def conv2d_acc(x_q, w_q, b_q, in_zp, folded=False) -> np.ndarray:
    """A stride-1 SAME convolution's int32 sums (outside taps add nothing); folded=True pads with in_zp and folds the bias, equal exactly."""
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
    return ipu.sx(acc, 32).reshape(bsz, h, wd, cout)


# ── 2. Frontend: each op becomes Y = sum_i X_i @ W_i + b, then an activation ─

@dataclass
class LayerRecord:
    """The neutral int8 conv / fully connected layer every importer produces: symmetric int8 weights, an int32 bias in units of
    s_in s_w[c], each output channel's real scale S[c] = s_in s_w[c] / s_out, the zero points and the fused activation's clamp."""
    kind: str  # "conv" or "fc"
    w_q: np.ndarray  # int8 codes: conv (cout, kh, kw, cin), fc (cout, cin)
    b_q: np.ndarray  # int32 bias per output channel, zeros where the layer has none
    scale: np.ndarray  # S[c], float64
    zp_in: int
    zp_out: int
    lo: int
    hi: int
    stride: tuple = (1, 1)
    pads: tuple = (0, 0, 0, 0)  # zero-point padding (top, bottom, left, right), explicit: each importer resolves its framework's rule
    name: str = ""
    s_out: float = None  # the output's real scale, for a layer whose output feeds an ADD
    clamp: tuple = (None, None)  # the fused activation's real range (None: unbounded), for the true-function gate


@dataclass
class AddRecord:
    """The neutral int8 ADD: y = (s1 (q1 - z1) + s2 (q2 - z2)) / s_out + z_out, clamped to [lo, hi]."""
    s1: float
    z1: int
    s2: float
    z2: int
    s_out: float
    z_out: int
    lo: int
    hi: int
    name: str = ""
    clamp: tuple = (None, None)  # the fused activation's real range (None: unbounded), for the true-function gate


FUSED_RANGE = {"none": (None, None), "relu": (0.0, None), "relu6": (0.0, 6.0), "relu_n1_to_1": (-1.0, 1.0)}  # a fused kind's real range


def real_range(relu=False, clip=None) -> tuple:
    """A following Relu and / or Clip(min, max) as a real range (None: unbounded)."""
    lo, hi = (None, None) if clip is None else (float(clip[0]), float(clip[1]))
    return (0.0 if lo is None else max(0.0, lo), hi) if relu else (lo, hi)


def clamp_codes(lo_real, hi_real, s_out: float, zp: int) -> tuple:
    """A real-valued clamp (None: unbounded) as int8 codes on an output of scale s_out and zero point zp, ties rounded up."""
    code = lambda v: zp + math.floor(v / s_out + 0.5)
    lo = -128 if lo_real is None else max(-128, code(lo_real))
    hi = 127 if hi_real is None else min(127, code(hi_real))
    return int(lo), int(hi)


def as_int8(codes, zps=None, unsigned=False):
    """Codes and zero points of an int8 or uint8 tensor as int8 ones: uint8 subtracts 128 from both (exact)."""
    shift = 128 if unsigned else 0
    c = np.asarray(codes, np.int64) - shift
    return c if zps is None else (c, np.asarray(zps, np.int64) - shift)


def _symmetric(w_zp, name):
    if np.any(np.asarray(w_zp) != 0):
        raise ValueError(f"layer {name or '(unnamed)'}: a non-zero weight zero point is not supported (SIENNA folds only the input zero point)")


def record_of_tflite(layer: dict, name: str = "") -> LayerRecord:
    """The TFLite importer's record: load_layer's tensors, scales from the stored float32 scales in float64 (layer_scales)."""
    inp, flt, b, out = layer["input"], layer["filter"], layer["bias"], layer["output"]
    _symmetric(flt["zp"], name)
    w = flt["data"].astype(np.int64)
    cout = w.shape[0]
    kind = "conv" if layer["kind"] == "CONV_2D" else "fc"
    bq = np.zeros(cout, np.int64) if b is None or b["data"] is None else b["data"].astype(np.int64)
    lo, hi = layer["act_range"]
    stride, pads = (1, 1), (0, 0, 0, 0)
    if kind == "conv":
        if layer.get("stride") is None or layer.get("same") is None:
            raise ValueError(f"layer {name or '(unnamed)'}: stride and padding are missing from the conv layer")
        stride = tuple(int(v) for v in layer["stride"])
        if layer["same"]:  # TFLite SAME: out = ceil(n / s), the odd pad at the end
            _, pt, pb = _same_pad(int(inp["shape"][1]), w.shape[1], stride[0])
            _, pl, pr = _same_pad(int(inp["shape"][2]), w.shape[2], stride[1])
            pads = (pt, pb, pl, pr)
    return LayerRecord(kind, w, bq, layer_scales(flt["scale"], inp["scale"][0], out["scale"][0], cout), int(inp["zp"][0]),
                       int(out["zp"][0]), int(lo), int(hi), stride, pads, name, float(out["scale"][0]), FUSED_RANGE[layer.get("fused", "none")])


def record_of_onnx(op: str, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias=None, strides=(1, 1), pads=None, relu=False,
                   clip=None, x_unsigned=False, y_unsigned=False, w_unsigned=False, group=1, dilations=(1, 1), auto_pad="NOTSET",
                   name: str = "") -> LayerRecord:
    """The ONNX importer's record for QLinearConv (w OIHW) or QLinearMatMul (w (K, N)), or their QDQ folding, from the initializers as
    numpy arrays: uint8 tensors become int8 by subtracting 128; a following Relu or Clip(min, max) is the clamp; pads [top, left, bottom,
    right] pass through as given; group, dilation and auto_pad SAME_* (resolve them to pads first) are refused."""
    w, w_zp = as_int8(w, w_zp, w_unsigned)
    _symmetric(w_zp, name)
    _, zin = as_int8(0, x_zp, x_unsigned)
    _, zout = as_int8(0, y_zp, y_unsigned)
    zin, zout = int(np.ravel(zin)[0]), int(np.ravel(zout)[0])
    if op == "QLinearConv":
        w = np.transpose(w, (0, 2, 3, 1))  # OIHW to OHWI
        if group != 1 or tuple(dilations) != (1, 1) or auto_pad not in ("NOTSET", "VALID"):
            raise ValueError(f"layer {name or '(unnamed)'}: group {group}, dilations {tuple(dilations)}, auto_pad {auto_pad}: only group 1, "
                             "dilation 1 and explicit pads are supported")
        p = [0, 0, 0, 0] if pads is None or auto_pad == "VALID" else [int(v) for v in pads]
        if len(p) != 4 or min(p) < 0:
            raise ValueError(f"layer {name or '(unnamed)'}: pads {list(pads)} are not four non-negative values")
        pads, kind = (p[0], p[2], p[1], p[3]), "conv"  # ONNX [top, left, bottom, right] to (top, bottom, left, right)
    elif op == "QLinearMatMul":
        w, kind, pads = w.T, "fc", (0, 0, 0, 0)
    else:
        raise ValueError(f"layer {name or '(unnamed)'}: {op} is not QLinearConv or QLinearMatMul")
    cout = w.shape[0]
    s_out = float(np.ravel(y_scale)[0])
    scale = np.asarray(layer_scales(np.atleast_1d(w_scale), float(np.ravel(x_scale)[0]), s_out, cout), np.float64)
    lo, hi = clamp_codes(0.0 if relu else (None if clip is None else clip[0]), None if clip is None else clip[1], s_out, zout)
    if relu and clip is not None:
        lo = max(lo, zout)
    bq = np.zeros(cout, np.int64) if bias is None else np.asarray(bias, np.int64)
    return LayerRecord(kind, w, bq, scale, zin, zout, lo, hi, tuple(int(v) for v in strides), pads, name, s_out, real_range(relu, clip))


def record_of_torch(kind: str, in_scale, in_zp, w_int, w_scales, w_zps, bias, out_scale, out_zp, relu=False, stride=(1, 1),
                    padding=(0, 0), unsigned=True, groups=1, dilation=(1, 1), padding_mode="zeros", name: str = "") -> LayerRecord:
    """The PyTorch importer's record for a quantized Conv2d / Linear (torch.ao, FX or PT2E): input q_scale and q_zero_point, the
    weight's int_repr (conv OIHW, linear (out, in)) with q_per_channel_scales of a symmetric qint8 weight, the float bias quantized to
    int32 at s_in s_w[c], the module's output scale and zero point; ConvReLU2d / LinearReLU clamp at the zero point; quint8 is shifted;
    padding (ph, pw) pads both sides of each axis; groups, dilation, padding_mode other than zeros and string padding are refused."""
    _symmetric(w_zps, name)
    w = np.asarray(w_int, np.int64)
    if kind == "conv":
        w = np.transpose(w, (0, 2, 3, 1))
        if groups != 1 or tuple(np.broadcast_to(dilation, 2)) != (1, 1) or padding_mode != "zeros" or isinstance(padding, str):
            raise ValueError(f"layer {name or '(unnamed)'}: groups {groups}, dilation {dilation}, padding {padding!r} ({padding_mode}): "
                             "only groups 1, dilation 1 and numeric zero padding are supported")
        ph, pw = (int(v) for v in np.broadcast_to(padding, 2))
        if min(ph, pw) < 0:
            raise ValueError(f"layer {name or '(unnamed)'}: padding {padding} is negative")
        pads = (ph, ph, pw, pw)
    elif kind == "linear":
        pads, kind = (0, 0, 0, 0), "fc"
    else:
        raise ValueError(f"layer {name or '(unnamed)'}: kind {kind} is not conv or linear")
    cout = w.shape[0]
    ws = np.broadcast_to(np.asarray(w_scales, np.float64), (cout,))
    zin = int(np.asarray(in_zp)) - (128 if unsigned else 0)
    zout = int(np.asarray(out_zp)) - (128 if unsigned else 0)
    bq = np.zeros(cout, np.int64) if bias is None else np.floor(np.asarray(bias, np.float64) / (float(in_scale) * ws) + 0.5).astype(np.int64)
    lo, hi = (zout, 127) if relu else (-128, 127)
    return LayerRecord(kind, w, bq, layer_scales(ws, float(in_scale), float(out_scale), cout), zin, zout, max(-128, lo), hi,
                       tuple(int(v) for v in np.broadcast_to(stride, 2)), pads, name, float(out_scale), real_range(relu))


def add_record_of_onnx(a_scale, a_zp, b_scale, b_zp, c_scale, c_zp, relu=False, clip=None, unsigned=False, name: str = "") -> AddRecord:
    """The ONNX importer's ADD record: QLinearAdd (com.microsoft) or the QDQ Add pattern, a following Relu or Clip as the clamp."""
    sh = 128 if unsigned else 0
    z1, z2, zo = (int(np.ravel(z)[0]) - sh for z in (a_zp, b_zp, c_zp))
    so = float(np.ravel(c_scale)[0])
    lo, hi = clamp_codes(0.0 if relu else (None if clip is None else clip[0]), None if clip is None else clip[1], so, zo)
    return AddRecord(float(np.ravel(a_scale)[0]), z1, float(np.ravel(b_scale)[0]), z2, so, zo, lo, hi, name, real_range(relu, clip))


def add_record_of_torch(a_scale, a_zp, b_scale, b_zp, out_scale, out_zp, relu=False, unsigned=True, name: str = "") -> AddRecord:
    """The PyTorch importer's ADD record: QFunctional.add / add_relu with the inputs' and the module's scale and zero point."""
    sh = 128 if unsigned else 0
    zo = int(out_zp) - sh
    return AddRecord(float(a_scale), int(a_zp) - sh, float(b_scale), int(b_zp) - sh, float(out_scale), zo, zo if relu else -128, 127, name,
                     real_range(relu))


def job_of_record(rec: LayerRecord, x) -> tuple:
    """(int8 job for RtlLayer, output shape) of a layer record on input codes x (images NHWC for conv, rows for fc); req holds each
    channel's real scale ('ratio') for the true-function gate."""
    w = np.asarray(rec.w_q, np.int64)
    cout = w.shape[0]
    mult, shift = encode_scales(rec.scale)
    if rec.kind == "conv":
        cols = [im2col(np.asarray(xi, np.float32), w.shape[1], w.shape[2], rec.stride, rec.pads,
                       pad_value=float(rec.zp_in)) for xi in x]  # every input image; rows in (image, y, x) order
        X, (oh, ow) = np.vstack([c for c, _ in cols]), cols[0][1]
        W = w.reshape(cout, -1).T  # OHWI filters: depth in (ky, kx, c) order, as im2col lays it out
        shape = (len(x), oh, ow, cout)
    else:
        W = w.T
        X = np.asarray(x, np.float32).reshape(-1, W.shape[0])
        shape = (X.shape[0], cout)
    hw_bias = wrap32(np.asarray(rec.b_q, np.int64) - rec.zp_in * W.sum(axis=0))
    req = dict(mult=mult, shift=shift, zp=rec.zp_out, amin=rec.lo, amax=rec.hi, mx=0, shx=0, mout=0, shout=0, zout=0,
               ratio=[float(v) for v in rec.scale])
    return {"terms": [(X.astype(np.float32), W.astype(np.float32))], "bias": hw_bias, "act": "linear", "shape": shape, "req": req}, shape



def job_of_add_record(conv: LayerRecord, add: AddRecord, x, R) -> tuple:
    """(int8 residual job for RtlLayer, output shape): conv on input codes x, then add's ADD of the residual codes R (input 1 is conv's output)."""
    if conv.s_out is None:
        raise ValueError(f"layer {conv.name or '(unnamed)'}: a conv feeding an ADD needs its output scale (s_out) on the record")
    if add.z1 != conv.zp_out or not math.isclose(add.s1, conv.s_out, rel_tol=1e-9):
        raise ValueError(f"ADD layer {add.name or '(unnamed)'}: input 1 ({add.s1:.9g}, zp {add.z1}) is not layer {conv.name or '(unnamed)'}'s "
                         f"output ({conv.s_out:.9g}, zp {conv.zp_out})")
    job, shape = job_of_record(conv, x)
    (X, W), = job["terms"]
    C = W.shape[1]
    R = np.asarray(R, np.int64).reshape(X.shape[0], C)
    words = add_record_words(add)
    req = dict(job["req"], s_out=float(conv.s_out))
    acc = wrap32(imatmul(X.astype(np.int64), W.astype(np.int64)) + np.asarray(job["bias"], np.int64)[None, :])
    ra = add_lane_params(words, add.s_out, qadd_codes(requantize(acc, req), R, words))
    req.update(mx=ra["mx"], shx=ra["shx"], mout=ra["mout"], shout=ra["shout"], zout=ra["zout"])  # the lane after the ADD
    job.update(terms=[(X, W), (R.astype(np.float32), np.eye(C, dtype=np.float32))], act="relu" if tuple(add.clamp) == (0.0, None) else "linear",
               shape=(X.shape[0], C), req=req, add=words, lane=ra, clamps=(tuple(conv.clamp), tuple(add.clamp)))
    return job, shape

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


def _same_pad(n, k, s):
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2, total - total // 2


def im2col(x, kh, kw, stride, same, channel_major=False, pad_value=0.0):
    """x is H x W x C; rows are output pixels, depth is (ky, kx, c), or (c, ky, kx) when channel_major; same is True (TFLite SAME),
    False (no padding) or explicit pads (top, bottom, left, right), filled with pad_value."""
    H, W, C = x.shape
    sh, sw = stride
    flag = isinstance(same, (bool, np.bool_))
    if flag and same:
        oh, pt, pb = _same_pad(H, kh, sh)
        ow, pl, pr = _same_pad(W, kw, sw)
    else:
        pt, pb, pl, pr = (0, 0, 0, 0) if flag else (int(v) for v in same)
        oh, ow = (H + pt + pb - kh) // sh + 1, (W + pl + pr - kw) // sw + 1
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), constant_values=pad_value)
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


def check_int8_add(op: dict, where: str) -> None:
    """Refuses an ADD the residual path cannot run: not int8, broadcast, a constant input, or per-axis scales or zero points."""
    a, b, out = op["a"], op["b"], op["output"]
    if any(v["type"] != 9 for v in (a, b, out)):  # 9 is INT8
        raise ValueError(f"{where}: inputs and output must be int8")
    if tuple(a["shape"]) != tuple(out["shape"]) or tuple(b["shape"]) != tuple(out["shape"]):
        raise ValueError(f"{where}: input shapes {tuple(a['shape'])} and {tuple(b['shape'])} differ from the output's {tuple(out['shape'])}; "
                         "a broadcast ADD is not supported")
    if a["data"] is not None or b["data"] is not None:
        raise ValueError(f"{where}: a constant input is not supported; both inputs must be activations")
    if any(np.size(v["scale"]) != 1 or np.size(v["zp"]) != 1 for v in (a, b, out)):
        raise ValueError(f"{where}: per-axis scales or zero points are not supported; the ADD takes one per tensor")


def load_int8_graph(path: str) -> dict:
    """An int8 .tflite's CONV_2D, FULLY_CONNECTED and ADD ops in order with their tensors (and indices), fused kinds and clamps."""
    import tflite
    from tflite.ActivationFunctionType import ActivationFunctionType as AF
    from tflite.BuiltinOperator import BuiltinOperator as BO

    names = {v: k for k, v in BO.__dict__.items() if not k.startswith("_")}
    acts = {AF.NONE: "none", AF.RELU: "relu", AF.RELU6: "relu6", AF.RELU_N1_TO_1: "relu_n1_to_1"}  # ipu.activation_clamp's kinds
    opts = {"CONV_2D": tflite.Conv2DOptions, "FULLY_CONNECTED": tflite.FullyConnectedOptions, "ADD": tflite.AddOptions}
    m = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
    g = m.Subgraphs(0)

    def tensor(i):
        if i < 0:
            return None
        t = g.Tensors(i)
        q = t.Quantization()
        buf = m.Buffers(t.Buffer()).DataAsNumpy()
        data = None
        if not isinstance(buf, int) and buf is not None and len(buf):
            data = np.frombuffer(buf.tobytes(), dtype={9: np.int8, 2: np.int32}[t.Type()]).reshape(tuple(t.ShapeAsNumpy()))
        return dict(index=i, type=t.Type(), shape=tuple(t.ShapeAsNumpy()), data=data,
                    scale=np.atleast_1d(q.ScaleAsNumpy()).astype(np.float32),
                    zp=np.atleast_1d(q.ZeroPointAsNumpy()).astype(np.int64))

    ops = []
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        c = m.OperatorCodes(op.OpcodeIndex())
        kind = names[max(c.BuiltinCode(), c.DeprecatedBuiltinCode())]
        if kind not in opts:
            raise ValueError(f"{path}: operator {k} is {kind}; the int8 importer takes CONV_2D, FULLY_CONNECTED and ADD")
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        out = tensor(op.Outputs(0))
        if kind == "ADD":
            d = dict(kind=kind, inputs=ins[:2], a=tensor(ins[0]), b=tensor(ins[1]), output=out)
            check_int8_add(d, f"{path}: operator {k} ADD")
        else:
            inp, flt = tensor(ins[0]), tensor(ins[1])
            bias = tensor(ins[2]) if len(ins) > 2 else None
            if inp["type"] != 9 or flt["type"] != 9 or out["type"] != 9:
                raise ValueError(f"{path}: input, filter and output must be int8")
            if np.any(flt["zp"] != 0):
                raise ValueError(f"{path}: TFLite's int8 filters are symmetric; a non-zero filter zero point is not supported")
            d = dict(kind=kind, input=inp, filter=flt, bias=bias, output=out)
        opt = opts[kind]()
        t = op.BuiltinOptions()
        opt.Init(t.Bytes, t.Pos)
        if kind == "CONV_2D":
            if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
                raise ValueError(f"{path}: dilation is not supported")
            d.update(same=opt.Padding() == 0, stride=(opt.StrideH(), opt.StrideW()))
        fa = opt.FusedActivationFunction()
        if fa not in acts:
            raise ValueError(f"{path}: fused activation {fa} is not supported")
        d["fused"] = acts[fa]
        d["act_range"] = ipu.activation_clamp(acts[fa], float(out["scale"][0]), int(out["zp"][0]))
        ops.append(d)
    return {"ops": ops, "input": g.Inputs(0), "output": g.Outputs(0)}


def load_layer(path: str) -> dict:
    """The one CONV_2D or FULLY_CONNECTED operator of an int8 .tflite, with its tensors' codes and quantization."""
    ops = load_int8_graph(path)["ops"]  # refuses any other operator
    kinds = [o["kind"] for o in ops]
    if kinds not in (["CONV_2D"], ["FULLY_CONNECTED"]):
        raise ValueError(f"{path}: operators {kinds}; expected one CONV_2D or FULLY_CONNECTED with int8 input and output")
    return ops[0]


def add_record_of_tflite(op: dict, first: int = None, name: str = "") -> AddRecord:
    """The TFLite importer's ADD record from load_int8_graph's op, tensor first (default its first input) as input 1."""
    first = op["inputs"][0] if first is None else first
    if first not in op["inputs"]:
        raise ValueError(f"ADD layer {name or '(unnamed)'}: tensor {first} is not one of its inputs {op['inputs']}")
    one, two = (op["a"], op["b"]) if first == op["inputs"][0] else (op["b"], op["a"])
    out = op["output"]
    lo, hi = op["act_range"]
    return AddRecord(float(one["scale"][0]), int(one["zp"][0]), float(two["scale"][0]), int(two["zp"][0]), float(out["scale"][0]),
                     int(out["zp"][0]), int(lo), int(hi), name, FUSED_RANGE[op["fused"]])


def job_of(layer: dict, x: np.ndarray, saved, name: str = "") -> tuple:
    """(int8 job for RtlLayer, output shape) of one TFLite layer on the saved input codes x, through its neutral record named name."""
    rec = record_of_tflite(layer, name)
    if max(abs(a - int(saved[k])) for a, k in zip((rec.lo, rec.hi), ("act_min", "act_max"))) > 1:  # rounding may move a bound one code
        raise ValueError(f"clamp ({rec.lo}, {rec.hi}) is not the npz's ({int(saved['act_min'])}, {int(saved['act_max'])}): the model and the npz do not belong together")
    return job_of_record(rec, x)


# ── 3. Middle end: jobs become N x N sets or a layer's streams; packing ───────

HW_BIAS = True  # the mesh adds the bias; False lowers it as a ones column and an extra depth row


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


WC_TILES = 128  # the weight cache (sienna_top's WC_TILES); a column block with more than half of it is streamed with its sets


def format_layer(job, N):
    """The configuration and the two input streams sienna_top's layer contract (sienna_sched) expects for a job, in its fixed order.

    Terms whose weight is an identity become the residual input; the rest are one product with their depths side by side.
    A depthwise layer gives each column block only its own channels' depth.
    A job with "req" is int8: every column block gets its bias beat, whose side words layer_epilogue gives; the beat's weight row is zeros."""
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
    int8 = job.get("req") is not None  # int8: every block has its bias beat, and the int32 bias rides beside it
    bias = job["bias"] if job["bias"] is not None and (int8 or np.any(job["bias"])) else None
    cached = rt > 1 and dt <= WC_TILES // 2

    def pad(a, rows, cols):
        out = np.zeros((rows, cols), np.float32)
        out[: a.shape[0], : a.shape[1]] = a
        return out

    Xp = pad(X, rt * N, max(X.shape[1], ct * N * (kk or 0)) if kk else dt * N)
    Wp = pad(W, max(W.shape[0], ct * N * (kk or 0)) if kk else dt * N, ct * N)
    Rp = pad(res[0], rt * N, ct * N) if res else None
    bp = np.zeros(ct * N, np.float32)
    if bias is not None and not int8:
        bp[: bias.size] = bias
    a_rows, w_rows = [], []
    for c in range(ct):
        k0 = c * N * kk if kk else 0  # depthwise: this block's channels only
        A_c = pad(Xp[:, k0 : k0 + kb], rt * N, dt * N)
        W_c = pad(Wp[k0 : k0 + kb, c * N : (c + 1) * N], dt * N, N)
        if bias is not None:
            w_rows.append(np.zeros((1, N), np.float32) if int8 else bp[c * N : (c + 1) * N][None, :])
        w_rows += [W_c] * (1 if cached else rt)
        for r in range(rt):
            for t in range(dt):  # the depth tiles of this row tile, N rows of N words each
                a_rows.append(A_c[r * N : (r + 1) * N, t * N : (t + 1) * N])
            if Rp is not None:
                a_rows.append(Rp[r * N : (r + 1) * N, c * N : (c + 1) * N])
    cfg = dict(m=M, kb=kb, n=C, residual=int(Rp is not None), bias=int(bias is not None), act=ACT_CODES[job["act"]])
    return cfg, np.vstack(a_rows), np.vstack(w_rows), (M, C, rt, ct)


def layer_epilogue(job, N):
    """int8: the words beside each column block's bias beat (N biases, N multipliers, N shifts, zero-padded), shape (blocks, 3, N)."""
    q, b = job["req"], np.asarray(job["bias"], np.int64)
    C = b.size
    ct = -(-C // N)
    pad = lambda v: np.concatenate([np.asarray(v, np.int64), np.zeros(ct * N - C, np.int64)])
    bb, mm, ss = pad(b), pad(q["mult"]), pad(q["shift"])
    return np.stack([np.stack([v[c * N:(c + 1) * N] for v in (bb, mm, ss)]) for c in range(ct)])


PACK_ENTRIES = 8  # sienna_top's parameter table


def pack_jobs(models: list, N: int, int8: bool) -> tuple:
    """Packs models into one layer (block c per model, an entry per distinct act / int8 output) -> (job, recipe); a set runs at its slowest act."""
    if not models:
        raise ValueError("nothing to pack")
    K = max(m["W"].shape[0] for m in models)
    C = max(m["W"].shape[1] for m in models)
    b = 2
    while b < max(K, C):
        b *= 2
    if b > N // 2:
        raise ValueError(f"a job of depth {K} and width {C} needs blocks of {b}; packing needs at most N/2 = {N // 2}")
    if len(models) > N // b:
        raise ValueError(f"{len(models)} models need {len(models)} blocks of {b}; N = {N} holds {N // b}")
    sh = (N // b).bit_length() - 1
    rows = max(sum(x.shape[0] for x in m["inputs"]) for m in models)
    M = -(-rows // N) * N
    A, B = np.zeros((M, N), np.float32), np.zeros((N, N), np.float32)
    bias = np.zeros(N, np.int64 if int8 else np.float32)
    mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
    keys, ents, mp, recipe = [], [], [0] * (N // 2), []
    for c, m in enumerate(models):
        k, cc = m["W"].shape
        X = np.vstack(m["inputs"]).astype(np.float32)
        A[:X.shape[0], c * b:c * b + k] = X
        B[c * b:c * b + k, c * b:c * b + cc] = m["W"]
        if m["bias"] is not None:
            bias[c * b:c * b + cc] = m["bias"]
        q = m.get("req")
        if int8:
            mult[c * b:c * b + cc], shift[c * b:c * b + cc] = q["mult"], q["shift"]
        key = (m["act"],) + (tuple(int(q[x]) for x in ("zp", "amin", "amax", "mx", "shx", "mout", "shout", "zout")) if int8 else ())
        if key not in keys:
            keys.append(key)
            ents.append((m["act"], {x: v for x, v in q.items() if x not in ("mult", "shift")} if int8 else None))
        mp[c] = keys.index(key)
        r0, spans = 0, []
        for x in m["inputs"]:
            spans.append((r0, x.shape[0]))
            r0 += x.shape[0]
        recipe.append((c * b, cc, spans))
    if len(ents) > PACK_ENTRIES:
        raise ValueError(f"{len(ents)} distinct activation / output settings; a packed set holds {PACK_ENTRIES}")
    ents += [("linear", None)] * (PACK_ENTRIES - len(ents))
    job = {"terms": [(A, B)], "bias": bias, "act": ents[0][0], "shape": (M, N), "pack": {"shift": sh, "map": mp, "ents": ents}}
    if int8:
        job["req"] = dict(ents[0][1], mult=mult, shift=shift)
    return job, recipe


def unpack(Y: np.ndarray, recipe: list) -> list:
    """Each model's outputs, one array per input, from a packed layer's result."""
    return [[Y[r0:r0 + m, c0:c0 + cc] for r0, m in spans] for c0, cc, spans in recipe]


def pack_precheck(pk, cfg, shape, N, lanes, tag="layer"):
    """Refuses a packed layer the layer contract cannot run: in silicon (no assertions) it would compute it silently wrong."""
    M, C, rt = shape
    if C != N or cfg["kb"] != N or rt * N != M:
        raise ValueError(f"{tag}: a packed layer is a whole number of row tiles, N columns and N deep")
    if cfg["residual"]:
        raise ValueError(f"{tag}: a packed layer takes no residual input")
    if lanes % N:
        raise ValueError(f"{tag}: packing needs NUM_LANES ({lanes}) to be N ({N}) or a multiple of it")
    if COLLAPSE_K == 0:
        raise ValueError(f"{tag}: the collapse-k 0 mesh refuses packed sets")
    if not 1 <= pk["shift"] < N.bit_length() - 1:
        raise ValueError(f"{tag}: pack shift {pk['shift']} is outside 1 .. log2(N) - 1 = {N.bit_length() - 2}")
    if len(pk["map"]) != N // 2 or not all(0 <= int(e) < PACK_ENTRIES for e in pk["map"]):
        raise ValueError(f"{tag}: the block map needs N/2 = {N // 2} entries in 0 .. {PACK_ENTRIES - 1}")
    if len(pk["ents"]) != PACK_ENTRIES:
        raise ValueError(f"{tag}: a packed layer needs all {PACK_ENTRIES} table entries, not {len(pk['ents'])}")


# ── 4. Device protocol: host to hardware; a silicon driver replaces 5-6 ──────

ACT_CODE = {"relu": 4, "linear": 5}  # the bypass modes of gpnae
ACT_CODES = {"linear": 5, "relu": 4, "selu": 1, "sigmoid": 2, "tanh": 3}


def bias_word_index(cfg, a, w) -> list:
    """TEST ONLY: the layer file word of each bias column; format_layer starts every column block's W rows with its bias row."""
    N = w.shape[1]
    ct = -(-cfg["n"] // N)
    if not cfg["bias"] or w.shape[0] % ct:
        raise ValueError(f"bias words: the layer has no bias row per column block ({w.shape[0]} W rows, {ct} blocks)")
    per = w.shape[0] // ct
    return [a.size + (c // N) * per * N + c % N for c in range(cfg["n"])]


def write_layer(path, cfg, a, w, fmt, req=None, pack=None, epilogue=None, bias_words=None, add=None):
    """TB_sienna (layer)'s layer file: L (configuration), int8's Q (requantize), an int8 residual layer's A (the ADD's words but zp1), a packed
    layer's P (shift, block map) and E (entries 1-7), rows of N words, int8's epilogue words."""
    if fmt == "int8" and cfg["residual"] and (add is None or int(add["zp1"]) != int(req["zp"])):  # zp1 is Q's zp, the conv output's
        raise ValueError("an int8 residual layer needs its ADD words (job['add']) with zp1 equal to the requantize zp")
    words = op_hex(np.concatenate([a.ravel(), w.ravel()]), fmt)
    if bias_words is not None:  # TEST ONLY: raw bias words, such as bf16 subnormals, which op_hex flushes to a signed zero
        for i, b in zip(bias_word_index(cfg, a, w), bias_words):
            if words[i] != op_hex(bits_float([b], fmt), fmt)[0]:  # the job's bias values must already be at these words
                raise ValueError(f"bias words: layer word {i} is {words[i]}, not the bias column's {int(b):x} flushed")
            words[i] = f"{int(b):0{len(words[i])}x}"
    with open(path, "w") as f:
        f.write(f"L {cfg['m']} {cfg['kb']} {cfg['n']} {cfg['residual']} {cfg['bias']} {cfg['act']} 0 0 {len(a)} {len(w)} {int(bool(pack))}\n")
        if fmt == "int8":
            f.write(f"Q {req['zp']} {req['amin']} {req['amax']} {req['mx']} {req['shx']} {req['mout']} {req['shout']} {req['zout']}\n")
            if cfg["residual"]:
                f.write("A " + " ".join(str(int(add[k])) for k in ADD_FIELDS[1:]) + "\n")
        if pack:
            f.write("P " + " ".join(str(int(v)) for v in [pack["shift"]] + list(pack["map"])) + "\n")
            for act, rq_e in pack["ents"][1:]:
                q = rq_e or {}
                f.write(f"E {ACT_CODES[act]} " + " ".join(str(int(q.get(x, 0))) for x in
                                                        ("zp", "amin", "amax", "mx", "shx", "mout", "shout", "zout")) + "\n")
        f.write("\n".join(words))
        f.write("\n")
        if fmt == "int8":
            f.write("".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in epilogue.ravel()))


def read_outputs(path, fmt="fp32", test=None):
    """The result file (an "S" line per output set, then its words) as float32 sets; words are in the build's format, a narrow one widened exactly; int8 as signed integer codes.
    An x or z word raises fourstate.Unknown (a ValueError) naming the word and test."""
    test, k = test or os.path.basename(path), 0
    word = lambda line: fourstate.word(line, test, k)
    if fmt == "int8":  # two's-complement codes
        sets, cur = [], None
        for line in open(path):
            if line.startswith("S "):
                cur = []
                sets.append(cur)
            else:
                cur.append(word(line))
                k += 1
        return [((np.array(s, np.int64) + 128) % 256) - 128 for s in sets]
    sh = 23 - FORMATS[fmt][1]
    sets = []
    cur = None
    for line in open(path):
        if line.startswith("S "):
            cur = []
            sets.append(cur)
        else:
            cur.append(word(line) << sh)
            k += 1
    return [np.array(s, dtype=np.uint32).view(np.float32) for s in sets]


def write_sets(path, groups, act):
    """TB_sienna (sets)'s set file: the set count, then per set "partial code bias" and its A, B (and bias) words as fp32 hex; returns the count."""
    code = ACT_CODES[act]  # every activation: the set engine runs the lanes too
    lines = []
    n = 0
    for _, passes, bias in groups:
        for k, (A, B) in enumerate(passes):
            partial = int(k < len(passes) - 1)
            with_bias = int(bias is not None and k == 0)  # the first pass carries the group's bias
            lines.append(f"{partial} {code} {with_bias}")
            parts = [A.ravel(), B.ravel()] + ([bias] if with_bias else [])
            words = np.concatenate(parts).astype(np.float32).view(np.uint32)
            lines.append("\n".join(f"{v:08x}" for v in words.tolist()))
            n += 1
    with open(path, "w") as f:
        f.write(f"{n}\n" + "\n".join(lines) + "\n")
    return n


# ── 5. Device build: the package the simulators compile against ──────────────

SETS_IN_FLIGHT = 2 + 2 + 4 + 4 + 2 + 1  # sienna_top's default credits (its banks, ACC_BANKS=RESULT_BANKS=4); the testbenches read it from the package
COLLAPSE_K = 1  # the mesh's COLLAPSE_K the build uses; --collapse-k sets it for the bit-exact golden
TB_DIR = os.path.join(ROOT, "testbenches")


TB_KNOBS = ("ACTIVATION_CODE", "NUM_SETS", "HAS_BIAS", "WEIGHT_CACHE", "ACCUM_PASSES", "MIXED_LEN", "MIXED_ACTS", "PACKED", "TRAINING_MODE",
            "DROPOUT_SEED", "RESIDUAL")  # TB_sienna (pipeline) reads these at run time (test_knobs.mem), so one build runs every test that agrees on the rest


def write_test_config(path: str, items: list) -> None:
    """test_config_pkg.sv with every item, and beside it test_knobs.mem: the TB_KNOBS items, lowercase name and value, one per line."""
    write_sv_package(path, items)
    with open(os.path.join(os.path.dirname(path), "test_knobs.mem"), "w") as f:
        f.writelines(f"{k.lower()} {v}\n" for k, v, _ in items if k in TB_KNOBS)


def write_sv_package(path: str, items: list) -> None:
    with open(path, "w") as f:
        f.write("// Auto-Generated Configuration Package\npackage test_config_pkg;\n\n")
        for name, val, vtype in items:
            kw = "shortreal" if vtype == "float" else "int"
            f.write(f"  localparam {kw} {name} = {val};\n")
        f.write("\nendpackage\n")


def _config_items(cfg: dict, fmt: str, act_type: str, num_sets: int, credits: int, passes: int, mixed: list,
                  use_bias: bool, drop_seed: int) -> list:
    """test_config_pkg's items: geometry, activation, pooling, dropout and the streamed sets; the format and its widths are the build's define's."""
    N = cfg.get("n", 16)
    sram_depth = N * N
    return [
        ("N", N, "int"),
        ("TILE_SIZE", cfg.get("tile_size", 4), "int"),
        ("NUM_LANES", cfg.get("lanes", 32), "int"),
        ("HOST_WORDS", cfg.get("host_words", N), "int"),
        ("EXACT_GOLDEN", int(fmt != "fp32"), "int"),  # the bench's golden: bit-exact in narrow formats; widths and the format are the define's
        ("SRAM_DEPTH", sram_depth, "int"),
        ("FIFO_DEPTH", cfg.get("fifo_depth", sram_depth), "int"),
        ("ACTIVATION_CODE", activation_to_code(act_type), "int"),
        ("IN_ROWS", N, "int"),
        ("IN_COLS", N, "int"),
        ("POOL_H", cfg.get("pool_h", 2), "int"),
        ("POOL_W", cfg.get("pool_w", 2), "int"),
        ("STRIDE_ROWS", cfg.get("pool_h", 2), "int"),
        ("STRIDE_COLS", cfg.get("pool_w", 2), "int"),
        ("PADDING", cfg.get("padding", 1), "int"),
        ("DROPOUT_P_PERCENT", int(round(cfg.get("dropout_p", 0.5) * 100)), "int"),
        ("LFSR_WIDTH", 32, "int"),
        ("CONTROL_WIDTH", 3, "int"),
        ("NUM_SETS", num_sets, "int"),
        ("SETS_IN_FLIGHT", credits, "int"),
        ("HAS_BIAS", int(use_bias), "int"),
        ("WEIGHT_CACHE", int(bool(cfg.get("cached", False))), "int"),
        ("ACCUM_PASSES", passes, "int"),
        ("MIXED_LEN", len(mixed), "int"),
        ("MIXED_ACTS", sum(activation_to_code(a) << (4 * i) for i, a in enumerate(mixed)), "int"),
        ("PACKED", int(bool(cfg.get("packed", False))), "int"),
        ("TRAINING_MODE", int(bool(cfg.get("training", False))), "int"),
        ("DROPOUT_SEED", drop_seed, "int"),
        ("RESIDUAL", int(bool(cfg.get("residual") or cfg.get("residual_every"))), "int"),  # int8 residual: add_<k>.mem gives each set's flags
        ("ADDR_LINES", max(1, math.ceil(math.log2(sram_depth))), "int"),
        ("COLLAPSE_K", COLLAPSE_K, "int"),  # TB_sienna (pipeline) sets sienna_top's mesh mode from it: a build item, not a run-time knob
    ]


def write_build_pkg(N: int, T: int, lanes: int, fmt: str) -> None:
    """test_config_pkg.sv of the model simulators' build (RtlSets, RtlLayer): matmul_relu_nopool's configuration in fmt, no stimulus."""
    os.makedirs(TB_DIR, exist_ok=True)
    cfg = {"n": N, "tile_size": T, "lanes": lanes, "host_words": N, "pool_h": 1, "pool_w": 1, "padding": 0}
    drop_seed = 0x2ACE0000 + 42 + int(os.environ.get("SIENNA_SEED", "0"))  # the generators' seed, so the package matches theirs
    write_test_config(os.path.join(TB_DIR, "test_config_pkg.sv"),
                      _config_items(cfg, fmt, "relu", SETS_IN_FLIGHT + 2, SETS_IN_FLIGHT, 1, [], fmt == "int8", drop_seed))


# ── 6. Backends: build(), run_job(job, tag) -> (outputs, sets, cycles) ───────

def check_layer_mode(text, tag):
    """Raises unless TB_sienna (layer)'s [LAYER] line gives the elaborated mesh's collapse_k and it is COLLAPSE_K."""
    m = re.search(r"\[LAYER\] sets=.*\bcollapse_k=(\d+)", text)
    if not m:
        raise RuntimeError(f"{tag}: the [LAYER] line has no collapse_k: the mesh's mode is unknown")
    if int(m.group(1)) != COLLAPSE_K:
        raise RuntimeError(f"{tag}: the build's mesh runs collapse-k {m.group(1)}, the run asked for {COLLAPSE_K}")


def read_stage_dump(path, N, rt, ct, np_):
    """TB_sienna (layer)'s int8 stage dump (the lines after the last H) as tile grids (rt N x ct N) per pass p of each output tile: {"X": {p: grid},
    "R": {p: grid}, "Q": {p: (q1, q2, y)}, "np": np_}; set k is tile (c, r)'s pass k % np_ with k // np_ = c rt + r. Raises on a duplicate or missing element."""
    lines = open(path).read().splitlines()
    last = max((i for i, l in enumerate(lines) if l.startswith("H ")), default=-1)
    got = {}
    for l in lines[last + 1:]:
        f = l.split()
        k, e = int(f[1]), int(f[2])
        t, p = divmod(k, np_)
        c, r = divmod(t, rt)
        key = (f[0], p)
        if key not in got:
            got[key] = np.full((rt * N, ct * N, 3 if f[0] == "Q" else 1), np.iinfo(np.int64).min, np.int64)
        g = got[key][r * N + e // N, c * N + e % N]
        if g[0] != np.iinfo(np.int64).min:
            raise RuntimeError(f"stage dump: set {k} element {e} has two {f[0]} lines")
        g[:] = [int(v) for v in f[3:]]
    out = {"np": np_}
    for (kind, p), g in got.items():
        if (g == np.iinfo(np.int64).min).any():
            raise RuntimeError(f"stage dump: pass {p}'s {kind} lines miss {int((g[..., 0] == np.iinfo(np.int64).min).sum())} of {g.shape[0] * g.shape[1]} elements")
        out.setdefault(kind, {})[p] = tuple(g[..., i] for i in range(3)) if kind == "Q" else g[..., 0]
    return out


INT8_FIXED_OUT = {"sigmoid": (1 / 256, -128), "tanh": (1 / 128, 0)}  # the converters' fixed output quantization (a convention)


def lane_quant(act, s_in, z_in, rq) -> tuple:
    """(s_in, z_in, s_out, z_out) the int8 lane's true function is judged on: SELU's calibrated output, tanh's and sigmoid's fixed one, ReLU and
    linear the input's own (their gate is the identity, so s_in may be None there)."""
    if s_in is None and act not in ("relu", "linear"):
        raise ValueError(f"the {act} lane's gate needs the requantize output scale (s_out)")
    s_in = 1.0 if s_in is None else s_in
    if act == "selu":
        return s_in, z_in, rq["s_selu"], rq["zout"]
    return (s_in, z_in) + INT8_FIXED_OUT.get(act, (s_in, z_in))


def _columns(job, C):
    """Per output column: its activation, requantize words (ratio, zp, amin, amax) and lane scales, from job['req'] or a packed layer's entries."""
    req, pk = job["req"], job.get("pack")
    ratio = req.get("ratio")
    ratio = code_scales(req["mult"], req["shift"]) if ratio is None else np.asarray(ratio, np.float64)
    if ratio.size < C:
        raise ValueError(f"the requantize gate needs a real scale per column: {ratio.size} for {C} columns")
    cols = []
    for c in range(C):
        act, q = job["act"], req
        if pk:
            act, q = pk["ents"][int(pk["map"][c // (C >> pk["shift"])])]
            q = q or req
        cols.append((act, float(ratio[c]), q))
    return cols


def true_clamp(kind, s_out, zp) -> tuple:
    """(lo, hi) in output codes, unrounded, of a fused kind's or a (lo, hi) real range on an output of scale s_out and zero point zp."""
    lo, hi = tuple(kind) if isinstance(kind, (tuple, list)) else FUSED_RANGE[kind]
    code = lambda v, d: d if v is None else min(127.0, max(-128.0, zp + v / s_out))
    return code(lo, -128.0), code(hi, 127.0)


def _stage(stages, kind, p):
    """One pass's grid of one kind from read_stage_dump; a missing one is a RuntimeError, as a missing element is."""
    if p not in stages.get(kind, {}):
        raise RuntimeError(f"stage dump: no {kind} lines for pass {p}")
    return stages[kind][p]


def gate_transfer(name, want, got) -> "truth.Result":
    """An exact stage (a buffer or a mux): every code must arrive unchanged."""
    want, got = (np.asarray(v, np.int64).ravel() for v in (want, got))
    bad = np.flatnonzero(want != got)
    return truth.Result(bad.size == 0 and want.size > 0, "int8", name, int(want.size), float(bad.size), 0.0, int(bad[0]) if bad.size else 0,
                        int(bad.size), int(bad.size), 0, 0, 0, [(int(i), int(got[i]), int(want[i])) for i in bad[:truth.FIRST]])


def layer_stage_gates(job, Y, stages, N) -> list:
    """[(stage, truth.Result)] of an int8 layer from TB_sienna (layer)'s stage dump: the conv's requantize on each exact int32 sum and its column's real
    scale; for a residual layer (job['add']) the identity requantize of the residual codes, hold_mem's transfer and the ADD on its real ratios, with
    the true clamps of job['clamps'] when given; the lane from the activation stage's input X to the output Y (ReLU and linear at truth.TARGET, the
    others at job['tol'], default their tier's tolerance, gpnae_model.TIER_TOL)."""
    eye = lambda W: W.shape[0] == W.shape[1] and np.array_equal(W, np.eye(W.shape[0], dtype=W.dtype))
    res = [X for X, W in job["terms"] if eye(W)] if job.get("add") is not None else []  # the residual term only in a layer with an ADD
    if job.get("add") is not None and len(res) != 1:
        raise ValueError(f"a residual layer needs one identity term, not {len(res)}")
    dense = [(X, W) for X, W in job["terms"] if not any(X is r for r in res)]
    A_q = np.hstack([np.asarray(x, np.int64) for x, _ in dense])
    W_q = np.vstack([np.asarray(w, np.int64) for _, w in dense])
    acc = wrap32(imatmul(A_q, W_q) + np.asarray(job["bias"], np.int64)[None, :])
    M, C = acc.shape
    cols = _columns(job, C)
    S = np.array([s for _, s, _ in cols])[None, :]
    zp, lo, hi = (np.array([q[k] for _, _, q in cols])[None, :] for k in ("zp", "amin", "amax"))
    relu = np.array([a == "relu" for a, _, _ in cols])[None, :]
    if not res:
        lo = np.where(relu, np.maximum(lo, zp), lo)  # ReLU's true clamp starts at its zero point
    np_ = stages["np"]
    crop = lambda g: g[:M, :C]
    out, Y = [], np.asarray(Y, np.int64)
    if res:  # the hold set (pass np_ - 2) and the add set (pass np_ - 1)
        add, R_q = job["add"], np.asarray(res[0], np.int64)[:M, :C]
        Rh = crop(_stage(stages, "R", np_ - 2))
        q1, q2, y = (crop(g) for g in _stage(stages, "Q", np_ - 1))
        lane = job["lane"]
        if job.get("clamps"):  # the conv's and the ADD's true clamps from their kinds, scales and zero points
            lo, hi = true_clamp(job["clamps"][0], job["req"]["s_out"], job["req"]["zp"])
            a_lo, a_hi = true_clamp(job["clamps"][1], lane["s_out"], add["zpo"])
        else:
            a_lo, a_hi = add["amin"], add["amax"]
        out.append(("requant", truth.gate_requantize(acc, S, zp, lo, hi, Rh)))
        out.append(("requant identity", truth.gate_requantize(R_q, 1.0, 0, -128, 127, q2)))
        out.append(("hold_mem", gate_transfer("hold_mem", Rh, q1)))  # each element its own hold beat
        r1, r2 = add["ratios"]
        a_lo = np.where(relu, np.maximum(a_lo, add["zpo"]), a_lo)  # a ReLU after the ADD is its clamp
        out.append(("add", truth.gate_quantized_add(q1, q2, r1, add["zp1"], r2, add["zp2"], 1.0, add["zpo"], a_lo, a_hi, y)))
        X = crop(_stage(stages, "X", np_ - 1))  # the ADD's output as the activation stage takes it (fill_d, the Q line's y too)
        quants = [lane_quant(a, lane["s_out"], add["zpo"], lane) for a, _, _ in cols]
    else:
        X = crop(_stage(stages, "X", np_ - 1))
        out.append(("requant", truth.gate_requantize(acc, S, zp, lo, hi, X)))
        quants = [lane_quant(a, q.get("s_out"), q["zp"], q) for a, _, q in cols]
    acts = np.array([a for a, _, _ in cols])
    for a in sorted(set(acts.tolist())):
        m = np.broadcast_to(acts[None, :] == a, X.shape)
        qa = np.array([quants[c] for c in range(C)], np.float64)[None, :, :].repeat(M, 0)[m]
        quant = tuple(qa[:, i] for i in range(4))
        bound = truth.TARGET if a in ("relu", "linear") else job.get("tol", gpnae_model.TIER_TOL[int(job.get("tier", 0))])
        r = truth.gate("int8", "linear" if a == "relu" else a, X[m], Y[m], quant=quant, bound=bound)
        out.append((f"activation {a}", r._replace(func=a)))
    return out


class RtlLayer:
    """TB_sienna (layer): one layer per run; software writes the configuration and the streams, then reads the results."""

    def __init__(self, N, lanes, work, fmt_name="fp32", tile_size=4):
        self.N, self.lanes, self.work, self.fmt_name, self.T = N, lanes, work, fmt_name, tile_size
        self.bin = os.path.join(SIM_DIR, "TB_sienna_layer_sim")
        self.cycles = self.sets = self.words = 0
        self.first_hold = self.first_free = self.last_log = self.last_stages = None
        self.retries = 0  # Verilator internal-error rebuilds

    def build(self):
        write_build_pkg(self.N, self.T, self.lanes, self.fmt_name)
        cmd = ["make", SIM_TARGET, "TB_MODE=layer", "TRACE=0", f"FMT={self.fmt_name}", f"N={self.N}",
               f"TILE={self.T}", f"LANES={self.lanes}", f"COLLAPSE_K={COLLAPSE_K}",
               "GEN_PKG=0"]  # the package write_build_pkg just wrote; the guard checks its COLLAPSE_K against this one
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 and "Internal Error" in r.stdout + r.stderr:  # Verilator's intermittent V3TSP error: one rebuild, never more
            self.retries += 1
            print(f"[RETRY] Verilator internal error, rebuilding TB_sienna (layer) (retry {self.retries}): "
                  f"{next((l for l in (r.stdout + r.stderr).splitlines() if 'Internal Error' in l), '').strip()}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(self.bin):
            sys.stdout.write(r.stdout[-4000:] + r.stderr[-4000:])
            raise RuntimeError("TB_sienna (layer) build failed")

    def run_job(self, job, tag, plusargs=()):
        """One job as one layer -> (outputs, sets, cycles): refuses what the RTL cannot run, writes the layer file, runs TB_sienna (layer) (with plusargs
        and $SIM_ARGS), reads the results; int8 keeps the per-stage dump in last_stages, and every run its log, first hold_done cycle and the cycle its hold window closed."""
        N = self.N
        int8 = self.fmt_name == "int8"
        if int8 and job.get("req") is None:
            raise ValueError(f"{tag}: an int8 job needs its requantize parameters (job['req'])")
        rq = job.get("req")
        cfg, a, w, (M, C, rt, ct) = format_layer(job, N)
        pk = job.get("pack")
        if pk:
            pack_precheck(pk, cfg, (M, C, rt), N, self.lanes, tag)
        lf = os.path.join(self.work, f"{tag}.layer")
        of = os.path.join(self.work, f"{tag}.out")
        df = os.path.join(self.work, f"{tag}.stages")
        write_layer(lf, cfg, a, w, self.fmt_name, rq, pk, layer_epilogue(job, N) if int8 else None, job.get("bias_words"), add=job.get("add"))
        dump = [f"+dump={df}"] if int8 else []
        lanes = [f"+tier={int(job.get('tier', 0))}"]  # the layer's lane tier (pick_tier), and a coefficient image loaded after reset
        cf = os.path.join(self.work, f"{tag}.coef")
        if job.get("coef") is not None:
            write_coef(cf, job["coef"])
            lanes.append(f"+coef={cf}")
        bf, ff = os.path.join(self.work, f"{tag}.between"), os.path.join(self.work, f"{tag}.first")
        if job.get("between") is not None:  # the layer runs at the first image, then again after this image's load with no reset
            write_coef(bf, job["between"])
            lanes += [f"+between={bf}", f"+out_first={ff}"]
        if job.get("coef_busy"):  # writes to the running activation's own words while the layer streams, which the interlock must refuse
            kf = os.path.join(self.work, f"{tag}.busy")
            write_writes(kf, busy_writes(job.get("coef") or gpnae_model.default_mem(self.fmt_name), job["act"], int(job.get("tier", 0))))
            lanes.append(f"+coef_busy={kf}")
        r = subprocess.run([self.bin, f"+layer={lf}", f"+out={of}", *dump, *lanes, *plusargs, *os.environ.get("SIM_ARGS", "").split()],
                           cwd=os.path.dirname(self.bin), capture_output=True, text=True)
        if job.get("coef") is not None:
            os.remove(cf)
            os.remove(cf + ".img")
        self.first_Y = None
        if FIRED.search(r.stdout + r.stderr) is None:  # a count with an x or z fails the layer by name (a ValueError its callers report)
            for pat, what in ((rf"\[LAYER\] sets=\S+ outputs=\S+ cycles=({fourstate.DEC})", "cycle count"), (rf"\[LAYER\] sets=({fourstate.DEC})", "set count")):
                try:
                    fourstate.field(pat, r.stdout, tag, what)
                except fourstate.Unknown as e:
                    raise RuntimeError(f"{tag}: {fourstate.fail_line(e)}")
        b = re.search(r"\[LAYER\] coef busy: (\d+) writes while the layer streamed, (\d+) refused", r.stdout)
        self.busy_refused = (int(b.group(1)), int(b.group(2))) if b else None
        g = re.search(r"\[Between\] loaded .* its first write (\d+) cycles after done_o", r.stdout)
        self.between_gap = int(g.group(1)) if g else None  # +between: how soon after the layer's done_o the load began
        self.last_log, self.last_stages = r.stdout, None
        h = re.search(r"\[RES\] first hold_done (\d+) cycles into the layer", r.stdout)
        self.first_hold = int(h.group(1)) if h else None
        f = re.search(r"\[RES\] first hold window closed (\d+) cycles into the layer", r.stdout)
        self.first_free = int(f.group(1)) if f else None
        if FIRED.search(r.stdout + r.stderr):  # neither assertions nor the TB's own checks change the exit code
            sys.stdout.write((r.stdout + r.stderr)[-3000:])
            raise RuntimeError(f"{tag}: an assertion or a testbench check failed in the layer simulation")
        coef_image_ran(r.stdout, tag, int(job.get("coef") is not None) + int(job.get("between") is not None))
        m = re.search(r"\[LAYER\] sets=(\d+) outputs=(\d+) cycles=(\d+) a_rows=(\d+)/(\d+) w_rows=(\d+)/(\d+)", r.stdout)
        e = re.search(r"epilogues=(\d+)/(\d+)", r.stdout)
        if not m or m.group(4) != m.group(5) or m.group(6) != m.group(7) or (int8 and (not e or e.group(1) != e.group(2))):
            sys.stdout.write(r.stdout[-3000:])
            raise RuntimeError(f"{tag}: layer simulation failed")
        check_layer_mode(r.stdout, tag)
        outs = read_outputs(of, self.fmt_name, tag)
        if job.get("between") is not None:
            self.first_Y = self._tiles(read_outputs(ff, self.fmt_name), rt, ct, int8, tag)[:M, :C]
            os.remove(bf)
            os.remove(bf + ".img")
            os.remove(ff)
        if int8:
            self.last_stages = read_stage_dump(df, N, rt, ct, -(-cfg["kb"] // N) + cfg["residual"])
            os.remove(df)
        os.remove(lf)
        os.remove(of)
        Y = self._tiles(outs, rt, ct, int8, tag)
        n, cyc = int(m.group(1)), int(m.group(3))
        self.sets += n
        self.cycles += cyc
        self.words += (len(a) + len(w)) * N
        return Y[:M, :C], n, cyc

    def _tiles(self, outs, rt, ct, int8, tag):
        """The output tiles in the spec's order (column blocks outer, row tiles inner) as one matrix."""
        N = self.N
        if len(outs) != rt * ct:
            raise RuntimeError(f"{tag}: {len(outs)} output tiles, expected {rt * ct}")
        Y = np.zeros((rt * N, ct * N), np.int64 if int8 else np.float32)
        k = 0
        for c in range(ct):
            for r_ in range(rt):
                Y[r_ * N : (r_ + 1) * N, c * N : (c + 1) * N] = outs[k].reshape(N, N)
                k += 1
        return Y

    def exact(self, job):
        """Bit-exact words run_job must return for a float job, from the streams format_layer sends: per tile its depth passes, the residual's identity pass, the bias."""
        N, fmt = self.N, self.fmt_name
        cfg, a, w, (M, C, rt, ct) = format_layer(job, N)
        dt = -(-cfg["kb"] // N)
        cached = rt > 1 and dt <= WC_TILES // 2  # sienna_sched's rule: a cached block's weight tiles are sent once
        ab, wb = fmt_bits(a, fmt), fmt_bits(w, fmt)
        eye = fmt_bits(np.eye(N, dtype=np.float32), fmt)  # the residual pass's B, sienna_sched's ONE_VALUE on the diagonal
        Y = np.zeros((rt * N, ct * N), np.int64)
        ai = wi = 0
        for c in range(ct):
            b = None
            if cfg["bias"]:
                b, wi = wb[wi], wi + 1
                if job.get("bias_words") is not None:  # TEST ONLY: the raw words write_layer sent
                    seg = np.asarray(job["bias_words"], np.int64)[c * N:(c + 1) * N]
                    b = np.concatenate([seg, b[seg.size:]])
            for r in range(rt):
                if r == 0 or not cached:
                    Wt, wi = [wb[wi + t * N:wi + (t + 1) * N] for t in range(dt)], wi + dt * N
                passes, ai = [(ab[ai + t * N:ai + (t + 1) * N], Wt[t]) for t in range(dt)], ai + dt * N
                if cfg["residual"]:
                    passes, ai = passes + [(ab[ai:ai + N], eye)], ai + N
                Y[r * N:(r + 1) * N, c * N:(c + 1) * N] = exact_sum(passes, b, job["act"], N, fmt, self.T, tier=int(job.get("tier", 0)),
                                                                     mem=job.get("coef"))
        if ai != len(ab) or wi != len(wb):
            raise RuntimeError(f"exact: decoded {ai}/{len(ab)} A rows and {wi}/{len(wb)} W rows of format_layer's streams")
        return Y[:M, :C]


class RtlSets:
    """The TB_sienna (sets) binary, built once per configuration and run once per layer."""

    def __init__(self, N, lanes, work, host_gaps=False, tile_size=4):
        self.N, self.lanes, self.work, self.T = N, lanes, work, tile_size
        self.host_gaps = host_gaps  # TB_sienna (pipeline)'s handshake instead of a streaming host
        self.bin = os.path.join(SIM_DIR, "TB_sienna_sets_sim")
        self.cycles = 0
        self.sets = 0

    def build(self):
        if COLLAPSE_K == 0:
            raise ValueError("TB_sienna (sets) builds collapse-k 1")
        write_build_pkg(self.N, self.T, self.lanes, "fp32")
        r = subprocess.run(["make", SIM_TARGET, "TB_MODE=sets", "TRACE=0",
                            "FMT=fp32", f"N={self.N}", f"TILE={self.T}", f"LANES={self.lanes}",
                            "GEN_PKG=0"],  # the package write_build_pkg just wrote, which is fp32
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(self.bin):
            sys.stdout.write(r.stdout[-4000:] + r.stderr[-4000:])
            raise RuntimeError("TB_sienna (sets) build failed")

    def run(self, groups, act, tag, tier=0, coef=None, busy=False):
        """groups' sets through TB_sienna (sets) at the lanes' tier, after loading the coefficient image coef (a gpnae_model.Mem) when given;
        busy: busy_writes to act's words while the sets stream, which the interlock must refuse (counted in busy_refused)."""
        sets_f = os.path.join(self.work, f"{tag}.sets")
        out_f = os.path.join(self.work, f"{tag}.out")
        n = write_sets(sets_f, groups, act)
        lanes = [f"+tier={int(tier)}"]
        if coef is not None:
            write_coef(os.path.join(self.work, f"{tag}.coef"), coef)
            lanes.append(f"+coef={os.path.join(self.work, f'{tag}.coef')}")
        if busy:
            kf = os.path.join(self.work, f"{tag}.busy")
            write_writes(kf, busy_writes(coef or gpnae_model.default_mem("fp32"), act, int(tier)))
            lanes.append(f"+coef_busy={kf}")
        r = subprocess.run([self.bin, f"+sets={sets_f}", f"+out={out_f}"] + lanes + (["+host_gaps"] if self.host_gaps else []),
                           cwd=os.path.dirname(self.bin),
                           capture_output=True, text=True)
        if FIRED.search(r.stdout + r.stderr):  # neither assertions nor the TB's own checks change the exit code
            sys.stdout.write((r.stdout + r.stderr)[-3000:])
            raise RuntimeError(f"{tag}: an assertion or a testbench check failed in the model simulation")
        coef_image_ran(r.stdout, tag, int(coef is not None))
        try:  # a count with an x or z fails the model run by name
            fourstate.field(rf"\[MODEL\] sets=\S+ outputs=\S+ cycles=({fourstate.DEC})", r.stdout, tag, "cycle count")
        except fourstate.Unknown as e:
            raise RuntimeError(f"{tag}: {fourstate.fail_line(e)}")
        b = re.search(r"\[MODEL\] coef busy: (\d+) writes while sets streamed, (\d+) refused", r.stdout)
        self.busy_refused = (int(b.group(1)), int(b.group(2))) if b else None
        m = re.search(r"\[MODEL\] sets=(\d+) outputs=(\d+) cycles=(\d+) mesh_busy=(\d+) act_busy=(\d+) order_errors=(\d+)", r.stdout)
        if not m or int(m.group(6)) != 0:
            sys.stdout.write(r.stdout[-3000:])
            raise RuntimeError(f"{tag}: simulation failed")
        outs = read_outputs(out_f, "fp32", tag)
        os.remove(sets_f)
        os.remove(out_f)
        cyc = int(m.group(3))
        self.cycles += cyc
        self.sets += n
        return outs, n, cyc

    def run_job(self, job, tag):
        """One job as N x N sets in tile order -> (outputs, sets, cycles); a tile's result is its last set's, partial sets return none."""
        N = self.N
        groups, (P, C, rt, ct) = tile_job(job, N)
        outs, n, cyc = self.run(groups, job["act"], tag, int(job.get("tier", 0)), job.get("coef"), bool(job.get("coef_busy")))
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

    def exact(self, job):
        """Bit-exact fp32 words run_job must return: per group its passes (zero weight tiles skipped, as tile_job sends them) and its bias, then the lane."""
        N = self.N
        groups, (P, C, rt, ct) = tile_job(job, N)
        Y = np.zeros((rt * N, ct * N), np.int64)
        for (r, c), passes, bias in groups:
            pb = [(word_bits(A, "fp32"), word_bits(B, "fp32")) for A, B in passes]  # write_sets sends the float32 words as they are
            Y[r * N:(r + 1) * N, c * N:(c + 1) * N] = exact_sum(pb, None if bias is None else word_bits(bias, "fp32"), job["act"], N, "fp32", self.T,
                                                                 tier=int(job.get("tier", 0)), mem=job.get("coef"))
        return Y[:P, :C]


class Emulator(RtlSets):
    """Numpy stand-in for the RTL with the same set stream: checks tiling and reassembly, not the hardware."""

    exact = None  # not the hardware: no bit-exact model, only the fp32 bound

    def build(self):
        pass

    def run(self, groups, act, tag, tier=0, coef=None, busy=False):  # ReLU and linear only: the lanes' tier, memory and load path do not apply
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


# ── 7. Runtime: a model graph through a backend ──────────────────────────────

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


# The model gate's bounds, one place; the hardware itself is judged bit for bit against the backend's exact model in every float format.
LAYER_BOUND = {  # max|hw-ref|/max|ref| per layer against float64 on the same operands: the check a bug shared by the RTL and its exact model must pass
    "fp32": 1e-4,  # also regression.py's gemm bound; measured worst 3.29e-05 over 187 layers (mg_fp32c), 3x margin
    "bf16": 0.02,  # fp32 sums: measured worst 3.77e-03 (vww L04, ba3_models_N16_bf16), about half a bf16 ulp of the largest output (2^-8 = 3.9e-03 is the RNE limit), 5x margin; bf16 sums were 3.35e-02 to 1.83 in every model, so they fail it
}
CLASSIFIERS = ("resnet8", "kws", "vww")  # top-1 must equal the float reference's in every format
SCORE_BOUND = {"fp32": 1e-3, "bf16": 1e-2}  # ad01 score |hw-ref|/ref vs the float model: fp32 measured 1.85e-05 (54x margin); bf16 with fp32 sums measured 2.24e-03 (ba3_models_N16_bf16, 4.5x margin)
REPORTED = set()  # (model, format) known limits: float checks shown as FAIL, not counted; bit-exact still gates; ad01 bf16 left it once bf16 summed in fp32


def parse_fault(spec):
    """TEST ONLY: word|shared:MODEL:LAYER:INDEX[:BIT] flips one output word's bit after the simulation (shared: in the exact model too); top1:MODEL swaps the top two outputs."""
    if not spec:
        return None
    p = spec.split(":")
    if p[0] == "top1" and len(p) == 2:
        return {"kind": "top1", "model": p[1], "spec": spec, "applied": False}
    if p[0] in ("word", "shared") and len(p) in (4, 5):
        return {"kind": p[0], "model": p[1], "layer": p[2], "index": int(p[3]), "bit": int(p[4]) if len(p) == 5 else 0, "spec": spec, "applied": False}
    raise ValueError(f"--fault {spec}: expected word:MODEL:LAYER:INDEX[:BIT], shared:MODEL:LAYER:INDEX[:BIT] or top1:MODEL")


def flip_bit(y, fmt, index, bit):
    """TEST ONLY: a layer's float32 outputs with bit `bit` of word `index` (in the build's format) flipped."""
    if not 0 <= bit < 1 + sum(FORMATS[fmt]):
        raise ValueError(f"--fault: bit {bit} is outside a {fmt} word")
    y = np.array(y, np.float32)
    y.view(np.uint32).flat[index] ^= np.uint32(1 << (bit + 23 - FORMATS[fmt][1]))
    return y


def check_layer(sim, job, y, fmt, fault=None):
    """Hardware correctness of one layer -> (y as the next layer takes it, record): every word against the backend's bit-exact model; a test-only fault flips one first."""
    if fault and 0 <= fault["index"] < np.size(y):  # an index outside the layer leaves the fault unapplied, which gate_verdict fails
        y = flip_bit(y, fmt, fault["index"], fault["bit"])
        fault["applied"] = True
    else:
        fault = None
    exact = getattr(sim, "exact", None)
    if exact is None:
        if isinstance(sim, Emulator):
            return y, None  # the numpy stand-in is not hardware; only its float bound applies
        raise RuntimeError(f"{type(sim).__name__} has no bit-exact model (exact): the model gate cannot judge it")
    want = exact(job)
    if fault and fault["kind"] == "shared":  # a bug the RTL and its model share: only the float bound can see it
        want = want.copy()
        want.flat[fault["index"]] ^= 1 << fault["bit"]
    got = word_bits(y, fmt)
    bad = np.flatnonzero(got != want)
    rec = {"differ": int(bad.size), "words": int(got.size)}
    if bad.size:
        i, w = int(bad[0]), (1 + sum(FORMATS[fmt])) // 4
        rec.update(index=i, row=i // got.shape[1], col=i % got.shape[1], expected=f"{int(want.flat[i]):0{w}x}", got=f"{int(got.flat[i]):0{w}x}")
    return y, rec


def judge(name, desc, fmt, stats, r):
    """The gate on one inference: hardware (each layer bit-exact, and within LAYER_BOUND of float64) and task (top-1, or ad01's score) against the float model.
    A REPORTED (model, format) moves only its float checks (bound, task) to reported; a bit-exact failure always gates."""
    rep = (name, fmt) in REPORTED
    why = " (reported, not gated: a known limit in model_runner.REPORTED)"
    hw, bad, task, reported = [], set(), [], []
    for s in stats:
        L, e = f"L{s['layer']:02d}", s["exact"]
        if e and e["differ"]:
            hw.append(f"FAIL hardware {name} [{desc}] {L}: {e['differ']} of {e['words']} words differ from the bit-exact model; "
                      f"first at index {e['index']} (row {e['row']}, col {e['col']}): expected {e['expected']} got {e['got']}")
            bad.add(L)
        if not s["err"] <= LAYER_BOUND[fmt]:  # NaN fails too
            m = f"FAIL hardware {name} [{desc}] {L}: max|hw-ref|/max|ref| {s['err']:.2e} above the {fmt} bound {LAYER_BOUND[fmt]:g}"
            if rep:
                reported.append(m + why)
            else:
                hw.append(m)
                bad.add(L)
    if name in CLASSIFIERS:
        line = f"top-1 hw {r['hw_top']} ref {r['ref_top']}"
        if r["hw_top"] != r["ref_top"]:
            m = f"FAIL task {name} [{desc}]: top-1 {r['hw_top']} differs from the float reference's {r['ref_top']}"
            (reported if rep else task).append(m + why if rep else m)
    else:  # ad01: the mean reconstruction error over the inference's slices
        sh, sr, bound = float(np.mean(r["score_hw"])), float(np.mean(r["score_ref"])), SCORE_BOUND[fmt]
        rel = abs(sh - sr) / sr
        line = f"anomaly score hw {sh:.6f} ref {sr:.6f}, |hw-ref|/ref {rel:.2e}, bound {bound:g}"
        if not rel <= bound:
            m = f"FAIL task {name} [{desc}]: anomaly score {sh:.6f} against the float reference's {sr:.6f}, |hw-ref|/ref {rel:.2e} above {bound:g}"
            (reported if rep else task).append(m + why if rep else m)
    return {"hw": hw, "bad_layers": sorted(bad), "task": task, "reported": reported, "task_line": line, "rep_key": (name, fmt) if rep else None}


def verdict_word(g, kind: str) -> str:
    """judge's word for one part ('hw' or 'task') of an inference: a reported failure reads FAIL (reported, not gated), never PASS."""
    if g[kind]:
        return "FAIL"
    tag = "FAIL hardware" if kind == "hw" else "FAIL task"
    return "FAIL (reported, not gated)" if any(m.startswith(tag) for m in g["reported"]) else "PASS"


def gate_verdict(tally, models, fault=None) -> tuple:
    """(ok, extra failures, notes) of a whole run: no hardware or task failure, every requested model ran, a test-only fault was applied."""
    extra = [f"FAIL coverage: {m} ran no inference" for m in models if not tally["runs"].get(m)]
    if not tally["layers"]:
        extra.append("FAIL coverage: no layer was checked")
    if fault and not fault.get("applied"):
        extra.append(f"FAIL fault: --fault {fault['spec']} was never applied (no such model, layer or index in its first inference, or a conv fused into its ADD)")
    notes = [f"NOTE: REPORTED entry {k} passed every check it covers; it can be dropped from REPORTED"
             for k in sorted(tally["rep_seen"]) if k not in tally["rep_failed"]]
    return not tally["hw"] and not tally["task"] and not extra, extra, notes


def execute(model, x, sim=None, log=None, fault=None):
    """Runs the graph. With sim, compute ops run on the RTL, each checked bit for bit (check_layer), and outputs feed the next layer; otherwise float64 reference."""
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
        _, ref_float = job_reference(job)
        fmt = getattr(sim, "fmt_name", "fp32")
        if fmt != "fp32":  # the hardware takes rounded operands; judge it on those, and report the format's own cost apart
            job = dict(job, terms=[(op_round(X, fmt), op_round(W, fmt)) for X, W in job["terms"]],
                       bias=None if job["bias"] is None else op_round(job["bias"], fmt))
        _, ref = job_reference(job)
        if sim is None:
            y = ref.astype(np.float32)
            n = cyc = 0
        else:
            y, n, cyc = sim.run_job(job, f"L{li:02d}")
        rec = None
        if sim is not None:
            y, rec = check_layer(sim, job, y, fmt, fault if fault and fault.get("layer") == f"L{li:02d}" else None)
        scale = float(np.max(np.abs(ref))) or 1.0
        err = float(np.max(np.abs(y.astype(np.float64) - ref))) / scale
        err_fmt = float(np.max(np.abs(y.astype(np.float64) - ref_float))) / (float(np.max(np.abs(ref_float))) or 1.0)
        stats.append({"layer": li, "kind": kind, "sets": n, "cycles": cyc, "macs": macs_of(job),
                      "passes": len(job["terms"]), "shape": list(job["shape"]), "err": err, "err_vs_float": err_fmt, "exact": rec})
        if log:
            ex = "" if rec is None else f"  bit-exact {rec['differ']}/{rec['words']} differ"
            log(f"    L{li:02d} {kind:<18} {str(job['shape']):<18} sets {n:6d}  cycles {cyc:8d}  MACs {stats[-1]['macs']:9d}  max err/max|ref| {err:.2e}{ex}")
        t[out] = y.reshape(job["shape"]).astype(np.float32)
    return t[model["output"]], stats


# ── 8. CLI: --action model (default), --action tflite [--pack] ───────────────

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
MODEL_DIR = os.path.join(ROOT, "testbenches", "tflite_int8")
ACTIONS = ("model", "tflite")


MODEL_INPUTS = {  # each model's input files beside its .tflite, and how to fetch one that is missing (models/NOTICE: sources and licenses)
    "resnet8": (("data/cifar-10-batches-py/test_batch", "perf_samples_idxs.npy"), "run models/fetch_cifar10.sh"),
    "ad01": (("normal_id_01_00000000_hist_librosa.bin",), "run models/fetch_inputs.sh"),
    "kws": (("marvin_617de221_0.wav",), "run models/fetch_inputs.sh"),
    "vww": ((), ""),
}


def missing_inputs(name, mdir):
    """The files model name needs under mdir that are not there (its .tflite first), and how to supply them."""
    files, how = MODEL_INPUTS[name]
    miss = [f for f in (MODELS[name],) + files if not os.path.isfile(os.path.join(mdir, f))]
    return miss, ("make regression MODEL_DIR=<directory with it>" if MODELS[name] in miss else how)


def model_main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="model", help="model: float models end to end (this CLI); tflite: the int8 TFLite runs")
    ap.add_argument("--models", default="resnet8,ad01,kws,vww")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--count", type=int, default=1, help="inferences per model on the RTL")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(FORMATS),
                    help="the layer engine's build format for inputs, weights and results (bf16 sums in fp32, each result rounded to bf16); --engine sets is fp32")
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--tile-size", type=int, default=4, help="mesh tile size T the RTL is built with")
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "models"))
    ap.add_argument("--ref-accuracy", type=int, default=0, help="CIFAR-10 test images for the float reference accuracy")
    ap.add_argument("--no-sim", action="store_true", help="float reference only")
    ap.add_argument("--emulate", action="store_true", help="numpy stand-in for the RTL, to check the lowering")
    ap.add_argument("--engine", choices=("layer", "sets"), default="layer",
                    help="layer: sienna_top's layer contract (sienna_sched) schedules everything; sets: the host drives sienna_top set by set")
    ap.add_argument("--host-gaps", action="store_true", help="host idles a cycle after each load and waits for the credit")
    ap.add_argument("--fault", default=None, help="TEST ONLY, to prove the gate fails: word:MODEL:LAYER:INDEX[:BIT] (flip a hardware output bit), "
                    "shared:... (flip it in the exact model too), top1:MODEL (swap the top two outputs); first inference of MODEL")
    ap.add_argument("--collapse-k", type=int, choices=(0, 1), default=1, help="the mesh's COLLAPSE_K the layer engine builds and the golden follows")
    a = ap.parse_args(argv)
    if a.engine == "sets" and a.collapse_k == 0:
        ap.error("--engine sets builds collapse-k 1 (TB_sienna (sets)); use --engine layer")
    global COLLAPSE_K
    COLLAPSE_K = a.collapse_k
    if a.fmt_name == "int8":
        ap.error("the int8 MLPerf models are sub-project 2b; single-layer TFLite int8 models run with --action tflite")
    if a.fmt_name != "fp32" and (a.emulate or a.engine == "sets"):
        ap.error(f"--format {a.fmt_name}: --engine sets and --emulate build and judge fp32 only; use --engine layer")
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
            sim = Emulator(a.n, a.lanes, a.work, a.host_gaps)
        elif a.engine == "layer":
            sim = RtlLayer(a.n, a.lanes, a.work, a.fmt_name, a.tile_size)
        else:
            sim = RtlSets(a.n, a.lanes, a.work, a.host_gaps, a.tile_size)
        t0 = time.time()
        sim.build()
        log(f"built {os.path.basename(sim.bin)[:-4] if hasattr(sim, 'bin') else 'emulator'} N={a.n} lanes={a.lanes} for collapse-k {COLLAPSE_K} (each layer checks the mesh's) in {time.time() - t0:.0f} s")
    fault = parse_fault(a.fault)
    if fault:
        log(f"TEST ONLY: --fault {a.fault} (a run without it is unchanged; a fault that is never applied fails the gate)")
    sim_fmt = getattr(sim, "fmt_name", "fp32")
    engine = "emulator" if a.emulate else f"engine {a.engine}"
    how = ("no bit-exact model: emulator" if isinstance(sim, Emulator) else "bit-exact") + f", {sim_fmt} bound {LAYER_BOUND[sim_fmt]:g}"
    tally = {"layers": 0, "bad_layers": 0, "inferences": 0, "bad_inferences": 0, "hw": [], "task": [], "reported": [],
             "runs": {}, "rep_seen": set(), "rep_failed": set()}
    results = {}
    models, skipped = a.models.split(","), []
    for name in list(models):  # a model whose input files are absent is skipped by name, not failed
        miss, supply = missing_inputs(name, a.model_dir)
        if miss:
            log(f"skipped: model {name}: {', '.join(miss)} not in {a.model_dir} ({supply})")
            models.remove(name)
            skipped.append(name)
    if not models:
        log(f"MODEL GATE: SKIPPED (no model has its files in {a.model_dir})")
        return
    for name in models:
        model = load_tflite(os.path.join(a.model_dir, MODELS[name]))
        if name == "resnet8" and a.ref_accuracy:
            x, y = cifar_test(os.path.join(a.model_dir, "data"))
            hits = sum(int(np.argmax(execute(model, x[i : i + 1])[0]) == y[i]) for i in range(a.ref_accuracy))
            log(f"{name}: float64 reference top-1 on the first {a.ref_accuracy} CIFAR-10 test images: {hits}/{a.ref_accuracy} = {100 * hits / a.ref_accuracy:.1f}%")
            results.setdefault(name, {})["ref_top1"] = [hits, a.ref_accuracy]
        inputs, source = model_inputs(name, a.model_dir, a.count, 7)
        log(f"\n== {name} ({MODELS[name]}), inputs: {source}")
        runs = []
        for ii, (x, label, desc) in enumerate(inputs):
            ref, _ = execute(model, x)
            if sim is None:
                runs.append({"input": desc, "label": label, "ref_top": int(np.argmax(ref))})
                continue
            c0, s0, w0, t0 = sim.cycles, sim.sets, getattr(sim, "words", 0), time.time()
            log(f"  inference on {desc}")
            f_here = fault if fault and fault["model"] == name and ii == 0 else None  # a test-only fault hits the model's first inference
            hw, stats = execute(model, x, sim, log, f_here)
            if f_here and f_here["kind"] == "top1":  # TEST ONLY: the hardware's top two outputs swapped
                hw = np.array(hw, np.float32)
                o = np.argsort(hw.ravel())[::-1][:2]
                hw.ravel()[o] = hw.ravel()[o[::-1]]
                f_here["applied"] = True
                log(f"  TEST ONLY fault: top1 swaps hw outputs {o[0]} and {o[1]}")
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
            r["gate"] = judge(name, desc, sim_fmt, stats, r)
            runs.append(r)
            util = macs / (sets * a.n ** 3) if sets else 0
            log(f"  result: hw top {r['hw_top']} ref top {r['ref_top']} label {label}  max |hw-ref| on outputs {diff:.2e}  "
                f"{cyc} cycles = {cyc / 950e3:.3f} ms @950 MHz (assumed)  {sets} sets  {hwords} host words  {macs} MACs  MAC-slot use {100 * util:.1f}%  wall {r['wall_s']:.0f} s")
            g = r["gate"]
            log(f"  gate: hardware {verdict_word(g, 'hw')} ({len(stats)} layers, {how})  task {verdict_word(g, 'task')} ({g['task_line']})")
            for m in g["hw"] + g["task"] + g["reported"]:
                log(f"    {m}")
            tally["layers"] += len(stats)
            tally["bad_layers"] += len(g["bad_layers"])
            tally["inferences"] += 1
            tally["bad_inferences"] += int(bool(g["task"]))
            tally["runs"][name] = tally["runs"].get(name, 0) + 1
            if g["rep_key"]:
                tally["rep_seen"].add(g["rep_key"])
                if g["reported"]:
                    tally["rep_failed"].add(g["rep_key"])
            for k in ("hw", "task", "reported"):
                tally[k] += g[k]
        results.setdefault(name, {})["runs"] = runs
        results[name]["source"] = source
        json.dump(results, open(js, "w"), indent=1, default=list)
    log(f"\nreport {report}\nresults {js}")
    if sim is None:
        log("MODEL GATE: not run (--no-sim: float reference only)")
        return
    ok, extra, notes = gate_verdict(tally, models, fault)
    log("")
    for m in tally["hw"] + tally["task"] + extra + tally["reported"] + notes:
        log(m)
    rep_names = ", ".join(f"{m} {f}" for m, f in sorted(tally["rep_failed"]))
    log(f"MODEL GATE: {'PASS' if ok else 'FAIL'}  {sim_fmt} N={a.n} T={a.tile_size} lanes={a.lanes} {engine}: {len(tally['runs'])} of {len(models)} models ran, "
        f"{tally['inferences']} inferences; hardware: {tally['bad_layers']} of {tally['layers']} layers fail ({how}); "
        f"task: {tally['bad_inferences']} of {tally['inferences']} inferences fail; other failures: {len(extra)}; "
        f"reported, not gated: {len(tally['reported'])}{' (' + rep_names + ')' if rep_names else ''}"
        + (f"; skipped (inputs absent): {', '.join(skipped)}" if skipped else ""))
    if not ok:
        sys.exit(1)


def tflite_main(argv=None):
    """Runs testbenches/tflite_int8/'s single-layer int8 models (TFLite importer) through sienna_top's layer contract in int8, each output gated on the true
    function of its exact int32 sum and the layer's real scales; agreement with the interpreter and the bit-exact model is reported, not gated."""
    ap = argparse.ArgumentParser(description=tflite_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="tflite")
    ap.add_argument("--pack", action="store_true", help="run the packed TFLite layers instead (python model_runner.py --action tflite --pack --help)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--models", default=MODEL_DIR, help="<name>.tflite files, each with <name>.npz holding x_test and y_test")
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "int8"))
    ap.add_argument("--collapse-k", type=int, choices=(0, 1), default=1, help="the mesh's COLLAPSE_K the layer engine builds")
    a = ap.parse_args(argv)
    global COLLAPSE_K
    COLLAPSE_K = a.collapse_k
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
    sim = RtlLayer(a.n, a.lanes, a.work, "int8", a.tile_size)
    t0 = time.time()
    sim.build()
    log(f"TB_sienna (layer) built in int8, N={a.n} T={a.tile_size} lanes={a.lanes} for collapse-k {COLLAPSE_K} (each layer checks the mesh's), in {time.time() - t0:.0f} s")
    bad, covered, hw_bad, n_interp, n_exact, n_words = 0, False, 0, 0, 0, 0
    for path in paths:
        name = os.path.basename(path)[:-len(".tflite")]
        ref = np.load(path[:-len(".tflite")] + ".npz")
        x, y = ref["x_test"], ref["y_test"].astype(np.int64)
        layer = load_layer(path)
        job, shape = job_of(layer, x, ref, name)
        got, sets, cyc = sim.run_job(job, name)
        r, low, acc = layer_truth(job, got)
        got, low, want = got.reshape(shape), low.reshape(shape), y.reshape(shape)
        per_channel = layer["filter"]["scale"].size > 1
        z_in = int(layer["input"]["zp"][0])
        padded = layer["kind"] == "CONV_2D" and any(record_of_tflite(layer, name).pads)  # a SAME conv with all-zero pads is unpadded
        covered |= padded and per_channel and z_in != 0
        m_rtl, m_low, m_hw = int(np.sum(got != want)), int(np.sum(low != want)), int(np.sum(got != low))
        bad += int(not r.ok)
        hw_bad += int(not r.ok)
        n_interp, n_exact, n_words = n_interp + m_rtl, n_exact + m_hw, n_words + want.size
        log(f"MODEL {name}: {layer['kind']} out {shape} z_in {z_in} {'per-channel' if per_channel else 'per-tensor'} "
            f"{'SAME-padded' if padded else 'unpadded'} clamp {layer['act_range']}: {truth.stage_line(r)}; reported: RTL {m_rtl}/{want.size} "
            f"differ from the interpreter, {m_hw} from the bit-exact model, the bit-exact model {m_low} from the interpreter; {sets} sets, {cyc} cycles")
        for f in r.first:
            log(f"FAIL {name}: output {f[0]} (int32 sum {f[1]}) is {f[2]}, the true value {f[3]:.4f}")
    if not covered:
        log("no SAME-padded per-channel conv with a non-zero input zero point among the models")
        bad += 1
    log(f"TFLITE_INT8: {len(paths)} models, {bad} failing")
    log(f"MODEL GATE: {'PASS' if bad == 0 else 'FAIL'}  int8 N={a.n} T={a.tile_size} lanes={a.lanes} TFLite layers: {len(paths)} models; "
        f"hardware: {hw_bad} of {len(paths)} fail the true function; reported, not gated: {n_interp} of {n_words} codes differ from the "
        f"interpreter, {n_exact} from the bit-exact model"
        f"{'; coverage missing (no SAME-padded per-channel conv with a non-zero zero point)' if not covered else ''}")
    log("RESULT: PASSED" if bad == 0 else "RESULT: FAILED")
    sys.exit(1 if bad else 0)


GROUPS = [["fc8x8_linear", "fc8x4_relu", "fc6x8_relu6"], ["conv3x3_6x6x1x8_linear", "fc16x16_relu"]]


def model_of(path):
    """(pack_jobs model, output shape, interpreter's outputs) of one saved .tflite with its npz."""
    ref = np.load(path[:-len(".tflite")] + ".npz")
    layer = load_layer(path)
    job, shape = job_of(layer, ref["x_test"], ref, os.path.basename(path)[:-len(".tflite")])
    (X, W), = job["terms"]
    return {"W": W, "bias": job["bias"], "act": "linear", "req": job["req"], "inputs": [X]}, shape, ref["y_test"].astype(np.int64)


def layer_truth(job, got):
    """(truth.gate_requantize Result, bit-exact model codes, int32 sums) of a linear int8 job's outputs got: each against the true
    requantize of its exact int32 sum by its channel's real scale (job['req']['ratio']), zero point and clamp."""
    (X, W), req = job["terms"][0], job["req"]
    acc = wrap32(imatmul(np.asarray(X, np.int64), np.asarray(W, np.int64)) + np.asarray(job["bias"], np.int64)[None, :])
    S = np.broadcast_to(np.asarray(req["ratio"], np.float64)[None, :acc.shape[1]], acc.shape)
    r = truth.gate_requantize(acc, S, req["zp"], req["amin"], req["amax"], np.asarray(got, np.int64).reshape(acc.shape))
    return r, requantize(acc, req), acc


def tflite_pack_main(argv=None):
    """Groups of TFLite layers share one packed layer on the layer contract; every model's outputs are gated on the true function, as each alone;
    agreement with the interpreter is reported."""
    ap = argparse.ArgumentParser(description=tflite_pack_main.__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--models", default=os.path.join(ROOT, "testbenches", "tflite_int8_pack"))
    a = ap.parse_args(argv)
    work = os.path.join(ROOT, "testbenches", "results", "int8")
    os.makedirs(work, exist_ok=True)
    rep = open(os.path.join(work, f"tflite_pack_N{a.n}_T{a.tile_size}.log"), "w")
    sim = RtlLayer(a.n, a.lanes, work, "int8", a.tile_size)
    sim.build()
    bad = 0

    def fits(names):  # the group, or its halves until each part packs at this N (three blocks of 8 need N = 32)
        try:
            pack_jobs([model_of(os.path.join(a.models, f"{n}.tflite"))[0] for n in names], a.n, int8=True)
            return [names]
        except ValueError as e:
            if len(names) == 1:  # a model wider than N / 2 cannot pack at this N: reported, not run
                line = f"SKIP {names[0]}: {e}"
                print(line, flush=True)
                rep.write(line + "\n")
                return []
            return fits(names[:len(names) // 2]) + fits(names[len(names) // 2:])

    parts = [part for grp in GROUPS for part in fits(grp)]
    if not parts:
        rep.write(f"TFLITE_PACK: no model packs at N = {a.n}\nRESULT: FAILED\n")
        sys.exit(f"TFLITE_PACK: no model packs at N = {a.n}")
    for g, names in enumerate(parts):
        got = [model_of(os.path.join(a.models, f"{n}.tflite")) for n in names]
        job, recipe = pack_jobs([m for m, _, _ in got], a.n, int8=True)
        Y, sets, cyc = sim.run_job(job, f"tflp{g}")
        for (m, shape, y), outs, n in zip(got, unpack(Y, recipe), names):
            o = np.vstack(outs)
            r = layer_truth({"terms": [(m["inputs"][0], m["W"])], "bias": m["bias"], "req": m["req"]}, o)[0]
            mism = int(np.sum(o.reshape(shape) != y.reshape(shape)))
            bad += int(not r.ok)
            line = (f"PACKED {n} (group {g}, b = {a.n >> job['pack']['shift']}): {truth.stage_line(r)}; reported: {mism}/{y.size} differ "
                    f"from the interpreter; {sets} sets, {cyc} cycles")
            print(line, flush=True)
            rep.write(line + "\n")
    tail = f"TFLITE_PACK: {sum(len(x) for x in parts)} of {sum(len(x) for x in GROUPS)} models in {len(parts)} packed layers at N = {a.n}, {bad} failing"
    print(tail)
    rep.write(tail + "\nRESULT: " + ("PASSED" if bad == 0 else "FAILED") + "\n")
    sys.exit(1 if bad else 0)


def main():
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--action", choices=ACTIONS, default="model")
    pre.add_argument("--pack", action="store_true")
    a, rest = pre.parse_known_args()
    if a.action == "model":
        if a.pack:
            pre.error("--pack belongs to --action tflite")
        model_main(rest)
    else:
        (tflite_pack_main if a.pack else tflite_main)(rest)


if __name__ == "__main__":
    main()
