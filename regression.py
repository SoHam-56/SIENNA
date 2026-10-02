#!/usr/bin/env python3
"""
regression.py — SIENNA Full Pipeline Unified Regression Suite
=============================================================
Contains:
  1. Golden Model & Vector Generator
  2. Live Status Streamer
  3. Formatted Matrix Trace Dumper
  4. Regression Orchestrator & Scoreboard
"""

import argparse
import math
import os
import re
import struct
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPNAE"))
import mesh_model  # noqa: E402
import gpnae_model  # noqa: E402
from mesh_model import fpu  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh", "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402
import tflite_ref  # noqa: E402

COLLAPSE_K = 1  # the mesh's COLLAPSE_K the build uses; --collapse-k sets it for the bit-exact golden

# Ensure we can import the mesh helpers
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh"))

from conv_tests import _basic_pair, _general_pair, _im2col_patches, _kernel_size
from matmul_tests import _f2h as float_to_hex
from matmul_tests import _ref_matmul, write_mem

RESULTS_DIR = os.path.join(ROOT, "testbenches", "results", "pipeline")
TB_DIR = os.path.join(ROOT, "testbenches")

# ── ANSI Colors ──────────────────────────────────────────────────────────────
_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_B = "\033[94m"
_X = "\033[0m"
_O = "\033[1m"
_D = "\033[90m"

ok = lambda s: f"{_G}{_O}{s}{_X}"
err = lambda s: f"{_R}{_O}{s}{_X}"
hdr = lambda s: f"{_O}{_B}{s}{_X}"

# =============================================================================
# PART 1: GOLDEN MODEL & VECTOR GENERATOR
# =============================================================================


# Activation control words: 001/010/011 are the GPNAE polynomial modes, 100/101 bypass the polynomial.
# No entry for 0 on purpose: a code the RTL does not implement must not be reachable from a test.
SETS_IN_FLIGHT = 2 + 2 + 4 + 4 + 2 + 1  # sienna_top's default credits (its banks, ACC_BANKS=RESULT_BANKS=4); the testbenches read it from the package

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


def write_op_mem(path: str, data, fmt: str) -> None:
    """Operands as hex words of the format's width; fp32 keeps the 8-digit words of write_mem."""
    if fmt == "fp32":
        write_mem(path, data)
        return
    with open(path, "w") as fh:
        fh.write("".join(w + "\n" for w in op_hex(data, fmt)))

# Polynomial terms per activation, as passed to the TYTAN controller.
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


def apply_maxpool_2d(
    mat: np.ndarray, pool_h=2, pool_w=2, stride_h=None, stride_w=None, padding=0
) -> np.ndarray:
    if stride_h is None:
        stride_h = pool_h
    if stride_w is None:
        stride_w = pool_w
    if padding:
        mat = np.pad(mat, padding, mode="constant", constant_values=-1e4)
    oh = (mat.shape[0] - pool_h) // stride_h + 1
    ow = (mat.shape[1] - pool_w) // stride_w + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    return np.array(
        [
            [
                mat[
                    i * stride_h : i * stride_h + pool_h,
                    j * stride_w : j * stride_w + pool_w,
                ].max()
                for j in range(ow)
            ]
            for i in range(oh)
        ],
        dtype=np.float32,
    )


def _lane_seed(seed: int, lane: int) -> int:
    # Mirrors sienna_top: x = (seed ^ 0x9E3779B9 * (lane + 1)) * 0x85EBCA6B; x ^ (x >> 16), zero -> ones.
    x = (seed ^ ((0x9E3779B9 * (lane + 1)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 16
    return x or 0xFFFFFFFF


def set_dropout_seed(base: int, k: int) -> int:
    # Set k's dropout seed, mirrored by set_seed() in TB_sienna_top.
    return (base ^ ((0x85EBCA6B * k) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _lfsr_next(s: int) -> int:
    # Mirrors dropout.sv: 32 steps of {s[30:0], s[31] ^ s[21] ^ s[1] ^ s[0]} per beat.
    for _ in range(32):
        bit = ((s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s) & 1
        s = ((s << 1) & 0xFFFFFFFF) | bit
    return s


def _check_dropout_generator() -> None:
    # The hardware generator must look like independent Bernoulli draws at any drop rate.
    for p_percent in (25, 50, 75):
        thr = ((2**32 - 1) * p_percent) // 100
        s, keeps = 0x2ACE002A, []
        for _ in range(20000):
            s = _lfsr_next(s)
            keeps.append(s >= thr)
        rate = sum(keeps) / len(keeps)
        both = sum(a and b for a, b in zip(keeps, keeps[1:])) / (len(keeps) - 1)
        if abs(rate - (1 - p_percent / 100)) > 0.02 or abs(both - rate * rate) > 0.02:
            raise RuntimeError(f"dropout generator at p={p_percent}%: keep rate {rate:.3f}, "
                               f"neighbours both kept {both:.3f}, independent would be {rate*rate:.3f}")
    # 200 sets must give 200 masks, and two lanes in one beat must look independent.
    ones = np.ones((9, 9), dtype=np.float32)
    ms = [tuple((apply_dropout(ones, 0.5, True, set_dropout_seed(0x2ACE002A, k)) != 0).flatten())
          for k in range(200)]
    rate = sum(sum(m) for m in ms) / (200 * 81)
    joint = sum(m[0] and m[1] for m in ms) / 200
    if len(set(ms)) != 200 or abs(joint - rate * rate) > 0.08:
        raise RuntimeError(f"dropout masks: {len(set(ms))} distinct of 200, lanes 0 and 1 both kept "
                           f"{joint:.3f}, independent would be {rate*rate:.3f}")


def dropout_keep(n: int, p: float, seed: int, num_lanes: int) -> np.ndarray:
    """dropout.sv's keep decisions for n outputs, window w on lane w % num_lanes, decided on the word the beat advances to."""
    thr = ((2**32 - 1) * int(round(p * 100))) // 100
    states = [_lane_seed(seed, lane) for lane in range(num_lanes)]
    keep = np.empty(n, dtype=bool)
    for w in range(n):
        lane = w % num_lanes
        states[lane] = _lfsr_next(states[lane])
        keep[w] = states[lane] >= thr
    return keep


def apply_dropout(x: np.ndarray, p=0.5, training=False, seed=1, num_lanes=16) -> np.ndarray:
    """Inference copies; training replays dropout.sv's per-lane LFSR, window w on lane w % num_lanes."""
    if not training:
        return x.copy()
    p_percent = int(round(p * 100))
    scale = np.float32(100.0 / (100 - p_percent))
    flat = x.astype(np.float32).flatten()
    keep = dropout_keep(flat.size, p, seed, num_lanes)
    out = np.where(keep, flat * scale, flat * np.float32(0.0)).astype(np.float32)  # a dropped negative stays -0.0
    return out.reshape(x.shape)


def fmt_bits(x, fmt: str) -> np.ndarray:
    """Values already rounded to the format (op_round), as its bit patterns."""
    x = np.asarray(x, dtype=np.float32)
    return np.array([int(h, 16) for h in op_hex(x, fmt)], dtype=np.int64).reshape(x.shape)


def bits_float(b, fmt: str) -> np.ndarray:
    f = fpu.FORMATS[fmt]
    return (np.asarray(b, dtype=np.int64) << (23 - f.m)).astype(np.uint32).view(np.float32)


def write_bits(path: str, bits, fmt: str) -> None:
    d = (fpu.FORMATS[fmt].w + 3) // 4
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v):0{d}x}\n" for v in np.asarray(bits).flatten()))


def _is_greater_bits(a: int, b: int, f) -> bool:
    """Maxpool_2D's float compare on bit patterns: +0 and -0 tie."""
    S = 1 << (f.w - 1)
    am, bm = a & (S - 1), b & (S - 1)
    if am == 0 and bm == 0:
        return False
    if (a & S) != (b & S):
        return not (a & S)
    return am > bm if not (a & S) else am < bm


def _maxpool_bits(x, ph: int, pw: int, pad: int, f) -> np.ndarray:
    """sienna_top's dispatcher order: window rows outer, columns inner, -infinity outside, a running max from -infinity."""
    H, W = x.shape
    ninf = (1 << (f.w - 1)) | (f.emax << f.m)
    oh, ow = (H + 2 * pad - ph) // ph + 1, (W + 2 * pad - pw) // pw + 1
    out = np.empty((oh, ow), dtype=np.int64)
    for i in range(oh):
        for j in range(ow):
            run = ninf
            for pr in range(ph):
                for pc in range(pw):
                    r, c = i * ph + pr - pad, j * pw + pc - pad
                    v = int(x[r, c]) if 0 <= r < H and 0 <= c < W else ninf
                    if _is_greater_bits(v, run, f):
                        run = v
            out[i, j] = run
    return out


def _pad_square(M, N: int) -> np.ndarray:
    """A set shorter than N x N fills the staging bank row-major; the rest of the bank reads zero."""
    M = np.asarray(M, dtype=np.int64)
    return np.concatenate([M.flatten(), np.zeros(N * N - M.size, dtype=np.int64)]).reshape(N, N)


def _golden_bits(passes, bias, cfg: dict, act: str, drop_seed: int, fmt: str) -> tuple:
    """Bit-exact mesh, activation, pooling and dropout for one set in a narrow format; passes are (A bits, B bits) in order."""
    f = fpu.FORMATS[fmt]
    N = cfg.get("n", 16)
    C = mesh_model.matmul(f, [(_pad_square(a, N), _pad_square(b, N)) for a, b in passes], N, cfg.get("tile_size", 4),
                          cfg.get("collapse_k", COLLAPSE_K), bias)
    rom = gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f)))
    A = gpnae_model.Lane(f, rom).run(C, activation_to_code(act))
    P = _maxpool_bits(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1), f)
    if not cfg.get("training", False):
        return C, A, P, P.copy()
    flat = P.flatten()
    keep = dropout_keep(flat.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    scale = fpu.from_fp32(int(np.float32(1.0 / (1.0 - cfg.get("dropout_p", 0.5))).view(np.uint32)), f.m)
    prod = fpu.mul(f, flat, np.full_like(flat, scale))[0]
    F = np.where(keep, prod, flat & (1 << (f.w - 1)))  # a dropped beat is a zero with the input's sign
    return C, A, P, F.reshape(P.shape)


# ===== int8 (D-6): TFLite-style quantization of the float tests' data, and the bit-exact golden =====

REQ_HEAD = 8  # layer-wide words heading requant_<k>.mem: zp, min, max, gp_mx, gp_shx, gp_mout, gp_shout, gp_zout


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
    """ipu.requant of int32 sums (rows x channels) with channel c's multiplier and shift on column c."""
    acc = np.asarray(acc, np.int64)
    m = np.broadcast_to(np.asarray(rq["mult"], np.int64)[None, :acc.shape[1]], acc.shape)
    s = np.broadcast_to(np.asarray(rq["shift"], np.int64)[None, :acc.shape[1]], acc.shape)
    return np.asarray(ipu.requant(acc, m, s, rq["zp"], rq["amin"], rq["amax"], tflite_ref.ROUNDING), np.int64)


def requant_params(acc, s_a: float, s_w, act: str, rng=None, zq=None) -> dict:
    """Requantize and GPNAE parameters for int32 sums acc (rows x channels) as TFLite PTQ picks them; with rng, random words per channel."""
    acc = np.asarray(acc, np.int64)
    s_w = np.asarray(s_w, np.float64)
    real = acc * (s_a * s_w)[None, :]
    if act == "relu":
        real = np.maximum(real, 0.0)
    _, s_out, z_out = quant_act(real)
    qm = [tflite_ref.quantize_multiplier(s_a * float(s) / s_out) for s in s_w]
    mult = np.array([m for m, _ in qm], np.int64)
    shift = np.array([e for _, e in qm], np.int64)
    if rng is not None:  # every normalized multiplier, and right shifts 0..12 (TFLite's left shift can overflow int32)
        mult = rng.randint(1 << 30, 1 << 31, s_w.size).astype(np.int64)
        shift = rng.randint(-12, 1, s_w.size).astype(np.int64)
    mx, shx = gpnae_model.rescale_params(s_out)
    rq = dict(mult=mult, shift=shift, zp=z_out, amin=max(-128, z_out) if act == "relu" else -128, amax=127,
              mx=mx, shx=shx, s_out=s_out)
    if zq is not None:  # zp_random: the given (zero point, min, max) in place of the calibrated ones
        rq.update(zp=zq[0], amin=zq[1], amax=zq[2])
    x = (requantize(acc, rq) - rq["zp"]) * s_out  # the lane's real inputs
    x_hi = gpnae_model.SELU_POS_SAT / 2048.0  # the lane's SELU: Q4.11 below 0, lambda * x exact up to here (int32 in 2^-22)
    _, s_selu, z_selu = quant_act(apply_activation(np.clip(x, -16.0, x_hi).astype(np.float32), "selu"))
    mout, shout = gpnae_model.quantize_multiplier(2.0 ** -22 / s_selu)  # the lane's SELU value, in units of 2^-22, to its int8 code (D-4)
    rq.update(mout=int(mout), shout=int(shout), zout=z_selu, s_selu=s_selu)
    return rq


def selu_saturates(mx: int, shx: int, z_in: int, q) -> np.ndarray:
    """True where lane input code q rescales (gp_mx, gp_shx, zero point z_in), unsaturated, to x >= 487.29, where lambda * x leaves int32 in 2^-22."""
    return gpnae_model.selu_pos_saturates(gpnae_model.rescale_wide(np.asarray(q, np.int64), int(z_in), int(mx), int(shx)))


def int8_lane():
    f = gpnae_model.FORMATS["int8"]
    return gpnae_model.Lane(f, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f))))


def activate_int8(R, act: str, rq: dict) -> np.ndarray:
    """The int8 lane stage: ReLU and linear pass the requantized value, the others run the fixed-point lane with the set's parameters."""
    code = activation_to_code(act)
    R = np.asarray(R, np.int64)
    if code in (4, 5):
        return R.copy()
    par = gpnae_model.Int8Params(mx=rq["mx"], shx=rq["shx"], zin=rq["zp"], mout=rq["mout"], shout=rq["shout"], zout=rq["zout"])
    return np.asarray(int8_lane().run(R, code, par), np.int64)


def drop_zp(act: str, rq: dict) -> int:
    """D-5: dropout's drop value, the output zero point of the set's activation (SELU gp_zout, sigmoid -128, ReLU / linear zp, else 0)."""
    return {1: rq["zout"], 2: -128, 4: rq["zp"], 5: rq["zp"]}.get(activation_to_code(act), 0)


def _maxpool_int(x, ph: int, pw: int, pad: int) -> np.ndarray:
    """Integer max pooling with -128 outside the input, stride = window, as sienna_top dispatches it."""
    x = np.asarray(x, np.int64)
    H, W = x.shape
    xp = np.full((H + 2 * pad, W + 2 * pad), -128, dtype=np.int64)
    xp[pad:pad + H, pad:pad + W] = x
    oh, ow = (H + 2 * pad - ph) // ph + 1, (W + 2 * pad - pw) // pw + 1
    return np.array([[xp[i * ph:i * ph + ph, j * pw:j * pw + pw].max() for j in range(ow)] for i in range(oh)], np.int64)


def _golden_int8(passes, hw_bias, rq: dict, cfg: dict, act: str, drop_seed: int) -> tuple:
    """Bit-exact int8 set (D-6): int32 sums, per-column requantize, the lane, integer max pooling, dropout (D-5); passes are (A, B) codes."""
    N = cfg.get("n", 16)
    for a, b in passes:  # matmul_int truncates non-integer operands silently, so only int64 int8 codes may reach it
        assert all(np.asarray(m).dtype == np.int64 and np.asarray(m).min() >= -128 and np.asarray(m).max() <= 127
                   for m in (a, b)), "int8 golden: operands must be int64 arrays of int8 values"
    C = mesh_model.matmul_int([(_pad_square(a, N), _pad_square(b, N)) for a, b in passes], N, hw_bias)
    R = requantize(C, rq)
    A = activate_int8(R, act, rq)
    P = _maxpool_int(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1))
    if not cfg.get("training", False):
        return C, R, A, P, P.copy()
    keep = dropout_keep(P.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    return C, R, A, P, np.where(keep, P.flatten(), drop_zp(act, rq)).reshape(P.shape)


def int8_layer_exact(A_q, B_q, hw_bias, rq: dict, act: str) -> np.ndarray:
    """sienna_layer's int8 output for one product: int32 sums, per-column requantize, the lane; the layer engine neither pools nor drops out."""
    assert all(np.asarray(m).dtype == np.int64 and np.asarray(m).min() >= -128 and np.asarray(m).max() <= 127
               for m in (A_q, B_q)), "int8_layer_exact: operands must be int64 arrays of int8 values"  # as _golden_int8
    acc = wrap32(imatmul(A_q, B_q) + np.asarray(hw_bias, np.int64)[None, :])
    return activate_int8(requantize(acc, rq), act, rq)


def _check_rounding() -> None:
    """sienna_fmt_pkg::REQ_ROUNDING must be the variant G0 pinned, as ipu and tflite_ref read it, or golden and RTL round apart."""
    pkg = os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "src", "sienna_fmt_pkg.sv")
    m = re.search(r'localparam string REQ_ROUNDING\s*=\s*"(\w+)"', open(pkg).read())
    rtl = m.group(1) if m else "(no REQ_ROUNDING)"
    if tflite_ref.ROUNDING is None or rtl != tflite_ref.ROUNDING or ipu.REQ_ROUNDING != tflite_ref.ROUNDING:
        raise ValueError(f"sienna_fmt_pkg rounds {rtl}, ipu.REQ_ROUNDING is {ipu.REQ_ROUNDING}, "
                         f"tflite_ref.ROUNDING (rounding.txt) is {tflite_ref.ROUNDING}")


def _decoy_params(N: int) -> dict:
    """A partial pass's requantize words: the activated pass's are the ones used, so these must never show in a result."""
    return dict(mult=np.full(N, 1 << 30, np.int64), shift=np.full(N, -1, np.int64), zp=5, amin=-100, amax=100,
                mx=1, shx=0, mout=1 << 30, shout=-1, zout=5)


def _write_s8(path: str, v) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"{int(x) & 0xFF:02x}\n" for x in np.asarray(v).flatten()))


def _write_w32(path: str, v) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"{int(x) & 0xFFFFFFFF:08x}\n" for x in np.asarray(v).flatten()))


def _requant_words(rq: dict) -> list:
    """requant_<k>.mem's words: the REQ_HEAD layer-wide words, then N multipliers, then N shifts."""
    return [rq["zp"], rq["amin"], rq["amax"], rq["mx"], rq["shx"], rq["mout"], rq["shout"], rq["zout"]] + \
        [int(v) for v in rq["mult"]] + [int(v) for v in rq["shift"]]


def _zp_draws(rng, groups: int, acts: list) -> list:
    """zp_random: per group a distinct non-zero zero point, the clamp minimum (the zero point after ReLU) and maximum around it."""
    zps = rng.choice([z for z in range(-100, 101) if z != 0], groups, replace=False)
    out = []
    for z, act in zip(zps, acts):
        z = int(z)
        out.append((z, z if act == "relu" else int(rng.randint(-128, z)), int(rng.randint(z + 1, 128))))
    return out


def _generate_vectors_int8(cfg: dict) -> None:
    """int8 stimulus and bit-exact golden: the float tests' matrices quantized as TFLite PTQ would, each set's parameters in requant_<k>.mem."""
    os.makedirs(TB_DIR, exist_ok=True)
    _check_rounding()
    N, mode = cfg.get("n", 16), cfg.get("mode", "matmul")
    act_type = cfg.get("activation", cfg.get("act", "idle"))
    seed = cfg.get("seed", 42) + int(os.environ.get("SIENNA_SEED", "0"))  # unset keeps the fixed stimulus
    test_name = cfg.get("name", "manual_gen")
    lo, hi = cfg.get("a_range", (-1.0, 1.0))  # a range not centred on 0 gives a non-zero input zero point
    scale = float(cfg.get("scale", 1.0))
    if mode == "conv":
        A0, B0 = build_conv_matrices(N, cfg.get("conv_type", "basic"), seed, cfg.get("conv_stride"))
    else:
        np.random.seed(seed)
        m_type = cfg.get("matrix_type", "random")
        if m_type == "identity":
            A0 = B0 = np.eye(N)
        elif m_type == "ones":
            A0 = B0 = np.ones((N, N))
        elif m_type == "small_exact":
            A0 = B0 = np.random.randint(-3, 4, (N, N))
        else:
            A0, B0 = np.random.uniform(lo, hi, (N, N)), np.random.uniform(-1.0, 1.0, (N, N))
    A0, B0 = np.array(A0, np.float64), np.array(B0, np.float64)  # copies: identity and ones share one array
    if cfg.get("zero_rows"):
        A0[0::4, :] = 0.0
        A0[1::4, :] = 0.0
    drop_seed = 0x2ACE0000 + seed
    passes = cfg.get("accum_passes", 1)
    mixed = cfg.get("mixed_acts", [])
    assert not mixed or mixed[0] == act_type, (test_name, "mixed_acts[0] must be the test's act")
    credits = cfg.get("credits", SETS_IN_FLIGHT)
    num_sets = cfg.get("num_sets", len(mixed) or -(-(credits + 2) // passes) * passes)
    use_bias = bool(cfg.get("bias", False))
    req_rng = np.random.RandomState(seed + 7000) if cfg.get("req_random") else None
    reals = [(A0 * scale, B0 * scale)]
    for k in range(1, num_sets):
        rng = np.random.RandomState(seed + 1000 + k)
        reals.append((rng.uniform(lo, hi, (N, N)) * scale, rng.uniform(-1.0, 1.0, (N, N)) * scale))
    first = None
    starts = list(range(0, num_sets, passes))
    acts_g = [mixed[(min(g0 + passes, num_sets) - 1) % len(mixed)] if mixed else act_type for g0 in starts]  # the activated pass's activation
    zqs = _zp_draws(np.random.RandomState(seed + 8000), len(starts), acts_g) if cfg.get("zp_random") else [None] * len(starts)
    sat = [0, 0]  # activated SELU sets: lane inputs where lambda * x saturates, and all their lane inputs
    for g, g0 in enumerate(starts):
        ks = list(range(g0, min(g0 + passes, num_sets)))
        act_g = acts_g[g]
        A_q, s_a, z_a = quant_act(np.hstack([reals[k][0] for k in ks]))
        B_q, s_w = quant_weights(np.vstack([reals[k][1] for k in ks]))
        bias_real = np.random.RandomState(seed + 5000 + g0).uniform(-1.0, 1.0, N) * scale if use_bias else None
        hw_bias = fold_bias(bias_real, s_a, s_w, z_a, B_q)
        parts = [(A_q[:, i * N:(i + 1) * N], B_q[i * N:(i + 1) * N, :]) for i in range(len(ks))]
        acc = wrap32(sum(imatmul(a, b) for a, b in parts) + hw_bias[None, :])
        rq = requant_params(acc, s_a, s_w, act_g, req_rng, zqs[g])
        complete = len(ks) == passes  # a trailing short group has only partial passes, as in the float tests
        if act_g == "selu" and complete:
            R = requantize(acc, rq)
            sat[0] += int(np.sum(selu_saturates(rq["mx"], rq["shx"], rq["zp"], R)))
            sat[1] += R.size
        for i, k in enumerate(ks):
            last = complete and i == len(ks) - 1
            rq_k = rq if last else _decoy_params(N)
            write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), parts[i][0], "int8")
            write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), parts[i][1], "int8")
            _write_w32(os.path.join(TB_DIR, f"bias_{k}.mem"), hw_bias if i == 0 else np.zeros(N, np.int64))
            _write_w32(os.path.join(TB_DIR, f"requant_{k}.mem"), _requant_words(rq_k))
            F = _golden_int8(parts, hw_bias, rq, cfg, act_g, set_dropout_seed(drop_seed, k))[4] if last \
                else np.zeros(0, np.int64)
            _write_s8(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F)  # empty for a partial set
            _write_s8(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(F))
            if k == 0:
                first = (parts[0][0], parts[0][1], hw_bias, rq_k)
    if sat[1]:
        print(f"      SELU saturation: {sat[0]} of {sat[1]} lane inputs at x >= {gpnae_model.SELU_POS_SAT / 2048:.2f}, where lambda * x "
              f"saturates at int32 (bit-exact against the saturating golden)")
    # The single-set pass starts set 0 alone, not partial, with set 0's bias and requantize words.
    a0, b0, hb0, rq0 = first
    C0, _, A0q, P0, F0 = _golden_int8([(a0, b0)], hb0, rq0, cfg, act_type, drop_seed)
    write_op_mem(os.path.join(TB_DIR, "matrix_west.mem"), a0, "int8")
    write_op_mem(os.path.join(TB_DIR, "matrix_north.mem"), b0, "int8")
    _write_s8(os.path.join(TB_DIR, "expected_output.mem"), F0)
    _write_s8(os.path.join(TB_DIR, "bound_output.mem"), np.zeros_like(F0))
    dump_golden_trace(test_name, C0.astype(np.float32), A0q.astype(np.float32), P0.astype(np.float32))
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"),
                     _config_items(cfg, "int8", act_type, num_sets, credits, passes, mixed, True, drop_seed))
    _check_mem_widths("int8", num_sets)


def _check_mem_widths(fmt: str, num_sets: int) -> None:
    """Every operand, bias and expected word the TB reads must be the format's width: a stale fp32 file would be truncated."""
    d = 2 if fmt == "int8" else (fpu.FORMATS[fmt].w + 3) // 4  # int8: operands and results 2 digits; int32 bias and requantize words 8
    per_set = ("matrix_west", "matrix_north", "bias", "expected_output", "bound_output") + (("requant",) if fmt == "int8" else ())
    names = [f"{b}.mem" for b in ("matrix_west", "matrix_north", "expected_output", "bound_output")]
    names += [f"{b}_{k}.mem" for k in range(num_sets) for b in per_set]
    for fn in names:
        want = 8 if fmt == "int8" and fn.startswith(("bias_", "requant_")) else d
        if os.path.exists(os.path.join(TB_DIR, fn)):
            for i, ln in enumerate(open(os.path.join(TB_DIR, fn))):
                if ln.strip() and len(ln.strip()) != want:
                    raise ValueError(f"{fn}:{i + 1}: word '{ln.strip()}' is not {want} hex digits ({fmt})")


def build_conv_matrices(N: int, conv_type: str, seed: int, stride=None) -> tuple:
    if conv_type != "basic":
        raise ValueError(f"Unknown conv_type '{conv_type}'")
    if math.isqrt(N) ** 2 == N:
        K = _kernel_size(N)
        return _basic_pair(img_size=K * K, K=K, seed=seed)
    return _general_pair(N, seed)[0]  # 3x3 kernel, depth zero-padded to N


def write_sv_package(path: str, items: list) -> None:
    with open(path, "w") as f:
        f.write("// Auto-Generated Configuration Package\npackage test_config_pkg;\n\n")
        for name, val, vtype in items:
            kw = "shortreal" if vtype == "float" else "int"
            f.write(f"  localparam {kw} {name} = {val};\n")
        f.write("\nendpackage\n")


def _config_items(cfg: dict, fmt: str, act_type: str, num_sets: int, credits: int, passes: int, mixed: list,
                  use_bias: bool, drop_seed: int) -> list:
    """test_config_pkg's items: geometry, the build's format, activation, pooling, dropout and the streamed sets."""
    N = cfg.get("n", 16)
    sram_depth = N * N
    return [
        ("N", N, "int"),
        ("TILE_SIZE", cfg.get("tile_size", 4), "int"),
        ("NUM_LANES", cfg.get("lanes", 32), "int"),
        ("HOST_WORDS", cfg.get("host_words", N), "int"),
        ("EXP_W", FORMATS[fmt][0], "int"),
        ("MAN_W", FORMATS[fmt][1], "int"),
        ("DATA_WIDTH", 1 + sum(FORMATS[fmt]), "int"),
        ("EXACT_GOLDEN", int(fmt != "fp32"), "int"),
        ("IS_INT", int(fmt == "int8"), "int"),
        ("ACC_W", 32 if fmt == "int8" else 1 + sum(FORMATS[fmt]), "int"),
        ("SRAM_DEPTH", sram_depth, "int"),
        ("FIFO_DEPTH", cfg.get("fifo_depth", sram_depth), "int"),
        ("ACTIVATION_CODE", activation_to_code(act_type), "int"),
        ("NUM_TERMS", get_polynomial_terms(act_type), "int"),
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
        ("TRAINING_MODE", int(bool(cfg.get("training", False))), "int"),
        ("DROPOUT_SEED", drop_seed, "int"),
        ("ADDR_LINES", max(1, math.ceil(math.log2(sram_depth))), "int"),
    ]


def float_to_hex_str(f: float) -> str:
    """Helper to cleanly convert python float to IEEE-754 hex string"""
    return struct.pack(">f", f).hex()


def dump_golden_trace(
    test_name: str, C: np.ndarray, C_act: np.ndarray, C_pooled: np.ndarray
):
    """Writes the expected intermediate values in the exact same format as the hardware trace."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{test_name}_expected_flow.txt")

    with open(out_path, "w") as f:
        f.write(
            f"{'='*100}\n SIENNA PIPELINE EXPECTED GOLDEN FLOW \n Test: {test_name}\n{'='*100}\n\n"
        )

        def write_matrix(name: str, mat: np.ndarray):
            flat = mat.flatten()
            total = len(flat)
            f.write(f"=== {name} ({total} elements) ===\n")
            if total == 0:
                f.write("  [ No data generated for this stage ]\n\n")
                return

            # Print row by row based on actual matrix dimensions
            for r in range(mat.shape[0]):
                row_data = mat[r, :]
                fmt_row = [
                    f"{val:>10.4f} (0x{float_to_hex_str(val)})" for val in row_data
                ]
                f.write("  [ " + "  ".join(fmt_row) + " ]\n")
            f.write("\n")

        write_matrix("Stage 1: Systolic Mesh Output (Input to GPNAE)", C)
        write_matrix("Stage 2: GPNAE Output (Input to Maxpool)", C_act)
        write_matrix("Stage 3: Maxpool Output (Input to Dropout)", C_pooled)


def _golden(A: np.ndarray, B: np.ndarray, cfg: dict, act_type: str, drop_seed: int = 1) -> tuple:
    return _golden_from_c(_ref_matmul(A, B), cfg, act_type, drop_seed)


def _golden_from_c(C: np.ndarray, cfg: dict, act_type: str, drop_seed: int = 1) -> tuple:
    """Activation, pooling and dropout of an already formed product (one matmul or a sum of partials)."""
    C_act = apply_activation(C, act_type)
    C_pooled = apply_maxpool_2d(
        C_act, cfg.get("pool_h", 2), cfg.get("pool_w", 2), padding=cfg.get("padding", 1)
    )
    C_final = apply_dropout(C_pooled, cfg.get("dropout_p", 0.5), cfg.get("training", False), drop_seed,
                            cfg.get("lanes", 32))
    return C, C_act, C_pooled, C_final


# Largest slope of each activation: an input error of e moves the output by at most slope * e.
ACTIVATION_SLOPE = {"selu": 1.76, "sigmoid": 0.25, "tanh": 1.0, "relu": 1.0, "linear": 1.0}


def fp32_error_bound(S: np.ndarray, products: int, cfg: dict, act_type: str, C: np.ndarray) -> np.ndarray:
    """Worst-case fp32 error of each output: (products + 8) * 2^-24 * sum|a*b|, carried through activation, pooling, dropout.

    S is the sum of |a_i * b_i| (and |bias|) behind each pre-activation value C. Near-zero outputs of a sum whose terms
    cancel can miss by this much with correct hardware, which a relative tolerance alone would call a failure."""
    B = (products + 8) * 2.0**-24 * S.astype(np.float64) * ACTIVATION_SLOPE.get(act_type.lower(), 1.0)
    if act_type.lower() == "relu":
        B = np.where(C.astype(np.float64) > -B, B, 0.0)  # clearly negative inputs must come out exactly 0
    B = apply_maxpool_2d(B, cfg.get("pool_h", 2), cfg.get("pool_w", 2), padding=cfg.get("padding", 1))
    if cfg.get("training", False):
        B = B / (1.0 - cfg.get("dropout_p", 0.5))
    return B.astype(np.float32)


def generate_vectors(cfg: dict) -> None:
    if cfg.get("fmt_name", "fp32") == "int8":
        return _generate_vectors_int8(cfg)
    os.makedirs(TB_DIR, exist_ok=True)
    N, tile_size, mode = (
        cfg.get("n", 16),
        cfg.get("tile_size", 4),
        cfg.get("mode", "matmul"),
    )
    act_type = cfg.get("activation", cfg.get("act", "idle"))
    seed = cfg.get("seed", 42) + int(os.environ.get("SIENNA_SEED", "0"))  # unset keeps the fixed stimulus
    test_name = cfg.get("name", "manual_gen")

    if mode == "conv":
        A, B = build_conv_matrices(
            N, cfg.get("conv_type", "basic"), seed, cfg.get("conv_stride")
        )
    else:
        np.random.seed(seed)
        m_type = cfg.get("matrix_type", "random")
        if m_type == "identity":
            A = B = np.eye(N, dtype=np.float32)
        elif m_type == "ones":
            A = B = np.ones((N, N), dtype=np.float32)
        elif m_type == "small_exact":
            A = B = np.random.randint(-3, 4, (N, N)).astype(np.float32)
        else:
            A = np.random.uniform(-1.0, 1.0, (N, N)).astype(np.float32)
            B = np.random.uniform(-1.0, 1.0, (N, N)).astype(np.float32)
    # "scale" widens the matmul outputs so every activation reaches its tails.
    if cfg.get("zero_rows"):  # rows of +0 and -0: every product a signed zero, and the PE's first add is +0 + p
        A[0::4, :] = np.float32(-0.0)
        A[1::4, :] = np.float32(0.0)
    scale = np.float32(cfg.get("scale", 1.0))
    fmt = cfg.get("fmt_name", "fp32")
    exact = fmt != "fp32"  # narrow formats: bit-exact golden, exact compare in the TB
    A, B = op_round(A * scale, fmt), op_round(B * scale, fmt)

    # Set k's dropout seed is set_dropout_seed(DROPOUT_SEED, k), as the TB drives it.
    drop_seed = 0x2ACE0000 + seed
    # "bias" gives the first pass of every group a random bias row, which the mesh adds to each column.
    use_bias = bool(cfg.get("bias", False))
    bias_raw = lambda k: (np.random.RandomState(seed + 5000 + k).uniform(-1.0, 1.0, N) * scale).astype(np.float32)
    bias_of = (lambda k: op_round(bias_raw(k), fmt)) if exact else bias_raw  # the bias in the format the hardware reads
    C0 = _ref_matmul(A, B) + (bias_of(0) if use_bias else np.float32(0.0))
    C, C_act, C_pooled, C_final = _golden_from_c(C0.astype(np.float32), cfg, act_type, drop_seed)

    # Write files for Verilator testbench
    write_op_mem(os.path.join(TB_DIR, "matrix_west.mem"), A, fmt)
    write_op_mem(os.path.join(TB_DIR, "matrix_north.mem"), B, fmt)
    if exact:
        F0 = _golden_bits([(fmt_bits(A, fmt), fmt_bits(B, fmt))], fmt_bits(bias_of(0), fmt) if use_bias else None,
                          cfg, act_type, drop_seed, fmt)[3]
        write_bits(os.path.join(TB_DIR, "expected_output.mem"), F0, fmt)
    else:
        write_mem(os.path.join(TB_DIR, "expected_output.mem"), C_final)
    S0 = np.abs(A).astype(np.float64) @ np.abs(B).astype(np.float64) + (np.abs(bias_of(0)) if use_bias else 0.0)
    if exact:
        write_bits(os.path.join(TB_DIR, "bound_output.mem"), np.zeros_like(F0), fmt)
    else:
        write_mem(os.path.join(TB_DIR, "bound_output.mem"), fp32_error_bound(S0, N, cfg, act_type, C0))

    # Streamed sets: set 0 is the test's own pattern, the rest random and distinct.
    # accum_passes P groups the streamed sets P at a time: P-1 partial products, then the set that is activated.
    passes = cfg.get("accum_passes", 1)
    mixed = cfg.get("mixed_acts", [])  # set k uses mixed[k % len], set 0 must match act
    assert not mixed or mixed[0] == act_type, (test_name, "mixed_acts[0] must be the test's act")
    # Enough sets to use every credit, so the credit-overrun pass is reachable; whole accumulate groups only.
    credits = cfg.get("credits", SETS_IN_FLIGHT)  # a test may build the pipeline with fewer credits than the default
    num_sets = cfg.get("num_sets", len(mixed) or -(-(credits + 2) // passes) * passes)
    masks = []
    run = None
    run_s = None  # sum|a*b| behind the running partial sum
    grp, gbias = [], None  # exact: the passes of the current group, in order, and its bias
    for k in range(num_sets):
        act_k = mixed[k % len(mixed)] if mixed else act_type
        if k == 0:
            Ak, Bk = A, B
        else:
            rng = np.random.RandomState(seed + 1000 + k)
            Ak = op_round(rng.uniform(-1.0, 1.0, (N, N)) * scale, fmt)
            Bk = op_round(rng.uniform(-1.0, 1.0, (N, N)) * scale, fmt)
        bk = bias_of(k) if use_bias and k % passes == 0 else np.zeros(N, dtype=np.float32)
        if exact:
            write_op_mem(os.path.join(TB_DIR, f"bias_{k}.mem"), bk, fmt)
            if k % passes == 0:
                grp, gbias = [], (fmt_bits(bk, fmt) if use_bias else None)
            grp.append((fmt_bits(Ak, fmt), fmt_bits(Bk, fmt)))
        else:
            write_mem(os.path.join(TB_DIR, f"bias_{k}.mem"), bk)
        Sk = np.abs(Ak).astype(np.float64) @ np.abs(Bk).astype(np.float64) + np.abs(bk)
        if passes > 1:
            Ck = (_ref_matmul(Ak, Bk) + bk).astype(np.float32)
            run = Ck if k % passes == 0 else (run + Ck).astype(np.float32)  # summed in pass order, as the hardware does
            run_s = Sk if k % passes == 0 else run_s + Sk
            partial = (k % passes) != passes - 1
            Fk = np.zeros(0, dtype=np.float32) if partial else _golden_from_c(run, cfg, act_k, set_dropout_seed(drop_seed, k))[3]
        else:
            partial = False
            Fk = C_final if k == 0 else _golden_from_c((_ref_matmul(Ak, Bk) + bk).astype(np.float32), cfg, act_k,
                                                       set_dropout_seed(drop_seed, k))[3]
        write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), Ak, fmt)
        write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), Bk, fmt)
        if exact:
            Fk = np.zeros(0, dtype=np.int64) if partial else \
                _golden_bits(grp, gbias, cfg, act_k, set_dropout_seed(drop_seed, k), fmt)[3]
            write_bits(os.path.join(TB_DIR, f"expected_output_{k}.mem"), Fk, fmt)  # empty for a partial set
            write_bits(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(Fk), fmt)
            Fk = bits_float(Fk, fmt)  # the dropout-mask check below reads values
        else:
            write_mem(os.path.join(TB_DIR, f"expected_output_{k}.mem"), Fk)  # empty for a partial set
            Bnd = np.zeros(0, dtype=np.float32) if partial else \
                fp32_error_bound(run_s if passes > 1 else Sk, N * passes, cfg, act_k,
                                 run if passes > 1 else (_ref_matmul(Ak, Bk) + bk).astype(np.float32))
            write_mem(os.path.join(TB_DIR, f"bound_output_{k}.mem"), Bnd)
        if cfg.get("training", False) and not partial:
            masks.append(tuple((Fk.flatten() != 0).tolist()))
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            agree = sum(a == b for a, b in zip(masks[i], masks[j])) / len(masks[i])
            if agree > 0.75:  # independent masks agree about half the time
                raise RuntimeError(f"{test_name}: sets {i} and {j} dropout masks agree on {agree:.0%}")

    # Dump the intermediate Golden Trace for debug comparisons
    dump_golden_trace(test_name, C, C_act, C_pooled)

    # Dump SV Config Package
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"),
                     _config_items(cfg, fmt, act_type, num_sets, credits, passes, mixed, use_bias, drop_seed))
    if exact:
        _check_mem_widths(fmt, num_sets)


# =============================================================================
# PART 2: TRACE ANALYZER & HW DUMPER
# =============================================================================


def _hex_to_float(hex_str: str) -> float:
    try:
        return struct.unpack(">f", bytes.fromhex(hex_str))[0]
    except ValueError:
        return float("nan")


def dump_hardware_trace(
    test_name: str,
    N: int,
    num_lanes: int = 8,
    pool_w=2,
    stride_w=2,
    padding=1,
    print_to_console=False,
):
    """Parses raw HW stream and writes a lane-aware flow log unconditionally."""
    in_path = os.path.join(TB_DIR, "hardware_trace.txt")
    out_path = os.path.join(RESULTS_DIR, f"{test_name}_data_flow.txt")
    if not os.path.exists(in_path):
        return

    stage1_flat = []  # Systolic -> GPNAE (single shared FIFO, no lane tag)
    stage2_lanes = {lane: [] for lane in range(num_lanes)}  # GPNAE -> Maxpool
    stage3_lanes = {lane: [] for lane in range(num_lanes)}  # Maxpool -> Dropout

    regex = re.compile(
        r"\[(\d+)\]\s+"
        r"(Systolic -> GPNAE|GPNAE -> Maxpool|Maxpool -> Dropout)"
        r"\s*(?:lane=(\d+))?\s*:\s*dec=[^\s]+\s+hex=([0-9a-fA-F]{8})"
    )

    with open(in_path, "r") as f:
        for line in f:
            match = regex.search(line)
            if not match:
                continue
            time_val, stage, lane_str, hex_val = match.groups()
            entry = {
                "time": int(time_val),
                "hex": hex_val,
                "float": _hex_to_float(hex_val),
            }
            if stage == "Systolic -> GPNAE":
                stage1_flat.append(entry)
            elif stage == "GPNAE -> Maxpool":
                stage2_lanes.setdefault(int(lane_str or 0), []).append(entry)
            elif stage == "Maxpool -> Dropout":
                stage3_lanes.setdefault(int(lane_str or 0), []).append(entry)

    with open(out_path, "w") as f:
        f.write(
            f"{'='*100}\n SIENNA PIPELINE HW CAPTURED FLOW \n Test: {test_name}\n{'='*100}\n\n"
        )

        def write_flat_stage(name: str, data: list, cols: int):
            total = len(data)
            f.write(f"=== {name} ({total} elements) ===\n")
            if total == 0:
                f.write("  [ No data emerged from this stage ]\n\n")
                return

            rows = math.ceil(total / cols)
            for r in range(rows):
                row_data = data[r * cols : (r + 1) * cols]
                fmt_row = [f"{it['float']:>10.4f} (0x{it['hex']})" for it in row_data]
                f.write("  [ " + "  ".join(fmt_row) + " ]\n")
            f.write("\n")

        def write_lane_stage(name: str, lane_data: dict, cols: int = 8):
            counts = {lane: len(v) for lane, v in lane_data.items()}
            total = sum(counts.values())
            f.write(f"=== {name} ({total} elements across {num_lanes} lanes) ===\n")
            if total == 0:
                f.write("  [ No data emerged from this stage ]\n\n")
                return

            distinct = set(counts.values())
            if len(distinct) > 1:
                f.write(f"  *** WARNING: lane element counts differ: {counts} ***\n\n")
            else:
                f.write(
                    f"  Lane element counts (uniform): {next(iter(distinct))} each\n\n"
                )

            for lane in sorted(lane_data.keys()):
                entries = lane_data[lane]
                f.write(f"  --- Lane {lane} ({len(entries)} elements) ---\n")
                if not entries:
                    f.write("    [ No data captured for this lane ]\n\n")
                    continue
                rows = math.ceil(len(entries) / cols)
                for r in range(rows):
                    row_data = entries[r * cols : (r + 1) * cols]
                    fmt_row = [
                        f"{it['float']:>10.4f} (0x{it['hex']}) @t={it['time']}"
                        for it in row_data
                    ]
                    f.write("    [ " + "  ".join(fmt_row) + " ]\n")
                f.write("\n")

        write_flat_stage(
            "Stage 1: Systolic Mesh Output (Input to GPNAE, shared FIFO1)",
            stage1_flat,
            cols=N,
        )
        write_lane_stage(
            "Stage 2: GPNAE Output (Input to Maxpool, per-lane FIFO2)",
            stage2_lanes,
        )
        write_lane_stage(
            "Stage 3: Maxpool Output (Input to Dropout, per-lane FIFO3)",
            stage3_lanes,
        )

    if print_to_console:
        print(f"\n{_R}>>> AUTO-DEBUG: PIPELINE CORRUPTION DETECTED <<<{_X}")
        print(f"{_D}Expected vs HW trace saved in {RESULTS_DIR}{_X}")
        with open(out_path, "r") as f:
            print(f.read())


# =============================================================================
# PART 3: REGRESSION ORCHESTRATOR
# =============================================================================

# Two tests here used "act": "idle", i.e. control word 0, which is not a mode the RTL implements -- they
# could only ever time out. Removed rather than adding an RTL bypass, which would be a design change.
# Their matrix types are still covered by mm_ones and mm_small_values in SystolicMesh/matmul_tests.py.
PIPELINE_TESTS = [
    {
        "name": "matmul_ident_selu",
        "mode": "matmul",
        "matrix_type": "identity",
        "act": "selu",
    },
    {
        "name": "matmul_random_sigm",
        "mode": "matmul",
        "matrix_type": "random",
        "act": "sigmoid",
    },
    {
        "name": "matmul_random_tanh",
        "mode": "matmul",
        "matrix_type": "random",
        "act": "tanh",
    },
    {"name": "matmul_random_selu", "mode": "matmul", "matrix_type": "random", "act": "selu"},  # ident_selu alone masks faults
    # The default credits cover every bank, so the host rarely runs out; three credits make it, for the no-credit start check.
    {"name": "matmul_relu_nopool_credits3", "mode": "matmul", "matrix_type": "random", "act": "relu", "credits": 3,
     "pool_h": 1, "pool_w": 1, "padding": 0},
    {"name": "matmul_tanh_credits3", "mode": "matmul", "matrix_type": "random", "act": "tanh", "credits": 3},
    {"name": "conv_basic_selu", "mode": "conv", "conv_type": "basic", "act": "selu"},
    {"name": "conv_basic_tanh", "mode": "conv", "conv_type": "basic", "act": "tanh"},
    {"name": "matmul_random_tanh_train", "mode": "matmul", "matrix_type": "random", "act": "tanh", "training": True},
    {"name": "matmul_random_sigm_train", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "training": True},
    {"name": "conv_basic_selu_train", "mode": "conv", "conv_type": "basic", "act": "selu", "training": True},
    {"name": "matmul_large_selu", "mode": "matmul", "matrix_type": "random", "act": "selu", "scale": 2.5},
    {"name": "matmul_large_sigm", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "scale": 2.5},
    {"name": "matmul_large_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "scale": 2.5},
    # Deep products split into passes: partial sums accumulate in the pipeline, only the last pass is activated.
    {"name": "matmul_accum2_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "accum_passes": 2},
    {"name": "matmul_accum3_selu", "mode": "matmul", "matrix_type": "random", "act": "selu", "accum_passes": 3},
    {"name": "matmul_accum2_tanh_train", "mode": "matmul", "matrix_type": "random", "act": "tanh",
     "accum_passes": 2, "training": True},
    # Modes the network layers use: ReLU and linear skip the polynomial, 1x1 pooling with no padding passes values through.
    {"name": "matmul_random_relu", "mode": "matmul", "matrix_type": "random", "act": "relu"},
    {"name": "matmul_zero_rows_relu", "mode": "matmul", "matrix_type": "random", "act": "relu", "zero_rows": True},
    # Linear keeps -0: integer sums cancel exactly (x + -x gives -0 when A is negative) and meet +0 in the pooling windows.
    {"name": "matmul_signed_zero_linear", "mode": "matmul", "matrix_type": "small_exact", "act": "linear", "zero_rows": True},
    {"name": "matmul_random_linear", "mode": "matmul", "matrix_type": "random", "act": "linear"},
    {"name": "matmul_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu",
     "pool_h": 1, "pool_w": 1, "padding": 0},
    {"name": "matmul_mixed_act", "mode": "matmul", "matrix_type": "random", "act": "tanh",
     "mixed_acts": ["tanh", "relu", "selu", "linear", "sigmoid", "relu"]},
    {"name": "matmul_accum2_mixed_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu",
     "accum_passes": 2, "pool_h": 1, "pool_w": 1, "padding": 0,
     "mixed_acts": ["relu", "relu", "linear", "linear", "tanh", "tanh", "relu", "relu"]},
    # A bias row added by the mesh: on every set, and on the first pass of each accumulate group.
    {"name": "matmul_bias_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "bias": True,
     "pool_h": 1, "pool_w": 1, "padding": 0},
    {"name": "matmul_bias_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "bias": True},
    {"name": "matmul_accum3_bias_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear",
     "bias": True, "accum_passes": 3, "pool_h": 1, "pool_w": 1, "padding": 0},
    # B from the mesh's weight cache: every set's B is written to its own tile once, then only A is sent.
    {"name": "matmul_cached_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "cached": True,
     "pool_h": 1, "pool_w": 1, "padding": 0},
    {"name": "matmul_accum2_cached_bias_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "cached": True,
     "bias": True, "accum_passes": 2},
    # int8 only: per-channel random requantize multipliers and shifts, and inputs whose zero point is far from 0.
    {"name": "int8_perchannel_random_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear",
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "formats": ("int8",)},
    {"name": "int8_input_zp_bias_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "bias": True,
     "a_range": (-0.25, 1.0), "pool_h": 1, "pool_w": 1, "padding": 0, "formats": ("int8",)},
    {"name": "int8_input_zp_perchannel_accum3_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh",
     "bias": True, "a_range": (0.0, 1.0), "req_random": True, "accum_passes": 3, "formats": ("int8",)},
    {"name": "int8_input_zp_selu_train", "mode": "matmul", "matrix_type": "random", "act": "selu",
     "a_range": (-0.1, 1.0), "training": True, "formats": ("int8",)},
    # int8 only: mixed activations in training, every set with its own zero point, clamps and multipliers; 24 sets reuse the 16 set ids.
    {"name": "int8_mixed_act_train", "mode": "matmul", "matrix_type": "random", "act": "sigmoid",
     "mixed_acts": ["sigmoid", "relu", "selu", "tanh", "linear", "selu"], "training": True, "req_random": True,
     "zp_random": True, "num_sets": 24, "formats": ("int8",)},
    # int8 only: all-negative A drives whole columns to tanh's -128, so edge windows hold only -128 beside the -128 pad.
    {"name": "int8_pad_negative_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "a_range": (-1.0, -0.5),
     "scale": 2.5, "formats": ("int8",)},
    # int8 only: each ReLU or linear set drains its requantize beats into a GPNAE set's fill; every set has its own zp, clamps, multipliers.
    {"name": "int8_mixed_bypass_lane_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu",
     "pool_h": 1, "pool_w": 1, "padding": 0, "mixed_acts": ["relu", "tanh", "linear", "selu"], "req_random": True,
     "zp_random": True, "num_sets": 16, "formats": ("int8",)},
    # int8 only: the same with a partial set right behind each draining ReLU or linear set, which must wait out the drain.
    {"name": "int8_accum2_mixed_bypass_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "accum_passes": 2,
     "pool_h": 1, "pool_w": 1, "padding": 0, "mixed_acts": ["relu", "relu", "tanh", "tanh", "linear", "linear", "selu", "selu"],
     "req_random": True, "zp_random": True, "num_sets": 16, "formats": ("int8",)},
]


def _run_make_live(log_path: str, fmt_name: str = "fp32") -> tuple:
    t0 = time.time()
    process = subprocess.Popen(
        ["make", "verilator", f"FMT={fmt_name}", "GEN_PKG=0"],  # GEN_PKG=0: build the package this test just wrote
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    raw_log = []

    with open(log_path, "w") as f:
        for line in process.stdout:
            raw_log.append(line)
            f.write(line)
            if "[STATUS" in line or "[STAGE]" in line or "[FATAL]" in line:
                print(f"      {_D}{line.strip()}{_X}")

    process.wait()
    return "".join(raw_log), time.time() - t0


def _parse_log(raw: str) -> dict:
    if "FATAL" in raw or "[FATAL] Timeout" in raw:
        return {
            "status": "TIMEOUT",
            "total": 0,
            "exact": 0,
            "tol": 0,
            "failed": 1,
            "cyc": 0,
        }
    ex = lambda p: int(m.group(1)) if (m := re.search(p, raw)) else 0
    tot, exact, tol, fail = (
        ex(r"Total\s+:\s+(\d+)"),
        ex(r"Exact\s+:\s+(\d+)"),
        ex(r"Tol pass\s+:\s+(\d+)"),
        ex(r"Failed\s+:\s+(\d+)"),
    )
    return {
        "status": "PASS" if fail == 0 and tot > 0 else "FAIL",
        "total": tot,
        "exact": exact,
        "tol": tol,
        "failed": fail,
        "cyc": ex(r"asserted @ \d+\s+\((\d+) cycles\)"),
    }


PKG_DEFAULT_TEST = "matmul_relu_nopool"  # in every format, so make pkg works with any FMT


def _pkg_fields(path: str) -> dict:
    """{name: int} of the package's int localparams, to check what was written."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"localparam int (\w+) = (-?\d+);", open(path).read())}


def write_pkg(N: int, T: int, test: str, lanes: int = 32, host_words: int = None, fmt_name: str = "fp32") -> None:
    """test_config_pkg.sv and one test's stimulus for one build; exits non-zero on a test the format does not run."""
    t = next((x for x in PIPELINE_TESTS if x["name"] == test), None)
    if t is None:
        near = [x["name"] for x in PIPELINE_TESTS if test in x["name"]]
        sys.exit(f"[ERROR] pkg: no pipeline test named '{test}'" + (f"; tests containing it: {', '.join(near)}" if near else ""))
    if fmt_name not in t.get("formats", tuple(FORMATS)):
        sys.exit(f"[ERROR] pkg: test '{test}' runs only in {', '.join(t['formats'])}, not {fmt_name}")
    generate_vectors({"n": N, "tile_size": T, "lanes": lanes, "host_words": host_words or N, "fmt_name": fmt_name, **t})
    path = os.path.join(TB_DIR, "test_config_pkg.sv")
    got = _pkg_fields(path)
    want = {"EXP_W": FORMATS[fmt_name][0], "MAN_W": FORMATS[fmt_name][1], "IS_INT": int(fmt_name == "int8"), "N": N, "TILE_SIZE": T}
    bad = {k: got.get(k) for k in want if got.get(k) != want[k]}
    if bad:
        sys.exit(f"[ERROR] pkg: {path} has {bad}, expected {want}")
    print(f"PACKAGE OK: {fmt_name} {test} N={N} TILE={T} LANES={lanes} COLLAPSE_K={COLLAPSE_K} -> {path}")


def run_regression(N: int, T: int, target_test: str = None, lanes: int = 32, host_words: int = None,
                   fmt_name: str = "fp32"):
    _check_dropout_generator()
    print(hdr(f"\n{'═'*70}\n  SIENNA PIPELINE — Regression Suite\n{'═'*70}"))
    tests_to_run = [t for t in PIPELINE_TESTS if fmt_name in t.get("formats", tuple(FORMATS))]  # int8-only tests skip the floats

    if target_test:
        tests_to_run = [t for t in tests_to_run if target_test in t["name"]]
        if not tests_to_run:
            print(f"  {_R}[ERROR] No tests found containing '{target_test}'{_X}")
            return

    print(
        f"  Matrix size : {N}×{N}\n  Tile size   : {T}×{T}\n  Format      : {fmt_name}\n  Total tests : {len(tests_to_run)}\n"
        + hdr(f"{'═'*70}")
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for idx, t in enumerate(tests_to_run):
        print(
            f"\n  ║  [{idx+1}/{len(tests_to_run)}] {_O}{t['name']}{_X}  (generating...)"
        )

        # 1. Generate Vectors & Dump Expected Traces
        cfg = {"n": N, "tile_size": T, "lanes": lanes, "host_words": host_words or N, "fmt_name": fmt_name, **t}
        generate_vectors(cfg)

        # 2. Run Verilator (Streams live status)
        log_path = os.path.join(RESULTS_DIR, f"{t['name']}.log")
        raw_log, wall = _run_make_live(log_path, fmt_name)

        # 3. Parse Log
        r = {
            "name": t["name"],
            "mode": t["mode"],
            "act": t["act"],
            "wall": wall,
            **_parse_log(raw_log),
        }
        results.append(r)

        # 4. Dump Formatted HW Matrix Trace
        dump_hardware_trace(t["name"], N, print_to_console=(r["status"] != "PASS"))

        # 5. Print Final Result
        sym = ok("PASS") if r["status"] == "PASS" else err("FAIL")
        tot = max(r["total"], 1)
        print(
            f"      {sym}   exact {100*r['exact']/tot:5.1f}%  tol {100*r['tol']/tot:5.1f}%  fail {r['failed']:3d}  {r['cyc']:6d} cyc  {r['wall']:5.1f}s"
        )

        if r["status"] != "PASS":
            print(f"  {_R}╚══  Sweep Aborted: {r['status']}.{_X}")
            sys.exit(1)

    print(f"\n  ╚══  Sweep Complete")

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(hdr(f"\n{'═'*70}\n  SUMMARY  —  N={N}  TILE={T}\n{'═'*70}"))
    print(f"  Passed      : {ok(passed)} / {len(results)}\n")
    print(
        hdr(
            f"  Verdict : {ok('✅ Sanity Clean') if passed == len(results) else err('❌ Failures detected')}"
        )
    )
    print(hdr(f"{'═'*70}\n"))
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SIENNA Pipeline Unified Tool")
    p.add_argument(
        "--action",
        default="regression",
        choices=["regression", "gen", "pkg", "analyze"],
        help="Action to perform; pkg writes test_config_pkg.sv and one --test's stimulus (default %s)" % PKG_DEFAULT_TEST,
    )
    p.add_argument("--matrix-size", "--n", type=int, default=16)
    p.add_argument("--tile-size", type=int, default=4)
    p.add_argument("--lanes", type=int, default=32)
    p.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(FORMATS),
                   help="number format of the build; narrow formats use a bit-exact golden")
    p.add_argument("--collapse-k", type=int, choices=[0, 1], default=1,
                   help="the mesh's COLLAPSE_K in this build, for the bit-exact golden (the RTL default is 1)")
    p.add_argument("--host-words", type=int, default=None, help="words per host write (default: N, one row)")
    p.add_argument("--mode", default="matmul", choices=["matmul", "conv"])
    p.add_argument("--conv-type", default="basic")
    p.add_argument("--activation", default="selu")
    p.add_argument(
        "--test", type=str, default=None, help="Run a specific test by name substring"
    )
    args, unknown = p.parse_known_args()
    COLLAPSE_K = args.collapse_k

    if args.action == "regression":
        run_regression(args.matrix_size, args.tile_size, args.test, args.lanes, args.host_words, args.fmt_name)
    elif args.action == "gen":
        generate_vectors(
            {
                "n": args.matrix_size,
                "tile_size": args.tile_size,
                "lanes": args.lanes,
                "host_words": args.host_words or args.matrix_size,
                "mode": args.mode,
                "conv_type": args.conv_type,
                "activation": args.activation,
                "name": "manual_gen",
                "fmt_name": args.fmt_name,
            }
        )
    elif args.action == "pkg":
        write_pkg(args.matrix_size, args.tile_size, args.test or PKG_DEFAULT_TEST, args.lanes, args.host_words, args.fmt_name)
    elif args.action == "analyze":
        dump_hardware_trace("manual_run", args.matrix_size, print_to_console=True)
