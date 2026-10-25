#!/usr/bin/env python3
"""
regression.py — SIENNA Full Pipeline Unified Regression Suite
=============================================================
Contains:
  1. Golden Model & Vector Generator
  2. Live Status Streamer
  3. Formatted Matrix Trace Dumper
  4. Regression Orchestrator & Scoreboard
  5. Hardware checks behind --action: pack, gemm, perf, oracle, pack-models, gpnae-tflite, rq-vectors, add-vectors
  6. Tool self-tests (--action selftest) and make regression's points and summary (--action points, point, summary)
"""

import argparse
import contextlib
import dataclasses
import functools
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPNAE"))
import mesh_model  # noqa: E402
import gpnae_model  # noqa: E402
from mesh_model import fpu  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))  # model_runner and check_hier
import model_runner as mr  # noqa: E402  the numerics and the device build package, one owner each
from model_runner import (  # noqa: E402
    FORMATS, ROOT, SETS_IN_FLIGHT, TB_DIR, _config_items, activate_int8, activation_to_code,
    apply_activation, bits_float, drop_zp, exact_layer, fmt_bits, fold_bias, get_polynomial_terms, imatmul, op_hex, op_round,
    quant_act, quant_weights, requant_params, requantize, wrap32, write_sv_package)
import ipu  # noqa: E402  AriL's integer model, on the path model_runner set
import fmt_pkg  # noqa: E402  the one parser of sienna_fmt_pkg's define map, on the same path
fourstate = mr.fourstate  # 4-state safe parsing: an x or z word, count or hash fails its test by name


def _load_truth():
    """SystolicMesh's ArithmeticLibrary copy of truth.py, by its path (GPNAE's copy may be first on sys.path); registered as truth."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh", "ArithmeticLibrary", "Common", "models", "truth.py")
    spec = importlib.util.spec_from_file_location("truth", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["truth"] = m  # any later import truth gets this one
    spec.loader.exec_module(m)
    return m


truth = _load_truth()  # AriL's true-function gates: every pipeline stage gates on them

# Ensure we can import the mesh helpers
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh"))

from conv_tests import _basic_pair, _general_pair, _kernel_size
from matmul_tests import _ref_matmul, pow2_tile_sizes, write_mem

RESULTS_DIR = os.path.join(ROOT, "testbenches", "results", "pipeline")
ACCUM_ZERO_PASSES = 4  # TB_sienna_top's accumulate pass: partial sets of zeros before the summed group
ACTIONS = ("regression", "gen", "pkg", "analyze", "pack", "gemm", "residual", "perf", "oracle", "pack-models", "gpnae-tflite", "rq-vectors", "add-vectors", "multi", "idle-fault", "selftest", "points", "point", "summary")

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


def write_op_mem(path: str, data, fmt: str) -> None:
    """Operands as hex words of the format's width; fp32 keeps the 8-digit words of write_mem."""
    if fmt == "fp32":
        write_mem(path, data)
        return
    with open(path, "w") as fh:
        fh.write("".join(w + "\n" for w in op_hex(data, fmt)))


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


MASK_PAIR_P = 1e-5  # chance that two independent half-rate masks agree as often as a failing pair


def check_masks_differ(test_name: str, masks: list, keep_p: float = 0.5) -> None:
    """Sets' dropout masks must differ: no two agree on more than 75% of their windows, where independent masks would with chance below MASK_PAIR_P (from 70 windows, just the 75%), nor all pairs together more than chance allows."""
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            n = len(masks[i])
            same = sum(a == b for a, b in zip(masks[i], masks[j]))
            if same / n <= 0.75:
                continue
            tail = sum(math.comb(n, m) for m in range(same, n + 1)) / 2**n  # independent masks agree about half the time
            if tail < MASK_PAIR_P:
                raise RuntimeError(f"{test_name}: sets {i} and {j} dropout masks agree on {same / n:.0%} ({same} of {n}, chance {tail:.1e} if independent)")
    if len(masks) > 1:  # all pairs together: weak correlation no single pair shows
        m, n = len(masks), len(masks[0])
        k = np.array(masks, dtype=np.int64).sum(axis=0)  # sets keeping each window
        same = int((k * (k - 1) // 2 + (m - k) * (m - k - 1) // 2).sum())
        tail = agree_total_chance(m, n, same, keep_p)
        if tail < MASK_PAIR_P:
            raise RuntimeError(f"{test_name}: the {m} sets' dropout masks agree on {same} of {m * (m - 1) // 2 * n} pair windows "
                               f"({same / (m * (m - 1) // 2 * n):.1%}, chance {tail:.1e} if independent)")


def agree_total_chance(m: int, n: int, same: int, q: float) -> float:
    """Exact chance that m independent masks, each keeping each of n windows with chance q, agree on at least same pair windows in all."""
    vmax = m * (m - 1) // 2
    pmf = np.zeros(vmax + 1)  # one window's agreeing pairs: k kept of m gives C(k,2) + C(m-k,2)
    for kk in range(m + 1):
        pmf[kk * (kk - 1) // 2 + (m - kk) * (m - kk - 1) // 2] += math.comb(m, kk) * q**kk * (1 - q) ** (m - kk)
    if same <= n * float(pmf @ np.arange(vmax + 1)):
        return 1.0  # not above the mean
    size = 1 << (n * vmax).bit_length()
    dist = np.fft.irfft(np.fft.rfft(pmf, size) ** n, size)[:n * vmax + 1]  # the n windows are independent: an n-fold convolution
    return float(max(dist[same:].sum(), 0.0))


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
                          cfg.get("collapse_k", mr.COLLAPSE_K), bias)
    A = gpnae_model.Lane(f).run(C, activation_to_code(act))  # the lanes' reset coefficient memory, tier 0
    P = _maxpool_bits(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1), f)
    return C, A, P, _dropout_bits(P, cfg, drop_seed, f)


def _special_bias(b, fmt: str) -> np.ndarray:
    """A bias row's bits with +0, -0 and the smallest and largest subnormals of each sign in 3 of every 4 columns (Review Focus 5)."""
    f = fpu.FORMATS[fmt]
    sg = 1 << (f.w - 1)
    special = [0, sg, 1, f.mmask, sg | 1, sg | f.mmask]
    b = np.array(b, np.int64)
    for i, c in enumerate(c for c in range(b.size) if c % 4 != 3):
        b[c] = special[i % len(special)]
    return b


def _dropout_bits(P, cfg: dict, drop_seed: int, f) -> np.ndarray:
    """Dropout of a set's pooled bits as the lanes apply it: scaled by 1/(1-p) where kept, a zero with the input's sign where dropped."""
    if not cfg.get("training", False):
        return P.copy()
    flat = P.flatten()
    keep = dropout_keep(flat.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    scale = fpu.from_fp32(int(np.float32(1.0 / (1.0 - cfg.get("dropout_p", 0.5))).view(np.uint32)), f.m)
    prod = fpu.mul(f, flat, np.full_like(flat, scale))[0]
    return np.where(keep, prod, flat & (1 << (f.w - 1))).reshape(P.shape)


# ===== int8 (D-6): affine quantization of the float tests' data, and the bit-exact golden (int8Requantizer, int8QuantizedAdder models) =====

REQ_HEAD = 8  # layer-wide words heading requant_<k>.mem: zp, min, max, gp_mx, gp_shx, gp_mout, gp_shout, gp_zout


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


def _golden_int8_residual(parts, hw_bias, rq: dict, R, add: dict, ra: dict, cfg: dict, act: str, drop_seed: int, q1=None) -> tuple:
    """Bit-exact int8 residual set: the depth passes' sums, the conv's requantize (q1, unless given), the ADD with R, the lane with the ADD's words, pooling, dropout (D-5: zp_o)."""
    N = cfg.get("n", 16)
    C = mesh_model.matmul_int([(_pad_square(a, N), _pad_square(b, N)) for a, b in parts], N, hw_bias)
    Y = mr.qadd_codes(requantize(C, rq) if q1 is None else q1, R, add)
    A = activate_int8(Y, act, ra)
    P = _maxpool_int(A, cfg.get("pool_h", 2), cfg.get("pool_w", 2), cfg.get("padding", 1))
    if not cfg.get("training", False):
        return C, Y, A, P, P.copy()
    keep = dropout_keep(P.size, cfg.get("dropout_p", 0.5), drop_seed, cfg.get("lanes", 32))
    return C, Y, A, P, np.where(keep, P.flatten(), drop_zp(act, ra)).reshape(P.shape)


ADD_TIGHT = (40, 91)  # a tight ADD clamp: amin and amax this many codes below and above zp_o, as a fused RELU6 or RELU_N1_TO_1 would cut
ADD_TRIES = 32  # draws of an add set before its writer gives up on a quarter of the outputs off the rails and 16 codes
LANE_ACTS = ("tanh", "sigmoid", "selu")  # a lane activation after the ADD: its inputs kept off saturation
ADD_SPAN = {"tanh": 2.0, "sigmoid": 4.0}  # tanh and sigmoid: R mostly cancels the conv, so the real sums lie in an uneven part of +-span


def _add_draw(rng, q1, s1: float, z1: int, act: str, zp_random: bool, setting=None, out_q=None, tight=False) -> tuple:
    """An add set's inputs for a conv output q1 (scale s1, zero point z1): the residual codes, the ADD's words and the lane's.
    setting fixes (s2 / s1, z2, s_out factor) and out_q (s_out, zp_o, amin, amax), a layer's words; tight clamps zp_o +- ADD_TIGHT codes.
    A lane activation takes s2 within 2^+-1 of s1 (tanh, sigmoid: 2^0..1, R cancelling the conv to ADD_SPAN) and s_out at or above its calibration;
    zp_random ReLU calibrates on the unclamped sums, so zp_o is interior and its clamp cuts the negative sums."""
    N = q1.shape[0]
    lane, span = act in LANE_ACTS, ADD_SPAN.get(act)
    f2, z2, fo = setting or (2.0 ** rng.uniform(*((0.0, 1.0) if span else (-1.0, 1.0) if lane else (-3.0, 3.0))),
                             int(rng.randint(-100, 101)) if zp_random else 0, 2.0 ** rng.uniform(*((0.0, 1.0) if lane else (-0.5, 1.0))))
    s2 = s1 * f2
    R = rng.randint(-128, 128, (N, N)).astype(np.int64)
    if span:
        lo, hi = -span * rng.uniform(0.2, 1.0), span * rng.uniform(0.2, 1.0)  # uneven, so zp_o is off 0
        R = np.clip(np.rint(z2 + (rng.uniform(lo, hi, (N, N)) - s1 * (q1 - z1)) / s2), -128, 127).astype(np.int64)
    y = s1 * (q1 - z1) + s2 * (R - z2)
    if out_q:
        s_out, z_out = out_q[:2]
    else:
        _, s_out, z_out = quant_act(np.maximum(y, 0.0) if act == "relu" and not zp_random else y)
        s_out *= fo if zp_random else 1.0  # random ADD scales: the output scale off its calibration, so some outputs clamp
    add = mr.add_words(s1, z1, s2, z2, s_out, z_out, act)
    if out_q:
        add.update(amin=out_q[2], amax=out_q[3])
    elif tight:
        add.update(amin=max(-128, z_out - int(rng.randint(*ADD_TIGHT))), amax=min(127, z_out + int(rng.randint(*ADD_TIGHT))))
    return R, add, dict(mr.add_lane_params(add, s_out, mr.qadd_codes(q1, R, add)), setting=(f2, z2, fo))


def _off_rails(F, ra: dict) -> tuple:
    """An add set's final outputs: the share off the rails (-128, 127 and the ADD's clamp) and the number of distinct codes."""
    F = np.asarray(F).ravel()
    return float(np.mean(~np.isin(F, [-128, 127, ra["amin"], ra["amax"]]))), len(np.unique(F))


def _identity_requant(N: int, ra: dict) -> dict:
    """An add set's requantize words: identity (mult 2^23, shift 23, zp 0, full clamp), with the lane's words after the ADD."""
    return dict(mult=np.full(N, 1 << 23, np.int64), shift=np.full(N, 23, np.int64), zp=0, amin=-128, amax=127,
                mx=ra["mx"], shx=ra["shx"], mout=ra["mout"], shout=ra["shout"], zout=ra["zout"])


def _decoy_params(N: int) -> dict:
    """A partial pass's requantize words: the activated pass's are the ones used, so these must never show in a result."""
    return dict(mult=np.full(N, 1 << 23, np.int64), shift=np.full(N, 24, np.int64), zp=5, amin=-100, amax=100,
                mx=1, shx=0, mout=1 << 23, shout=24, zout=5)


def _write_s8(path: str, v) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"{int(x) & 0xFF:02x}\n" for x in np.asarray(v).flatten()))


def _write_w32(path: str, v) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"{int(x) & 0xFFFFFFFF:08x}\n" for x in np.asarray(v).flatten()))


def _res_flags(partial=False, bias=False, hold=False, add=False, act="linear") -> int:
    """add_<k>.mem's first word: bit 0 partial, 1 bias, 2 hold, 3 add, bits 7:4 the set's activation code."""
    return int(partial) | int(bias) << 1 | int(hold) << 2 | int(add) << 3 | activation_to_code(act) << 4


def _requant_words(rq: dict) -> list:
    """requant_<k>.mem's words: the REQ_HEAD layer-wide words, then N 24-bit mantissas, then N right shifts (32-bit words)."""
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


def _residual_layout(cfg: dict, passes: int, mixed: list, credits: int, act: str) -> list:
    """A residual test's groups as (first set, depth passes, residual, activation): a residual group takes one more set, its add set."""
    every, want, out, k = 1 if cfg.get("residual") else cfg["residual_every"], cfg.get("num_sets"), [], 0
    while True:
        g = len(out)
        res = g % every == 0
        out.append((k, passes, res, mixed[g % len(mixed)] if mixed else act))
        k += passes + res
        if want is not None and k >= want:
            if k > want:
                raise ValueError(f"{cfg['name']}: {want} sets is not a whole number of this test's groups")
            return out
        if want is None and k > credits + 2 and len(out) >= len(mixed):
            return out


def _write_residual_groups(cfg: dict, groups: list, reals: list, seed: int, req_rng, drop_seed: int) -> tuple:
    """Every set's files of a residual test (operands, bias, requantize words, expected outputs, add_<k>.mem, set 1 alone); returns set 0's
    (A, B, bias, words, act) and the truth gate's lane scales."""
    N = cfg.get("n", 16)
    zqs = (_zp_draws(np.random.RandomState(seed + 8000), len(groups), ["linear" if r else a for _, _, r, a in groups])
           if cfg.get("zp_random") else [None] * len(groups))
    layers = cfg.get("add_layers", 0)  # add_layers L: groups g and g + L are one layer, with the same conv quantization and ADD words
    layer = {}  # layer -> (s_a, z_a, s_w, rq, out_q, add, setting): its first group's
    w = lambda name, k: os.path.join(TB_DIR, f"{name}_{k}.mem")
    first, scales = None, {}

    def alone(k, a, b, bias, rq_k, act_k):  # the back-to-back pass: set 1 alone, not partial, no hold or add, its own flags' bias and act
        if k == 1:
            Fb = _golden_int8([(a, b)], bias, rq_k, cfg, act_k, set_dropout_seed(drop_seed, 1))[4]
            _write_s8(os.path.join(TB_DIR, "expected_b2b.mem"), Fb)
            _write_s8(os.path.join(TB_DIR, "bound_b2b.mem"), np.zeros_like(Fb))

    for g, (g0, P, res, act_g) in enumerate(groups):
        L = g % layers if layers else None
        A_r, B_r = np.hstack([reals[k][0] for k in range(g0, g0 + P)]), np.vstack([reals[k][1] for k in range(g0, g0 + P)])
        if L in layer:  # a later group of a layer: its first group's scales and zero points
            s_a, z_a, s_w, rq = layer[L][:4]
            A_q = np.clip(np.rint(A_r / s_a) + z_a, -128, 127).astype(np.int64)
            B_q = np.clip(np.rint(B_r / s_w[None, :]), -127, 127).astype(np.int64)
        else:
            (A_q, s_a, z_a), (B_q, s_w) = quant_act(A_r), quant_weights(B_r)
        hw_bias = fold_bias(None, s_a, s_w, z_a, B_q)
        parts = [(A_q[:, i * N:(i + 1) * N], B_q[i * N:(i + 1) * N, :]) for i in range(P)]
        acc = wrap32(sum(imatmul(a, b) for a, b in parts) + hw_bias[None, :])
        if L not in layer:
            rq = requant_params(acc, s_a, s_w, "linear" if res else act_g, req_rng, zqs[g])
        if res:  # redrawn until a quarter of the add set's final outputs are off the rails, in 16 codes or more
            arng, q1 = np.random.RandomState(seed + 9000 + g0), requantize(acc, rq)
            for _ in range(ADD_TRIES):
                fixed = layer.get(L, (None,) * 7)  # a later group of a layer: its (s2 / s1, z2, s_out factor) and output words
                R, add, ra = _add_draw(arng, q1, rq["s_out"], rq["zp"], act_g, cfg.get("zp_random", False), fixed[6], fixed[4],
                                       act_g != "relu" and (g % 2 == 1 or act_g in cfg.get("tight_acts", ())))
                F = _golden_int8_residual(parts, hw_bias, rq, R, add, ra, cfg, act_g, set_dropout_seed(drop_seed, g0 + P))[4]
                off, codes = _off_rails(F, ra)
                if off >= 0.25 and codes >= min(16, F.size // 4):  # 16 codes, or a quarter of a small set's outputs (N = 8 pooled: 16)
                    break
            else:
                raise ValueError(f"{cfg['name']}: group {g}'s add set: no draw in {ADD_TRIES} with a quarter of its outputs off the rails "
                                 "and 16 codes")
            if layers:
                layer.setdefault(L, (s_a, z_a, s_w, rq, (ra["s_out"], add["zpo"], add["amin"], add["amax"]), add, ra["setting"]))
                assert add == layer[L][5] and rq is layer[L][3], f"{cfg['name']}: group {g}'s words are not layer {L}'s"
        for i in range(P):
            k, last = g0 + i, i == P - 1
            act_k, rq_k = ("linear" if res and last else act_g), (rq if last else _decoy_params(N))
            bias_k = hw_bias if i == 0 else np.zeros(N, np.int64)
            write_op_mem(w("matrix_west", k), parts[i][0], "int8")
            write_op_mem(w("matrix_north", k), parts[i][1], "int8")
            _write_w32(w("bias", k), bias_k)
            _write_w32(w("requant", k), _requant_words(rq_k))
            F = _golden_int8(parts, hw_bias, rq, cfg, act_g, set_dropout_seed(drop_seed, k))[4] if last and not res else np.zeros(0, np.int64)
            _write_s8(w("expected_output", k), F)  # empty for a partial or hold set
            _write_s8(w("bound_output", k), np.zeros_like(F))
            _write_w32(w("add", k), [_res_flags(partial=not last, bias=i == 0, hold=res and last, act=act_k)] + [0] * len(mr.ADD_FIELDS))
            if last:
                scales[str(k)] = {"ratio": rq["ratio"]} | ({} if res else {"0": [float(rq["s_out"]), float(rq["s_selu"])]})
            first = first or (parts[0][0], parts[0][1], hw_bias, rq_k, act_k)
            alone(k, parts[i][0], parts[i][1], bias_k, rq_k, act_k)
        if res:
            k = g0 + P
            rq_i = _identity_requant(N, ra)
            write_op_mem(w("matrix_west", k), R, "int8")
            write_op_mem(w("matrix_north", k), np.eye(N, dtype=np.int64), "int8")  # the identity pass: int8 code 1 on the diagonal
            _write_w32(w("bias", k), np.zeros(N, np.int64))
            _write_w32(w("requant", k), _requant_words(rq_i))
            F = _golden_int8_residual(parts, hw_bias, rq, R, add, ra, cfg, act_g, set_dropout_seed(drop_seed, k))[4]
            _write_s8(w("expected_output", k), F)
            _write_s8(w("bound_output", k), np.zeros_like(F))
            _write_w32(w("add", k), [_res_flags(add=True, act=act_g)] + [add[f] for f in mr.ADD_FIELDS])
            scales[str(k)] = {"0": [float(ra["s_out"]), float(ra["s_selu"])], "ratio": None, "add": [float(v) for v in add["ratios"]]}
            alone(k, R, np.eye(N, dtype=np.int64), np.zeros(N, np.int64), rq_i, act_g)
    return first, scales


def _generate_vectors_int8(cfg: dict) -> None:
    """int8 stimulus and bit-exact golden: the float tests' matrices quantized affinely (min/max), each set's parameters in requant_<k>.mem."""
    os.makedirs(TB_DIR, exist_ok=True)
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
    residual = bool(cfg.get("residual") or cfg.get("residual_every"))
    groups = _residual_layout(cfg, passes, mixed, credits, act_type) if residual else None
    num_sets = groups[-1][0] + passes + groups[-1][2] if residual else cfg.get("num_sets", len(mixed) or -(-(credits + 2) // passes) * passes)
    if not residual and num_sets % passes:  # TB_sienna_top's drains wait for every sum to close, so a stream must end on a whole group
        raise ValueError(f"{cfg.get('name', 'manual_gen')}: {num_sets} sets is not a whole number of {passes}-pass accumulate groups")
    use_bias = bool(cfg.get("bias", False))
    req_rng = np.random.RandomState(seed + 7000) if cfg.get("req_random") else None
    reals = [(A0 * scale, B0 * scale)]
    for k in range(1, num_sets):
        rng = np.random.RandomState(seed + 1000 + k)
        reals.append((rng.uniform(lo, hi, (N, N)) * scale, rng.uniform(-1.0, 1.0, (N, N)) * scale))
    if residual:
        first, scales = _write_residual_groups(cfg, groups, reals, seed, req_rng, drop_seed)
        starts = []  # the residual writer wrote every set: the group loop below runs none
    else:
        first, scales = None, {}  # scales: per activated set its lane's real input scale and SELU output scale, for the truth gate
        starts = list(range(0, num_sets, passes))
    acts_g = [mixed[(min(g0 + passes, num_sets) - 1) % len(mixed)] if mixed else act_type for g0 in starts]  # the activated pass's activation
    zqs = _zp_draws(np.random.RandomState(seed + 8000), len(starts), acts_g) if cfg.get("zp_random") else [None] * len(starts)
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
        complete = len(ks) == passes  # always, since num_sets is a whole number of groups
        for i, k in enumerate(ks):
            last = complete and i == len(ks) - 1
            rq_k = rq if last else _decoy_params(N)
            if last:
                scales[str(k)] = {"0": [float(rq["s_out"]), float(rq["s_selu"])], "ratio": rq["ratio"]}
            write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), parts[i][0], "int8")
            write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), parts[i][1], "int8")
            _write_w32(os.path.join(TB_DIR, f"bias_{k}.mem"), hw_bias if i == 0 else np.zeros(N, np.int64))
            _write_w32(os.path.join(TB_DIR, f"requant_{k}.mem"), _requant_words(rq_k))
            F = _golden_int8(parts, hw_bias, rq, cfg, act_g, set_dropout_seed(drop_seed, k))[4] if last \
                else np.zeros(0, np.int64)
            _write_s8(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F)  # empty for a partial set
            _write_s8(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(F))
            if k == 0:
                first = (parts[0][0], parts[0][1], hw_bias, rq_k, act_type)
            if k == 1:  # the back-to-back pass: set 1 alone, not partial, with its own bias, requantize words and activation
                Fb = _golden_int8([parts[i]], hw_bias if i == 0 else np.zeros(N, np.int64), rq_k, cfg,
                                  mixed[1 % len(mixed)] if mixed else act_type, set_dropout_seed(drop_seed, 1))[4]
                _write_s8(os.path.join(TB_DIR, "expected_b2b.mem"), Fb)
                _write_s8(os.path.join(TB_DIR, "bound_b2b.mem"), np.zeros_like(Fb))
    # The single-set pass starts set 0 alone, not partial, with set 0's bias and requantize words.
    a0, b0, hb0, rq0, act0 = first  # residual: set 0's flags give its activation (linear for a hold set), and the TB clears its hold and add
    C0, _, A0q, P0, F0 = _golden_int8([(a0, b0)], hb0, rq0, cfg, act0, drop_seed)
    write_op_mem(os.path.join(TB_DIR, "matrix_west.mem"), a0, "int8")
    write_op_mem(os.path.join(TB_DIR, "matrix_north.mem"), b0, "int8")
    _write_s8(os.path.join(TB_DIR, "expected_output.mem"), F0)
    _write_s8(os.path.join(TB_DIR, "bound_output.mem"), np.zeros_like(F0))
    dump_golden_trace(test_name, C0.astype(np.float32), A0q.astype(np.float32), P0.astype(np.float32))
    with open(TRUTH_SCALES, "w") as fh:
        json.dump(scales, fh)
    mr.write_test_config(os.path.join(TB_DIR, "test_config_pkg.sv"),
                         _config_items(cfg, "int8", act_type, num_sets, credits, passes, mixed, True, drop_seed))
    _check_mem_widths("int8", num_sets, residual)


def _check_mem_widths(fmt: str, num_sets: int, residual: bool = False) -> None:
    """Every operand, bias and expected word the TB reads must be the format's width: a stale fp32 file would be truncated."""
    d = 2 if fmt == "int8" else (fpu.FORMATS[fmt].w + 3) // 4  # int8: operands and results 2 digits; int32 bias, requantize and ADD words 8
    per_set = ("matrix_west", "matrix_north", "bias", "expected_output", "bound_output") + (("requant",) if fmt == "int8" else ()) + \
        (("add",) if residual else ())
    names = [f"{b}.mem" for b in ("matrix_west", "matrix_north", "expected_output", "bound_output", "expected_b2b", "bound_b2b")]
    names += [f"{b}_{k}.mem" for k in range(num_sets) for b in per_set]
    names += [] if fmt == "int8" else ["expected_accum.mem"]  # int8's accumulate pass reads its group's own golden
    for fn in names:
        want = 8 if fmt == "int8" and fn.startswith(("bias_", "requant_", "add_")) else d
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
    return _general_pair(N, seed)[0]  # 3x3 kernel (2x2 below N = 9), depth zero-padded to N


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


PACK_ENTRIES = 8  # sienna_top's PACK_ENTRIES; entry 0 is the per-set ports


def pack_shift_of(N: int, k: int) -> int:
    """Set k's pack shift in the packed tests: packed, unpacked, b = 2 (fewer columns than the PE's slots), b = N / 4."""
    lg = N.bit_length() - 1
    return [1, 0, lg - 1, min(2, lg - 1)][k % 4]


def pack_map_of(N: int, sh: int, k: int) -> list:
    """Entry of each of the N/2 block slots: the set's 2^sh blocks rotate through the entries; an unpacked set uses entry 0."""
    return [((c + k) % PACK_ENTRIES if sh and c < (1 << sh) else 0) for c in range(N // 2)]


def _write_pack(path: str, sh: int, mp: list, ents: list) -> None:
    """pack_<k>.mem: the shift, N/2 map words, then entries 1..7 as act, zp, min, max, mx, shx, mout, shout, zout."""
    words = [sh] + list(mp)
    for act, rq in ents[1:]:
        q = rq or {}
        words += [activation_to_code(act)] + [int(q.get(x, 0)) for x in ("zp", "amin", "amax", "mx", "shx", "mout", "shout", "zout")]
    _write_w32(path, np.array(words, np.int64))


def _packed_float(cfg, k, A, B, bias, sh, col_ent, acts, drop):
    """One packed float set's files; the golden is each column's block through its entry's activation, then dropout."""
    N, fmt = cfg.get("n", 16), cfg.get("fmt_name", "fp32")
    A, B = op_round(A, fmt), op_round(B, fmt)
    b = op_round(bias, fmt) if bias is not None else np.zeros(N, np.float32)
    write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), A, fmt)
    write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), B, fmt)
    if fmt == "fp32":
        write_mem(os.path.join(TB_DIR, f"bias_{k}.mem"), b)
        C = (_ref_matmul(A, B) + b).astype(np.float32)
        S = np.abs(A).astype(np.float64) @ np.abs(B).astype(np.float64) + np.abs(b)
        Ca, Bnd = np.zeros_like(C), np.zeros_like(C)
        for e in sorted(set(col_ent.tolist())):
            cols = col_ent == e
            Ca[:, cols] = apply_activation(C, acts[e])[:, cols]
            Bnd[:, cols] = fp32_error_bound(S, N, cfg, acts[e], C)[:, cols]
        F = apply_dropout(Ca, cfg.get("dropout_p", 0.5), cfg.get("training", False), drop, cfg.get("lanes", 32))
        write_mem(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F)
        write_mem(os.path.join(TB_DIR, f"bound_output_{k}.mem"), Bnd)
        return
    f = fpu.FORMATS[fmt]
    write_op_mem(os.path.join(TB_DIR, f"bias_{k}.mem"), b, fmt)
    Ab, Bb = fmt_bits(A, fmt), fmt_bits(B, fmt)
    bb = fmt_bits(b, fmt) if bias is not None else None
    C = mesh_model.matmul_packed(f, Ab, Bb, N, sh, bb) if sh else mesh_model.matmul(f, [(Ab, Bb)], N, cfg.get("tile_size", 4), 1, bb)
    lane = gpnae_model.Lane(f)  # the lanes' reset coefficient memory, tier 0
    Aout = np.zeros_like(C)
    for e in sorted(set(col_ent.tolist())):
        cols = col_ent == e
        Aout[:, cols] = lane.run(C, activation_to_code(acts[e]))[:, cols]
    F = _dropout_bits(Aout, cfg, drop, f)
    write_bits(os.path.join(TB_DIR, f"expected_output_{k}.mem"), F, fmt)
    write_bits(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(F), fmt)


def _packed_int8(cfg, k, A, B, bias, sh, col_ent, acts, drop, req_rng, zqs):
    """One packed int8 set: one input scale, per-column weights, each entry's parameters from its columns' sums, drop to its zp (D-5)."""
    N = cfg.get("n", 16)
    A_q, s_a, z_a = quant_act(A)
    B_q, s_w = quant_weights(B)
    hw_bias = fold_bias(bias, s_a, s_w, z_a, B_q)
    acc = wrap32(imatmul(A_q, B_q) + hw_bias[None, :])
    mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
    ents = [(acts[e], None) for e in range(PACK_ENTRIES)]
    Y, dzp = np.zeros_like(acc), np.zeros(N, np.int64)
    for e in sorted(set(col_ent.tolist())):
        cols = col_ent == e
        rq = requant_params(acc[:, cols], s_a, s_w[cols], acts[e], req_rng, zqs[e])
        mult[cols], shift[cols] = rq["mult"], rq["shift"]
        ents[e] = (acts[e], rq)
        Y[:, cols] = activate_int8(requantize(acc[:, cols], rq), acts[e], rq)
        dzp[cols] = drop_zp(acts[e], rq)
    if cfg.get("training", False):
        keep = dropout_keep(N * N, cfg.get("dropout_p", 0.5), drop, cfg.get("lanes", 32)).reshape(N, N)
        Y = np.where(keep, Y, dzp[None, :])
    head = ents[0][1] or requant_params(acc, s_a, s_w, acts[0], req_rng, zqs[0])  # entry 0 rides on the per-set ports
    write_op_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), A_q, "int8")
    write_op_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), B_q, "int8")
    _write_w32(os.path.join(TB_DIR, f"bias_{k}.mem"), hw_bias)
    _write_w32(os.path.join(TB_DIR, f"requant_{k}.mem"), _requant_words(dict(head, mult=mult, shift=shift)))
    _write_s8(os.path.join(TB_DIR, f"expected_output_{k}.mem"), Y)
    _write_s8(os.path.join(TB_DIR, f"bound_output_{k}.mem"), np.zeros_like(Y))
    return ents


def _generate_vectors_packed(cfg: dict) -> None:
    """Packed sets: block-diagonal B, a shift per set, an 8-entry act (int8: output) table by block; golden = each job alone; 1x1 pool."""
    os.makedirs(TB_DIR, exist_ok=True)
    N, fmt, act_type, acts = cfg.get("n", 16), cfg.get("fmt_name", "fp32"), cfg["act"], cfg["pack_acts"]
    assert len(acts) == PACK_ENTRIES and acts[0] == act_type, (cfg["name"], "pack_acts[0] must be the test's act")
    assert (cfg.get("pool_h"), cfg.get("pool_w"), cfg.get("padding")) == (1, 1, 0), (cfg["name"], "packed sets need a 1x1 pool")
    seed = cfg.get("seed", 42) + int(os.environ.get("SIENNA_SEED", "0"))
    credits = cfg.get("credits", SETS_IN_FLIGHT)
    num_sets = cfg.get("num_sets", credits + 2)
    drop_seed = 0x2ACE0000 + seed
    use_bias = bool(cfg.get("bias", False))
    req_rng = np.random.RandomState(seed + 7000) if cfg.get("req_random") else None
    zqs = _zp_draws(np.random.RandomState(seed + 8000), PACK_ENTRIES, acts) if cfg.get("zp_random") else [None] * PACK_ENTRIES
    scales = {}  # int8: per set and pack entry, the lane's real input scale and SELU output scale, for the truth gate
    for k in range(num_sets):
        sh, rng = pack_shift_of(N, k), np.random.RandomState(seed + 1000 + k)
        mp = pack_map_of(N, sh, k)
        col_ent = np.array([mp[j // (N >> sh)] if sh else mp[0] for j in range(N)])
        b = N >> sh
        mask = np.kron(np.eye(N // b), np.ones((b, b))).astype(bool) if sh else np.ones((N, N), bool)
        A = rng.uniform(-1.0, 1.0, (N, N))
        B = np.where(mask, rng.uniform(-1.0, 1.0, (N, N)), 0.0)
        bias = rng.uniform(-1.0, 1.0, N) if use_bias else None
        drop = set_dropout_seed(drop_seed, k)
        if fmt == "int8":
            ents = _packed_int8(cfg, k, A, B, bias, sh, col_ent, acts, drop, req_rng, zqs)
            scales[str(k)] = {str(e): [float(q["s_out"]), float(q["s_selu"])] for e, (_, q) in enumerate(ents) if q}
            ratio = [None] * N  # per column, its entry's real ratio s_a s_w / s_out (none where the multipliers are random)
            for e, (_, q) in enumerate(ents):
                if q and q.get("ratio") is not None:
                    for c, r in zip(np.flatnonzero(col_ent == e), q["ratio"]):
                        ratio[int(c)] = r
            scales[str(k)]["ratio"] = ratio
        else:
            _packed_float(cfg, k, A, B, bias, sh, col_ent, acts, drop)
            ents = [(a, None) for a in acts]
        _write_pack(os.path.join(TB_DIR, f"pack_{k}.mem"), sh, mp, ents)
    for name in ("matrix_west", "matrix_north", "expected_output", "bound_output"):  # the single-set pass runs set 0
        shutil.copy(os.path.join(TB_DIR, f"{name}_0.mem"), os.path.join(TB_DIR, f"{name}.mem"))
    for name in ("expected", "bound"):  # the back-to-back pass runs set 1 alone; packed sets never accumulate
        shutil.copy(os.path.join(TB_DIR, f"{name}_output_1.mem"), os.path.join(TB_DIR, f"{name}_b2b.mem"))
    if fmt == "int8":
        with open(TRUTH_SCALES, "w") as fh:
            json.dump(scales, fh)
    mr.write_test_config(os.path.join(TB_DIR, "test_config_pkg.sv"),  # int8 always sends bias_<k>: it carries the folded input zero point
                         _config_items(cfg, fmt, act_type, num_sets, credits, 1, [], use_bias or fmt == "int8", drop_seed))
    if fmt != "fp32":
        _check_mem_widths(fmt, num_sets)


def generate_vectors(cfg: dict) -> None:
    if cfg.get("packed"):
        return _generate_vectors_packed(cfg)
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
    special = bool(cfg.get("bias_special", False))  # bias words +0, -0 and subnormals of each sign in 3 of every 4 columns
    if special and not (exact and use_bias):
        raise ValueError(f"{test_name}: bias_special needs a bias and a bit-exact format, not {fmt}")
    bias_bits = lambda k: _special_bias(fmt_bits(bias_of(k), fmt), fmt) if special else fmt_bits(bias_of(k), fmt)
    C0 = _ref_matmul(A, B) + (bias_of(0) if use_bias else np.float32(0.0))
    C, C_act, C_pooled, C_final = _golden_from_c(C0.astype(np.float32), cfg, act_type, drop_seed)

    # Write files for Verilator testbench
    write_op_mem(os.path.join(TB_DIR, "matrix_west.mem"), A, fmt)
    write_op_mem(os.path.join(TB_DIR, "matrix_north.mem"), B, fmt)
    if exact:
        F0 = _golden_bits([(fmt_bits(A, fmt), fmt_bits(B, fmt))], bias_bits(0) if use_bias else None,
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
    if num_sets % passes:  # TB_sienna_top's drains wait for every sum to close, so a stream must end on a whole group
        raise ValueError(f"{cfg.get('name', 'manual_gen')}: {num_sets} sets is not a whole number of {passes}-pass accumulate groups")
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
            bbk = bias_bits(k) if use_bias and k % passes == 0 else fmt_bits(bk, fmt)
            write_bits(os.path.join(TB_DIR, f"bias_{k}.mem"), bbk, fmt)  # the operand format's words; the TB widens them
            if k % passes == 0:
                grp, gbias = [], (bbk if use_bias else None)
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
            if k == passes - 1:  # TB_sienna_top's accumulate pass: zero partials with set 0's B, then group 0; the mesh's slot rotation runs on through them
                Z = np.zeros_like(grp[0][0])
                write_bits(os.path.join(TB_DIR, "expected_accum.mem"), _golden_bits([(Z, grp[0][1])] * ACCUM_ZERO_PASSES + grp, gbias, cfg,
                                                                                  act_k, set_dropout_seed(drop_seed, k), fmt)[3], fmt)
            Fk = bits_float(Fk, fmt)  # the dropout-mask check below reads values
        else:
            write_mem(os.path.join(TB_DIR, f"expected_output_{k}.mem"), Fk)  # empty for a partial set
            Bnd = np.zeros(0, dtype=np.float32) if partial else \
                fp32_error_bound(run_s if passes > 1 else Sk, N * passes, cfg, act_k,
                                 run if passes > 1 else (_ref_matmul(Ak, Bk) + bk).astype(np.float32))
            write_mem(os.path.join(TB_DIR, f"bound_output_{k}.mem"), Bnd)
        if k == 1:  # the back-to-back pass: set 1 alone, not partial, with its own bias (a group's first pass only) and activation
            seed1 = set_dropout_seed(drop_seed, 1)
            if exact:
                Fb = _golden_bits([(fmt_bits(Ak, fmt), fmt_bits(Bk, fmt))], bbk if use_bias and k % passes == 0 else None, cfg, act_k, seed1, fmt)[3]
                write_bits(os.path.join(TB_DIR, "expected_b2b.mem"), Fb, fmt)
                write_bits(os.path.join(TB_DIR, "bound_b2b.mem"), np.zeros_like(Fb), fmt)
            else:
                C1 = (_ref_matmul(Ak, Bk) + bk).astype(np.float32)
                write_mem(os.path.join(TB_DIR, "expected_b2b.mem"), _golden_from_c(C1, cfg, act_k, seed1)[3])
                write_mem(os.path.join(TB_DIR, "bound_b2b.mem"), fp32_error_bound(Sk, N, cfg, act_k, C1))
        if cfg.get("training", False) and not partial:  # the set's keep decisions, not Fk != 0: an output already zero is not a drop
            keep = dropout_keep(Fk.size, cfg.get("dropout_p", 0.5), set_dropout_seed(drop_seed, k), cfg.get("lanes", 32))
            if (Fk.flatten()[~keep] != 0).any():
                raise RuntimeError(f"{test_name}: set {k}'s golden keeps a window its dropout mask drops")
            masks.append(tuple(keep.tolist()))
    check_masks_differ(test_name, masks, 1 - cfg.get("dropout_p", 0.5))

    # Dump the intermediate Golden Trace for debug comparisons
    dump_golden_trace(test_name, C, C_act, C_pooled)

    # Dump SV Config Package
    mr.write_test_config(os.path.join(TB_DIR, "test_config_pkg.sv"),
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
# PART 2b: TRUE-FUNCTION GATES PER STAGE (truth.py)
# =============================================================================
# Each stage of a pipeline test is judged against the true function of the words that stage actually received, from
# TB_sienna_top's truth dump of every pass: the single-set pass (P0), the back-to-back pass (B2B, in a -DBACK_TO_BACK build), each
# streaming pass (S1..S4: plain, credit overrun, after a reset, the output hold) and the accumulate pass (A). The TB's bit-exact
# (or fp32 tolerance) compare of the outputs stays as a diagnostic, still required.

TRUTH_DUMP = os.path.join(TB_DIR, "truth_dump.txt")
TRUTH_SCALES = os.path.join(TB_DIR, "truth_scales.json")  # int8: each activated set's real lane scales per pack entry, from the generator
ACT_NAMES = {v: k for k, v in mr.ACTIVATION_CODES.items()}
INT8_FIXED_OUT = mr.INT8_FIXED_OUT  # the converters' fixed output quantization (a convention), one owner
TRUTH_STAGES = ("matmul", "narrow", "transfer", "requant", "add", "scale", "relu", "activation", "maxpool", "dropout")
TRUTH_STREAMS = ("S1", "S2", "S3", "S4")  # TB_sienna_top's streaming passes, in order; every one is dumped and gated


def _tb_words(name: str) -> np.ndarray:
    with open(os.path.join(TB_DIR, name)) as fh:
        return np.array([int(t, 16) for t in fh.read().split()], dtype=np.int64)


def _sx(v, bits: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64) & ((1 << bits) - 1)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v)


def _tb_knobs() -> dict:
    with open(os.path.join(TB_DIR, "test_knobs.mem")) as fh:
        return {k: int(v) for k, v in (ln.split() for ln in fh if ln.strip())}


def read_truth_dump(path: str, lanes: int, test: str = "-") -> dict:
    """{pass: {"M": {r: {e: (sum, word)}}, "X"/"A"/"R": {k: {e: word}}, "Q": {k: {e: (q1, q2, y)}}, "P": {k: {window: (word, keep)}},
    "D": {k: {i: word}}}}; R: a hold set's beats into hold_mem, Q: an add set's ADD inputs and output."""
    d, n = {}, 0
    h = lambda x: fourstate.word(x, test, n, "dumped word")  # an x or z raises fourstate.Unknown, numbered by the dump's line
    with open(path) as fh:
        for n, ln in enumerate(fh):
            t = ln.split()
            if not t:
                continue
            if t[0] == "H":  # the build's passes
                d.setdefault("_hdr", {})[t[1]] = int(t[2])
                continue
            g = d.setdefault(t[1], {"M": {}, "X": {}, "A": {}, "P": {}, "D": {}, "Q": {}, "R": {}})
            if t[0] == "M":
                g["M"].setdefault(int(t[2]), {})[int(t[3])] = (h(t[4]), h(t[5]))
            elif t[0] == "Q":
                g["Q"].setdefault(int(t[2]), {})[int(t[3])] = (h(t[4]), h(t[5]), h(t[6]))
            elif t[0] in ("X", "A", "D", "R"):
                g[t[0]].setdefault(int(t[2]), {})[int(t[3])] = h(t[4])
            elif t[0] == "P":
                g["P"].setdefault(int(t[2]), {})[int(t[4]) * lanes + int(t[3])] = (h(t[5]), fourstate.word(t[6], test, n, "dropout keep bit", 10))
            else:
                raise ValueError(f"truth dump: unknown line {ln.strip()!r}")
    return d


def _full(m: dict, n: int, what: str) -> list:
    """m's values in key order 0..n-1; anything missing or extra is a dump fault."""
    if sorted(m) != list(range(n)):
        raise RuntimeError(f"truth dump: {what} holds {len(m)} of {n} entries")
    return [m[i] for i in range(n)]


def _res_flags_of(k: int) -> dict:
    """A residual test's set k as add_<k>.mem gives it: partial, bias, hold and add flags, its activation, and the ADD's words (zero points
    and clamp signed, mantissas and shift unsigned)."""
    w = _tb_words(f"add_{k}.mem")
    f = int(w[0])
    words = {n: int(v & 0xFFFFFFFF) if n in ("m1", "m2") else int(v & 0x3F) if n == "shift" else int(_sx(v, 8))
             for n, v in zip(mr.ADD_FIELDS, w[1:])}
    return {"partial": bool(f & 1), "bias": bool(f & 2), "hold": bool(f & 4), "add": bool(f & 8), "act": ACT_NAMES[(f >> 4) & 15],
            "words": words}


def _set_words(k: int, knobs: dict, N: int, fmt: str) -> dict:
    """Set k as the TB drives it: each column's pack entry, each entry's activation and (int8) requantize and lane words."""
    code0 = (knobs["mixed_acts"] >> (4 * (k % knobs["mixed_len"]))) & 15 if knobs["mixed_len"] else knobs["activation_code"]
    if knobs.get("residual"):  # TB_sienna_top's act_of(k): a residual test's flags
        code0 = (int(_tb_words(f"add_{k}.mem")[0]) >> 4) & 15
    ents = {0: {"act": ACT_NAMES[code0]}}
    col_ent = np.zeros(N, dtype=np.int64)
    sh = 0
    if knobs["packed"]:
        pk = _tb_words(f"pack_{k}.mem")
        sh = int(pk[0])
        mp = pk[1:1 + N // 2]
        lg = N.bit_length() - 1
        col_ent = np.array([mp[0] if sh == 0 else mp[c >> (lg - sh)] for c in range(N)], dtype=np.int64)
        for e in range(1, PACK_ENTRIES):
            o = 1 + N // 2 + 9 * (e - 1)
            ents[e] = {"act": ACT_NAMES.get(int(pk[o]) & 7, "linear"), "zp": int(_sx(pk[o + 1], 8)), "amin": int(_sx(pk[o + 2], 8)),
                       "amax": int(_sx(pk[o + 3], 8)), "mx": int(pk[o + 4] & 0xFFFF), "shx": int(pk[o + 5] & 0x1F),
                       "mout": int(pk[o + 6] & 0xFFFFFF), "shout": int(pk[o + 7] & 0x3F), "zout": int(_sx(pk[o + 8], 8))}
    if fmt == "int8":
        rq = _tb_words(f"requant_{k}.mem")
        ents[0].update(zp=int(_sx(rq[0], 8)), amin=int(_sx(rq[1], 8)), amax=int(_sx(rq[2], 8)), mx=int(rq[3] & 0xFFFF), shx=int(rq[4] & 0x1F),
                       mout=int(rq[5] & 0xFFFFFF), shout=int(rq[6] & 0x3F), zout=int(_sx(rq[7], 8)))
        mult, shift = rq[8:8 + N] & 0xFFFFFF, rq[8 + N:8 + 2 * N] & 0x3F
    else:
        mult = shift = None
    return {"ents": ents, "col_ent": col_ent, "sh": sh, "mult": mult, "shift": shift}


def _int8_quant(ents: dict, scales: dict, k: int):
    """Per entry (s_in, z_in, s_out, z_out) of the lane: the generator's real scales, the zero points the TB drove."""
    out = {}
    for e, w in ents.items():
        sc = scales.get(str(k), {}).get(str(e))
        if sc is None:  # a set the generator calibrated no lane for (P0's or B2B's partial): the scales its words encode
            s_in = w["mx"] * 2.0 ** -w["shx"] / 2048  # SELU's output: acc 2^-(shx + 18) times mout 2^-shout is the code
            sc = [s_in, s_in * 2.0 ** (w["shout"] - 7) / (w["mx"] * w["mout"]) if w["mout"] and w["mx"] else np.inf]
        s_in, s_selu = sc
        a = w["act"]
        s_out, z_out = (s_selu, w["zout"]) if a == "selu" else INT8_FIXED_OUT.get(a, (s_in, w["zp"]))
        out[e] = (s_in, w["zp"], s_out, z_out)
    return out


def _relu_clamp(w: dict, k: int, e: int) -> "truth.Result":
    """An int8 ReLU entry is the requantize clamp (sienna_top runs the lane as linear): its minimum must be max(zp, -128), the fused ReLU."""
    ok = w["amin"] == max(w["zp"], -128)
    return truth.Result(ok, "int8", "relu clamp", 1, 0.0 if ok else float(w["zp"] - w["amin"]), 0.0, 0, int(not ok), int(not ok), 0, 0, 0,
                        [] if ok else [(k, e, w["amin"], w["zp"])])


def _drop_value(w: dict) -> int:
    """D-5: dropout's int8 drop value, the output zero point of the entry's activation (SELU zout, sigmoid -128, ReLU and linear zp, tanh 0)."""
    return {"selu": w["zout"], "sigmoid": -128, "relu": w["zp"], "linear": w["zp"]}.get(w["act"], 0)


def _transfer(fmt: str, want, got) -> "truth.Result":
    """The activation stage's inputs must be the mesh's words, exactly (an identity: L3 carries them unchanged)."""
    want, got = np.asarray(want, dtype=np.int64), np.asarray(got, dtype=np.int64)
    bad = np.flatnonzero(want != got)
    return truth.Result(bad.size == 0, fmt, "transfer", int(want.size), float(bad.size), 0.0, int(bad[0]) if bad.size else 0, int(bad.size),
                        int(bad.size), 0, 0, 0, [(int(i), int(got[i]), int(want[i])) for i in bad[:truth.FIRST]])


def _act_bound(func: str, tier: int = 0) -> float:
    """An activation stage's bound: ReLU and linear at the 1% target, the lane activations at their tier's tolerance (1% at tier 0)."""
    return truth.TARGET if func in ("relu", "linear") else gpnae_model.TIER_TOL[tier]


def truth_test(N: int, fmt: str, lanes: int) -> dict:
    """{stage (activation: 'activation <func>'): Result} over every activated set of the dumped passes of the test just run."""
    knobs, pkg = _tb_knobs(), _pkg_fields(os.path.join(TB_DIR, "test_config_pkg.sv"))
    if not os.path.exists(TRUTH_DUMP):
        raise RuntimeError(f"no truth dump at {TRUTH_DUMP}")
    dump = read_truth_dump(TRUTH_DUMP, lanes, f"N{N} {fmt}")
    hdr = dump.pop("_hdr", {})
    if "b2b" not in hdr:
        raise RuntimeError("truth dump: no 'H b2b' line")
    res_t = bool(knobs.get("residual"))  # a residual test: each set's flags from add_<k>.mem, and no accumulate pass
    want = {"P0"} | set(TRUTH_STREAMS) | ({"B2B"} if hdr["b2b"] else set()) | (set() if knobs["packed"] or res_t else {"A"})  # no A for packed sets
    if set(dump) != want:
        raise RuntimeError(f"truth dump: passes {sorted(dump)}, the TB runs {sorted(want)}")
    scales = json.load(open(TRUTH_SCALES)) if fmt == "int8" else {}
    P, K, W = knobs["accum_passes"], knobs["num_sets"], (pkg["IN_ROWS"] + 2 * pkg["PADDING"] - pkg["POOL_H"]) // pkg["POOL_H"] + 1
    W *= (pkg["IN_COLS"] + 2 * pkg["PADDING"] - pkg["POOL_W"]) // pkg["POOL_W"] + 1
    ph, pw, pad, p = pkg["POOL_H"], pkg["POOL_W"], pkg["PADDING"], pkg["DROPOUT_P_PERCENT"] / 100.0
    acc = truth.acc_fmt(fmt)
    res, mult_seen = {}, set()
    def add(name, r):
        res.setdefault(name, []).append(r)
        if not r.ok:
            res.setdefault("_passes", {}).setdefault(name, set()).add(tag)
    def multipliers(k, sw):  # the host's scale codes, once per set: each within s 2^-24 of its real scale s_a s_w / s_out
        ratio = scales.get(str(k), {}).get("ratio") or []
        cs = [c for c in range(min(N, len(ratio))) if ratio[c] is not None]
        if cs and k not in mult_seen:
            mult_seen.add(k)
            add("scale", truth.gate_scale([ratio[c] for c in cs], sw["mult"][cs], sw["shift"][cs]))
    def real_scales(k, sw, cols):  # per element its channel's real scale; the words' own exact scale where the generator kept none
        ratio = scales.get(str(k), {}).get("ratio") or [None] * N
        code = mr.code_scales(sw["mult"], sw["shift"])
        return np.array([code[c] if ratio[c] is None else ratio[c] for c in range(N)], dtype=np.float64)[cols]
    alone = {"P0": 0, "B2B": 1}  # the single-set and back-to-back passes put set 0 and set 1 alone, from their own files
    fl = {k: _res_flags_of(k) for k in range(K)} if res_t else {}
    for tag, g in sorted(dump.items()):
        if tag.startswith("S"):  # the sets with a mesh result: a residual test's hold and add sets too
            acts = [k for k in range(K) if not fl[k]["partial"]] if res_t else [k for k in range(K) if k % P == P - 1]
        else:
            acts = [alone[tag]] if tag in alone else [P - 1]
        M = _full(g["M"], len(acts), f"pass {tag}'s results")
        held = {}  # per hold set of the pass: its requantized beats, which hold_mem gives its add set
        for r, k in enumerate(acts):
            sw = _set_words(k, knobs, N, fmt)
            hold, addk = (fl[k]["hold"], fl[k]["add"]) if res_t and tag.startswith("S") else (False, False)
            g0 = k  # the first pass of k's group
            while res_t and tag.startswith("S") and g0 > 0 and fl[g0 - 1]["partial"]:
                g0 -= 1
            ce = sw["col_ent"]
            pairs = _full(M[r], N * N, f"pass {tag} result {r}")
            pre = np.array([a for a, _ in pairs], dtype=np.int64)
            post = np.array([b for _, b in pairs], dtype=np.int64)
            # matmul: the sum of the group's passes as the host sent them, the bias with the first, a packed set's blocks alone
            passes = []
            if tag == "P0":  # matrix_west.mem; B from cache tile 0 when cached
                files = [("matrix_west.mem", "matrix_north_0.mem" if knobs["weight_cache"] else "matrix_north.mem")]
            elif tag == "B2B":
                files = [("matrix_west_1.mem", "matrix_north_1.mem")]
            else:
                files = [(f"matrix_west_{j}.mem", f"matrix_north_{j}.mem") for j in range(g0 if res_t else k - P + 1, k + 1)]
            for fw, fn in files:
                A, B = _tb_words(fw).reshape(N, N), _tb_words(fn).reshape(N, N)
                if sw["sh"]:
                    b = N >> sw["sh"]
                    B = np.where(np.kron(np.eye(N // b), np.ones((b, b))).astype(bool), B, 0)
                passes.append((A, B))
            bias = None
            first = k if tag in alone else g0 if res_t else k - P + 1  # apply_bias: the bias rides on a set k with k % P == 0
            if fl[first]["bias"] if res_t else knobs["has_bias"] and first % P == 0:
                bw = _tb_words(f"bias_{first}.mem")[:N]
                bias = bw if fmt != "bf16" else truth.encode("fp32", truth.values("bf16", bw))  # the TB widens the row exactly
            add("matmul", truth.gate_matmul(fmt, passes, pre.reshape(N, N), bias))
            if fmt == "bf16":
                add("narrow", truth.gate_narrow("bf16", pre, post))
            cols = np.arange(N * N) % N
            ent = ce[cols]
            if hold:  # a hold set's requantized beats go to hold_mem: requantize gated on them; no activation, pooling or dropout
                Rh = np.array(_full(g["R"].get(k, {}), N * N, f"pass {tag} set {k}'s hold beats"), dtype=np.int64)
                zp, lo, hi = (np.array([sw["ents"][int(e)][x] for e in ent]) for x in ("zp", "amin", "amax"))
                add("requant", truth.gate_requantize(pre, real_scales(k, sw, cols), zp, lo, hi, Rh))
                multipliers(k, sw)
                held[k] = Rh
                continue
            if addk:  # an add set: requantize (the identity) gives q2, hold_mem q1, the ADD y, the activation stage's input
                Q = _full(g["Q"].get(k, {}), N * N, f"pass {tag} set {k}'s ADD outputs")
                q1, q2, y = (np.array([v[i] for v in Q], dtype=np.int64) for i in range(3))
                if k - 1 not in held:
                    raise RuntimeError(f"truth dump: pass {tag} add set {k} has no hold set before it")
                add("transfer", _transfer("int8", held[k - 1], q1))  # hold_mem: each element its hold set's own beat
                aw = fl[k]["words"]
                r1, r2 = scales.get(str(k), {}).get("add") or [float(np.ldexp(aw[m], -aw["shift"])) for m in ("m1", "m2")]
                add("add", truth.gate_quantized_add(q1, q2, r1, aw["zp1"], r2, aw["zp2"], 1.0, aw["zpo"], aw["amin"], aw["amax"], y))
            X = np.array(_full(g["X"].get(k, {}), N * N, f"pass {tag} set {k}'s activation inputs"), dtype=np.int64)
            Aw = np.array(_full(g["A"].get(k, {}), N * N, f"pass {tag} set {k}'s activation bank"), dtype=np.int64)
            if fmt == "int8":
                zp, lo, hi = (np.array([sw["ents"][int(e)][x] for e in ent]) for x in ("zp", "amin", "amax"))
                add("requant", truth.gate_requantize(pre, real_scales(k, sw, cols), zp, lo, hi, q2 if addk else X))
                if addk:  # the lanes after an ADD: input zero point zp_o, the ADD's clamp, the drop value zp_o (P-3, D-5)
                    aw = fl[k]["words"]
                    sw["ents"] = {e: dict(w, zp=aw["zpo"], amin=aw["amin"], amax=aw["amax"]) for e, w in sw["ents"].items()}
                multipliers(k, sw)
            else:
                add("transfer", _transfer(fmt, post, X))
            # activation: each element through its column's entry, on the word the stage received
            q8 = _int8_quant(sw["ents"], scales, k) if fmt == "int8" else {}
            for e in sorted(set(ent.tolist())):
                func = sw["ents"][e]["act"]
                m = ent == e
                if fmt == "int8":
                    if e not in q8:
                        raise RuntimeError(f"truth_scales.json has no scales for set {k} entry {e}")
                    if func == "relu":  # sienna_top runs int8 ReLU as the requantize clamp: its activation stage is the identity
                        w = sw["ents"][e]
                        add("activation relu", truth.gate("int8", "linear", X[m], Aw[m], quant=(1.0, w["zp"], 1.0, w["zp"]),
                                                          bound=truth.TARGET)._replace(func="relu"))
                        if str(e) in scales.get(str(k), {}) and not (res_t and fl[k]["add"] and not addk):  # a calibrated ReLU entry: its
                            add("relu clamp", _relu_clamp(w, k, e))  # clamp must start at its zero point; an add set alone has the identity's
                        continue
                    add(f"activation {func}", truth.gate("int8", func, X[m], Aw[m], quant=q8[e], bound=_act_bound(func)))
                else:
                    add(f"activation {func}", truth.gate(fmt, func, X[m], Aw[m], bound=_act_bound(func)))
            Pw = _full(g["P"].get(k, {}), W, f"pass {tag} set {k}'s dropout inputs")
            Pv, keep = np.array([a for a, _ in Pw], dtype=np.int64), np.array([b for _, b in Pw], dtype=bool)
            add("maxpool", truth.gate_maxpool(fmt, Aw.reshape(N, N), Pv, ph, pw, pad))
            D = np.array(_full(g["D"].get(k, {}), W, f"pass {tag} set {k}'s outputs"), dtype=np.int64)
            zpw = np.array([_drop_value(sw["ents"][int(ce[w % N])]) for w in range(W)]) if fmt == "int8" else 0
            add("dropout", truth.gate_dropout(fmt, Pv, keep, D, p, bool(knobs["training_mode"]), zpw))
            model = dropout_keep(W, p, set_dropout_seed(knobs["dropout_seed"], k), lanes)
            if knobs["training_mode"] and (model != keep).any():
                res.setdefault("_mask", []).append(int((model != keep).sum()))
    return {k: (truth.merge(v) if not k.startswith("_") else v) for k, v in res.items()} | {"_npass": len(dump)}


def truth_report(res: dict) -> tuple:
    """(lines, misses) for one test: a line per stage, and a '[FAIL] truth:' line per stage that misses."""
    lines, bad = [], []
    for name in sorted((k for k in res if not k.startswith("_")), key=lambda x: (TRUTH_STAGES.index(x.split()[0]), x)):
        r = res[name]
        lines.append("        " + (truth.line(r, bound=_act_bound(r.func)) if name.startswith("activation") else truth.stage_line(r)))
        if not r.ok:
            w = {"matmul": lambda: f"{truth.worst_text(r.worst)} of sum|ab|", "narrow": lambda: f"{r.worst_lsb} ULP",
                 "requant": lambda: f"{r.worst:.3f} steps from the true value", "add": lambda: f"{r.worst:.3f} steps from the true value",
                 "scale": lambda: f"{r.worst:.3f} x 2^-24 relative",
                 "activation": lambda: truth.worst_text(r.worst) +
                 (f" ({r.worst_lsb} LSB)" if r.fmt == "int8" else "")}.get(name.split()[0], lambda: f"|out - want| {r.worst:.6g}")()
            where = ", ".join(sorted(res.get("_passes", {}).get(name, ())))
            bad.append(f"[FAIL] truth: {name.replace(' ', '/')} {w}, {r.over} of {r.n} over the bound in passes {where}; first: {r.first[:3]}")
    lines.append(f"        {res.get('_npass', 0)} passes gated (P0, B2B when built, {', '.join(TRUTH_STREAMS)}, and A unless packed)")
    if "_mask" in res:
        lines.append(f"        diagnostic: dropout's keep decisions differ from regression.py's LFSR model in {sum(res['_mask'])} windows")
    return lines, bad


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
    # bf16 only: -0 and subnormal bias words beside zero-product rows; TB_sienna_top compares the mesh's bias queue entry with the row widened.
    {"name": "bf16_bias_special_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "bias": True,
     "bias_special": True, "zero_rows": True, "pool_h": 1, "pool_w": 1, "padding": 0, "formats": ("bf16",)},
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
    # sienna-packing: shifts 1, 0, b = 2, b = N / 4 per set, an 8-entry table of activations rotating over the blocks
    {"name": "packed_mixed_act_nopool", "mode": "matmul", "matrix_type": "random", "act": "tanh", "pool_h": 1, "pool_w": 1,
     "padding": 0, "packed": True, "pack_acts": ["tanh", "relu", "selu", "linear", "sigmoid", "tanh", "relu", "selu"]},
    {"name": "packed_bias_cached_train_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "pool_h": 1,
     "pool_w": 1, "padding": 0, "packed": True, "bias": True, "cached": True, "training": True,
     "pack_acts": ["relu", "linear", "tanh", "sigmoid", "relu", "selu", "linear", "tanh"]},
    {"name": "packed_all_bypass_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "pool_h": 1, "pool_w": 1,
     "padding": 0, "packed": True, "pack_acts": ["relu", "linear"] * 4},  # every lane ReLU or linear: the bypass path, per lane
    {"name": "int8_packed_zp_random_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "pool_h": 1,
     "pool_w": 1, "padding": 0, "packed": True, "req_random": True, "zp_random": True, "formats": ("int8",),
     "pack_acts": ["linear", "relu", "tanh", "selu", "sigmoid", "linear", "relu", "tanh"]},
    # int8 residual ADD (sienna-int8-residual): a residual group's last depth pass is a hold set, then an add set (the identity pass) meets it in add_lanes.
    {"name": "int8_residual_one_pair", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual": True, "num_sets": 2,
     "req_random": True, "zp_random": True, "formats": ("int8",)},  # one hold/add pair per pass, the first residual test run: the end-of-pass hold check sees a stale hold first
    {"name": "int8_residual_relu", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual": True,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "residual": True,
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "residual": True, "accum_passes": 2,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_sigmoid_selu_nopool", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "residual": True,
     "mixed_acts": ["sigmoid", "selu"], "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "num_sets": 16,
     "tight_acts": ("sigmoid",), "formats": ("int8",)},  # SELU's output calibrates on the clamped sums, so only sigmoid shows its clamp
    {"name": "int8_residual_train", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual": True, "training": True,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_then_relu_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual_every": 2,
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_two_layers_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "residual": True,
     "add_layers": 2, "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_accum2_mixed_nopool", "mode": "matmul", "matrix_type": "random", "act": "relu", "accum_passes": 2,
     "residual_every": 2, "mixed_acts": ["relu", "tanh", "linear", "selu", "sigmoid"], "pool_h": 1, "pool_w": 1, "padding": 0,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
]


OUT_WORD = re.compile(r"Dropout -> Output \(Lane (\d+)\)\s*: dec=\S+\s+hex=([0-9a-fA-FxXzZ]+)")  # every L9 word TB_sienna_top prints, lane and word (x, z: 4-state)


def build_define(key_file: str) -> str:
    """The +define+FMT_<F> on a build's flag line (build_key.txt's last line), or none; unreadable if the build wrote no key."""
    try:
        last = open(key_file).read().rstrip("\n").splitlines()[-1]
    except (OSError, IndexError):
        return "unreadable"
    d = [w for w in last.split() if w.startswith("+define+FMT_")]
    return " ".join(d) if d else "none"


def build_key(pkg_text: str) -> str:
    """What a TB_sienna_top build depends on in a package: every line but the run-time knobs' (make's guard adds the inputs and flags)."""
    knob = re.compile(r"^\s*localparam\s+\w+\s+(?:" + "|".join(mr.TB_KNOBS) + r")\s*=")
    return "".join(x for x in pkg_text.splitlines(True) if not knob.match(x))


def build_sources() -> list:
    """Every SV file TB_sienna_top's build compiles (make -n lint's command line) and every include file in its -I directories."""
    r = subprocess.run(["make", "-s", "-n", "lint", "GEN_PKG=0"], cwd=ROOT, env=MAKE_ENV, capture_output=True, text=True)
    words = r.stdout.split()
    files = {w for w in words if w.endswith((".sv", ".svh", ".v")) and os.path.isfile(w)}
    for d in [w[2:] for w in words if w.startswith("-I")]:
        files |= {os.path.join(d, f) for f in os.listdir(d) if f.endswith((".svh", ".vh"))} if os.path.isdir(d) else set()
    return sorted(files)


def knob_references(files: list) -> list:
    """(file, line, knob) for each run-time knob a file names, but the package's own 'localparam int KNOB =' declarations."""
    decl = re.compile(r"^\s*localparam\s+\w+\s+(" + "|".join(mr.TB_KNOBS) + r")\s*=")
    out = []
    for f in files:
        for i, line in enumerate(open(f, errors="replace"), 1):
            m = decl.match(line) if os.path.basename(f) == "test_config_pkg.sv" else None
            out += [(f, i, k) for k in mr.TB_KNOBS if re.search(rf"\b{k}\b", line[m.end():] if m else line)]
    return out


def predicted_key(t: dict, N: int, T: int, lanes: int, fmt: str) -> str:
    """A test's build key from its configuration alone, to order the tests; the build decision reads the package the test wrote."""
    cfg = {"n": N, "tile_size": T, "lanes": lanes, "host_words": N, "fmt_name": fmt, **t}
    items = _config_items(cfg, fmt, "relu", 1, cfg.get("credits", SETS_IN_FLIGHT), 1, [], False, 0)
    return build_key("".join(f"localparam int {k} = {v};\n" for k, v, _ in items))


def build_order(tests: list, N: int, T: int, lanes: int, fmt: str) -> list:
    """The tests with each build group together, groups in order of first appearance, tests in order within a group."""
    keys = {}
    for t in tests:
        keys.setdefault(predicted_key(t, N, T, lanes, fmt), []).append(t)
    return [t for g in keys.values() for t in g]


def _run_make_live(log_path: str, fmt_name: str = "fp32", N: int = 16, T: int = 4, lanes: int = 32, build: bool = True) -> tuple:
    t0 = time.time()
    process = subprocess.Popen(
        ["make", mr.SIM_TARGET if build else "sim", f"FMT={fmt_name}", f"N={N}", f"TILE={T}", f"LANES={lanes}", f"COLLAPSE_K={mr.COLLAPSE_K}",
         "GEN_PKG=0", f"SIM={mr.SIM}"],  # GEN_PKG=0: build the package this test just wrote, in its format and geometry
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


def _parse_log(raw: str, test: str = "-") -> dict:
    fired = re.search(r'^.*(?:Assertion failed|%Error).*$|^(?:Error|Fatal): ".*$|^Error-\[.*$|^.*: started at .* failed at.*$', raw, re.M)  # Verilator's and VCS's forms
    if fired:  # a firing assertion leaves the exit code and the TB's counts untouched, so the log is the only witness
        return {"status": "ASSERT", "total": 0, "exact": 0, "tol": 0, "failed": 1, "cyc": 0, "fired": fired.group(0).strip()}
    if "FATAL" in raw or "[FATAL] Timeout" in raw:
        return {
            "status": "TIMEOUT",
            "total": 0,
            "exact": 0,
            "tol": 0,
            "failed": 1,
            "cyc": 0,
        }
    D = fourstate.DEC
    try:  # a 4-state simulator's x in a count fails the test by name
        ex = lambda p, what="count": fourstate.field(p, raw, test, what) or 0
        tot, exact, tol, fail = (
            ex(rf"Total\s+:\s+({D})"),
            ex(rf"Exact\s+:\s+({D})"),
            ex(rf"Tol pass\s+:\s+({D})"),
            ex(rf"Failed\s+:\s+({D})"),
        )
        cyc = ex(rf"asserted @ \d+\s+\(({D}) cycles\)", "cycle count")
    except fourstate.Unknown as e:
        return {"status": "FAIL", "total": 0, "exact": 0, "tol": 0, "failed": 1, "cyc": 0, "fired": fourstate.fail_line(e)}
    return {
        "status": "PASS" if fail == 0 and tot > 0 else "FAIL",
        "total": tot,
        "exact": exact,
        "tol": tol,
        "failed": fail,
        "cyc": cyc,
        "b2b": bool(re.search(r"^ BACK_TO_BACK: (?:PASSED|FAILED)", raw, re.M)),  # a -DBACK_TO_BACK build ran its back-to-back pass
    }


HOLD_ADD = re.compile(r"^\[HOLD/ADD\] hold sets done (\d+), add beats (\d+), add sets of (\d+) beats (\d+), add beats of other sets (\d+), "
                      r"hold done with an L3 grant (\d+), hold done with a bank done (\d+)", re.M)  # TB_sienna_top's line per stream pass


def _residual_pairs_bad(raw: str, per_lane: int) -> str:
    """A residual test's '[FAIL] residual: ...' line, or '': each stream pass's [HOLD/ADD] line must count a hold set done per add set,
    each add set with exactly per_lane add_lanes beats (want * per_lane in all), and none for another set."""
    k_sets = _tb_knobs()["num_sets"]  # the files past num_sets may be a longer earlier test's
    want = sum(int(open(os.path.join(TB_DIR, f"add_{k}.mem")).readline(), 16) >> 3 & 1 for k in range(k_sets))
    got = [tuple(int(x) for x in m[:5]) for m in HOLD_ADD.findall(raw)]
    if not got or any(g != (want, want * per_lane, per_lane, want, 0) for g in got):
        return (f"[FAIL] residual: {'; '.join('%d hold sets done, %d add beats, %d add sets of %d beats, %d of other sets' % (g[0], g[1], g[3], g[2], g[4]) for g in got) or 'no [HOLD/ADD] line'}, "
                f"expected {want}, {want * per_lane}, {want} of {per_lane} and 0 in each stream pass")
    return ""


RESIDUAL_CROSSES = ("hold done with an L3 grant", "hold done with a bank done")  # spec Risk 3's same-cycle crosses, each counted on its own
RESIDUAL_CROSS = re.compile(r"^\s*RESIDUAL CROSSES: hold done with an L3 grant (\d+), hold done with a bank done (\d+)", re.M)  # a point's line
RESIDUAL_CROSS_GATE = True  # the sweep needs each cross non-zero at some int8 pipeline point, or a ruling for every point where it is 0


def _residual_cross_line(counts, ran: int) -> str:
    """run_regression's RESIDUAL CROSSES line over a point's residual tests (RESIDUAL_CROSS parses it)."""
    return (f"  RESIDUAL CROSSES: {RESIDUAL_CROSSES[0]} {counts[0]}, {RESIDUAL_CROSSES[1]} {counts[1]} over {ran} residual tests"
            " (a 0: not reachable at this point)")
RESIDUAL_CROSS_UNREACHABLE = {  # cross (RESIDUAL_CROSSES) -> {point: ruling text, or "*" (every point): (guard assertion, ruling text)}
    "hold done with a bank done": {"*": ("a_hold_done_no_bank",
                                         "unreachable, checked every cycle by sienna_top's a_hold_done_no_bank (fires under m_t26_hdb_force): "
                                         "hold_done is a hold beat leaving requantize, which is then no rq_out beat; no add_lanes beat is in flight "
                                         "(the hold set's grant waited on rq_room, so add_lanes' busy_o was low); and a lane set's g_done needs its own beats "
                                         "through requantize after the hold's last (a_rq_one_set)")},
}
RTL_TOP = os.path.join(ROOT, "src", "pipeline", "sienna_pipe.sv")  # the pipeline, where the stage assertions live


def _ruling_ok(point: str, ruling, rtl: str) -> bool:
    """A point's ruling is its text; a "*" ruling (every point) is (guard, text), and sienna_pipe.sv must declare that exact guard."""
    if point != "*":
        return isinstance(ruling, str) and bool(ruling.strip())
    if not (isinstance(ruling, tuple) and len(ruling) == 2 and all(isinstance(x, str) for x in ruling)):
        return False
    guard, text = ruling
    if not text.strip() or not re.fullmatch(r"a_\w+", guard):
        return False
    return re.search(rf"^\s*{guard}\s*:\s*assert\s+property\b", rtl, re.M) is not None


def _residual_cross_sweep_bad(counts: dict, ruling: dict, rtl: str = None) -> list:
    """The sweep's residual-cross failures, counts {point: (L3 grant cross, bank done cross)}: each cross must be non-zero at some point,
    or every point where it is 0 must carry a ruling (ruling {cross: {point: text}}); a 0 at one point alone is not a failure."""
    bad, rtl = [], open(RTL_TOP).read() if rtl is None else rtl
    for i, c in enumerate(RESIDUAL_CROSSES):
        if any(v[i] for v in counts.values()):
            continue
        r = ruling.get(c, {})
        unruled = sorted(q for q in counts if not _ruling_ok(q, r.get(q, ""), rtl) and not _ruling_ok("*", r.get("*", ""), rtl))
        if unruled:
            bad.append(f"{c}: 0 at every point, and no unreachable ruling for {', '.join(unruled)}")
    return bad


def _residual_cross_report(results: str, points: list) -> tuple:
    """(lines, failures) of the residual crosses over a sweep's int8 pipeline points, from each point's log under results."""
    counts, lines = {}, []
    for q in points:
        if "fmt" not in SPEC.parse(q) or SPEC.parse(q)["fmt"] != "int8" or SPEC.parse(q)["kind"] not in PIPELINE_KINDS:  # selftest and the unit points have none
            continue
        log = os.path.join(results, q + ".log")
        m = RESIDUAL_CROSS.findall(regress.ANSI.sub("", open(log, errors="replace").read())) if os.path.exists(log) else []
        if m:
            counts[q] = tuple(int(x) for x in m[-1])
        lines.append(f"  {q}: " + (", ".join(f"{c} {v}" for c, v in zip(RESIDUAL_CROSSES, counts[q])) if m else "no RESIDUAL CROSSES line"))
    if not lines:
        if not RESIDUAL_CROSS_GATE:
            return ["RESIDUAL CROSSES: NOT CHECKED, the sweep gate is off (RESIDUAL_CROSS_GATE) and the sweep has no int8 pipeline point"], []
        if "int8" not in {SPEC.parse(q).get("fmt") for q in points}:  # FMT=fp32 or bf16: nothing residual runs
            return ["RESIDUAL CROSSES: NOT APPLICABLE, the sweep runs no int8 point"], []
        return ["RESIDUAL CROSSES: FAIL: no int8 pipeline point in a sweep with int8 points"], ["no int8 pipeline point"]
    if not RESIDUAL_CROSS_GATE:
        return ["RESIDUAL CROSSES per point:"] + lines + ["RESIDUAL CROSSES: NOT CHECKED, the sweep gate is off (RESIDUAL_CROSS_GATE)"], []
    bad = _residual_cross_sweep_bad(counts, RESIDUAL_CROSS_UNREACHABLE)
    bad += [f"{q}: no RESIDUAL CROSSES line" for q in (ln.split(":")[0].strip() for ln in lines) if q not in counts]
    return ["RESIDUAL CROSSES per point:"] + lines + [f"RESIDUAL CROSSES: {'FAIL: ' + '; '.join(bad) if bad else 'PASS'}"], bad


def summary_main(argv=None) -> None:
    """make regression's table (ArithmeticLibrary's regress.summary), then the residual crosses over the sweep's int8 pipeline points."""
    try:
        regress.main(SPEC, ["--action", "summary", *(argv or [])])
        rc = 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--results", default=SPEC.results)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--formats", default=None)
    a, _ = ap.parse_known_args(argv)
    res = os.path.abspath(a.results)
    lines, bad = _residual_cross_report(res, SPEC.points(a.small, regress._formats(SPEC, a.formats)))
    print("\n".join(lines))
    with open(os.path.join(res, "regression.log"), "a") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(1 if rc or bad else 0)


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
    want = {"N": N, "TILE_SIZE": T}  # the format is the build's define, not the package's
    bad = {k: got.get(k) for k in want if got.get(k) != want[k]}
    if bad:
        sys.exit(f"[ERROR] pkg: {path} has {bad}, expected {want}")
    print(f"PACKAGE OK: {fmt_name} {test} N={N} TILE={T} LANES={lanes} COLLAPSE_K={mr.COLLAPSE_K} -> {path}")


def run_regression(N: int, T: int, target_test: str = None, lanes: int = 32, host_words: int = None,
                   fmt_name: str = "fp32"):
    _check_dropout_generator()
    print(hdr(f"\n{'═'*70}\n  SIENNA PIPELINE — Regression Suite\n{'═'*70}"))
    tests_to_run = [t for t in PIPELINE_TESTS if fmt_name in t.get("formats", tuple(FORMATS))]  # int8-only tests skip the floats
    if mr.COLLAPSE_K == 0:  # the collapse-k 0 mesh refuses packed sets (a_pack_collapsed), as the mesh regression skips them
        tests_to_run = [t for t in tests_to_run if not t.get("packed")]

    if target_test:
        tests_to_run = [t for t in tests_to_run if target_test in t["name"]]
        if not tests_to_run:
            print(f"  {_R}[ERROR] No tests found containing '{target_test}' in {fmt_name}{_X}")
            sys.exit(1)
    if not tests_to_run:  # a run that simulates nothing is not a pass
        print(f"  {_R}[ERROR] No tests found for {fmt_name}{_X}")
        sys.exit(1)

    print(
        f"  Matrix size : {N}×{N}\n  Tile size   : {T}×{T}\n  Format      : {fmt_name}\n  Total tests : {len(tests_to_run)}\n"
        + hdr(f"{'═'*70}")
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    tests_to_run = build_order(tests_to_run, N, T, lanes, fmt_name)
    built = None  # the build key of the TB_sienna_top binary on disk, None until this run builds one
    res_cross, res_ran = [0, 0], 0  # residual tests' same-cycle crosses (RESIDUAL_CROSSES), each on its own, and the residual tests run

    for idx, t in enumerate(tests_to_run):
        print(
            f"\n  ║  [{idx+1}/{len(tests_to_run)}] {_O}{t['name']}{_X}  (generating...)"
        )

        # 1. Generate Vectors & Dump Expected Traces
        cfg = {"n": N, "tile_size": T, "lanes": lanes, "host_words": host_words or N, "fmt_name": fmt_name, **t}
        generate_vectors(cfg)

        # 2. Build TB_sienna_top when the package this test wrote needs another build, then run it (streams live status)
        key = build_key(open(os.path.join(TB_DIR, "test_config_pkg.sv")).read())
        log_path = os.path.join(RESULTS_DIR, f"{t['name']}.log")
        print(f"      {_D}{'build' if key != built else 'the build of the test before'}{_X}")
        if os.path.exists(TRUTH_DUMP):
            os.remove(TRUTH_DUMP)  # a stale dump would be gated as this test's
        raw_log, wall = _run_make_live(log_path, fmt_name, N, T, lanes, build=key != built)
        if key != built:  # the build's precision define, from the flag line make wrote into build_key.txt
            print(f"      BUILD DEFINE: {build_define(os.path.join(mr.SIM_DIR, 'TB_sienna_top_sim.build_key.txt' if mr.SIM == 'vcs' else 'build_key.txt'))}")
        built = key if "-- Done" in raw_log or "-- Running simulation" in raw_log else None  # a failed build leaves no binary to reuse
        w = OUT_WORD.findall(raw_log)
        print(f"      WORDS {t['name']}: {len(w)} outputs, sha1 {hashlib.sha1(' '.join(f'{a}:{b}' for a, b in w).encode()).hexdigest()}")
        xw = next((k for k, (_, b) in enumerate(w) if fourstate.XZ.search(b)), None)  # the hash covers the words as printed; an unknown one fails the test
        if xw is not None:
            raw_log += "\n" + fourstate.fail_line(fourstate.Unknown(f"{'X' if re.search('[xX]', w[xw][1]) else 'Z'} in output word {xw} of test {t['name']} ({w[xw][1]!r})")) + "\n"

        # 3. Parse Log
        r = {
            "name": t["name"],
            "mode": t["mode"],
            "act": t["act"],
            "wall": wall,
            **_parse_log(raw_log, t["name"]),
        }
        if "fired" in r and r["fired"].startswith("FAIL: "):
            print(f"      {err(r['fired'])}")
        elif re.search(r"^FAIL: [XZ] in output word", raw_log, re.M):
            print(f"      {err(raw_log.rstrip().splitlines()[-1])}")
            r.update(status="FAIL", failed=max(r["failed"], 1))
        if t.get("residual") or t.get("residual_every"):  # the hold sets and add beats each stream pass counted
            bad = _residual_pairs_bad(raw_log, N * N // lanes)
            res_ran += 1
            for m in HOLD_ADD.findall(raw_log):
                res_cross[0], res_cross[1] = res_cross[0] + int(m[5]), res_cross[1] + int(m[6])
            if bad:
                print(f"      {err(bad)}")
                r.update(status="FAIL" if r["status"] == "PASS" else r["status"], failed=max(r["failed"], 1))
        results.append(r)

        if os.environ.get("SIENNA_KEEP_TRUTH") and os.path.exists(TRUTH_DUMP):  # comparison runs: the dump and each set's per-column activations
            shutil.copy(TRUTH_DUMP, os.path.join(RESULTS_DIR, f"{t['name']}_truth_dump.txt"))
            kn = _tb_knobs()
            with open(os.path.join(RESULTS_DIR, f"{t['name']}_acts.txt"), "w") as fh:
                ph, pw, pad = t.get("pool_h", 2), t.get("pool_w", 2), t.get("padding", 1)  # the pool geometry, so a comparison can map P words to columns
                fh.write(f"G {ph} {pw} {pad} {lanes} {(N + 2 * pad - pw) // pw + 1}\n")
                for k in range(kn["num_sets"]):
                    sw = _set_words(k, kn, N, fmt_name)
                    fh.write(f"{k} " + " ".join(sw["ents"][int(sw["col_ent"][c])]["act"] for c in range(N)) + "\n")

        # 4. Dump Formatted HW Matrix Trace
        dump_hardware_trace(t["name"], N, print_to_console=(r["status"] != "PASS"))

        # 5. Print Final Result: the TB's compare (bit-exact; fp32 within tolerance) is a diagnostic, still required
        sym = ok("PASS") if r["status"] == "PASS" else err("FAIL")
        tot = max(r["total"], 1)
        print(
            f"      {sym}   exact {100*r['exact']/tot:5.1f}%  tol {100*r['tol']/tot:5.1f}%  fail {r['failed']:3d}  {r['cyc']:6d} cyc  {r['wall']:5.1f}s"
            f"  (TB compare, diagnostic)"
        )

        # 6. True-function gates per stage, on the TB's truth dump; a miss fails the test but the sweep goes on
        if r["status"] in ("PASS", "FAIL"):
            try:
                r["truth"] = truth_test(N, fmt_name, lanes)
                tl, tbad = truth_report(r["truth"])
            except fourstate.Unknown as e:
                tl, tbad = [], [f"[FAIL] truth: none ({fourstate.fail_line(e)} {t['name']})"]
            except (RuntimeError, ValueError, OSError, KeyError) as e:
                tl, tbad = [], [f"[FAIL] truth: none ({e})"]
            print("      true-function gates (truth.py), every stage on its actual inputs:")
            for ln in tl:
                print(f"      {ln.strip()}")
            for ln in tbad:
                print(f"      {err(ln)}")
            r["truth_ok"] = not tbad
            if tbad and r["status"] == "PASS":
                r["status"] = "FAIL"
                continue

        if r["status"] != "PASS":
            print(f"  {_R}╚══  Sweep Aborted: {r['status']}.{_X}" + (f"\n      {r['fired']}" if "fired" in r else ""))
            sys.exit(1)

    print(f"\n  ╚══  Sweep Complete")
    if res_ran:  # characterization: a 0 is a cross not reachable at this point; the sweep gates them (summary_main)
        print(_residual_cross_line(res_cross, res_ran))

    gated = [r for r in results if "truth" in r]
    names = sorted({k for r in gated for k in r["truth"] if not k.startswith("_")},
                   key=lambda x: (TRUTH_STAGES.index(x.split()[0]), x))
    print(hdr(f"\n  TRUE-FUNCTION GATES, all tests ({fmt_name}, N={N}, TILE={T})"))
    for name in names:
        m = truth.merge([r["truth"].get(name) for r in gated])
        print("  " + (truth.line(m, bound=_act_bound(m.func)) if name.startswith("activation") else truth.stage_line(m)))
        for r in gated:
            x = r["truth"].get(name)
            if x is not None and not x.ok:
                print(f"    miss: {r['name']}: {x.over} of {x.n}")
    print(f"  TRUTH: {len(gated)} tests gated, {sum(not r.get('truth_ok', False) for r in gated)} with misses")
    print(f"  BACK_TO_BACK: {sum(r.get('b2b', False) for r in results)} of {len(results)} tests ran the back-to-back pass")

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


# ── pack (was pack_regression.py) ────────────────────────────────────────────

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


def float_layer_over(A_q, z_a, s_a, B_q, s_w, bias, s_out, zp, relu, Y) -> tuple:
    """(over, n) of int8 outputs Y against a float64 reference of the whole layer from its real numbers (s_a (A_q - z_a), s_w B_q and the
    bias rounded to s_a s_w), independent of fold_bias: |Y - t| <= max(1% |t|, one step), t in codes clamped to -128..127."""
    s_w = np.asarray(s_w, np.float64)
    b_real = np.rint(np.asarray(bias, np.float64) / (s_a * s_w)) * s_a * s_w
    t = (s_a * (np.asarray(A_q, np.float64) - z_a)) @ (s_w[None, :] * np.asarray(B_q, np.float64)) + b_real[None, :]
    if relu:
        t = np.maximum(t, 0.0)
    tc = np.clip(t / s_out + zp, -128, 127)
    return int(np.sum(np.abs(np.asarray(Y, np.float64) - tc) > np.maximum(truth.TARGET * np.abs(tc - zp), 1.0))), int(tc.size)


def run_case(a, sim, N, sh, R, seed, log, zero_rows=False):
    rng = np.random.RandomState(seed)
    A, B, bias, mp, b = layer(N, sh, R, rng, a.fmt, zero_rows)
    col_ent = [mp[j // b] for j in range(N)]
    ents = [(a.acts[e], None) for e in range(8)]
    if a.fmt == "int8":
        A_q, s_a, z_a = mr.quant_act(A)
        B_q, s_w = mr.quant_weights(B)
        hw = mr.fold_bias(bias, s_a, s_w, z_a, B_q)
        acc = mr.wrap32(mr.imatmul(A_q, B_q) + hw[None, :])
        mult, shift, ratio = np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N)
        for e in sorted(set(col_ent)):
            cols = np.array(col_ent) == e
            act = "linear" if a.acts[e] == "selu" else a.acts[e]  # SELU's int8 saturation is pack_jobs' to refuse (Task 7)
            rq = mr.requant_params(acc[:, cols], s_a, s_w[cols], act)
            mult[cols], shift[cols], ratio[cols] = rq["mult"], rq["shift"], rq["ratio"]
            ents[e] = (act, rq)
        head = ents[0][1] or mr.requant_params(acc, s_a, s_w, "linear")
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw, "act": ents[0][0], "shape": (R * N, N),
               "req": dict(head, mult=mult, shift=shift, ratio=list(ratio)), "pack": {"shift": sh, "map": mp, "ents": ents}}
    else:
        A, B, bias = (mr.op_round(v, a.fmt) for v in (A, B, bias))
        job = {"terms": [(A, B)], "bias": bias, "act": a.acts[0], "shape": (R * N, N), "pack": {"shift": sh, "map": mp, "ents": ents}}
    t0 = time.time()
    Yp, sets_p, cyc_p = sim.run_job(job, f"pk_s{sh}")
    bad = cyc_a = 0
    gated = over = f_over = n_exact = 0  # int8: outputs gated per stage on the true function, those over the rule, over the float64 layer, off the bit-exact model
    if a.fmt == "int8":  # every entry per stage (requantize, then its lane) from the packed layer's dump
        for st, r in mr.layer_stage_gates(job, Yp, sim.last_stages, N):
            gated, over = gated + (r.n if st.startswith("activation") else 0), over + r.over
            bad += r.over + int(r.n == 0)
            if not r.ok:
                print(f"  FAIL {st}: {truth.stage_line(r)}", flush=True)
    for c in range(N // b):
        cols = slice(c * b, (c + 1) * b)
        e = mp[c]
        if a.fmt == "int8":
            rq = dict(ents[e][1], mult=mult[cols], shift=shift[cols])
            alone = {"terms": [(A_q[:, cols].astype(np.float32), B_q[cols, cols].astype(np.float32))], "bias": hw[cols],
                     "act": ents[e][0], "shape": (R * N, b), "req": rq}
            n_exact += int(np.sum(Yp[:, cols] != mr.int8_layer_exact(A_q[:, cols], B_q[cols, cols], hw[cols], rq, ents[e][0])))  # reported
            gold = None
            if ents[e][0] in ("linear", "relu"):  # the whole layer from its real numbers: a zero-point fold bug shared by host and golden shows here
                fo, fn = float_layer_over(A_q[:, cols], z_a, s_a, B_q[cols, cols], s_w[cols], bias[cols], rq["s_out"], rq["zp"],
                                          ents[e][0] == "relu", Yp[:, cols])
                f_over, bad = f_over + fo, bad + fo + int(fn == 0)
                if fo:
                    print(f"  FAIL block {c} ({ents[e][0]}): {fo} of {fn} outputs off the float64 layer", flush=True)
        else:
            alone = {"terms": [(A[:, cols], B[cols, cols])], "bias": bias[cols], "act": a.acts[e], "shape": (R * N, b)}
            gold = exact_layer(A[:, cols], B[cols, cols], bias[cols] if np.any(bias[cols]) else None, a.acts[e], N, a.fmt, a.tile) if a.fmt == "bf16" else None  # fp32: RTL against RTL (F-GP1)
        Ya, _, cyc = sim.run_job(alone, f"al_s{sh}_{c}")
        cyc_a += cyc
        bits = (lambda y: y) if a.fmt == "int8" else (lambda y: mr.fmt_bits(y, a.fmt))
        bad += int(np.sum(bits(Yp[:, cols]) != bits(Ya)))
        if gold is not None:
            bad += int(np.sum(bits(Yp[:, cols]) != gold))
    line = (f"{a.fmt} N={N} T={a.tile} b={b:<3} rows={R * N:<4} {'zero rows ' if zero_rows else ''}packed: {sets_p} sets "
            f"{cyc_p} cycles | {N // b} jobs alone: {cyc_a} cycles | speedup {cyc_a / cyc_p:5.2f}x | mismatches {bad} | "
            + (f"true function per stage: {gated} outputs, {over} over the rule; float64 layer (linear/ReLU): {f_over} over; "
               f"reported: {n_exact} off the bit-exact model | " if a.fmt == "int8" else "")
            + f"wall {time.time() - t0:.0f}s")
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
            m["W"], m["bias"] = mr.op_round(m["W"], a.fmt), mr.op_round(m["bias"], a.fmt)
            m["inputs"] = [mr.op_round(x, a.fmt) for x in m["inputs"]]
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
        bad += int(np.sum(mr.fmt_bits(got, a.fmt) != mr.fmt_bits(Ya, a.fmt)))
    line = f"{a.fmt} N={N} T={a.tile} pack_jobs: {len(models)} models in {N // (N >> job['pack']['shift'])} blocks, mismatches {bad}"
    print(line, flush=True)
    log.write(line + "\n")
    return bad


def pack_main(argv=None):
    """Packed layers on sienna_top's layer contract: each job's block equals the job alone on the RTL and its golden, bit for bit; cycles packed vs alone."""
    ap = argparse.ArgumentParser(description=pack_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="pack")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--rows", type=int, nargs="+", default=[2], help="row tiles per packed layer; several run in turn")
    ap.add_argument("--format", dest="fmt", default="int8", choices=sorted(mr.FORMATS))
    ap.add_argument("--act", choices=sorted(set(ACTS)), help="every entry holds this activation (a same-activation cycle sweep); default the mixed table")
    a = ap.parse_args(argv)
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


# ── gemm (was gemm_sweep.py) ─────────────────────────────────────────────────

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


def run_int8(a, sim, shapes) -> None:
    """int8 on the layer engine, affinely quantized products: every case per stage on the true function from TB_model_run's dump
    (mr.layer_stage_gates: the requantize of each exact int32 sum, then the lane); the bit-exact model and the quantization error against the
    float product are reported, not gated."""
    rep = open(os.path.join(a.work, f"gemm_sweep_N{a.n}.log"), "w")
    peak = a.n * a.n
    head = (f"{'shape':<22} {'M':>5} {'K':>5} {'N':>5} {'sets':>7} {'cycles':>10} {'MAC/cycle':>9} {'PE use':>7} "
            f"{'slot use':>8} {'mism':>6} {'q err':>8} {'wall s':>6}")
    for line in (f"GEMM sweep on the RTL, mesh N={a.n} for collapse-k {mr.COLLAPSE_K} (each layer checks the mesh's), {a.lanes} lanes, int8 operands, int32 sums, requantized output; "
                 f"peak {peak} MAC/cycle", head):
        print(line, flush=True)
        rep.write(line + "\n")
    cases = [(name, m, k, n, "linear", {}) for name, m, k, n in shapes]
    cases += [(f"layer_{act}_bias", 64, 48, 40, act, {}) for act in ("tanh", "sigmoid", "selu")]
    # the lanes' tier field and load path (lane_cases): tier 1, the loaded swapped image after reset and between two runs, writes while busy
    cases += [(t, 32, 48, 24, "tanh", dict(e, tol=tol)) for t, e, tol in lane_cases("int8", True)]
    rows, bad, over, all_mism = [], 0, 0, 0
    for name, m, k, n, act, extra in cases:
        rng = np.random.RandomState(m * 7 + k * 13 + n)
        A, B = rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))
        bias = None if act == "linear" else rng.uniform(-0.5, 0.5, n)
        A_q, s_a, z_a = mr.quant_act(A)
        B_q, s_w = mr.quant_weights(B)
        hw_bias = mr.fold_bias(bias, s_a, s_w, z_a, B_q)
        req = mr.requant_params(mr.wrap32(mr.imatmul(A_q, B_q) + hw_bias[None, :]), s_a, s_w, act)
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw_bias, "act": act,
               "shape": (m, n), "req": req, **extra}
        t0 = time.time()
        y, sets, cyc = sim.run_job(job, name)
        img = extra.get("coef", extra.get("between"))
        model = mr.int8_layer_exact(A_q, B_q, hw_bias, req, act, extra.get("tier", 0), img)
        mism = int(np.sum(y != model))
        if extra:  # the case must exercise the field: its model's words are not the reset image's tier 0 (the refused case's are)
            dflt = mr.int8_layer_exact(A_q, B_q, hw_bias, req, act)
            differ = int(np.sum(model != dflt))
            note = ""
            if "between" in extra:
                fm = int(np.sum(sim.first_Y != dflt))
                note = f"; its first run {fm} outputs off the reset image's; loaded {sim.between_gap} cycles after done_o"
                if fm:
                    print(f"FAIL {name}: the run before the load differs from the reset image in {fm} outputs", flush=True)
                    bad += 1
                if sim.between_gap is None:
                    print(f"FAIL {name}: the testbench did not report when the load began", flush=True)
                    bad += 1
            if extra.get("coef_busy"):
                t, rf = sim.busy_refused or (0, 0)
                note = f"; {rf} of {t} writes while busy refused"
                if not (t == 4 and rf == 4):
                    print(f"FAIL {name}: {rf} of {t} writes while busy refused", flush=True)
                    bad += 1
            print(f"    {name}: {differ} outputs of the model differ from tier 0 of the reset image{note}", flush=True)
            rep.write(f"    {name}: {differ} outputs of the model differ from tier 0 of the reset image{note}\n")
            if (differ == 0) != bool(extra.get("coef_busy")):
                print(f"FAIL {name}: the case does not exercise the tier or the loaded image", flush=True)
                bad += 1
            if mism:  # a lane that ignored its tier or image would pass the true function at a stricter tier: these gate bit for bit too
                print(f"FAIL {name}: {mism} outputs differ from the bit-exact model of tier {extra.get('tier', 0)}{' of the loaded image' if img is not None else ''}", flush=True)
                bad += 1
        ref = A @ B + (0.0 if bias is None else bias[None, :])
        if act != "linear":
            ref = mr.apply_activation(ref.astype(np.float32), act).astype(np.float64)
        if act == "tanh":
            deq = y / 128.0  # D-4: tanh y * 128, zero point 0
        elif act == "sigmoid":
            deq = (y + 128) / 256.0  # sigmoid y * 256, zero point -128
        elif act == "selu":
            deq = (y - req["zout"]) * req["s_selu"]
        else:
            deq = (y - req["zp"]) * req["s_out"]
        err = float(np.max(np.abs(deq - ref)) / (np.max(np.abs(ref)) or 1.0))
        macs = m * k * n
        r = {"shape": name, "M": m, "K": k, "N": n, "act": act, "sets": sets, "cycles": cyc, "macs": macs,
             "mac_per_cycle": macs / cyc if cyc else 0, "pe_use": macs / (cyc * peak) if cyc else 0,
             "slot_use": macs / (sets * a.n ** 3), "mism": mism, "err": err, "wall": time.time() - t0}
        rows.append(r)
        line = (f"{name:<22} {m:>5} {k:>5} {n:>5} {sets:>7} {cyc:>10} {r['mac_per_cycle']:>9.1f} {100 * r['pe_use']:>6.1f}% "
                f"{100 * r['slot_use']:>7.1f}% {mism:>6} {err:>8.1e} {r['wall']:>6.0f}")
        print(line, flush=True)
        rep.write(line + "\n")
        rep.flush()
        json.dump(rows, open(os.path.join(a.work, f"gemm_sweep_N{a.n}.json"), "w"), indent=1)
        all_mism += mism
        gates = mr.layer_stage_gates(job, y, sim.last_stages, a.n)
        over += sum(r.over for _, r in gates)
        fails = [(st, r) for st, r in gates if not r.ok]
        for st, r in gates:  # each stage's worst
            sl = f"    {st}: {truth.line(r, bound=extra.get('tol', _act_bound(r.func))) if st.startswith('activation') else truth.stage_line(r)}"
            print(sl, flush=True)
            rep.write(sl + "\n")
        for st, r in fails:
            print(f"FAIL {name}: {st}: {truth.stage_line(r)}", flush=True)
        bad += int(bool(fails))
    line = (f"GEMM int8: {len(rows)} cases, {bad} failing (every case per stage on the true function, requantize and lane, {over} outputs "
            f"over the rule); reported: {all_mism} outputs differ from the bit-exact model")
    print(line, flush=True)
    rep.write(line + "\n")
    if bad:
        sys.exit(1)


def gemm_main(argv=None):
    """Model-agnostic benchmark: C = A (M x K) @ B (K x N) on the RTL over a grid of shapes, plus transformer-sized shapes."""
    ap = argparse.ArgumentParser(description=gemm_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="gemm")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4, help="mesh tile size T the RTL is built with")
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "gemm"))
    ap.add_argument("--emulate", action="store_true", help="numpy stand-in for the RTL")
    ap.add_argument("--quick", action="store_true", help="a few small shapes only")
    ap.add_argument("--engine", choices=("layer", "sets"), default="layer",
                    help="layer: sienna_top's layer contract (sienna_sched) schedules everything; sets: the host drives sienna_top set by set")
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(mr.FORMATS),
                    help="format of A and B on the layer engine; int8 sums in int32 and requantizes")
    ap.add_argument("--host-gaps", action="store_true", help="host idles a cycle after each load and waits for the credit")
    ap.add_argument("--collapse-k", type=int, choices=(0, 1), default=1, help="the mesh's COLLAPSE_K the layer engine builds and the float goldens follow")
    a = ap.parse_args(argv)
    if a.engine == "sets" and a.collapse_k == 0:
        ap.error("--engine sets builds collapse-k 1 (TB_sienna_model); use --engine layer")
    mr.COLLAPSE_K = a.collapse_k
    if a.fmt_name == "int8" and (a.emulate or a.engine != "layer"):
        ap.error("int8 runs on the layer engine only")
    os.makedirs(a.work, exist_ok=True)
    if a.emulate:
        sim = mr.Emulator(a.n, a.lanes, a.work, tile_size=a.tile)
    elif a.engine == "layer":
        sim = mr.RtlLayer(a.n, a.lanes, a.work, a.fmt_name, a.tile)
    else:
        sim = mr.RtlSets(a.n, a.lanes, a.work, a.host_gaps, a.tile)
    sim.build()
    shapes = [(f"grid_{m}x{k}x{n}", m, k, n) for m in GRID_M for k in GRID_K for n in GRID_N] + TRANSFORMER
    if a.quick:
        shapes = [s for s in shapes if s[1] * s[2] * s[3] <= 64 * 64 * 64][:6]
    if a.fmt_name == "int8":
        run_int8(a, sim, shapes)
        return
    rep = open(os.path.join(a.work, f"gemm_sweep_N{a.n}.log"), "w")
    peak = a.n * a.n  # collapse-k mesh: N^2 PEs, one product per PE per cycle at best
    head = f"{'shape':<22} {'M':>5} {'K':>5} {'N':>5} {'sets':>7} {'cycles':>10} {'MAC/cycle':>9} {'PE use':>7} {'slot use':>8} {'max err':>8} {'wall s':>6}"
    for line in (f"GEMM sweep on the RTL, mesh N={a.n} for collapse-k {mr.COLLAPSE_K} (each layer checks the mesh's), {a.lanes} lanes, linear activation, {a.fmt_name} operands and results, fp32 sums; "
                 f"peak {peak} MAC/cycle", head):
        print(line, flush=True)
        rep.write(line + "\n")
    rows = []
    for name, m, k, n in shapes:
        rng = np.random.RandomState(m * 7 + k * 13 + n)
        A = mr.op_round(rng.uniform(-1, 1, (m, k)), a.fmt_name)
        B = mr.op_round(rng.uniform(-1, 1, (k, n)), a.fmt_name)
        job = {"terms": [(A, B)], "bias": None, "act": "linear", "shape": (m, n)}
        t0 = time.time()
        y, sets, cyc = sim.run_job(job, name)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = float(np.max(np.abs(y - ref)) / (np.max(np.abs(ref)) or 1.0))
        mism = 0
        if a.fmt_name != "fp32" and isinstance(sim, mr.RtlLayer):  # narrow formats: every output bit-exact
            mism = int(np.sum(mr.fmt_bits(y, a.fmt_name) != exact_layer(A, B, None, "linear", a.n, a.fmt_name, a.tile)))
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
        if not err <= mr.LAYER_BOUND[a.fmt_name]:  # the model gate's layer bound in this format, one constant; NaN fails
            print(f"FAIL {name}: error {err:.2e} above {mr.LAYER_BOUND[a.fmt_name]:g}", flush=True)
            sys.exit(1)
        if a.fmt_name != "fp32" and mism:
            print(f"FAIL {name}: {mism} outputs differ from the bit-exact model", flush=True)
            sys.exit(1)
    if isinstance(sim, mr.RtlLayer):
        # The polynomial activations through the layer engine, with a bias; GPNAE approximates within about 2%.
        rng = np.random.RandomState(5)
        A = mr.op_round(rng.uniform(-1, 1, (64, 48)), a.fmt_name)
        B = mr.op_round(rng.uniform(-0.3, 0.3, (48, 40)), a.fmt_name)
        b = mr.op_round(rng.uniform(-0.5, 0.5, 40), a.fmt_name)
        for act in ("tanh", "sigmoid", "selu"):
            y, sets, cyc = sim.run_job({"terms": [(A, B)], "bias": b, "act": act, "shape": (64, 40)}, f"act_{act}")
            ref = mr.apply_activation((A.astype(np.float64) @ B + b).astype(np.float32), act)
            err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
            line = f"layer_{act}_bias{'':<9} {64:>5} {48:>5} {40:>5} {sets:>7} {cyc:>10}  max err {err:.1e} of the output range"
            print(line, flush=True)
            rep.write(line + "\n")
            if a.fmt_name == "fp32" and err > 3e-2:
                print(f"FAIL layer_{act}: error {err:.2e} above 3e-2", flush=True)
                sys.exit(1)
            if a.fmt_name != "fp32":  # narrow formats: bit-exact against the model; the error above is reported, not gated
                mism = int(np.sum(mr.fmt_bits(y, a.fmt_name) != exact_layer(A, B, b, act, a.n, a.fmt_name, a.tile)))
                print(f"  layer_{act}: {mism} outputs differ from the bit-exact model", flush=True)
                rep.write(f"  layer_{act}: {mism} outputs differ from the bit-exact model\n")
                if mism:
                    sys.exit(1)
        if a.fmt_name == "bf16" and not layer_bias_special(a, sim, rep):
            sys.exit(1)
        if not lane_features(a, sim, rep):
            sys.exit(1)
        if a.fmt_name == "fp32" and a.engine == "layer" and not a.emulate and a.tile == 2 and mr.COLLAPSE_K == 1:  # the set contract's tier and load path, once per N
            sets = mr.RtlSets(a.n, a.lanes, a.work, False, a.tile)
            sets.build()
            if not lane_features(a, sets, rep, "sets "):
                sys.exit(1)


LANE_CASES = {"lanes_tier1": 1, "lanes_loaded": 0, "lanes_between": 0, "lanes_refused": 0}  # the gemm points' tier and load runs and their tiers
SET_LANE_CASES = ("lanes_tier1", "lanes_loaded", "lanes_refused")  # those the set contract runs (no between load there)


def lane_case_line(tag: str, tier: int) -> str:
    """The gemm verdict's pattern for one float tier or load run: bit-exact at its tier and within its tolerance of the true function; whether
    it moved off the reset image's tier 0 (or, refused, did not) is the run's own FAIL line."""
    return rf"^{tag} +\d+ +\d+ +\d+ +\d+ +\d+  0 outputs differ from the bit-exact model of tier {tier}(?: of the loaded image)?; \d+ differ from tier 0 of the reset image; true function PASS "


def swapped_image(fmt: str):
    """The lanes' reset image with tiers 0 and 2 swapped (descriptors and tail sets): loaded, tier 0 runs the 10% design."""
    m = gpnae_model.default_mem(fmt)
    m.desc[0], m.desc[2] = m.desc[2], m.desc[0]
    m.tail[0], m.tail[2] = m.tail[2], m.tail[0]
    return m


def lane_cases(fmt: str, layer: bool) -> list:
    """The lanes' tier and load cases: (tag, the job's extra keys, the tolerance its words are gated at, its expected tier label). tier 1; tier 0
    of the swapped image loaded after reset (it holds the 10% design); on the layer contract also that image loaded between two runs of the
    layer with no reset; and writes to the running activation's own words while it streams (busy_writes), which the interlock must refuse."""
    tol = gpnae_model.TIER_TOL
    out = [("lanes_tier1", {"tier": 1}, tol[1]), ("lanes_loaded", {"tier": 0, "coef": swapped_image(fmt)}, tol[2])]
    if layer:
        out += [("lanes_between", {"tier": 0, "between": swapped_image(fmt)}, tol[2])]
    return out + [("lanes_refused", {"tier": 0, "coef_busy": True}, tol[0])]


def lane_features(a, sim, rep, contract: str = "") -> bool:
    """The lanes' tier field and coefficient load path on sim's contract (the layer's; "sets " the set contract's), on a tanh layer (lane_cases):
    each run bit-exact against the model of its tier and image and within its tier's tolerance of the true function; the tier and load runs
    must differ from the reset image's tier 0, the between run's first pass must equal it, and every write while busy must be refused."""
    rng = np.random.RandomState(6)
    A = mr.op_round(rng.uniform(-1, 1, (32, 48)), a.fmt_name)
    B = mr.op_round(rng.uniform(-0.6, 0.6, (48, 24)), a.fmt_name)
    b = mr.op_round(rng.uniform(-0.5, 0.5, 24), a.fmt_name)
    good = True
    base = {"terms": [(A, B)], "bias": b, "act": "tanh", "shape": (32, 24)}
    dflt = np.asarray(sim.exact(dict(base, tier=0)))
    pre = np.asarray(sim.exact(dict(base, act="linear")))  # the lanes' inputs: the layer's words before the activation
    for tag, extra, tol in lane_cases(a.fmt_name, isinstance(sim, mr.RtlLayer)):
        job = dict(base, **extra)
        y, sets, cyc = sim.run_job(job, tag)
        img = {"tier": job["tier"]}
        if "coef" in job or "between" in job:
            img["coef"] = job.get("coef", job.get("between"))
        want = np.asarray(sim.exact(dict(base, **img)))
        words = mr.word_bits(y, a.fmt_name)
        mism = int(np.sum(words != want))
        differ = int(np.sum(want != dflt))
        r = truth.gate(a.fmt_name, "tanh", pre.ravel(), np.asarray(words).ravel(), bound=tol)
        ok = mism == 0 and r.ok and (differ > 0 if tag != "lanes_refused" else differ == 0)
        note = ""
        if tag == "lanes_between":  # the first run, before the load, is the reset image's
            fm = int(np.sum(mr.word_bits(sim.first_Y, a.fmt_name) != dflt))
            ok &= fm == 0 and sim.between_gap is not None
            note = f"; its first run {fm} outputs off the reset image's; loaded {sim.between_gap} cycles after done_o"
        if tag == "lanes_refused":
            t, rf = sim.busy_refused or (0, 0)
            ok &= t == 4 and rf == 4
            note = f"; {rf} of {t} writes while busy refused"
        img_txt = " of the loaded image" if "coef" in img else ""
        line = (f"{contract + tag:<22} {32:>5} {48:>5} {24:>5} {sets:>7} {cyc:>10}  {mism} outputs differ from the bit-exact model of tier {job['tier']}"
                f"{img_txt}; {differ} differ from tier 0 of the reset image; true function {'PASS' if r.ok else 'FAIL'} worst "
                f"{truth.worst_text(r.worst)} at {100 * tol:g}%{note}")
        print(line, flush=True)
        rep.write(line + "\n")
        if not ok:
            print(f"FAIL {contract + tag}: {line.split('  ', 1)[1]}", flush=True)
        good &= ok
    return good


def layer_bias_special(a, sim, rep) -> bool:
    """One bf16 layer whose bias row holds +0, -0 and subnormals (_special_bias): bit-exact, and sienna_sched's a_bias_widened, the only check that sees them, must not fire."""
    rng = np.random.RandomState(4711)
    A = mr.op_round(rng.uniform(-1, 1, (32, 40)), a.fmt_name)
    B = mr.op_round(rng.uniform(-1, 1, (40, 24)), a.fmt_name)
    words = _special_bias(mr.fmt_bits(mr.op_round(rng.uniform(-0.5, 0.5, 24), a.fmt_name), a.fmt_name), a.fmt_name)
    job = {"terms": [(A, B)], "bias": mr.bits_float(words, a.fmt_name), "act": "linear", "shape": (32, 24), "bias_words": words}
    try:
        y, sets, cyc = sim.run_job(job, "bias_special")
    except RuntimeError as e:  # a firing assertion or a TB check
        line, good = f"FAIL layer_bias_special: {e}", False
    else:
        mism = int(np.sum(mr.word_bits(y, a.fmt_name) != sim.exact(job)))
        line, good = f"layer_bias_special{'':<4} {32:>5} {40:>5} {24:>5} {sets:>7} {cyc:>10}  {mism} outputs differ from the bit-exact model", mism == 0
    print(line, flush=True)
    rep.write(line + "\n")
    return good


# ── residual: int8 ADD layers on sienna_top's layer contract (sienna-int8-residual) ─────────

RES_CASES = {  # name: N -> (rows M, depth K, columns C, conv clamp, ADD's fused activation, the lane after the ADD)
    "resnet_relu": lambda N: (64, 144, 16, "none", "relu", "relu"),  # ResNet-8's shape: 8x8 rows, 3x3x16 depth, 16 channels
    "deep_cached": lambda N: (2 * N, 3 * N + 1, N, "none", "relu", "relu"),  # dt 4, rt 2: cached
    "deep_uncached": lambda N: (N, 3 * N + 1, N, "none", "none", "linear"),  # dt 4, rt 1: uncached
    "tiles": lambda N: (2 * N + 3, N + 2, 2 * N + 5, "none", "relu", "relu"),  # rt 3, ct 3, dt 2
    "relu6_conv_linear_add": lambda N: (2 * N, 2 * N, N, "relu6", "none", "linear"),  # Review Focus 3
    "tanh_after_add": lambda N: (N, 2 * N, N, "none", "none", "tanh"),
    "shallow_cached": lambda N: (2 * N, N - 3, N, "none", "relu", "relu"),  # dt 1, rt 2: the hold set is the first pass and the bias pass, cached
    "shallow_uncached": lambda N: (N, N - 1, N, "relu6", "none", "sigmoid"),  # dt 1, rt 1: uncached, a lane after the ADD
}
RES_SEED = ("deep_cached", "deep_uncached", "relu6_conv_linear_add", "resnet_relu", "tanh_after_add", "tiles", "shallow_cached",
            "shallow_uncached")  # seed offsets: the first six sorted, as they were, then each later case appended


def residual_case(name: str, N: int, rng) -> tuple:
    """One int8 residual layer job (random codes, per-channel words and ADD scales; the ADD words from an AddRecord named after the layer), its
    bit-exact output (a diagnostic) and the ADD outputs of two wrong ADDs (zp_1 taken as 0, no ADD) the stage gate must reject."""
    M, K, C, conv_act, add_act, lane_act = RES_CASES[name](N)
    A_q = rng.randint(-128, 128, (M, K)).astype(np.int64)
    W_q = rng.randint(-127, 128, (K, C)).astype(np.int64)
    s_w = np.exp(rng.uniform(np.log(1e-3), np.log(2e-2), C))
    hw_bias = rng.randint(-2000, 2000, C).astype(np.int64)
    acc = wrap32(imatmul(A_q, W_q) + hw_bias[None, :])
    rq = requant_params(acc, 0.02, s_w, "linear" if conv_act == "none" else "relu", np.random.RandomState(rng.randint(1 << 30)))
    rq["amin"], rq["amax"] = ipu.activation_clamp(conv_act, rq["s_out"], rq["zp"])
    q1 = requantize(acc, rq)
    s2, z2 = rq["s_out"] * 2.0 ** rng.uniform(-3.0, 3.0), int(rng.randint(-100, 101))
    R_q = rng.randint(-128, 128, (M, C)).astype(np.int64)
    y = rq["s_out"] * (q1 - rq["zp"]) + s2 * (R_q - z2)
    _, s_out, z_out = quant_act(np.maximum(y, 0.0) if add_act == "relu" else y)
    s_out *= 2.0 ** rng.uniform(-0.5, 0.5)
    lo, hi = ipu.activation_clamp(add_act, s_out, z_out)
    add = mr.add_record_words(mr.AddRecord(float(rq["s_out"]), int(rq["zp"]), float(s2), z2, float(s_out), int(z_out), int(lo), int(hi), name))
    ra = mr.add_lane_params(add, s_out, mr.qadd_codes(q1, R_q, add))
    want = mr.int8_residual_exact(A_q, W_q, hw_bias, rq, R_q, add, ra, lane_act)
    wrong = (mr.qadd_codes(q1, R_q, dict(add, zp1=0)), q1)  # the ADD's outputs with zp_1 taken as the add set's 0, and with no ADD
    req = dict(rq, mx=ra["mx"], shx=ra["shx"], mout=ra["mout"], shout=ra["shout"], zout=ra["zout"])  # Q: the conv's zp and clamp, the lane's words
    job = {"terms": [(A_q.astype(np.float32), W_q.astype(np.float32)), (R_q.astype(np.float32), np.eye(C, dtype=np.float32))],
           "bias": hw_bias, "act": lane_act, "shape": (M, C), "req": req, "add": add, "lane": ra, "clamps": (conv_act, add_act)}
    return job, want, wrong, q1, R_q


def residual_rejects(job, q1, R_q, wrong) -> tuple:
    """Whether the ADD stage gate fails each wrong ADD's outputs (zp_1 = 0, no ADD) on the host's q1 and q2 = R_q."""
    a = job["add"]
    lo = max(a["amin"], a["zpo"]) if job["act"] == "relu" else a["amin"]
    g = lambda y: truth.gate_quantized_add(q1, R_q, a["ratios"][0], a["zp1"], a["ratios"][1], a["zp2"], 1.0, a["zpo"], lo, a["amax"], y)
    return tuple(not g(y).ok for y in wrong)


def _stage_gates_bad(label: str, gates: list) -> list:
    """Prints each stage gate's line under label; returns the failing stages' names."""
    bad = []
    for stage, r in gates:
        print(f"    {stage}: {truth.line(r, bound=_act_bound(r.func)) if stage.startswith('activation') else truth.stage_line(r)}", flush=True)
        if not r.ok:
            bad.append(f"{label} ({stage})")
    return bad


def _res_block_models(seed: int) -> dict:
    """Two int8 residual blocks on 8x8x16 from the converter (test vehicles): conv 3x3 + ADD(x) + ReLU, and conv 3x3 / 2 + ADD(conv 1x1 / 2) + ReLU."""
    rng = np.random.default_rng(seed)
    w3 = tf.constant((rng.standard_normal((3, 3, 16, 16)) * 0.08).astype(np.float32))
    w1 = tf.constant((rng.standard_normal((1, 1, 16, 16)) * 0.25).astype(np.float32))
    b3 = tf.constant((rng.standard_normal(16) * 0.1).astype(np.float32))
    b1 = tf.constant((rng.standard_normal(16) * 0.1).astype(np.float32))
    spec = [tf.TensorSpec((1, 8, 8, 16), tf.float32)]

    @tf.function(input_signature=spec)
    def block(x):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w3, 1, "SAME"), b3) + x)

    @tf.function(input_signature=spec)
    def block_s2(x):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w3, 2, "SAME"), b3) + tf.nn.bias_add(tf.nn.conv2d(x, w1, 2, "VALID"), b1))

    rep = np.random.default_rng(seed + 1)
    out = {}
    for name, fn in (("block", block), ("block_s2", block_s2)):
        holder = tf.Module()
        holder.f = fn
        conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], holder)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([rep.uniform(-1.0, 2.0, (1, 8, 8, 16)).astype(np.float32)] for _ in range(100))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        out[name] = conv.convert()
    return out


def _interp_tensors(model: bytes, x) -> tuple:
    """The interpreter's (BUILTIN_REF) output codes per image and every tensor it kept, as {index: (images, ...) codes}; for reported agreement only."""
    it = tf.lite.Interpreter(model_content=model, experimental_op_resolver_type=REF, experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    keep = [int(o["outputs"][0]) for o in it._get_ops_details()]
    ys, tens = [], {k: [] for k in keep}
    for img in x:
        it.set_tensor(i_in, img[None].astype(np.int8))
        it.invoke()
        ys.append(it.get_tensor(i_out)[0].astype(np.int64))
        for k in keep:
            tens[k].append(it.get_tensor(k)[0].astype(np.int64))
    return np.stack(ys), {k: np.stack(v) for k, v in tens.items()}


def _sets_line(sets: int, want: int) -> str:
    """A layer's set count against the expected rt ct passes."""
    return f"sets {sets} (expected {want})"


def onnx_res_block(seed: int) -> tuple:
    """A hand-built ONNX-style residual block (QLinearConv 3x3 pads 1 + QLinearAdd(x) + Relu, uint8, 2 x 8x8x16): (conv record, ADD record, x, R)."""
    rng = np.random.default_rng(seed)
    x_u8, xs, xz = rng.integers(0, 256, (2, 8, 8, 16)), 0.02, 120
    w, ws, b = rng.integers(-127, 128, (16, 16, 3, 3)), rng.uniform(0.0005, 0.002, 16), rng.integers(-2000, 2000, 16)
    xr = xs * (x_u8 - xz)
    Wr = np.transpose(w, (0, 2, 3, 1)).reshape(16, -1).T * ws[None, :]
    y = np.vstack([mr.im2col(img.astype(np.float32), 3, 3, (1, 1), (1, 1, 1, 1))[0].astype(np.float64) @ Wr for img in xr]) + b * xs * ws
    ys = float(y.max() - y.min()) / 255.0
    yz = int(np.clip(np.rint(-y.min() / ys), 0, 255))
    conv = mr.record_of_onnx("QLinearConv", xs, xz, w, ws, np.zeros(16), ys, yz, b, pads=[1, 1, 1, 1], x_unsigned=True, y_unsigned=True,
                             name="onnx_conv")
    cs = float(np.maximum(y + xr.reshape(-1, 16), 0.0).max()) / 255.0
    add = mr.add_record_of_onnx(ys, yz, xs, xz, cs, 0, relu=True, unsigned=True, name="onnx_add")
    x = x_u8 - 128
    return conv, add, x, x.reshape(-1, 16)


def _shortcut_conv(convs: dict):
    """The 1x1 conv of block_s2's two; a ValueError when there is none."""
    one = [o for o in convs.values() if o["filter"]["shape"][1] == 1]
    if len(one) != 1:
        raise ValueError(f"block_s2: {len(one)} 1x1 convs, expected one shortcut")
    return one[0]


def residual_tflite(sim, seed: int, work: str) -> list:
    """The converter-made residual blocks through the TFLite importer and a hand-built ONNX-style block, on TB_model_run, gated per stage."""
    _tf_imports()
    bad = []
    x = np.random.default_rng(seed + 2).integers(-128, 128, (4, 8, 8, 16)).astype(np.int64)
    for name, model in _res_block_models(seed).items():
        path = os.path.join(work, f"{name}.tflite")
        with open(path, "wb") as f:
            f.write(model)
        try:
            g = mr.load_int8_graph(path)
        except ValueError as e:
            print(f"tflite {name}: FAIL: {e}", flush=True)
            bad.append(f"tflite {name}")
            continue
        kinds = sorted(o["kind"] for o in g["ops"])
        if kinds != (["ADD", "CONV_2D"] if name == "block" else ["ADD", "CONV_2D", "CONV_2D"]):
            print(f"tflite {name}: FAIL: the converter made {kinds}", flush=True)
            bad.append(f"{name}: the converter made {kinds}")
            continue
        try:
            ys, tens = _interp_tensors(model, x)
            add_op = next(o for o in g["ops"] if o["kind"] == "ADD")
            convs = {o["output"]["index"]: o for o in g["ops"] if o["kind"] == "CONV_2D"}
            a1, a2 = add_op["inputs"]
            if name == "block":
                main = convs[a1] if a1 in convs else convs[a2]
                res_i, R = (a2 if a1 in convs else a1), x.reshape(-1, 16)
                if res_i != g["input"]:
                    raise ValueError(f"the ADD's other input is tensor {res_i}, not the graph's input {g['input']}")
            else:
                short = _shortcut_conv(convs)
                main = next(o for o in convs.values() if o is not short)
                if {a1, a2} != {short["output"]["index"], main["output"]["index"]}:
                    raise ValueError(f"the ADD's inputs {a1, a2} are not the two convs' outputs")
                sjob, _ = mr.job_of_record(mr.record_of_tflite(short, f"{name}_short"), x)
                S, sets, cyc = sim.run_job(sjob, f"{name}_short")
                (Xs, _), = sjob["terms"]
                want_s = -(-Xs.shape[0] // sim.N) * -(-S.shape[1] // sim.N) * -(-Xs.shape[1] // sim.N)
                ws = tens[short["output"]["index"]].reshape(-1, 16)
                print(f"tflite {name} shortcut layer: {S.size} outputs, {_sets_line(sets, want_s)}, cycles {cyc}; reported: "
                      f"{int(np.sum(S != ws))} differ from the interpreter's tensor", flush=True)
                bad += _stage_gates_bad(f"{name} shortcut", mr.layer_stage_gates(sjob, S, sim.last_stages, sim.N))
                if sets != want_s:
                    bad.append(f"tflite {name} shortcut (sets)")
                res_i, R = short["output"]["index"], S
            job, _ = mr.job_of_add_record(mr.record_of_tflite(main, f"{name}_conv"),
                                          mr.add_record_of_tflite(add_op, main["output"]["index"], name), x, R)
            Y, sets, cyc = sim.run_job(job, name)
            gates = mr.layer_stage_gates(job, Y, sim.last_stages, sim.N)
        except (RuntimeError, ValueError) as e:
            print(f"tflite {name}: FAIL: {e}", flush=True)
            bad.append(f"tflite {name}")
            continue
        M, C = job["shape"]
        np_ = sim.last_stages["np"]
        q1 = sim.last_stages["R"][np_ - 2][:M, :C]  # the hold beats: the conv's own codes
        want = -(-M // sim.N) * -(-C // sim.N) * (-(-job["terms"][0][0].shape[1] // sim.N) + 1)
        print(f"tflite {name}: {Y.size} outputs, {_sets_line(sets, want)}, cycles {cyc}; ADD input 1 is the residual: "
              f"{'yes' if a1 == res_i else 'no'}; reported: {int(np.sum(Y != ys.reshape(-1, C)))} differ from the interpreter (BUILTIN_REF), "
              f"the conv's codes (hold beats) {int(np.sum(q1 != tens[main['output']['index']].reshape(-1, C)))} differ from its tensor", flush=True)
        bad += _stage_gates_bad(f"tflite {name}", gates)
        if sets != want:
            bad.append(f"tflite {name} (sets)")
    try:  # the same neutral path from another importer's records
        conv, add, xo, Ro = onnx_res_block(seed + 3)
        job, _ = mr.job_of_add_record(conv, add, xo, Ro)
        Y, sets, cyc = sim.run_job(job, "onnx_block")
        gates = mr.layer_stage_gates(job, Y, sim.last_stages, sim.N)
    except (RuntimeError, ValueError) as e:
        print(f"onnx-style block: FAIL: {e}", flush=True)
        return bad + ["onnx-style block"]
    M, C = job["shape"]
    want = -(-M // sim.N) * -(-C // sim.N) * (-(-job["terms"][0][0].shape[1] // sim.N) + 1)
    print(f"onnx-style block (QLinearConv + QLinearAdd + Relu records): {Y.size} outputs, {_sets_line(sets, want)}, cycles {cyc}", flush=True)
    bad += _stage_gates_bad("onnx-style block", gates)
    return bad + (["onnx-style block (sets)"] if sets != want else [])


RES_LINE = re.compile(r"\[RES\] reset (\d+) cycles into the layer, inside the hold window, (\d+) of (\d+) hold beats unread")


def residual_mid_reset(sim, job, Y0, want) -> list:
    """Resets with hold_mem holding beats: a control must fail +hold_check; "full" lands in the first window, "waiting" after it closed."""
    first, free = sim.first_hold, sim.first_free  # the plain run's: the runs below overwrite them
    if first is None or free is None:
        return [f"mid-reset: the plain run printed no first hold_done ({first}) or hold window close ({free})"]
    bad = []
    try:
        sim.run_job(job, "tiles_mid_reset_ctl", plusargs=("+mid_reset=1", "+hold_check"))
        bad.append("mid-reset control: a reset outside the hold window passed +hold_check")
    except RuntimeError:
        if "+hold_check: the mid-stream reset" not in (sim.last_log or ""):
            bad.append("mid-reset control: the run failed, but not on +hold_check")
    print(f"residual tiles mid-reset control (+mid_reset=1 +hold_check): {'FAIL ' + bad[0] if bad else 'the check fires outside the hold window'}",
          flush=True)
    for label, c in (("full", first + 1), ("waiting", free)):
        try:
            Y, _, _ = sim.run_job(job, f"tiles_mid_reset_{label}", plusargs=(f"+mid_reset={c}", "+hold_wait", "+hold_check"))
            gates = mr.layer_stage_gates(job, Y, sim.last_stages, sim.N)
        except (RuntimeError, ValueError) as e:
            print(f"residual tiles mid-reset ({label}) from {c}: FAIL: {e}", flush=True)
            bad.append(f"mid-reset {label}")
            continue
        r = RES_LINE.search(sim.last_log or "")
        at, unread, per = (int(v) for v in r.groups()) if r else (None, None, None)
        n = int(np.sum(Y != Y0))
        print(f"residual tiles mid-reset ({label}) asked from {c} cycles (first hold_done {first}, first window closed {free}), landed at {at}, "
              f"inside the hold window with {unread} of {per} hold beats unread: {Y.size} outputs, {n} differ from the run without it; reported: "
              f"{int(np.sum(Y != want))} differ from the bit-exact model", flush=True)
        ok = at is not None and (c <= at < free if label == "full" else at > c)  # full: inside the first window; waiting: past its close
        if not ok:
            bad.append(f"mid-reset {label} (landed at {at}; asked from {c}, the first window closes at {free})")
        if n:
            bad.append(f"mid-reset {label} (outputs differ from the run without it)")
        bad += _stage_gates_bad(f"mid-reset {label}", gates)
    return bad


def residual_main(argv=None) -> None:
    """int8 residual layers on sienna_top's layer contract (TB_model_run), each stage gated on the true function (the conv's requantize, the identity
    requantize, hold_mem, the ADD, the fill, the lane); int8_residual_exact is a diagnostic. --tflite adds the converter-made residual blocks
    through the TFLite importer and a hand-built ONNX-style block, --mid-reset resets of the tiles case with hold_mem holding beats."""
    ap = argparse.ArgumentParser(description=residual_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="residual")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--cases", nargs="*", default=list(RES_CASES), choices=list(RES_CASES))
    ap.add_argument("--tflite", action="store_true")
    ap.add_argument("--mid-reset", action="store_true")
    ap.add_argument("--collapse-k", type=int, choices=[0, 1], default=1, help="the mesh's COLLAPSE_K the layer engine builds")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--work", default=os.path.join(ROOT, "testbenches", "results", "residual"))
    a = ap.parse_args(argv)
    if a.mid_reset and "tiles" not in a.cases:
        ap.error("--mid-reset resets the tiles case: add it to --cases")
    mr.COLLAPSE_K = a.collapse_k
    os.makedirs(a.work, exist_ok=True)
    print(f"residual: N={a.n} T={a.tile} lanes={a.lanes} collapse-k {a.collapse_k} sim args '{os.environ.get('SIM_ARGS', '')}'", flush=True)
    sim = mr.RtlLayer(a.n, a.lanes, a.work, "int8", a.tile)
    sim.build()
    bad = []
    for name in a.cases:
        job, want, wrong, q1, R_q = residual_case(name, a.n, np.random.RandomState(a.seed + RES_SEED.index(name)))
        M, K, C = RES_CASES[name](a.n)[:3]
        rt, ct, dt = -(-M // a.n), -(-C // a.n), -(-K // a.n)
        try:
            Y, sets, cyc = sim.run_job(job, name)
            gates = mr.layer_stage_gates(job, Y, sim.last_stages, a.n)
        except (RuntimeError, ValueError) as e:  # an assertion, a TB check, a refusal or a missing stage: this case fails, the others still run
            print(f"residual {name} N={a.n} T={a.tile}: FAIL: {e}", flush=True)
            bad.append(name)
            continue
        mism = int(np.sum(Y != want))
        w1, w2 = residual_rejects(job, q1, R_q, wrong)
        print(f"residual {name} N={a.n} T={a.tile}: {M} x {K} x {C}, {Y.size} outputs, sets {sets} (expected {rt * ct * (dt + 1)}), "
              f"cycles {cyc}; reported: {mism} outputs differ from the bit-exact model; the ADD gate rejects zp_1 = 0: {w1}, no ADD: {w2}", flush=True)
        bad += _stage_gates_bad(name, gates)
        if sets != rt * ct * (dt + 1):
            bad.append(f"{name} (sets)")
        if (job["add"]["zp1"] != 0 and not w1) or not w2:
            bad.append(f"{name} (its gate cannot tell a wrong ADD)")
        if a.mid_reset and name == "tiles":  # right after its plain run, whose first hold_done sim.first_hold holds
            bad += residual_mid_reset(sim, job, Y, want)
    if a.tflite:
        bad += residual_tflite(sim, a.seed, a.work)
    print(f"RESIDUAL: {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
    sys.exit(1 if bad else 0)


# ── perf (was perf_analysis.py) ──────────────────────────────────────────────

REPORT = os.path.join(ROOT, "testbenches", "results", "perf", "pipeline_performance_report.log")
CONFIGS = [t["name"] for t in PIPELINE_TESTS]


GEOM = {"n": 16, "tile_size": 4, "lanes": 32}  # set from the command line in perf_main()


def run(name: str, num_sets: int, build_dir: str) -> str:
    cfg = next(t for t in PIPELINE_TESTS if t["name"] == name)
    generate_vectors({**GEOM, **cfg, "num_sets": num_sets})
    cmd = ["make", mr.SIM_TARGET, "TRACE=0", "EXTRA_FLAGS=-DPERF", f"{'VCS' if mr.SIM == 'vcs' else 'VERILATOR'}_DIR={build_dir}",
           f"FMT={GEOM.get('fmt_name', 'fp32')}", f"N={GEOM['n']}", f"TILE={GEOM['tile_size']}", f"LANES={GEOM['lanes']}",
           "GEN_PKG=0"]  # the package generate_vectors just wrote
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    raw_dir = os.path.join(os.path.dirname(REPORT), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    open(os.path.join(raw_dir, f"{name}_N{GEOM['n']}.log"), "w").write(r.stdout + r.stderr)  # to re-parse without a rerun
    return r.stdout + r.stderr


def events(raw: str) -> dict:
    """PERF lines of the first plain stream pass, as {kind: [(cycle, value, value2), ...]}."""
    ev, on = {}, False
    for m in re.finditer(r"^PERF (\d+) (\w+)(?: (-?\d+))?(?: (-?\d+))?", raw, re.M):
        c, kind = int(m.group(1)), m.group(2)
        a = int(m.group(3)) if m.group(3) is not None else None
        b = int(m.group(4)) if m.group(4) is not None else None
        if kind == "PASS":
            if on:
                break  # only the first stream pass, which has no overrun and no reset
            on = b == 0
            continue
        if on:
            ev.setdefault(kind, []).append((c, a, b))
    return ev


def leaves(ev: dict, kind: str, idle: int) -> tuple:
    """Cycles a state machine leaves and re-enters its idle state: one pair per set it takes."""
    up, down, prev = [], [], idle
    for c, a, _ in ev.get(kind, []):
        if prev == idle and a != idle:
            up.append(c)
        if prev != idle and a == idle:
            down.append(c)
        prev = a
    return up, down


def analyse(ev: dict, k_sets: int, passes: int = 1, flags: list = None) -> dict:
    """Per-set stage times; with passes > 1 only every passes-th set outputs (the others are partial sums); with flags, analyse_res."""
    load, start = [c for c, _, _ in ev["HOST_LOAD"]], [c for c, _, _ in ev["HOST_START"]]
    launch = [c for c, a, _ in ev.get("MESH", []) if a == 1]
    written = [c for c, a, _ in ev.get("MESH", []) if a == 7]
    reduce = [c for c, a, _ in ev.get("MESH", []) if a == 5]
    g_up, g_dn = leaves(ev, "G", 0)
    p_up, p_dn = leaves(ev, "P", 0)
    done = [c for c, _, _ in ev["DONE"]]
    if flags is not None:
        return analyse_res(ev, k_sets, flags, load, start, launch, written, reduce, g_up, g_dn, p_up, p_dn, done)
    n = min(k_sets, len(done), len(start), len(launch), passes * min(len(written), len(g_dn)))
    n -= n % passes  # whole outputs only
    sets = []
    for k in range(n):
        j, f = k // passes, k - k % passes  # output index (reduce, write and activation are per output); its first set
        s = dict(load=start[k] - load[k], wait_mesh=launch[k] - start[k], mesh=None, wait_act=None, act=None, out=None,
                 latency=None)
        if (k + 1) % passes == 0:  # the set that completes an output
            s.update(mesh=written[j] - launch[f], wait_act=g_up[j] - written[j], act=g_dn[j] - g_up[j],
                     out=done[k] - g_dn[j], latency=done[k] - start[f])
        sets.append(s)
    last = lambda k: (k + 1) % passes == 0
    gap = lambda xs: [xs[k] - xs[k - 1] for k in range(1, min(n, len(xs)))]
    gaps = gap(done)
    full = [done[k] for k in range(n) if last(k)]
    per_out = [b - a for a, b in zip(full, full[1:])]
    steady = [g / passes for g in per_out[len(per_out) // 2:]]  # cycles per set: an output's interval over its passes
    lanes = [a / (GEOM["lanes"] * b) for _, a, b in ev.get("LANES", []) if b]
    pool = [d - u for u, d in zip(p_up, p_dn)]  # sets that output; partial sums skip pooling
    return dict(sets=sets, gaps=gaps, steady=steady, n=n, launch_gaps=gap(launch), host_gaps=gap(start),
                first_latency=next((x["latency"] for x in sets if x["latency"] is not None), 0),
                act_gaps=gap(g_up), pool=pool, lanes=lanes, reduce_to_written=[w - r for r, w in zip(reduce, written)])


def analyse_res(ev, k_sets, flags, load, start, launch, written, reduce, g_up, g_dn, p_up, p_dn, done) -> dict:
    """analyse() for a residual test: written, g_up and g_dn index the staged sets (no partial bit); a hold set's stage interval goes to hold."""
    staged = [k for k in range(k_sets) if not flags[k] & 1]
    m = min(len(written), len(g_dn), len(staged))
    n = min(k_sets, len(done), len(start), len(launch), staged[m - 1] + 1 if m else 0)
    sets, hold, outs, j, pf, gf = [], [], [], 0, 0, 0  # pf: the first pass of the set's depth group; gf: of its output (a hold set's group included)
    for k in range(n):
        s = dict(load=start[k] - load[k], wait_mesh=launch[k] - start[k], mesh=None, wait_act=None, act=None, out=None, latency=None)
        if not flags[k] & 1:
            s.update(mesh=written[j] - launch[pf], wait_act=g_up[j] - written[j])
            if flags[k] & 4:  # a hold set: its beats wait in hold_mem for the add set, pooling passes it as a null
                hold.append(g_dn[j] - g_up[j])
            else:
                s.update(act=g_dn[j] - g_up[j], out=done[k] - g_dn[j], latency=done[k] - start[gf], add=bool(flags[k] & 8))
                outs.append(k)
                gf = k + 1
            j, pf = j + 1, k + 1
        sets.append(s)
    gap = lambda xs: [xs[k] - xs[k - 1] for k in range(1, min(n, len(xs)))]
    per_set = [(done[b] - done[a]) / (b - a) for a, b in zip(outs, outs[1:])]  # an output's completion interval over its group's sets
    return dict(sets=sets, gaps=gap(done), steady=per_set[len(per_set) // 2:], n=n, launch_gaps=gap(launch), host_gaps=gap(start),
                first_latency=next((x["latency"] for x in sets if x["latency"] is not None), 0), act_gaps=gap(g_up),
                pool=[d - u for u, d in zip(p_up, p_dn)], lanes=[a / (GEOM["lanes"] * b) for _, a, b in ev.get("LANES", []) if b],
                reduce_to_written=[w - r for r, w in zip(reduce, written)], hold=hold)


LANE_ACTS = ("selu", "sigmoid", "tanh")  # the activations the lanes run as polynomials
FMT_KNOBS = tuple(fmt_pkg.FORMATS)  # the builds' formats, one define each


def rtl_lat() -> tuple:
    """sienna_fmt_pkg's unit latencies under each define, through fmt_pkg.py: ({fmt: (MUL_LATENCY, ADD_LATENCY)}, fixedPointMultiplyAdd's, REQUANT_LATENCY, QUANT_ADD_LATENCY)."""
    unit = {f: (fmt_pkg.get(f, "MUL_LATENCY"), fmt_pkg.get(f, "ADD_LATENCY")) for f in FMT_KNOBS}
    fx = fmt_pkg.get("int8", "HORNER_STEP_LATENCY") - 1  # the int8 Horner step is one register, then fixedPointMultiplyAdd
    return unit, fx, fmt_pkg.get("int8", "REQUANT_LATENCY"), fmt_pkg.get("int8", "QUANT_ADD_LATENCY")


UNIT_LAT = FX_LAT = REQ_LAT = QADD_LAT = None  # mul_lat and add_lat per format, the int8 Horner step's multiply-add, int8Requantizer, add_lanes' ADD; perf_init() parses them
MUL_LAT = ADD_LAT = None  # valid in to done out; perf_init() sets fp32's, perf_main() the build's format's values
FMT = "fp32"  # the build's format; perf_main() sets it


def lane_lat(f: str) -> dict:
    """The gpnae lane's latencies in a format, from the package: the Horner step (LOOP), the input stage (ACT_IN), the post multiply and add."""
    return {"LOOP": fmt_pkg.get(f, "HORNER_STEP_LATENCY"), "ACT_IN": fmt_pkg.get(f, "LANE_MUL_LATENCY") + 1, "ML": fmt_pkg.get(f, "LANE_MUL_LATENCY"),
            "AL": fmt_pkg.get(f, "LANE_ADD_LATENCY"), "RQ": fmt_pkg.get(f, "REQUANT_LATENCY")}


def pkg() -> dict:
    """The generated test_config_pkg's integer parameters: the geometry this run was built with."""
    raw = open(os.path.join(ROOT, "testbenches", "test_config_pkg.sv")).read()
    return {k: int(v) for k, v in re.findall(r"localparam int (\w+) = (-?\d+);", raw)}


def clog2(x: int) -> int:
    return max(0, (x - 1).bit_length())


def model(cfg: dict, collapse: bool = True, P: dict = None, words: int = None) -> dict:
    """Cycles each stage should take per set, from the RTL's structure; P and words (matrix_west_0's) default to the generated test's."""
    P = P or pkg()
    N, T, lanes = P["N"], P["TILE_SIZE"], P["NUM_LANES"]
    per_lane = P["SRAM_DEPTH"] // lanes
    K = N if collapse else T  # depth each array multiplies
    U = min(K, ADD_LAT + 1)  # partial sums per PE pixel: the adder loop plus one (6 in fp32 and bf16, 2 in int8)
    RP = 1 if collapse else N // T  # depth slices summed per output tile
    LAT = 1 + ADD_LAT * clog2(RP * U + 1)  # reducer read to write: tree over the partials and the bias
    words = len(open(os.path.join(mr.TB_DIR, "matrix_west_0.mem")).read().split()) if words is None else words
    rows = -(-words // P["HOST_WORDS"])
    m = dict(
        # TB_sienna_top: one row per cycle, enable drops for a cycle, start, one cycle for the credit to land.
        host=rows + 3,
        # Broadcast T rows plus commit, the array feed of K products, the reducer's T^2 reads: whichever is longest.
        mesh=max(T + 2, K, T * T),
        # Launch to written: broadcast T+2, feed registers 2, skew 2(T-1), depth K-1, multiply, add, final flag 2,
        # then T^2 reads and the tree.
        mesh_lat=(T + 2) + 2 + 2 * (T - 1) + (K - 1) + MUL_LAT + ADD_LAT + 2 + T * T + LAT,
        per_lane=per_lane, LAT=LAT)
    m["mesh_lat"] += (cfg.get("accum_passes", 1) - 1) * m["host"]  # each further depth pass launches one host interval later (TB_sienna_top is host-bound)
    act = cfg.get("act")
    windows = ((P["IN_ROWS"] + 2 * P["PADDING"] - P["POOL_H"]) // P["STRIDE_ROWS"] + 1) * \
              ((P["IN_COLS"] + 2 * P["PADDING"] - P["POOL_W"]) // P["STRIDE_COLS"] + 1)
    m["pool_dispatch"] = -(-windows // lanes) * P["POOL_H"] * P["POOL_W"]  # every lane takes a window element per cycle
    if cfg.get("packed"):  # each lane runs its own column's pack entry
        m["act"] = packed_stage(P, collapse)
    elif (cfg.get("residual") or cfg.get("residual_every")) and not cfg.get("mixed_acts"):  # the add set's stage
        # A lane set's beats reach the lanes qadd_lat() later through add_lanes; a ReLU or linear set leaves at byp_all_in, before add_lanes drains.
        m["act"] = per_lane + 1 if act in ("relu", "linear") else lane_stage(act, P, collapse) + QADD_LAT
        m["hold"] = per_lane + 1  # a hold set passes the stage into hold_mem
    elif act in ("relu", "linear") and not cfg.get("mixed_acts") and cfg.get("accum_passes", 1) == 1:
        m["act"] = per_lane + 1  # from the first beat (TB g_eff): per_lane wide reads, g_done at byp_all_in, idle; FEED/LATCH precede or overlap the reads (per_lane >= 2)
    elif act in LANE_ACTS and not cfg.get("mixed_acts") and cfg.get("accum_passes", 1) == 1:
        m["act"] = lane_stage(act, P, collapse)
    return m


def lane_params() -> tuple:
    """gpnae's K and TAIL_CONTEXTS defaults, parsed from the RTL; fails if sienna_pipe overrides them."""
    src = open(os.path.join(ROOT, "GPNAE", "src", "gpnae.sv")).read()
    inst = re.search(r"\bgpnae #\((.*?)\) gpnae", open(os.path.join(ROOT, "src", "pipeline", "sienna_pipe.sv")).read(), re.S)
    k, t = (re.search(rf"parameter int\s+{n}\s*=\s*(\d+)", src) for n in ("K", "TAIL_CONTEXTS"))
    if not (k and t and inst) or re.search(r"\.(K|TAIL_CONTEXTS)\s*\(", inst.group(1)):
        raise RuntimeError("regression.py --action perf: gpnae's K / TAIL_CONTEXTS not found, or sienna_pipe overrides them")
    return int(k.group(1)), int(t.group(1))


GROUP_K = TAIL_CTX = None  # gpnae's K and TAIL_CONTEXTS; perf_init() parses them


def perf_init() -> None:
    """Parses the RTL's unit latencies and lane parameters for the perf model; at perf's entry, not at import, so an unparsable RTL fails only perf."""
    global UNIT_LAT, FX_LAT, REQ_LAT, QADD_LAT, MUL_LAT, ADD_LAT, GROUP_K, TAIL_CTX
    UNIT_LAT, FX_LAT, REQ_LAT, QADD_LAT = rtl_lat()
    MUL_LAT, ADD_LAT = UNIT_LAT["fp32"]
    GROUP_K, TAIL_CTX = lane_params()


def lane_inputs(k: int, P: dict, collapse: bool) -> list:
    """Set k's activation inputs from the bit-exact mesh model, row-major as the wide read hands them to the lanes."""
    rd = lambda f: np.array([int(w, 16) for w in open(os.path.join(mr.TB_DIR, f)).read().split()], np.int64)
    n, f = P["N"], fpu.FORMATS[FMT]
    A, B = (rd(f"matrix_{s}_{k}.mem").reshape(n, n) for s in ("west", "north"))
    C = mesh_model.matmul(f, [(A, B)], n, P["TILE_SIZE"], int(collapse), rd(f"bias_{k}.mem") if P["HAS_BIAS"] else None)
    return mr.bits_float(np.asarray(C, np.int64), FMT).astype(np.float64).flatten().tolist()  # bf16: each fp32 sum rounded once, as the lanes see it


def lane_kinds(xs: list, act: str) -> tuple:
    """Each element's kind and Horner rounds at tier 0 (gpnae_model): float inputs as the format's words; int8 has no per-element data here,
    so every element is taken as a polynomial one (the activation's tier-0 degree), the worst case."""
    code = mr.activation_to_code(act)
    if FMT == "int8":
        m = gpnae_model.default_mem("int8")
        d = {1: [gpnae_model.D_SELU], 2: [gpnae_model.D_SIGP, gpnae_model.D_SIGN], 3: [gpnae_model.D_TANH]}[code]
        r = max(m.desc[0][i].deg for i in d) + 1
        return [gpnae_model.K_MUL if code == 1 else gpnae_model.K_PASS] * len(xs), [r] * len(xs)
    f = fpu.FORMATS[FMT]
    w = np.asarray(np.float32(xs)).view(np.uint32).astype(np.int64) >> (23 - f.m)
    k, r = gpnae_model.Lane(f).rounds(w, code, 0)
    return [int(v) for v in k], [int(v) for v in r]


def tail_ops(tier: int = 0) -> int:
    """gpnae_tail's cycles for one element alone, start to T_DONE: each fp32 step is request, issue, unit, update (latency + 2)."""
    ts = gpnae_model.default_mem("bf16").tail[tier]
    cm, ca = UNIT_LAT["fp32"][0] + 2, UNIT_LAT["fp32"][1] + 2
    return cm + (ts.deg + 1) * (cm + ca) + 2 * ca + ts.newton * (cm + ca + cm) + cm  # k; Horner; D and y0; Newton; E y


def lane_cycles(xs: list, act: str, s: int) -> int:
    """Cycle of a lane's last result put, its first G_CAP at s: groups of GROUP_K run capture (each into the input stage), the MAC, the post
    issues and emit in turn; per group the MAC runs its highest element's rounds."""
    L = lane_lat(FMT)
    kinds, rounds = lane_kinds(xs, act)
    K = gpnae_model
    for g in range(0, len(xs), GROUP_K):
        kd, rd = kinds[g:g + GROUP_K], rounds[g:g + GROUP_K]
        n, R = len(kd), max(rounds[g:g + GROUP_K])
        t0 = s + n + 2 + L["ACT_IN"]  # the last capture's input-stage result loads with the start
        d = t0 + R * max(n, L["LOOP"] + 1) + L["LOOP"] + 3 + n  # RUN rounds, DRAIN LOOP + 2, EMIT n: barrel_mac's done
        exp = any(k == K.K_EXP for k in kd)
        e0 = d + n + 1 + ((L["AL"] + 2) + n if exp else 0)  # G_EMIT, after the post issues (and fp32's second pass)
        ready = []
        for i, k in enumerate(kd):
            iss = d + 1 + i
            if FMT == "int8":
                ready.append(iss + 1 + L["ML"] + L["RQ"] + 1 if act == "selu" else iss + 2)
            elif k in (K.K_PASS, K.K_SAT):
                ready.append(iss + 2)
            elif k == K.K_REFL:
                ready.append(iss + L["AL"] + 2)
            elif k == K.K_EXP:
                ready.append(d + n + L["AL"] + 3 + i + L["ML"] + 2)
            elif k == K.K_TAIL:
                ready.append(-1)  # set below
            else:
                ready.append(iss + L["ML"] + 2)
        if FMT == "bf16":  # tail elements start as classified, one per cycle, on TAIL_CTX contexts; unit contention is not modelled
            free, last = [-1] * TAIL_CTX, -1
            for i, k in enumerate(kd):
                if k == K.K_TAIL:
                    c = free.index(min(free))
                    t_s = max(s + 3 + i + L["ACT_IN"] + 1, last + 1, free[c])
                    free[c], last, ready[i] = t_s + tail_ops() + 2, t_s, t_s + tail_ops() + 3
        e = e0 - 1
        for r in ready:
            e = max(e + 1, r)
        s = e + 2  # G_NEXT, then the next group's G_CAP
    return e + 1


def lane_stage(act: str, P: dict, collapse: bool = True) -> int:
    """Activation stage cycles of a lane set (median over the streamed sets): FEED to the cycle after every lane is collected."""
    per_lane, lanes = P["SRAM_DEPTH"] // P["NUM_LANES"], P["NUM_LANES"]
    start = per_lane + 3 + (REQ_LAT if FMT == "int8" else 0)  # last_i: FEED, LATCH, the wide reads, the fill count, the requantize
    if FMT == "int8":  # no per-element data here: every set costs the same (lane_kinds)
        return lane_cycles([0.0] * per_lane, act, start + 1) + 2
    out = []
    for k in range(P["NUM_SETS"]):
        xs = lane_inputs(k, P, collapse)
        out.append(max(lane_cycles(xs[L * per_lane:(L + 1) * per_lane], act, start + 1) for L in range(lanes)) + 2)
    return med(out)  # done_o, lane_collected, g_done, back in G_IDLE


def packed_stage(P: dict, collapse: bool = True) -> int:
    """Activation stage cycles of a packed set (median over the streamed sets): lane L runs column L % N's entry down its rows; ReLU and linear lanes do not bound it."""
    N, per_lane, lanes = P["N"], P["SRAM_DEPTH"] // P["NUM_LANES"], P["NUM_LANES"]
    start, knobs, out = per_lane + 3 + (REQ_LAT if FMT == "int8" else 0), _tb_knobs(), []
    for k in range(P["NUM_SETS"]):
        sw = _set_words(k, knobs, N, FMT)
        acts = [sw["ents"][int(sw["col_ent"][L % N])]["act"] for L in range(lanes)]
        if not any(a in LANE_ACTS for a in acts):  # every lane ReLU or linear: the bypass path
            out.append(per_lane + 1)
            continue
        xs = [0.0] * (N * N) if FMT == "int8" else lane_inputs(k, P, collapse)  # int8 has no tail path: data-free
        col = lambda L: [xs[((L // N) * per_lane + i) * N + L % N] for i in range(per_lane)]  # sienna_top's elem() for a packed set
        out.append(max(lane_cycles(col(L), a, start + 1) for L, a in enumerate(acts) if a in LANE_ACTS) + 2)
    return med(out)


def fmt_table(rows: list, cols: list) -> list:
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    line = "  " + "  ".join(str(c).rjust(w[i]) for i, c in enumerate(cols))
    out = [line, "  " + "  ".join("-" * x for x in w)]
    out += ["  " + "  ".join(str(r[i]).rjust(w[i]) for i in range(len(cols))) for r in rows]
    return out


def med(xs: list) -> int:
    return int(statistics.median(xs)) if xs else 0


def summary_cols() -> list:
    """The summary table's columns; int8 counts integer operations and TOPS where the floats count FLOP and GFLOPS."""
    op, rate = ("OP/cyc", "TOPS*") if FMT == "int8" else ("FLOP/cyc", "GFLOPS*")
    return ["config", "latency", "cycles/set", op, rate, "PE use", "limit", "its cycles",
            "host model", "mesh model", "mesh lat model", "mesh lat", "sim"]


def one_config(name: str, args) -> tuple:
    """(detail lines, summary row) for one regression config."""
    flop = 2 * args.n ** 3
    op = "OP" if FMT == "int8" else "FLOP"
    rate = lambda s: flop / s * args.clock_mhz / (1e6 if FMT == "int8" else 1000)  # TOPS in int8, GFLOPS otherwise
    rate_s = lambda s: f"{rate(s):.3f} TOPS" if FMT == "int8" else f"{rate(s):.1f} GFLOPS"
    cfg = next(t for t in PIPELINE_TESTS if t["name"] == name)
    res = bool(cfg.get("residual") or cfg.get("residual_every"))
    sets = cfg.get("num_sets") if res else args.sets  # a residual test keeps its own set count: its groups need not divide args.sets
    if args.reparse:  # the saved trace; only the stimulus is regenerated, for the model's geometry
        generate_vectors({**GEOM, **cfg, "num_sets": sets})
        raw = open(os.path.join(args.reparse, f"{name}_N{args.n}.log"), errors="ignore").read()
    else:
        raw = run(name, sets, args.build_dir)
    flags = [int(_tb_words(f"add_{k}.mem")[0]) for k in range(_tb_knobs()["num_sets"])] if res else None  # files past num_sets may be an earlier test's
    mdl = model(cfg, not args.slices)
    if "RESULT: PASSED" not in raw or mr.FIRED.search(raw.replace("[FAIL]", "")):
        return [f"--- {name}: SIMULATION DID NOT PASS; numbers omitted", ""], [name] + ["-"] * 11 + ["FAIL"]
    a = analyse(events(raw), len(flags) if res else args.sets, cfg.get("accum_passes", 1), flags)
    s = a["sets"]
    P = cfg.get("accum_passes", 1)
    first = next(x for x in s if x["mesh"] is not None)  # the first output's last set
    steady = statistics.mean(a["steady"]) if a["steady"] else 0
    # Each stage's own cost per set: the host's start interval; activation runs one set at a time; the mesh is
    # pipelined, so its cost is its tightest launch interval, not the time a set spends inside it. Activation and
    # pooling run once per output, so an accumulate config spreads them over its P sets.
    stage = {"host": min(a["host_gaps"] or [0]), "mesh": min(a["launch_gaps"] or [0]),
             "activation": round(med([x["act"] for x in s if x["act"] is not None]) / P), "pooling": round(med(a["pool"]) / P)}
    if res:  # the add sets' own stage, which the model gives, not spread over the group's passes
        stage["activation"] = med([x["act"] for x in s if x.get("add")])
    lim = max(stage, key=stage.get)
    L = [f"--- {name}", ""]
    dash = lambda v: "-" if v is None else v  # a partial-sum set has no output of its own
    L += fmt_table([[k, x["load"], x["wait_mesh"], x["mesh"], x["wait_act"], x["act"], dash(x["out"]), dash(x["latency"])]
                    for k, x in enumerate(s)],
                   ["set", "load", "wait", "mesh", "wait", "activ", "pool+out", "latency"])
    L += ["",
          f"  Stage cost per set: host load {stage['host']}, mesh launch interval min {stage['mesh']} median "
          f"{med(a['launch_gaps'])}, activation {stage['activation']}, pooling {stage['pooling']} (cycles)",
          f"  Mesh inside  : reduce start to result written {med(a['reduce_to_written'])} cycles (median)"
          + (f"; GPNAE lanes busy {100 * statistics.median(a['lanes']):.0f}% of a round" if a["lanes"] else ""),
          f"  Completion gaps: {a['gaps']}",
          f"  Steady state : {steady:.1f} cycles per set (mean of the last {len(a['steady'])} output intervals"
          + (f", each over its {cfg['accum_passes']} depth passes" if cfg.get("accum_passes", 1) > 1 else "")
          + f"); first-output latency {a['first_latency']} cycles",
          f"  Limit        : {lim} ({stage[lim]} cycles per set)",
          f"  Design model : host {mdl['host']} (measured {stage['host']}), mesh interval {mdl['mesh']} (measured min "
          f"{stage['mesh']}, host-bound), mesh latency {mdl['mesh_lat']} (measured {first['mesh']} on the first output"
          + (f", over its {P} depth passes" if P > 1 else "") + "), "
          + f"activation {mdl.get('act', '-')} (measured {stage['activation']})"
          + f", pooling dispatch {mdl['pool_dispatch']} of the measured {stage['pooling']}"
          + (f", hold set {mdl['hold']} (measured {med(a['hold'])})" if res and "hold" in mdl else ""),
          f"  Matmul rate  : {flop} {op} per set -> {flop / steady:.1f} {op}/cycle, {rate_s(steady)}"
          f" at an ASSUMED {args.clock_mhz:.0f} MHz, {100 * flop / 2 / steady / args.n ** 2:.1f}% of the mesh's"
          f" {args.n ** 2} MAC/cycle" if steady else "", ""]
    rates = [f"{steady:.1f}", f"{flop / steady:.1f}", f"{rate(steady):.3f}" if FMT == "int8" else f"{rate(steady):.1f}",
             f"{100 * flop / 2 / steady / args.n ** 2:.0f}%"] if steady else ["-"] * 4  # one output (int8_residual_one_pair): no interval
    row = [name, a["first_latency"], *rates, lim, stage[lim], mdl["host"], mdl["mesh"], mdl["mesh_lat"], first["mesh"], "pass"]
    return L, row


def header(args) -> list:
    return ["=" * 100, " SIENNA PIPELINE PERFORMANCE (measured in simulation, cycles)", "=" * 100,
            f" Generated {time.strftime('%Y-%m-%d %H:%M')}  N={args.n}  TILE={args.tile_size}  lanes={args.lanes}  "
            f"{args.sets} streamed sets per config, streaming host (one row of N words per operand per cycle)",
            " Every number is measured from TB_sienna_top's PERF trace. Per set: load = host rows, wait = to mesh launch,",
            " mesh = launch to result written (sets overlap inside it), wait = to the activation stage, activ = activation",
            " stage occupancy (a partial sum, summed in the PEs, passes as a null), pool+out = to the set's completion pulse.",
            " A residual test's hold set passes the stage into hold_mem with no output; its activation is the add sets' stage.", ""]


def footer(args, summary: list) -> list:
    L = ["=" * 100, " SUMMARY", "=" * 100]
    L += fmt_table(summary, summary_cols())
    L += [" latency = host start of the first output's first set to its completion pulse, on an idle pipeline; cycles/set",
          " counts each depth pass of an accumulate config as a set. host/mesh model: cycles per set the",
          " design needs (host N+3 is TB_sienna_top's handshake; the mesh alone needs max(T+2, K, T^2)).",
          " mesh lat model: launch to result written, 3T+K+T^2+LAT+16; mesh lat: the same, measured on set 0."]
    unit = "TOPS (2 x MACs per second)" if FMT == "int8" else "GFLOPS"
    L += [f" * {unit} at an ASSUMED {args.clock_mhz:.0f} MHz clock, not a timing result; they count the set's"
          f" {2 * args.n ** 3}-{'OP' if FMT == 'int8' else 'FLOP'} matmul only.",
          " PE use = multiply-accumulates per cycle over the mesh's N^2. limit = the stage with the largest cost per set."]
    return L


def perf_main(argv=None) -> None:
    """Cycle, latency and throughput analysis of the streamed SIENNA pipeline from TB_sienna_top's PERF trace."""
    ap = argparse.ArgumentParser(description=perf_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="perf")
    ap.add_argument("--sets", type=int, default=24, help="streamed sets; 24 fits every accumulate and mixed pattern")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--clock-mhz", type=float, default=950.0,
                    help="assumed clock for GFLOPS; no timing run has demonstrated one")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--build-dir", default=os.environ.get("PERF_BUILD_DIR", os.path.join(ROOT, "Verilator_perf")))
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--reparse", help="directory of saved traces (<config>_N<n>.log) to analyse instead of simulating")
    ap.add_argument("--slices", action="store_true", help="the build has COLLAPSE_K=0 (depth slices); for the model only")
    ap.add_argument("--merge", nargs="*", help="combine the .json parts of earlier runs into --report, in order")
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(FMT_KNOBS), help="number format of the build (its define)")
    args = ap.parse_args(argv)
    perf_init()
    global MUL_LAT, ADD_LAT, FMT
    MUL_LAT, ADD_LAT = UNIT_LAT[args.fmt_name]
    FMT = args.fmt_name
    fmts = {t["name"]: t.get("formats", tuple(UNIT_LAT)) for t in PIPELINE_TESTS}
    args.configs = [c for c in args.configs if args.fmt_name in fmts[c]]  # int8-only tests run only in int8
    GEOM.update(n=args.n, tile_size=args.tile_size, lanes=args.lanes, fmt_name=args.fmt_name)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)  # the .json part lands there first
    if args.merge:
        parts = [json.load(open(f)) for f in args.merge]
        L = header(args) + [x for p in parts for x in p["lines"]] + footer(args, [r for p in parts for r in p["rows"]])
    else:
        lines, rows = [], []
        for name in args.configs:
            d, r = one_config(name, args)
            lines += d
            rows.append(r)
            print("\n".join(d), flush=True)
        json.dump({"lines": lines, "rows": rows}, open(os.path.splitext(args.report)[0] + ".json", "w"))
        L = header(args) + lines + footer(args, rows)
    open(args.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-len(rows if not args.merge else parts) - 8:]))
    print(f"\nReport: {args.report}")
    failed = [r[0] for r in (rows if not args.merge else [r for p in parts for r in p["rows"]]) if r[-1] == "FAIL"]
    if failed:  # a config that did not pass fails the action, so sweeps see it
        print(f"[ERROR] perf: {len(failed)} config(s) did not pass: {', '.join(failed)}")
        sys.exit(1)


# ── oracle (was tflite_oracle.py) ────────────────────────────────────────────

tf = tflite = FUSED = ACT_FN = REF = None  # TensorFlow, the tflite reader and what needs them; _tf_imports() sets them


def _tf_imports() -> None:
    """TensorFlow and the tflite flatbuffer reader, for oracle, pack-models and gpnae-tflite only: regression.py imports without them."""
    global tf, tflite, FUSED, ACT_FN, REF
    import tensorflow as tf
    import tflite
    FUSED = {"none": tflite.ActivationFunctionType.NONE, "relu": tflite.ActivationFunctionType.RELU,
             "relu6": tflite.ActivationFunctionType.RELU6}
    ACT_FN = {"none": tf.identity, "relu": tf.nn.relu, "relu6": tf.nn.relu6}
    REF = tf.lite.experimental.OpResolverType.BUILTIN_REF


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
OP_NAME = {"fc": "FULLY_CONNECTED", "conv": "CONV_2D"}
MIN_UNCLAMPED = 0.25  # per model: the outputs must not be dominated by saturated ones
N_SAVED = 64  # test inputs and interpreter outputs kept per npz


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
    p["amin"], p["amax"] = ipu.activation_clamp(act, float(p["out_scale"]), p["out_zp"])
    return p


def layer_sums(p, xs, folded=False):
    """The layer's exact int32 sums on input codes xs, as (rows, channels); folded is SIENNA's zero-point algebra."""
    if p["layer"] == "fc":
        acc = mr.fc_acc(xs, p["w_q"], p["b_q"], p["in_zp"], folded=folded)
    else:
        acc = mr.conv2d_acc(xs, p["w_q"], p["b_q"], p["in_zp"], folded=folded)
    return acc.reshape(-1, p["w_q"].shape[0])


def test_inputs(p, lo, hi, n, rng):
    """Mostly the calibration distribution quantized with the model's input parameters; a tenth uniform over all of int8."""
    shape = (n,) + p["in_shape"][1:]
    xq = np.clip(np.round(rng.uniform(lo, hi, shape) / p["in_scale"]) + p["in_zp"], -128, 127).astype(np.int64)
    k = n // 10
    xq[:k] = rng.integers(-128, 128, (k,) + shape[1:])
    return xq


def pick_saved(differ_in, k, n):
    """Indices of the saved inputs: those whose interpreter outputs differ most from the generic model first (up to n), then alternately
    calibration and uniform ones."""
    top = [int(i) for i in np.argsort(-differ_in, kind="stable") if differ_in[i] > 0][:n]
    rest = [i for i in range(differ_in.size) if i not in set(top)]
    fill = [i for pair in itertools.zip_longest([i for i in rest if i >= k], [i for i in rest if i < k]) for i in pair if i is not None]
    return np.array(sorted(top + fill[:n - len(top)]))


def save(out, name, model, p, xs, ys):
    with open(os.path.join(out, name + ".tflite"), "wb") as f:
        f.write(model)
    np.savez(os.path.join(out, name + ".npz"), layer=p["layer"], activation=p["activation"], in_shape=np.array(p["in_shape"]),
             w_q=p["w_q"].astype(np.int8), b_q=p["b_q"].astype(np.int32), w_scales=p["w_scales"], in_scale=p["in_scale"],
             in_zp=p["in_zp"], out_scale=p["out_scale"], out_zp=p["out_zp"], act_min=p["amin"], act_max=p["amax"],
             x_test=xs.astype(np.int8), y_test=ys.astype(np.int8), tf_version=tf.__version__)


def oracle_main(argv=None) -> None:
    """The single-layer int8 test vehicles: TensorFlow converts each model, the interpreter (BUILTIN_REF) runs it, and every output of the
    interpreter and of the generic bit-exact model (encode_scale, ipu.requantize) is judged on the true function of its exact int32 sum;
    the generic model must pass, the interpreter's misses and the two's disagreements are reported. Seed 0 of each kind is saved."""
    ap = argparse.ArgumentParser(description=oracle_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="oracle")
    ap.add_argument("--out", required=True, help="directory for the models and npz files")
    ap.add_argument("--report", required=True, help="report (.log)")
    ap.add_argument("--seeds", type=int, default=8, help="models per kind; seed 0 is saved")
    ap.add_argument("--inputs", type=int, default=1000, help="random test inputs per model")
    a = ap.parse_args(argv)
    _tf_imports()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    lines = [f"int8 layer models: TensorFlow {tf.__version__}, numpy {np.__version__}, op resolver BUILTIN_REF, "
             f"{a.seeds} seeds x {len(MODELS)} models, {a.inputs} inputs each"]
    tot = {"outputs": 0, "interp_miss": 0, "model_miss": 0, "differ": 0, "folded": 0}
    problems, keep = [], {}
    for name, (layer, act) in MODELS.items():
        for seed in range(a.seeds):
            model, lo, hi = build_model(layer, act, seed, OUT_RANGE.get(name))
            it = interpreter(model, REF)
            p = extract(it, model, layer, act)
            xs = test_inputs(p, lo, hi, a.inputs, np.random.default_rng(seed + 2000))
            ys_raw = invoke_all(it, xs)
            ys = ys_raw.reshape(-1, p["w_q"].shape[0])
            acc = layer_sums(p, xs)
            fold = int(np.sum(layer_sums(p, xs, folded=True) != acc))
            S = mr.layer_scales(p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0])
            m, e = mr.encode_scales(S)
            ym = mr.requantize(acc, dict(mult=m, shift=e, zp=p["out_zp"], amin=p["amin"], amax=p["amax"]))
            Sx = np.broadcast_to(S[None, :], acc.shape)
            r_i = truth.gate_requantize(acc, Sx, p["out_zp"], p["amin"], p["amax"], ys)
            r_m = truth.gate_requantize(acc, Sx, p["out_zp"], p["amin"], p["amax"], ym)
            differ_in = (ys != ym).reshape(len(xs), -1).sum(axis=1)
            unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
            per_ch = p["w_scales"].size > 1
            lines.append(f"{name} seed {seed}: in scale {p['in_scale']:.6g} zp {p['in_zp']}, out scale {p['out_scale']:.6g} "
                         f"zp {p['out_zp']}, act [{p['amin']}, {p['amax']}], weights {'per-channel' if per_ch else 'per-tensor'} "
                         f"({p['w_scales'].size}), outputs {ys.size}: generic model {r_m.over} over the rule (worst {r_m.worst:.3f} steps), "
                         f"interpreter {r_i.over} over (worst {r_i.worst:.3f}), {int(differ_in.sum())} differ between the two, "
                         f"folded {fold}, unclamped {100 * unclamped:.1f}%")
            if not r_m.ok:
                problems.append(f"{name} seed {seed}: the generic model fails the true function on {r_m.over} outputs")
            if p["in_zp"] == 0:
                problems.append(f"{name} seed {seed}: input zero point is 0")
            if layer == "conv" and (not per_ch or np.unique(p["w_scales"]).size < 2):
                problems.append(f"{name} seed {seed}: conv weights are not per-channel")
            if unclamped < MIN_UNCLAMPED:
                problems.append(f"{name} seed {seed}: only {100 * unclamped:.1f}% of outputs unclamped")
            tot["outputs"] += ys.size
            tot["interp_miss"] += r_i.over
            tot["model_miss"] += r_m.over
            tot["differ"] += int(differ_in.sum())
            tot["folded"] += fold
            if seed == 0:
                idx = pick_saved(differ_in, a.inputs // 10, N_SAVED)
                lines.append(f"  saved {name}: {idx.size} inputs ({int(np.sum(idx < a.inputs // 10))} uniform int8, "
                             f"{int(np.sum(idx >= a.inputs // 10))} calibration), {int(differ_in[idx].sum())} outputs where the interpreter and the "
                             f"generic model differ")
                keep[name] = (model, p, xs[idx], ys_raw[idx])
    lines.append(f"totals: outputs {tot['outputs']}, generic model over the rule {tot['model_miss']}, interpreter over the rule "
                 f"{tot['interp_miss']} (reported), interpreter and generic model differ {tot['differ']} (reported), folded {tot['folded']}")
    if tot["folded"]:
        problems.append(f"SIENNA's folded zero-point algebra differs from the plain sums on {tot['folded']} outputs")
    if problems:
        lines += [f"problem: {m}" for m in problems] + ["MODELS: FAIL"]
    else:
        for name, (model, p, xs, ys) in keep.items():
            save(a.out, name, model, p, xs, ys)
        lines += ["MODELS: PASS"]
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


# ── pack-models (was tflite_pack_models.py) ──────────────────────────────────

PACK_MODELS = {  # name: (layer, act, inputs, outputs); the first three fit blocks of 8, the last two blocks of 16
    "fc8x8_linear": ("fc", "none", 8, 8),
    "fc8x4_relu": ("fc", "relu", 8, 4),
    "fc6x8_relu6": ("fc", "relu6", 6, 8),
    "conv3x3_6x6x1x8_linear": ("conv", "none", 1, 8),
    "fc16x16_relu": ("fc", "relu", 16, 16),
}


def pack_models_main(argv=None) -> None:
    """Five small int8 TFLite layers for packing, saved as the model generator (--action oracle) saves its models; arg: output directory."""
    ap = argparse.ArgumentParser(description=pack_models_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="pack-models")
    ap.add_argument("out", help="output directory")
    out = ap.parse_args(argv).out
    _tf_imports()
    global FC_IN, FC_OUT, CONV_CIN, CONV_COUT, CONV_HW
    os.makedirs(out, exist_ok=True)
    for name, (layer, act, cin, cout) in PACK_MODELS.items():
        FC_IN, FC_OUT, CONV_CIN, CONV_COUT, CONV_HW = cin, cout, cin, cout, 6
        model, lo, hi = build_model(layer, act, 0)
        it = interpreter(model, REF)
        p = extract(it, model, layer, act)
        xs = test_inputs(p, lo, hi, 32, np.random.default_rng(2000))
        ys = invoke_all(it, xs)
        unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
        save(out, name, model, p, xs, ys)
        print(f"{name}: in zp {p['in_zp']} out zp {p['out_zp']} act [{p['amin']}, {p['amax']}] unclamped {100 * unclamped:.1f}% "
              f"y [{ys.min()}, {ys.max()}] w_scales {p['w_scales'].size}")


# ── gpnae-tflite (was gpnae_int8_tflite.py) ──────────────────────────────────

GPTFL_OPS = (("tanh", 3, "tanh"), ("sigmoid", 2, "logistic"))  # (truth.py function, GPNAE control word, TFLite op)


LANE_TOL = gpnae_model.TIER_TOL[0]  # the reset image's tier 0: the lanes gate at 1%


def gpnae_tflite_judge(rows):
    """Report lines and pass flag: each row (func, op, quant, q, hw, tfl, ex) gated on truth.py, TFLite agreement characterization only."""
    L = ["GPNAE int8 lane (gpnae_model, bit-exact with the RTL at G2) on every int8 input, quantization from TFLite's converter (BUILTIN_REF)",
         "Gate: truth.py, the lane's words against the true function of the real input (bound "
         f"max({100 * LANE_TOL:g}% |t|, one output step), target {100 * truth.TARGET:g}%); TFLite agreement is characterization, not gated",
         f"{'op':<9}{'s_in':>11}{'z_in':>6}{'truth':>6}{'worst':>9}{'LSB':>4}{'>bound':>7}{'>1%':>5}"
         f"{'equal':>7}{'|d|=1':>7}{'max |d|':>8}{'TFLite-exact':>13}{'lane-exact':>11}"]
    res = {}
    for func, op, quant, q, hw, tfl, ex in rows:
        r = truth.gate("int8", func, q, hw, quant=quant, bound=LANE_TOL)
        res.setdefault(func, []).append(r)
        d = np.abs(hw - tfl)
        L.append(f"{op:<9}{quant[0]:>11.6f}{quant[1]:>6}{'PASS' if r.ok else 'FAIL':>6}{truth.worst_text(r.worst):>9}{r.worst_lsb:>4}{r.over:>7}"
                 f"{r.over_target:>5}{int((d == 0).sum()):>7}{int((d == 1).sum()):>7}{int(d.max()):>8}"
                 f"{int(np.abs(tfl - ex).max()):>13}{int(np.abs(hw - ex).max()):>11}")
        L += [f"  miss: q {m[1]} lane {m[2]} true {m[3]}" for m in r.first]
    if not res:
        return L + ["ACCURACY: FAIL (no gated case)"], False
    merged = {f: truth.merge(rs) for f, rs in res.items()}
    L += [truth.line(m) for m in merged.values()]
    ok = all(m.ok for m in merged.values())
    body = "; ".join(f"{f} worst {truth.worst_text(m.worst)} ({m.worst_lsb} LSB), {m.over} of {m.n} over {100 * LANE_TOL:g}%, "
                     f"{m.over_target} over {100 * truth.TARGET:g}%" for f, m in merged.items())
    L.append(f"ACCURACY: {'PASS' if ok else 'FAIL'} (true function, truth.py, int8; bound {100 * LANE_TOL:g}%, target {100 * truth.TARGET:g}%: {body})")
    agree = "; ".join(f"{op} {sum(int((r[4] == r[5]).sum()) for r in rows if r[1] == op)} of {sum(r[3].size for r in rows if r[1] == op)} equal, "
                      f"max |d| {max(int(np.abs(r[4] - r[5]).max()) for r in rows if r[1] == op)}" for op in dict.fromkeys(r[1] for r in rows))
    L.append(f"TFLITE AGREEMENT (characterization, not gated): {agree}")
    return L, ok


def gpnae_tflite_main(argv=None):
    """GPNAE's int8 tanh and sigmoid lane on TFLite-derived scales, gated on the true function (truth.py); TFLite's TANH and LOGISTIC agreement is reported, not gated."""
    p = argparse.ArgumentParser(description=gpnae_tflite_main.__doc__)
    p.add_argument("--action", choices=ACTIONS, default="gpnae-tflite")
    p.add_argument("--report", required=True)
    a = p.parse_args(argv)
    _tf_imports()
    lane = gpnae_model.Lane(gpnae_model.INT8)  # the lanes' reset coefficient memory, tier 0
    q = np.arange(-128, 128, dtype=np.int64)
    rows = []
    for func, code, op in GPTFL_OPS:
        for case in gpnae_model.INT8_CASES[func]:
            tfl, s, z = activation_int8(op, case.s_in, case.z_in)  # asserts TFLite's output quantization is the lane's (case.s_out, case.z_out)
            c = case._replace(s_in=s, z_in=z)
            hw = lane.run(q, code, gpnae_model.int8_params(c, code))
            rows.append((func, op, (s, z, c.s_out, c.z_out), q, hw, tfl, gpnae_model.exact_int8(q, code, c)))
    L, ok = gpnae_tflite_judge(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if not ok:
        sys.exit(1)


# ── rq-vectors (was testbenches/gen_rq_lanes.py) ─────────────────────────────

RQ_N, RQ_LANES, RQ_SETS = 16, 32, 6  # TB_requant_lanes' geometry
RQ_PER = RQ_N * RQ_N // RQ_LANES
PACKED_SETS = (1, 2, 5)  # set 3 is unpacked at an odd multiple of RQ_PER beats, so a beat counter that ignores clear_i is caught


def _lane_codes(path: str, beats: int, lanes: int) -> np.ndarray:
    """A lane bench's output file (one line per output beat, each lane's code in hex, lane 0 first) as signed codes; ValueError on another shape."""
    rows = [ln.split() for ln in open(path) if ln.strip()]
    if len(rows) != beats or any(len(r) != lanes for r in rows):
        raise ValueError(f"{path}: {len(rows)} output beats of {sorted({len(r) for r in rows})} lanes, the vectors have {beats} of {lanes}")
    return truth.int8_codes(np.array([fourstate.words(r, os.path.basename(path), start=k * lanes) for k, r in enumerate(rows)], np.int64))


def _bench_check(name: str, gate, out: str, beats: int, lanes: int) -> None:
    """A lane bench's true-function gate on its output file: the stage line, then '<NAME>: PASS' or FAIL; exit 1 unless it passed."""
    try:
        r = gate(_lane_codes(out, beats, lanes))
        print(truth.stage_line(r))
        for m in r.first:
            print(f"  miss (index, inputs..., got, true): {m}")
        ok, why = r.ok, f"{r.n} outputs, {r.over} over the bound"
    except fourstate.Unknown as e:
        ok, why = False, str(e)
        print(fourstate.fail_line(e))
    except (OSError, ValueError) as e:
        ok, why = False, str(e)
    print(f"{name}: {'PASS' if ok else 'FAIL'} ({why})")
    sys.exit(0 if ok else 1)


def rq_vectors_main(argv=None) -> None:
    """Vectors for TB_requant_lanes through its channel map: real scales per channel, their words from ipu.encode_scale, the bit-exact codes from
    ipu.requantize (reported by the bench); --truth keeps the true function's inputs, --check gates the bench's outputs on it (truth.gate_requantize)."""
    ap = argparse.ArgumentParser(description=rq_vectors_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="rq-vectors")
    ap.add_argument("path", nargs="?", help="output .mem path")
    ap.add_argument("--truth", help="the true function's inputs (JSON): written with the vectors, read by --check")
    ap.add_argument("--check", metavar="OUT", help="gate TB_requant_lanes' outputs (rq_lanes_out.mem) on the true function of --truth")
    a = ap.parse_args(argv)
    if a.check:
        if not a.truth:
            ap.error("--check needs --truth")
        t = json.load(open(a.truth))
        acc, scale = (np.array([x[k] for x in t["sets"]], np.float64).reshape(-1, RQ_LANES) for k in ("acc", "scale"))
        zp, lo, hi = (np.repeat(np.array([x[k] for x in t["sets"]], np.int64), RQ_PER, axis=0) for k in ("zp", "lo", "hi"))
        _bench_check("REQUANT LANES", lambda q: truth.gate_requantize(acc.astype(np.int64), scale, zp, lo, hi, q), a.check, acc.shape[0], RQ_LANES)
    if not a.path:
        ap.error("the output .mem path is needed without --check")
    path = a.path
    rng = np.random.RandomState(8)
    out, sets = [RQ_SETS], []
    for s in range(RQ_SETS):
        e = (16 + (s * RQ_N + np.arange(RQ_N)) % 40).astype(np.int64)  # every right shift 16..55 across the sets
        real = rng.uniform(1.0, 2.0, RQ_N) * np.ldexp(1.0, 23 - e)  # each channel's real scale, its mantissa normalized at shift e
        if s == 0:
            real[0], real[1] = np.ldexp(float((1 << 24) - 1), -int(e[0])), np.ldexp(float(1 << 23), -int(e[1]))  # the largest mantissa and the smallest
        mult, shift = mr.encode_scales(real)
        zp = int(rng.randint(-128, 128))
        amin, amax = [(-128, 127), (zp, 127), (zp, min(127, zp + 50)), (-128, 127), (-40, 40), (-128, zp)][s]
        amin, amax = min(amin, amax), max(amin, amax)
        small = rng.randint(-(1 << 15), 1 << 15, (RQ_PER, RQ_LANES))
        full = rng.randint(-(1 << 31), (1 << 31) - 1, (RQ_PER, RQ_LANES))
        acc = np.where(rng.rand(RQ_PER, RQ_LANES) < 0.5, small, full).astype(np.int64)
        if s == 0:
            acc[0, :4] = [-(1 << 31), (1 << 31) - 1, 0, -1]
        packed = s in PACKED_SETS  # lane k takes column k % RQ_N, and its block's zero point and clamp
        c = (np.arange(RQ_LANES)[None, :] % RQ_N + 0 * np.arange(RQ_PER)[:, None]) if packed else \
            (np.arange(RQ_LANES)[None, :] * RQ_PER + np.arange(RQ_PER)[:, None]) % RQ_N  # channel of lane k at beat b
        if packed:
            ents = []
            for _ in range(4):  # zero point, then a clamp low <= high
                z = int(rng.randint(-128, 128))
                lo_, hi_ = sorted(int(v) for v in rng.randint(-128, 128, 2))
                ents.append((z, lo_, hi_))
            blk = (np.arange(RQ_LANES) % RQ_N) // (RQ_N // 4)  # four blocks of RQ_N / 4 columns
            zpL = np.array([ents[e][0] for e in blk]); aminL = np.array([ents[e][1] for e in blk]); amaxL = np.array([ents[e][2] for e in blk])
        else:
            zpL, aminL, amaxL = np.full(RQ_LANES, zp), np.full(RQ_LANES, amin), np.full(RQ_LANES, amax)
        want = np.stack([ipu.requantize(acc[:, k], mult[c[:, k]], shift[c[:, k]], int(zpL[k]), int(aminL[k]), int(amaxL[k]))
                         for k in range(RQ_LANES)], axis=1)
        out += [int(packed)] + zpL.tolist() + aminL.tolist() + amaxL.tolist() + mult.tolist() + shift.tolist()
        for b in range(RQ_PER):
            out += acc[b].tolist() + [int(v) for v in want[b]]
        sets.append({"acc": acc.tolist(), "scale": real[c].tolist(), "zp": [int(v) for v in zpL], "lo": [int(v) for v in aminL],
                     "hi": [int(v) for v in amaxL]})
    with open(path, "w") as f:
        f.write("".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in out))
    if a.truth:
        json.dump({"sets": sets}, open(a.truth, "w"))


# ── add-vectors: TB_add_lanes' vectors ────────────────────────────────────────

ADD_LANES, ADD_SETS, ADD_PER = 32, 6, 8  # TB_add_lanes' geometry: lanes, sets, beats per set
ADD_EXP = {"none": (-8, 1), "relu": (-8, 1), "relu6": (-7, -5)}  # input scales' log2 range: relu6's 6.0 within a few input codes' sums
ADD_ZIN = {"none": 128, "relu": -32, "relu6": -32}  # input zero points below this: a ReLU set's inputs mostly positive, off its zp_o rail
ADD_ZOUT = {"none": 128, "relu": 1, "relu6": -63}  # zp_o below this, so 127 stays above a ReLU's or ReLU6's outputs


def _add_set_draw(rng, s: int) -> tuple:
    """Set s's activation, scales (s1, s2, s_out) and zero points (zp1, zp2, zp_o): s_out is 2 max(s1, s2) 2^U(-2, 1)."""
    act = ("none", "relu", "relu6")[s % 3]
    s1, s2 = (float(np.float32(2.0 ** x)) for x in rng.uniform(*ADD_EXP[act], 2))
    s_out = float(np.float32(2.0 * max(s1, s2) * 2.0 ** rng.uniform(-2, 1)))
    return act, [s1, s2, s_out], [int(v) for v in rng.integers(-128, ADD_ZIN[act], 2)] + [int(rng.integers(-128, ADD_ZOUT[act]))]


def add_vectors_main(argv=None) -> None:
    """Vectors for TB_add_lanes: per set its ADD words, then per beat a random tag and each lane's q1, q2 and bit-exact code (ipu.qadd, reported by
    the bench); --truth keeps the true function's inputs, --check gates the bench's outputs on it (truth.gate_quantized_add)."""
    ap = argparse.ArgumentParser(description=add_vectors_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="add-vectors")
    ap.add_argument("out", nargs="?", help="output .mem path")
    ap.add_argument("--truth", help="the true function's inputs (JSON): written with the vectors, read by --check")
    ap.add_argument("--check", metavar="OUT", help="gate TB_add_lanes' outputs (add_lanes_out.mem) on the true function of --truth")
    a = ap.parse_args(argv)
    if a.check:
        if not a.truth:
            ap.error("--check needs --truth")
        t = json.load(open(a.truth))["sets"]
        q1, q2 = (np.array([x[k] for x in t], np.int64).reshape(-1, ADD_LANES) for k in ("q1", "q2"))
        r1, r2, z1, z2, zo, lo, hi = (np.repeat(np.array([x[k] for x in t], np.float64), ADD_PER)[:, None] for k in ("r1", "r2", "z1", "z2", "zo", "lo", "hi"))
        _bench_check("ADD LANES", lambda q: truth.gate_quantized_add(q1, q2, r1, z1, r2, z2, 1.0, zo, lo, hi, q), a.check, q1.shape[0], ADD_LANES)
    if not a.out:
        ap.error("the output .mem path is needed without --check")
    rng = np.random.default_rng(29)
    lines, sets = [], []
    for s in range(ADD_SETS):
        for _ in range(ADD_TRIES):  # redrawn until half the set's outputs are off the clamp rails
            act, sc, z = _add_set_draw(rng, s)
            add = mr.add_words(sc[0], z[0], sc[1], z[1], sc[2], z[2], act)
            q1, q2 = rng.integers(-128, 128, (2, ADD_PER, ADD_LANES))
            want = mr.qadd_codes(q1, q2, add)
            off = np.mean((want != add["amin"]) & (want != add["amax"]))
            if off >= 0.5:
                break
        else:
            raise ValueError(f"add-vectors: set {s} ({act}): no draw in {ADD_TRIES} with half its outputs off the clamp rails")
        lines.append(" ".join(f"{add[k] & 0xFFFFFFFF:08x}" for k in mr.ADD_FIELDS))
        lo, hi = mr.true_clamp(act, sc[2], z[2])  # the fused activation's real range on the output, unrounded
        sets.append({"q1": q1.tolist(), "q2": q2.tolist(), "r1": add["ratios"][0], "r2": add["ratios"][1], "z1": z[0], "z2": z[1], "zo": z[2],
                     "lo": lo, "hi": hi})
        for b in range(ADD_PER):
            lines.append(f"{int(rng.integers(0, 1 << 16)):04x} " + " ".join(f"{int(x) & 0xFF:02x}{int(y) & 0xFF:02x}{int(w) & 0xFF:02x}"
                                                                         for x, y, w in zip(q1[b], q2[b], want[b])))
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if a.truth:
        json.dump({"sets": sets}, open(a.truth, "w"))
    print(f"{a.out}: {ADD_SETS} sets x {ADD_PER} beats x {ADD_LANES} lanes, bit-exact words from ipu.qadd" + (f"; true function's inputs {a.truth}" if a.truth else ""))


# ── Self-tests ───────────────────────────────────────────────────────────────
# (were test_makefile_fmt.py, test_pack_jobs.py, test_perf_analysis.py, test_regression_parse.py, test_tflite_ref.py)

reg = pa = sys.modules[__name__]  # the moved tests name this module reg or pa
SELFTESTS = []  # the registry: --action selftest runs exactly these, in order, whatever else is named test_*


def selftest(fn):
    """Registers fn as a self-test."""
    SELFTESTS.append(fn)
    return fn


def perf_selftest(fn):
    """Registers a self-test of the perf section; perf_init() runs first, as at perf's entry."""

    @functools.wraps(fn)
    def run():
        perf_init()
        fn()

    return selftest(run)


# ── tool: Makefile FMT flow (was test_makefile_fmt.py) ──

MAKE_ENV = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", "MAKEOVERRIDES")}  # an outer make's variables would leak in


def make_n(*args) -> subprocess.CompletedProcess:
    """make -n from the repo root: prints recipes, runs nothing but parse-time checks."""
    return subprocess.run(["make", "-s", "-n", *args], cwd=ROOT, env=MAKE_ENV, capture_output=True, text=True)


def fixture_pkg(d: str, fmt: str, n: int = 16, tile: int = 4, lanes: int = 32, drop: tuple = (), edit: dict = None) -> str:
    """A fixture test_config_pkg.sv in d, written by regression.py's own writer, minus the localparams in drop, with edit's values."""
    path = os.path.join(d, f"pkg_{fmt}_{n}_{tile}_{lanes}{'_old' if drop or edit else ''}.sv")
    items = reg._config_items({"n": n, "tile_size": tile, "lanes": lanes}, fmt, "relu", 4, 15, 1, [], False, 1)
    reg.write_sv_package(path, [(k, (edit or {}).get(k, v), t) for k, v, t in items if k not in drop])
    return path


def pkg_guard(pkg: str, *args) -> tuple:
    """(exit, output) of the shell make -n prints for pkg-geometry GEN_PKG=0 on pkg."""
    r = make_n("pkg-geometry", "GEN_PKG=0", f"PKG_FILE={pkg}", *args)
    assert r.returncode == 0, r.stderr
    g = subprocess.run(["bash", "-c", r.stdout], capture_output=True, text=True)
    return g.returncode, g.stdout + g.stderr


@selftest
def test_busy_writes_hit_the_running_activation():
    g = gpnae_model  # a landed +coef_busy write must change the running activation's own words, so a bypassed interlock shows in its outputs
    for fmt in reg.FORMATS:
        for act, d in (("tanh", g.D_TANH), ("selu", g.D_SELU), ("sigmoid", g.D_SIGP)):
            m = g.default_mem(fmt)
            w = mr.busy_writes(m, act, 0)
            ds = m.desc[0][d]
            b = g.A_DESC + g.DESC_STRIDE * d
            assert len(w) == 4 and all(ds.base <= a < ds.base + ds.np * (ds.deg + 1) or b <= a < b + g.DESC_STRIDE for a, _ in w), (fmt, act, w)
            for a, v in w:  # each write alone changes the memory
                m1 = g.default_mem(fmt)
                m1.load(a, v)
                assert m1.to_dict() != m.to_dict(), (fmt, act, a)
            if fmt != "int8":
                x = mr.word_bits(np.linspace(-3, 3, 241).astype(np.float32), fmt)
                m2 = g.default_mem(fmt)
                for a, v in w[:1]:  # the first word alone moves outputs
                    m2.load(a, v)
                f = fpu.FORMATS[fmt]
                code = {"tanh": g.TANH, "selu": g.SELU, "sigmoid": g.SIGMOID}[act]
                assert np.sum(g.Lane(f, m2).run(x, code) != g.Lane(f, m).run(x, code)) > 0, (fmt, act)


@selftest
def test_guard_passes_any_format():
    with tempfile.TemporaryDirectory() as d:  # the package carries no format: one geometry's package serves every define
        for f in reg.FORMATS:
            for want in reg.FORMATS:
                rc, out = pkg_guard(fixture_pkg(d, f), f"FMT={want}")
                assert rc == 0 and "matches N=16 TILE=4 LANES=32 COLLAPSE_K=1" in out and f"+define+FMT_{want.upper()}" in out, (f, want, out)


@selftest
def test_package_has_no_format():
    for f in reg.FORMATS:  # the define gives the format and its widths: test_config_pkg must not, or a bench could read a stale one
        names = {k for k, _, _ in reg._config_items({}, f, "relu", 1, 15, 1, [], False, 1)}
        assert not names & {"EXP_W", "MAN_W", "IS_INT", "ACC_W", "OUT_W", "DATA_WIDTH"}, (f, names)
    mk = open(os.path.join(ROOT, "Makefile")).read()
    assert "FMT_FIELDS" not in mk and "pkg-check" not in mk and "include $(SM_LIB_DIR)/Common/fmt.mk" in mk


@selftest
def test_guard_checks_geometry():
    with tempfile.TemporaryDirectory() as d:
        big = fixture_pkg(d, "bf16", 32, 8)
        rc, out = pkg_guard(big, "FMT=bf16")  # a stale package of another size
        assert rc != 0 and "N=32 TILE_SIZE=8 NUM_LANES=32 COLLAPSE_K=1, but the build asks for N=16 TILE=4 LANES=32 COLLAPSE_K=1" in out, out
        rc, out = pkg_guard(big, "FMT=bf16", "N=32", "TILE=8")
        assert rc == 0 and "N=32 TILE=8 LANES=32" in out, out
        rc, out = pkg_guard(fixture_pkg(d, "int8"), "FMT=int8", "LANES=64")
        assert rc != 0 and "LANES=64" in out, out


def _synthetic_truth(mod=None) -> dict:
    """truth_test on a dump built from the bit-exact models for bf16 matmul_random_tanh_train at N=8 T=2, in a scratch TB_DIR;
    mod(tag, k, arrs) may plant a fault in one set's arrays (pre, post, X, A, P, keep, D) first."""
    t = next(x for x in PIPELINE_TESTS if x["name"] == "matmul_random_tanh_train")
    N, T, lanes, fmt = 8, 2, 32, "bf16"
    g = globals()
    saved = {k: g[k] for k in ("TB_DIR", "RESULTS_DIR", "TRUTH_DUMP", "TRUTH_SCALES")}
    with tempfile.TemporaryDirectory() as d:
        g.update(TB_DIR=d, RESULTS_DIR=os.path.join(d, "results"), TRUTH_DUMP=os.path.join(d, "truth_dump.txt"),
                 TRUTH_SCALES=os.path.join(d, "truth_scales.json"))
        try:
            cfg = {"n": N, "tile_size": T, "lanes": lanes, "host_words": N, "fmt_name": fmt, **t}
            with contextlib.redirect_stdout(io.StringIO()):
                generate_vectors(cfg)
            knobs, pkg = _tb_knobs(), _pkg_fields(os.path.join(d, "test_config_pkg.sv"))
            W = ((N + 2 * pkg["PADDING"] - pkg["POOL_H"]) // pkg["POOL_H"] + 1) ** 2
            p = pkg["DROPOUT_P_PERCENT"] / 100.0
            out = ["H b2b 0"]
            for tag in ("P0", "S1", "S2", "S3", "S4", "A"):
                ks = range(knobs["num_sets"]) if tag.startswith("S") else [0]
                for r, k in enumerate(ks):
                    Ab = _tb_words("matrix_west.mem" if tag == "P0" else f"matrix_west_{k}.mem").reshape(N, N)
                    Bb = _tb_words("matrix_north.mem" if tag == "P0" else f"matrix_north_{k}.mem").reshape(N, N)
                    seed = set_dropout_seed(knobs["dropout_seed"], k)
                    C, Aout, Pool, Dw = _golden_bits([(Ab, Bb)], None, cfg, "tanh", seed, fmt)
                    C = np.asarray(C, np.int64).ravel()
                    arrs = {"pre": C << 16, "post": C.copy(), "X": C.copy(), "A": np.asarray(Aout, np.int64).ravel(),
                            "P": np.asarray(Pool, np.int64).ravel(), "keep": dropout_keep(W, p, seed, lanes).astype(int),
                            "D": np.asarray(Dw, np.int64).ravel(), "Ab": Ab, "Bb": Bb}
                    if mod:
                        mod(tag, k, arrs)
                    out += [f"M {tag} {r} {e} {int(arrs['pre'][e]) & 0xFFFFFFFF:08x} {int(arrs['post'][e]):04x}" for e in range(N * N)]
                    out += [f"X {tag} {k} {e} {int(arrs['X'][e]):04x}" for e in range(N * N)]
                    out += [f"A {tag} {k} {e} {int(arrs['A'][e]):04x}" for e in range(N * N)]
                    out += [f"P {tag} {k} {w % lanes} {w // lanes} {int(arrs['P'][w]):04x} {int(arrs['keep'][w])}" for w in range(W)]
                    out += [f"D {tag} {k} {i} {int(arrs['D'][i]):04x}" for i in range(W)]
            with open(TRUTH_DUMP, "w") as fh:
                fh.write("\n".join(out) + "\n")
            return truth_test(N, fmt, lanes)
        finally:
            g.update(saved)


@selftest
def test_truth_synthetic_dump():
    # the models' own words pass every stage of every pass; each planted fault fails its own gate, in its own pass
    clean = _synthetic_truth()
    assert clean["_npass"] == 6 and all(r.ok for k, r in clean.items() if not k.startswith("_")), {k: r.ok for k, r in clean.items() if not k.startswith("_")}
    assert {"matmul", "narrow", "transfer", "activation tanh", "maxpool", "dropout"} <= set(clean), sorted(clean)

    def drop(tag, k, a):  # the first product of every element left out of set 3's sums in S2
        if (tag, k) == ("S2", 3):
            A, B = truth.values("bf16", a["Ab"]), truth.values("bf16", a["Bb"])
            a["pre"] = truth.encode("fp32", truth.values("fp32", a["pre"]) - np.outer(A[:, 0], B[0, :]).ravel())
            a["post"] = a["X"] = truth.narrow_ref("bf16", a["pre"])

    def trunc(tag, k, a):  # set 2's sums in S3 sit above half an ULP, and the narrowing truncates them
        if (tag, k) == ("S3", 2):
            a["pre"] = (a["post"] << 16) | 0xC000

    def swap(tag, k, a):  # two activation inputs of set 5 in S1 swapped
        if (tag, k) == ("S1", 5):
            i, j = [int(x) for x in np.flatnonzero(a["X"] != a["X"][0])[:1]] + [0]
            a["X"][[i, j]] = a["X"][[j, i]]

    def keep(tag, k, a):  # dropout's keep decision flipped on set 1's first nonzero window in S4
        if (tag, k) == ("S4", 1):
            w = int(np.flatnonzero(truth.values("bf16", a["P"]) != 0)[0])
            a["keep"][w] ^= 1

    for plant, stage, tag in ((drop, "matmul", "S2"), (trunc, "narrow", "S3"), (swap, "transfer", "S1"), (keep, "dropout", "S4")):
        res = _synthetic_truth(plant)
        assert not res[stage].ok and res["_passes"][stage] == {tag}, (plant.__name__, stage, res[stage].ok, res.get("_passes"))
        lines, bad = truth_report(res)
        assert any(x.startswith(f"[FAIL] truth: {stage} ") and f"in passes {tag};" in x for x in bad), (plant.__name__, bad)


def _tb_b2b_files() -> tuple:
    """The expected and bound files TB_sienna_top's back-to-back pass reads."""
    src = open(os.path.join(ROOT, "testbenches", "top", "TB_sienna_top.sv")).read()
    blk = src[src.index("[STAGE] BACK TO BACK"):]
    blk = blk[:blk.index("collect_outputs")]
    get = lambda q: re.search(r'read_mem_file\("([^"]+)",\s*' + q + r"\)", blk).group(1)
    return get("expected_results"), get("bound_results")


def _set1_alone(cfg: dict, fmt: str) -> tuple:
    """Set 1 as the back-to-back pass drives it, from its files in TB_DIR: alone, not partial; (golden words, fp32 bound or None)."""
    knobs, N = _tb_knobs(), cfg["n"]
    sw = _set_words(1, knobs, N, fmt)
    act, seed = sw["ents"][0]["act"], set_dropout_seed(knobs["dropout_seed"], 1)
    biased = knobs["has_bias"] != 0 and 1 % knobs["accum_passes"] == 0  # apply_bias: only a group's first pass sends its bias
    A, B = _tb_words("matrix_west_1.mem"), _tb_words("matrix_north_1.mem")
    if fmt == "int8":
        rq = dict(sw["ents"][0], mult=sw["mult"], shift=sw["shift"])
        hb = _sx(_tb_words("bias_1.mem"), 32) if biased else np.zeros(N, np.int64)
        return np.asarray(_golden_int8([(_sx(A, 8), _sx(B, 8))], hb, rq, cfg, act, seed)[4]).ravel() & 0xFF, None
    if fmt != "fp32":
        bb = _tb_words("bias_1.mem") if biased else None
        return np.asarray(_golden_bits([(A, B)], bb, cfg, act, seed, fmt)[3]).ravel(), None
    A, B = _pad_square(A, N).astype(np.uint32).view(np.float32), _pad_square(B, N).astype(np.uint32).view(np.float32)
    b = _tb_words("bias_1.mem").astype(np.uint32).view(np.float32) if biased else np.zeros(N, np.float32)
    C = (_ref_matmul(A, B) + b).astype(np.float32)
    S = np.abs(A).astype(np.float64) @ np.abs(B).astype(np.float64) + np.abs(b)
    F = np.asarray(_golden_from_c(C, cfg, act, seed)[3], np.float32).ravel().view(np.uint32).astype(np.int64)
    return F, np.asarray(fp32_error_bound(S, N, cfg, act, C), np.float32).ravel().view(np.uint32).astype(np.int64)


@selftest
def test_b2b_golden_is_set1_alone():
    # the back-to-back pass runs set 1 alone, not partial, so its golden is set 1's own, never its accumulate group's sum
    exp_f, bnd_f = _tb_b2b_files()
    g = globals()
    saved = {k: g[k] for k in ("TB_DIR", "RESULTS_DIR", "TRUTH_DUMP", "TRUTH_SCALES")}
    ran, acc = 0, 0
    for fmt in FORMATS:
        for t in [x for x in PIPELINE_TESTS if fmt in x.get("formats", tuple(FORMATS))]:
            with tempfile.TemporaryDirectory() as d:
                g.update(TB_DIR=d, RESULTS_DIR=os.path.join(d, "results"), TRUTH_DUMP=os.path.join(d, "truth_dump.txt"),
                         TRUTH_SCALES=os.path.join(d, "truth_scales.json"))
                try:
                    cfg = {"n": 8, "tile_size": 2, "lanes": 32, "host_words": 8, "fmt_name": fmt, **t}
                    with contextlib.redirect_stdout(io.StringIO()):
                        generate_vectors(cfg)
                    got = _tb_words(exp_f)
                    if t.get("packed"):  # packed sets never accumulate: set 1 alone is the stream's set 1
                        want, bnd = _tb_words("expected_output_1.mem"), _tb_words("bound_output_1.mem")
                    else:
                        want, bnd = _set1_alone(cfg, fmt)
                    n_bad = int((got != want).sum()) if got.size == want.size else -1
                    assert n_bad == 0, (fmt, t["name"], exp_f, f"{n_bad} of {want.size} words differ (-1: {got.size} words, want {want.size})")
                    if bnd is not None:
                        gb = _tb_words(bnd_f)
                        assert gb.size == bnd.size and (gb == bnd).all(), (fmt, t["name"], bnd_f, "bound differs")
                    ran += 1
                    acc += t.get("accum_passes", 1) > 1
                finally:
                    g.update(saved)
    runs = [x for f in FORMATS for x in PIPELINE_TESTS if f in x.get("formats", tuple(FORMATS))]
    assert ran == len(runs) and acc == sum(x.get("accum_passes", 1) > 1 for x in runs) >= 20, (ran, acc)


@selftest
def test_int8_relu_is_the_clamp():
    # int8 ReLU: the requantize clamp from the zero point, then an identity activation stage
    assert _relu_clamp({"amin": 5, "zp": 5}, 0, 0).ok and _relu_clamp({"amin": -128, "zp": -128}, 0, 0).ok
    assert not _relu_clamp({"amin": -100, "zp": 5}, 3, 2).ok and _relu_clamp({"amin": -100, "zp": 5}, 3, 2).first == [(3, 2, -100, 5)]


@selftest
def test_residual_golden():
    """_golden_int8_residual against an element-by-element composition (scalar ipu.qadd on each code), and it moves with zp1, the ADD and zp_o."""
    rng = np.random.RandomState(3)
    N = 8
    A = rng.randint(-128, 128, (N, N)).astype(np.int64)
    B = rng.randint(-127, 128, (N, N)).astype(np.int64)
    bias = rng.randint(-500, 500, N).astype(np.int64)
    acc = wrap32(imatmul(A, B) + bias[None, :])
    rq = requant_params(acc, 0.02, np.full(N, 0.01), "linear", np.random.RandomState(4), (-9, -128, 127))
    q1 = requantize(acc, rq)
    R, add, ra = _add_draw(rng, q1, rq["s_out"], rq["zp"], "tanh", True)
    assert add["zpo"] != 0, "the fixture needs zp_o != 0, or the lane's zero point cannot show; pick another seed"
    cfg = {"n": N, "pool_h": 1, "pool_w": 1, "padding": 0}
    Y = _golden_int8_residual([(A, B)], bias, rq, R, add, ra, cfg, "tanh", 1)[1]
    for i in range(N):
        for j in range(N):
            assert int(Y[i, j]) == int(ipu.qadd(int(q1[i, j]), int(R[i, j]), *[add[k] for k in mr.ADD_FIELDS]))
    for k, v in (("zp1", 0), ("zpo", add["zpo"] + 7)):
        assert np.any(_golden_int8_residual([(A, B)], bias, rq, R, dict(add, **{k: v}), ra, cfg, "tanh", 1)[1] != Y), k
    assert np.any(_golden_int8_residual([(A, B)], bias, rq, R, add, dict(ra, zp=0), cfg, "tanh", 1)[2] !=
                  _golden_int8_residual([(A, B)], bias, rq, R, add, ra, cfg, "tanh", 1)[2]), "the lane must see zp_o"
    assert np.array_equal(_golden_int8_residual([(A, B)], bias, rq, R, add, ra, cfg, "tanh", 1)[2],
                          mr.int8_residual_exact(A, B, bias, rq, R, add, ra, "tanh"))  # the layer engine's helper agrees


def _residual_tests() -> list:
    """The int8 residual pipeline tests' names."""
    return [t["name"] for t in PIPELINE_TESTS if t.get("residual") or t.get("residual_every")]


def _residual_add_sets(name: str, N: int = 16, T: int = 4) -> list:
    """A residual test's add sets as the golden's arguments (the last call per set, the one written), from its writer in a scratch directory."""
    t = next(x for x in PIPELINE_TESTS if x["name"] == name)
    rec, orig, g = {}, _golden_int8_residual, globals()

    def spy(*a, **k):
        if not k:
            rec[a[8]] = a
        return orig(*a, **k)

    with _scratch_tb_dir():
        g["_golden_int8_residual"] = spy
        try:
            generate_vectors({"n": N, "tile_size": T, "lanes": 32, "host_words": N, "fmt_name": "int8", **t})
        finally:
            g["_golden_int8_residual"] = orig
    return list(rec.values())


@selftest
def test_residual_clamp_bites():
    """Each residual test has an add set whose golden moves with the ADD's clamp forced to -128..127, so a DUT that drops add_min / add_max, or the ReLU after an ADD, fails."""
    for name in _residual_tests():
        moved = 0
        for a in _residual_add_sets(name):
            add, ra = a[4], a[5]
            F = _golden_int8_residual(*a)[4]
            moved += bool(np.any(_golden_int8_residual(*a[:4], dict(add, amin=-128, amax=127), dict(ra, amin=-128, amax=127), *a[6:])[4] != F))
        assert moved, f"{name}: no add set's outputs move with the ADD's clamp forced to the rails"


@selftest
def test_residual_add_discriminates():
    """Each add set's golden differs in 10% or more of its outputs from each wrong DUT: zp1 dropped, the lane's z_in 0 (lane activations),
    q1 and q2 swapped, q1 a beat late, the previous add set's q1."""
    for name in _residual_tests():
        prev = None
        for a in _residual_add_sets(name):
            parts, hw_bias, rq, R, add, ra, cfg, act = a[:8]
            N, lanes = cfg.get("n", 16), cfg.get("lanes", 32)
            q1 = requantize(mesh_model.matmul_int([(_pad_square(x, N), _pad_square(y, N)) for x, y in parts], N, hw_bias), rq)
            F = np.asarray(_golden_int8_residual(*a)[4])
            alt = {"zp1 dropped": _golden_int8_residual(*a[:4], dict(add, zp1=0), *a[5:])[4],
                   "q1 and q2 swapped": _golden_int8_residual(*a[:3], q1, *a[4:], q1=R)[4],
                   "q1 a beat late": _golden_int8_residual(*a, q1=np.roll(q1.reshape(lanes, -1), 1, axis=1).reshape(q1.shape))[4]}
            if act in LANE_ACTS:
                alt["the lane's z_in 0"] = _golden_int8_residual(*a[:5], dict(ra, zp=0), *a[6:])[4]
            if prev is not None:
                alt["the previous add set's q1"] = _golden_int8_residual(*a, q1=prev)[4]
            for k, v in alt.items():
                d = float(np.mean(np.asarray(v) != F))
                assert d >= 0.10, f"{name}: an add set ({act}) differs from '{k}' in {d:.0%} of its outputs, under 10%"
            prev = q1


@selftest
def test_residual_layers_repeat():
    """An add_layers test's add sets take its layers' ADD words in turn (A, B, A, B), each layer its own."""
    for t in (x for x in PIPELINE_TESTS if x.get("add_layers")):
        L, words = t["add_layers"], [tuple(a[4][k] for k in mr.ADD_FIELDS) for a in _residual_add_sets(t["name"])]
        assert len(words) > L and all(w == words[i % L] for i, w in enumerate(words)) and len(set(words[:L])) == L, \
            f"{t['name']}: its add sets' words do not repeat {L} layers' in turn"


@contextlib.contextmanager
def _scratch_tb_dir():
    """TB_DIR, RESULTS_DIR and TRUTH_SCALES pointed at a temporary directory while a stimulus writer runs, then restored: a self-test never writes testbenches/."""
    global TB_DIR, RESULTS_DIR, TRUTH_SCALES
    old = TB_DIR, RESULTS_DIR, TRUTH_SCALES
    with tempfile.TemporaryDirectory() as d:
        TB_DIR, RESULTS_DIR, TRUTH_SCALES = d, os.path.join(d, "results"), os.path.join(d, "truth_scales.json")
        try:
            yield d
        finally:
            TB_DIR, RESULTS_DIR, TRUTH_SCALES = old


@selftest
def test_residual_stimulus():
    """A residual test writes add_<k>.mem for every set: per group its partial and hold flags, then the add set; hold and partial sets expect no output."""
    cfg = dict(next(t for t in PIPELINE_TESTS if t["name"] == "int8_residual_tanh"), n=8, lanes=32)
    with _scratch_tb_dir() as d:
        with contextlib.redirect_stdout(io.StringIO()):
            _generate_vectors_int8(cfg)
        knobs = dict(x.split() for x in open(os.path.join(d, "test_knobs.mem")).read().splitlines())
        flags = [int(open(os.path.join(d, f"add_{k}.mem")).readline(), 16) for k in range(int(knobs["num_sets"]))]
        assert knobs["residual"] == "1" and len(flags) == 18, (knobs, len(flags))  # 6 groups of 2 depth passes and an add set
        assert flags[:3] == [0x33, 0x54, 0x38], [hex(f) for f in flags[:3]]  # partial with bias (tanh), hold (linear), add (tanh)
        assert [f & 0xF for f in flags] == [0x3, 0x4, 0x8] * 6, [hex(f) for f in flags]
        assert all(os.path.getsize(os.path.join(d, f"expected_output_{k}.mem")) == 0 for k in (0, 1))
        assert os.path.getsize(os.path.join(d, "expected_output_2.mem")) > 0
    with _scratch_tb_dir() as d:  # a plain test writes the knob as 0 and no add_<k>.mem
        with contextlib.redirect_stdout(io.StringIO()):
            _generate_vectors_int8(dict(next(t for t in PIPELINE_TESTS if t["name"] == "int8_perchannel_random_linear_nopool"), n=8, lanes=32))
        assert "residual 0" in open(os.path.join(d, "test_knobs.mem")).read() and not os.path.exists(os.path.join(d, "add_0.mem"))
    layouts = {}  # every residual test's group layout on the default credits (brief Step 3.1)
    for t in [x for x in PIPELINE_TESTS if x.get("residual") or x.get("residual_every")]:
        gs = _residual_layout(t, t.get("accum_passes", 1), t.get("mixed_acts", []), SETS_IN_FLIGHT, t["act"])
        layouts[t["name"]] = (gs[-1][0] + gs[-1][1] + gs[-1][2], [a for _, _, r, a in gs if r], [a for _, _, r, a in gs if not r])
    assert layouts["int8_residual_sigmoid_selu_nopool"] == (16, ["sigmoid", "selu"] * 4, []), layouts
    assert layouts["int8_residual_accum2_mixed_nopool"] == (18, ["relu", "linear", "sigmoid", "tanh"], ["tanh", "selu", "relu"]), layouts
    assert layouts["int8_residual_one_pair"] == (2, ["relu"], []), layouts
    assert all(n == 18 for k, (n, _, _) in layouts.items() if k not in ("int8_residual_sigmoid_selu_nopool", "int8_residual_one_pair")), layouts


def _tb_hold_add_line(*v) -> str:
    """TB_sienna_top's [HOLD/ADD] $display with the values v put in its %0d fields, from the TB's own source."""
    src = open(os.path.join(ROOT, "testbenches", "top", "TB_sienna_top.sv")).read()
    fmt = re.search(r'\$display\("(\[HOLD/ADD\][^"]*)"', src).group(1)
    assert fmt.count("%0d") == len(v), (fmt, v)
    return fmt.replace("%0d", "{}").format(*v) + "\n"


@selftest
def test_residual_pairs_check():
    """run_regression's residual counts on lines built from TB_sienna_top's own $display: per stream pass a hold set done per add set
    below num_sets, every add set with exactly per_lane add beats, none for another set; a missing line fails."""
    with _scratch_tb_dir() as d:
        open(os.path.join(d, "test_knobs.mem"), "w").write("num_sets 4\n")
        for k, f in enumerate((0x54, 0x58, 0x54, 0x58, 0x58)):  # add_4.mem lies past num_sets: a longer earlier test's
            _write_w32(os.path.join(d, f"add_{k}.mem"), [f] + [0] * 11)
        line = lambda h, b, w, o=0, x=(0, 0): _tb_hold_add_line(h, b, 8, w, o, *x, 0)
        assert HOLD_ADD.findall(line(2, 16, 2, 0, (3, 4)))[0] == ("2", "16", "8", "2", "0", "3", "4")  # the regex reads the TB's line
        assert _residual_pairs_bad(line(2, 16, 2) * 4, 8) == ""
        assert "expected 2, 16, 2 of 8 and 0" in _residual_pairs_bad(line(0, 0, 0) * 4, 8)
        assert _residual_pairs_bad(line(2, 16, 2) * 3 + line(1, 16, 2), 8) and _residual_pairs_bad(line(2, 15, 1) * 4, 8)
        assert _residual_pairs_bad(line(2, 16, 1) * 4, 8)  # 16 beats in all, but one set 9 and one 7: counted per add set
        assert _residual_pairs_bad(line(2, 16, 2, 1) * 4, 8) and "no [HOLD/ADD] line" in _residual_pairs_bad("RESULT: PASSED\n", 8)


@selftest
def test_residual_cross_sweep():
    """The sweep's residual-cross check on synthetic counts: a cross 0 at every point fails unless every such point is ruled unreachable,
    a 0 at some points passes, and run_regression's own line (_residual_cross_line) parses into its two counts."""
    g, b = RESIDUAL_CROSSES
    assert _residual_cross_sweep_bad({"p1": (1, 0), "p2": (0, 2)}, {}) == []  # both present, each at some point
    bad = _residual_cross_sweep_bad({"p1": (1, 0), "p2": (3, 0)}, {})
    assert len(bad) == 1 and bad[0].startswith(b) and "p1, p2" in bad[0], bad  # the bank done cross missing
    assert _residual_cross_sweep_bad({"p1": (0, 0), "p2": (0, 0)}, {}) == [f"{x}: 0 at every point, and no unreachable ruling for p1, p2" for x in (g, b)]
    assert _residual_cross_sweep_bad({"p1": (1, 0), "p2": (3, 0)}, {b: {"p1": "ruled", "p2": "ruled"}}) == []  # ruled unreachable everywhere
    assert _residual_cross_sweep_bad({"p1": (1, 0), "p2": (3, 0)}, {b: {"p1": "ruled"}})  # p2 unruled
    rtl = "  a_x_never: assert property (@(posedge clk_i) 1'b1);\n"
    assert _residual_cross_sweep_bad({"p1": (1, 0), "p2": (3, 0)}, {b: {"*": ("a_x_never", "ruled")}}, rtl) == []  # every point, on its guard
    assert _residual_cross_sweep_bad({"p1": (1, 0)}, {b: {"*": "ruled, a_x_never"}}, rtl)  # "*" as text alone is refused, even naming an assertion
    assert _residual_cross_sweep_bad({"p1": (1, 0)}, {b: {"*": ("a_x_gone", "ruled, a_x_never")}}, rtl)  # a guard the RTL does not declare, another named in the text
    assert _residual_cross_sweep_bad({"p1": (1, 0)}, {b: {"p1": "ruled by prose"}}, rtl) == []  # a single point's ruling needs text only
    assert _residual_cross_sweep_bad({"p1": (1, 0)}, {b: {"*": ("a_x", "ruled")}}, rtl + "  a_x_y: assert property (1'b1);\n")  # a prefix is no match
    assert _residual_cross_sweep_bad({"p1": (0, 0)}, {b: {"*": ("a_x_never", " ")}}, rtl)  # a ruling needs its written reason
    assert _residual_cross_sweep_bad({"p1": (0, 0)}, {b: {"*": ("a_x_never", "ruled")}}, rtl) and RESIDUAL_CROSS_UNREACHABLE.get(g) is None  # the L3 grant cross has none
    assert _ruling_ok("*", RESIDUAL_CROSS_UNREACHABLE[b]["*"], open(RTL_TOP).read())  # the committed ruling's assertion is in sienna_pipe
    gone = "".join(x for x in open(RTL_TOP).read().splitlines(True) if not re.match(r"^\s*a_hold_done_no_bank\s*:", x))
    assert not _ruling_ok("*", RESIDUAL_CROSS_UNREACHABLE[b]["*"], gone)  # the guard deleted from the RTL: the ruling fails
    assert RESIDUAL_CROSS.findall(_residual_cross_line([4, 0], 8) + "\n") == [("4", "0")]
    assert RESIDUAL_CROSS.findall(regress.ANSI.sub("", "\x1b[1m" + _residual_cross_line([12, 3], 8) + "\x1b[0m\n")) == [("12", "3")]


@selftest
def test_residual_cross_report():
    """_residual_cross_report over a results directory: gate off is NOT CHECKED and never fails, gate on fails a point with no line and
    a cross 0 everywhere unruled, passes both crosses present, fails a sweep with int8 points but no int8 pipeline point, and says a
    sweep without int8 is not applicable."""
    global RESIDUAL_CROSS_GATE, RESIDUAL_CROSS_UNREACHABLE
    keep = (RESIDUAL_CROSS_GATE, RESIDUAL_CROSS_UNREACHABLE)
    pts = ["selftest", "int8/N8_T2/pipeline", "int8/N8_T4/pipeline", "int8/N8_T2/gemm", "fp32/N8_T2/pipeline"]
    with tempfile.TemporaryDirectory() as d:
        def log(q, text):
            os.makedirs(os.path.dirname(os.path.join(d, q)), exist_ok=True)
            open(os.path.join(d, q + ".log"), "w").write(text)
        log("fp32/N8_T2/pipeline", "no residual tests\n")
        log("int8/N8_T2/pipeline", "x\n" + _residual_cross_line([2, 0], 8) + "\n")
        try:
            RESIDUAL_CROSS_GATE, RESIDUAL_CROSS_UNREACHABLE = False, {}
            lines, bad = _residual_cross_report(d, pts)  # T4 has no log: no line
            assert bad == [] and lines[-1].startswith("RESIDUAL CROSSES: NOT CHECKED"), lines
            RESIDUAL_CROSS_GATE = True
            lines, bad = _residual_cross_report(d, pts)
            assert "int8/N8_T4/pipeline: no RESIDUAL CROSSES line" in bad and any(x.startswith(RESIDUAL_CROSSES[1]) for x in bad), bad
            assert lines[-1].startswith("RESIDUAL CROSSES: FAIL") and "  int8/N8_T2/pipeline: hold done with an L3 grant 2, " in lines[1], lines
            log("int8/N8_T4/pipeline", _residual_cross_line([0, 5], 8) + "\n")
            lines, bad = _residual_cross_report(d, pts)
            assert bad == [] and lines[-1] == "RESIDUAL CROSSES: PASS", (lines, bad)
            log("int8/N8_T4/pipeline", _residual_cross_line([0, 0], 8) + "\n")
            assert _residual_cross_report(d, pts)[1], "the bank done cross 0 everywhere and unruled"
            RESIDUAL_CROSS_UNREACHABLE = {RESIDUAL_CROSSES[1]: {"int8/N8_T2/pipeline": "r", "int8/N8_T4/pipeline": "r"}}
            assert _residual_cross_report(d, pts)[1] == []
            lines, bad = _residual_cross_report(d, ["selftest", "int8/N8_T2/gemm", "fp32/N8_T2/pipeline"])
            assert bad and lines[-1].startswith("RESIDUAL CROSSES: FAIL"), lines
            lines, bad = _residual_cross_report(d, ["selftest", "fp32/N8_T2/pipeline"])
            assert bad == [] and lines == ["RESIDUAL CROSSES: NOT APPLICABLE, the sweep runs no int8 point"], lines
        finally:
            RESIDUAL_CROSS_GATE, RESIDUAL_CROSS_UNREACHABLE = keep


def _residual_syn_dump(name: str, fault=None) -> list:
    """The truth gates' misses on a dump of residual test name (N = 8, tile 2) built from its golden stages; fault (pass, set, f(q1, y))
    plants an error in one add set's hold_mem beat q1 or ADD output y."""
    global TRUTH_DUMP
    t, N, lanes = next(x for x in PIPELINE_TESTS if x["name"] == name), 8, 32
    keep = TRUTH_DUMP
    with _scratch_tb_dir() as d:
        TRUTH_DUMP = os.path.join(d, "truth_dump.txt")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                generate_vectors({"n": N, "tile_size": 2, "lanes": lanes, "host_words": N, "fmt_name": "int8", **t})
            kn, pkg = _tb_knobs(), _pkg_fields(os.path.join(d, "test_config_pkg.sv"))
            K = kn["num_sets"]
            fl = {k: _res_flags_of(k) for k in range(K)}
            ph, pw, pad, p = pkg["POOL_H"], pkg["POOL_W"], pkg["PADDING"], pkg["DROPOUT_P_PERCENT"] / 100
            sx = lambda f, b: _sx(_tb_words(f), b)
            def rq_of(k):
                w = _set_words(k, kn, N, "int8")
                return dict(w["ents"][0], mult=w["mult"], shift=w["shift"])
            out = ["H b2b 1"]
            def emit(tag, ridx, k, C, X, Aout, act_rq, act, q12=None, Rh=None):
                Cf = np.asarray(C, np.int64).ravel()
                out.extend(f"M {tag} {ridx} {e} {int(Cf[e]) & 0xFFFFFFFF:08x} {int(Cf[e]) & 0xFFFFFFFF:08x}" for e in range(N * N))
                if Rh is not None:  # a hold set: its beats into hold_mem, nothing after
                    out.extend(f"R {tag} {k} {e} {int(Rh[e]) & 0xFF:02x}" for e in range(N * N))
                    return
                if q12 is not None:
                    out.extend(f"Q {tag} {k} {e} {int(q12[0][e]) & 0xFF:02x} {int(q12[1][e]) & 0xFF:02x} {int(X[e]) & 0xFF:02x}" for e in range(N * N))
                out.extend(f"X {tag} {k} {e} {int(X[e]) & 0xFF:02x}" for e in range(N * N))
                out.extend(f"A {tag} {k} {e} {int(Aout[e]) & 0xFF:02x}" for e in range(N * N))
                Pl = _maxpool_int(np.asarray(Aout).reshape(N, N), ph, pw, pad).ravel()
                W = Pl.size
                kp = dropout_keep(W, p, set_dropout_seed(kn["dropout_seed"], k), lanes) if kn["training_mode"] else np.ones(W, bool)
                Dv = np.where(kp, Pl, drop_zp(act, act_rq))
                out.extend(f"P {tag} {k} {w % lanes} {w // lanes} {int(Pl[w]) & 0xFF:02x} {int(kp[w])}" for w in range(W))
                out.extend(f"D {tag} {k} {i} {int(Dv[i]) & 0xFF:02x}" for i in range(W))
            def alone(tag, k, fw, fn):
                A, B = sx(fw, 8).reshape(N, N), sx(fn, 8).reshape(N, N)
                hb = sx(f"bias_{k}.mem", 32) if fl[k]["bias"] else np.zeros(N, np.int64)
                C = wrap32(imatmul(A, B) + hb[None, :])
                rq = rq_of(k)
                q = requantize(C, rq)
                emit(tag, 0, k, C, q.ravel(), activate_int8(q, fl[k]["act"], rq).ravel(), rq, fl[k]["act"])
            alone("P0", 0, "matrix_west.mem", "matrix_north.mem")
            alone("B2B", 1, "matrix_west_1.mem", "matrix_north_1.mem")
            for tag in TRUTH_STREAMS:
                ridx, g0, held = 0, 0, None
                for k in range(K):
                    if fl[k]["partial"]:
                        continue
                    ks, g0 = range(g0, k + 1), k + 1
                    hb = sx(f"bias_{ks[0]}.mem", 32) if fl[ks[0]]["bias"] else np.zeros(N, np.int64)
                    C = wrap32(sum(imatmul(sx(f"matrix_west_{j}.mem", 8).reshape(N, N), sx(f"matrix_north_{j}.mem", 8).reshape(N, N)) for j in ks)
                               + hb[None, :])
                    rq = rq_of(k)
                    q = requantize(C, rq).ravel()
                    if fl[k]["hold"]:
                        held = q.copy()
                        emit(tag, ridx, k, C, None, None, None, None, Rh=q)
                    elif fl[k]["add"]:
                        w = fl[k]["words"]
                        y, q1 = mr.qadd_codes(held, q, w), held.copy()
                        ra = dict(rq, zp=w["zpo"], amin=w["amin"], amax=w["amax"])
                        if fault and fault[:2] == (tag, k):
                            fault[2](q1, y)
                        emit(tag, ridx, k, C, y, activate_int8(y.reshape(N, N), fl[k]["act"], ra).ravel(), ra, fl[k]["act"], q12=(q1, q))
                    else:
                        emit(tag, ridx, k, C, q, activate_int8(q.reshape(N, N), fl[k]["act"], rq).ravel(), rq, fl[k]["act"])
                    ridx += 1
            open(TRUTH_DUMP, "w").write("\n".join(out) + "\n")
            return truth_report(truth_test(N, "int8", lanes))[1]
        finally:
            TRUTH_DUMP = keep


@selftest
def test_residual_truth_path():
    """The residual truth path (R and Q lines, truth.gate_add) on dumps built from the golden stages: clean dumps of four residual tests
    pass every gate; one hold_mem beat off fails transfer, one ADD output off fails add."""
    for name in ("int8_residual_tanh", "int8_residual_accum2_mixed_nopool", "int8_residual_train", "int8_residual_sigmoid_selu_nopool"):
        assert _residual_syn_dump(name) == [], name
    def q1_off(q1, y):
        q1[3] += 1 if q1[3] < 127 else -1
    def y_off(q1, y):
        y[5] += 1 if y[5] < 127 else -1
    bad = _residual_syn_dump("int8_residual_tanh", ("S2", 5, q1_off))
    assert any(x.startswith("[FAIL] truth: transfer") for x in bad), bad
    bad = _residual_syn_dump("int8_residual_tanh", ("S3", 8, y_off))
    assert len(bad) == 1 and bad[0].startswith("[FAIL] truth: add ") and "passes S3" in bad[0], bad


@selftest
def test_truth_is_systolicmesh_copy():
    # the gates load SystolicMesh's ArithmeticLibrary copy by path, and GPNAE's copy must be the same bytes
    sm = os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "models")
    gp = os.path.join(ROOT, "GPNAE", "ArithmeticLibrary", "Common", "models")
    assert os.path.realpath(truth.__file__) == os.path.realpath(os.path.join(sm, "truth.py")), truth.__file__
    assert sys.modules["truth"] is truth
    for f in ("truth.py", "test_truth.py"):
        a, b = open(os.path.join(sm, f), "rb").read(), open(os.path.join(gp, f), "rb").read()
        assert a == b, f"{f}: SystolicMesh's and GPNAE's ArithmeticLibrary copies differ ({len(a)} and {len(b)} bytes); fast-forward GPNAE's"


@selftest
def test_guard_checks_collapse_k():
    with tempfile.TemporaryDirectory() as d:
        ck0 = fixture_pkg(d, "bf16", edit={"COLLAPSE_K": 0})
        rc, out = pkg_guard(ck0, "FMT=bf16")
        assert rc != 0 and "COLLAPSE_K=0, but the build asks for N=16 TILE=4 LANES=32 COLLAPSE_K=1" in out, out
        rc, out = pkg_guard(ck0, "FMT=bf16", "COLLAPSE_K=0")
        assert rc == 0, out
        rc, out = pkg_guard(fixture_pkg(d, "bf16", drop=("COLLAPSE_K",)), "FMT=bf16")
        assert rc != 0 and "COLLAPSE_K=missing" in out, out


@selftest
def test_guard_missing_file():
    with tempfile.TemporaryDirectory() as d:
        rc, out = pkg_guard(os.path.join(d, "none.sv"), "FMT=fp32")
        assert rc != 0 and "missing or unreadable" in out, out


@selftest
def test_parse_time_rejections():
    r = make_n("pipeline", "FMT=fp16")
    assert r.returncode != 0 and "FMT=fp16: not one of fp32 bf16 int8" in r.stderr, r.stderr
    for t in ("regression", "pack", "perf-analysis"):
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode != 0 and "COLLAPSE_K=0 reaches TB_sienna_top and the layer contract's model, gemm and tflite runs" in r.stderr, (t, r.stderr)
    for t in ("gemm", "model", "tflite"):  # the layer contract's runs build the mesh mode they are given
        r = make_n(t, "COLLAPSE_K=0", "MODEL_DIR=/m", "FMT=int8" if t == "tflite" else "FMT=fp32", "PYTHON=py")
        assert r.returncode == 0 and "--collapse-k 0" in r.stdout, (t, r.stdout + r.stderr)
    for ck, want in (("0", False), ("1", True)):  # make tflite adds the packed layers at N >= 16 on the collapse-k 1 mesh only
        r = make_n("tflite", "N=16", f"COLLAPSE_K={ck}", "FMT=int8", "PYTHON=py")
        assert r.returncode == 0 and ("--pack" in r.stdout) == want, (ck, r.stdout + r.stderr)
    for t in ("pipeline", "verilator", "lint", "pkg", "sim", "build-key"):  # TB_sienna_top sets sienna_top's COLLAPSE_K from the package
        r = make_n(t, "COLLAPSE_K=0")
        assert "COLLAPSE_K=0 reaches" not in r.stderr, (t, r.stderr)
    r = make_n("lint", "COLLAPSE_K=2")
    assert r.returncode != 0 and "Invalid COLLAPSE_K=2" in r.stderr, r.stderr
    for t in ("help", "sm-verilator"):  # the only targets collapse-k 0 is meaningful for, or harmless in
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode == 0, (t, r.stderr)


@selftest
def test_build_key_per_format():
    with tempfile.TemporaryDirectory() as d:  # make build-key with an empty dependency file: the key's flag line carries the build's define
        open(os.path.join(d, "VTB_sienna_top__ver.d"), "w").close()
        key = lambda *a: subprocess.run(["make", "-s", "build-key", f"VERILATOR_DIR={d}", *a], cwd=ROOT, env=MAKE_ENV, capture_output=True, text=True).stdout
        keys = {f: key(f"FMT={f}") for f in reg.FORMATS}
        keys["none"] = key("FMT=fp32", "FMT_DEF=")
        assert all(keys.values()) and len(set(keys.values())) == 4, {k: v[-80:] for k, v in keys.items()}
        assert all(f"+define+FMT_{f.upper()}" in keys[f] for f in reg.FORMATS) and "+define+FMT_" not in keys["none"]


@selftest
def test_multi_point():
    pts = reg.SPEC.points(True, ["bf16"])
    assert "bf16/N8_T2/pipeline-multi" in pts and not any(p.endswith("pipeline-multi") and "N8_T2" not in p for p in pts), pts
    argv = reg.SPEC.argv("int8/N8_T2/pipeline-multi", argparse.Namespace())
    assert argv[:3] == ["make", "multi", "FMT=int8"] and "COPIES=2" in argv, argv
    try:
        reg.SPEC.parse("fp32/N16_T4/pipeline-multi")
        raise AssertionError("pipeline-multi offered at N16")
    except ValueError:
        pass
    r = make_n("multi", "FMT=bf16", "PYTHON=py")
    assert r.returncode == 0 and r.stdout.split()[:7] == "py regression.py --action multi --format bf16 --n".split(), r.stdout
    ok = "MULTI COPIES=2 N=8 lanes=32\nRESULT: PASSED\nMULTI: PASS (COPIES=2 fp32 N=8 T=2 48 sets of x, make exit 0; l)\n"
    assert reg.SPEC.verdict("fp32/N8_T2/pipeline-multi", ok, 0) == "PASS"
    assert reg.SPEC.verdict("fp32/N8_T2/pipeline-multi", ok.replace("MULTI: PASS", "MULTI: FAIL"), 1).startswith("FAIL")


@selftest
def test_default_point():
    pts = reg.SPEC.points(True, ["fp32"])
    assert reg.DEFAULT_POINT in pts and reg.DEFAULT_POINT not in reg.SPEC.points(True, ["bf16", "int8"]), pts[:6]
    argv = reg.SPEC.argv(reg.DEFAULT_POINT, argparse.Namespace())
    assert argv[:3] == ["make", "pipeline", "FMT=fp32"] and argv[-1] == "FMT_DEF=" and "FMT_DEF=" not in reg.SPEC.argv(reg.DEFAULT_TWIN, argparse.Namespace())
    r = make_n("lint", "GEN_PKG=0", "FMT_DEF=")
    assert r.returncode == 0 and "+define+FMT_" not in r.stdout and "+define+FMT_FP32" in make_n("lint", "GEN_PKG=0").stdout, r.stdout[-300:]
    log = "  Passed : 4 / 4\n  Sanity Clean\npipeline_complete_o asserted @ 100  (311 cycles)\n  Exact     : 512\n      WORDS t: 2 outputs, sha1 ab\n"
    assert reg.compared_lines("# command: x\n" + log) == reg.compared_lines(log) and len(reg.compared_lines(log)) == 4
    assert reg.twin_verdict(log.replace("sha1 ab", "sha1 ac"), log).startswith("FAIL (differs"), "a word hash must be compared"
    assert reg.OUT_WORD.findall("[DEBUG 5] Dropout -> Output (Lane 3)   : dec=1.0000  hex=3f800000") == [("3", "3f800000")]
    with tempfile.TemporaryDirectory() as d:
        k = os.path.join(d, "build_key.txt")
        open(k, "w").write("abc  f.sv\nTB_sienna_top --timing +define+FMT_BF16 -Wno-X\n")
        assert reg.build_define(k) == "+define+FMT_BF16"
        open(k, "w").write("abc  f.sv\nTB_sienna_top --timing -Wno-X\n")
        assert reg.build_define(k) == "none" and reg.build_define(os.path.join(d, "missing")) == "unreadable"
    assert reg.twin_verdict(log, "# other host\n" + log) == "PASS"
    assert reg.twin_verdict(log.replace("311", "312"), log).startswith("FAIL (differs from fp32/N8_T2/pipeline at result line 2")
    assert reg.twin_verdict(log, log.replace("512", "511")).startswith("FAIL (differs")
    assert reg.SPEC.verdict(reg.DEFAULT_POINT, "nothing\n", 0).startswith("FAIL"), "a failing pipeline verdict decides first"
    mk = open(os.path.join(ROOT, "Makefile")).read()
    assert f"regression/{reg.DEFAULT_POINT}: | regression/{reg.DEFAULT_TWIN}" in mk, "the default point runs after its twin"


@selftest
def test_pack_target():
    r = make_n("pack", "FMT=bf16", "N=32", "TILE=8", "LANES=64", "PYTHON=py")
    assert r.returncode == 0 and r.stdout.split() == "py regression.py --action pack --format bf16 --n 32 --tile 8 --lanes 64".split(), r.stdout + r.stderr
    r = make_n("pack", "FMT=fp16")
    assert r.returncode != 0 and "FMT=fp16: not one of" in r.stderr, r.stderr


@selftest
def test_regression_points():
    small, full = SPEC.points(True, list(FORMATS)), SPEC.points(False, list(FORMATS))
    sweep = lambda pts: sorted({(SPEC.parse(p)["n"], SPEC.parse(p)["tile"]) for p in pts if SPEC.parse(p)["n"]})
    assert sweep(small) == [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8), (16, 16)], sweep(small)
    assert sweep(full) == sweep(small) + [(32, t) for t in (2, 4, 8, 16, 32)] + [(64, t) for t in (2, 4, 8, 16, 32, 64)], sweep(full)
    assert small[0] == full[0] == "selftest" and "selftest" not in small[1:]  # format-independent: once
    assert small[1:3] == full[1:3] == ["add-lanes", "rq-lanes"] and not set(UNIT_POINTS) & set(small[3:])  # int8's unit benches: once
    assert small[3:6] == full[3:6] == [DEFAULT_POINT, *FAULT_POINTS] and [p for p in small if p.endswith(MULTI_KIND)] == [f"{f}/N8_T2/{MULTI_KIND}" for f in FORMATS]
    extra = 1 + 2 + 3  # the default point, the two idle faults and pipeline-multi per format
    assert len(small) == len(set(small)) == 1 + 2 + 7 * 28 + extra == 205 and len(full) == len(set(full)) == 1 + 2 + 18 * 12 + 7 * 9 + 11 * 3 + 18 * 6 + 18 + extra == 447, (len(small), len(full))
    layer_ck0 = {"fp32": ["model-ck0", "gemm-ck0"], "bf16": ["model-ck0", "gemm-ck0"], "int8": ["gemm-ck0", "tflite-ck0"]}
    for f, kinds in (("fp32", ["pipeline", "model", "pack", "gemm"]), ("bf16", ["pipeline", "model", "pack", "gemm"]), ("int8", ["pipeline", "pack", "gemm", "tflite"])):
        for n, t in sweep(full):  # back-to-back and the layer contract's collapse-k 0 at every N, the pipeline's collapse-k 0 at N = 8 and 16; packing needs collapse-k 1
            want = kinds[:1] + (["pipeline-ck0", "pipeline-b2b", "pipeline-ck0-b2b"] if n <= 16 else ["pipeline-b2b"]) + kinds[1:] + layer_ck0[f] + (["residual"] if f == "int8" else [])
            assert [p.rsplit("/", 1)[1] for p in full if p.startswith(f"{f}/N{n}_T{t}/") and not p.endswith(MULTI_KIND)] == want, (f, n, t)
    assert SPEC.parse("bf16/N16_T2/pipeline-ck0")["ck"] == 0 and SPEC.parse("bf16/N16_T2/pipeline")["ck"] == 1
    assert SPEC.parse("fp32/N32_T8/gemm-ck0")["ck"] == 0
    assert [(SPEC.parse(f"int8/N8_T2/{k}")["ck"], SPEC.parse(f"int8/N8_T2/{k}")["b2b"]) for k in ("pipeline", "pipeline-ck0", "pipeline-b2b", "pipeline-ck0-b2b")] \
        == [(1, False), (0, False), (1, True), (0, True)]
    assert [SPEC.parse(p)[x] for p in ("fp32/N64_T64/pipeline-b2b",) for x in ("ck", "b2b", "lanes")] == [1, True, 128]
    assert SPEC.points(True, ["int8"]) == ["selftest", "add-lanes", "rq-lanes"] + [p for p in small if p.startswith("int8/")] and SPEC.points(False, []) == []
    assert SPEC.points(True, ["fp32", "bf16"]) == ["selftest", DEFAULT_POINT, *FAULT_POINTS] + [p for p in small if p[:4] in ("fp32", "bf16")]
    assert SPEC.argv("idle-fault-sp1", argparse.Namespace())[:3] == ["make", "idle-fault", "SET_PORT=1"]
    r = make_n("idle-fault", "SET_PORT=1", "PYTHON=py")
    assert r.returncode == 0 and r.stdout.split()[-5:] == "py regression.py --action idle-fault --set-port 1".split()[-5:], r.stdout + r.stderr
    assert make_n("idle-fault", "PYTHON=py").returncode == 0 and "SET_PORT=0 or SET_PORT=1" in make_n("idle-fault", "PYTHON=py").stdout
    assert SPEC.verdict("idle-fault-sp0", "%Error: x.sv:5: Assertion failed in TOP.dut: a_idle_contract\nIDLE FAULT: CAUGHT (SET_PORT 0)\n", 0) == "PASS"
    assert SPEC.verdict("idle-fault-sp1", "IDLE FAULT: CAUGHT (SET_PORT 1)\n", 1).startswith("FAIL"), "the action exits 0 only when it caught the fault"
    assert SPEC.verdict("idle-fault-sp0", "IDLE FAULT: SURVIVED (SET_PORT 0)\n", 1).startswith("FAIL")
    assert [SPEC.parse(p)["lanes"] for p in ("fp32/N32_T8/pipeline", "fp32/N64_T8/pipeline", "fp32/N8_T2/model")] == [32, 128, 32]
    for bad in ("fp32/N8_T16/pipeline", "int8/N8_T2/model", "fp32/N16_T4/tflite", "fp16/N8_T2/pack", "fp32/N8_T2/selftest", "fp32/pipeline",
                "fp32/N12_T4/pack", "fp32/N8_T2/mesh", "fp32/gpnae-verilator", "fp32/aril-fpu", "fp32/N32_T8/pipeline-ck0",
                "fp32/N8_T2/pack-ck0", "int8/N8_T2/model-ck0", "fp32/N8_T2/tflite-ck0", "fp32/N32_T8/pipeline-ck0-b2b", "bf16/N64_T8/pipeline-ck0-b2b",
                "fp32/N8_T2/pack-b2b", "fp32/N8_T2/pipeline-b2b-ck0", "fp32/N8_T2/residual", "bf16/N16_T4/residual"):
        try:
            SPEC.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse accepted {bad}")


@selftest
def test_point_commands():
    py, opts = sys.executable, type("O", (), {"model_dir": "/m"})()
    at = lambda f, n, t, l: [f"FMT={f}", f"PYTHON={py}", f"N={n}", f"TILE={t}", f"LANES={l}"]
    assert SPEC.argv("fp32/N64_T64/pipeline", opts) == ["make", "pipeline", *at("fp32", 64, 64, 128)]
    assert SPEC.argv("bf16/N8_T4/model", opts) == ["make", "model", *at("bf16", 8, 4, 32), "MODEL_DIR=/m", "MODEL_COUNT=3"]
    assert SPEC.argv("int8/N32_T8/gemm", opts) == ["make", "gemm", *at("int8", 32, 8, 32), "QUICK=1"]
    assert SPEC.argv("int8/N8_T2/tflite", opts) == ["make", "tflite", *at("int8", 8, 2, 32)] and SPEC.argv("fp32/N16_T2/pack", opts)[1] == "pack"
    assert SPEC.argv("selftest", opts)[1:] == ["regression.py", "--action", "selftest"]
    assert SPEC.argv("bf16/N16_T4/pipeline-ck0", opts) == ["make", "pipeline", *at("bf16", 16, 4, 32), "COLLAPSE_K=0"]
    assert SPEC.argv("int8/N8_T2/pipeline-b2b", opts) == ["make", "pipeline", *at("int8", 8, 2, 32), "EXTRA_FLAGS=-DBACK_TO_BACK"]
    assert SPEC.argv("fp32/N16_T8/pipeline-ck0-b2b", opts) == ["make", "pipeline", *at("fp32", 16, 8, 32), "COLLAPSE_K=0", "EXTRA_FLAGS=-DBACK_TO_BACK"]
    assert SPEC.argv("int8/N8_T2/tflite-ck0", opts) == ["make", "tflite", *at("int8", 8, 2, 32), "COLLAPSE_K=0"]
    assert SPEC.argv("bf16/N64_T8/model-ck0", opts) == ["make", "model", *at("bf16", 64, 8, 128), "MODEL_DIR=/m", "MODEL_COUNT=3", "COLLAPSE_K=0"]
    assert SPEC.argv("fp32/N16_T4/gemm-ck0", opts) == ["make", "gemm", *at("fp32", 16, 4, 32), "QUICK=1", "COLLAPSE_K=0"]
    assert SPEC.argv("int8/N8_T2/residual", opts) == ["make", "residual", *at("int8", 8, 2, 32)]
    assert SPEC.env("int8/N8_T2/gemm-ck0", {}) == {"OPT_FAST": "-O0"} and SPEC.env("int8/N8_T2/tflite-ck0", {}) == {}
    old = mr.COLLAPSE_K
    try:
        for main, args in ((mr.model_main, ["--model-dir", "/m"]), (gemm_main, [])):  # the sets engine builds collapse-k 1: a usage error, before any file
            try:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    main(args + ["--engine", "sets", "--collapse-k", "0", "--work", "/nonexistent/lck0"])
            except SystemExit as e:
                assert e.code == 2 and "--engine sets builds collapse-k 1" in err.getvalue(), (main.__name__, e.code, err.getvalue())
                continue
            raise AssertionError(f"{main.__name__} accepted --engine sets --collapse-k 0")
    finally:
        mr.COLLAPSE_K = old
    with tempfile.TemporaryDirectory() as d:  # a -b2b point's make pipeline passes the define to the Verilator flags of the make regression.py starts
        fake = os.path.join(d, "py")
        open(fake, "w").write(f"#!/bin/sh\nexec make -s -n -C {ROOT} lint GEN_PKG=0 FMT=int8 N=8 TILE=2\n")
        os.chmod(fake, 0o755)
        for extra, want in (([], False), (["EXTRA_FLAGS=-DBACK_TO_BACK"], True)):
            r = subprocess.run(["make", "-s", "pipeline", "FMT=int8", "N=8", "TILE=2", f"PYTHON={fake}", *extra], cwd=ROOT, env=MAKE_ENV,
                               capture_output=True, text=True)
            assert r.returncode == 0 and "--lint-only" in r.stdout and ("-DBACK_TO_BACK" in r.stdout.split()) == want, (extra, r.stdout[-300:], r.stderr[-300:])
    r = make_n("pipeline", "FMT=int8", "N=8", "TILE=2", "COLLAPSE_K=0", "PYTHON=py")
    assert r.returncode == 0 and "--collapse-k 0" in r.stdout, r.stdout + r.stderr
    saved = {k: os.environ.get(k) for k in ("GPNAE_TIER", "TEST", "N", "LINK_STAGES")}
    os.environ.update(GPNAE_TIER="2", TEST="selu", N="32", LINK_STAGES="1")
    try:
        env = regress.point_env(SPEC, "fp32/N32_T2/pipeline", "s -j4 --jobserver-auth=3,4", 8)
    finally:
        for k, v in saved.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert env["TRACE"] == "0" and env["OPT_FAST"] == "-O0" and env["SIM_THREADS"] == "2" and not {"GPNAE_TIER", "TEST", "N", "LINK_STAGES"} & set(env)
    pe = lambda n: regress.point_env(SPEC, n, "s", 8).get("OPT_FAST")
    assert [pe(f"fp32/N{n}_T2/{k}") for n, k in ((8, "pack"), (16, "pack"), (8, "gemm"), (16, "gemm"))] == ["-O0"] * 4  # the short pack and gemm simulations build at -O0 at every N
    assert pe("fp32/N8_T2/pipeline") is None and pe("fp32/N16_T4/pipeline") is None and pe("int8/N8_T2/tflite") is None and pe("bf16/N16_T4/model") is None and pe("selftest") is None  # the long ones keep -Os below N = 32
    assert pe("fp32/N32_T2/pipeline") == pe("fp32/N32_T2/model") == "-O0"  # from N = 32 every point builds at -O0 (point_env)
    for n, t in ((8, 2), (16, 4), (32, 2), (64, 16)):  # a -b2b point builds and runs as its plain pipeline point, plus the define
        plain, b2b = f"bf16/N{n}_T{t}/pipeline", f"bf16/N{n}_T{t}/pipeline-b2b"
        assert pe(b2b) == pe(plain) == ("-O0" if n >= 32 else None), (n, pe(b2b), pe(plain))
        assert SPEC.argv(b2b, opts) == SPEC.argv(plain, opts) + ["EXTRA_FLAGS=-DBACK_TO_BACK"] and not any("LARGE_CODE" in a for a in SPEC.argv(b2b, opts)), n
    cores = regress._cores()
    assert f"-j 1 --build-jobs {cores} " in make_n("verilator").stdout  # one verilation thread (-j 8 segfaulted 3 of 63 times on od-r7a-2xl), every core for the C++ build
    for args in (("-C", "SystolicMesh"), ("-C", "GPNAE")):
        assert f"-j {cores} --build-jobs {cores} " in make_n(*args, "verilator").stdout, args  # Verilator takes every core in the submodules
    for args in (("lint", "GEN_PKG=0"), ("-C", "SystolicMesh", "lint"), ("-C", "GPNAE", "verilator")):  # every Verilator build of this tree takes SIM_THREADS
        out = make_n(*args, "SIM_THREADS=3").stdout
        assert "--threads 3" in out and "--threads 8" not in out, (args, out[:300])
    r = make_n("sm-verilator", "FMT=bf16", "N=32", "TILE=8")  # the mesh alone, for one N, as before
    assert r.returncode == 0 and "mesh-regression MATRIX_SIZE=32" in r.stdout and "--tiles 8" in r.stdout, r.stdout[-300:]


@selftest
def test_point_verdicts():
    n32 = len([t for t in PIPELINE_TESTS if "fp32" in t.get("formats", tuple(FORMATS))])  # every fp32 pipeline test
    n32p = len([t for t in PIPELINE_TESTS if "fp32" in t.get("formats", tuple(FORMATS)) and t.get("packed")])  # collapse-k 0 skips these
    clean = (f"  TRUTH: {n32} tests gated, 0 with misses\n  BACK_TO_BACK: 0 of {n32} tests ran the back-to-back pass\n"
             f"\x1b[1m  Passed      : \x1b[92m{n32}\x1b[0m / {n32}\n  Verdict : \x1b[92m✅ Sanity Clean\x1b[0m\n")
    plain = regress.ANSI.sub("", clean)
    tf = "; true function PASS worst 0.9% at 5%\n"
    lanes_ok = ("lanes_tier1               32    48    24      12      2101  0 outputs differ from the bit-exact model of tier 1; 7 differ from tier 0 of the reset image" + tf +
                "lanes_loaded              32    48    24      12      2101  0 outputs differ from the bit-exact model of tier 0 of the loaded image; 7 differ from tier 0 of the reset image" + tf)
    lanes_ref = "lanes_refused             32    48    24      12      2101  0 outputs differ from the bit-exact model of tier 0; 0 differ from tier 0 of the reset image" + tf
    lanes_all = lanes_ok + ("lanes_between             32    48    24      12      2101  0 outputs differ from the bit-exact model of tier 0 of the loaded image; 7 differ from tier 0 of the reset image" + tf +
                            lanes_ref)
    lanes8 = "".join(f"    {t}: 31 outputs of the model differ from tier 0 of the reset image\n" for t in LANE_CASES)
    v = lambda p, t, rc=0: SPEC.verdict(p, t, rc)
    assert n32 > 20 and v("int8/N8_T2/pipeline", clean) == f"FAIL ({n32} tests ran, int8 has {len([t for t in PIPELINE_TESTS if 'int8' in t.get('formats', tuple(FORMATS))])})"
    for p, text, rc, want in (
            ("fp32/N8_T2/pipeline", clean, 0, "PASS"), ("fp32/N8_T2/pipeline", clean, 1, "FAIL (exit 1)"),
            ("fp32/N8_T2/pipeline", plain.replace(f"{n32} / {n32}", "5 / 5"), 0, f"FAIL (5 tests ran, fp32 has {n32})"),  # a test lost on the way
            ("fp32/N8_T2/pipeline", plain.replace(f"{n32} / {n32}", f"{n32 - 1} / {n32}"), 0, f"FAIL ({n32 - 1} of {n32} tests passed)"),
            ("fp32/N8_T2/pipeline", plain.replace(f"{n32} / {n32}", "0 / 0"), 0, "FAIL (0 of 0 tests passed)"),
            ("fp32/N8_T2/pipeline", plain.replace("Sanity Clean", "Failures detected"), 0, "FAIL (no 'Passed : X / X' with Sanity Clean)"),
            ("fp32/N8_T2/pipeline", "make: done\n", 0, "FAIL (no 'Passed : X / X' with Sanity Clean)"),
            ("fp32/N8_T2/pipeline", clean + "%Error: x.sv:3: bad\n", 0, "FAIL (log: %Error: x.sv:3: bad)"),
            ("fp32/N8_T2/pipeline", plain.replace(f"TRUTH: {n32} tests", f"TRUTH: {n32 - 1} tests"), 0, f"FAIL ({n32 - 1} of {n32} tests gated on the true function)"),
            ("fp32/N8_T2/pipeline", plain.replace(f"  TRUTH: {n32} tests gated, 0 with misses\n", ""), 0, f"FAIL (0 of {n32} tests gated on the true function)"),
            ("fp32/N8_T2/pipeline", "      [FAIL] truth: maxpool |out - want| 1, 3 of 81 over the bound\n" + plain.replace("0 with misses", "1 with misses"), 0,
             "FAIL (true function: maxpool; 1 tests with misses)"),
            ("fp32/N8_T2/model", "MODEL GATE: PASS  fp32 N=8\n", 0, "PASS"), ("fp32/N8_T2/model", "MODEL GATE: FAIL  fp32\n", 0, "FAIL (log: MODEL GATE: FAIL  fp32)"),
            ("fp32/N8_T2/model", "built\n", 0, "FAIL (no 'MODEL GATE: PASS')"), ("int8/N8_T2/tflite", "MODEL GATE: PASS int8\n", 0, "PASS"),
            ("int8/N8_T2/tflite", "ok\n", 0, "FAIL (no 'MODEL GATE: PASS')"),
            ("int8/N16_T4/tflite", "MODEL GATE: PASS int8\nTFLITE_PACK: 3 of 5 models in 2 packed layers at N = 16, 0 failing\n", 0, "PASS"),
            ("int8/N16_T4/tflite", "MODEL GATE: PASS int8\n", 0, "FAIL (no 'TFLITE_PACK: \\d+ of \\d+ models in \\d+ packed layers at N = \\d+, 0 failing')"),
            ("int8/N16_T4/tflite", "MODEL GATE: PASS int8\nTFLITE_PACK: 3 of 5 models in 2 packed layers at N = 16, 1 failing\n", 0,
             "FAIL (log: TFLITE_PACK: 3 of 5 models in 2 packed layers at N = 16, 1 f)"),  # the verdict cuts the line
            ("int8/N16_T4/tflite-ck0", "MODEL GATE: PASS int8\n", 0, "PASS"),  # collapse-k 0 runs no packed layers
            ("int8/N16_T4/tflite-ck0", "MODEL GATE: FAIL int8\n", 0, "FAIL (log: MODEL GATE: FAIL int8)"),
            ("int8/N8_T2/residual", "RESIDUAL: PASS\n", 0, "PASS"), ("int8/N8_T2/residual", "RESIDUAL: FAIL tiles\n", 0, "FAIL (log: RESIDUAL: FAIL tiles)"),
            ("bf16/N8_T2/pack", "PACK REGRESSION PASS: 0 mismatching outputs\n", 0, "PASS"), ("bf16/N8_T2/pack", "done\n", 0, "FAIL (no 'PACK REGRESSION PASS: 0 mismatching outputs')"),
            ("int8/N8_T2/gemm", "grid_1x64x16  1 64 16\n" + lanes8 + "GEMM int8: 9 cases, 0 failing (every case per stage on the true function)\n", 0, "PASS"),
            ("int8/N8_T2/gemm", "grid_1x64x16  1 64 16\n" + lanes8.replace("    lanes_tier1", "    lanes_t1") + "GEMM int8: 9 cases, 0 failing (every case per stage on the true function)\n", 0,
             "FAIL (no '    lanes_tier1: \\d+ outputs of the model differ from tier 0 of the reset image')"),
            ("int8/N8_T2/gemm", "grid_1x64x16  1 64 16\n" + lanes8 + "FAIL lanes_refused: 3 of 4 writes while busy refused\nGEMM int8: 9 cases, 1 failing\n", 0,
             "FAIL (log: FAIL lanes_refused: 3 of 4 writes while busy refused)"),
            ("int8/N8_T2/gemm", "grid_1x64x16  1 64 16\n", 0, "FAIL (no 'GEMM int8: \\d+ cases, 0 failing ')"),
            ("fp32/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"layer_{x}_bias  64 48 40 240 7633  max err 9.3e-04 of the output range\n" for x in ("tanh", "sigmoid", "selu"))
             + lanes_all + (lanes_ok + lanes_ref).replace("lanes_", "sets lanes_"), 0, "PASS"),
            ("fp32/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"layer_{x}_bias  64 48 40 240 7633  max err 9.3e-04 of the output range\n" for x in ("tanh", "sigmoid", "selu"))
             + lanes_all + lanes_ok.replace("lanes_", "sets lanes_"), 0, "FAIL (no '" + lane_case_line("sets lanes_refused", 0).strip("^$") + "')"),
            ("fp32/N8_T4/gemm", "grid_1x8x8 1\n" + "".join(f"layer_{x}_bias  64 48 40 240 7633  max err 9.3e-04 of the output range\n" for x in ("tanh", "sigmoid", "selu"))
             + lanes_all, 0, "PASS"),  # the set contract's runs only at T = 2
            ("fp32/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"layer_{x}_bias  64 48 40 240 7633  max err 9.3e-04 of the output range\n" for x in ("tanh", "sigmoid", "selu"))
             + lanes_all, 0, "FAIL (no '" + lane_case_line("sets lanes_tier1", 1).strip("^$") + "')"),
            ("bf16/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"  layer_{x}: 0 outputs differ from the bit-exact model\n" for x in ("tanh", "sigmoid", "selu"))
             + "layer_bias_special  32 40 24 12 308  0 outputs differ from the bit-exact model\n" + lanes_all.replace("true function PASS", "true function FAIL", 1), 0,
             "FAIL (no '" + lane_case_line("lanes_tier1", 1).strip("^$") + "')"),  # a tier run off the true function fails
            ("bf16/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"  layer_{x}: 0 outputs differ from the bit-exact model\n" for x in ("tanh", "sigmoid", "selu"))
             + "layer_bias_special  32 40 24 12 308  0 outputs differ from the bit-exact model\n" + lanes_all + "FAIL lanes_tier1: 0 differ from tier 0\n", 0,
             "FAIL (log: FAIL lanes_tier1: 0 differ from tier 0)"),  # a tier run whose outputs equal tier 0's does not exercise it
            ("fp32/N8_T2/gemm", "grid_1x8x8 1\nlayer_tanh_bias  64 max err 9.3e-04 of the output range\n", 0, "FAIL (no 'layer_sigmoid_bias .* max err ')"),
            ("bf16/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"  layer_{x}: 0 outputs differ from the bit-exact model\n" for x in ("tanh", "sigmoid", "selu"))
             + "layer_bias_special  32 40 24 12 308  0 outputs differ from the bit-exact model\n" + lanes_all, 0, "PASS"),
            ("bf16/N8_T2/gemm", "grid_1x8x8 1\n  layer_tanh: 0 outputs differ from the bit-exact model\n", 0, "FAIL (no '  layer_sigmoid: 0 outputs differ from the bit-exact model')"),
            ("int8/N8_T2/gemm", "GEMM int8: 0 cases\n", 0, "FAIL (no 'grid_\\d+x\\d+x\\d+ ')"), ("fp32/N8_T2/gemm", "grid_1x8x8 1\nFAIL grid_1x8x8: error\n", 0, "FAIL (log: FAIL grid_1x8x8: error)"),
            ("selftest", "ALL 9 PASSED\n", 0, "PASS"), ("selftest", "1 of 9 FAILED\n", 0, "FAIL (no 'ALL \\d+ PASSED')"),
            ("fp32/N8_T2/pipeline-ck0", plain.replace(str(n32), str(n32 - n32p)), 0, "PASS"),
            ("fp32/N8_T2/pipeline-ck0", plain, 0, f"FAIL ({n32} tests ran, fp32 has {n32 - n32p})"),
            ("fp32/N8_T2/pipeline", plain.replace(f"0 of {n32} tests ran the back", f"{n32} of {n32} tests ran the back"), 0,
             f"FAIL ({n32} of {n32} tests ran the back-to-back pass)"),  # a plain point built with -DBACK_TO_BACK
            ("fp32/N8_T2/pipeline", plain.replace(f"  BACK_TO_BACK: 0 of {n32} tests ran the back-to-back pass\n", ""), 0, "FAIL (no BACK_TO_BACK count)"),
            ("fp32/N8_T2/pipeline-b2b", plain.replace(f"0 of {n32} tests ran the back", f"{n32 - 1} of {n32 - 1} tests ran the back"), 0,
             f"FAIL (BACK_TO_BACK counts {n32 - 1} tests, Passed counts {n32})"),  # the two counts disagree on the tests
            ("fp32/N8_T2/pipeline-b2b", plain.replace(f"0 of {n32} tests ran the back", f"{n32} of {n32} tests ran the back"), 0, "PASS"),
            ("fp32/N8_T2/pipeline-b2b", plain, 0, f"FAIL (0 of {n32} tests ran the back-to-back pass)"),  # built without the define
            ("fp32/N8_T2/pipeline-b2b", plain.replace(f"0 of {n32} tests ran the back", f"{n32 - 1} of {n32} tests ran the back"), 0,
             f"FAIL ({n32 - 1} of {n32} tests ran the back-to-back pass)"),
            ("fp32/N8_T2/pipeline-ck0-b2b", plain.replace(str(n32), str(n32 - n32p)).replace(f"0 of {n32 - n32p} tests ran the back",
             f"{n32 - n32p} of {n32 - n32p} tests ran the back"), 0, "PASS"),
            ("fp32/N8_T2/pipeline-ck0-b2b", plain.replace(f"0 of {n32} tests ran the back", f"{n32} of {n32} tests ran the back"), 0,
             f"FAIL ({n32} tests ran, fp32 has {n32 - n32p})")):
        got = v(p, text, rc)
        assert got == want, (p, text, rc, got, want)


@selftest
def test_residual_points():
    """Every int8 (N, T) has a residual point and no float format has one; it runs make residual like the other kinds; its verdict needs RESIDUAL: PASS."""
    pts = SPEC.points(False, list(FORMATS))
    nt = {p.rsplit("/", 1)[0] for p in pts if p.startswith("int8/N") and p.endswith("/pipeline")}
    res = {p.rsplit("/", 1)[0] for p in pts if p.endswith("/residual")}
    assert res == nt and all(p.startswith("int8/") for p in pts if p.endswith("/residual")), (sorted(res), sorted(nt))
    py = sys.executable
    assert SPEC.argv("int8/N16_T4/residual", None) == ["make", "residual", "FMT=int8", f"PYTHON={py}", "N=16", "TILE=4", "LANES=32"]
    assert SPEC.argv("int8/N64_T8/residual", None)[-1] == "LANES=128"
    for bad in ("fp32/N8_T2/residual", "bf16/N16_T4/residual"):
        try:
            SPEC.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse accepted {bad}")
    v = lambda text, rc=0: SPEC.verdict("int8/N8_T2/residual", text, rc)
    assert v("RESIDUAL: PASS\n") == "PASS" and v("RESIDUAL: PASS\n", 1) == "FAIL (exit 1)" and v("residual tiles ...\n") == "FAIL (no 'RESIDUAL: PASS')"
    assert v("RESIDUAL: FAIL tiles\n").startswith("FAIL (log: RESIDUAL: FAIL") and v("RESIDUAL: PASS\n[x] Assertion failed in t\n").startswith("FAIL (log:")


@selftest
def test_unit_points():
    """The int8 unit benches are points run once when int8 is selected: make add-lanes and make rq-lanes, each passing only on its bench's lines and its true-function gate."""
    assert SPEC.points(True, ["int8"])[:3] == ["selftest", "add-lanes", "rq-lanes"] and "add-lanes" not in SPEC.points(True, ["fp32", "bf16"])
    py = sys.executable
    assert SPEC.argv("add-lanes", None) == ["make", "add-lanes", f"PYTHON={py}"] and SPEC.argv("rq-lanes", None) == ["make", "rq-lanes", f"PYTHON={py}"]
    ok = ("add_lanes: 48 beats, 0 errors, 0 latency errors, 0 tag errors, 0 busy errors (64 busy cycles); 3 outputs differ from the bit-exact model (reported)\n"
          "RESULT: PASSED\nTRUTH int8 qadd: PASS  n 1536\nADD LANES: PASS (1536 outputs, 0 over the bound)\n")
    av = lambda text, rc=0: SPEC.verdict("add-lanes", text, rc)
    assert av(ok) == "PASS" and av(ok, 2) == "FAIL (exit 2)"
    assert av(ok.replace("0 tag errors", "3 tag errors").replace("PASSED", "FAILED")).startswith("FAIL")
    assert av(ok.replace("0 busy errors", "1 busy errors")).startswith("FAIL") and av(ok.replace("ADD LANES: PASS (1536", "ADD LANES: FAIL (1536")).startswith("FAIL")
    assert av(ok.replace("ADD LANES: PASS (1536 outputs, 0 over the bound)\n", "")) == "FAIL (no 'ADD LANES: PASS')"  # the bench alone is not the gate
    rok = ("TB_requant_lanes: 48 beats, 0 errors, 0 latency errors; 0 outputs differ from the bit-exact model (reported)\nRESULT: PASSED\n"
           "REQUANT LANES: PASS (1536 outputs, 0 over the bound)\n")
    rv = lambda text, rc=0: SPEC.verdict("rq-lanes", text, rc)
    assert rv(rok) == "PASS" and rv("RESULT: PASSED\n") == "FAIL (no 'TB_requant_lanes: \\d+ beats, 0 errors, 0 latency errors;')" and rv("built\n").startswith("FAIL (no")
    assert rv(rok.replace("REQUANT LANES: PASS", "REQUANT LANES: FAIL")).startswith("FAIL (log: REQUANT LANES: FAIL")
    assert SPEC.parse("rq-lanes") == {"n": 0, "kind": "rq-lanes"} and regress.point_env(SPEC, "add-lanes", "s", 8)["TRACE"] == "0"
    r = make_n("rq-lanes", f"PYTHON={py}")
    assert r.returncode == 0 and "--action rq-vectors" in r.stdout and "--top-module TB_requant_lanes" in r.stdout and "--check" in r.stdout, r.stdout[-300:]
    assert "Requantizer/src/int8Requantizer.sv" in r.stdout and "tfliteRequant" not in r.stdout, r.stdout[-300:]


@selftest
def test_add_lanes_real_clamp():
    """add-lanes' check keeps the fused activation's real clamp: in add-vectors' set 2 (ReLU6, hi -56.426 codes) an output one code above the rail fails."""
    with tempfile.TemporaryDirectory() as d:
        mem, tj, out = (os.path.join(d, x) for x in ("add.mem", "add.json", "out.mem"))
        with contextlib.redirect_stdout(io.StringIO()):
            add_vectors_main(["--action", "add-vectors", mem, "--truth", tj])
        hi = json.load(open(tj))["sets"][2]["hi"]
        assert abs(hi + 56.426) < 1e-3, hi
        want = [[int(x[4:6], 16) for x in ln.split()[1:]] for ln in open(mem) if len(ln.split()[0]) == 4]  # each beat's bit-exact codes

        def check(codes):
            open(out, "w").write("".join(" ".join(f"{c:02x}" for c in r) + "\n" for r in codes))
            return subprocess.run([sys.executable, os.path.join(ROOT, "regression.py"), "--action", "add-vectors", "--check", out, "--truth", tj],
                                  capture_output=True, text=True)
        r = check(want)
        assert r.returncode == 0 and "ADD LANES: PASS" in r.stdout, r.stdout[-300:]
        rail = [(b, k) for b in range(2 * ADD_PER, 3 * ADD_PER) for k in range(ADD_LANES) if truth.int8_codes(want[b][k]) == -56]
        bumped = [[(c + 1) & 0xFF if (b, k) in rail else c for k, c in enumerate(row)] for b, row in enumerate(want)]
        r = check(bumped)
        assert rail and r.returncode == 1 and f"ADD LANES: FAIL (1536 outputs, {len(rail)} over the bound)" in r.stdout, (len(rail), r.stdout[-400:])


@selftest
def test_model_points_skip_without_models():
    """A model point with no model directory (unset, missing or holding no model) is SKIPPED, not failed; make's default is the repo's models/;
    inside a run a model whose input files are absent is skipped by name, and a run that ran no model fails."""
    with tempfile.TemporaryDirectory() as d:
        o = lambda md: type("O", (), {"model_dir": md})()
        assert SPEC.skipped("fp32/N8_T2/pipeline", o("")) is None and SPEC.skipped("selftest", o("")) is None and SPEC.skipped("add-lanes", o("")) is None
        assert SPEC.skipped("fp32/N8_T2/model", o("")) == "no MODEL_DIR" and SPEC.skipped("bf16/N8_T2/model-ck0", o(d + "/x")).startswith("no MODEL_DIR")
        assert SPEC.skipped("fp32/N8_T2/model", o(d)) == f"no MODEL_DIR ({d} holds no model)"
        open(os.path.join(d, "vww_96_float.tflite"), "w").write("")
        assert SPEC.skipped("fp32/N8_T2/model", o(d)) is None
        assert mr.missing_inputs("vww", d) == ([], "") and mr.missing_inputs("kws", d)[0] == ["kws_ref_model_float32.tflite", "marvin_617de221_0.wav"]
    shipped = os.path.join(ROOT, "models")
    assert all(os.path.isfile(os.path.join(shipped, m)) for m in mr.MODELS.values()) and SPEC.skipped("fp32/N8_T2/model", o(shipped)) is None
    assert mr.missing_inputs("resnet8", shipped)[1] == "run models/fetch_cifar10.sh" or not mr.missing_inputs("resnet8", shipped)[0]
    r = subprocess.run(["make", "-s", "-n", "model", "PYTHON=py"], cwd=ROOT, env=MAKE_ENV, capture_output=True, text=True)
    assert f"--model-dir {shipped} " in r.stdout and "/proj/" not in r.stdout.replace(ROOT, "<root>"), r.stdout[-300:]
    assert SPEC.verdict("fp32/N8_T2/model", "MODEL GATE: SKIPPED (no model has its files in /m)\n", 0) == "FAIL (MODEL GATE: SKIPPED: no model ran)"


@selftest
def test_tb_knobs_are_run_time():
    # TB_sienna_top reads the TB-only settings from test_knobs.mem, so no knob value can reach its elaboration or the DUT's
    src = open(os.path.join(TB_DIR, "top", "TB_sienna_top.sv")).read()
    found = knob_references(build_sources())  # every file TB_sienna_top's build compiles or can include, the package's own declarations aside
    assert not found and len(build_sources()) > 30, found
    with tempfile.TemporaryDirectory() as d:  # the scan finds a knob anywhere but a package declaration
        ref, pkg = os.path.join(d, "unit.svh"), os.path.join(d, "test_config_pkg.sv")
        open(ref, "w").write("localparam int K = test_config_pkg::NUM_SETS;\n// PACKED only in a comment is a reference too\n")
        open(pkg, "w").write("package test_config_pkg;\n  localparam int NUM_SETS = 17;\n  localparam int X = NUM_SETS;\nendpackage\n")
        assert knob_references([ref, pkg]) == [(ref, 1, "NUM_SETS"), (ref, 2, "PACKED"), (pkg, 3, "NUM_SETS")], knob_references([ref, pkg])
    labels = set(re.findall(r'^\s*"(\w+)"\s*:', src[src.index("task automatic read_knobs"):], re.M))
    task = src[src.index("task automatic read_knobs"):src.index("endtask", src.index("task automatic read_knobs"))]
    assert re.search(r"if \(seen\[i\]\) \$fatal\(1, .*twice", task), task  # a setting given twice stops the run (simulated: the duplicate-knob run)
    assert labels >= {k.lower() for k in mr.TB_KNOBS} and len(labels & {k.lower() for k in mr.TB_KNOBS}) == len(mr.TB_KNOBS), labels
    mk = re.search(r"^TB_KNOBS\s*=\s*(.*)$", open(os.path.join(ROOT, "Makefile")).read(), re.M)
    assert mk and mk.group(1).split() == list(mr.TB_KNOBS), mk and mk.group(1)
    items = reg._config_items({"n": 16}, "fp32", "tanh", 9, 15, 3, ["tanh", "relu"], True, 77)
    with tempfile.TemporaryDirectory() as d:
        mr.write_test_config(os.path.join(d, "test_config_pkg.sv"), items)
        knobs = dict(x.split() for x in open(os.path.join(d, "test_knobs.mem")).read().splitlines())
        pkg = open(os.path.join(d, "test_config_pkg.sv")).read()
    want = {k: v for k, v, _ in items if k in mr.TB_KNOBS}
    assert knobs == {k.lower(): str(v) for k, v in want.items()} and len(knobs) == len(mr.TB_KNOBS), knobs
    assert all(f"localparam int {k} = {v};" in pkg for k, v, _ in items)  # the package keeps every item (TB_sienna_multi reads them)


@selftest
def test_build_groups():
    # two tests share a build exactly when their packages agree on everything but the run-time knobs
    base = reg._config_items({"n": 16, "tile_size": 4, "pool_h": 2, "pool_w": 2, "padding": 1}, "bf16", "tanh", 17, 15, 1, [], False, 5)
    text = lambda items: "".join(f"  localparam int {k} = {v};\n" for k, v, _ in items)
    knob_only = [(k, (v + 1) if k in mr.TB_KNOBS else v, t) for k, v, t in base]
    assert build_key(text(base)) == build_key(text(knob_only))
    for k in [k for k, _, _ in base if k not in mr.TB_KNOBS]:  # any other item, POOL_H, SETS_IN_FLIGHT or DROPOUT_P_PERCENT among them, needs its own build
        other = [(n, (v + 1) if n == k else v, t) for n, v, t in base]
        assert build_key(text(base)) != build_key(text(other)), k
    for N, T in ((8, 2), (16, 4), (32, 8)):
        for f in ("fp32", "bf16", "int8"):
            tests = [t for t in PIPELINE_TESTS if f in t.get("formats", tuple(FORMATS))]
            order = build_order(tests, N, T, 128 if N >= 64 else 32, f)
            assert sorted(t["name"] for t in order) == sorted(t["name"] for t in tests), (N, T, f)  # every test once
            keys = [predicted_key(t, N, T, 32, f) for t in order]
            assert all(keys.index(k) == i or keys[i - 1] == k for i, k in enumerate(keys)), (N, T, f)  # each group contiguous


@selftest
def test_sim_reuses_only_its_own_build():
    r = make_n("sim", "GEN_PKG=0", "FMT=fp32")
    assert r.returncode == 0 and "build_key.txt" in r.stdout and "cmp" in r.stdout and "verilator --binary" not in r.stdout, r.stdout[-400:]
    mk = open(os.path.join(ROOT, "Makefile")).read()
    cp = mk[mk.index("define copy_mem_files"):mk.index("endef", mk.index("define copy_mem_files"))]
    assert "cp $$d/$$p $(1)/" in cp and "cp -u" not in cp, cp  # every test's stimulus copied, whatever the timestamps: cp -u skipped a file of the same second
    rec = mk[mk.index("\nverilator: $(PKG_DEP)"):mk.index("\n\n", mk.index("\nverilator: $(PKG_DEP)") + 2)]  # make verilator: the key goes only after a whole build
    assert rec.index("rm -f $(VERILATOR_DIR)/build_key.txt") < rec.index("$(VERILATOR) --binary") < rec.index("-f V$(TOP_MODULE).mk") < rec.index("> $(VERILATOR_DIR)/build_key.txt"), rec
    with tempfile.TemporaryDirectory() as d:  # the key itself: knob values leave it unchanged; any other item, an input file or a flag changes it
        pkg, vdir = fixture_pkg(d, "fp32"), os.path.join(d, "V")
        os.makedirs(vdir)
        rtl = os.path.join(d, "unit.sv")
        open(rtl, "w").write("module unit; endmodule\n")
        open(os.path.join(vdir, "VTB_sienna_top__ver.d"), "w").write(f"{vdir}/x.cpp : {pkg} {rtl}\n")
        key = lambda *a: subprocess.run(["make", "-s", "build-key", f"PKG_FILE={pkg}", f"VERILATOR_DIR={vdir}", *a], cwd=ROOT, env=MAKE_ENV,
                                        capture_output=True, text=True).stdout
        k0, body = key(), open(pkg).read()
        assert k0 and "unit.sv" in k0 and "NUM_SETS" not in k0 and "POOL_H" in k0, k0
        open(pkg, "w").write(re.sub(r"NUM_SETS = \d+", "NUM_SETS = 99", re.sub(r"ACTIVATION_CODE = \d+", "ACTIVATION_CODE = 3", body)))
        assert key() == k0
        open(pkg, "w").write(re.sub(r"POOL_H = \d+", "POOL_H = 3", body))
        assert key() != k0
        open(pkg, "w").write(body)
        assert key() == k0 and key("LINK_STAGES=1") != k0
        open(rtl, "w").write("module unit; wire w; endmodule\n")
        assert key() != k0


@selftest
def test_regression_target():
    py = f"PYTHON={sys.executable}"
    def recipe(*args):
        r = make_n(*args, py)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    for args, pts in ((("SMALL=1",), SPEC.points(True, list(FORMATS))), ((), SPEC.points(False, list(FORMATS))),
                      (("SMALL=1", "FMT=int8"), SPEC.points(True, ["int8"]))):
        lines = recipe("regression", "MODEL_DIR=/m", *args)
        names = [x.split("--name ")[1].split()[0] for x in lines if "--action point " in x]
        if DEFAULT_POINT in pts:  # make runs the default point's twin first
            assert names.index(DEFAULT_TWIN) < names.index(DEFAULT_POINT), names[:8]
            pts = [p for p in pts if p != DEFAULT_TWIN]
            pts.insert(pts.index(DEFAULT_POINT), DEFAULT_TWIN)
        assert names == pts, (args, sorted(set(names) ^ set(pts))[:5], names[:8])
        assert all("--results " in x and "--model-dir=/m" in x and "--run " in x and x.rstrip().endswith("--keep-going") for x in lines if "--action point " in x), lines[:2]
        assert "--action summary" in lines[-1] and "--run " in lines[-1] and ("--small" in lines[-1]) == ("SMALL=1" in args), lines[-1]
        assert lines[-1].split("--formats ")[1].split()[0] == ("int8" if "FMT=int8" in args else "fp32,bf16,int8"), lines[-1]
    small = [x.split("--name ")[1].split()[0] for x in recipe("regression", "SMALL=1", "MODEL_DIR=/m") if "--action point " in x]
    assert [p for p in small if p.endswith("-ck0")] == [f"{f}/N{n}_T{t}/{k}" for f in FORMATS for n in (8, 16) for t in pow2_tile_sizes(n)
                                                       for k in ["pipeline-ck0"] + (["gemm-ck0", "tflite-ck0"] if f == "int8" else ["model-ck0", "gemm-ck0"])]
    assert [p for p in small if p.endswith("-b2b")] == [f"{f}/N{n}_T{t}/{k}" for f in FORMATS for n in (8, 16) for t in pow2_tile_sizes(n)
                                                       for k in ("pipeline-b2b", "pipeline-ck0-b2b")]
    full = [x.split("--name ")[1].split()[0] for x in recipe("regression", "MODEL_DIR=/m") if "--action point " in x]
    assert {SPEC.parse(p)["n"] for p in full if p.endswith("-b2b")} == {8, 16, 32, 64} and len([p for p in full if p.endswith("-b2b")]) == 3 * (7 * 2 + 11)
    assert {SPEC.parse(p)["n"] for p in full if p.endswith("-ck0-b2b")} == {8, 16}
    one = recipe("regression/fp32/N8_T2/pipeline")
    assert len(one) == 1 and "--name fp32/N8_T2/pipeline" in one[0] and "--keep-going" not in one[0] and "--run" not in one[0], one
    s = recipe("regression-summary", "SMALL=1")
    assert len(s) == 1 and "--action summary" in s[0] and "--small" in s[0] and "--run" not in s[0], s
    r = make_n("regression", "FMT=fp16")
    assert r.returncode != 0 and "FMT=fp16: not one of" in r.stderr, r.stderr
    r = make_n("regression", "SMALL=2")
    assert r.returncode != 0 and "Invalid SMALL=2" in r.stderr, r.stderr
    assert recipe("model", "MODEL_DIR=/m")[-1].split()[-2:] == ["--count", "1"] and recipe("model", "MODEL_DIR=/m", "MODEL_COUNT=3")[-1].split()[-2:] == ["--count", "3"]
    assert make_n("check").returncode != 0  # the gate's old name is gone
    r = make_n("gemm", "COLLAPSE_K=0", "FMT=fp32", "PYTHON=py")
    assert r.returncode == 0 and "--collapse-k 0" in r.stdout, r.stdout + r.stderr
    h = subprocess.run(["make", "-s", "help"], cwd=ROOT, env=MAKE_ENV, capture_output=True, text=True).stdout
    assert "make sim " in h and "make build-key " in h and "sienna_ck0" not in open(os.path.join(ROOT, "Makefile")).read(), h[-300:]
    with tempfile.TemporaryDirectory() as d:  # a relative MODEL_DIR reaches the points absolute: they run in a private copy
        rel = os.path.relpath(d, ROOT)
        lines = recipe("regression", "SMALL=1", f"MODEL_DIR={rel}")
        assert all(f"--model-dir={os.path.realpath(d)}" in x.replace(d, os.path.realpath(d)) for x in lines if "--action " in x), lines[0]
    env = dict(MAKE_ENV, FMT="bf16", SMALL="1")  # exported, FMT and SMALL do not narrow it: only make's command line does
    r = subprocess.run(["make", "-s", "-n", "regression", py], cwd=ROOT, env=env, capture_output=True, text=True)
    names = [x.split("--name ")[1].split()[0] for x in r.stdout.splitlines() if "--action point " in x]
    assert r.returncode == 0 and sorted(names) == sorted(SPEC.points(False, list(FORMATS))), (r.stderr[-200:], len(names))


@selftest
def test_gpnae_target_tier():
    r = make_n("gpnae-verilator", "FMT=bf16", "PYTHON=py")
    assert r.returncode == 0 and "py regression.py --format bf16 --run gate --tier 0" in r.stdout, r.stdout + r.stderr
    r = make_n("gpnae-verilator", "FMT=bf16", "GPNAE_TIER=2", "PYTHON=py")
    assert r.returncode == 0 and "--tier 2" in r.stdout, r.stdout + r.stderr


@selftest
def test_layer_mode_check():
    line = "[LAYER] sets=4 outputs=64 cycles=99 a_rows=2/2 w_rows=2/2 epilogues=0/0 out_stall=0 in_stall=0"
    tb = open(os.path.join(ROOT, "testbenches", "top", "TB_model_run.sv")).read()  # the mode printed is the elaborated mesh's, not the package's
    assert "in_stall=%0d collapse_k=%0d" in tb and "dut.G_LAYER.pipe.systolic_array_inst.COLLAPSE_K);" in tb, "TB_model_run's [LAYER] line lost the mesh's COLLAPSE_K"
    old = mr.COLLAPSE_K
    try:
        for ck in (0, 1):
            mr.COLLAPSE_K = ck
            mr.check_layer_mode(f"{line} collapse_k={ck}\n", "t")
            for text, want in ((f"{line} collapse_k={1 - ck}\n", f"t: the build's mesh runs collapse-k {1 - ck}, the run asked for {ck}"),
                               (line + "\n", "t: the [LAYER] line has no collapse_k")):
                try:
                    mr.check_layer_mode(text, "t")
                except RuntimeError as e:
                    assert want in str(e), (want, str(e))
                    continue
                raise AssertionError(f"collapse-k {ck}: accepted {text!r}")
    finally:
        mr.COLLAPSE_K = old


@selftest
def test_layer_file_format():
    N, rng = 16, np.random.default_rng(5)
    req = dict(zp=3, amin=-128, amax=127, mx=0, shx=0, mout=0, shout=0, zout=0)
    models = [{"W": rng.integers(-5, 5, (6, 8)).astype(np.float32), "bias": rng.integers(-9, 9, 8), "act": a,
               "req": dict(req, mult=np.full(8, 1 << 30), shift=np.full(8, -3)), "inputs": [rng.integers(-9, 9, (5, 6)).astype(np.float32)]}
              for a in ("linear", "relu")]
    job, _ = mr.pack_jobs(models, N, int8=True)
    cfg, a, w, _ = mr.format_layer(job, N)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.layer")
        mr.write_layer(path, cfg, a, w, "int8", job["req"], job["pack"], mr.layer_epilogue(job, N))
        lines = open(path).read().splitlines()
    head = [x for x in lines if x[:1] in ("L", "Q", "P", "E")]
    assert [x[0] for x in head] == ["L", "Q", "P"] + ["E"] * (mr.PACK_ENTRIES - 1), head
    assert len(head[0].split()) == 12 and head[0].split()[-1] == "1" and len(head[2].split()) == 2 + N // 2, head
    words, blocks = (len(a) + len(w)) * N, -(-cfg["n"] // N)
    assert len(lines) == len(head) + words + blocks * 3 * N, (len(lines), len(a), len(w))
    assert all(len(x) == 2 for x in lines[len(head):len(head) + words])  # int8 words: two hex digits
    assert all(len(x) == 8 for x in lines[len(head) + words:])  # the epilogue's int32 words


@selftest
def test_backends_share_run_job():
    fjob = {"terms": [(np.eye(20, dtype=np.float32), np.arange(20 * 9, dtype=np.float32).reshape(20, 9) - 90)], "bias": np.ones(9, np.float32),
            "act": "relu", "shape": (20, 9)}
    for cls in (mr.RtlLayer, mr.RtlSets, mr.Emulator):
        assert callable(getattr(cls, "build")) and callable(getattr(cls, "run_job")), cls
    y, sets, cyc = mr.Emulator(16, 32, tempfile.gettempdir()).run_job(fjob, "t")
    assert np.array_equal(y, mr.job_reference(fjob)[1].astype(np.float32)) and sets == 4 and cyc == 0, (sets, cyc)


@selftest
def test_pkg_action_refusals():
    run = lambda *a: subprocess.run([sys.executable, "regression.py", "--action", "pkg", *a], cwd=ROOT, capture_output=True, text=True)
    r = run("--format", "fp32", "--test", "no_such_test")
    assert r.returncode != 0 and "no pipeline test named 'no_such_test'" in r.stderr, r.stderr
    only8 = next(t["name"] for t in reg.PIPELINE_TESTS if t.get("formats") == ("int8",))
    r = run("--format", "fp32", "--test", only8)
    assert r.returncode != 0 and f"test '{only8}' runs only in int8, not fp32" in r.stderr, r.stderr
    assert "formats" not in next(t for t in reg.PIPELINE_TESTS if t["name"] == reg.PKG_DEFAULT_TEST)  # the default runs everywhere


@selftest
def test_regression_selects_nothing_fails():
    r = subprocess.run([sys.executable, "regression.py", "--format", "fp32", "--test", "no_such_test"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "No tests found containing 'no_such_test'" in r.stdout + r.stderr, (r.returncode, r.stdout[-400:])
    only8 = next(t["name"] for t in reg.PIPELINE_TESTS if t.get("formats") == ("int8",))
    r = subprocess.run([sys.executable, "regression.py", "--format", "fp32", "--test", only8], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "No tests found" in r.stdout + r.stderr and "Sanity Clean" not in r.stdout, (r.returncode, r.stdout[-400:])
    saved, reg.PIPELINE_TESTS = reg.PIPELINE_TESTS, []
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                reg.run_regression(16, 4)
                code = 0
            except SystemExit as e:
                code = e.code
    finally:
        reg.PIPELINE_TESTS = saved
    assert code == 1 and "No tests found for fp32" in out.getvalue(), (code, out.getvalue()[-300:])


@selftest
def test_dropout_mask_check():
    # Independent masks pass (N = 8: 25 windows, N = 16: 81); equal seeds, more than 75% of 81, and a weak shared component over 17 sets fail
    ms = [tuple(reg.dropout_keep(25, 0.5, reg.set_dropout_seed(0x2ACE0000 + 42, k), 32)) for k in range(17)]
    reg.check_masks_differ("n8_independent", ms)
    for bad in ([ms[0]] * 3, [(True,) * 81, (True,) * 61 + (False,) * 20]):
        try:
            reg.check_masks_differ("dependent", bad)
            raise AssertionError(f"masks agreeing on {sum(a == b for a, b in zip(bad[0], bad[1]))} of {len(bad[0])} passed")
        except RuntimeError as e:
            assert "dropout masks agree" in str(e), e
    reg.check_masks_differ("n16_58_of_81", [(True,) * 81, (True,) * 58 + (False,) * 23])  # chance 6e-5 if independent
    reg.check_masks_differ("n16_independent", [tuple(reg.dropout_keep(81, 0.5, reg.set_dropout_seed(0x2ACE0000 + 42, k), 32)) for k in range(17)])
    # 17 sets sharing a component (each window the shared draw with chance 0.45): pairs agree about 60%, below the pair rule's 92% at 25 windows
    rng = np.random.default_rng(3)
    shared = rng.random(25) < 0.5
    corr = [tuple(np.where(rng.random(25) < 0.45, shared, rng.random(25) < 0.5).tolist()) for _ in range(17)]
    pairs = [sum(a == b for a, b in zip(corr[i], corr[j])) / 25 for i in range(17) for j in range(i + 1, 17)]
    assert 0.55 < sum(pairs) / len(pairs) < 0.65 and max(pairs) < 0.92, (sum(pairs) / len(pairs), max(pairs))
    try:
        reg.check_masks_differ("correlated", corr)
        raise AssertionError("17 sets at about 60% pairwise agreement passed")
    except RuntimeError as e:
        assert "pair windows" in str(e), e

@selftest
def test_formats_match_fmt_mk():
    mk = open(os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "fmt.mk")).read()
    assert re.search(r"^FORMATS\s*=\s*(.*)$", mk, re.M).group(1).split() == list(reg.FORMATS) == list(fmt_pkg.FORMATS)


# ── tool: _parse_log (was test_regression_parse.py) ──

LOG_CLEAN = """  Total     : 512
  Exact     : 512
  Tol pass  : 0
  Failed    : 0
pipeline_complete_o asserted @ 3165000  (311 cycles)
RESULT: PASSED
"""

LOG_FIRED = ("[3165000] %Error: sienna_top.sv:1128: Assertion failed in TB_sienna_top.dut.a_pack_one_pass: "
             "sienna_top: a packed set cannot be a partial sum or continue one\n")


@selftest
def test_clean_log_passes():
    r = reg._parse_log(LOG_CLEAN)
    assert r["status"] == "PASS" and r["total"] == 512 and r["cyc"] == 311, r


@selftest
def test_assertion_firing_fails():
    r = reg._parse_log(LOG_CLEAN + LOG_FIRED)  # the TB's own counts still read clean
    assert r["status"] == "ASSERT" and r["failed"] == 1 and "a_pack_one_pass" in r["fired"], r


@selftest
def test_assertion_without_error_prefix_fails():
    r = reg._parse_log(LOG_CLEAN + "Assertion failed in TB_sienna_top.dut.a_credit_accept\n")
    assert r["status"] == "ASSERT", r


@selftest
def test_error_line_fails():
    r = reg._parse_log(LOG_CLEAN + "%Error: TB_sienna_top.sv:12: some runtime error\n")
    assert r["status"] == "ASSERT", r


@selftest
def test_warning_is_not_a_firing():
    r = reg._parse_log("%Warning-UNUSEDSIGNAL: sienna_top.sv:40: Signal is not used\n" + LOG_CLEAN)
    assert r["status"] == "PASS", r


# ── tool: pack_jobs and the packed-layer prechecks (was test_pack_jobs.py) ──


def _models(rng, shapes, acts):
    return [{"W": rng.uniform(-1, 1, (K, C)), "bias": rng.uniform(-1, 1, C), "act": a, "req": None,
             "inputs": [rng.uniform(-1, 1, (m, K)) for m in ms]} for (K, C, ms), a in zip(shapes, acts)]


@selftest
def test_round_trip_float():
    rng = np.random.RandomState(3)
    models = _models(rng, [(5, 3, [2, 7]), (8, 8, [1]), (2, 6, [4, 4, 3])], ["tanh", "relu", "linear"])
    job, recipe = mr.pack_jobs(models, 32, int8=False)
    (A, B), = job["terms"]
    assert job["pack"]["shift"] == 2 and A.shape == (32, 32) and B.shape == (32, 32)  # b = 8, rows padded to N
    Y = A @ B + job["bias"][None, :]
    for m, outs in zip(models, mr.unpack(Y, recipe)):
        for x, y in zip(m["inputs"], outs):
            assert np.allclose(y, x @ m["W"] + m["bias"]), "a job's rows and columns came back wrong"
    pk = job["pack"]
    assert pk["map"][:3] == [0, 1, 2] and set(pk["map"][3:]) == {0}, "each model's block must point at its own entry"
    assert [e[0] for e in pk["ents"][:3]] == ["tanh", "relu", "linear"] and len(pk["ents"]) == 8


@selftest
def test_partial_packing_and_entries():
    rng = np.random.RandomState(4)
    models = _models(rng, [(3, 3, [1]), (4, 2, [2])], ["selu", "selu"])
    job, recipe = mr.pack_jobs(models, 16, int8=False)
    pk = job["pack"]
    assert pk["shift"] == 2 and pk["map"][:4] == [0, 0, 0, 0] and pk["ents"][0][0] == "selu"  # same setting, one entry; blocks 2, 3 empty
    Y = job["terms"][0][0] @ job["terms"][0][1]
    assert np.all(Y[:, 8:] == 0), "an empty block's columns must stay zero before bias"


@selftest
def test_refusals():
    rng = np.random.RandomState(5)
    big = _models(rng, [(9, 4, [1])], ["linear"])
    for bad, why in ((big, "K > N/2"), (_models(rng, [(2, 2, [1])] * 9, ["tanh", "relu", "linear", "selu", "sigmoid", "tanh",
                                                                          "relu", "linear", "selu"]), "more models than blocks")):
        try:
            mr.pack_jobs(bad, 16, int8=False)
        except ValueError:
            continue
        raise AssertionError(f"pack_jobs accepted {why}")


@selftest
def test_refuses_wide_output():
    rng = np.random.RandomState(6)
    try:
        mr.pack_jobs(_models(rng, [(2, 9, [1])], ["linear"]), 16, int8=False)
    except ValueError:
        return
    raise AssertionError("pack_jobs accepted C > N/2")


def _int8_model(rng, i, act="linear"):
    req = dict(mult=np.array([1000 + 10 * i, 2000 + 10 * i]), shift=np.array([i, i + 1]), zp=i, amin=-128, amax=127, mx=0, shx=0,
               mout=0, shout=0, zout=0)
    return {"W": rng.randint(-5, 5, (2, 2)).astype(float), "bias": rng.randint(-9, 9, 2), "act": act, "req": req,
            "inputs": [rng.randint(-5, 5, (3, 2)).astype(float)]}


@selftest
def test_int8_layout_and_entry_limit():
    rng = np.random.RandomState(7)
    models = [_int8_model(rng, i) for i in range(9)]  # K = C = 2 at N = 32: b = 2, 16 blocks, 9 distinct zero points
    try:
        mr.pack_jobs(models, 32, int8=True)
    except ValueError as e:
        assert "distinct" in str(e), f"refused for the wrong reason: {e}"
    else:
        raise AssertionError("pack_jobs accepted 9 settings in a set that holds 8")
    job, recipe = mr.pack_jobs(models[:8], 32, int8=True)
    pk, q = job["pack"], job["req"]
    assert pk["shift"] == 4 and pk["map"] == list(range(8)) + [0] * 8 and len(pk["ents"]) == 8
    assert q["zp"] == 0 and all(q[x] == models[0]["req"][x] for x in ("amin", "amax", "mx", "shx", "mout", "shout", "zout"))
    assert [e[1]["zp"] for e in pk["ents"]] == list(range(8)) and "mult" not in pk["ents"][3][1], "entries carry the output words only"
    assert job["bias"].dtype == np.int64 and q["mult"].shape == (32,) and q["shift"].shape == (32,)
    for c, m in enumerate(models[:8]):
        assert list(q["mult"][2 * c:2 * c + 2]) == list(m["req"]["mult"]) and list(q["shift"][2 * c:2 * c + 2]) == list(m["req"]["shift"])
        assert list(job["bias"][2 * c:2 * c + 2]) == list(m["bias"]), f"model {c}'s bias is not at its columns"
    assert not np.any(q["mult"][16:]) and not np.any(q["shift"][16:]) and not np.any(job["bias"][16:]), "padding columns must be zero"
    Y = job["terms"][0][0] @ job["terms"][0][1] + job["bias"][None, :]
    for m, outs in zip(models[:8], mr.unpack(Y, recipe)):
        assert np.array_equal(outs[0], m["inputs"][0] @ m["W"] + m["bias"])


@selftest
def test_int8_shared_entry():
    rng = np.random.RandomState(8)
    a, b, c = _int8_model(rng, 0), _int8_model(rng, 0), _int8_model(rng, 0, "relu")
    b["req"] = dict(b["req"], mult=np.array([5, 6]))  # a column's own multiplier is not part of the entry
    job, _ = mr.pack_jobs([a, b, c], 16, int8=True)
    assert job["pack"]["map"][:3] == [0, 0, 1] and [e[0] for e in job["pack"]["ents"][:2]] == ["linear", "relu"]


@selftest
def test_selu_large_scale_accepted():
    # Task 1.16: the int8 lane's SELU is exact past lambda * x = 512, so a large-scale SELU entry packs and runs; its output scale is an encode_scale code.
    acc = np.arange(-2000, 2000, dtype=np.int64).reshape(-1, 2) * 500  # real inputs to +-1000: s_out about 7.8, lambda * x to 1051
    req = mr.requant_params(acc, 1e-3, np.ones(2), "selu")
    x = (mr.requantize(acc, req) - req["zp"]) * req["s_out"]
    assert x.max() * gpnae_model.LAMBDA_F > 512 and 0 <= req["shout"] <= ipu.REQ_MAX_SHIFT and req["mout"] < 1 << 24 and (req["mout"], req["shout"]) == \
        gpnae_model.selu_out_params(req["s_out"], req["mx"], req["s_selu"]), (x.max(), req["mout"], req["shout"])
    m = {"W": np.ones((2, 2)), "bias": None, "act": "selu", "req": req, "inputs": [np.ones((1, 2))]}
    job, _ = mr.pack_jobs([m], 16, int8=True)
    assert job["pack"]["ents"][0][0] == "selu"
    y = mr.activate_int8(mr.requantize(acc, req), "selu", req)
    t = np.clip(np.floor(np.where(x > 0, gpnae_model.LAMBDA_F * x, gpnae_model.LA_F * np.expm1(np.minimum(x, 0))) / req["s_selu"] + 0.5) + req["zout"], -128, 127)
    assert np.abs(y - t).max() <= 1, np.abs(y - t).max()  # the lane's words against the true function, past 512 too


def _packed(N, rng_seed=9):
    rng = np.random.RandomState(rng_seed)
    job, _ = mr.pack_jobs(_models(rng, [(3, 3, [2]), (2, 4, [3])], ["tanh", "relu"]), N, int8=False)
    return job


def _refused(job, N, lanes, why):
    with tempfile.TemporaryDirectory() as d:
        try:
            mr.RtlLayer(N, lanes, d).run_job(job, "t")  # every refusal is raised before the simulator would run
        except ValueError as e:
            assert not os.listdir(d), f"{why}: a layer file was written before the refusal"
            return str(e)
    raise AssertionError(f"RtlLayer accepted {why}")


@selftest
def test_precheck_accepts_legal():
    for N, lanes in ((16, 32), (32, 32), (64, 64), (64, 128), (8, 8)):
        job = _packed(N)
        cfg, _, _, (M, C, rt, _) = mr.format_layer(job, N)
        mr.pack_precheck(job["pack"], cfg, (M, C, rt), N, lanes)


@selftest
def test_precheck_lanes():
    assert "NUM_LANES" in _refused(_packed(64), 64, 32, "N = 64 at the default 32 lanes")
    assert "NUM_LANES" in _refused(_packed(16), 16, 24, "24 lanes at N = 16")


@selftest
def test_precheck_collapse_k0():
    old = mr.COLLAPSE_K
    mr.COLLAPSE_K = 0
    try:
        assert "collapse-k 0" in _refused(_packed(16), 16, 32, "a packed layer on the collapse-k 0 mesh")
    finally:
        mr.COLLAPSE_K = old


@selftest
def test_precheck_residual():
    job = _packed(16)
    A = job["terms"][0][0]
    job["terms"].append((np.ones_like(A), np.eye(16, dtype=np.float32)))  # an identity term is format_layer's residual
    assert "residual" in _refused(job, 16, 32, "a packed layer with a residual")


@selftest
def test_precheck_table():
    for edit, why in ((lambda pk: pk.update(map=pk["map"][:-1]), "a map shorter than N/2"),
                      (lambda pk: pk.update(map=pk["map"] + [0]), "a map longer than N/2"),
                      (lambda pk: pk["map"].__setitem__(1, 8), "a map entry past the table"),
                      (lambda pk: pk.update(ents=pk["ents"][:7]), "7 table entries"),
                      (lambda pk: pk.update(shift=0), "pack shift 0"),
                      (lambda pk: pk.update(shift=4), "pack shift log2(N)")):
        job = _packed(16)
        edit(job["pack"])
        _refused(job, 16, 32, why)


# ── tool: the model gate (model_runner's parse_fault, flip_bit, check_layer, judge, gate_verdict) ──

class _ExactSim:
    """A backend stand-in whose bit-exact model returns the given words."""

    def __init__(self, want):
        self.want = np.asarray(want, np.int64)

    def exact(self, job):
        return self.want


def _layer(err=0.0, differ=0, layer=3):
    """One layer's stats as execute records them; differ > 0 adds a first mismatch."""
    e = {"differ": differ, "words": 16}
    if differ:
        e.update(index=5, row=0, col=5, expected="3de8", got="3de9")
    return {"layer": layer, "err": err, "exact": e}


def _tally(runs=None, layers=9, hw=(), task=(), rep_seen=(), rep_failed=()):
    return {"layers": layers, "hw": list(hw), "task": list(task), "runs": dict(runs or {"resnet8": 1}),
            "rep_seen": set(rep_seen), "rep_failed": set(rep_failed)}


@selftest
def test_gate_parse_fault():
    assert mr.parse_fault(None) is None and mr.parse_fault("") is None
    f = mr.parse_fault("word:resnet8:L00:17")
    assert (f["kind"], f["model"], f["layer"], f["index"], f["bit"], f["applied"]) == ("word", "resnet8", "L00", 17, 0, False)
    assert mr.parse_fault("shared:ad01:L09:0:20")["bit"] == 20 and mr.parse_fault("top1:kws")["kind"] == "top1"
    for bad in ("word:resnet8:L00", "top1", "flip:resnet8:L00:1", "word:resnet8:L00:x"):
        try:
            mr.parse_fault(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_fault accepted {bad!r}")


@selftest
def test_gate_flip_bit():
    y = mr.bits_float(np.array([0x3DE8, 0x0000]), "bf16")
    assert mr.word_bits(mr.flip_bit(y, "bf16", 0, 0), "bf16").tolist() == [0x3DE9, 0]
    assert mr.word_bits(y, "bf16").tolist() == [0x3DE8, 0], "flip_bit must not change its input"
    assert mr.word_bits(mr.flip_bit(np.float32([0.0]), "fp32", 0, 30), "fp32").tolist() == [0x40000000]
    for fmt, bit in (("bf16", 16), ("fp32", 32), ("bf16", -1)):
        try:
            mr.flip_bit(y, fmt, 0, bit)
        except ValueError:
            continue
        raise AssertionError(f"flip_bit took bit {bit} of a {fmt} word")


@selftest
def test_gate_check_layer():
    want = np.array([[0x3F80, 0x4000], [0x0000, 0xBF80]])
    y = mr.bits_float(want, "bf16")
    _, rec = mr.check_layer(_ExactSim(want), None, y, "bf16")
    assert rec["differ"] == 0 and rec["words"] == 4, rec
    bad = want.copy()
    bad[1, 0] = 0x0001
    _, rec = mr.check_layer(_ExactSim(bad), None, y, "bf16")
    assert (rec["differ"], rec["index"], rec["row"], rec["col"], rec["expected"], rec["got"]) == (1, 2, 1, 0, "0001", "0000"), rec
    f = mr.parse_fault("word:m:L00:3:0")
    y2, rec = mr.check_layer(_ExactSim(want), None, y, "bf16", f)
    assert f["applied"] and rec["differ"] == 1 and rec["got"] == "bf81" and mr.word_bits(y2, "bf16")[1, 1] == 0xBF81, rec
    f = mr.parse_fault("shared:m:L00:3:0")
    _, rec = mr.check_layer(_ExactSim(want), None, y, "bf16", f)
    assert f["applied"] and rec["differ"] == 0, "a shared fault must pass the bit-exact check"
    f = mr.parse_fault("word:m:L00:4:0")  # one past the last word
    _, rec = mr.check_layer(_ExactSim(want), None, y, "bf16", f)
    assert not f["applied"] and rec["differ"] == 0
    with tempfile.TemporaryDirectory() as d:
        assert mr.check_layer(mr.Emulator(16, 32, d), None, y, "fp32")[1] is None
    try:
        mr.check_layer(object(), None, y, "bf16")
    except RuntimeError:
        return
    raise AssertionError("check_layer judged a backend without a bit-exact model")


@selftest
def test_gate_judge():
    top = {"hw_top": 3, "ref_top": 3}
    g = mr.judge("resnet8", "x", "fp32", [_layer(), _layer(3e-5)], top)
    assert not g["hw"] and not g["task"] and not g["reported"], g
    g = mr.judge("resnet8", "x", "bf16", [_layer(differ=1)], top)
    assert g["bad_layers"] == ["L03"] and "expected 3de8 got 3de9" in g["hw"][0], g
    assert mr.judge("resnet8", "x", "fp32", [_layer(float("nan"))], top)["hw"], "NaN error must fail"
    assert mr.judge("resnet8", "x", "fp32", [_layer(2e-4)], top)["hw"], "fp32 error above 1e-4 must fail"
    assert mr.judge("kws", "x", "bf16", [_layer(0.03)], top)["hw"], "bf16 error above 0.02 must fail"
    assert not mr.judge("kws", "x", "bf16", [_layer(0.01)], top)["hw"]
    g = mr.judge("vww", "x", "fp32", [_layer()], {"hw_top": 1, "ref_top": 0})
    assert g["task"] and "top-1 1 differs" in g["task"][0], g
    ad = lambda hw: {"hw_top": 0, "ref_top": 0, "score_hw": [hw], "score_ref": [10.0]}
    assert not mr.judge("ad01", "x", "fp32", [_layer()], ad(10.005))["task"]
    assert mr.judge("ad01", "x", "fp32", [_layer()], ad(10.02))["task"], "fp32 score above 1e-3 must fail"
    assert ("ad01", "bf16") not in mr.REPORTED, "bf16 sums in fp32: ad01 bf16 is gated"
    g = mr.judge("ad01", "x", "bf16", [_layer(1.8)], ad(27.0))
    assert g["hw"] and g["task"] and not g["reported"] and g["rep_key"] is None, "a bf16-sum ad01 (layer 1.8, score 2.7x) must fail"
    assert not mr.judge("ad01", "x", "bf16", [_layer(3e-3)], ad(10.0224))["task"], "ad01 bf16 at 2.24e-03 passes"
    shipped, mr.REPORTED = mr.REPORTED, {("ad01", "bf16")}  # the REPORTED mechanism on a stand-in entry
    try:
        g = mr.judge("ad01", "x", "bf16", [_layer(1.8)], ad(27.0))
        assert not g["hw"] and not g["task"] and len(g["reported"]) == 2 and g["rep_key"] == ("ad01", "bf16"), g
        assert mr.verdict_word(g, "hw") == mr.verdict_word(g, "task") == "FAIL (reported, not gated)", "a reported failure must not read PASS"
        g = mr.judge("ad01", "x", "bf16", [_layer(0.01)], ad(27.0))
        assert (mr.verdict_word(g, "hw"), mr.verdict_word(g, "task")) == ("PASS", "FAIL (reported, not gated)"), g
        g = mr.judge("ad01", "x", "bf16", [_layer(1.8, differ=1)], ad(27.0))
        assert len(g["hw"]) == 1 and g["bad_layers"] == ["L03"], "REPORTED must never suppress a bit-exact failure"
        assert mr.verdict_word(g, "hw") == "FAIL"
    finally:
        mr.REPORTED = shipped
    assert mr.verdict_word(mr.judge("vww", "x", "fp32", [_layer()], {"hw_top": 1, "ref_top": 0}), "task") == "FAIL"
    assert mr.verdict_word(mr.judge("vww", "x", "fp32", [_layer()], {"hw_top": 0, "ref_top": 0}), "task") == "PASS"


@selftest
def test_gate_verdict():
    assert mr.gate_verdict(_tally(), ["resnet8"])[0]
    ok, extra, _ = mr.gate_verdict(_tally(runs={"resnet8": 1}), ["resnet8", "kws"])
    assert not ok and "kws ran no inference" in extra[0], extra
    assert not mr.gate_verdict(_tally(runs={}, layers=0), ["resnet8"])[0], "zero inferences must fail"
    assert not mr.gate_verdict(_tally(hw=["FAIL hardware"]), ["resnet8"])[0]
    assert not mr.gate_verdict(_tally(task=["FAIL task"]), ["resnet8"])[0]
    f = mr.parse_fault("word:resnet8:L01:0")
    ok, extra, _ = mr.gate_verdict(_tally(), ["resnet8"], f)
    assert not ok and "never applied" in extra[0], extra
    f["applied"] = True
    assert mr.gate_verdict(_tally(), ["resnet8"], f)[0]
    ok, _, notes = mr.gate_verdict(_tally(rep_seen=[("ad01", "bf16")], rep_failed=[("ad01", "bf16")]), ["resnet8"])
    assert ok and not notes
    ok, _, notes = mr.gate_verdict(_tally(rep_seen=[("ad01", "bf16")]), ["resnet8"])
    assert ok and "can be dropped" in notes[0], notes


# ── tool: perf section (was test_perf_analysis.py) ──

PERF_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbenches", "perf_fixtures")


def perf_trace(name):
    return pa.events(open(os.path.join(PERF_FIX, f"{name}_N16_T4_bf16.trace")).read())


@perf_selftest
def test_accumulate_outputs_every_third_set():
    a = pa.analyse(perf_trace("matmul_accum3_bias_linear_nopool"), 24, passes=3)
    assert a["first_latency"] == 133  # host start of pass 0 to the completion of pass 2, read off the trace by hand
    assert statistics.mean(a["steady"]) == 19.0  # cycles per depth pass: 57 per output of 3 passes
    outs = [x for x in a["sets"] if x["latency"] is not None]
    assert len(outs) == 8 and all(x["latency"] > 0 and x["out"] >= 0 for x in outs)
    assert all(x["latency"] is None and x["out"] is None for k, x in enumerate(a["sets"]) if (k + 1) % 3)


@perf_selftest
def test_single_pass_unchanged():
    a = pa.analyse(perf_trace("matmul_relu_nopool"), 24)
    assert a["first_latency"] == a["sets"][0]["latency"] == 95  # the pre-fix report's value
    assert statistics.mean(a["steady"]) == 19.0
    assert len(a["sets"]) == 24 and all(x["out"] >= 0 for x in a["sets"])


@perf_selftest
def test_residual_hold_add_pairs():
    ev = pa.events(open(os.path.join(PERF_FIX, "int8_residual_relu_N8_T2_int8.trace")).read())
    a = pa.analyse(ev, 18, 1, [0x56, 0x58] * 9)  # add_<k>.mem's flags: hold sets and add sets in turn, ReLU, bias on the hold set
    assert a["hold"] == [3] * 9 and [x["act"] for x in a["sets"][1::2]] == [3] * 9  # G 115-118 (hold), 126-129 (add), off the trace by hand
    assert all(x["act"] is None and x["latency"] is None and x["mesh"] == 26 for x in a["sets"][0::2])  # a hold set outputs nothing
    assert a["first_latency"] == 146 - 85 and a["sets"][1]["add"]  # the hold set's host start to the add set's completion
    assert a["steady"] and all(g == 11.0 for g in a["steady"])  # 22 cycles per hold/add pair


PERF_P8 = dict(N=8, TILE_SIZE=2, NUM_LANES=32, HOST_WORDS=8, SRAM_DEPTH=64, IN_ROWS=8, IN_COLS=8, POOL_H=1, POOL_W=1, STRIDE_ROWS=1,
               STRIDE_COLS=1, PADDING=0, NUM_SETS=18)  # rs2_perf2_int8_n8_t2's test_config_pkg


@perf_selftest
def test_mesh_lat_passes():
    with perf_fmt("bf16"):  # matmul_accum3_bias_linear_nopool at N16 T4: first pass's launch to the third pass's result written, read off the trace
        a = pa.analyse(perf_trace("matmul_accum3_bias_linear_nopool"), 24, passes=3)
        P16 = {**PERF_P8, "N": 16, "TILE_SIZE": 4, "HOST_WORDS": 16, "SRAM_DEPTH": 256}
        m1, m3 = pa.model({"act": "linear"}, True, P16, 256), pa.model({"act": "linear", "accum_passes": 3}, True, P16, 256)
        assert m1["mesh_lat"] == 71 and m3["mesh_lat"] == 71 + 2 * min(a["host_gaps"]) == next(x["mesh"] for x in a["sets"] if x["mesh"] is not None) == 109


@perf_selftest
def test_bypass_stage():
    with perf_fmt("int8"):  # the ReLU fixture's G intervals (hold 115-118, add 126-129) are the model's bypass stage, plain or residual
        a = pa.analyse(pa.events(open(os.path.join(PERF_FIX, "int8_residual_relu_N8_T2_int8.trace")).read()), 18, 1, [0x56, 0x58] * 9)
        plain, res = pa.model({"act": "relu"}, True, PERF_P8, 64), pa.model({"act": "relu", "residual": True}, True, PERF_P8, 64)
        assert plain["act"] == res["act"] == res["hold"] == a["hold"][0] == a["sets"][1]["act"] == 3, (plain, res)
    with perf_fmt("bf16"):  # matmul_relu_nopool at N16 T4 on today's TB (rs2_perf2_bf16_n16_t4): G from the first beat to idle, 9 = per_lane 8 + 1
        g_up, g_dn = pa.leaves(pa.events(open(os.path.join(PERF_FIX, "matmul_relu_nopool_N16_T4_bf16_geff.trace")).read()), "G", 0)
        assert pa.model({"act": "relu"}, True, {**PERF_P8, "N": 16, "TILE_SIZE": 4, "HOST_WORDS": 16, "SRAM_DEPTH": 256}, 256)["act"] == \
            statistics.median(d - u for u, d in zip(g_up, g_dn)) == 9


@contextlib.contextmanager
def perf_fmt(f):
    """The perf section's build format for the block, restored afterwards so no test depends on the order they run in."""
    old = pa.FMT, pa.MUL_LAT, pa.ADD_LAT
    pa.FMT, (pa.MUL_LAT, pa.ADD_LAT) = f, pa.UNIT_LAT[f]
    try:
        yield
    finally:
        pa.FMT, pa.MUL_LAT, pa.ADD_LAT = old


@perf_selftest
def test_rtl_lat():
    assert pa.UNIT_LAT == {"fp32": (8, 5), "bf16": (3, 5), "int8": (1, 1)}  # sienna_fmt_pkg's MUL_LATENCY, ADD_LATENCY per define, through fmt_pkg.py
    assert (pa.FX_LAT, pa.REQ_LAT, pa.QADD_LAT, pa.GROUP_K, pa.TAIL_CTX) == (2, 4, 6, 16, 8)  # fixedPointMultiplyAdd, REQUANT_LATENCY, QUANT_ADD_LATENCY, gpnae's K, TAIL_CONTEXTS


@perf_selftest
def test_lane_stage_int8():
    with perf_fmt("int8"):  # no tail path, so the stage is data-free; rs3_b2_perf_int8_n8/n16 measured each 2 below, as Batch A's model and sim differ
        stage = lambda act, n: pa.lane_stage(act, {"SRAM_DEPTH": n * n, "NUM_LANES": 32})
        assert stage("selu", 16) == 101 and stage("tanh", 16) == 109 and stage("sigmoid", 16) == 109
        assert stage("selu", 8) == 56 and stage("tanh", 8) == 55 and stage("sigmoid", 8) == 55


@perf_selftest
def test_lane_group_float():
    with perf_fmt("fp32"):  # one group from its first G_CAP at s; the stage model built on it is 2 above rs3_b2_perf_fp32_n8 on every config, as Batch A's
        assert pa.lane_cycles([0.0] * 8, "selu", 12) == 88 and pa.lane_cycles([0.0] * 2, "tanh", 6) == 64
        assert pa.lane_cycles([-3.0] * 8, "sigmoid", 12) == 184 and pa.lane_cycles([0.0] * 32, "tanh", 36) == 253
    with perf_fmt("bf16"):  # the same against rs3_b2_perf_bf16_n8; -3.0 runs on the fp32 tail
        assert pa.lane_cycles([0.0] * 8, "selu", 12) == 71 and pa.lane_cycles([0.0] * 32, "tanh", 36) == 233
        assert pa.lane_cycles([-3.0] * 8, "sigmoid", 12) == 204 and pa.lane_cycles([-3.0] + [0.5] * 7, "sigmoid", 12) == 204


@perf_selftest
def test_lane_kinds():
    K = gpnae_model
    with perf_fmt("fp32"):  # tier 0: below -6 the negative sigmoid is 2^-n P(f) on the MAC (no tail in fp32), above 0 P itself
        k, r = pa.lane_kinds([-8.0, -3.0, 0.5, 5.0], "sigmoid")
        assert k == [K.K_EXP, K.K_REFL, K.K_PASS, K.K_PASS] and r == [5, 8, 8, 8], (k, r)
    with perf_fmt("bf16"):  # tier 0: below -1.5 the negative sigmoid goes to the fp32 tail
        k, r = pa.lane_kinds([-3.0, -1.0, 0.5, 5.0], "sigmoid")
        assert k == [K.K_TAIL, K.K_PASS, K.K_PASS, K.K_PASS] and r == [1, 4, 5, 5], (k, r)


@perf_selftest
def test_tail_ops_bf16():
    with perf_fmt("bf16"):  # fp32 units, multiply 8 + 2 and add 5 + 2 per step: k, Horner (deg + 1 rounds), D and y0, Newton (mul, add, mul), E y
        assert [pa.tail_ops(t) for t in range(3)] == [10 + 5 * 17 + 14 + 2 * 27 + 10, 10 + 3 * 17 + 14 + 27 + 10, 10 + 3 * 17 + 14 + 10] == [173, 112, 85]


# ── tool: the int8 layer references (was test_tflite_ref.py) ──

@selftest
def test_layer_ref():
    """The int8 layer references on hand values: fold_input_zp's algebra equals the plain sums, ipu.activation_clamp's bounds, and
    requantize (encode_scale words) rounding ties half up."""
    fails = checks = 0

    def check(what, got, want):
        nonlocal fails, checks
        checks += 1
        g, w = np.asarray(got), np.asarray(want)
        if g.shape != w.shape or np.any(g != w):
            fails += 1
            print(f"[FAIL] {what}: got {g.tolist()}, want {w.tolist()}")

    check("relu6 clamp", ipu.activation_clamp("relu6", 0.05, -128), (-128, -8))  # 6 / 0.05 = 120 levels
    check("relu clamp", ipu.activation_clamp("relu", 0.1, 3), (3, 127))
    check("relu6 above int8", ipu.activation_clamp("relu6", 0.02, -10), (-10, 127))  # 300 levels: capped
    check("none clamp", ipu.activation_clamp("none", 0.1, 3), (-128, 127))
    check("layer_scales per channel and per tensor", mr.layer_scales([0.25, 0.5], 0.5, 2.0, 2).tolist() + mr.layer_scales([0.25], 0.5, 2.0, 2).tolist(),
          [0.0625, 0.125, 0.0625, 0.0625])
    # FC: acc = 3 * (10 + 2) - 4 * (-20 + 2) + b = 108 + b; scale 0.5 * 0.25 / 1.0 = 2^-3; out zp 3; ties round half up.
    x, w = np.array([[10, -20]]), np.array([[3, -4]])
    m, e = mr.encode_scales(mr.layer_scales([0.25], 0.5, 1.0, 1))
    check("encode_scales(2^-3)", (m.tolist(), e.tolist()), ([1 << 23], [26]))
    for b, want in ((100, 29), (104, 30), (-176, -5)):  # 26; 26.5 up; -8.5 up
        for folded in (False, True):
            acc = mr.fc_acc(x, w, np.array([b]), -2, folded=folded)
            check(f"fc b={b} folded={folded}: sum", acc, [[108 + b]])
            check(f"fc b={b} folded={folded}: requantize", mr.requantize(acc, dict(mult=m, shift=e, zp=3, amin=-128, amax=127)), [[want]])
    # Conv 3x3 SAME on 2x2: in-image taps of (x - 1) give [[49, 43], [31, 25]]; scale 0.5 makes each a tie, half up; out zp -3.
    xc = np.array([1, 2, 3, 4]).reshape(1, 2, 2, 1)
    wc = np.arange(1, 10).reshape(1, 3, 3, 1)
    for folded in (False, True):
        acc = mr.conv2d_acc(xc, wc, np.array([0]), 1, folded=folded)
        check(f"conv folded={folded}: sums", acc.ravel(), [49, 43, 31, 25])
        check(f"conv folded={folded}: requantize", mr.requantize(acc.reshape(-1, 1), dict(mult=[1 << 23], shift=[24], zp=-3, amin=-128, amax=127)).ravel(),
              [22, 19, 13, 10])
    check("fold_input_zp", mr.fold_input_zp(np.array([0]), np.ones((1, 3, 3, 1)), 1), [-9])
    check("im2col_same corner", mr.im2col_same(xc, 3, 3, 1)[0], [1, 1, 1, 1, 1, 2, 1, 3, 4])  # pad value 1 outside
    job = {"terms": [(np.array([[10.0, -20.0]]), np.array([[3.0], [-4.0]]))], "bias": np.array([102]),  # 104 - in_zp sum(w) = 104 - (-2)(-1)
           "req": dict(mult=m, shift=e, zp=3, amin=-128, amax=127, ratio=[0.125])}
    check("layer_truth passes the half-up tie and the other side of it", [mr.layer_truth(job, [[v]])[0].ok for v in (30, 29, 32)], [True, True, False])
    print(f"test_layer_ref: {checks} checks, {fails} failures")
    print(f"RESULT: {'PASSED' if fails == 0 else 'FAILED'}")
    assert fails == 0, f"{fails} of {checks} checks failed"


def _float_layer(kind, x_real, w_real, b_real, stride, pads):
    """An independent float64 reference of a conv (NHWC x, OHWI w, explicit loops, zero pads (top, bottom, left, right) in real values) or an fc layer."""
    if kind == "fc":
        return x_real @ w_real.T + b_real[None, :]
    n, h, wd, _ = x_real.shape
    co, kh, kw, _ = w_real.shape
    ph, pb, pw, pr = pads
    oh, ow = (h + ph + pb - kh) // stride[0] + 1, (wd + pw + pr - kw) // stride[1] + 1
    y = np.zeros((n, oh, ow, co))
    for b in range(n):
        for i in range(oh):
            for j in range(ow):
                for dy in range(kh):
                    for dx in range(kw):
                        yy, xx = i * stride[0] + dy - ph, j * stride[1] + dx - pw
                        if 0 <= yy < h and 0 <= xx < wd:
                            y[b, i, j] += w_real[:, dy, dx, :] @ x_real[b, yy, xx, :]
    return y + b_real


def _record_check(rec, x_codes, x_real, w_real, b_real, s_out, out_zp, lo_real=None, hi_real=None, pads=(0, 0, 0, 0)):
    """The record's job through the bit-exact host model, against the float reference of the layer's real numbers: every output
    within max(1% |t|, one step) of t, t clamped to the output range and the activation's real bounds; pads are the framework's, from
    the test, not the record; returns the over-count (an output-shape mismatch counts every output)."""
    job, shape = mr.job_of_record(rec, x_codes)
    (X, W), req = job["terms"][0], job["req"]
    y = mr.int8_layer_exact(X.astype(np.int64), W.astype(np.int64), job["bias"], req, "linear").reshape(shape)
    t = _float_layer(rec.kind, x_real, w_real, b_real, rec.stride, pads)
    if t.shape != y.shape:
        return y.size, y.size
    t = np.clip(t, -np.inf if lo_real is None else lo_real, np.inf if hi_real is None else hi_real)
    tc = np.clip(t / s_out + out_zp, -128, 127)
    over = int(np.sum(np.abs(y - tc) > np.maximum(0.01 * np.abs(tc - out_zp), 1.0)))
    return over, y.size


@selftest
def test_layer_file_add_line():
    """An int8 residual layer file carries A after Q (zp2 .. amax, add_words_t's order); write_layer refuses one without ADD words or with zp1
    off the Q line's zp; the A line's words come from an AddRecord named after the layer, so a refused ADD names it."""
    job, _, _, _, _ = residual_case("deep_uncached", 8, np.random.RandomState(1))
    cfg, a_rows, w_rows, _ = mr.format_layer(job, 8)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.layer")
        mr.write_layer(p, cfg, a_rows, w_rows, "int8", job["req"], None, mr.layer_epilogue(job, 8), add=job["add"])
        lines = open(p).read().splitlines()
        assert lines[1].startswith("Q ") and lines[2].split()[0] == "A" and len(lines[2].split()) == 8, lines[:3]
        assert [int(v) for v in lines[2].split()[1:]] == [job["add"][k] for k in mr.ADD_FIELDS[1:]]
        for add in (None, dict(job["add"], zp1=job["req"]["zp"] + 1)):
            try:
                mr.write_layer(p, cfg, a_rows, w_rows, "int8", job["req"], None, mr.layer_epilogue(job, 8), add=add)
                raise AssertionError("write_layer accepted a residual layer with bad ADD words")
            except ValueError:
                pass
        plain = dict(job, terms=job["terms"][:1])  # no residual term: no A line, and no ADD words needed
        cfg0, a0, w0, _ = mr.format_layer(plain, 8)
        mr.write_layer(p, cfg0, a0, w0, "int8", job["req"], None, mr.layer_epilogue(plain, 8))
        assert not any(l.startswith("A ") for l in open(p).read().splitlines()[:3])
    try:
        mr.add_record_words(mr.AddRecord(1.0, 0, 1e9, 0, 1.0, 0, -128, 127, "res_blk3"))
        raise AssertionError("an ADD outside the adder's domain was accepted")
    except ValueError as e:
        assert "res_blk3" in str(e), e


@selftest
def test_stage_dump_gates():
    """layer_stage_gates on stage dumps built from the host models (read_stage_dump's format): a clean residual layer passes every stage; a
    hold_mem beat off fails hold_mem, an ADD output off by 3 fails add, zp_1 taken as 0 fails add, a lane output off fails the activation, a
    missing element and a duplicate line are refused; a plain tanh layer's requantize off by 2 codes fails requant."""
    N = 8
    job, want, wrong, q1, R_q = residual_case("tiles", N, np.random.RandomState(3))
    M, C = job["shape"]
    rt, ct, np_ = -(-M // N), -(-C // N), -(-RES_CASES["tiles"](N)[1] // N) + 1
    y = mr.qadd_codes(q1, R_q, job["add"])

    pad = lambda g: np.pad(np.asarray(g, np.int64), ((0, rt * N - M), (0, ct * N - C)))  # the TB dumps whole tiles

    def dump(path, R, Q1, Q2, Yq, X):  # the TB's lines, set k = (c rt + r) np_ + p, element e row-major in the tile
        R, Q1, Q2, Yq, X = (pad(v) for v in (R, Q1, Q2, Yq, X))
        with open(path, "w") as f:
            f.write("H pass 0\nX 0 0 9\nH pass 1\n")  # a cut pass's lines come before the last H
            for c in range(ct):
                for r in range(rt):
                    t = (c * rt + r) * np_
                    for e in range(N * N):
                        i, j = r * N + e // N, c * N + e % N
                        f.write(f"R {t + np_ - 2} {e} {R[i, j]}\nQ {t + np_ - 1} {e} {Q1[i, j]} {Q2[i, j]} {Yq[i, j]}\nX {t + np_ - 1} {e} {X[i, j]}\n")

    def gates(R=q1, Q1=q1, Yq=y, X=y, out=want, edit=None):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s")
            dump(p, R, Q1, R_q, Yq, X)
            if edit:
                ls = open(p).readlines()
                open(p, "w").writelines(edit(ls))
            st = mr.read_stage_dump(p, N, rt, ct, np_)
        return {s: r.ok for s, r in mr.layer_stage_gates(job, out, st, N)}

    ok = gates()
    assert all(ok.values()) and set(ok) == {"requant", "requant identity", "hold_mem", "add", "activation relu"}, ok
    off = q1.copy()
    off[1, 2] += 1 if off[1, 2] < 127 else -1
    assert not gates(Q1=off)["hold_mem"]
    y3 = y.copy()
    y3[2, 1] += 3 if y3[2, 1] < 120 else -3
    assert not gates(Yq=y3, X=y3)["add"]
    if job["add"]["zp1"] != 0:
        assert not gates(Yq=wrong[0], X=wrong[0])["add"]
    o = want.copy()
    o[o > -100] -= 3
    assert not gates(out=o)["activation relu"]
    for edit, msg in ((lambda ls: ls[:-1], "miss"), (lambda ls: ls + ls[-1:], "two")):
        try:
            gates(edit=edit)
            raise AssertionError(f"read_stage_dump accepted a dump with {msg}")
        except RuntimeError as e:
            assert msg in str(e), e
    # a plain tanh layer: requantize and the lane
    rng = np.random.RandomState(4)
    A_q, W_q = rng.randint(-128, 128, (N, 2 * N)), rng.randint(-127, 128, (2 * N, N))
    hw = rng.randint(-500, 500, N)
    acc = wrap32(imatmul(A_q, W_q) + hw[None, :])
    rq = requant_params(acc, 0.02, np.full(N, 0.01), "tanh")
    X = requantize(acc, rq)
    plain = {"terms": [(A_q.astype(np.float32), W_q.astype(np.float32))], "bias": hw, "act": "tanh", "shape": (N, N), "req": rq}
    Yt = activate_int8(X, "tanh", rq)
    run = lambda Xd: {s: r.ok for s, r in mr.layer_stage_gates(plain, Yt, {"np": 2, "X": {1: Xd}}, N)}
    assert run(X) == {"requant": True, "activation tanh": True}, run(X)
    X2 = X.copy()
    X2[0, 0] += 2 if X2[0, 0] < 120 else -2
    assert not run(X2)["requant"]
    for st, kind in (({"np": 2}, "X"), ({"np": 2, "X": {0: X}}, "X")):  # no X lines for the output pass: a RuntimeError, not a KeyError
        try:
            mr.layer_stage_gates(plain, Yt, st, N)
            raise AssertionError("a dump without the output pass's X lines was gated")
        except RuntimeError as e:
            assert f"no {kind} lines for pass 1" in str(e), e
    eye = dict(plain, terms=[(A_q[:, :N].astype(np.float32), np.eye(N, dtype=np.float32))], bias=np.zeros(N, np.int64))
    Xe = requantize(wrap32(A_q[:, :N]), rq)  # an identity dense term in a layer with no ADD is the product, not a residual
    assert [s for s, _ in mr.layer_stage_gates(eye, activate_int8(Xe, "tanh", rq), {"np": 1, "X": {0: Xe}}, N)] == ["requant", "activation tanh"]
    try:
        mr.lane_quant("tanh", None, 0, rq)
        raise AssertionError("a tanh lane was judged without its input scale")
    except ValueError:
        pass
    assert mr.true_clamp("relu6", 0.05, -10) == (-10.0, 110.0) and mr.true_clamp("none", 0.05, 3) == (-128.0, 127.0)


@selftest
def test_residual_mid_reset():
    """residual_mid_reset on a stub engine: passes on in-window resets matching the plain run; fails on each way one can go wrong."""
    Y0 = np.arange(12).reshape(3, 4)
    ok = truth.Result(True, "int8", "requantize", 1, 0.0, 0, 0, 0, 0, 0, 0, 0, [])

    class Stub:
        def __init__(self, ctl="check", at=(101, 160), inside=True, Y=Y0, first=100, free=150):
            self.N, self.first_hold, self.first_free, self.last_log, self.last_stages = 8, first, free, None, {}
            self.ctl, self.at, self.inside, self.Y = ctl, at, inside, Y

        def run_job(self, job, tag, plusargs=()):
            self.first_hold = self.first_free = -5  # a later run overwrites them
            if "+mid_reset=1" in plusargs:
                self.last_log = {"check": "[FAIL] +hold_check: the mid-stream reset at 3 cycles is outside the hold window",
                                 "other": "[FATAL] nothing moved", "pass": ""}[self.ctl]
                if self.ctl != "pass":
                    raise RuntimeError(f"{tag}: an assertion or a testbench check failed in the layer simulation")
                return Y0, 1, 1
            k = int("waiting" in tag)
            assert plusargs == (f"+mid_reset={(101, 150)[k]}", "+hold_wait", "+hold_check"), plusargs
            self.last_log = f"[RES] reset {self.at[k]} cycles into the layer, {'inside' if self.inside else 'outside'} the hold window, 4 of 4 hold beats unread"
            return self.Y, 1, 1

    real = mr.layer_stage_gates
    try:
        for stub, gate_ok, want in ((Stub(), True, None), (Stub(at=(149, 151)), True, None), (Stub(first=None), True, "no first hold_done"),
                                    (Stub(free=None), True, "window close (None)"), (Stub(ctl="pass"), True, "control: a reset outside"),
                                    (Stub(ctl="other"), True, "not on +hold_check"), (Stub(at=(100, 160)), True, "full (landed at 100"),
                                    (Stub(at=(150, 160)), True, "full (landed at 150"), (Stub(at=(101, 150)), True, "waiting (landed at 150"),
                                    (Stub(inside=False), True, "landed at None"), (Stub(Y=Y0 + 1), True, "outputs differ"),
                                    (Stub(), False, "mid-reset full (add)")):
            mr.layer_stage_gates = lambda *a, g=gate_ok: [("add", ok._replace(ok=g))]
            with contextlib.redirect_stdout(io.StringIO()):
                bad = residual_mid_reset(stub, {}, Y0, Y0)
            assert (not bad) if want is None else any(want in b for b in bad), (want, bad)
    finally:
        mr.layer_stage_gates = real


@selftest
def test_job_of_add_record():
    """job_of_add_record on ONNX and PyTorch records: the job's words pass the requantize and ADD true-function gates; misuse is refused."""
    conv, add, x, R = onnx_res_block(4)
    job, shape = mr.job_of_add_record(conv, add, x, R)
    assert shape == (2, 8, 8, 16) and job["act"] == "relu" and job["clamps"] == ((None, None), (0.0, None)), (shape, job["act"], job["clamps"])
    assert job["req"]["s_out"] == conv.s_out and job["add"]["zp1"] == job["req"]["zp"] == conv.zp_out
    rng = np.random.default_rng(6)
    tconv = mr.record_of_torch("conv", 0.02, 120, rng.integers(-127, 128, (16, 16, 3, 3)), rng.uniform(0.0005, 0.002, 16), np.zeros(16),
                               rng.uniform(-0.5, 0.5, 16), 0.15, 130, padding=(1, 1), name="t_conv")
    tadd = mr.add_record_of_torch(0.15, 130, 0.02, 120, 0.2, 128, relu=True, name="t_add")
    for c, a, xx, RR in ((conv, add, x, R), (tconv, tadd, x, R)):
        job, _ = mr.job_of_add_record(c, a, xx, RR)
        (X, W), (Rf, _) = job["terms"]
        acc = wrap32(imatmul(X.astype(np.int64), W.astype(np.int64)) + np.asarray(job["bias"], np.int64)[None, :])
        q1 = requantize(acc, job["req"])
        y = mr.qadd_codes(q1, Rf.astype(np.int64), job["add"])
        lo, hi = mr.true_clamp(job["clamps"][0], c.s_out, c.zp_out)
        a_lo, a_hi = mr.true_clamp(job["clamps"][1], a.s_out, a.z_out)
        r1, r2 = job["add"]["ratios"]
        g1 = truth.gate_requantize(acc, np.broadcast_to(c.scale[None, :], acc.shape), c.zp_out, lo, hi, q1)
        g2 = truth.gate_quantized_add(q1, Rf.astype(np.int64), r1, a.z1, r2, a.z2, 1.0, a.z_out, a_lo, a_hi, y)
        assert g1.ok and g2.ok and np.mean(y > a_lo) > 0.25, (c.name, g1.over, g2.over)
    for bad, why in ((lambda: mr.job_of_add_record(conv, mr.AddRecord(add.s2, add.z2, add.s1, add.z1, add.s_out, add.z_out, add.lo, add.hi, "sw"),
                                                   x, R), "ADD layer sw: input 1"),
                     (lambda: mr.job_of_add_record(dataclasses.replace(conv, s_out=None), add, x, R), "needs its output scale")):
        try:
            bad()
        except ValueError as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"no refusal: {why}")


@selftest
def test_build_retry():
    """RtlLayer.build rebuilds once, printing [RETRY], when make prints a Verilator Internal Error; a second internal error or any other failure
    raises; a clean build runs make once."""
    real_run, real_exists, real_pkg = mr.subprocess.run, mr.os.path.exists, mr.write_build_pkg
    R = lambda rc, out: subprocess.CompletedProcess([], rc, out, "")
    ierr = R(2, "%Error: Internal Error: ../V3TSP.cpp:353: No unmarked edges found in tour\nmake: *** [Makefile:431: verilator] Error 1\n")
    other = R(2, "%Error: TB_model_run.sv:12: syntax error\nmake: *** [Makefile:431: verilator] Error 1\n")
    ok = R(0, "built\n")
    try:
        mr.write_build_pkg = lambda *a: None
        mr.os.path.exists = lambda p: True
        for seq, want_calls, want_ok in (([ok], 1, True), ([ierr, ok], 2, True), ([ierr, ierr], 2, False), ([other, ok], 1, False),
                                         ([ierr, other], 2, False)):
            calls = []
            mr.subprocess.run = lambda *a, **k: (calls.append(1), seq[len(calls) - 1])[1]
            sim = mr.RtlLayer(8, 32, "/nonexistent", "int8", 2)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    sim.build()
                    good = True
                except RuntimeError:
                    good = False
            assert (len(calls), good) == (want_calls, want_ok), (seq, len(calls), good)
            assert ("[RETRY] Verilator internal error" in buf.getvalue()) == (seq[0] is ierr) and sim.retries == int(seq[0] is ierr), buf.getvalue()
    finally:
        mr.subprocess.run, mr.os.path.exists, mr.write_build_pkg = real_run, real_exists, real_pkg


@selftest
def test_pack_float_reference():
    """float_layer_over passes a layer whose int32 bias folds the input zero point (fold_bias) and fails one that drops the fold, a bug the host,
    the golden and the per-stage requantize gate would share; ReLU's reference clamps at 0."""
    rng = np.random.RandomState(9)
    A, B, bias = rng.uniform(-0.2, 1, (16, 8)), rng.uniform(-1, 1, (8, 8)), rng.uniform(-0.5, 0.5, 8)
    A_q, s_a, z_a = mr.quant_act(A)
    B_q, s_w = mr.quant_weights(B)
    assert z_a != 0, z_a
    for act in ("linear", "relu"):
        for fold_zp, want in ((z_a, 0), (0, None)):
            hw = mr.fold_bias(bias, s_a, s_w, fold_zp, B_q)
            acc = mr.wrap32(mr.imatmul(A_q, B_q) + hw[None, :])
            rq = mr.requant_params(mr.wrap32(mr.imatmul(A_q, B_q) + mr.fold_bias(bias, s_a, s_w, z_a, B_q)[None, :]), s_a, s_w, act)
            Y = mr.activate_int8(mr.requantize(acc, rq), act, rq)
            over, n = float_layer_over(A_q, z_a, s_a, B_q, s_w, bias, rq["s_out"], rq["zp"], act == "relu", Y)
            assert n == Y.size and (over == 0 if want == 0 else over > 0), (act, fold_zp, over, n)


@selftest
def test_importer_records():
    """The neutral layer and ADD records from hand-built TFLite, ONNX (QLinearConv, QLinearMatMul + Clip, QLinearAdd, uint8) and
    PyTorch (Conv2d + ReLU with a float bias, Linear, add_relu, quint8) tensors, each run through job_of_record or add_record_words and
    the bit-exact host models and judged against an independent float reference; and the refusals (a weight zero point, three ONNX pads, string PyTorch padding,
    an ADD outside the adder's domain naming its layer, groups, dilation, ONNX auto_pad, a TFLite conv without its stride); stride-2
    convolutions with padding on even inputs, where TFLite SAME and ONNX / PyTorch pads differ, check that the record keeps each one's."""
    rng = np.random.default_rng(11)
    # ONNX QLinearConv, uint8 activations, per-channel int8 weights (OIHW), int32 bias, SAME pads, a following Relu
    x_u8, xs, xz = rng.integers(0, 256, (2, 5, 5, 3)), 0.02, 131
    w = rng.integers(-127, 128, (4, 3, 3, 3)); ws = rng.uniform(0.002, 0.01, 4)
    b = rng.integers(-300, 300, 4); ys, yz = 0.05, 140
    rec = mr.record_of_onnx("QLinearConv", xs, xz, w, ws, np.zeros(4), ys, yz, b, pads=[1, 1, 1, 1], relu=True, x_unsigned=True,
                            y_unsigned=True, name="c1")
    assert (rec.kind, rec.zp_in, rec.zp_out, rec.lo, rec.hi, rec.pads) == ("conv", 3, 12, 12, 127, (1, 1, 1, 1)), rec
    w_real = np.transpose(w, (0, 2, 3, 1)) * ws[:, None, None, None]
    over, n = _record_check(rec, x_u8 - 128, xs * (x_u8 - xz), w_real, b * xs * ws, ys, yz - 128, lo_real=0.0, pads=(1, 1, 1, 1))
    assert over == 0 and n == 200, (over, n)
    # ONNX QLinearConv 3x3, stride 2, pads 1 on a 6x6 input: ONNX pads (1, 1), where TFLite SAME would pad (0, 1)
    x6 = rng.integers(0, 256, (2, 6, 6, 3))
    rec = mr.record_of_onnx("QLinearConv", xs, xz, w, ws, np.zeros(4), ys, yz, b, strides=(2, 2), pads=[1, 1, 1, 1], x_unsigned=True,
                            y_unsigned=True, name="c2")
    over, n = _record_check(rec, x6 - 128, xs * (x6 - xz), w_real, b * xs * ws, ys, yz - 128, pads=(1, 1, 1, 1))
    assert over == 0 and n == 72, ("onnx stride 2 pads 1", over, n)
    # ONNX asymmetric pads [top 0, left 0, bottom 1, right 1] (as tf2onnx emits for SAME at stride 2): checks the reorder to (t, b, l, r)
    x5 = rng.integers(0, 256, (1, 5, 5, 3))
    rec = mr.record_of_onnx("QLinearConv", xs, xz, w, ws, np.zeros(4), ys, yz, b, strides=(2, 2), pads=[0, 0, 1, 1], x_unsigned=True,
                            y_unsigned=True, name="c5")
    over, n = _record_check(rec, x5 - 128, xs * (x5 - xz), w_real, b * xs * ws, ys, yz - 128, pads=(0, 1, 0, 1))
    assert over == 0 and n == 16, ("onnx asymmetric pads", over, n)
    assert rec.pads == (0, 1, 0, 1), rec.pads
    # ONNX QLinearMatMul (w (K, N)), int8, Clip(-1, 2)
    x, w2, ws2 = rng.integers(-128, 128, (6, 8)), rng.integers(-127, 128, (8, 5)), np.full(5, 0.004)
    rec = mr.record_of_onnx("QLinearMatMul", 0.03, -7, w2, ws2, np.zeros(1), 0.02, 9, clip=(-1.0, 2.0), name="m1")
    assert (rec.kind, rec.lo, rec.hi) == ("fc", -41, 109), (rec.lo, rec.hi)
    over, n = _record_check(rec, x, 0.03 * (x + 7), w2.T * ws2[:, None], np.zeros(5), 0.02, 9, lo_real=-1.0, hi_real=2.0)
    assert over == 0 and n == 30, (over, n)
    # PyTorch quantized Conv2d + ReLU (ConvReLU2d), quint8 input and output, symmetric qint8 weight, float bias, stride 2, no padding
    xq, w3, ws3 = rng.integers(0, 256, (1, 7, 7, 2)), rng.integers(-127, 128, (3, 2, 3, 3)), rng.uniform(0.003, 0.006, 3)
    bias = rng.uniform(-0.5, 0.5, 3)
    rec = mr.record_of_torch("conv", 0.015, 120, w3, ws3, np.zeros(3), bias, 0.04, 100, relu=True, stride=(2, 2), padding=(0, 0),
                             name="t1")
    assert (rec.zp_in, rec.zp_out, rec.lo, rec.pads) == (-8, -28, -28, (0, 0, 0, 0)), rec
    over, n = _record_check(rec, xq - 128, 0.015 * (xq - 120), np.transpose(w3, (0, 2, 3, 1)) * ws3[:, None, None, None],
                            rec.b_q * 0.015 * ws3, 0.04, -28, lo_real=0.0)  # the bias as PyTorch quantizes it, round(b / (s_in s_w))
    assert over == 0 and n == 27, (over, n)
    assert np.all(np.abs(rec.b_q * 0.015 * ws3 - bias) <= 0.5 * 0.015 * ws3), "the bias is quantized to the nearest step"
    # PyTorch Conv2d 3x3, stride 2, padding 1 on an 8x8 input (the ResNet downsampling conv): pads (1, 1) on each axis
    x8 = rng.integers(0, 256, (1, 8, 8, 2))
    rec = mr.record_of_torch("conv", 0.015, 120, w3, ws3, np.zeros(3), bias, 0.04, 100, stride=(2, 2), padding=(1, 1), name="t3")
    over, n = _record_check(rec, x8 - 128, 0.015 * (x8 - 120), np.transpose(w3, (0, 2, 3, 1)) * ws3[:, None, None, None],
                            rec.b_q * 0.015 * ws3, 0.04, -28, pads=(1, 1, 1, 1))
    assert over == 0 and n == 48, ("torch stride 2 padding 1", over, n)
    # PyTorch Linear, quint8
    rec = mr.record_of_torch("linear", 0.01, 128, rng.integers(-127, 128, (4, 6)), 0.005, 0, None, 0.02, 128, name="t2")
    assert (rec.kind, rec.zp_in, rec.zp_out, rec.lo, rec.hi) == ("fc", 0, 0, -128, 127)
    # TFLite through its record: load_layer's dict shape, per-tensor weight scale broadcast, the clamp as given
    t = {"kind": "FULLY_CONNECTED", "act_range": (-5, 127), "input": {"scale": np.float32([0.1]), "zp": np.array([-5])},
         "filter": {"data": rng.integers(-127, 128, (3, 4)).astype(np.int8), "scale": np.float32([0.25]), "zp": np.array([0])},
         "bias": None, "output": {"scale": np.float32([0.5]), "zp": np.array([-5])}}
    rec = mr.record_of_tflite(t, "f1")
    assert np.allclose(rec.scale, 0.05, rtol=1e-7) and (rec.zp_in, rec.lo, rec.hi, rec.b_q.tolist()) == (-5, -5, 127, [0, 0, 0])
    # TFLite CONV_2D SAME, stride 2 on a 6x6 input: TFLite's rule pads (0, 1), the odd pad at the end
    xt, wt = rng.integers(-128, 128, (1, 6, 6, 2)), rng.integers(-127, 128, (2, 3, 3, 2))
    tcv = {"kind": "CONV_2D", "act_range": (-128, 127), "stride": (2, 2), "same": True,
          "input": {"shape": (1, 6, 6, 2), "scale": np.float32([0.02]), "zp": np.array([3])},
          "filter": {"data": wt.astype(np.int8), "scale": np.float32([0.01, 0.005]), "zp": np.array([0, 0])},
          "bias": None, "output": {"scale": np.float32([0.25]), "zp": np.array([-2])}}
    rec = mr.record_of_tflite(tcv, "c3")
    assert rec.pads == (0, 1, 0, 1), rec.pads
    over, n = _record_check(rec, xt, np.float32(0.02) * (xt - 3.0), wt * np.float32([0.01, 0.005])[:, None, None, None].astype(np.float64),
                            np.zeros(2), np.float32(0.25), -2, pads=(0, 1, 0, 1))
    assert over == 0 and n == 18, ("tflite stride 2 SAME", over, n)
    # ADD records: ONNX QLinearAdd (uint8, Relu) and PyTorch add_relu, through add_record_words and ipu.qadd against the float sum
    for a in (mr.add_record_of_onnx(0.05, 140, 0.03, 120, 0.06, 128, relu=True, unsigned=True, name="a1"),
              mr.add_record_of_torch(0.05, 140, 0.03, 120, 0.06, 128, relu=True, name="a2")):
        wds = mr.add_record_words(a)
        q1, q2 = (v.ravel() for v in np.meshgrid(np.arange(-128, 128), np.arange(-128, 128)))
        y = mr.qadd_codes(q1, q2, wds)
        tc = np.clip(np.maximum(a.s1 * (q1 - a.z1) + a.s2 * (q2 - a.z2), 0.0) / a.s_out + a.z_out, -128, 127)
        assert (a.z1, a.z2, a.z_out, a.lo) == (12, -8, 0, 0) and not np.any(np.abs(y - tc) > np.maximum(0.01 * np.abs(tc - a.z_out), 1.0))
    # TFLite ADD through its record (load_int8_graph's op): the host picks input 1, the other is input 2; a tensor not an input is refused
    tadd = {"kind": "ADD", "inputs": [5, 9], "fused": "relu", "act_range": (-3, 127), "a": {"index": 5, "scale": np.float32([0.05]), "zp": np.array([12])},
            "b": {"index": 9, "scale": np.float32([0.03]), "zp": np.array([-8])}, "output": {"index": 11, "scale": np.float32([0.06]), "zp": np.array([-3])}}
    r0, r1 = mr.add_record_of_tflite(tadd, name="t5"), mr.add_record_of_tflite(tadd, 9, "t6")
    assert (r0.z1, r0.z2, r0.z_out, r0.lo, r0.hi) == (12, -8, -3, -3, 127) and (r1.z1, r1.z2) == (-8, 12), (r0, r1)
    assert (r0.s1, r0.s2) == (r1.s2, r1.s1) == (float(np.float32(0.05)), float(np.float32(0.03)))
    for a in (r0, r1):
        q1, q2 = (v.ravel() for v in np.meshgrid(np.arange(-128, 128), np.arange(-128, 128)))
        y = mr.qadd_codes(q1, q2, mr.add_record_words(a))
        tc = np.clip(np.maximum(a.s1 * (q1 - a.z1) + a.s2 * (q2 - a.z2), 0.0) / a.s_out + a.z_out, -128, 127)
        assert not np.any(np.abs(y - tc) > np.maximum(0.01 * np.abs(tc - a.z_out), 1.0)), a.name
    assert r0.clamp == (0.0, None) and mr.record_of_tflite(dict(tcv, fused="relu6"), "c6").clamp == (0.0, 6.0), (r0.clamp,)
    # the ADD importer's checks: a plain int8 ADD passes; broadcast, constant and per-axis inputs are refused below
    ten = lambda **k: dict({"type": 9, "shape": (1, 4, 4, 8), "data": None, "scale": np.float32([0.1]), "zp": np.array([0])}, **k)
    plain = {"a": ten(), "b": ten(), "output": ten()}
    mr.check_int8_add(plain, "a0")
    # refusals
    for bad, why in ((lambda: mr.record_of_onnx("QLinearMatMul", 0.1, 0, np.ones((2, 2)), 0.1, 3, 0.1, 0), "weight zero point"),
                     (lambda: mr.record_of_onnx("QLinearConv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, 0.1, 0, pads=[0, 1, 0]), "pads"),
                     (lambda: mr.record_of_torch("conv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, None, 0.1, 0, padding="same"), "padding"),
                     (lambda: mr.record_of_onnx("QLinearConv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, 0.1, 0, group=2), "group"),
                     (lambda: mr.record_of_onnx("QLinearConv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, 0.1, 0, dilations=(2, 2)), "dilation"),
                     (lambda: mr.record_of_onnx("QLinearConv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, 0.1, 0, auto_pad="SAME_LOWER"), "auto_pad"),
                     (lambda: mr.record_of_torch("conv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, None, 0.1, 0, groups=2), "groups"),
                     (lambda: mr.record_of_torch("conv", 0.1, 0, np.ones((1, 1, 3, 3)), 0.1, 0, None, 0.1, 0, dilation=(2, 2)), "dilation"),
                     (lambda: mr.record_of_tflite(dict(tcv, stride=None), "c4"), "layer c4: stride"),
                     (lambda: mr.add_record_of_tflite(tadd, 11, "t7"), "ADD layer t7: tensor 11"),
                     (lambda: mr.check_int8_add(dict(plain, b=ten(shape=(1, 1, 1, 8))), "a1"), "a1: input shapes"),
                     (lambda: mr.check_int8_add(dict(plain, b=ten(data=np.zeros((1, 4, 4, 8), np.int8))), "a2"), "a2: a constant input"),
                     (lambda: mr.check_int8_add(dict(plain, a=ten(scale=np.float32([0.1, 0.2]))), "a3"), "a3: per-axis"),
                     (lambda: mr.check_int8_add(dict(plain, output=ten(type=3)), "a4"), "a4: inputs and output must be int8"),
                     (lambda: mr.add_record_words(mr.AddRecord(2.0 ** 24, 0, 1.0, 0, 1.0, 0, -128, 127, "res3")), "ADD layer res3")):
        try:
            bad()
        except ValueError as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"no refusal: {why}")


# ── gpnae-tflite's gate (no TensorFlow: GPNAE's own case scales stand in for the converter's) ──

def gptfl_rows(mem=None, tfl=None):
    """gpnae_tflite_judge's rows for the lane (on the coefficient memory mem, default the reset image) on INT8_CASES' tanh and sigmoid; tfl(hw, ex)
    gives the TFLite words (default ex)."""
    lane, q, rows = gpnae_model.Lane(gpnae_model.INT8, mem), np.arange(-128, 128, dtype=np.int64), []
    for func, code, op in GPTFL_OPS:
        for c in gpnae_model.INT8_CASES[func]:
            hw, ex = lane.run(q, code, gpnae_model.int8_params(c, code)), gpnae_model.exact_int8(q, code, c)
            rows.append((func, op, (c.s_in, c.z_in, c.s_out, c.z_out), q, hw, ex if tfl is None else tfl(hw, ex), ex))
    return rows


@selftest
def test_gptfl_gate():
    L, ok = gpnae_tflite_judge(gptfl_rows())
    assert ok and L[-2].startswith("ACCURACY: PASS (true function, truth.py") and "TFLITE AGREEMENT (characterization, not gated)" in L[-1], L[-3:]
    broken = gpnae_model.default_mem("int8")
    broken.desc[0][gpnae_model.D_SIGP].bounds[0] = 2048  # sigmoid x >= 0 saturates from 1.0: the gate must fail it
    L, ok = gpnae_tflite_judge(gptfl_rows(mem=broken))
    assert not ok and L[-2].startswith("ACCURACY: FAIL") and "TRUTH int8 sigmoid: FAIL" in "\n".join(L) and "TRUTH int8 tanh: PASS" in "\n".join(L), L[-3:]
    L, ok = gpnae_tflite_judge(gptfl_rows(tfl=lambda hw, ex: 127 - hw))  # TFLite far off: characterization only, the verdict stands
    assert ok and L[-2].startswith("ACCURACY: PASS") and "tanh 0 of 1280 equal" in L[-1], L[-2:]
    off = [r[:4] + (np.clip(r[4] + 3 * (r[0] == "tanh"), -128, 127),) + r[5:] for r in gptfl_rows()]  # tanh 3 LSB high
    L, ok = gpnae_tflite_judge(off)
    assert not ok and "TRUTH int8 tanh: FAIL" in "\n".join(L) and "TRUTH int8 sigmoid: PASS" in "\n".join(L), L[-4:]
    L, ok = gpnae_tflite_judge([])
    assert not ok and L[-1] == "ACCURACY: FAIL (no gated case)"  # nothing gated is not a pass


@selftest
def test_vcs_commands():
    """SIM=vcs: every make point names SIM=vcs, make exports SIENNA_SIM, each build target names vcs (never verilator) with the same define, -G overrides
    as -pvalue+ and -D defines, the lane benches build on VCS, and the binaries and keys are VCS/<top>_sim."""
    import copy
    spec = copy.copy(SPEC)
    spec.sim = "vcs"
    opts = argparse.Namespace(model_dir="/m")
    for name in spec.points(True, list(FORMATS)):
        v, a = SPEC.argv(name, opts), spec.argv(name, opts)
        assert a == (v + ["SIM=vcs"] if v[0] == "make" else v), (name, a)
    env = {k: x for k, x in MAKE_ENV.items() if k not in ("SIM", "SIENNA_SIM")}
    run = lambda *x: subprocess.run(["make", "-s", "-n", *x], cwd=ROOT, env=env, capture_output=True, text=True)
    for args, want in ((("vcs", "GEN_PKG=0", "FMT=int8", "EXTRA_FLAGS=-GFAULT=8 -DBACK_TO_BACK", "LINK_STAGES=2"),
                        ("-top TB_sienna_top", "+define+FMT_INT8", "-pvalue+TB_sienna_top.FAULT=8", "+define+BACK_TO_BACK", "-pvalue+TB_sienna_top.LINK_STAGES=2",
                         "-o " + os.path.join(ROOT, "VCS", "TB_sienna_top_sim"), "./TB_sienna_top_sim  +vcs+dumpvarsoff")),
                       (("vcs", "GEN_PKG=0", "TOP_MODULE=TB_model_run", "TESTBENCH=TB_model_run.sv"), ("-top TB_model_run", "VCS/TB_model_run_sim.build_key.txt")),
                       (("sim", "GEN_PKG=0", "SIM=vcs"), ("VCS/TB_sienna_top_sim.build_key.txt", "cd " + os.path.join(ROOT, "VCS") + " && ./TB_sienna_top_sim")),
                       (("add-lanes", "SIM=vcs", "PYTHON=python3"), ("-top TB_add_lanes", "+define+FMT_INT8", "VCS/add_lanes/sim  +vcs+dumpvarsoff")),
                       (("rq-lanes", "SIM=vcs", "PYTHON=python3"), ("-top TB_requant_lanes", "VCS/rq_lanes/sim")),
                       (("sm-verilator", "SIM=vcs"), ("mesh-regression MATRIX_SIZE=16 SIM=vcs",)),
                       (("gpnae-verilator", "SIM=vcs", "PYTHON=python3"), ("--run gate --tier 0 --sim vcs",))):
        r = run(*args)
        assert r.returncode == 0, (args, r.stderr[-300:])
        for w in want:
            assert w in r.stdout, (args, w, r.stdout[-600:])
        assert not re.search(r"(?m)^\s*\$?\(?verilator |^verilator ", r.stdout) and "-debug" not in r.stdout, (args, r.stdout[-300:])
    assert "-debug_access+all +define+ENABLE_TRACE" in run("vcs", "GEN_PKG=0", "TRACE=vcd").stdout
    r = run("pipeline", "SIM=vcs", "PYTHON=python3", "N=8", "TILE=2")
    assert r.returncode == 0 and "regression.py" in r.stdout, r.stdout[-300:]
    r = subprocess.run(["make", "-s", "-p", "-n", "help", "SIM=vcs"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert re.search(r"(?m)^SIENNA_SIM := vcs$", r.stdout), "make exports SIENNA_SIM"
    r = subprocess.run([sys.executable, "-c", "import model_runner as m; print(m.SIM, m.SIM_TARGET, os.path.basename(m.SIM_DIR))".replace("os.path", "__import__('os').path")],
                       cwd=os.path.join(ROOT, "scripts"), env=dict(env, SIENNA_SIM="vcs"), capture_output=True, text=True)
    assert r.stdout.split() == ["vcs", "vcs", "VCS"], (r.stdout, r.stderr[-300:])
    r = run("regression", "SIM=vcs", "SMALL=1", "FMT=int8", f"PYTHON={sys.executable}")
    pts = [x for x in r.stdout.splitlines() if "--action point " in x]
    assert pts and all(" --sim vcs" in x for x in pts) and " --sim vcs" in r.stdout.splitlines()[-1], r.stdout[-300:]


@selftest
def test_x_words_fail_by_name():
    """A 4-state simulator's x or z: a count, a cycle count, a dumped word, a result file word, a lane bench code or the multi hash fails by name."""
    clean = "  Total    : 64\n  Exact    : 64\n  Tol pass : 0\n  Failed   : 0\n  done asserted @ 900 (90 cycles)\n"
    assert _parse_log(clean, "t")["status"] == "PASS" and _parse_log(clean, "t")["cyc"] == 90
    r = _parse_log(clean.replace("Failed   : 0", "Failed   : x"), "mm_relu")
    assert r["status"] == "FAIL" and r["fired"] == "FAIL: X in count of test mm_relu ('x')", r
    r = _parse_log(clean.replace("(90 cycles)", "(X cycles)"), "mm_relu")
    assert r["status"] == "FAIL" and r["fired"].startswith("FAIL: X in cycle count of test mm_relu"), r
    assert _parse_log(clean + 'Error: "/t/sienna_top.sv", 40: TB_sienna_top.dut.a_credit_accept: at time 5 ns\n')["status"] == "ASSERT"
    w = OUT_WORD.findall("Dropout -> Output (Lane 3) : dec=x hex=3fx00000\n")
    assert w == [("3", "3fx00000")] and fourstate.XZ.search(w[0][1])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dump.txt")
        open(p, "w").write("H b2b 0\nM P0 0 0 3f800000 3f80\nM P0 0 1 3f8x0000 3f80\n")
        try:
            read_truth_dump(p, 32, "mm_relu N8")
            raise AssertionError("the truth dump's parser took an x word")
        except fourstate.Unknown as e:
            assert str(e).startswith("X in dumped word 2 of test mm_relu N8"), e
        p = os.path.join(d, "l.out")
        open(p, "w").write("S 0\n3f80\n3fz0\n")
        try:
            mr.read_outputs(p, "bf16", "conv1")
            raise AssertionError("read_outputs took a z word")
        except fourstate.Unknown as e:
            assert str(e).startswith("Z in output word 1 of test conv1"), e
        open(os.path.join(d, "m.out"), "w").write("S 0\n3f80\n")
        assert mr.read_outputs(os.path.join(d, "m.out"), "bf16")[0][0] == 1.0
        p = os.path.join(d, "rq.mem")
        open(p, "w").write("01 02\n03 x4\n")
        try:
            _lane_codes(p, 2, 2)
            raise AssertionError("_lane_codes took an x code")
        except fourstate.Unknown as e:
            assert str(e).startswith("X in output word 3 of test rq.mem"), e
    try:
        fourstate.field(rf"^MULTI_HASH ({fourstate.HEX}) over", "MULTI_HASH 00000000x0000000 over 96 words\n", "multi", "word hash", 16, flags=re.M)
        raise AssertionError("the multi hash took an x")
    except fourstate.Unknown as e:
        assert str(e).startswith("X in word hash of test multi"), e
    assert SPEC.verdict("int8/N8_T2/pipeline", "FAIL: X in output word 7 of test mm_relu ('xx')\n", 0).startswith("FAIL (X in output word 7")


def selftest_main(argv=None) -> None:
    """Runs every registered self-test (the tools only, no simulator): PASS or FAIL per test, exit 1 if any fails."""
    argparse.ArgumentParser(description=selftest_main.__doc__).add_argument("--action", choices=ACTIONS, default="selftest")
    bad = 0
    for t in SELFTESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except (Exception, SystemExit) as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"ALL {len(SELFTESTS)} PASSED" if bad == 0 else f"{bad} of {len(SELFTESTS)} FAILED")
    sys.exit(1 if bad else 0)


# ── make regression: SIENNA's tests at every N, tile size and format ─────────


def _engine():
    """ArithmeticLibrary's make regression engine, under its own name (SystolicMesh and GPNAE each have a regression.py too)."""
    import importlib.util
    s = importlib.util.spec_from_file_location("aril_regress", os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "regress.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


regress = _engine()
KINDS = ("pipeline", "pipeline-ck0", "pipeline-b2b", "pipeline-ck0-b2b", "model", "pack", "gemm", "tflite", "model-ck0", "gemm-ck0", "tflite-ck0",
         "residual")  # per N, tile size and format; model in fp32 and bf16, tflite and residual in int8
PIPELINE_KINDS = KINDS[:4]  # TB_sienna_top's sweep: collapse-k 1 or 0 (-ck0), built plain or with -DBACK_TO_BACK (-b2b)
CK0_SIZES = (8, 16)  # pipeline-ck0: the pipeline on the collapse-k 0 mesh, at the small sizes
LAYER_CK0 = KINDS[-4:-1]  # the layer contract's kinds on the collapse-k 0 mesh, at every N (pack refuses collapse-k 0)
B2B_SIZES = regress.SIZES  # pipeline-b2b at every N (pipeline-ck0-b2b also takes CK0_SIZES): the pipeline built with -DBACK_TO_BACK
MODEL_COUNT = 3  # inferences per model and point, as the model sweeps run them
UNIT_POINTS = ("add-lanes", "rq-lanes")  # int8 unit benches, run once per make regression when int8 is selected
FAULT_POINTS = ("idle-fault-sp0", "idle-fault-sp1")  # a put on the unused contract must fire a_idle_contract: TB_model_run FAULT 4, TB_sienna_top FAULT 8 (fp32)
MULTI_KIND = "pipeline-multi"  # TB_sienna_multi on sienna_top at COPIES = 2 (SET_PORT = 1), per format at N8_T2
DEFAULT_POINT = "default/N8_T2/pipeline"  # fp32's N8_T2 pipeline built with no precision define (FMT_DEF=): words and cycles must equal fp32's
DEFAULT_TWIN = "fp32/N8_T2/pipeline"
COMPARED = re.compile(r"cycles|^\s*(Total|Exact|Tol pass|Failed|Passed)\s*:|TRUTH:|stall|RESULT:|^\s*WORDS ")  # the lines whose words (hashes) and cycles a twin must repeat


def twin_verdict(text: str, twin_text: str) -> str:
    """PASS when the default point's result lines equal its twin's, else the first line that differs."""
    a, b = compared_lines(text), compared_lines(twin_text)
    if a == b:
        return "PASS"
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"FAIL (differs from {DEFAULT_TWIN} at result line {d + 1}: {(a[d:d+1] or ['<none>'])[0][:60]!r} against {(b[d:d+1] or ['<none>'])[0][:60]!r})"


def compared_lines(text: str) -> list:
    """A pipeline point's result lines (counts, cycles, verdicts) without its command, host and paths: what the default point and its twin must share."""
    out = []
    for ln in regress.ANSI.sub("", text).splitlines():
        if ln.startswith("#") or not COMPARED.search(ln):
            continue
        out.append(re.sub(r"/tmp/\S+|\d+(\.\d+)? s\b", "<run>", ln).rstrip())
    return out


def reg_lanes(n: int) -> int:
    """Activation lanes of a point, as the sweeps build them: 32 up to N = 32, 128 at N = 64."""
    return 128 if n >= 64 else 32


class SiennaSpec(regress.Spec):
    title = "SIENNA REGRESSION"
    formats = tuple(FORMATS)
    sized = True
    drop = ("GPNAE_TIER", "TEST", "QUICK", "COLLAPSE_K", "N", "TILE", "LANES", "MODEL_DIR", "MODEL_COUNT", "GEN_PKG", "LINK_STAGES", "SIENNA_SIM")

    results_dir = None  # the --results a point or summary run names; the default point reads its twin's log there

    def points(self, small, formats):
        if not formats:
            return []
        return ["selftest"] + (list(UNIT_POINTS) if "int8" in formats else []) + ([DEFAULT_POINT, *FAULT_POINTS] if "fp32" in formats else []) + [f"{f}/N{n}_T{t}/{k}" for f in formats for n in (regress.SIZES_SMALL if small else regress.SIZES)
                               for t in pow2_tile_sizes(n) for k in KINDS if self._offered(f, n, k)] \
            + [f"{f}/N8_T2/{MULTI_KIND}" for f in formats]

    @staticmethod
    def _offered(f, n, k):
        """model in fp32 and bf16, tflite (-ck0 too) and residual in int8, the pipeline's -ck0 kinds at CK0_SIZES, LAYER_CK0 at every N, the -b2b kinds at B2B_SIZES."""
        b = k.removesuffix("-ck0")
        return ((b != "model" or f != "int8") and (b != "tflite" or f == "int8") and (k != "residual" or f == "int8")
                and ("-ck0" not in k or k in LAYER_CK0 or n in CK0_SIZES)
                and (not k.endswith("-b2b") or n in B2B_SIZES))

    def parse(self, name):
        if name == "selftest" or name in UNIT_POINTS or name in FAULT_POINTS:
            return {"n": 0, "kind": name}
        if name == DEFAULT_POINT:
            return {"fmt": "fp32", "n": 8, "tile": 2, "lanes": reg_lanes(8), "kind": "pipeline", "ck": 1, "b2b": False, "no_define": True}
        m = re.fullmatch(r"(\w+)/N(\d+)_T(\d+)/([a-z0-9-]+)", name)
        if m and m.group(1) in FORMATS and m.group(4) == MULTI_KIND:
            if (m.group(2), m.group(3)) != ("8", "2"):
                raise ValueError(f"{name}: {MULTI_KIND} runs at N8_T2 only")
            return {"fmt": m.group(1), "n": 8, "tile": 2, "lanes": reg_lanes(8), "kind": MULTI_KIND, "ck": 1, "b2b": False}
        if not m or m.group(1) not in FORMATS or m.group(4) not in KINDS:
            raise ValueError(f"{name}: points are selftest, {', '.join(UNIT_POINTS)} and <fmt>/N<n>_T<t>/<kind>, kind one of {', '.join(KINDS)}")
        f, n, t, k = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if n not in regress.SIZES or t not in pow2_tile_sizes(n):
            raise ValueError(f"{name}: N is 8, 16, 32 or 64 and the tile size a power-of-2 divisor of N")
        if not self._offered(f, n, k):
            raise ValueError(f"{name}: model runs in fp32 and bf16, tflite and residual in int8, the pipeline's -ck0 kinds at N = {', '.join(map(str, CK0_SIZES))}"
                             f" and the -b2b kinds at N = {', '.join(map(str, B2B_SIZES))}")
        return {"fmt": f, "n": n, "tile": t, "lanes": reg_lanes(n), "kind": k, "ck": 0 if "-ck0" in k else 1, "b2b": k.endswith("-b2b")}

    def env(self, name, env):
        if self.parse(name)["kind"].removesuffix("-ck0") in ("pack", "gemm"):  # -O0 builds these ~30% faster at N = 8 and 16 (rs1_spd_sienna); pipeline, tflite and model simulations are long enough to lose more (pipeline N = 16 +40-125%)
            env = dict(env, OPT_FAST="-O0")
        return env

    def argv(self, name, opts):
        v = self.verilator_argv(name, opts)
        return v + ["SIM=vcs"] if self.sim == "vcs" and v[0] == "make" else v

    def verilator_argv(self, name, opts):
        p, py = self.parse(name), sys.executable
        if p["kind"] == "selftest":
            return [py, "regression.py", "--action", "selftest"]
        if p["kind"] in UNIT_POINTS:
            return ["make", p["kind"], f"PYTHON={py}"]
        if p["kind"] in FAULT_POINTS:
            return ["make", "idle-fault", f"SET_PORT={p['kind'][-1]}", f"PYTHON={py}"]
        if p["kind"] == MULTI_KIND:
            return ["make", "multi", f"FMT={p['fmt']}", f"PYTHON={py}", f"N={p['n']}", f"TILE={p['tile']}", f"LANES={p['lanes']}", "COPIES=2"]
        k = "pipeline" if p["kind"] in PIPELINE_KINDS else p["kind"].removesuffix("-ck0")
        mk = ["make", k, f"FMT={p['fmt']}", f"PYTHON={py}", f"N={p['n']}", f"TILE={p['tile']}", f"LANES={p['lanes']}"]
        return mk + {"model": [f"MODEL_DIR={getattr(opts, 'model_dir', '')}", f"MODEL_COUNT={MODEL_COUNT}"], "gemm": ["QUICK=1"]}.get(k, []) \
            + (["COLLAPSE_K=0"] if p["ck"] == 0 else []) + (["EXTRA_FLAGS=-DBACK_TO_BACK"] if p["b2b"] else []) \
            + (["FMT_DEF="] if p.get("no_define") else [])

    def verdict(self, name, text, rc):
        if name == DEFAULT_POINT:  # fp32's verdict, built with no define, and the same results as fp32's own point
            v = self.verdict(DEFAULT_TWIN, text, rc)
            if v != "PASS":
                return v
            plain = regress.ANSI.sub("", text)
            defs = re.findall(r"^\s*BUILD DEFINE: (.*)$", plain, re.M)
            if not defs or set(defs) != {"none"}:
                return f"FAIL (its builds' defines are {sorted(set(defs)) or 'not printed'}, not none)"
            twin = os.path.join(self.results_dir or self.results, DEFAULT_TWIN + ".log")
            if not os.path.exists(twin):
                return f"FAIL (no {DEFAULT_TWIN} result to compare with in {os.path.dirname(os.path.dirname(os.path.dirname(twin)))})"
            tt = regress.ANSI.sub("", open(twin, errors="replace").read())
            tdefs = re.findall(r"^\s*BUILD DEFINE: (.*)$", tt, re.M)
            if not tdefs or set(tdefs) != {"+define+FMT_FP32"}:
                return f"FAIL ({DEFAULT_TWIN}'s builds' defines are {sorted(set(tdefs)) or 'not printed'}, not +define+FMT_FP32)"
            run = lambda x: (re.findall(r"^# point \S+\s+run (\S+)", x, re.M) or ["-"])[0]
            if run(plain) != "-" and run(tt) != run(plain):
                return f"FAIL ({DEFAULT_TWIN}'s result is of run {run(tt)}, not {run(plain)})"
            return twin_verdict(text, tt)
        k = self.parse(name)["kind"]
        k = k if k in PIPELINE_KINDS else k.removesuffix("-ck0")  # a LAYER_CK0 point is judged as its base kind
        if k in FAULT_POINTS:  # its log shows the assertion it provokes: judged on the action's own line, not on the firing
            return regress.marker_verdict(regress.ANSI.sub("", text), rc, (r"^IDLE FAULT: CAUGHT",), (r"^IDLE FAULT: (SURVIVED|OTHER)",), firing=False)
        v = regress.common_verdict(text, rc)
        if v:
            return v
        text = regress.ANSI.sub("", text)
        if k == "selftest":
            return regress.marker_verdict(text, rc, (r"^ALL \d+ PASSED$",))
        if k == "add-lanes":  # the bench's beats, latency, tags and busy_o, then its outputs on the true function
            return regress.marker_verdict(text, rc, (r"^add_lanes: \d+ beats, 0 errors, 0 latency errors, 0 tag errors, 0 busy errors \(\d+ busy cycles\);",
                                                     r"^RESULT: PASSED", r"^ADD LANES: PASS"), (r"RESULT: FAILED", r"^ADD LANES: FAIL"))
        if k == MULTI_KIND:
            return regress.marker_verdict(text, rc, (r"^MULTI: PASS \(COPIES=2",), (r"^MULTI: FAIL",))
        if k == "rq-lanes":
            return regress.marker_verdict(text, rc, (r"^TB_requant_lanes: \d+ beats, 0 errors, 0 latency errors;", r"^RESULT: PASSED", r"^REQUANT LANES: PASS"),
                                          (r"RESULT: FAILED", r"^REQUANT LANES: FAIL"))
        if k in PIPELINE_KINDS:
            m = re.findall(r"^\s*Passed\s*:\s*(\d+)\s*/\s*(\d+)\s*$", text, re.M)
            g = re.findall(r"^\s*TRUTH: (\d+) tests gated, (\d+) with misses$", text, re.M)
            miss = sorted(set(re.findall(r"\[FAIL\] truth: (\S+)", text)))
            if miss:
                return f"FAIL (true function: {', '.join(miss)}; {g[-1][1] if g else '?'} tests with misses)"
            if not m or "Sanity Clean" not in text:
                return "FAIL (no 'Passed : X / X' with Sanity Clean)"
            x, y = map(int, m[-1])
            want = len([t for t in PIPELINE_TESTS if self.parse(name)["fmt"] in t.get("formats", tuple(FORMATS))
                        and not ("-ck0" in k and t.get("packed"))])  # collapse-k 1 runs them all, collapse-k 0 all but the packed
            if x != y or y == 0:
                return f"FAIL ({x} of {y} tests passed)"
            if y != want:
                return f"FAIL ({y} tests ran, {self.parse(name)['fmt']} has {want})"
            if not g or int(g[-1][0]) != y:
                return f"FAIL ({g[-1][0] if g else 0} of {y} tests gated on the true function)"
            b = re.findall(r"^\s*BACK_TO_BACK: (\d+) of (\d+) tests ran the back-to-back pass$", text, re.M)
            if b and int(b[-1][1]) != y:
                return f"FAIL (BACK_TO_BACK counts {b[-1][1]} tests, Passed counts {y})"
            nb = int(b[-1][0]) if b else None
            if nb != (y if k.endswith("-b2b") else 0):  # a -b2b point runs the pass in every test, a plain point in none
                return f"FAIL ({'no BACK_TO_BACK count' if nb is None else f'{nb} of {y} tests ran the back-to-back pass'})"
            return "PASS"
        if k == "tflite" and self.parse(name)["n"] >= 16 and self.parse(name)["ck"] == 1:  # the packed TFLite layers run there too (collapse-k 1 only)
            return regress.marker_verdict(text, rc, (r"^MODEL GATE: PASS", r"^TFLITE_PACK: \d+ of \d+ models in \d+ packed layers at N = \d+, 0 failing$"),
                                          (r"^MODEL GATE: FAIL", r"^TFLITE_PACK: .*, [1-9]\d* failing$"))
        if k == "model" and re.search(r"^MODEL GATE: SKIPPED", text, re.M):  # a directory with models but none with its inputs
            return "FAIL (MODEL GATE: SKIPPED: no model ran)"
        if k in ("model", "tflite"):
            return regress.marker_verdict(text, rc, (r"^MODEL GATE: PASS",), (r"^MODEL GATE: FAIL",))
        if k == "residual":
            return regress.marker_verdict(text, rc, (r"^RESIDUAL: PASS",), (r"^RESIDUAL: FAIL",))
        if k == "pack":
            return regress.marker_verdict(text, rc, (r"^PACK REGRESSION PASS: 0 mismatching outputs$",))
        acts = ("tanh", "sigmoid", "selu")  # gemm: its rows, every activation's own check, the lanes' tier and load runs, and no FAIL line
        p = self.parse(name)
        need = {"fp32": [rf"^layer_{x}_bias .* max err " for x in acts],
                "bf16": [rf"^  layer_{x}: 0 outputs differ from the bit-exact model$" for x in acts] + [r"^layer_bias_special .* 0 outputs differ from the bit-exact model$"],
                "int8": [r"^GEMM int8: \d+ cases, 0 failing "] + [rf"^    {t}: \d+ outputs of the model differ from tier 0 of the reset image" for t in LANE_CASES]}[p["fmt"]]
        if p["fmt"] != "int8":  # the layer contract's, and at fp32 T = 2 the set contract's (TB_sienna_model)
            need += [lane_case_line(t, n) for t, n in LANE_CASES.items()]
            if p["fmt"] == "fp32" and p["tile"] == 2 and p["ck"] == 1:
                need += [lane_case_line("sets " + t, LANE_CASES[t]) for t in SET_LANE_CASES]
        return regress.marker_verdict(text, rc, [r"^grid_\d+x\d+x\d+ "] + need, (r"^FAIL ",))

    def skipped(self, name, opts):
        """A model point with no model directory, or one holding none of the models, is skipped (make regression defaults MODEL_DIR to models/)."""
        if name in ("selftest", DEFAULT_POINT) or name in UNIT_POINTS or name in FAULT_POINTS or self.parse(name)["kind"].removesuffix("-ck0") != "model":
            return None
        d = getattr(opts, "model_dir", "")
        if not d or not os.path.isdir(d) or not any(os.path.isfile(os.path.join(d, m)) for m in mr.MODELS.values()):
            return "no MODEL_DIR" + (f" ({d} holds no model)" if d else "")
        return None


SPEC = SiennaSpec(ROOT, os.path.join(ROOT, "testbenches", "results", "regression"))
if "--results" in sys.argv[:-1]:  # point and summary runs: where the default point's twin is
    SPEC.results_dir = os.path.abspath(sys.argv[sys.argv.index("--results") + 1])


def idle_fault_main(argv) -> None:
    """A put on the unused contract (TB_model_run FAULT 4 at SET_PORT 0 through make gemm, TB_sienna_top FAULT 8 at SET_PORT 1): IDLE FAULT: CAUGHT
    when the run fails with sienna_top's a_idle_contract message, SURVIVED when it does not fire, OTHER when the run fails some other way."""
    ap = argparse.ArgumentParser(prog="regression.py --action idle-fault")
    ap.add_argument("--set-port", type=int, choices=(0, 1), required=True)
    a = ap.parse_args(argv)
    env = dict(os.environ, EXTRA_FLAGS=f"-GFAULT={4 if a.set_port == 0 else 8}")
    if a.set_port == 0:
        cmd = ["make", "gemm", "FMT=fp32", "N=8", "TILE=2", "LANES=32", "QUICK=1", f"PYTHON={sys.executable}"]
    else:
        t = next(x for x in PIPELINE_TESTS if x["name"] == PKG_DEFAULT_TEST)
        generate_vectors({"n": 8, "tile_size": 2, "lanes": 32, **t, "fmt_name": "fp32"})  # the package and vectors TB_sienna_top reads
        cmd = ["make", mr.SIM_TARGET, "TRACE=0", "FMT=fp32", "N=8", "TILE=2", "LANES=32", "GEN_PKG=0", "EXTRA_FLAGS=-GFAULT=8"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    text = r.stdout + r.stderr
    fired = "a_idle_contract" in text and re.search(r'Assertion failed|%Error|^(?:Error|Fatal): "|: started at .* failed at', text, re.M) is not None
    planted = f"[FAULT {4 if a.set_port == 0 else 8}]" in text
    print("\n".join(x for x in text.splitlines() if "a_idle_contract" in x or "FAULT" in x)[:2000])
    v = "CAUGHT" if fired and planted else ("SURVIVED" if planted else "OTHER")
    print(f"IDLE FAULT: {v} (SET_PORT {a.set_port}: the planted put {'ran' if planted else 'never ran'}, a_idle_contract {'fired' if fired else 'did not fire'}, exit {r.returncode})")
    sys.exit(0 if v == "CAUGHT" else 1)


def multi_main(argv) -> None:
    """TB_sienna_multi on sienna_top at COPIES copies (SET_PORT = 1): one pipeline test's sets, every set against its golden, then MULTI: PASS or FAIL."""
    ap = argparse.ArgumentParser(prog="regression.py --action multi")
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(FORMATS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tile", type=int, default=2)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--sets", type=int, default=48)
    ap.add_argument("--test", default="matmul_relu_nopool")
    a = ap.parse_args(argv)
    t = next(x for x in PIPELINE_TESTS if x["name"] == a.test)
    generate_vectors({"n": a.n, "tile_size": a.tile, "lanes": a.lanes, **t, "num_sets": a.sets, "fmt_name": a.fmt_name})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lp = os.path.join(RESULTS_DIR, f"multi_{a.fmt_name}_N{a.n}_T{a.tile}.log")
    cmd = ["make", mr.SIM_TARGET, "TOP_MODULE=TB_sienna_multi", "TESTBENCH=TB_sienna_multi.sv", "TRACE=0", f"FMT={a.fmt_name}", f"N={a.n}",
           f"TILE={a.tile}", f"LANES={a.lanes}", "GEN_PKG=0", f"EXTRA_FLAGS=-GCOPIES={a.copies} -GCOLLAPSE_K=1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = r.stdout + r.stderr
    open(lp, "w").write(text)
    print("\n".join(x for x in text.splitlines() if x.startswith(("MULTI", "RESULT", "  [FAIL]", "%Error", "[FATAL]")) or "Assertion failed" in x))
    try:  # the copies' word hash: an x or z in it fails the run by name
        fourstate.field(rf"^MULTI_HASH ({fourstate.HEX}) over", text, f"multi {a.fmt_name} N{a.n} T{a.tile}", "word hash", 16, flags=re.M)
        xline = ""
    except fourstate.Unknown as e:
        xline = fourstate.fail_line(e)
        print(xline)
    bad = r.returncode != 0 or bool(xline) or mr.FIRED.search(text.replace("[FAIL]", "")) is not None or not re.search(r"^RESULT: PASSED", text, re.M) \
        or not re.search(rf"^MULTI COPIES={a.copies} ", text, re.M)
    print(f"MULTI: {'FAIL' if bad else 'PASS'} (COPIES={a.copies} {a.fmt_name} N={a.n} T={a.tile} {a.sets} sets of {a.test}, make exit {r.returncode}; {lp})")
    sys.exit(1 if bad else 0)


MAINS = {"idle-fault": idle_fault_main, "multi": multi_main, "pack": pack_main, "gemm": gemm_main, "residual": residual_main, "perf": perf_main, "oracle": oracle_main, "pack-models": pack_models_main,
         "gpnae-tflite": gpnae_tflite_main, "rq-vectors": rq_vectors_main, "add-vectors": add_vectors_main, "selftest": selftest_main,
         "summary": summary_main, **{x: (lambda argv, x=x: regress.main(SPEC, ["--action", x, *argv])) for x in ("points", "point")}}  # each parses its own options (regression.py --action A --help)

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # no abbreviation: pack's --act is not --action
    pre.add_argument("--action", choices=ACTIONS, default="regression")
    pre_args, rest = pre.parse_known_args()
    if pre_args.action in MAINS:
        MAINS[pre_args.action](rest)
        sys.exit(0)
    p = argparse.ArgumentParser(description="SIENNA Pipeline Unified Tool")
    p.add_argument(
        "--action",
        default="regression",
        choices=ACTIONS,
        help="Action to perform; pkg writes test_config_pkg.sv and one --test's stimulus (default %s); "
             "pack, gemm, perf, oracle, pack-models, gpnae-tflite, rq-vectors and add-vectors take their own options (--action A --help)" % PKG_DEFAULT_TEST,
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
        "--test", type=str, default=None, help="regression: tests whose name contains this; pkg: one exact test name"
    )
    args, unknown = p.parse_known_args()
    mr.COLLAPSE_K = args.collapse_k

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
