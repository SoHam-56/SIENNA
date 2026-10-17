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
import functools
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
import model_runner as mr  # noqa: E402  the numerics and the device build package, one owner each
from model_runner import (  # noqa: E402
    FORMATS, ROOT, SETS_IN_FLIGHT, TB_DIR, _check_rounding, _config_items, activate_int8, activation_to_code,
    apply_activation, bits_float, drop_zp, exact_layer, fmt_bits, fold_bias, get_polynomial_terms, imatmul, op_hex, op_round,
    quant_act, quant_weights, requant_params, requantize, wrap32, write_sv_package)
import ipu  # noqa: E402  AriL's integer model, on the path model_runner set


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
ACTIONS = ("regression", "gen", "pkg", "analyze", "pack", "gemm", "perf", "oracle", "pack-models", "gpnae-tflite", "rq-vectors", "add-vectors", "selftest", "points", "point", "summary")

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
    rom = gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f)))
    A = gpnae_model.Lane(f, rom).run(C, activation_to_code(act))
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


# ===== int8 (D-6): TFLite-style quantization of the float tests' data, and the bit-exact golden =====

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
    """Bit-exact int8 residual set: the depth passes' sums, the conv's requantize (q1, unless given), TFLite's ADD with R, the lane with the ADD's words, pooling, dropout (D-5: zp_o)."""
    N = cfg.get("n", 16)
    C = mesh_model.matmul_int([(_pad_square(a, N), _pad_square(b, N)) for a, b in parts], N, hw_bias)
    Y = mr.tflite_add_codes(requantize(C, rq) if q1 is None else q1, R, add)
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
    return R, add, dict(mr.add_lane_params(add, s_out, mr.tflite_add_codes(q1, R, add)), setting=(f2, z2, fo))


def _off_rails(F, ra: dict) -> tuple:
    """An add set's final outputs: the share off the rails (-128, 127 and the ADD's clamp) and the number of distinct codes."""
    F = np.asarray(F).ravel()
    return float(np.mean(~np.isin(F, [-128, 127, ra["amin"], ra["amax"]]))), len(np.unique(F))


def _identity_requant(N: int, ra: dict) -> dict:
    """An add set's requantize words: identity (mult 2^30, shift +1, zp 0, full clamp), with the lane's words after the ADD."""
    return dict(mult=np.full(N, 1 << 30, np.int64), shift=np.full(N, 1, np.int64), zp=0, amin=-128, amax=127,
                mx=ra["mx"], shx=ra["shx"], mout=ra["mout"], shout=ra["shout"], zout=ra["zout"])


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


def _res_flags(partial=False, bias=False, hold=False, add=False, act="linear") -> int:
    """add_<k>.mem's first word: bit 0 partial, 1 bias, 2 hold, 3 add, bits 7:4 the set's activation code."""
    return int(partial) | int(bias) << 1 | int(hold) << 2 | int(add) << 3 | activation_to_code(act) << 4


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
                                       act_g != "relu" and g % 2 == 1)
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
            _write_w32(w("add", k), [_res_flags(partial=not last, bias=i == 0, hold=res and last, act=act_k)] + [0] * 11)
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
            scales[str(k)] = {"0": [float(ra["s_out"]), float(ra["s_selu"])], "ratio": None}
            alone(k, R, np.eye(N, dtype=np.int64), np.zeros(N, np.int64), rq_i, act_g)
    return first, scales


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
    lane = gpnae_model.Lane(f, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f))))
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
    if fmt == "int8":
        _check_rounding()
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
INT8_FIXED_OUT = {"sigmoid": (1 / 256, -128), "tanh": (1 / 128, 0)}  # TFLite's fixed output quantization
TRUTH_STAGES = ("matmul", "narrow", "transfer", "requant", "add", "multiplier", "relu", "activation", "maxpool", "dropout")
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


def read_truth_dump(path: str, lanes: int) -> dict:
    """{pass: {"M": {r: {e: (sum, word)}}, "X"/"A"/"R": {k: {e: word}}, "Q": {k: {e: (q1, q2, y)}}, "P": {k: {window: (word, keep)}},
    "D": {k: {i: word}}}}; R: a hold set's beats into hold_mem, Q: an add set's ADD inputs and output."""
    d = {}
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if not t:
                continue
            if t[0] == "H":  # the build's passes
                d.setdefault("_hdr", {})[t[1]] = int(t[2])
                continue
            g = d.setdefault(t[1], {"M": {}, "X": {}, "A": {}, "P": {}, "D": {}, "Q": {}, "R": {}})
            if t[0] == "M":
                g["M"].setdefault(int(t[2]), {})[int(t[3])] = (int(t[4], 16), int(t[5], 16))
            elif t[0] == "Q":
                g["Q"].setdefault(int(t[2]), {})[int(t[3])] = (int(t[4], 16), int(t[5], 16), int(t[6], 16))
            elif t[0] in ("X", "A", "D", "R"):
                g[t[0]].setdefault(int(t[2]), {})[int(t[3])] = int(t[4], 16)
            elif t[0] == "P":
                g["P"].setdefault(int(t[2]), {})[int(t[4]) * lanes + int(t[3])] = (int(t[5], 16), int(t[6]))
            else:
                raise ValueError(f"truth dump: unknown line {ln.strip()!r}")
    return d


def _full(m: dict, n: int, what: str) -> list:
    """m's values in key order 0..n-1; anything missing or extra is a dump fault."""
    if sorted(m) != list(range(n)):
        raise RuntimeError(f"truth dump: {what} holds {len(m)} of {n} entries")
    return [m[i] for i in range(n)]


def _res_flags_of(k: int) -> dict:
    """A residual test's set k as add_<k>.mem gives it: partial, bias, hold and add flags, its activation, and the ADD's words (signed)."""
    w = _tb_words(f"add_{k}.mem")
    f = int(w[0])
    words = {n: int(_sx(v, 32 if n in ("m1", "m2", "mo") else 8)) for n, v in zip(mr.ADD_FIELDS, w[1:])}
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
                       "mout": int(pk[o + 6] & 0xFFFFFFFF), "shout": int(_sx(pk[o + 7], 8)), "zout": int(_sx(pk[o + 8], 8))}
    if fmt == "int8":
        rq = _tb_words(f"requant_{k}.mem")
        ents[0].update(zp=int(_sx(rq[0], 8)), amin=int(_sx(rq[1], 8)), amax=int(_sx(rq[2], 8)), mx=int(rq[3] & 0xFFFF), shx=int(rq[4] & 0x1F),
                       mout=int(rq[5] & 0xFFFFFFFF), shout=int(_sx(rq[6], 8)), zout=int(_sx(rq[7], 8)))
        mult, shift = rq[8:8 + N] & 0xFFFFFFFF, _sx(rq[8 + N:8 + 2 * N], 8)
    else:
        mult = shift = None
    return {"ents": ents, "col_ent": col_ent, "sh": sh, "mult": mult, "shift": shift}


def _int8_quant(ents: dict, scales: dict, k: int):
    """Per entry (s_in, z_in, s_out, z_out) of the lane: the generator's real scales, the zero points the TB drove."""
    out = {}
    for e, w in ents.items():
        sc = scales.get(str(k), {}).get(str(e))
        if sc is None:  # a set the generator calibrated no lane for (P0's or B2B's partial): the scales its words encode
            s_in = w["mx"] * 2.0 ** -w["shx"] / 2048  # SELU's output: acc 2^-(shx + 18) times mout 2^(shout - 31) is the code
            sc = [s_in, s_in * 2.0 ** (24 - w["shout"]) / (w["mx"] * w["mout"]) if w["mout"] and w["mx"] else np.inf]
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


def truth_test(N: int, fmt: str, lanes: int) -> dict:
    """{stage (activation: 'activation <func>'): Result} over every activated set of the dumped passes of the test just run."""
    knobs, pkg = _tb_knobs(), _pkg_fields(os.path.join(TB_DIR, "test_config_pkg.sv"))
    if not os.path.exists(TRUTH_DUMP):
        raise RuntimeError(f"no truth dump at {TRUTH_DUMP}")
    dump = read_truth_dump(TRUTH_DUMP, lanes)
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
    def multipliers(k, sw):  # the host's multipliers, once per set: within their own quantization of s_a s_w / s_out
        ratio = scales.get(str(k), {}).get("ratio") or []
        cs = [c for c in range(min(N, len(ratio))) if ratio[c] is not None]
        if cs and k not in mult_seen:
            mult_seen.add(k)
            add("multiplier", truth.gate_multiplier(sw["mult"][cs], sw["shift"][cs], [ratio[c] for c in cs]))
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
                add("requant", truth.gate_requant(pre, sw["mult"][cols], sw["shift"][cols], zp, lo, hi, Rh))
                multipliers(k, sw)
                held[k] = Rh
                continue
            if addk:  # an add set: requantize (the identity) gives q2, hold_mem q1, TFLite's ADD y, the activation stage's input
                Q = _full(g["Q"].get(k, {}), N * N, f"pass {tag} set {k}'s ADD outputs")
                q1, q2, y = (np.array([v[i] for v in Q], dtype=np.int64) for i in range(3))
                if k - 1 not in held:
                    raise RuntimeError(f"truth dump: pass {tag} add set {k} has no hold set before it")
                add("transfer", _transfer("int8", held[k - 1], q1))  # hold_mem: each element its hold set's own beat
                add("add", truth.gate_add(q1, q2, fl[k]["words"], y, mr.ROUNDING))
            X = np.array(_full(g["X"].get(k, {}), N * N, f"pass {tag} set {k}'s activation inputs"), dtype=np.int64)
            Aw = np.array(_full(g["A"].get(k, {}), N * N, f"pass {tag} set {k}'s activation bank"), dtype=np.int64)
            if fmt == "int8":
                zp, lo, hi = (np.array([sw["ents"][int(e)][x] for e in ent]) for x in ("zp", "amin", "amax"))
                add("requant", truth.gate_requant(pre, sw["mult"][cols], sw["shift"][cols], zp, lo, hi, q2 if addk else X))
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
                        add("activation relu", truth.gate("int8", "linear", X[m], Aw[m], quant=(1.0, w["zp"], 1.0, w["zp"]))._replace(func="relu"))
                        if str(e) in scales.get(str(k), {}) and not (res_t and fl[k]["add"] and not addk):  # a calibrated ReLU entry: its
                            add("relu clamp", _relu_clamp(w, k, e))  # clamp must start at its zero point; an add set alone has the identity's
                        continue
                    add(f"activation {func}", truth.gate("int8", func, X[m], Aw[m], quant=q8[e]))
                else:
                    add(f"activation {func}", truth.gate(fmt, func, X[m], Aw[m]))
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
        lines.append("        " + (truth.line(r) if name.startswith("activation") else truth.stage_line(r)))
        if not r.ok:
            w = {"matmul": lambda: f"{truth.worst_text(r.worst)} of sum|ab|", "narrow": lambda: f"{r.worst_lsb} ULP",
                 "requant": lambda: f"{r.worst_lsb} LSB from round(real)", "multiplier": lambda: f"{r.worst:.3f} LSB of the multiplier",
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
    {"name": "int8_residual_relu", "mode": "matmul", "matrix_type": "random", "act": "relu", "residual": True,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_linear_nopool", "mode": "matmul", "matrix_type": "random", "act": "linear", "residual": True,
     "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "residual": True, "accum_passes": 2,
     "req_random": True, "zp_random": True, "formats": ("int8",)},
    {"name": "int8_residual_sigmoid_selu_nopool", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "residual": True,
     "mixed_acts": ["sigmoid", "selu"], "pool_h": 1, "pool_w": 1, "padding": 0, "req_random": True, "zp_random": True, "num_sets": 16,
     "formats": ("int8",)},
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
        ["make", "verilator" if build else "sim", f"FMT={fmt_name}", f"N={N}", f"TILE={T}", f"LANES={lanes}", f"COLLAPSE_K={mr.COLLAPSE_K}",
         "GEN_PKG=0"],  # GEN_PKG=0: build the package this test just wrote, in its format and geometry
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
    fired = re.search(r"^.*(?:Assertion failed|%Error).*$", raw, re.M)
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
        "b2b": bool(re.search(r"^ BACK_TO_BACK: (?:PASSED|FAILED)", raw, re.M)),  # a -DBACK_TO_BACK build ran its back-to-back pass
    }


HOLD_ADD = re.compile(r"^\[HOLD/ADD\] hold sets done (\d+), add beats (\d+), add beats of other sets (\d+), hold release with an L3 grant (\d+), "
                      r"hold done with a bank done (\d+)", re.M)  # TB_sienna_top's line per stream pass of a residual test


def _residual_pairs_bad(raw: str, per_lane: int) -> str:
    """A residual test's '[FAIL] residual: ...' line, or '': each stream pass's [HOLD/ADD] line must count a hold set done per add set,
    per_lane add_lanes beats per add set and none for another set."""
    k_sets = _tb_knobs()["num_sets"]  # the files past num_sets may be a longer earlier test's
    want = sum(int(open(os.path.join(TB_DIR, f"add_{k}.mem")).readline(), 16) >> 3 & 1 for k in range(k_sets))
    got = [tuple(int(x) for x in m[:3]) for m in HOLD_ADD.findall(raw)]
    if not got or any(g != (want, want * per_lane, 0) for g in got):
        return (f"[FAIL] residual: {'; '.join('%d hold sets done, %d add beats, %d of other sets' % g for g in got) or 'no [HOLD/ADD] line'}, "
                f"expected {want}, {want * per_lane} and 0 in each stream pass")
    return ""


def _residual_cross_bad(ran: list, crosses: int, every=None) -> str:
    """'[FAIL] residual: ...' when every residual test (every: names, default PIPELINE_TESTS') ran and no hold done met an L3 grant or a bank done in any (spec Risk 3)."""
    every = set(_residual_tests() if every is None else every)
    if crosses == 0 and every and every <= set(ran):
        return "[FAIL] residual: in no residual test did a hold set's done meet an L3 grant or a bank done, so neither cross was exercised"
    return ""


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
    res_cross = 0  # residual tests' same-cycle crosses, summed: gated once every residual test has run

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
        built = key if "-- Done" in raw_log or "-- Running simulation" in raw_log else None  # a failed build leaves no binary to reuse

        # 3. Parse Log
        r = {
            "name": t["name"],
            "mode": t["mode"],
            "act": t["act"],
            "wall": wall,
            **_parse_log(raw_log),
        }
        if t.get("residual") or t.get("residual_every"):  # the hold sets and add beats each stream pass counted
            bad = _residual_pairs_bad(raw_log, N * N // lanes)
            res_cross += sum(int(m[3]) + int(m[4]) for m in HOLD_ADD.findall(raw_log))
            if bad:
                print(f"      {err(bad)}")
                r.update(status="FAIL" if r["status"] == "PASS" else r["status"], failed=max(r["failed"], 1))
        results.append(r)

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
    cross_bad = _residual_cross_bad([r["name"] for r in results], res_cross)
    if cross_bad:
        print(f"  {err(cross_bad)}")
        sys.exit(1)

    gated = [r for r in results if "truth" in r]
    names = sorted({k for r in gated for k in r["truth"] if not k.startswith("_")},
                   key=lambda x: (TRUTH_STAGES.index(x.split()[0]), x))
    print(hdr(f"\n  TRUE-FUNCTION GATES, all tests ({fmt_name}, N={N}, TILE={T})"))
    for name in names:
        m = truth.merge([r["truth"].get(name) for r in gated])
        print("  " + (truth.line(m) if name.startswith("activation") else truth.stage_line(m)))
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
        mult, shift = np.zeros(N, np.int64), np.zeros(N, np.int64)
        for e in sorted(set(col_ent)):
            cols = np.array(col_ent) == e
            act = "linear" if a.acts[e] == "selu" else a.acts[e]  # SELU's int8 saturation is pack_jobs' to refuse (Task 7)
            rq = mr.requant_params(acc[:, cols], s_a, s_w[cols], act)
            mult[cols], shift[cols] = rq["mult"], rq["shift"]
            ents[e] = (act, rq)
        head = ents[0][1] or mr.requant_params(acc, s_a, s_w, "linear")
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw, "act": ents[0][0], "shape": (R * N, N),
               "req": dict(head, mult=mult, shift=shift), "pack": {"shift": sh, "map": mp, "ents": ents}}
    else:
        A, B, bias = (mr.op_round(v, a.fmt) for v in (A, B, bias))
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
            gold = mr.int8_layer_exact(A_q[:, cols], B_q[cols, cols], hw[cols], rq, ents[e][0])
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
    """Packed layers on sienna_layer: each job's block equals the job alone on the RTL and its golden, bit for bit; cycles packed vs alone."""
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
    """int8 on the layer engine: TFLite-PTQ-quantized products must equal the model bit for bit; the quantization error is reported, not gated."""
    rep = open(os.path.join(a.work, f"gemm_sweep_N{a.n}.log"), "w")
    peak = a.n * a.n
    head = (f"{'shape':<22} {'M':>5} {'K':>5} {'N':>5} {'sets':>7} {'cycles':>10} {'MAC/cycle':>9} {'PE use':>7} "
            f"{'slot use':>8} {'mism':>6} {'q err':>8} {'wall s':>6}")
    for line in (f"GEMM sweep on the RTL, mesh N={a.n}, {a.lanes} lanes, int8 operands, int32 sums, requantized output; "
                 f"peak {peak} MAC/cycle", head):
        print(line, flush=True)
        rep.write(line + "\n")
    cases = [(name, m, k, n, "linear") for name, m, k, n in shapes]
    cases += [(f"layer_{act}_bias", 64, 48, 40, act) for act in ("tanh", "sigmoid", "selu")]
    rows, bad = [], 0
    for name, m, k, n, act in cases:
        rng = np.random.RandomState(m * 7 + k * 13 + n)
        A, B = rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))
        bias = None if act == "linear" else rng.uniform(-0.5, 0.5, n)
        A_q, s_a, z_a = mr.quant_act(A)
        B_q, s_w = mr.quant_weights(B)
        hw_bias = mr.fold_bias(bias, s_a, s_w, z_a, B_q)
        req = mr.requant_params(mr.wrap32(mr.imatmul(A_q, B_q) + hw_bias[None, :]), s_a, s_w, act)
        job = {"terms": [(A_q.astype(np.float32), B_q.astype(np.float32))], "bias": hw_bias, "act": act,
               "shape": (m, n), "req": req}
        t0 = time.time()
        y, sets, cyc = sim.run_job(job, name)
        mism = int(np.sum(y != mr.int8_layer_exact(A_q, B_q, hw_bias, req, act)))
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
        if mism:
            print(f"FAIL {name}: {mism} outputs differ from the bit-exact model", flush=True)
            bad += 1
    line = f"GEMM int8: {len(rows)} cases, {bad} with outputs that differ from the bit-exact model"
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
                    help="layer: sienna_layer schedules everything; sets: the host drives sienna_top set by set")
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(mr.FORMATS),
                    help="format of A and B on the layer engine; int8 sums in int32 and requantizes")
    ap.add_argument("--host-gaps", action="store_true", help="host idles a cycle after each load and waits for the credit")
    a = ap.parse_args(argv)
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
    for line in (f"GEMM sweep on the RTL, mesh N={a.n}, {a.lanes} lanes, linear activation, {a.fmt_name} operands and results, fp32 sums; "
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


def layer_bias_special(a, sim, rep) -> bool:
    """One bf16 layer whose bias row holds +0, -0 and subnormals (_special_bias): bit-exact, and sienna_layer's a_bias_widened, the only check that sees them, must not fire."""
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


# ── perf (was perf_analysis.py) ──────────────────────────────────────────────

REPORT = os.path.join(ROOT, "testbenches", "results", "perf", "pipeline_performance_report.log")
CONFIGS = [t["name"] for t in PIPELINE_TESTS]


GEOM = {"n": 16, "tile_size": 4, "lanes": 32}  # set from the command line in perf_main()


def run(name: str, num_sets: int, build_dir: str) -> str:
    cfg = next(t for t in PIPELINE_TESTS if t["name"] == name)
    generate_vectors({**GEOM, **cfg, "num_sets": num_sets})
    cmd = ["make", "verilator", "TRACE=0", "EXTRA_FLAGS=-DPERF", f"VERILATOR_DIR={build_dir}",
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


def analyse(ev: dict, k_sets: int, passes: int = 1) -> dict:
    """Per-set stage times; with passes > 1 only every passes-th set outputs (the others are partial sums)."""
    load, start = [c for c, _, _ in ev["HOST_LOAD"]], [c for c, _, _ in ev["HOST_START"]]
    launch = [c for c, a, _ in ev.get("MESH", []) if a == 1]
    written = [c for c, a, _ in ev.get("MESH", []) if a == 7]
    reduce = [c for c, a, _ in ev.get("MESH", []) if a == 5]
    g_up, g_dn = leaves(ev, "G", 0)
    p_up, p_dn = leaves(ev, "P", 0)
    done = [c for c, _, _ in ev["DONE"]]
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


DEGREE = {"selu": 8, "sigmoid": 6, "tanh": 8}  # gpnae_poly coefficient table in fp32 and bf16; int8 takes gpnae_model.SETS_INT8's
FMT_KNOBS = {"fp32": (8, 23), "bf16": (8, 7), "int8": (0, 7)}  # (EXP_W, MAN_W) of each build
FMT_PKG = os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "src", "sienna_fmt_pkg.sv")


def rtl_lat() -> tuple:
    """sienna_fmt_pkg's unit latencies, parsed from the RTL: ({fmt: (mul_lat, add_lat)}, fx_lat, req_lat)."""
    raw = open(FMT_PKG).read()

    def body(fn: str, form: str) -> tuple:
        m = re.search(rf"function automatic int {fn}\([^)]*\);(.*?)endfunction", raw, re.S)
        b = re.fullmatch(form, re.sub(r"//[^\n]*", "", m.group(1)).strip()) if m else None
        if not b:
            raise RuntimeError(f"regression.py --action perf: {FMT_PKG}: {fn}() is not in the form the model reads; update rtl_lat()")
        return tuple(int(x) for x in b.groups())

    mi, mw, mhi, mlo = body("mul_lat", r"if \(is_int\(exp_w\)\) return (\d+);\s*return \(man_w \+ 1 > (\d+)\) \? (\d+) : (\d+);")
    ai, af = body("add_lat", r"return is_int\(exp_w\) \? (\d+) : (\d+);")
    unit = {f: ((mi if e == 0 else mhi if m + 1 > mw else mlo), (ai if e == 0 else af)) for f, (e, m) in FMT_KNOBS.items()}
    return unit, body("fx_lat", r"return (\d+);")[0], body("req_lat", r"return (\d+);")[0]


UNIT_LAT = FX_LAT = REQ_LAT = None  # mul_lat and add_lat per format, fxMac, tfliteRequant; perf_init() parses them
MUL_LAT = ADD_LAT = None  # valid in to done out; perf_init() sets fp32's, perf_main() the build's format's values
MAC_LAT = {"fp32": 13, "bf16": 8, "int8": 3}  # barrel_mac's Horner loop: multiplier then adder, or fxMac behind a register stage
FMT = "fp32"  # the build's format; perf_main() sets it


def degree(act: str) -> int:
    """The polynomial degree of act's coefficient set in the build's format: Task 10's SETS_INT8 in int8."""
    if FMT == "int8":
        return gpnae_model.SETS_INT8[mr.activation_to_code(act)][1]
    return DEGREE[act]


def pkg() -> dict:
    """The generated test_config_pkg's integer parameters: the geometry this run was built with."""
    raw = open(os.path.join(ROOT, "testbenches", "test_config_pkg.sv")).read()
    return {k: int(v) for k, v in re.findall(r"localparam int (\w+) = (-?\d+);", raw)}


def clog2(x: int) -> int:
    return max(0, (x - 1).bit_length())


def model(cfg: dict, collapse: bool = True) -> dict:
    """Cycles each stage should take per set, from the RTL's structure."""
    P = pkg()
    N, T, lanes = P["N"], P["TILE_SIZE"], P["NUM_LANES"]
    per_lane = P["SRAM_DEPTH"] // lanes
    K = N if collapse else T  # depth each array multiplies
    U = min(K, ADD_LAT + 1)  # partial sums per PE pixel: the adder loop plus one (6 in fp32 and bf16, 2 in int8)
    RP = 1 if collapse else N // T  # depth slices summed per output tile
    LAT = 1 + ADD_LAT * clog2(RP * U + 1)  # reducer read to write: tree over the partials and the bias
    words = len(open(os.path.join(mr.TB_DIR, "matrix_west_0.mem")).read().split())
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
    act = cfg.get("act")
    windows = ((P["IN_ROWS"] + 2 * P["PADDING"] - P["POOL_H"]) // P["STRIDE_ROWS"] + 1) * \
              ((P["IN_COLS"] + 2 * P["PADDING"] - P["POOL_W"]) // P["STRIDE_COLS"] + 1)
    m["pool_dispatch"] = -(-windows // lanes) * P["POOL_H"] * P["POOL_W"]  # every lane takes a window element per cycle
    if act in ("relu", "linear") and not cfg.get("mixed_acts") and cfg.get("accum_passes", 1) == 1:
        m["act"] = per_lane + 4  # FEED, LATCH, one wide beat per cycle plus a cycle of read latency, then done; int8 leaves while the requantize drains
    elif act in DEGREE and not cfg.get("mixed_acts") and cfg.get("accum_passes", 1) == 1:
        m["act"] = lane_stage(act, P, collapse)
    return m


def lane_params() -> tuple:
    """gpnae_poly's K and TAIL_CONTEXTS defaults, parsed from the RTL; fails if sienna_top overrides either."""
    poly = open(os.path.join(ROOT, "GPNAE", "src", "gpnae_poly.sv")).read()
    inst = re.search(r"gpnae_poly #\((.*?)\) gpnae_inst", open(os.path.join(ROOT, "src", "sienna_top.sv")).read(), re.S)
    k, t = (re.search(rf"parameter int\s+{n}\s*=\s*(\d+)", poly) for n in ("K", "TAIL_CONTEXTS"))
    if not (k and t and inst) or re.search(r"\.(K|TAIL_CONTEXTS)\s*\(", inst.group(1)):
        raise RuntimeError("regression.py --action perf: gpnae_poly's K / TAIL_CONTEXTS defaults not found, or sienna_top overrides them")
    return int(k.group(1)), int(t.group(1))


GROUP_K = TAIL_CTX = None  # gpnae_poly's K and TAIL_CONTEXTS, which sienna_top leaves at their defaults; perf_init() parses them


def perf_init() -> None:
    """Parses the RTL's unit latencies and lane parameters for the perf model; at perf's entry, not at import, so an unparsable RTL fails only perf."""
    global UNIT_LAT, FX_LAT, REQ_LAT, MUL_LAT, ADD_LAT, GROUP_K, TAIL_CTX
    UNIT_LAT, FX_LAT, REQ_LAT = rtl_lat()
    MUL_LAT, ADD_LAT = UNIT_LAT["fp32"]
    GROUP_K, TAIL_CTX = lane_params()
DN_LAT = {"fp32": 6, "bf16": 5}  # sigmoid's P - 1: fp32_down's valid_stage6, or the format's fpAdder


def lane_inputs(k: int, P: dict, collapse: bool) -> list:
    """Set k's activation inputs from the bit-exact mesh model, row-major as the wide read hands them to the lanes."""
    rd = lambda f: np.array([int(w, 16) for w in open(os.path.join(mr.TB_DIR, f)).read().split()], np.int64)
    n, f = P["N"], fpu.FORMATS[FMT]
    A, B = (rd(f"matrix_{s}_{k}.mem").reshape(n, n) for s in ("west", "north"))
    C = mesh_model.matmul(f, [(A, B)], n, P["TILE_SIZE"], int(collapse), rd(f"bias_{k}.mem") if P["HAS_BIAS"] else None)
    return mr.bits_float(np.asarray(C, np.int64), FMT).astype(np.float64).flatten().tolist()  # bf16: each fp32 sum rounded once, as the lanes see it


def in_tail(x: float, act: str) -> bool:
    """gpnae_poly's sig_in_tail: past the fitted range, gpnae_tail computes the element."""
    return x < -4.0 if act == "selu" else abs(x) > (3.5 if act == "sigmoid" else 4.0)


def tail_ops(x: float, act: str) -> int:
    """gpnae_tail's cycles for one element alone, from its first state to T_DONE: each step is request, issue, unit, state update."""
    cm, ca = MUL_LAT + 2, ADD_LAT + 2
    a = x if act == "selu" else -abs(x) if act == "sigmoid" else -2 * abs(x)  # e^a, a <= 0
    if act == "tanh" and abs(x) >= 64.0:
        a = -math.inf  # x's exponent at BIAS + 6: the underflow path
    out = ca if act == "tanh" or (act == "sigmoid" and x >= 0) else 0  # 1 - 2s, 1 - s, or s as it is
    if abs(a) > 104.0:  # e^a underflows: straight to the closing steps
        return cm if act == "selu" else ca + cm + ca + cm + out
    mant, m = math.frexp(abs(a))  # |a| = mant * 2^m, mant in [0.5, 1): the Taylor z = a / 2^m and m doublings back
    z = -mant if m > 0 else a
    m = max(m, 0)
    ops = 10 * (cm + ca) + cm  # Taylor e^z - 1 from 1/11!: ten multiply-add rounds, then d = z * acc
    if act == "selu":
        return ops + m * (ca + cm) + cm  # m rounds of d = d * (d + 2), then lambda * alpha * d
    ops += ca  # E = 1 + d
    for j in range(m):  # E squared m times, cut short once E drops below 2^-63
        if z * 2 ** j < -63 * math.log(2):
            ops += 1
            break
        ops += cm
    return ops + ca + cm + ca + cm + out  # u = 1 - E, v = E u, w = 1 - v, s = E w, then the output step


def lane_cycles(xs: list, act: str, s: int) -> int:
    """Cycle of a lane's last done_o, its first G_CAP at s: groups of GROUP_K run capture, load, MAC, post and emit in turn."""
    int8, tanh, ncoef, loop = FMT == "int8", act == "tanh", degree(act) + 1, MAC_LAT[FMT]
    for g in range(0, len(xs), GROUP_K):
        grp = xs[g:g + GROUP_K]
        n = len(grp)
        # G_CAP n + 3, then G_LOAD n and G_LDRAIN to barrel_mac's RUN: the float lane loads while capturing unless tanh squares.
        if int8:
            run = s + 2 * n + MUL_LAT + 7 + (FX_LAT + 1 if tanh else 0)
        else:
            run = s + (2 * n + MUL_LAT + 7 if tanh else n + 5)
        post = run + ncoef * max(n, loop + 1) + loop + 3 + n  # RUN rounds of max(n, loop + 1), DRAIN loop + 2, EMIT n: G_POST
        if int8:  # result flag MUL_LAT (+ REQ_LAT for SELU) after the issue; sigmoid in place; tanh's saturated inputs read as unsaturated
            lat = [MUL_LAT + REQ_LAT if act == "selu" else MUL_LAT if tanh else -1] * n
        else:
            lat = [(-1 if x >= 0 else DN_LAT[FMT]) if act == "sigmoid" else MUL_LAT for x in grp]
        ready = [post + i + 2 + lat[i] for i in range(n)]
        if not int8:  # tail elements start as captured, one per cycle, on TAIL_CTX contexts; unit contention is not modelled
            free, last = [-1] * TAIL_CTX, -1
            for i, x in enumerate(grp):
                if in_tail(x, act):
                    c = free.index(min(free))
                    t0 = max(s + 4 + i, last + 1, free[c])
                    ops = tail_ops(x, act)
                    free[c], last, ready[i] = t0 + 2 + ops, t0, t0 + 3 + ops
        e = post + n - 1  # G_EMIT from post + n: one element per cycle once its result is in
        for r in ready:
            e = max(e + 1, r)
        s = e + 2  # G_NEXT, then the next group's G_CAP
    return e + 1


def lane_stage(act: str, P: dict, collapse: bool = True) -> int:
    """Activation stage cycles of a lane set (median over the streamed sets): FEED to the cycle after every lane is collected."""
    per_lane, lanes = P["SRAM_DEPTH"] // P["NUM_LANES"], P["NUM_LANES"]
    start = per_lane + 3 + (REQ_LAT if FMT == "int8" else 0)  # last_i: FEED, LATCH, the wide reads, the fill count, the requantize
    if FMT == "int8":  # no tail path: every set costs the same
        return lane_cycles([0.0] * per_lane, act, start + 1) + 2
    out = []
    for k in range(P["NUM_SETS"]):
        xs = lane_inputs(k, P, collapse)
        out.append(max(lane_cycles(xs[L * per_lane:(L + 1) * per_lane], act, start + 1) for L in range(lanes)) + 2)
    return med(out)  # done_o, lane_collected, g_done, back in G_IDLE


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
    if args.reparse:  # the saved trace; only the stimulus is regenerated, for the model's geometry
        generate_vectors({**GEOM, **cfg, "num_sets": args.sets})
        raw = open(os.path.join(args.reparse, f"{name}_N{args.n}.log"), errors="ignore").read()
    else:
        raw = run(name, args.sets, args.build_dir)
    mdl = model(cfg, not args.slices)
    if "RESULT: PASSED" not in raw or re.search(r"Assertion failed|%Error|%Fatal", raw):
        return [f"--- {name}: SIMULATION DID NOT PASS; numbers omitted", ""], [name] + ["-"] * 11 + ["FAIL"]
    a = analyse(events(raw), args.sets, cfg.get("accum_passes", 1))
    s = a["sets"]
    P = cfg.get("accum_passes", 1)
    first = next(x for x in s if x["mesh"] is not None)  # the first output's last set
    steady = statistics.mean(a["steady"]) if a["steady"] else 0
    # Each stage's own cost per set: the host's start interval; activation runs one set at a time; the mesh is
    # pipelined, so its cost is its tightest launch interval, not the time a set spends inside it. Activation and
    # pooling run once per output, so an accumulate config spreads them over its P sets.
    stage = {"host": min(a["host_gaps"] or [0]), "mesh": min(a["launch_gaps"] or [0]),
             "activation": round(med([x["act"] for x in s if x["act"] is not None]) / P), "pooling": round(med(a["pool"]) / P)}
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
          + f", pooling dispatch {mdl['pool_dispatch']} of the measured {stage['pooling']}",
          f"  Matmul rate  : {flop} {op} per set -> {flop / steady:.1f} {op}/cycle, {rate_s(steady)}"
          f" at an ASSUMED {args.clock_mhz:.0f} MHz, {100 * flop / 2 / steady / args.n ** 2:.1f}% of the mesh's"
          f" {args.n ** 2} MAC/cycle" if steady else "", ""]
    row = [name, a["first_latency"], f"{steady:.1f}", f"{flop / steady:.1f}", f"{rate(steady):.3f}" if FMT == "int8" else f"{rate(steady):.1f}",
           f"{100 * flop / 2 / steady / args.n ** 2:.0f}%", lim, stage[lim], mdl["host"], mdl["mesh"], mdl["mesh_lat"],
           first["mesh"], "pass"]
    return L, row


def header(args) -> list:
    return ["=" * 100, " SIENNA PIPELINE PERFORMANCE (measured in simulation, cycles)", "=" * 100,
            f" Generated {time.strftime('%Y-%m-%d %H:%M')}  N={args.n}  TILE={args.tile_size}  lanes={args.lanes}  "
            f"{args.sets} streamed sets per config, streaming host (one row of N words per operand per cycle)",
            " Every number is measured from TB_sienna_top's PERF trace. Per set: load = host rows, wait = to mesh launch,",
            " mesh = launch to result written (sets overlap inside it), wait = to the activation stage, activ = activation",
            " stage occupancy (a partial sum, summed in the PEs, passes as a null), pool+out = to the set's completion pulse.",
            " residual tests (int8 ADD) are not modelled and not run here", ""]


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
    ap.add_argument("--format", dest="fmt_name", default="fp32", choices=sorted(FMT_KNOBS), help="number format of the build")
    args = ap.parse_args(argv)
    perf_init()
    global MUL_LAT, ADD_LAT, FMT
    MUL_LAT, ADD_LAT = UNIT_LAT[args.fmt_name]
    FMT = args.fmt_name
    fmts = {t["name"]: t.get("formats", tuple(UNIT_LAT)) for t in PIPELINE_TESTS}
    args.configs = [c for c in args.configs if args.fmt_name in fmts[c]]  # int8-only tests run only in int8
    cfgs = {t["name"]: t for t in PIPELINE_TESTS}
    args.configs = [c for c in args.configs if not (cfgs[c].get("residual") or cfgs[c].get("residual_every"))]  # P-9: not modelled yet
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
ROUNDINGS = ("DOUBLE", "SINGLE")
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
    p["amin"], p["amax"] = mr.activation_range(act, p["out_scale"], p["out_zp"])
    return p


def reference(p, xs, rounding, folded=False, scale_product=None):
    args = (xs, p["w_q"], p["b_q"], p["in_zp"], p["w_scales"], p["in_scale"], p["out_scale"], p["out_zp"], p["amin"],
            p["amax"], rounding)
    if p["layer"] == "fc":
        return mr.fc_int8(*args, folded=folded, scale_product=scale_product).astype(np.int64)
    return mr.conv2d_int8(*args, folded=folded).astype(np.int64)


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
    mults, shifts = mr.layer_multipliers(p["layer"], p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], rounding)
    np.savez(os.path.join(out, name + ".npz"), layer=p["layer"], activation=p["activation"], in_shape=np.array(p["in_shape"]),
             w_q=p["w_q"].astype(np.int8), b_q=p["b_q"].astype(np.int32), w_scales=p["w_scales"], in_scale=p["in_scale"],
             in_zp=p["in_zp"], out_scale=p["out_scale"], out_zp=p["out_zp"], act_min=p["amin"], act_max=p["amax"],
             mults=mults, shifts=shifts, x_test=xs.astype(np.int8), y_test=ys.astype(np.int8), rounding=rounding,
             tf_version=tf.__version__)


def oracle_main(argv=None) -> None:
    """Gate G0: single-layer int8 TFLite models on the interpreter's BUILTIN_REF kernels against model_runner's reference kernels in both roundings; pins the rounding."""
    ap = argparse.ArgumentParser(description=oracle_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="oracle")
    ap.add_argument("--out", required=True, help="directory for the G4 models, npz files and rounding files")
    ap.add_argument("--report", required=True, help="G0 report (.log)")
    ap.add_argument("--seeds", type=int, default=8, help="models per kind; seed 0 is saved")
    ap.add_argument("--inputs", type=int, default=1000, help="random test inputs per model")
    a = ap.parse_args(argv)
    _tf_imports()
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
            _, shifts = mr.layer_multipliers(layer, p["w_scales"], p["in_scale"], p["out_scale"], p["w_q"].shape[0], "DOUBLE")
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


# ── pack-models (was tflite_pack_models.py) ──────────────────────────────────

PACK_MODELS = {  # name: (layer, act, inputs, outputs); the first three fit blocks of 8, the last two blocks of 16
    "fc8x8_linear": ("fc", "none", 8, 8),
    "fc8x4_relu": ("fc", "relu", 8, 4),
    "fc6x8_relu6": ("fc", "relu6", 6, 8),
    "conv3x3_6x6x1x8_linear": ("conv", "none", 1, 8),
    "fc16x16_relu": ("fc", "relu", 16, 16),
}


def pack_models_main(argv=None) -> None:
    """Five small int8 TFLite layers for packing, saved as the oracle saves its G4 models; arg: output directory."""
    ap = argparse.ArgumentParser(description=pack_models_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="pack-models")
    ap.add_argument("out", help="output directory")
    out = ap.parse_args(argv).out
    _tf_imports()
    global FC_IN, FC_OUT, CONV_CIN, CONV_COUT, CONV_HW
    os.makedirs(out, exist_ok=True)
    rounding = open(os.path.join(ROOT, "testbenches", "tflite_int8", "rounding.txt")).read().strip()
    for name, (layer, act, cin, cout) in PACK_MODELS.items():
        FC_IN, FC_OUT, CONV_CIN, CONV_COUT, CONV_HW = cin, cout, cin, cout, 6
        model, lo, hi = build_model(layer, act, 0)
        it = interpreter(model, REF)
        p = extract(it, model, layer, act)
        xs = test_inputs(p, lo, hi, 32, np.random.default_rng(2000))
        ys = invoke_all(it, xs)
        unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
        save(out, name, model, p, xs, ys, rounding)
        print(f"{name}: in zp {p['in_zp']} out zp {p['out_zp']} act [{p['amin']}, {p['amax']}] unclamped {100 * unclamped:.1f}% "
              f"y [{ys.min()}, {ys.max()}] w_scales {p['w_scales'].size}")
    shutil.copy(os.path.join(ROOT, "testbenches", "tflite_int8", "rounding.txt"), out)


# ── gpnae-tflite (was gpnae_int8_tflite.py) ──────────────────────────────────

GPTFL_OPS = (("tanh", 3, "tanh"), ("sigmoid", 2, "logistic"))  # (truth.py function, GPNAE control word, TFLite op)


def gpnae_tflite_judge(rows):
    """Report lines and pass flag: each row (func, op, quant, q, hw, tfl, ex) gated on truth.py, TFLite agreement characterization only."""
    L = ["GPNAE int8 lane (gpnae_model, bit-exact with the RTL at G2) on every int8 input, quantization from TFLite's converter (BUILTIN_REF)",
         "Gate: truth.py, the lane's words against the true function of the real input (bound "
         f"{100 * truth.BOUND:g}%, int8 floor {truth.I8_FLOOR} LSB, target {100 * truth.TARGET:g}%); TFLite agreement is characterization, not gated",
         f"{'op':<9}{'s_in':>11}{'z_in':>6}{'truth':>6}{'worst':>9}{'LSB':>4}{'>bound':>7}{'>1%':>5}"
         f"{'equal':>7}{'|d|=1':>7}{'max |d|':>8}{'TFLite-exact':>13}{'lane-exact':>11}"]
    res = {}
    for func, op, quant, q, hw, tfl, ex in rows:
        r = truth.gate("int8", func, q, hw, quant=quant)
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
    body = "; ".join(f"{f} worst {truth.worst_text(m.worst)} ({m.worst_lsb} LSB), {m.over} of {m.n} over {100 * truth.BOUND:g}%, "
                     f"{m.over_target} over {100 * truth.TARGET:g}%" for f, m in merged.items())
    L.append(f"ACCURACY: {'PASS' if ok else 'FAIL'} (true function, truth.py, int8; bound {100 * truth.BOUND:g}%, target {100 * truth.TARGET:g}%: {body})")
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
    lane = gpnae_model.Lane(gpnae_model.INT8, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", gpnae_model.coeff_file(gpnae_model.INT8))))
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
SEP_SLOTS, SEP_BOUND, SEP_TRIES, SEP_BATCH = 4, 1 << 20, 1 << 20, 1 << 16  # accumulators that separate DOUBLE from SINGLE


def separating(rng, mult: int, shift: int, zp: int, amin: int, amax: int):
    """A small accumulator (|acc| < SEP_BOUND) whose requantize differs between DOUBLE and SINGLE for this channel, or None."""
    for _ in range(SEP_TRIES // SEP_BATCH):
        a = rng.randint(-SEP_BOUND + 1, SEP_BOUND, SEP_BATCH).astype(np.int64)
        hit = np.flatnonzero(ipu.requant(a, mult, shift, zp, amin, amax, "DOUBLE") != ipu.requant(a, mult, shift, zp, amin, amax, "SINGLE"))
        if hit.size:
            return int(a[hit[0]])
    return None


def rq_vectors_main(argv=None) -> None:
    """Vectors for TB_requant_lanes through its channel map, expected values from ipu.requant in model_runner's rounding; arg: output .mem path."""
    ap = argparse.ArgumentParser(description=rq_vectors_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="rq-vectors")
    ap.add_argument("path", help="output .mem path")
    path = ap.parse_args(argv).path
    rng = np.random.RandomState(8)
    out = [RQ_SETS]
    for s in range(RQ_SETS):
        mult = rng.randint(1 << 30, 1 << 31, RQ_N).astype(np.int64)
        if s == 0:
            mult[0], mult[1] = (1 << 31) - 1, 1 << 30  # the largest multiplier and the smallest normalized one
        shift = -((s * RQ_N + np.arange(RQ_N)) % 32).astype(np.int64)  # every right shift 0..31 across the sets
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
        srng, found = np.random.RandomState(1900 + s), []
        for ch in (np.arange(RQ_N) + 5 * s) % RQ_N:
            k0 = int(ch)  # packed: lane ch reads channel ch at every beat
            a = separating(srng, int(mult[ch]), int(shift[ch]), int(zpL[k0]), int(aminL[k0]), int(amaxL[k0]))
            if a is None:
                continue
            if packed:
                b, k = len(found) % RQ_PER, int(ch) + RQ_N * (len(found) % (RQ_LANES // RQ_N))
            else:
                b, k = int(ch) % RQ_PER, 8 + 2 * len(found) + int(ch) // RQ_PER
            assert c[b, k] == ch
            acc[b, k] = a
            found.append(int(ch))
            if len(found) == SEP_SLOTS:
                break
        if not found:
            raise RuntimeError(f"set {s}: no channel has a small accumulator that separates DOUBLE from SINGLE")
        print(f"set {s}{' (packed)' if packed else ''}: DOUBLE/SINGLE separating accumulators on channels {found}")
        want = np.stack([ipu.requant(acc[:, k], mult[c[:, k]], shift[c[:, k]], int(zpL[k]), int(aminL[k]), int(amaxL[k]),
                                     mr.ROUNDING) for k in range(RQ_LANES)], axis=1)
        out += [int(packed)] + zpL.tolist() + aminL.tolist() + amaxL.tolist() + mult.tolist() + shift.tolist()
        for b in range(RQ_PER):
            out += acc[b].tolist() + [int(v) for v in want[b]]
    with open(path, "w") as f:
        f.write("".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in out))


# ── add-vectors: TB_add_lanes' vectors ────────────────────────────────────────

ADD_LANES, ADD_SETS, ADD_PER = 32, 6, 8  # TB_add_lanes' geometry: lanes, sets, beats per set
ADD_TIE = 3  # the set of halves: s1 = s2 = 2^e, s_out = 2^(e+1), so its odd sums are ties that DOUBLE and SINGLE round apart
ADD_EXP = {"none": (-8, 1), "relu": (-8, 1), "relu6": (-7, -5)}  # input scales' log2 range: relu6's 6.0 within a few input codes' sums
ADD_ZIN = {"none": 128, "relu": -32, "relu6": -32}  # input zero points below this: a ReLU set's inputs mostly positive, off its zp_o rail
ADD_ZOUT = {"none": 128, "relu": 1, "relu6": -63}  # zp_o below this, so 127 stays above a ReLU's or ReLU6's outputs


def _add_set_draw(rng, s: int) -> tuple:
    """Set s's activation, scales (s1, s2, s_out) and zero points (zp1, zp2, zp_o): s_out is 2 max(s1, s2) 2^U(-2, 1), ADD_TIE's are halves."""
    if s == ADD_TIE:
        e = int(rng.integers(-8, 1))
        return "none", [2.0 ** e, 2.0 ** e, 2.0 ** (e + 1)], [int(v) for v in rng.integers(-20, 21, 3)]
    act = ("none", "relu", "relu6")[s % 3]
    s1, s2 = (float(np.float32(2.0 ** x)) for x in rng.uniform(*ADD_EXP[act], 2))
    s_out = float(np.float32(2.0 * max(s1, s2) * 2.0 ** rng.uniform(-2, 1)))
    return act, [s1, s2, s_out], [int(v) for v in rng.integers(-128, ADD_ZIN[act], 2)] + [int(rng.integers(-128, ADD_ZOUT[act]))]


def add_vectors_main(argv=None) -> None:
    """Vectors for TB_add_lanes: per set its ADD words, then per beat a random tag and each lane's q1, q2 and expected code; arg: output .mem path."""
    ap = argparse.ArgumentParser(description=add_vectors_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="add-vectors")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(29)
    lines = []
    for s in range(ADD_SETS):
        act, sc, z = _add_set_draw(rng, s)
        add = mr.add_words(sc[0], z[0], sc[1], z[1], sc[2], z[2], act)
        lines.append(" ".join(f"{add[k] & 0xFFFFFFFF:08x}" for k in mr.ADD_FIELDS))
        q1, q2 = rng.integers(-128, 128, (2, ADD_PER, ADD_LANES))
        want = mr.tflite_add_codes(q1, q2, add)
        off = np.mean((want != add["amin"]) & (want != add["amax"]))
        assert off >= 0.5, f"add-vectors: set {s} ({act}) has {off:.0%} of its outputs off the clamp rails, under half"
        for b in range(ADD_PER):
            lines.append(f"{int(rng.integers(0, 1 << 16)):04x} " + " ".join(f"{int(x) & 0xFF:02x}{int(y) & 0xFF:02x}{int(w) & 0xFF:02x}"
                                                                         for x, y, w in zip(q1[b], q2[b], want[b])))
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {ADD_SETS} sets x {ADD_PER} beats x {ADD_LANES} lanes, rounding {mr.ROUNDING}")


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
    """(exit, output) of the shell make -n prints for pkg-check GEN_PKG=0 on pkg."""
    r = make_n("pkg-check", "GEN_PKG=0", f"PKG_FILE={pkg}", *args)
    assert r.returncode == 0, r.stderr
    g = subprocess.run(["bash", "-c", r.stdout], capture_output=True, text=True)
    return g.returncode, g.stdout + g.stderr


@selftest
def test_guard_passes_matching_package():
    with tempfile.TemporaryDirectory() as d:
        for f in reg.FORMATS:
            rc, out = pkg_guard(fixture_pkg(d, f), f"FMT={f}")
            assert rc == 0 and f"matches FMT={f} N=16 TILE=4 LANES=32" in out, out


@selftest
def test_guard_rejects_other_format():
    with tempfile.TemporaryDirectory() as d:
        for f in reg.FORMATS:
            for want in reg.FORMATS:
                if want != f:
                    rc, out = pkg_guard(fixture_pkg(d, f), f"FMT={want}")
                    assert rc != 0 and f"but FMT={want} needs" in out, (f, want, out)


@selftest
def test_guard_rejects_pre_format_package():
    with tempfile.TemporaryDirectory() as d:
        rc, out = pkg_guard(fixture_pkg(d, "fp32", drop=("EXP_W", "MAN_W", "IS_INT")), "FMT=fp32")
        assert rc != 0 and "EXP_W=missing" in out, out


@selftest
def test_guard_rejects_bf16_sum_package():
    with tempfile.TemporaryDirectory() as d:  # a bf16 package from before fp32 sums: ACC_W 16 and no OUT_W
        rc, out = pkg_guard(fixture_pkg(d, "bf16", drop=("OUT_W",), edit={"ACC_W": 16}), "FMT=bf16")
        assert rc != 0 and "ACC_W=16 OUT_W=missing, but FMT=bf16 sums in 32 bits and gives 16-bit results" in out, out
        for f, o in (("fp32", 32), ("bf16", 16), ("int8", 32)):
            rc, out = pkg_guard(fixture_pkg(d, f), f"FMT={f}")
            assert rc == 0 and f"ACC_W=32 OUT_W={o}" in out, (f, out)


@selftest
def test_guard_checks_geometry():
    with tempfile.TemporaryDirectory() as d:
        big = fixture_pkg(d, "bf16", 32, 8)
        rc, out = pkg_guard(big, "FMT=bf16")  # a stale same-format package of another size
        assert rc != 0 and "N=32 TILE_SIZE=8 NUM_LANES=32, but the build asks for N=16 TILE=4 LANES=32" in out, out
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
    src = open(os.path.join(ROOT, "testbenches", "TB_sienna_top.sv")).read()
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
    """_golden_int8_residual against an element-by-element composition (scalar ipu on each code), and it moves with zp1, the ADD and zp_o."""
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
            assert int(Y[i, j]) == int(ipu.tflite_add(int(q1[i, j]), int(R[i, j]), *[add[k] for k in mr.ADD_FIELDS], mr.ROUNDING))
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
    assert all(n == 18 for k, (n, _, _) in layouts.items() if k != "int8_residual_sigmoid_selu_nopool"), layouts


@selftest
def test_residual_pairs_check():
    """run_regression's residual counts: per stream pass a hold set done per add set below num_sets, per_lane add beats each, none for
    another set, a missing line fails; and once every residual test ran, some hold done must have met an L3 grant or a bank done."""
    with _scratch_tb_dir() as d:
        open(os.path.join(d, "test_knobs.mem"), "w").write("num_sets 4\n")
        for k, f in enumerate((0x54, 0x58, 0x54, 0x58, 0x58)):  # add_4.mem lies past num_sets: a longer earlier test's
            _write_w32(os.path.join(d, f"add_{k}.mem"), [f] + [0] * 11)
        line = lambda h, b, o=0: (f"[HOLD/ADD] hold sets done {h}, add beats {b}, add beats of other sets {o}, hold release with an L3 grant 0, "
                                  "hold done with a bank done 0, grants held for an add drain 0\n")
        assert _residual_pairs_bad(line(2, 16) * 4, 8) == ""
        assert "expected 2, 16 and 0" in _residual_pairs_bad(line(0, 0) * 4, 8)
        assert _residual_pairs_bad(line(2, 16) * 3 + line(1, 16), 8) and _residual_pairs_bad(line(2, 15) * 4, 8)
        assert _residual_pairs_bad(line(2, 16, 1) * 4, 8) and "no [HOLD/ADD] line" in _residual_pairs_bad("RESULT: PASSED\n", 8)
    every = _residual_tests()
    assert _residual_cross_bad(every, 0).startswith("[FAIL]") and _residual_cross_bad(every, 1) == "" and _residual_cross_bad(every[:-1], 0) == ""
    assert _residual_cross_bad(["matmul_relu_nopool"], 0, every=[]) == ""  # no residual test defined: nothing to gate


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
        assert rc != 0 and "COLLAPSE_K=0, but the build asks for COLLAPSE_K=1" in out, out
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
    assert r.returncode != 0 and "Invalid FMT=fp16: must be one of fp32, bf16, int8" in r.stderr, r.stderr
    for t in ("regression", "pack", "gemm", "model", "tflite", "perf-analysis"):
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode != 0 and "COLLAPSE_K=0 reaches TB_sienna_top only" in r.stderr, (t, r.stderr)
    for t in ("pipeline", "verilator", "lint", "pkg", "sim", "build-key"):  # TB_sienna_top sets sienna_top's COLLAPSE_K from the package
        r = make_n(t, "COLLAPSE_K=0")
        assert "COLLAPSE_K=0 reaches" not in r.stderr, (t, r.stderr)
    r = make_n("lint", "COLLAPSE_K=2")
    assert r.returncode != 0 and "Invalid COLLAPSE_K=2" in r.stderr, r.stderr
    for t in ("help", "sm-verilator"):  # the only targets collapse-k 0 is meaningful for, or harmless in
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode == 0, (t, r.stderr)


@selftest
def test_pack_target():
    r = make_n("pack", "FMT=bf16", "N=32", "TILE=8", "LANES=64", "PYTHON=py")
    assert r.returncode == 0 and r.stdout.split() == "py regression.py --action pack --format bf16 --n 32 --tile 8 --lanes 64".split(), r.stdout + r.stderr
    r = make_n("pack", "FMT=fp16")
    assert r.returncode != 0 and "Invalid FMT=fp16" in r.stderr, r.stderr


@selftest
def test_regression_points():
    small, full = SPEC.points(True, list(FORMATS)), SPEC.points(False, list(FORMATS))
    sweep = lambda pts: sorted({(SPEC.parse(p)["n"], SPEC.parse(p)["tile"]) for p in pts if p != "selftest"})
    assert sweep(small) == [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8), (16, 16)], sweep(small)
    assert sweep(full) == sweep(small) + [(32, t) for t in (2, 4, 8, 16, 32)] + [(64, t) for t in (2, 4, 8, 16, 32, 64)], sweep(full)
    assert small[0] == full[0] == "selftest" and "selftest" not in small[1:]  # format-independent: once
    assert len(small) == len(set(small)) == 1 + 7 * 21 and len(full) == len(set(full)) == 1 + 18 * 12 + 7 * 9 + 11 * 3 == 313, (len(small), len(full))
    for f, kinds in (("fp32", ["pipeline", "model", "pack", "gemm"]), ("bf16", ["pipeline", "model", "pack", "gemm"]), ("int8", ["pipeline", "pack", "gemm", "tflite"])):
        for n, t in sweep(full):  # back-to-back at every N, collapse-k 0 at N = 8 and 16; packing needs collapse-k 1
            want = kinds[:1] + (["pipeline-ck0", "pipeline-b2b", "pipeline-ck0-b2b"] if n <= 16 else ["pipeline-b2b"]) + kinds[1:]
            assert [p.rsplit("/", 1)[1] for p in full if p.startswith(f"{f}/N{n}_T{t}/")] == want, (f, n, t)
    assert SPEC.parse("bf16/N16_T2/pipeline-ck0")["ck"] == 0 and SPEC.parse("bf16/N16_T2/pipeline")["ck"] == 1
    assert [(SPEC.parse(f"int8/N8_T2/{k}")["ck"], SPEC.parse(f"int8/N8_T2/{k}")["b2b"]) for k in ("pipeline", "pipeline-ck0", "pipeline-b2b", "pipeline-ck0-b2b")] \
        == [(1, False), (0, False), (1, True), (0, True)]
    assert [SPEC.parse(p)[x] for p in ("fp32/N64_T64/pipeline-b2b",) for x in ("ck", "b2b", "lanes")] == [1, True, 128]
    assert SPEC.points(True, ["int8"]) == ["selftest"] + [p for p in small if p.startswith("int8/")] and SPEC.points(False, []) == []
    assert [SPEC.parse(p)["lanes"] for p in ("fp32/N32_T8/pipeline", "fp32/N64_T8/pipeline", "fp32/N8_T2/model")] == [32, 128, 32]
    for bad in ("fp32/N8_T16/pipeline", "int8/N8_T2/model", "fp32/N16_T4/tflite", "fp16/N8_T2/pack", "fp32/N8_T2/selftest", "fp32/pipeline",
                "fp32/N12_T4/pack", "fp32/N8_T2/mesh", "fp32/gpnae-verilator", "fp32/aril-fpu", "fp32/N32_T8/pipeline-ck0",
                "fp32/N8_T2/pack-ck0", "int8/N8_T2/tflite-ck0", "fp32/N8_T2/model-ck0", "fp32/N32_T8/pipeline-ck0-b2b", "bf16/N64_T8/pipeline-ck0-b2b",
                "fp32/N8_T2/pack-b2b", "fp32/N8_T2/pipeline-b2b-ck0"):
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
    saved = {k: os.environ.get(k) for k in ("GPNAE_MODEL", "TEST", "N", "LINK_STAGES")}
    os.environ.update(GPNAE_MODEL="hw", TEST="selu", N="32", LINK_STAGES="1")
    try:
        env = regress.point_env(SPEC, "fp32/N32_T2/pipeline", "s -j4 --jobserver-auth=3,4", 8)
    finally:
        for k, v in saved.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert env["TRACE"] == "0" and env["OPT_FAST"] == "-O0" and env["SIM_THREADS"] == "2" and not {"GPNAE_MODEL", "TEST", "N", "LINK_STAGES"} & set(env)
    pe = lambda n: regress.point_env(SPEC, n, "s", 8).get("OPT_FAST")
    assert [pe(f"fp32/N{n}_T2/{k}") for n, k in ((8, "pack"), (16, "pack"), (8, "gemm"), (16, "gemm"))] == ["-O0"] * 4  # the short pack and gemm simulations build at -O0 at every N
    assert pe("fp32/N8_T2/pipeline") is None and pe("fp32/N16_T4/pipeline") is None and pe("int8/N8_T2/tflite") is None and pe("bf16/N16_T4/model") is None and pe("selftest") is None  # the long ones keep -Os below N = 32
    assert pe("fp32/N32_T2/pipeline") == pe("fp32/N32_T2/model") == "-O0"  # from N = 32 every point builds at -O0 (point_env)
    for n, t in ((8, 2), (16, 4), (32, 2), (64, 16)):  # a -b2b point builds and runs as its plain pipeline point, plus the define
        plain, b2b = f"bf16/N{n}_T{t}/pipeline", f"bf16/N{n}_T{t}/pipeline-b2b"
        assert pe(b2b) == pe(plain) == ("-O0" if n >= 32 else None), (n, pe(b2b), pe(plain))
        assert SPEC.argv(b2b, opts) == SPEC.argv(plain, opts) + ["EXTRA_FLAGS=-DBACK_TO_BACK"] and not any("LARGE_CODE" in a for a in SPEC.argv(b2b, opts)), n
    cores = regress._cores()
    for args in ((), ("-C", "SystolicMesh"), ("-C", "GPNAE")):
        assert f"-j {cores} --build-jobs {cores} " in make_n(*args, "verilator").stdout, args  # Verilator takes every core in all three repos
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
            ("bf16/N8_T2/pack", "PACK REGRESSION PASS: 0 mismatching outputs\n", 0, "PASS"), ("bf16/N8_T2/pack", "done\n", 0, "FAIL (no 'PACK REGRESSION PASS: 0 mismatching outputs')"),
            ("int8/N8_T2/gemm", "grid_1x64x16  1 64 16\nGEMM int8: 9 cases, 0 with outputs that differ from the bit-exact model\n", 0, "PASS"),
            ("int8/N8_T2/gemm", "grid_1x64x16  1 64 16\n", 0, "FAIL (no 'GEMM int8: \\d+ cases, 0 with outputs that differ from the bit-exact model')"),
            ("fp32/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"layer_{x}_bias  64 48 40 240 7633  max err 9.3e-04 of the output range\n" for x in ("tanh", "sigmoid", "selu")), 0, "PASS"),
            ("fp32/N8_T2/gemm", "grid_1x8x8 1\nlayer_tanh_bias  64 max err 9.3e-04 of the output range\n", 0, "FAIL (no 'layer_sigmoid_bias .* max err ')"),
            ("bf16/N8_T2/gemm", "grid_1x8x8 1\n" + "".join(f"  layer_{x}: 0 outputs differ from the bit-exact model\n" for x in ("tanh", "sigmoid", "selu"))
             + "layer_bias_special  32 40 24 12 308  0 outputs differ from the bit-exact model\n", 0, "PASS"),
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
def test_model_points_need_model_dir():
    with tempfile.TemporaryDirectory() as d:
        o = lambda md: type("O", (), {"model_dir": md})()
        assert SPEC.not_run("fp32/N8_T2/pipeline", o("")) is None and SPEC.not_run("fp32/N8_T2/model", o(d)).startswith(f"MODEL_DIR={d} is missing pretrainedResnet.tflite")
        assert "MODEL_DIR unset" in SPEC.not_run("bf16/N8_T2/model", o("")) and "does not exist" in SPEC.not_run("bf16/N8_T2/model", o(d + "/x"))
        assert all("make regression MODEL_DIR=<directory with pretrainedResnet.tflite, ad01_fp32.tflite" in SPEC.not_run("fp32/N8_T2/model", o(x)) for x in ("", d))
        for m in mr.MODELS.values():
            open(os.path.join(d, m), "w").write("")
        assert SPEC.not_run("fp32/N8_T2/model", o(d)) is None


@selftest
def test_tb_knobs_are_run_time():
    # TB_sienna_top reads the TB-only settings from test_knobs.mem, so no knob value can reach its elaboration or the DUT's
    src = open(os.path.join(TB_DIR, "TB_sienna_top.sv")).read()
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
    rec = mk[mk.index("\nverilator: pkg-check"):mk.index("\n\n", mk.index("\nverilator: pkg-check") + 2)]  # make verilator: the key goes only after a whole build
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
        assert names == pts, (args, sorted(set(names) ^ set(pts))[:5])
        assert all("--results " in x and "--model-dir=/m" in x and "--run " in x and x.rstrip().endswith("--keep-going") for x in lines if "--action point " in x), lines[:2]
        assert "--action summary" in lines[-1] and "--run " in lines[-1] and ("--small" in lines[-1]) == ("SMALL=1" in args), lines[-1]
        assert lines[-1].split("--formats ")[1].split()[0] == ("int8" if "FMT=int8" in args else "fp32,bf16,int8"), lines[-1]
    small = [x.split("--name ")[1].split()[0] for x in recipe("regression", "SMALL=1", "MODEL_DIR=/m") if "--action point " in x]
    assert [p for p in small if p.endswith("-ck0")] == [f"{f}/N{n}_T{t}/pipeline-ck0" for f in FORMATS for n in (8, 16) for t in pow2_tile_sizes(n)]
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
    assert r.returncode != 0 and "Invalid FMT=fp16" in r.stderr, r.stderr
    r = make_n("regression", "SMALL=2")
    assert r.returncode != 0 and "Invalid SMALL=2" in r.stderr, r.stderr
    assert recipe("model", "MODEL_DIR=/m")[-1].split()[-2:] == ["--count", "1"] and recipe("model", "MODEL_DIR=/m", "MODEL_COUNT=3")[-1].split()[-2:] == ["--count", "3"]
    assert make_n("check").returncode != 0  # the gate's old name is gone
    h = subprocess.run(["make", "-s", "help"], cwd=ROOT, env=MAKE_ENV, capture_output=True, text=True).stdout
    assert "make sim " in h and "make build-key " in h and "sienna_ck0" not in open(os.path.join(ROOT, "Makefile")).read(), h[-300:]
    with tempfile.TemporaryDirectory() as d:  # a relative MODEL_DIR reaches the points absolute: they run in a private copy
        rel = os.path.relpath(d, ROOT)
        lines = recipe("regression", "SMALL=1", f"MODEL_DIR={rel}")
        assert all(f"--model-dir={os.path.realpath(d)}" in x.replace(d, os.path.realpath(d)) for x in lines if "--action " in x), lines[0]
    env = dict(MAKE_ENV, FMT="bf16", SMALL="1")  # exported, FMT and SMALL do not narrow it: only make's command line does
    r = subprocess.run(["make", "-s", "-n", "regression", py], cwd=ROOT, env=env, capture_output=True, text=True)
    names = [x.split("--name ")[1].split()[0] for x in r.stdout.splitlines() if "--action point " in x]
    assert r.returncode == 0 and names == SPEC.points(False, list(FORMATS)), (r.stderr[-200:], len(names))


@selftest
def test_gpnae_target_model():
    r = make_n("gpnae-verilator", "FMT=bf16", "PYTHON=py")
    assert r.returncode == 0 and "py regression.py --lane poly --format bf16 --model exact" in r.stdout, r.stdout + r.stderr
    r = make_n("gpnae-verilator", "FMT=bf16", "GPNAE_MODEL=hw", "PYTHON=py")
    assert r.returncode == 0 and "--model hw" in r.stdout, r.stdout + r.stderr


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
def test_fmt_fields_match_formats():
    mk = open(os.path.join(ROOT, "Makefile")).read()
    fields = {f: tuple(int(x) for x in v.split()) for f, v in re.findall(r"^FMT_FIELDS_(\w+) = ([\d ]+)$", mk, re.M)}
    assert re.search(r"^FORMATS = (.*)$", mk, re.M).group(1).split() == list(reg.FORMATS)
    pkg_of = lambda f: {k: v for k, v, _ in reg._config_items({}, f, "relu", 1, 15, 1, [], False, 1)}  # what the package writes
    assert fields == {f: tuple(pkg_of(f)[k] for k in ("EXP_W", "MAN_W", "IS_INT", "ACC_W", "OUT_W")) for f in reg.FORMATS}, fields
    assert fields == {f: (*reg.FORMATS[f], int(f == "int8"), 32, 32 if f == "int8" else 1 + sum(reg.FORMATS[f])) for f in reg.FORMATS}, fields


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
    # Task 1.16: the int8 lane's SELU is exact past lambda * x = 512, so a large-scale SELU entry packs and runs; its output shift is <= 0.
    acc = np.arange(-2000, 2000, dtype=np.int64).reshape(-1, 2) * 500  # real inputs to +-1000: s_out about 7.8, lambda * x to 1051
    req = mr.requant_params(acc, 1e-3, np.ones(2), "selu")
    x = (mr.requantize(acc, req) - req["zp"]) * req["s_out"]
    assert x.max() * gpnae_model.LAMBDA_F > 512 and req["shout"] <= 0 and (req["mout"], req["shout"]) == \
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
    assert pa.UNIT_LAT == {"fp32": (8, 5), "bf16": (3, 5), "int8": (1, 1)}  # sienna_fmt_pkg's mul_lat, add_lat, as parsed
    assert (pa.FX_LAT, pa.REQ_LAT, pa.GROUP_K, pa.TAIL_CTX) == (2, 4, 16, 4)  # fx_lat, req_lat, gpnae_poly's K, TAIL_CONTEXTS


@perf_selftest
def test_lane_stage_int8():
    with perf_fmt("int8"):  # no tail path, so the stage is data-free; each value derived from gpnae_poly_int8 and barrel_mac, and measured
        stage = lambda act, n: pa.lane_stage(act, {"SRAM_DEPTH": n * n, "NUM_LANES": 32})
        assert stage("selu", 16) == 96 and stage("tanh", 16) == 107  # sp_p16t2
        assert stage("selu", 32) == 327 and stage("tanh", 32) == 365  # sp_p32t2, sp_p32t4
        assert stage("selu", 8) == 53 and stage("tanh", 8) == 56  # sp_p8t2


@perf_selftest
def test_lane_group_float():
    with perf_fmt("fp32"):  # one group, no tail element: last_i at per_lane + 3, so the first G_CAP at per_lane + 4
        assert pa.lane_cycles([0.0] * 8, "selu", 12) + 2 == 195 and pa.lane_cycles([0.0] * 2, "tanh", 6) + 2 == 183
    with perf_fmt("bf16"):
        assert pa.lane_cycles([0.0] * 8, "selu", 12) + 2 == 143 and pa.lane_cycles([0.0] * 32, "tanh", 36) + 2 == 529


@perf_selftest
def test_tail_ops_fp32():
    with perf_fmt("fp32"):  # multiply 8 + 2, add 5 + 2 cycles per gpnae_tail step, by hand
        assert pa.tail_ops(5.0, "tanh") == 10 * 17 + 10 + 7 + 4 * 10 + 7 + 10 + 7 + 10 + 7  # a = -10: four squarings
        assert pa.tail_ops(-5.0, "selu") == 10 * 17 + 10 + 3 * 17 + 10  # a = -5: three doublings
        assert pa.tail_ops(60.0, "tanh") == 7 + 10 + 7 + 10 + 7  # |a| = 120 > 104: e^a underflows
        assert pa.tail_ops(-4.5, "sigmoid") == 10 * 17 + 10 + 7 + 3 * 10 + 7 + 10 + 7 + 10  # x < 0 keeps s, no output step
        assert pa.lane_cycles([5.0] + [0.0] * 7, "tanh", 12) + 2 == 16 + 3 + 268 + 8 + 2  # the tail result holds the group's emit


# ── tool: TFLite reference kernels (was test_tflite_ref.py) ──

TFLITE_QM = [  # real -> (mult, shift): frexp, then the mantissa * 2^31 rounded half away from zero
    (0.5, (1 << 30, 0)),
    (1.0, (1 << 30, 1)),
    (0.75, (1610612736, 0)),
    (0.1, (1717986918, -3)),  # 0.8 * 2^31 = 1717986918.4
    (2.0 ** -32, (1 << 30, -31)),  # shift -31 is kept
    (2.0 ** -33, (0, 0)),  # shift -32 flushes to zero
    (0.0, (0, 0)),
    (1.0 - 2.0 ** -40, (1 << 30, 1)),  # the mantissa rounds to 2^31: halved, shift + 1
    (1.0 / 255.0, (1077952576, -7)),
    (0.5 + 2.0 ** -32, ((1 << 30) + 1, 0)),  # 2^30 + 0.5 rounds away from zero; numpy.round would give 2^30
]


@selftest
def test_tflite_ref():
    fails = checks = 0

    def check(what, got, want):
        nonlocal fails, checks
        checks += 1
        g, w = np.asarray(got), np.asarray(want)
        if g.shape != w.shape or np.any(g != w):
            fails += 1
            print(f"[FAIL] {what}: got {g.tolist()}, want {w.tolist()}")

    ref = mr
    for real, want in TFLITE_QM:
        check(f"quantize_multiplier({real!r})", ref.quantize_multiplier(real), want)
    check("quantize_multiplier(2^31, SINGLE)", ref.quantize_multiplier(2.0 ** 31, "SINGLE"), ((1 << 31) - 1, 30))
    check("quantize_multiplier(2^31, DOUBLE)", ref.quantize_multiplier(2.0 ** 31, "DOUBLE"), (1 << 30, 32))
    check("relu6 range", ref.activation_range("relu6", 0.05, -128), (-128, -8))  # 6 / 0.05 = 120 levels
    check("relu range", ref.activation_range("relu", 0.1, 3), (3, 127))
    check("relu6 above int8", ref.activation_range("relu6", 0.02, -10), (-10, 127))  # 300 levels: capped
    check("none range", ref.activation_range("none", 0.1, 3), (-128, 127))

    # FC: acc = 3 * (10 + 2) - 4 * (-20 + 2) + b = 108 + b; scale 0.5 * 0.25 / 1.0 = 2^-3 (QM (2^30, -2)); out zp 3.
    x, w = np.array([[10, -20]]), np.array([[3, -4]])
    fc_cases = [(100, 29, 29),  # 208 / 8 = 26
                (104, 30, 30),  # 212 / 8 = 26.5: a positive tie, both up
                (-176, -6, -5)]  # -68 / 8 = -8.5: DOUBLE away (-9), SINGLE up (-8)
    for b, dbl, sgl in fc_cases:
        for r, want in (("DOUBLE", dbl), ("SINGLE", sgl)):
            for folded in (False, True):
                check(f"fc b={b} {r} folded={folded}",
                      ref.fc_int8(x, w, np.array([b]), -2, [0.25], 0.5, 1.0, 3, -128, 127, r, folded=folded), [[want]])

    # Conv 3x3 SAME on 2x2: in-image taps of (x - 1) give [[49, 43], [31, 25]]; QM (2^30, 0) makes each a positive tie; out zp -3.
    xc = np.array([1, 2, 3, 4]).reshape(1, 2, 2, 1)
    wc = np.arange(1, 10).reshape(1, 3, 3, 1)
    for r in ("DOUBLE", "SINGLE"):
        for folded in (False, True):
            check(f"conv {r} folded={folded}",
                  ref.conv2d_int8(xc, wc, np.array([0]), 1, [1.0], 0.5, 1.0, -3, -128, 127, r, folded=folded),
                  np.array([22, 19, 13, 10]).reshape(1, 2, 2, 1))
    check("fold_input_zp", ref.fold_input_zp(np.array([0]), np.ones((1, 3, 3, 1)), 1), [-9])
    check("im2col_same corner", ref.im2col_same(xc, 3, 3, 1)[0], [1, 1, 1, 1, 1, 2, 1, 3, 4])  # pad value 1 outside
    print(f"test_tflite_ref: {checks} checks, {fails} failures")
    print(f"RESULT: {'PASSED' if fails == 0 else 'FAILED'}")
    assert fails == 0, f"{fails} of {checks} checks failed"


# ── gpnae-tflite's gate (no TensorFlow: GPNAE's own case scales stand in for the converter's) ──

def gptfl_rows(ladder=True, tfl=None):
    """gpnae_tflite_judge's rows for the lane on INT8_CASES' tanh and sigmoid; tfl(hw, ex) gives the TFLite words (default ex)."""
    rom = gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", gpnae_model.coeff_file(gpnae_model.INT8)))
    lane, q, rows = gpnae_model.Lane(gpnae_model.INT8, rom, ladder=ladder), np.arange(-128, 128, dtype=np.int64), []
    for func, code, op in GPTFL_OPS:
        for c in gpnae_model.INT8_CASES[func]:
            hw, ex = lane.run(q, code, gpnae_model.int8_params(c, code)), gpnae_model.exact_int8(q, code, c)
            rows.append((func, op, (c.s_in, c.z_in, c.s_out, c.z_out), q, hw, ex if tfl is None else tfl(hw, ex), ex))
    return rows


@selftest
def test_gptfl_gate():
    L, ok = gpnae_tflite_judge(gptfl_rows())
    assert ok and L[-2].startswith("ACCURACY: PASS (true function, truth.py") and "TFLITE AGREEMENT (characterization, not gated)" in L[-1], L[-3:]
    L, ok = gpnae_tflite_judge(gptfl_rows(ladder=False))  # the 2a lane: sigmoid saturated past 3.5
    assert not ok and L[-2].startswith("ACCURACY: FAIL") and "sigmoid worst 100.000% (7 LSB), 34 of 1280 over 10%" in L[-2], L[-3:]
    L, ok = gpnae_tflite_judge(gptfl_rows(tfl=lambda hw, ex: 127 - hw))  # TFLite far off: characterization only, the verdict stands
    assert ok and L[-2].startswith("ACCURACY: PASS") and "tanh 0 of 1280 equal" in L[-1], L[-2:]
    off = [r[:4] + (np.clip(r[4] + 3 * (r[0] == "tanh"), -128, 127),) + r[5:] for r in gptfl_rows()]  # tanh 3 LSB high
    L, ok = gpnae_tflite_judge(off)
    assert not ok and "TRUTH int8 tanh: FAIL" in "\n".join(L) and "TRUTH int8 sigmoid: PASS" in "\n".join(L), L[-4:]
    L, ok = gpnae_tflite_judge([])
    assert not ok and L[-1] == "ACCURACY: FAIL (no gated case)"  # nothing gated is not a pass


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
KINDS = ("pipeline", "pipeline-ck0", "pipeline-b2b", "pipeline-ck0-b2b", "model", "pack", "gemm", "tflite")  # per N, tile size and format; model in fp32 and bf16, tflite in int8
PIPELINE_KINDS = KINDS[:4]  # TB_sienna_top's sweep: collapse-k 1 or 0 (-ck0), built plain or with -DBACK_TO_BACK (-b2b)
CK0_SIZES = (8, 16)  # pipeline-ck0: the pipeline on the collapse-k 0 mesh, at the small sizes
B2B_SIZES = regress.SIZES  # pipeline-b2b at every N (pipeline-ck0-b2b also takes CK0_SIZES): the pipeline built with -DBACK_TO_BACK
MODEL_COUNT = 3  # inferences per model and point, as the model sweeps run them


def reg_lanes(n: int) -> int:
    """Activation lanes of a point, as the sweeps build them: 32 up to N = 32, 128 at N = 64."""
    return 128 if n >= 64 else 32


class SiennaSpec(regress.Spec):
    title = "SIENNA REGRESSION"
    formats = tuple(FORMATS)
    sized = True
    drop = ("GPNAE_MODEL", "TEST", "QUICK", "COLLAPSE_K", "N", "TILE", "LANES", "MODEL_DIR", "MODEL_COUNT", "GEN_PKG", "LINK_STAGES")

    def points(self, small, formats):
        if not formats:
            return []
        return ["selftest"] + [f"{f}/N{n}_T{t}/{k}" for f in formats for n in (regress.SIZES_SMALL if small else regress.SIZES)
                               for t in pow2_tile_sizes(n) for k in KINDS if self._offered(f, n, k)]

    @staticmethod
    def _offered(f, n, k):
        """model in fp32 and bf16, tflite in int8, the -ck0 kinds at CK0_SIZES, the -b2b kinds at B2B_SIZES."""
        return ((k != "model" or f != "int8") and (k != "tflite" or f == "int8") and ("-ck0" not in k or n in CK0_SIZES)
                and (not k.endswith("-b2b") or n in B2B_SIZES))

    def parse(self, name):
        if name == "selftest":
            return {"n": 0, "kind": "selftest"}
        m = re.fullmatch(r"(\w+)/N(\d+)_T(\d+)/([a-z0-9-]+)", name)
        if not m or m.group(1) not in FORMATS or m.group(4) not in KINDS:
            raise ValueError(f"{name}: points are selftest and <fmt>/N<n>_T<t>/<kind>, kind one of {', '.join(KINDS)}")
        f, n, t, k = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if n not in regress.SIZES or t not in pow2_tile_sizes(n):
            raise ValueError(f"{name}: N is 8, 16, 32 or 64 and the tile size a power-of-2 divisor of N")
        if not self._offered(f, n, k):
            raise ValueError(f"{name}: model runs in fp32 and bf16, tflite in int8, the -ck0 kinds at N = {', '.join(map(str, CK0_SIZES))}"
                             f" and the -b2b kinds at N = {', '.join(map(str, B2B_SIZES))}")
        return {"fmt": f, "n": n, "tile": t, "lanes": reg_lanes(n), "kind": k, "ck": 0 if "-ck0" in k else 1, "b2b": k.endswith("-b2b")}

    def env(self, name, env):
        if self.parse(name)["kind"] in ("pack", "gemm"):  # -O0 builds these ~30% faster at N = 8 and 16 (rs1_spd_sienna); pipeline, tflite and model simulations are long enough to lose more (pipeline N = 16 +40-125%)
            env = dict(env, OPT_FAST="-O0")
        return env

    def argv(self, name, opts):
        p, py = self.parse(name), sys.executable
        if p["kind"] == "selftest":
            return [py, "regression.py", "--action", "selftest"]
        k = "pipeline" if p["kind"] in PIPELINE_KINDS else p["kind"]
        mk = ["make", k, f"FMT={p['fmt']}", f"PYTHON={py}", f"N={p['n']}", f"TILE={p['tile']}", f"LANES={p['lanes']}"]
        return mk + {"model": [f"MODEL_DIR={getattr(opts, 'model_dir', '')}", f"MODEL_COUNT={MODEL_COUNT}"], "gemm": ["QUICK=1"]}.get(k, []) \
            + (["COLLAPSE_K=0"] if p["ck"] == 0 else []) + (["EXTRA_FLAGS=-DBACK_TO_BACK"] if p["b2b"] else [])

    def verdict(self, name, text, rc):
        k = self.parse(name)["kind"]
        v = regress.common_verdict(text, rc)
        if v:
            return v
        text = regress.ANSI.sub("", text)
        if k == "selftest":
            return regress.marker_verdict(text, rc, (r"^ALL \d+ PASSED$",))
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
        if k in ("model", "tflite"):
            return regress.marker_verdict(text, rc, (r"^MODEL GATE: PASS",), (r"^MODEL GATE: FAIL",))
        if k == "pack":
            return regress.marker_verdict(text, rc, (r"^PACK REGRESSION PASS: 0 mismatching outputs$",))
        acts = ("tanh", "sigmoid", "selu")  # gemm: its rows, every activation's own check, and no FAIL line
        need = {"fp32": [rf"^layer_{x}_bias .* max err " for x in acts],
                "bf16": [rf"^  layer_{x}: 0 outputs differ from the bit-exact model$" for x in acts] + [r"^layer_bias_special .* 0 outputs differ from the bit-exact model$"],
                "int8": [r"^GEMM int8: \d+ cases, 0 with outputs that differ from the bit-exact model$"]}[self.parse(name)["fmt"]]
        return regress.marker_verdict(text, rc, [r"^grid_\d+x\d+x\d+ "] + need, (r"^FAIL ",))

    def not_run(self, name, opts):
        if self.parse(name)["kind"] != "model":
            return None
        d, need = getattr(opts, "model_dir", ""), list(mr.MODELS.values())
        how = f"make regression MODEL_DIR=<directory with {', '.join(need)}>"
        if not d:
            return f"MODEL_DIR unset: the model points need {how}"
        if not os.path.isdir(d):
            return f"MODEL_DIR={d} does not exist: {how}"
        missing = [m for m in need if not os.path.isfile(os.path.join(d, m))]
        return f"MODEL_DIR={d} is missing {', '.join(missing)}: {how}" if missing else None


SPEC = SiennaSpec(ROOT, os.path.join(ROOT, "testbenches", "results", "regression"))


MAINS = {"pack": pack_main, "gemm": gemm_main, "perf": perf_main, "oracle": oracle_main, "pack-models": pack_models_main,
         "gpnae-tflite": gpnae_tflite_main, "rq-vectors": rq_vectors_main, "add-vectors": add_vectors_main, "selftest": selftest_main,
         **{x: (lambda argv, x=x: regress.main(SPEC, ["--action", x, *argv])) for x in ("points", "point", "summary")}}  # each parses its own options (regression.py --action A --help)

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
