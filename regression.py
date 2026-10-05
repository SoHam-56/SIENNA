#!/usr/bin/env python3
"""
regression.py — SIENNA Full Pipeline Unified Regression Suite
=============================================================
Contains:
  1. Golden Model & Vector Generator
  2. Live Status Streamer
  3. Formatted Matrix Trace Dumper
  4. Regression Orchestrator & Scoreboard
  5. Hardware checks behind --action: pack, gemm, perf, oracle, pack-models, gpnae-tflite, rq-vectors
  6. Tool self-tests (--action selftest) and the one-verdict gate (--action all, make check)
"""

import argparse
import contextlib
import functools
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
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SystolicMesh"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GPNAE"))
import mesh_model  # noqa: E402
import gpnae_model  # noqa: E402
from mesh_model import fpu  # noqa: E402
import model_runner as mr  # noqa: E402  the numerics and the device build package, one owner each
from model_runner import (  # noqa: E402
    FORMATS, ROOT, SETS_IN_FLIGHT, TB_DIR, _check_rounding, _config_items, activate_int8, activation_to_code,
    apply_activation, bits_float, drop_zp, fmt_bits, fold_bias, get_polynomial_terms, imatmul, op_hex, op_round,
    quant_act, quant_weights, requant_params, requantize, selu_saturates, wrap32, write_sv_package)
import ipu  # noqa: E402  AriL's integer model, on the path model_runner set

# Ensure we can import the mesh helpers
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh"))

from conv_tests import _basic_pair, _general_pair, _im2col_patches, _kernel_size
from matmul_tests import _f2h as float_to_hex
from matmul_tests import _ref_matmul, write_mem

RESULTS_DIR = os.path.join(ROOT, "testbenches", "results", "pipeline")
ACTIONS = ("regression", "gen", "pkg", "analyze", "pack", "gemm", "perf", "oracle", "pack-models", "gpnae-tflite", "rq-vectors", "selftest", "all")

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
        else:
            _packed_float(cfg, k, A, B, bias, sh, col_ent, acts, drop)
            ents = [(a, None) for a in acts]
        _write_pack(os.path.join(TB_DIR, f"pack_{k}.mem"), sh, mp, ents)
    for name in ("matrix_west", "matrix_north", "expected_output", "bound_output"):  # the single-set pass runs set 0
        shutil.copy(os.path.join(TB_DIR, f"{name}_0.mem"), os.path.join(TB_DIR, f"{name}.mem"))
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"),  # int8 always sends bias_<k>: it carries the folded input zero point
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
]


def _run_make_live(log_path: str, fmt_name: str = "fp32", N: int = 16, T: int = 4, lanes: int = 32) -> tuple:
    t0 = time.time()
    process = subprocess.Popen(
        ["make", "verilator", f"FMT={fmt_name}", f"N={N}", f"TILE={T}", f"LANES={lanes}",
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
        raw_log, wall = _run_make_live(log_path, fmt_name, N, T, lanes)

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
            print(f"  {_R}╚══  Sweep Aborted: {r['status']}.{_X}" + (f"\n      {r['fired']}" if "fired" in r else ""))
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


def exact_layer(A, B, bias, act, N, fmt, T=4):
    """Bit-exact output of sienna_layer for one product in a narrow format: per output tile, the depth blocks as passes in
    order (format_layer's order), the bias with the first, then the lane; T is the build's tile size."""
    f = fpu.FORMATS[fmt]
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
    rom = gpnae_model.read_rom(os.path.join(mr.ROOT, "GPNAE", "src", "TYTAN", "Memory", gpnae_model.coeff_file(f)))
    lane = gpnae_model.Lane(f, rom)
    Y = np.zeros((rt * N, ct * N), np.int64)
    for c in range(ct):
        for r in range(rt):
            passes = [(mr.fmt_bits(Ap[r * N:(r + 1) * N, t * N:(t + 1) * N], fmt), mr.fmt_bits(Bp[t * N:(t + 1) * N, c * N:(c + 1) * N], fmt))
                      for t in range(dt)]
            b = mr.fmt_bits(bp[c * N:(c + 1) * N], fmt) if bias is not None else None
            Ct = mesh_model.matmul(f, passes, N, T, 1, b)
            Y[r * N:(r + 1) * N, c * N:(c + 1) * N] = lane.run(Ct, mr.activation_to_code(act))
    return Y[:M, :C]


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
    for line in (f"GEMM sweep on the RTL, mesh N={a.n}, {a.lanes} lanes, linear activation, {a.fmt_name} operands and sums; "
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
        y, sets, cyc = sim.run_job(job, name) if isinstance(sim, mr.RtlLayer) else mr.run_job_hw(job, sim, name)
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
        if a.fmt_name == "fp32" and err > 1e-4:
            print(f"FAIL {name}: error {err:.2e} above 1e-4", flush=True)
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
    return mr.bits_float(np.asarray(C, np.int64), FMT).astype(np.float64).flatten().tolist()  # bf16 rounds every sum: tails shift


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
    if "RESULT: PASSED" not in raw or "Assertion failed" in raw:
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
            " stage occupancy (a partial sum, summed in the PEs, passes as a null), pool+out = to the set's completion pulse.", ""]


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

def gpnae_tflite_main(argv=None):
    """GPNAE's int8 tanh and sigmoid (the lane model, bit-exact with the RTL at G2) against TFLite's int8 TANH and LOGISTIC; reported, not gated."""
    p = argparse.ArgumentParser(description=gpnae_tflite_main.__doc__)
    p.add_argument("--action", choices=ACTIONS, default="gpnae-tflite")
    p.add_argument("--report", required=True)
    a = p.parse_args(argv)
    _tf_imports()
    lane = gpnae_model.Lane(gpnae_model.INT8, gpnae_model.read_rom(os.path.join(ROOT, "GPNAE", gpnae_model.coeff_file(gpnae_model.INT8))))
    q = np.arange(-128, 128, dtype=np.int64)
    L = ["GPNAE int8 lane against TFLite int8 (BUILTIN_REF), every int8 input; the input quantization is the converter's",
         f"{'op':<9}{'s_in':>11}{'z_in':>6}{'equal':>7}{'|d|=1':>7}{'max |d|':>8}{'TFLite-exact':>13}{'lane-exact':>11}"]
    for act, code, op in (("tanh", 3, "tanh"), ("sigmoid", 2, "logistic")):
        for case in gpnae_model.INT8_CASES[act]:
            tfl, s, z = activation_int8(op, case.s_in, case.z_in)
            c = case._replace(s_in=s, z_in=z)
            hw = lane.run(q, code, gpnae_model.int8_params(c, code))
            ex = gpnae_model.exact_int8(q, code, c)
            d = np.abs(hw - tfl)
            L.append(f"{op:<9}{s:>11.6f}{z:>6}{int((d == 0).sum()):>7}{int((d == 1).sum()):>7}{int(d.max()):>8}"
                     f"{int(np.abs(tfl - ex).max()):>13}{int(np.abs(hw - ex).max()):>11}")
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


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


def fixture_pkg(d: str, fmt: str, n: int = 16, tile: int = 4, lanes: int = 32, drop: tuple = ()) -> str:
    """A fixture test_config_pkg.sv in d, written by regression.py's own writer, minus the localparams in drop."""
    path = os.path.join(d, f"pkg_{fmt}_{n}_{tile}_{lanes}{'_old' if drop else ''}.sv")
    items = reg._config_items({"n": n, "tile_size": tile, "lanes": lanes}, fmt, "relu", 4, 15, 1, [], False, 1)
    reg.write_sv_package(path, [x for x in items if x[0] not in drop])
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
def test_guard_checks_geometry():
    with tempfile.TemporaryDirectory() as d:
        big = fixture_pkg(d, "bf16", 32, 8)
        rc, out = pkg_guard(big, "FMT=bf16")  # a stale same-format package of another size
        assert rc != 0 and "N=32 TILE_SIZE=8 NUM_LANES=32, but the build asks for N=16 TILE=4 LANES=32" in out, out
        rc, out = pkg_guard(big, "FMT=bf16", "N=32", "TILE=8")
        assert rc == 0 and "N=32 TILE=8 LANES=32" in out, out
        rc, out = pkg_guard(fixture_pkg(d, "int8"), "FMT=int8", "LANES=64")
        assert rc != 0 and "LANES=64" in out, out


@selftest
def test_guard_missing_file():
    with tempfile.TemporaryDirectory() as d:
        rc, out = pkg_guard(os.path.join(d, "none.sv"), "FMT=fp32")
        assert rc != 0 and "missing or unreadable" in out, out


@selftest
def test_parse_time_rejections():
    r = make_n("regression", "FMT=fp16")
    assert r.returncode != 0 and "Invalid FMT=fp16: must be one of fp32, bf16, int8" in r.stderr, r.stderr
    for t in ("regression", "verilator", "lint", "pkg", "pack"):
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode != 0 and "sienna_ck0.sh" in r.stderr, (t, r.stderr)
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
def test_check_target():
    r = make_n("check", "FMT=int8", "N=32", "TILE=8", "LANES=64", "PYTHON=py")
    assert r.returncode == 0 and r.stdout.split() == "py regression.py --action all --format int8 --n 32 --tile 8 --lanes 64".split(), r.stdout + r.stderr
    r = make_n("check", "FMT=fp16")
    assert r.returncode != 0 and "Invalid FMT=fp16" in r.stderr, r.stderr


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
def test_fmt_fields_match_formats():
    mk = open(os.path.join(ROOT, "Makefile")).read()
    fields = {f: tuple(int(x) for x in v.split()) for f, v in re.findall(r"^FMT_FIELDS_(\w+) = ([\d ]+)$", mk, re.M)}
    assert re.search(r"^FORMATS = (.*)$", mk, re.M).group(1).split() == list(reg.FORMATS)
    assert fields == {f: (*reg.FORMATS[f], int(f == "int8")) for f in reg.FORMATS}, fields


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
def test_selu_saturation_refused():
    # Review Focus 5: an int8 SELU entry whose lane input reaches x >= 487.29 must be refused, not packed.
    req = dict(mult=np.full(2, 1 << 30), shift=np.zeros(2, np.int64), zp=-128, amin=-128, amax=127, mx=(1 << 15) - 1, shx=0,
               mout=1, shout=0, zout=0)
    assert np.any(mr.selu_saturates(req["mx"], req["shx"], req["zp"], np.arange(req["amin"], req["amax"] + 1))), \
        "the test's req does not saturate"
    m = {"W": np.ones((2, 2)), "bias": None, "act": "selu", "req": req, "inputs": [np.ones((1, 2))]}
    try:
        mr.pack_jobs([m], 16, int8=True)
    except ValueError:
        return
    raise AssertionError("pack_jobs accepted a saturating SELU entry")


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


# ── all (the one-verdict gate) ───────────────────────────────────────────────

CHECK_DIR = os.path.join(ROOT, "testbenches", "results", "check")
FIRING = re.compile(r"^.*(?:Assertion failed|%Error).*$", re.M)  # _parse_log's witness: a firing assertion leaves the exit status clean


def check_steps(a) -> list:
    """The gate's steps as (name, argv): the self-tests, then make targets in the build's format, N, TILE and LANES."""
    mk = lambda target, *extra: ["make", target, f"FMT={a.fmt}", f"N={a.n}", f"TILE={a.tile}", f"LANES={a.lanes}", f"PYTHON={sys.executable}", *extra]
    steps = [("selftest", [sys.executable, os.path.abspath(__file__), "--action", "selftest"]), ("sm-verilator", mk("sm-verilator")),
             ("gpnae-verilator", mk("gpnae-verilator")), ("regression", mk("regression")), ("pack", mk("pack")), ("gemm", mk("gemm", "QUICK=1"))]
    return steps + ([("tflite", mk("tflite"))] if a.fmt == "int8" else [])


def run_step(argv: list, log_path: str) -> str:
    """Runs one step with its output teed to log_path: PASS, or FAIL naming the exit status or the firing the log holds."""
    with open(log_path, "w") as log:
        try:
            p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        except OSError as e:
            log.write(f"cannot run {argv[0]}: {e}\n")
            return f"FAIL (cannot run: {e})"
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = p.wait()
    fired = FIRING.search(open(log_path, errors="replace").read())
    return f"FAIL (exit {rc})" if rc else f"FAIL (log: {fired.group(0).strip()[:60]})" if fired else "PASS"


def all_main(argv=None, steps=check_steps) -> None:
    """Self-tests, then the SystolicMesh and GPNAE regressions, the SIENNA regression, pack, gemm --quick and (int8) tflite via make; one table, exit 1 if any step failed."""
    ap = argparse.ArgumentParser(description=all_main.__doc__)
    ap.add_argument("--action", choices=ACTIONS, default="all")
    ap.add_argument("--format", dest="fmt", default="fp32", choices=sorted(mr.FORMATS))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    a = ap.parse_args(argv)
    os.makedirs(CHECK_DIR, exist_ok=True)
    tag = f"{a.fmt}_N{a.n}_T{a.tile}_L{a.lanes}"
    rows = []
    for name, cmd in steps(a):
        log_path = os.path.join(CHECK_DIR, f"{name}_{tag}.log")
        print(hdr(f"\n=== check: {name}  ({' '.join(cmd)}) ==="), flush=True)
        t0 = time.time()
        rows.append((name, run_step(cmd, log_path), time.time() - t0, log_path))
    failed = sum(v != "PASS" for _, v, _, _ in rows)
    table = [f"CHECK {tag}", f"{'step':<16}{'verdict':<48}{'secs':>8}  log"] + [f"{n:<16}{v:<48}{s:8.1f}  {p}" for n, v, s, p in rows]
    table.append(f"CHECK {'PASS' if failed == 0 else 'FAIL'}: {len(rows) - failed} of {len(rows)} steps passed")
    open(os.path.join(CHECK_DIR, f"check_{tag}.log"), "w").write("\n".join(table) + "\n")
    print("\n" + "\n".join(table))
    sys.exit(1 if failed else 0)


MAINS = {"pack": pack_main, "gemm": gemm_main, "perf": perf_main, "oracle": oracle_main, "pack-models": pack_models_main,
         "gpnae-tflite": gpnae_tflite_main, "rq-vectors": rq_vectors_main, "selftest": selftest_main, "all": all_main}  # each parses its own options (regression.py --action A --help)

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
             "pack, gemm, perf, oracle, pack-models, gpnae-tflite and rq-vectors take their own options (--action A --help)" % PKG_DEFAULT_TEST,
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
