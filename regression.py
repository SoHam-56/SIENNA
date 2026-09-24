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

# Ensure we can import the mesh helpers
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh"))

from conv_tests import _basic_pair, _im2col_patches, _kernel_size
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


# GPNAE control words. 01/10/11 are the only modes the hardware decodes; anything else stalls the lane.
# No entry for 0 on purpose: a code the RTL does not implement must not be reachable from a test.
ACTIVATION_CODES = {"selu": 1, "sigmoid": 2, "tanh": 3}

# Polynomial terms per activation, as passed to the TYTAN controller.
ACTIVATION_TERMS = {"selu": 14, "sigmoid": 15, "tanh": 30}


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
    return x.copy()  # IDLE


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


def apply_dropout(x: np.ndarray, p=0.5, training=False, seed=1, num_lanes=16) -> np.ndarray:
    """Inference copies; training replays dropout.sv's per-lane LFSR, window w on lane w % num_lanes."""
    if not training:
        return x.copy()
    p_percent = int(round(p * 100))
    thr = ((2**32 - 1) * p_percent) // 100
    scale = np.float32(100.0 / (100 - p_percent))
    flat = x.astype(np.float32).flatten()
    out = np.empty_like(flat)
    states = [_lane_seed(seed, lane) for lane in range(num_lanes)]
    for w, v in enumerate(flat):
        lane = w % num_lanes
        states[lane] = _lfsr_next(states[lane])
        keep = states[lane] >= thr  # decided on the word the beat advances to, as dropout.sv does
        out[w] = v * scale if keep else v * np.float32(0.0)  # a dropped negative stays -0.0
    return out.reshape(x.shape)


def build_conv_matrices(N: int, conv_type: str, seed: int, stride=None) -> tuple:
    K = _kernel_size(N)
    if conv_type == "basic":
        return _basic_pair(img_size=K * K, K=K, seed=seed)
    raise ValueError(f"Unknown conv_type '{conv_type}'")


def write_sv_package(path: str, items: list) -> None:
    with open(path, "w") as f:
        f.write("// Auto-Generated Configuration Package\npackage test_config_pkg;\n\n")
        for name, val, vtype in items:
            kw = "shortreal" if vtype == "float" else "int"
            f.write(f"  localparam {kw} {name} = {val};\n")
        f.write("\nendpackage\n")


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
    C = _ref_matmul(A, B)
    C_act = apply_activation(C, act_type)
    C_pooled = apply_maxpool_2d(
        C_act, cfg.get("pool_h", 2), cfg.get("pool_w", 2), padding=cfg.get("padding", 1)
    )
    C_final = apply_dropout(C_pooled, cfg.get("dropout_p", 0.5), cfg.get("training", False), drop_seed)
    return C, C_act, C_pooled, C_final


def generate_vectors(cfg: dict) -> None:
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
    scale = np.float32(cfg.get("scale", 1.0))
    A, B = (A * scale).astype(np.float32), (B * scale).astype(np.float32)

    # Set k's dropout seed is set_dropout_seed(DROPOUT_SEED, k), as the TB drives it.
    drop_seed = 0x2ACE0000 + seed
    C, C_act, C_pooled, C_final = _golden(A, B, cfg, act_type, drop_seed)

    # Write files for Verilator testbench
    write_mem(os.path.join(TB_DIR, "matrix_west.mem"), A)
    write_mem(os.path.join(TB_DIR, "matrix_north.mem"), B)
    write_mem(os.path.join(TB_DIR, "expected_output.mem"), C_final)

    # Streamed sets: set 0 is the test's own pattern, the rest random and distinct.
    num_sets = cfg.get("num_sets", 4)
    masks = []
    for k in range(num_sets):
        if k == 0:
            Ak, Bk, Fk = A, B, C_final
        else:
            rng = np.random.RandomState(seed + 1000 + k)
            Ak = (rng.uniform(-1.0, 1.0, (N, N)) * scale).astype(np.float32)
            Bk = (rng.uniform(-1.0, 1.0, (N, N)) * scale).astype(np.float32)
            Fk = _golden(Ak, Bk, cfg, act_type, set_dropout_seed(drop_seed, k))[3]
        write_mem(os.path.join(TB_DIR, f"matrix_west_{k}.mem"), Ak)
        write_mem(os.path.join(TB_DIR, f"matrix_north_{k}.mem"), Bk)
        write_mem(os.path.join(TB_DIR, f"expected_output_{k}.mem"), Fk)
        if cfg.get("training", False):
            masks.append(tuple((Fk.flatten() != 0).tolist()))
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            agree = sum(a == b for a, b in zip(masks[i], masks[j])) / len(masks[i])
            if agree > 0.75:  # independent masks agree about half the time
                raise RuntimeError(f"{test_name}: sets {i} and {j} dropout masks agree on {agree:.0%}")

    # Dump the intermediate Golden Trace for debug comparisons
    dump_golden_trace(test_name, C, C_act, C_pooled)

    # Dump SV Config Package
    sram_depth = N * N
    items = [
        ("N", N, "int"),
        ("TILE_SIZE", tile_size, "int"),
        ("NUM_LANES", cfg.get("lanes", 16), "int"),
        ("HOST_WORDS", cfg.get("host_words", N), "int"),
        ("DATA_WIDTH", 32, "int"),
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
        ("CONTROL_WIDTH", 2, "int"),
        ("NUM_SETS", num_sets, "int"),
        ("TRAINING_MODE", int(bool(cfg.get("training", False))), "int"),
        ("DROPOUT_SEED", drop_seed, "int"),
        ("ADDR_LINES", max(1, math.ceil(math.log2(sram_depth))), "int"),
    ]
    write_sv_package(os.path.join(TB_DIR, "test_config_pkg.sv"), items)


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
    {"name": "conv_basic_selu", "mode": "conv", "conv_type": "basic", "act": "selu"},
    {"name": "conv_basic_tanh", "mode": "conv", "conv_type": "basic", "act": "tanh"},
    {"name": "matmul_random_tanh_train", "mode": "matmul", "matrix_type": "random", "act": "tanh", "training": True},
    {"name": "matmul_random_sigm_train", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "training": True},
    {"name": "conv_basic_selu_train", "mode": "conv", "conv_type": "basic", "act": "selu", "training": True},
    {"name": "matmul_large_selu", "mode": "matmul", "matrix_type": "random", "act": "selu", "scale": 2.5},
    {"name": "matmul_large_sigm", "mode": "matmul", "matrix_type": "random", "act": "sigmoid", "scale": 2.5},
    {"name": "matmul_large_tanh", "mode": "matmul", "matrix_type": "random", "act": "tanh", "scale": 2.5},
]


def _run_make_live(log_path: str) -> tuple:
    t0 = time.time()
    process = subprocess.Popen(
        ["make", "verilator"],
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


def run_regression(N: int, T: int, target_test: str = None, lanes: int = 16, host_words: int = None):
    _check_dropout_generator()
    print(hdr(f"\n{'═'*70}\n  SIENNA PIPELINE — Regression Suite\n{'═'*70}"))
    tests_to_run = PIPELINE_TESTS

    if target_test:
        tests_to_run = [t for t in PIPELINE_TESTS if target_test in t["name"]]
        if not tests_to_run:
            print(f"  {_R}[ERROR] No tests found containing '{target_test}'{_X}")
            return

    print(
        f"  Matrix size : {N}×{N}\n  Tile size   : {T}×{T}\n  Total tests : {len(tests_to_run)}\n"
        + hdr(f"{'═'*70}")
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for idx, t in enumerate(tests_to_run):
        print(
            f"\n  ║  [{idx+1}/{len(tests_to_run)}] {_O}{t['name']}{_X}  (generating...)"
        )

        # 1. Generate Vectors & Dump Expected Traces
        cfg = {"n": N, "tile_size": T, "lanes": lanes, "host_words": host_words or N, **t}
        generate_vectors(cfg)

        # 2. Run Verilator (Streams live status)
        log_path = os.path.join(RESULTS_DIR, f"{t['name']}.log")
        raw_log, wall = _run_make_live(log_path)

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
        choices=["regression", "gen", "analyze"],
        help="Action to perform",
    )
    p.add_argument("--matrix-size", "--n", type=int, default=16)
    p.add_argument("--tile-size", type=int, default=4)
    p.add_argument("--lanes", type=int, default=16)
    p.add_argument("--host-words", type=int, default=None, help="words per host write (default: N, one row)")
    p.add_argument("--mode", default="matmul", choices=["matmul", "conv"])
    p.add_argument("--conv-type", default="basic")
    p.add_argument("--activation", default="selu")
    p.add_argument(
        "--test", type=str, default=None, help="Run a specific test by name substring"
    )
    args, unknown = p.parse_known_args()

    if args.action == "regression":
        run_regression(args.matrix_size, args.tile_size, args.test, args.lanes, args.host_words)
    elif args.action == "gen":
        generate_vectors(
            {
                "n": args.matrix_size,
                "tile_size": args.tile_size,
                "mode": args.mode,
                "conv_type": args.conv_type,
                "activation": args.activation,
                "name": "manual_gen",
            }
        )
    elif args.action == "analyze":
        dump_hardware_trace("manual_run", args.matrix_size, print_to_console=True)
