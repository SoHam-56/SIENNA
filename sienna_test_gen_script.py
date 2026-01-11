#!/usr/bin/env python3
import argparse
import math
import os
import struct
from typing import List, Tuple, Union

import numpy as np

# ==========================================
# 1. Utility Functions
# ==========================================


def float_to_hex(f):
    """Convert a float32 to an 8-character hex string (big-endian)."""
    return "".join(f"{b:02x}" for b in struct.pack(">f", np.float32(f)))


def _is_float_like(v):
    return isinstance(v, (float, np.floating))


def _is_int_like(v):
    return isinstance(v, (int, np.integer))


# ==========================================
# 2. Mathematical Golden Models
# ==========================================


def activation_to_code(activation_type):
    """Map string to 2-bit hardware code."""
    mapping = {
        "none": 0b00,
        "linear": 0b00,
        "relu": 0b01,
        "sigmoid": 0b10,
        "tanh": 0b11,
    }
    return mapping.get(activation_type.lower(), 0b01)


def apply_activation(x, activation_type="relu"):
    """Apply activation function element-wise."""
    if activation_type == "relu":
        return np.maximum(0, x)
    elif activation_type == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation_type == "tanh":
        return np.tanh(x)
    return x


def apply_maxpool_2d(
    input_matrix, pool_h=2, pool_w=2, stride_h=None, stride_w=None, padding=0
):
    """Apply 2D max pooling with padding."""
    if stride_h is None:
        stride_h = pool_h
    if stride_w is None:
        stride_w = pool_w

    if padding > 0:
        # Pad with -inf equivalent (or 0 for ReLU outputs)
        input_matrix = np.pad(
            input_matrix,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=-10000.0,
        )

    in_h, in_w = input_matrix.shape
    out_h = (in_h - pool_h) // stride_h + 1
    out_w = (in_w - pool_w) // stride_w + 1

    if out_h <= 0 or out_w <= 0:
        return np.zeros((1, 1), dtype=np.float32)

    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            window = input_matrix[
                h_start : h_start + pool_h, w_start : w_start + pool_w
            ]
            output[i, j] = np.max(window)

    return output


def apply_dropout(x, dropout_p=0.5, training_mode=False):
    """Inference mode = pass through (no dropout applied)."""
    if training_mode:
        mask = np.random.binomial(1, 1 - dropout_p, size=x.shape)
        return x * mask / (1 - dropout_p)
    return x


# ==========================================
# 3. File Writers
# ==========================================


def write_mif_file(filename, data, width=32, depth=None):
    values = data.flatten() if isinstance(data, np.ndarray) else list(data)
    if depth is None:
        depth = len(values)

    with open(filename, "w") as f:
        f.write(f"-- Memory Initialization File (.mif)\n")
        f.write(f"WIDTH={width};\nDEPTH={depth};\n")
        f.write(f"ADDRESS_RADIX=UNS;\nDATA_RADIX=HEX;\n\n")
        f.write(f"CONTENT BEGIN\n")
        for addr, val in enumerate(values):
            if _is_float_like(val):
                hex_val = float_to_hex(val)
            elif _is_int_like(val):
                hex_val = f"{int(val) & 0xFFFFFFFF:08x}"
            else:
                hex_val = "00000000"
            f.write(f"    {addr} : {hex_val};\n")
        f.write(f"END;\n")


def write_sv_package(filename, items):
    with open(filename, "w") as f:
        f.write("// Auto-Generated Configuration Package\n")
        f.write("package test_config_pkg;\n\n")
        for name, val, vtype in items:
            if vtype == "float":
                f.write(f"  localparam shortreal {name} = {val};\n")
            else:
                f.write(f"  localparam int {name} = {val};\n")
        f.write("\nendpackage\n")


def write_intermediate_debug_file(filename, config_dict, arrays):
    """Writes a human-readable text file with matrix snapshots."""
    with open(filename, "w") as f:
        # Header
        f.write("=" * 60 + "\n")
        f.write("INTERMEDIATE VALUES (for debugging)\n")
        f.write("=" * 60 + "\n\n")

        # Config Section
        act_name = config_dict.get("activation", "unknown")
        act_code = config_dict.get("activation_code", 0)
        n_val = config_dict.get("n", 32)

        f.write("Test Configuration (key fields):\n")
        f.write(f"  Number of tests: {config_dict.get('num_tests', 1)}\n")
        f.write(f"  Activation: {act_name} (code: {bin(act_code)})\n")
        f.write(f"  Number of terms: {n_val * n_val}\n")
        f.write(f"  Matrix size: {n_val}x{n_val}\n\n")

        # Helper to print matrices
        def print_matrix_slice(name, mat, rows=3, cols=3):
            f.write(f"{name}:\n")
            # Slice safely
            r_end = min(rows, mat.shape[0])
            c_end = min(cols, mat.shape[1])
            slice_mat = mat[:r_end, :c_end]
            f.write(str(slice_mat))
            f.write("\n\n")

        # Matrix dumps
        print_matrix_slice("Input Matrix A (first 3x3)", arrays["A"])
        print_matrix_slice("Input Matrix B (first 3x3)", arrays["B"])
        print_matrix_slice("After MatMul C = AxB (first 3x3)", arrays["C"])

        act_label = f"After Activation ({act_name}) (first 3x3)"
        print_matrix_slice(act_label, arrays["C_act"])

        # MaxPool Input (Displaying 5x5 as requested in prompt, usually same as Act output)
        print_matrix_slice("MaxPool Input (5x5)", arrays["C_act"], rows=5, cols=5)

        # Post MaxPool (Display whole or slices depending on size, showing 5x5 cap)
        print_matrix_slice("After MaxPool", arrays["C_pooled"], rows=5, cols=5)

        # Final Dropout
        print_matrix_slice("After Dropout (inference mode)", arrays["C_final"], rows=5, cols=5)

        # Final Scalar Check
        final_flat = arrays["C_final"].flatten()
        if len(final_flat) > 0:
            val = final_flat[0]
            f.write(f"Expected Final Output (first element): {val}\n")
            f.write(f"Expected Final Output (hex): {float_to_hex(val)}\n")


# ==========================================
# 4. Configuration Builder
# ==========================================


def build_config_items(**kwargs) -> List[Tuple[str, Union[int, float], str]]:
    items = []

    def I(name, v):
        items.append((name, int(v), "int"))

    def F(name, v):
        items.append((name, float(v), "float"))

    N = kwargs.get("n", 32)
    sram_depth = N * N

    # Auto-Calculate Address Lines
    if sram_depth > 1:
        min_addr_lines = math.ceil(math.log2(sram_depth))
    else:
        min_addr_lines = 1

    final_addr_lines = max(kwargs.get("addr_lines", 0), min_addr_lines)

    # Parameter Map
    I("NUM_TESTS", kwargs.get("num_tests", 1))
    I("ACTIVATION_CODE", kwargs.get("activation_code", 1))
    I("NUM_TERMS", kwargs.get("num_terms", sram_depth))
    I("N", N)
    I("DATA_WIDTH", kwargs.get("data_width", 32))
    I("SRAM_DEPTH", sram_depth)
    I("ADDR_LINES", final_addr_lines)
    I("CONTROL_WIDTH", kwargs.get("control_width", 2))
    I("IN_ROWS", kwargs.get("in_rows", 5))
    I("IN_COLS", kwargs.get("in_cols", 5))
    I("POOL_H", kwargs.get("pool_h", 2))
    I("POOL_W", kwargs.get("pool_w", 2))
    I("STRIDE_ROWS", kwargs.get("pool_h", 2))
    I("STRIDE_COLS", kwargs.get("pool_w", 2))
    I("PADDING", kwargs.get("padding", 1))
    I("DROPOUT_P_PERCENT", int(round(kwargs.get("dropout_p", 0.5) * 100)))
    I("LFSR_WIDTH", kwargs.get("lfsr_width", 32))
    I("INTERMEDIATE_BUFFER_DEPTH", sram_depth * 2)
    I("FIFO_DEPTH", kwargs.get("fifo_depth", 16))
    I("SEED", kwargs.get("seed", 42))

    matrix_type_map = {"random": 0, "identity": 1, "ones": 2, "small_int": 3}
    I("MATRIX_TYPE", matrix_type_map.get(kwargs.get("matrix_type", "random"), 0))

    val_range = kwargs.get("value_range", (-1.0, 1.0))
    F("MIN_VAL", val_range[0])
    F("MAX_VAL", val_range[1])

    return items


# ==========================================
# 5. Main Generation Logic
# ==========================================


def generate_pipeline_test_vectors(output_dir="testbenches", **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    # Unpack Arguments needed for logic
    N = kwargs.get("n", 32)
    seed = kwargs.get("seed", 42)
    m_type = kwargs.get("matrix_type", "random")
    v_range = kwargs.get("value_range", (-1.0, 1.0))
    act_type = kwargs.get("activation", "relu")
    pool_h = kwargs.get("pool_h", 2)
    pool_w = kwargs.get("pool_w", 2)
    padding = kwargs.get("padding", 1)
    dropout_p = kwargs.get("dropout_p", 0.5)

    print(f"Generating vectors in '{output_dir}' for N={N}...")

    # --- Step 1: Matrix Gen ---
    np.random.seed(seed)
    if m_type == "identity":
        A = np.eye(N, dtype=np.float32)
        B = np.eye(N, dtype=np.float32)
    elif m_type == "ones":
        A = np.ones((N, N), dtype=np.float32)
        B = np.ones((N, N), dtype=np.float32)
    elif m_type == "small_int":
        A = np.random.randint(-3, 4, (N, N)).astype(np.float32)
        B = np.random.randint(-3, 4, (N, N)).astype(np.float32)
    else:
        A = np.random.uniform(v_range[0], v_range[1], (N, N)).astype(np.float32)
        B = np.random.uniform(v_range[0], v_range[1], (N, N)).astype(np.float32)

    # --- Step 2: Golden Calculation ---
    C = np.matmul(A, B)
    C_act = apply_activation(C, act_type)
    C_pooled = apply_maxpool_2d(C_act, pool_h, pool_w, padding=padding)
    C_final = apply_dropout(C_pooled, dropout_p)

    # --- Step 3: Flatten ---
    flat_west = A.flatten(order="C")  # Row Major
    flat_north = B.flatten(order="C")  # Row Major
    flat_out = C_final.flatten(order="C")

    # --- Step 4: Write MIFs ---
    write_mif_file(os.path.join(output_dir, "matrix_west.mif"), flat_west)
    write_mif_file(os.path.join(output_dir, "matrix_north.mif"), flat_north)
    write_mif_file(os.path.join(output_dir, "expected_output.mif"), flat_out)

    print(f"    ✓ matrix_west.mif ({len(flat_west)} items)")
    print(f"    ✓ matrix_north.mif ({len(flat_north)} items) [Row Major]")
    print(f"    ✓ expected_output.mif ({len(flat_out)} items)")

    # --- Step 5: Write Config (Package Only) ---
    act_code = activation_to_code(act_type)

    # Update kwargs with calculated values
    kwargs["activation_code"] = act_code

    config_items = build_config_items(**kwargs)

    # REMOVED: write_config_mif(...) for test_config.mif
    write_sv_package(os.path.join(output_dir, "test_config_pkg.sv"), config_items)
    print(f"    ✓ test_config_pkg.sv")

    # --- Step 6: Intermediate Values File ---
    # Prepare data dictionary for the writer
    array_data = {
        "A": A,
        "B": B,
        "C": C,
        "C_act": C_act,
        "C_pooled": C_pooled,
        "C_final": C_final
    }

    write_intermediate_debug_file(
        os.path.join(output_dir, "intermediate_values.txt"),
        kwargs,
        array_data
    )
    print(f"    ✓ intermediate_values.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sienna Test Vector Generator")

    parser.add_argument("--n", type=int, default=32, help="Matrix Size")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh", "linear", "none"],
    )
    parser.add_argument("--pool-h", type=int, default=2)
    parser.add_argument("--pool-w", type=int, default=2)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument(
        "--matrix-type",
        type=str,
        default="random",
        choices=["random", "identity", "ones", "small_int"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="testbenches")
    parser.add_argument("--addr-lines", type=int, default=0)
    parser.add_argument("--data-width", type=int, default=32)

    args = parser.parse_args()

    # Pass arguments as kwargs to avoid "multiple values" error
    generate_pipeline_test_vectors(
        n=args.n,
        activation=args.activation,
        pool_h=args.pool_h,
        pool_w=args.pool_w,
        padding=args.padding,
        dropout_p=args.dropout_p,
        matrix_type=args.matrix_type,
        seed=args.seed,
        output_dir=args.output_dir,
        addr_lines=args.addr_lines,
        data_width=args.data_width,
    )
