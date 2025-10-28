import argparse
import os
import struct

import numpy as np


def float_to_hex(f):
    """Convert a float32 to an 8-character hex string (big-endian)."""
    return "".join(f"{b:02x}" for b in struct.pack(">f", f))


def hex_to_float(hex_str):
    """Convert an 8-character hex string to float32."""
    return struct.unpack(">f", bytes.fromhex(hex_str))[0]


def apply_activation(x, activation_type="relu"):
    """Apply activation function element-wise."""
    if activation_type == "relu":
        return np.maximum(0, x)
    elif activation_type == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation_type == "tanh":
        return np.tanh(x)
    elif activation_type == "none":
        return x
    else:
        return x


def apply_maxpool_2d(
    input_matrix, pool_h=2, pool_w=2, stride_h=None, stride_w=None, padding=0
):
    """
    Apply 2D max pooling to input matrix.

    Parameters:
    - input_matrix: 2D numpy array (IN_ROWS x IN_COLS)
    - pool_h, pool_w: pooling window size
    - stride_h, stride_w: stride (defaults to pool size if None)
    - padding: zero padding around input
    """
    if stride_h is None:
        stride_h = pool_h
    if stride_w is None:
        stride_w = pool_w

    # Add padding
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode="constant", constant_values=0)

    in_h, in_w = input_matrix.shape
    out_h = (in_h - pool_h) // stride_h + 1
    out_w = (in_w - pool_w) // stride_w + 1

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


def apply_dropout(x, dropout_p=0.5, training_mode=False, seed=None):
    """
    Apply dropout (for golden reference, we'll assume inference mode = no dropout).

    Parameters:
    - x: input array
    - dropout_p: dropout probability
    - training_mode: if True, randomly drop elements
    - seed: random seed
    """
    if training_mode:
        if seed is not None:
            np.random.seed(seed)
        mask = np.random.binomial(1, 1 - dropout_p, size=x.shape)
        return x * mask / (1 - dropout_p)
    else:
        # Inference mode - no dropout
        return x


def generate_pipeline_test_vectors(
    n=32,
    activation_type="relu",
    pool_h=2,
    pool_w=2,
    pool_padding=1,
    dropout_p=0.5,
    output_dir="testbenches",
    seed=42,
    matrix_type="random",
    value_range=(-1.0, 1.0),
):
    """
    Generate complete pipeline test vectors:
    Systolic Array (Matrix Mult) -> Activation (GPNAE) -> MaxPool -> Dropout

    Parameters:
    - n: Matrix size (N x N)
    - activation_type: 'relu', 'sigmoid', 'tanh', 'none'
    - pool_h, pool_w: pooling window size
    - pool_padding: padding for maxpool
    - dropout_p: dropout probability
    - output_dir: Directory to save test files
    - seed: Random seed for reproducibility
    - matrix_type: "random", "identity", "ones", "small_int"
    - value_range: Tuple (min, max) for random values
    """

    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Sienna Pipeline Test Vectors")
    print("=" * 60)

    # ========== STEP 1: Generate Input Matrices ==========
    print(f"\n1. Generating {n}x{n} input matrices...")

    if matrix_type == "identity":
        A = np.eye(n, dtype=np.float32)
        B = np.eye(n, dtype=np.float32)
    elif matrix_type == "ones":
        A = np.ones((n, n), dtype=np.float32)
        B = np.ones((n, n), dtype=np.float32)
    elif matrix_type == "small_int":
        A = np.random.randint(-3, 4, size=(n, n)).astype(np.float32)
        B = np.random.randint(-3, 4, size=(n, n)).astype(np.float32)
    else:  # random
        A = np.random.uniform(value_range[0], value_range[1], size=(n, n)).astype(
            np.float32
        )
        B = np.random.uniform(value_range[0], value_range[1], size=(n, n)).astype(
            np.float32
        )

    # ========== STEP 2: Matrix Multiplication (Systolic Array) ==========
    print(f"2. Computing matrix multiplication: C = A × B...")
    C = np.matmul(A, B).astype(np.float32)

    # ========== STEP 3: Activation Function (GPNAE) ==========
    print(f"3. Applying activation function: {activation_type}...")
    activated = apply_activation(C, activation_type)

    # ========== STEP 4: MaxPooling ==========
    print(f"4. Applying MaxPool2D (pool={pool_h}x{pool_w}, padding={pool_padding})...")
    # Note: For a full NxN matrix, we need to decide how to apply maxpool
    # Option 1: Take first K×K submatrix where K fits maxpool input size
    # Option 2: Apply maxpool to entire matrix (if dimensions work)

    # For simplicity, let's take a small region or reshape
    # Assuming IN_ROWS=5, IN_COLS=5 from your parameters
    in_rows, in_cols = 5, 5

    # Extract top-left corner for maxpool input
    maxpool_input = activated[:in_rows, :in_cols]
    pooled = apply_maxpool_2d(maxpool_input, pool_h, pool_w, padding=pool_padding)

    # ========== STEP 5: Dropout ==========
    print(f"5. Applying Dropout (p={dropout_p}, inference mode)...")
    final_output = apply_dropout(pooled, dropout_p, training_mode=False)

    # ========== STEP 6: Write Output Files ==========
    print(f"\n6. Writing test files to '{output_dir}'...")

    # Write North Matrix (Matrix A)
    north_file = os.path.join(output_dir, "matrix_north.txt")
    with open(north_file, "w") as f:
        for val in A.flatten():
            f.write(float_to_hex(val) + "\n")
    print(f"   ✓ {north_file} ({n}×{n} = {n*n} elements)")

    # Write West Matrix (Matrix B)
    west_file = os.path.join(output_dir, "matrix_west.txt")
    with open(west_file, "w") as f:
        for val in B.flatten():
            f.write(float_to_hex(val) + "\n")
    print(f"   ✓ {west_file} ({n}×{n} = {n*n} elements)")

    # Write Expected Final Output
    # Note: The actual output depends on what your design outputs
    # It could be a single scalar, or the flattened pooled matrix
    output_file = os.path.join(output_dir, "expected_output.txt")
    with open(output_file, "w") as f:
        # Write the first element of final output as the expected result
        # Adjust this based on your actual design output
        f.write(float_to_hex(final_output.flatten()[0]) + "\n")
    print(f"   ✓ {output_file} (1 element)")

    # Write intermediate results for debugging
    debug_file = os.path.join(output_dir, "intermediate_values.txt")
    with open(debug_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("INTERMEDIATE VALUES (for debugging)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Input Matrix A (first 3×3):\n")
        f.write(f"{A[:3, :3]}\n\n")

        f.write(f"Input Matrix B (first 3×3):\n")
        f.write(f"{B[:3, :3]}\n\n")

        f.write(f"After MatMul C = A×B (first 3×3):\n")
        f.write(f"{C[:3, :3]}\n\n")

        f.write(f"After Activation ({activation_type}) (first 3×3):\n")
        f.write(f"{activated[:3, :3]}\n\n")

        f.write(f"MaxPool Input ({in_rows}×{in_cols}):\n")
        f.write(f"{maxpool_input}\n\n")

        f.write(f"After MaxPool:\n")
        f.write(f"{pooled}\n\n")

        f.write(f"After Dropout (inference mode):\n")
        f.write(f"{final_output}\n\n")

        f.write(f"Expected Final Output (first element): {final_output.flatten()[0]}\n")
        f.write(
            f"Expected Final Output (hex): {float_to_hex(final_output.flatten()[0])}\n"
        )

    print(f"   ✓ {debug_file} (intermediate values for verification)")

    # ========== STEP 7: Summary ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Pipeline: MatMul -> {activation_type} -> MaxPool({pool_h}×{pool_w}) -> Dropout({dropout_p})"
    )
    print(f"Matrix size: {n}×{n}")
    print(f"Seed: {seed}")
    print(f"\nData preview:")
    print(f"  A[0,0] = {A[0,0]:.6f} (hex: {float_to_hex(A[0,0])})")
    print(f"  B[0,0] = {B[0,0]:.6f} (hex: {float_to_hex(B[0,0])})")
    print(f"  C[0,0] = {C[0,0]:.6f} (hex: {float_to_hex(C[0,0])})")
    print(
        f"  Final[0] = {final_output.flatten()[0]:.6f} (hex: {float_to_hex(final_output.flatten()[0])})"
    )
    print("=" * 60)

    return {
        "A": A,
        "B": B,
        "matmul": C,
        "activated": activated,
        "pooled": pooled,
        "final": final_output,
    }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate test vectors for Sienna pipeline"
    )
    parser.add_argument("--n", type=int, default=32, help="Matrix size (N×N)")
    parser.add_argument(
        "--activation",
        choices=["relu", "sigmoid", "tanh", "none"],
        default="relu",
        help="Activation function",
    )
    parser.add_argument("--pool-h", type=int, default=2, help="Pooling height")
    parser.add_argument("--pool-w", type=int, default=2, help="Pooling width")
    parser.add_argument("--pool-padding", type=int, default=1, help="Pooling padding")
    parser.add_argument(
        "--dropout-p", type=float, default=0.5, help="Dropout probability"
    )
    parser.add_argument(
        "--output-dir", type=str, default="testbench", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--matrix-type",
        choices=["random", "identity", "ones", "small_int"],
        default="random",
        help="Type of input matrices",
    )
    parser.add_argument("--min-val", type=float, default=-1.0, help="Min random value")
    parser.add_argument("--max-val", type=float, default=1.0, help="Max random value")

    args = parser.parse_args()

    generate_pipeline_test_vectors(
        n=args.n,
        activation_type=args.activation,
        pool_h=args.pool_h,
        pool_w=args.pool_w,
        pool_padding=args.pool_padding,
        dropout_p=args.dropout_p,
        output_dir=args.output_dir,
        seed=args.seed,
        matrix_type=args.matrix_type,
        value_range=(args.min_val, args.max_val),
    )


if __name__ == "__main__":
    # If run without arguments, generate default test case
    import sys

    if len(sys.argv) == 1:
        print("No arguments provided, generating default test case...")
        print("Use --help to see all options\n")
        generate_pipeline_test_vectors(
            n=32, activation_type="relu", seed=42, matrix_type="small_int"
        )
    else:
        main()
