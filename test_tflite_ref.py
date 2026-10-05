#!/usr/bin/env python3
"""Known values for tflite_ref.py (numpy only): QuantizeMultiplier, activation ranges, a hand-computed FC and 3x3 conv, the fold."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_runner as ref  # noqa: E402

fails = checks = 0


def check(what, got, want):
    global fails, checks
    checks += 1
    g, w = np.asarray(got), np.asarray(want)
    if g.shape != w.shape or np.any(g != w):
        fails += 1
        print(f"[FAIL] {what}: got {g.tolist()}, want {w.tolist()}")


QM = [  # real -> (mult, shift): frexp, then the mantissa * 2^31 rounded half away from zero
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


def main() -> None:
    for real, want in QM:
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
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
