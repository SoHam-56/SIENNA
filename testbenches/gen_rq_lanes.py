#!/usr/bin/env python3
"""Vectors for TB_requant_lanes through its channel map, expected values from ipu.requant in tflite_ref's rounding; arg: output .mem path."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SystolicMesh", "ArithmeticLibrary", "Common", "models"))
import ipu  # noqa: E402
import tflite_ref  # noqa: E402

N, LANES, SETS = 16, 32, 6  # TB_requant_lanes' geometry
PER = N * N // LANES


def main(path: str) -> None:
    rng = np.random.RandomState(8)
    out = [SETS]
    for s in range(SETS):
        mult = rng.randint(1 << 30, 1 << 31, N).astype(np.int64)
        if s == 0:
            mult[0], mult[1] = (1 << 31) - 1, 1 << 30  # the largest multiplier and the smallest normalized one
        shift = -((s * N + np.arange(N)) % 32).astype(np.int64)  # every right shift 0..31 across the sets
        zp = int(rng.randint(-128, 128))
        amin, amax = [(-128, 127), (zp, 127), (zp, min(127, zp + 50)), (-128, 127), (-40, 40), (-128, zp)][s]
        amin, amax = min(amin, amax), max(amin, amax)
        small = rng.randint(-(1 << 15), 1 << 15, (PER, LANES))
        full = rng.randint(-(1 << 31), (1 << 31) - 1, (PER, LANES))
        acc = np.where(rng.rand(PER, LANES) < 0.5, small, full).astype(np.int64)
        if s == 0:
            acc[0, :4] = [-(1 << 31), (1 << 31) - 1, 0, -1]
        c = (np.arange(LANES)[None, :] * PER + np.arange(PER)[:, None]) % N  # channel of lane k at beat b
        want = ipu.requant(acc, mult[c], shift[c], zp, amin, amax, tflite_ref.ROUNDING)
        out += [zp, amin, amax] + mult.tolist() + shift.tolist()
        for b in range(PER):
            out += acc[b].tolist() + [int(v) for v in want[b]]
    with open(path, "w") as f:
        f.write("".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in out))


if __name__ == "__main__":
    main(sys.argv[1])
