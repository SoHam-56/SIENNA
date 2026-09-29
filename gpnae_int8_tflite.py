#!/usr/bin/env python3
"""GPNAE's int8 tanh and sigmoid (the lane model, bit-exact with the RTL at G2) against TFLite's int8 TANH and LOGISTIC; reported, not gated."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "GPNAE"))
import gpnae_model as gm  # noqa: E402
import tflite_oracle  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    lane = gm.Lane(gm.INT8, gm.read_rom(os.path.join(ROOT, "GPNAE", gm.coeff_file(gm.INT8))))
    q = np.arange(-128, 128, dtype=np.int64)
    L = ["GPNAE int8 lane against TFLite int8 (BUILTIN_REF), every int8 input; the input quantization is the converter's",
         f"{'op':<9}{'s_in':>11}{'z_in':>6}{'equal':>7}{'|d|=1':>7}{'max |d|':>8}{'TFLite-exact':>13}{'lane-exact':>11}"]
    for act, code, op in (("tanh", 3, "tanh"), ("sigmoid", 2, "logistic")):
        for case in gm.INT8_CASES[act]:
            tfl, s, z = tflite_oracle.activation_int8(op, case.s_in, case.z_in)
            c = case._replace(s_in=s, z_in=z)
            hw = lane.run(q, code, gm.int8_params(c, code))
            ex = gm.exact_int8(q, code, c)
            d = np.abs(hw - tfl)
            L.append(f"{op:<9}{s:>11.6f}{z:>6}{int((d == 0).sum()):>7}{int((d == 1).sum()):>7}{int(d.max()):>8}"
                     f"{int(np.abs(tfl - ex).max()):>13}{int(np.abs(hw - ex).max()):>11}")
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    open(a.report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
