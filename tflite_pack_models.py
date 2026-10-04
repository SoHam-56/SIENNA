#!/usr/bin/env python3
"""Five small int8 TFLite layers for packing, saved as tflite_oracle saves its G4 models; arg: output directory."""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import tflite_oracle as to  # noqa: E402

MODELS = {  # name: (layer, act, inputs, outputs); the first three fit blocks of 8, the last two blocks of 16
    "fc8x8_linear": ("fc", "none", 8, 8),
    "fc8x4_relu": ("fc", "relu", 8, 4),
    "fc6x8_relu6": ("fc", "relu6", 6, 8),
    "conv3x3_6x6x1x8_linear": ("conv", "none", 1, 8),
    "fc16x16_relu": ("fc", "relu", 16, 16),
}


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    rounding = open(os.path.join(ROOT, "testbenches", "tflite_int8", "rounding.txt")).read().strip()
    for name, (layer, act, cin, cout) in MODELS.items():
        to.FC_IN, to.FC_OUT, to.CONV_CIN, to.CONV_COUT, to.CONV_HW = cin, cout, cin, cout, 6
        model, lo, hi = to.build_model(layer, act, 0)
        it = to.interpreter(model, to.REF)
        p = to.extract(it, model, layer, act)
        xs = to.test_inputs(p, lo, hi, 32, np.random.default_rng(2000))
        ys = to.invoke_all(it, xs)
        unclamped = float(np.mean((ys != p["amin"]) & (ys != p["amax"])))
        to.save(out, name, model, p, xs, ys, rounding)
        print(f"{name}: in zp {p['in_zp']} out zp {p['out_zp']} act [{p['amin']}, {p['amax']}] unclamped {100 * unclamped:.1f}% "
              f"y [{ys.min()}, {ys.max()}] w_scales {p['w_scales'].size}")
    shutil.copy(os.path.join(ROOT, "testbenches", "tflite_int8", "rounding.txt"), out)


if __name__ == "__main__":
    main(sys.argv[1])
