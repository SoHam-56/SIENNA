#!/usr/bin/env python3
"""Groups of TFLite layers share one packed sienna_layer layer; every model's outputs must equal the interpreter's bit for bit."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import model_runner as mr  # noqa: E402
import tflite_int8_run as tr  # noqa: E402

GROUPS = [["fc8x8_linear", "fc8x4_relu", "fc6x8_relu6"], ["conv3x3_6x6x1x8_linear", "fc16x16_relu"]]


def model_of(path):
    """(pack_jobs model, output shape, interpreter's outputs) of one saved .tflite with its npz."""
    ref = np.load(path[:-len(".tflite")] + ".npz")
    layer = tr.load_layer(path)
    job, shape = tr.job_of(layer, ref["x_test"], ref)
    (X, W), = job["terms"]
    return {"W": W, "bias": job["bias"], "act": "linear", "req": job["req"], "inputs": [X]}, shape, ref["y_test"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tile-size", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--models", default=os.path.join(ROOT, "testbenches", "tflite_int8_pack"))
    a = ap.parse_args()
    work = os.path.join(ROOT, "testbenches", "results", "int8")
    os.makedirs(work, exist_ok=True)
    rep = open(os.path.join(work, f"tflite_pack_N{a.n}_T{a.tile_size}.log"), "w")
    sim = mr.LayerSim(a.n, a.lanes, work, "int8", a.tile_size)
    sim.build()
    bad = 0
    for g, names in enumerate(GROUPS):
        got = [model_of(os.path.join(a.models, f"{n}.tflite")) for n in names]
        job, recipe = mr.pack_jobs([m for m, _, _ in got], a.n, int8=True)
        Y, sets, cyc = sim.run_job(job, f"tflp{g}")
        for (m, shape, y), outs, n in zip(got, mr.unpack(Y, recipe), names):
            mism = int(np.sum(np.vstack(outs).reshape(shape) != y.reshape(shape)))
            bad += int(mism != 0)
            line = f"PACKED {n} (group {g}, b = {a.n >> job['pack']['shift']}): {mism}/{y.size} differ from the interpreter; {sets} sets, {cyc} cycles"
            print(line, flush=True)
            rep.write(line + "\n")
    tail = f"TFLITE_PACK: {sum(len(x) for x in GROUPS)} models in {len(GROUPS)} packed layers, {bad} failing"
    print(tail)
    rep.write(tail + "\nRESULT: " + ("PASSED" if bad == 0 else "FAILED") + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
