#!/usr/bin/env python3
"""model_runner.pack_jobs / unpack, without a simulator: layout, refusals, and every job recovered from Y = A @ B + bias."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_runner as mr  # noqa: E402


def _models(rng, shapes, acts):
    return [{"W": rng.uniform(-1, 1, (K, C)), "bias": rng.uniform(-1, 1, C), "act": a, "req": None,
             "inputs": [rng.uniform(-1, 1, (m, K)) for m in ms]} for (K, C, ms), a in zip(shapes, acts)]


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


def test_partial_packing_and_entries():
    rng = np.random.RandomState(4)
    models = _models(rng, [(3, 3, [1]), (4, 2, [2])], ["selu", "selu"])
    job, recipe = mr.pack_jobs(models, 16, int8=False)
    pk = job["pack"]
    assert pk["shift"] == 2 and pk["map"][:4] == [0, 0, 0, 0] and pk["ents"][0][0] == "selu"  # same setting, one entry; blocks 2, 3 empty
    Y = job["terms"][0][0] @ job["terms"][0][1]
    assert np.all(Y[:, 8:] == 0), "an empty block's columns must stay zero before bias"


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


def test_selu_saturation_refused():
    # Review Focus 5: an int8 SELU entry whose lane input reaches x >= 487.29 must be refused, not packed.
    req = dict(mult=np.full(2, 1 << 30), shift=np.zeros(2, np.int64), zp=-128, amin=-128, amax=127, mx=(1 << 15) - 1, shx=0,
               mout=1, shout=0, zout=0)
    assert np.any(mr.regression.selu_saturates(req["mx"], req["shx"], req["zp"], np.arange(req["amin"], req["amax"] + 1))), \
        "the test's req does not saturate"
    m = {"W": np.ones((2, 2)), "bias": None, "act": "selu", "req": req, "inputs": [np.ones((1, 2))]}
    try:
        mr.pack_jobs([m], 16, int8=True)
    except ValueError:
        return
    raise AssertionError("pack_jobs accepted a saturating SELU entry")


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL {name}: {e!r}")
    print("ALL PASS" if bad == 0 else f"{bad} FAILED")
    sys.exit(1 if bad else 0)
