#!/usr/bin/env python3
"""model_runner.pack_jobs / unpack, without a simulator: layout, refusals, and every job recovered from Y = A @ B + bias."""
import os
import sys
import tempfile

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
    pk = job["pack"]
    assert pk["map"][:3] == [0, 1, 2] and set(pk["map"][3:]) == {0}, "each model's block must point at its own entry"
    assert [e[0] for e in pk["ents"][:3]] == ["tanh", "relu", "linear"] and len(pk["ents"]) == 8


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


def test_int8_shared_entry():
    rng = np.random.RandomState(8)
    a, b, c = _int8_model(rng, 0), _int8_model(rng, 0), _int8_model(rng, 0, "relu")
    b["req"] = dict(b["req"], mult=np.array([5, 6]))  # a column's own multiplier is not part of the entry
    job, _ = mr.pack_jobs([a, b, c], 16, int8=True)
    assert job["pack"]["map"][:3] == [0, 0, 1] and [e[0] for e in job["pack"]["ents"][:2]] == ["linear", "relu"]


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
            mr.LayerSim(N, lanes, d).run_job(job, "t")  # every refusal is raised before the simulator would run
        except ValueError as e:
            assert not os.listdir(d), f"{why}: a layer file was written before the refusal"
            return str(e)
    raise AssertionError(f"LayerSim accepted {why}")


def test_precheck_accepts_legal():
    for N, lanes in ((16, 32), (32, 32), (64, 64), (64, 128), (8, 8)):
        job = _packed(N)
        cfg, _, _, (M, C, rt, _) = mr.format_layer(job, N)
        mr.pack_precheck(job["pack"], cfg, (M, C, rt), N, lanes)


def test_precheck_lanes():
    assert "NUM_LANES" in _refused(_packed(64), 64, 32, "N = 64 at the default 32 lanes")
    assert "NUM_LANES" in _refused(_packed(16), 16, 24, "24 lanes at N = 16")


def test_precheck_collapse_k0():
    old = mr.COLLAPSE_K
    mr.COLLAPSE_K = 0
    try:
        assert "collapse-k 0" in _refused(_packed(16), 16, 32, "a packed layer on the collapse-k 0 mesh")
    finally:
        mr.COLLAPSE_K = old


def test_precheck_residual():
    job = _packed(16)
    A = job["terms"][0][0]
    job["terms"].append((np.ones_like(A), np.eye(16, dtype=np.float32)))  # an identity term is format_layer's residual
    assert "residual" in _refused(job, 16, 32, "a packed layer with a residual")


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
