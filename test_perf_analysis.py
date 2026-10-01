#!/usr/bin/env python3
"""perf_analysis on saved PERF traces (N=16, T=4, bf16, 24 streamed sets) and its activation-stage model of the GPNAE lanes."""
import contextlib
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_analysis as pa  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbenches", "perf_fixtures")


def trace(name):
    return pa.events(open(os.path.join(FIX, f"{name}_N16_T4_bf16.trace")).read())


def test_accumulate_outputs_every_third_set():
    a = pa.analyse(trace("matmul_accum3_bias_linear_nopool"), 24, passes=3)
    assert a["first_latency"] == 133  # host start of pass 0 to the completion of pass 2, read off the trace by hand
    assert statistics.mean(a["steady"]) == 19.0  # cycles per depth pass: 57 per output of 3 passes
    outs = [x for x in a["sets"] if x["latency"] is not None]
    assert len(outs) == 8 and all(x["latency"] > 0 and x["out"] >= 0 for x in outs)
    assert all(x["latency"] is None and x["out"] is None for k, x in enumerate(a["sets"]) if (k + 1) % 3)


def test_single_pass_unchanged():
    a = pa.analyse(trace("matmul_relu_nopool"), 24)
    assert a["first_latency"] == a["sets"][0]["latency"] == 95  # the pre-fix report's value
    assert statistics.mean(a["steady"]) == 19.0
    assert len(a["sets"]) == 24 and all(x["out"] >= 0 for x in a["sets"])


@contextlib.contextmanager
def fmt(f):
    """perf_analysis's build format for the block, restored afterwards so no test depends on the order they run in."""
    old = pa.FMT, pa.MUL_LAT, pa.ADD_LAT
    pa.FMT, (pa.MUL_LAT, pa.ADD_LAT) = f, pa.UNIT_LAT[f]
    try:
        yield
    finally:
        pa.FMT, pa.MUL_LAT, pa.ADD_LAT = old


def test_rtl_lat():
    assert pa.UNIT_LAT == {"fp32": (8, 5), "bf16": (3, 5), "int8": (1, 1)}  # sienna_fmt_pkg's mul_lat, add_lat, as parsed
    assert (pa.FX_LAT, pa.REQ_LAT, pa.GROUP_K, pa.TAIL_CTX) == (2, 4, 16, 4)  # fx_lat, req_lat, gpnae_poly's K, TAIL_CONTEXTS


def test_lane_stage_int8():
    with fmt("int8"):  # no tail path, so the stage is data-free; each value derived from gpnae_poly_int8 and barrel_mac, and measured
        stage = lambda act, n: pa.lane_stage(act, {"SRAM_DEPTH": n * n, "NUM_LANES": 32})
        assert stage("selu", 16) == 96 and stage("tanh", 16) == 107  # sp_p16t2
        assert stage("selu", 32) == 327 and stage("tanh", 32) == 365  # sp_p32t2, sp_p32t4
        assert stage("selu", 8) == 53 and stage("tanh", 8) == 56  # sp_p8t2


def test_lane_group_float():
    with fmt("fp32"):  # one group, no tail element: last_i at per_lane + 3, so the first G_CAP at per_lane + 4
        assert pa.lane_cycles([0.0] * 8, "selu", 12) + 2 == 195 and pa.lane_cycles([0.0] * 2, "tanh", 6) + 2 == 183
    with fmt("bf16"):
        assert pa.lane_cycles([0.0] * 8, "selu", 12) + 2 == 143 and pa.lane_cycles([0.0] * 32, "tanh", 36) + 2 == 529


def test_tail_ops_fp32():
    with fmt("fp32"):  # multiply 8 + 2, add 5 + 2 cycles per gpnae_tail step, by hand
        assert pa.tail_ops(5.0, "tanh") == 10 * 17 + 10 + 7 + 4 * 10 + 7 + 10 + 7 + 10 + 7  # a = -10: four squarings
        assert pa.tail_ops(-5.0, "selu") == 10 * 17 + 10 + 3 * 17 + 10  # a = -5: three doublings
        assert pa.tail_ops(60.0, "tanh") == 7 + 10 + 7 + 10 + 7  # |a| = 120 > 104: e^a underflows
        assert pa.tail_ops(-4.5, "sigmoid") == 10 * 17 + 10 + 7 + 3 * 10 + 7 + 10 + 7 + 10  # x < 0 keeps s, no output step
        assert pa.lane_cycles([5.0] + [0.0] * 7, "tanh", 12) + 2 == 16 + 3 + 268 + 8 + 2  # the tail result holds the group's emit


if __name__ == "__main__":
    for t in (test_accumulate_outputs_every_third_set, test_single_pass_unchanged, test_rtl_lat, test_lane_stage_int8,
              test_lane_group_float, test_tail_ops_fp32):
        t()
        print(f"PASS {t.__name__}")
