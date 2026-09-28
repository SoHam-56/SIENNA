#!/usr/bin/env python3
"""perf_analysis on saved PERF traces (N=16, T=4, bf16, 24 streamed sets): accumulate configs output every passes-th set."""
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


if __name__ == "__main__":
    for t in (test_accumulate_outputs_every_third_set, test_single_pass_unchanged):
        t()
        print(f"PASS {t.__name__}")
