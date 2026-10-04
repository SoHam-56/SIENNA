#!/usr/bin/env python3
"""regression._parse_log: a log holding an assertion firing or a %Error fails, whatever the testbench's counts say; a clean log passes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression as reg  # noqa: E402

CLEAN = """  Total     : 512
  Exact     : 512
  Tol pass  : 0
  Failed    : 0
pipeline_complete_o asserted @ 3165000  (311 cycles)
RESULT: PASSED
"""
FIRED = ("[3165000] %Error: sienna_top.sv:1128: Assertion failed in TB_sienna_top.dut.a_pack_one_pass: "
         "sienna_top: a packed set cannot be a partial sum or continue one\n")


def test_clean_log_passes():
    r = reg._parse_log(CLEAN)
    assert r["status"] == "PASS" and r["total"] == 512 and r["cyc"] == 311, r


def test_assertion_firing_fails():
    r = reg._parse_log(CLEAN + FIRED)  # the TB's own counts still read clean
    assert r["status"] == "ASSERT" and r["failed"] == 1 and "a_pack_one_pass" in r["fired"], r


def test_assertion_without_error_prefix_fails():
    r = reg._parse_log(CLEAN + "Assertion failed in TB_sienna_top.dut.a_credit_accept\n")
    assert r["status"] == "ASSERT", r


def test_error_line_fails():
    r = reg._parse_log(CLEAN + "%Error: TB_sienna_top.sv:12: some runtime error\n")
    assert r["status"] == "ASSERT", r


def test_warning_is_not_a_firing():
    r = reg._parse_log("%Warning-UNUSEDSIGNAL: sienna_top.sv:40: Signal is not used\n" + CLEAN)
    assert r["status"] == "PASS", r


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} PASSED")
