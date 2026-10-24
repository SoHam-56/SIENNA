#!/usr/bin/env python3
"""Hierarchy check: at each define and at SET_PORT 0 and 1, the ArithmeticLibrary units elaborated under sienna_top are that precision's only.
Outside the GPNAE lanes (the mesh, the lane feed, the residual ADD, dropout) the set must be exactly the precision's; the lanes, GPNAE as in B,
may hold no unit of another precision. Args: --files (DESIGN_FILES and the package), --inc (the -I directories)."""
import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
UNITS = {  # every arithmetic unit of ArithmeticLibrary: per-format units, generic units and the int8-only units
    "fp32Multiplier", "fp32Adder", "fp32Divider", "bf16Multiplier", "bf16Adder", "bf16WideMultiplier", "fp32ToBf16",
    "int8Multiplier", "int16Multiplier", "int24Multiplier", "int32Adder", "fpMultiplier", "fpAdder", "fpMulWiden", "fpNarrow",
    "intMultiplier", "intAdder", "fxMac", "int8Requantizer", "int8QuantizedAdder",
}
INT8 = {"int8Multiplier", "int16Multiplier", "int24Multiplier", "int32Adder", "intMultiplier", "intAdder", "fxMac", "int8Requantizer", "int8QuantizedAdder"}
BF16 = {"bf16Multiplier", "bf16Adder", "bf16WideMultiplier", "fp32ToBf16", "fpMulWiden", "fpNarrow"}
FORBIDDEN = {"fp32": BF16 | INT8, "bf16": INT8, "int8": UNITS - INT8}  # another precision's units; bf16 builds keep fp32 sums and GPNAE's fp32 tail
FORBIDDEN["lanes fp32"] = FORBIDDEN["fp32"] - {"fpNarrow"}  # B's gpnae_poly elaborates fpNarrow at (8, 23), the fp32 identity (gpnae_poly.sv:274); Batch B's gpnae removes it
FORBIDDEN["default"] = FORBIDDEN["fp32"]
EXPECTED = {  # outside the lanes: the mesh's product and sums (and narrowing), dropout's keep multiply, the lane feed's requantize and the ADD
    "fp32": {"fp32Multiplier", "fp32Adder"},
    "bf16": {"bf16WideMultiplier", "fp32Adder", "fp32ToBf16", "bf16Multiplier"},
    "int8": {"int8Multiplier", "int32Adder", "int8Requantizer", "int8QuantizedAdder"},
}
EXPECTED["default"] = EXPECTED["fp32"]  # no define elaborates fp32
BENCH = {1: "TB_sienna_top", 0: "TB_model_run"}  # the set and the layer contract; a top with interface ports cannot be elaborated


def units_by_place(xml: str, top: str) -> tuple:
    """(origNames of the cells under dut outside the GPNAE lanes, those inside them, those outside dut) from the XML's cell hierarchy."""
    orig = {m.group(1): m.group(2) for m in re.finditer(r'<module\b[^>]*\bname="([^"]+)"[^>]*\borigName="([^"]+)"', xml)}
    pipe, lanes, other = set(), set(), set()
    for m in re.finditer(r'<cell\b[^>]*\bsubmodname="([^"]+)"[^>]*\bhier="([^"]+)"', xml):
        u, h = orig.get(m.group(1), m.group(1)), m.group(2)
        if not h.startswith(f"{top}.dut."):
            other.add(u)
        elif ".gpnae." in h:
            lanes.add(u)
        else:
            pipe.add(u)
    if not pipe:
        raise RuntimeError(f"no cell under {top}.dut in the XML")
    return pipe & UNITS, lanes & UNITS, other & UNITS


def elaborate(fmt: str, sp: int, files: list, inc: list, verilator: str, out: str) -> str:
    d = [] if fmt == "default" else [f"+define+FMT_{fmt.upper()}"]
    x = os.path.join(out, f"hier_{fmt}_sp{sp}.xml")
    top = BENCH[sp]
    cmd = [verilator, "--xml-only", "--xml-output", x, "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", *d, *[f"-I{i}" for i in inc],
           "--top-module", top, "--Mdir", os.path.join(out, f"obj_{fmt}_sp{sp}"), *files, os.path.join(ROOT, "testbenches", f"{top}.sv")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        errs = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("%Error")]
        raise RuntimeError(f"{fmt} SET_PORT {sp}: verilator exit {r.returncode}: {' | '.join(errs[:3]) or (r.stdout + r.stderr)[-400:]}")
    return open(x).read()


def lane_forbidden(fmt: str) -> set:
    return FORBIDDEN["lanes fp32"] if fmt in ("fp32", "default") else FORBIDDEN[fmt]


def judge(fmt: str, pipe: set, lanes: set, other: set) -> bool:
    return pipe == EXPECTED[fmt] and not (lanes & lane_forbidden(fmt)) and not other


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--inc", nargs="*", default=[])
    ap.add_argument("--verilator", default="verilator")
    ap.add_argument("--out", default=os.path.join(ROOT, "Verilator_hier"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(f, sp) for f in ("fp32", "bf16", "int8", "default") for sp in (1, 0)]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as ex:  # one elaboration per core
        xmls = dict(zip(jobs, ex.map(lambda j: elaborate(*j, a.files, a.inc, a.verilator, a.out), jobs)))
    places = {j: units_by_place(x, BENCH[j[1]]) for j, x in xmls.items()}
    bad = 0
    for (f, sp), (pipe, lanes, other) in places.items():
        ok = judge(f, pipe, lanes, other)
        bad += not ok
        print(f"check-hier {f} SET_PORT {sp}: {'OK' if ok else 'FAIL'}: pipeline {', '.join(sorted(pipe)) or 'none'}; lanes {', '.join(sorted(lanes)) or 'none'}"
              + ("" if ok else f" (want pipeline {', '.join(sorted(EXPECTED[f]))}, no {', '.join(sorted(lanes & lane_forbidden(f)))} in the lanes, nothing beside the DUT: {other or 'none'})"))
    ok = judge("fp32", *places[("bf16", 1)])  # control: bf16's hierarchy is not fp32's
    print(f"check-hier control (bf16 hierarchy judged as fp32): {'FAIL, as it must' if not ok else 'PASSED: the check cannot fail'}")
    bad += ok
    print(f"check-hier: {len(places)} elaborations, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
