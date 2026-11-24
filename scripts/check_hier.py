#!/usr/bin/env python3
"""Hierarchy check: at each define and at SET_PORT 0 and 1, the arithmetic units elaborated under sienna_top are that precision's only.
Outside the GPNAE lanes (the mesh, the lane feed, the residual ADD, dropout) and inside them (gpnae) each set must be exactly the precision's.
Args: --files (DESIGN_FILES and the package), --inc (the -I directories)."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # the SIENNA repository
UNITS = {  # every arithmetic unit of ArithmeticLibrary: per-format units, the generic integer units and the int8-only units
    "fp32Multiplier", "fp32Adder", "fp32Divider", "bf16Multiplier", "bf16Adder", "bf16WideMultiplier", "fp32ToBf16",
    "int8Multiplier", "int16Multiplier", "int24Multiplier", "int32Adder", "intMultiplier", "intAdder", "fixedPointMultiplyAdd",
    "int8Requantizer", "int8QuantizedAdder",
}  # and GPNAE's own fixedPointMultiplyAdd
INT8 = {"int8Multiplier", "int16Multiplier", "int24Multiplier", "int32Adder", "intMultiplier", "intAdder", "fixedPointMultiplyAdd",
        "int8Requantizer", "int8QuantizedAdder"}
BF16 = {"bf16Multiplier", "bf16Adder", "bf16WideMultiplier", "fp32ToBf16"}
FORBIDDEN = {"fp32": BF16 | INT8, "bf16": INT8, "int8": UNITS - INT8}  # another precision's units; bf16 builds keep fp32 sums and GPNAE's fp32 tail
FORBIDDEN["default"] = FORBIDDEN["fp32"]
EXPECTED = {  # outside the lanes: the mesh's product and sums (and narrowing), the lane feed's requantize and the ADD; dropout's 2.0 keep scale is an
    # exponent increment in both float formats since 833dee1 (G_POW2), so no float multiplier of its own
    "fp32": {"fp32Multiplier", "fp32Adder"},
    "bf16": {"bf16WideMultiplier", "fp32Adder", "fp32ToBf16"},
    "int8": {"int8Multiplier", "int32Adder", "int8Requantizer", "int8QuantizedAdder"},
}
EXPECTED["default"] = EXPECTED["fp32"]  # no define elaborates fp32
LANE_EXPECTED = {  # inside the lanes (GPNAE's check_hier.py): the Horner step, the operand and post units; bf16's fp32 tail and its narrowing
    "fp32": {"fp32Multiplier", "fp32Adder"},
    "bf16": {"bf16Multiplier", "bf16Adder", "fp32Multiplier", "fp32Adder", "fp32ToBf16"},
    "int8": {"int16Multiplier", "int24Multiplier", "int8Requantizer", "fixedPointMultiplyAdd"},
}
LANE_EXPECTED["default"] = LANE_EXPECTED["fp32"]
BENCH = {1: (0, "g_pipeline"), 0: (1, "g_layer")}  # TB_sienna's MODE and branch for the set and the layer contract; a top with interface ports cannot be elaborated


def hier_cells(tree: dict, top: str) -> list:
    """Every instance below the top module as (hierarchical name, origName of the module it instantiates), from Verilator's --json-only
    tree: a MODULE holds its CELLs, inside any named generate block (GENBLOCK), and a CELL's modp names the MODULE it instantiates."""
    modules = {m["addr"]: m for m in tree["modulesp"] if m.get("type") == "MODULE"}

    def cells(node, path):  # the CELLs in a module's body with their generate-block path, not descending into another module
        if isinstance(node, dict):
            t = node.get("type")
            if t == "CELL":
                yield node, path
                return
            if t == "GENBLOCK" and node.get("name"):
                path = f"{path}.{node['name']}"
            for v in node.values():
                if isinstance(v, (list, dict)):
                    yield from cells(v, path)
        elif isinstance(node, list):
            for v in node:
                yield from cells(v, path)

    root = next((m for m in modules.values() if m.get("origName") == top), None)
    if root is None:
        raise RuntimeError(f"no {top} module in the JSON")
    out, stack = [], [(root, top)]
    while stack:
        mod, path = stack.pop()
        for body in (v for k, v in mod.items() if isinstance(v, (list, dict))):
            for c, p in cells(body, path):
                child = modules.get(c.get("modp"))
                if child is None:
                    continue
                h = f"{p}.{c['origName']}"
                out.append((h, child.get("origName", child["name"])))
                stack.append((child, h))
    return out


def units_by_place(tree: dict, top: str) -> tuple:
    """(origNames of the modules under dut outside the GPNAE lanes, those inside them, those outside dut)."""
    pipe, lanes, other = set(), set(), set()
    for h, u in hier_cells(tree, "TB_sienna"):
        if not h.startswith(f"{top}.dut."):
            other.add(u)
        elif ".gpnae." in h:
            lanes.add(u)
        else:
            pipe.add(u)
    if not pipe:
        raise RuntimeError(f"no cell under {top}.dut in the JSON")
    return pipe & UNITS, lanes & UNITS, other & UNITS


def elaborate(fmt: str, sp: int, files: list, inc: list, verilator: str, out: str) -> dict:
    d = [] if fmt == "default" else [f"+define+FMT_{fmt.upper()}"]
    x = os.path.join(out, f"hier_{fmt}_sp{sp}.json")
    mode, _ = BENCH[sp]
    cmd = [verilator, "--json-only", "--json-only-output", x, "--json-only-meta-output", x[:-5] + ".meta.json", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style", *d, *[f"-I{i}" for i in inc],
           "--top-module", "TB_sienna", f"-GMODE={mode}", "--Mdir", os.path.join(out, f"obj_{fmt}_sp{sp}"), *files,
           os.path.join(ROOT, "testbenches", "top", "TB_sienna.sv")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        errs = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("%Error")]
        raise RuntimeError(f"{fmt} SET_PORT {sp}: verilator exit {r.returncode}: {' | '.join(errs[:3]) or (r.stdout + r.stderr)[-400:]}")
    with open(x) as fh:
        return json.load(fh)


def judge(fmt: str, pipe: set, lanes: set, other: set) -> bool:
    return pipe == EXPECTED[fmt] and lanes == LANE_EXPECTED[fmt] and not (lanes & FORBIDDEN[fmt]) and not other


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
        trees = dict(zip(jobs, ex.map(lambda j: elaborate(*j, a.files, a.inc, a.verilator, a.out), jobs)))
    places = {j: units_by_place(x, f"TB_sienna.{BENCH[j[1]][1]}") for j, x in trees.items()}
    bad = 0
    for (f, sp), (pipe, lanes, other) in places.items():
        ok = judge(f, pipe, lanes, other)
        bad += not ok
        print(f"check-hier {f} SET_PORT {sp}: {'OK' if ok else 'FAIL'}: pipeline {', '.join(sorted(pipe)) or 'none'}; lanes {', '.join(sorted(lanes)) or 'none'}"
              + ("" if ok else f" (want pipeline {', '.join(sorted(EXPECTED[f]))}, lanes {', '.join(sorted(LANE_EXPECTED[f]))}, nothing beside the DUT: {other or 'none'})"))
    ok = judge("fp32", *places[("bf16", 1)])  # control: bf16's hierarchy is not fp32's
    print(f"check-hier control (bf16 hierarchy judged as fp32): {'FAIL, as it must' if not ok else 'PASSED: the check cannot fail'}")
    bad += ok
    print(f"check-hier: {len(places)} elaborations, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
