#!/usr/bin/env python3
"""The Makefile's FMT flow without building: the pkg-check guard on fixture packages, the parse-time FMT and COLLAPSE_K checks, regression.py --action pkg's refusals, and FMT_FIELDS_* against regression.FORMATS."""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import regression as reg  # noqa: E402

ENV = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", "MAKEOVERRIDES")}  # an outer make's variables would leak in


def make_n(*args) -> subprocess.CompletedProcess:
    """make -n from the repo root: prints recipes, runs nothing but parse-time checks."""
    return subprocess.run(["make", "-s", "-n", *args], cwd=ROOT, env=ENV, capture_output=True, text=True)


def write_pkg(d: str, fmt: str, n: int = 16, tile: int = 4, lanes: int = 32, drop: tuple = ()) -> str:
    """A fixture test_config_pkg.sv in d, written by regression.py's own writer, minus the localparams in drop."""
    path = os.path.join(d, f"pkg_{fmt}_{n}_{tile}_{lanes}{'_old' if drop else ''}.sv")
    items = reg._config_items({"n": n, "tile_size": tile, "lanes": lanes}, fmt, "relu", 4, 15, 1, [], False, 1)
    reg.write_sv_package(path, [x for x in items if x[0] not in drop])
    return path


def guard(pkg: str, *args) -> tuple:
    """(exit, output) of the shell make -n prints for pkg-check GEN_PKG=0 on pkg."""
    r = make_n("pkg-check", "GEN_PKG=0", f"PKG_FILE={pkg}", *args)
    assert r.returncode == 0, r.stderr
    g = subprocess.run(["bash", "-c", r.stdout], capture_output=True, text=True)
    return g.returncode, g.stdout + g.stderr


def test_guard_passes_matching_package():
    with tempfile.TemporaryDirectory() as d:
        for f in reg.FORMATS:
            rc, out = guard(write_pkg(d, f), f"FMT={f}")
            assert rc == 0 and f"matches FMT={f} N=16 TILE=4 LANES=32" in out, out


def test_guard_rejects_other_format():
    with tempfile.TemporaryDirectory() as d:
        for f in reg.FORMATS:
            for want in reg.FORMATS:
                if want != f:
                    rc, out = guard(write_pkg(d, f), f"FMT={want}")
                    assert rc != 0 and f"but FMT={want} needs" in out, (f, want, out)


def test_guard_rejects_pre_format_package():
    with tempfile.TemporaryDirectory() as d:
        rc, out = guard(write_pkg(d, "fp32", drop=("EXP_W", "MAN_W", "IS_INT")), "FMT=fp32")
        assert rc != 0 and "EXP_W=missing" in out, out


def test_guard_checks_geometry():
    with tempfile.TemporaryDirectory() as d:
        big = write_pkg(d, "bf16", 32, 8)
        rc, out = guard(big, "FMT=bf16")  # a stale same-format package of another size
        assert rc != 0 and "N=32 TILE_SIZE=8 NUM_LANES=32, but the build asks for N=16 TILE=4 LANES=32" in out, out
        rc, out = guard(big, "FMT=bf16", "N=32", "TILE=8")
        assert rc == 0 and "N=32 TILE=8 LANES=32" in out, out
        rc, out = guard(write_pkg(d, "int8"), "FMT=int8", "LANES=64")
        assert rc != 0 and "LANES=64" in out, out


def test_guard_missing_file():
    with tempfile.TemporaryDirectory() as d:
        rc, out = guard(os.path.join(d, "none.sv"), "FMT=fp32")
        assert rc != 0 and "missing or unreadable" in out, out


def test_parse_time_rejections():
    r = make_n("regression", "FMT=fp16")
    assert r.returncode != 0 and "Invalid FMT=fp16: must be one of fp32, bf16, int8" in r.stderr, r.stderr
    for t in ("regression", "verilator", "lint", "pkg"):
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode != 0 and "sienna_ck0.sh" in r.stderr, (t, r.stderr)
    r = make_n("lint", "COLLAPSE_K=2")
    assert r.returncode != 0 and "Invalid COLLAPSE_K=2" in r.stderr, r.stderr
    for t in ("help", "sm-verilator"):  # the only targets collapse-k 0 is meaningful for, or harmless in
        r = make_n(t, "COLLAPSE_K=0")
        assert r.returncode == 0, (t, r.stderr)


def test_pkg_action_refusals():
    run = lambda *a: subprocess.run([sys.executable, "regression.py", "--action", "pkg", *a], cwd=ROOT, capture_output=True, text=True)
    r = run("--format", "fp32", "--test", "no_such_test")
    assert r.returncode != 0 and "no pipeline test named 'no_such_test'" in r.stderr, r.stderr
    only8 = next(t["name"] for t in reg.PIPELINE_TESTS if t.get("formats") == ("int8",))
    r = run("--format", "fp32", "--test", only8)
    assert r.returncode != 0 and f"test '{only8}' runs only in int8, not fp32" in r.stderr, r.stderr
    assert "formats" not in next(t for t in reg.PIPELINE_TESTS if t["name"] == reg.PKG_DEFAULT_TEST)  # the default runs everywhere


def test_fmt_fields_match_formats():
    mk = open(os.path.join(ROOT, "Makefile")).read()
    fields = {f: tuple(int(x) for x in v.split()) for f, v in re.findall(r"^FMT_FIELDS_(\w+) = ([\d ]+)$", mk, re.M)}
    assert re.search(r"^FORMATS = (.*)$", mk, re.M).group(1).split() == list(reg.FORMATS)
    assert fields == {f: (*reg.FORMATS[f], int(f == "int8")) for f in reg.FORMATS}, fields


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} PASSED")
