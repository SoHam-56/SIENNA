---
name: sienna-report
description: Use when updating, regenerating or extending SIENNA's journey-and-performance report (the HTML report in /proj/work/spramanik/sienna_report/, later a PDF), when adding a new measurement to it (a new format such as int8, a new N or tile size, a new model), or when someone asks how SIENNA's performance evolved from day one, what the credit interface or GPNAE changes bought, or where an old measured number came from.
---

# SIENNA report: the journey and the numbers

Soham's living report for seniors and his own notes. Started 2026-09-28; it replaced every earlier
generated report (their verbatim copies are in `history/`). Keep it updated as work lands; a PDF is made
only when Soham asks (headless Chrome: `google-chrome --headless=new --print-to-pdf=... file://...`).

## Build

`python3 /proj/work/spramanik/sienna_jobs/report/build_report.py /proj/work/spramanik/sienna_report/sienna_report.html`
runs on the login node (pure Python, no simulation). Charts are inline SVG from `report/svgplot.py`
(no matplotlib on this system). Check the render with a headless Chrome screenshot before telling Soham.

## Sections (Soham's order, 2026-09-28)

| # | Section | Data |
|---|---|---|
| 1 | How SIENNA got here: architecture changes, ready/valid to credits, GPNAE changes, day one against today | `report/journey.json` (curated from `journey.md`); day one measured from `sienna_jobs/dayone_tree` (SIENNA 0eeac7e + build fix e988875), runs `dayone_*` |
| 2 | Regressions: what each level tests; latency and throughput against N and tile size | g4 sweep runs `g4p_N*_T*_{fp32,bf16}` via `sienna_jobs/g4_table.py` |
| 3 | Real models (MLPerf Tiny) for each N and tile size | runs `ms_N*_T*_{fp32,bf16}` (`cmds/model_sweep.sh`, `model_runner.py --tile-size`) |
| 4 | Number formats: fp32, bf16, int8 | g4 runs, int8 perf runs `g8p_N*_T*_int8`, GEMM runs `s22_gemm_*` and `g8_gemm_int8`, `history/2026-09-28_gpnae_gate.txt` and the int8 G2 summary (`testbenches/int8/gpnae_int8_accuracy.json` in the int8 tree, read by build_report.py) |

## Rules

- Every number is measured in simulation or labeled as an estimate. The 950 MHz clock is an assumption.
- Tables and charts, highlights only; no dumps of every result. One consistent config per chart series.
- Day-one comparisons use the same tests at the same N and T. Day one ran one set at a time, so its
  cycles per set are load + latency (a lower bound).
- Accumulate configs: use `g4_table.py`'s values (perf_analysis before f1cc285 mis-reported them).

## history/

Verbatim copies of the old generated reports, file names prefixed with the date they were written.
They are the only record of the intermediate steps (2026-09-21 latency and throughput analyses, back-to-back
findings, architecture changes, credit sweep, GPNAE review, synthesis readiness, the bf16 and int8 gate reports).
