# SIENNA

**A neural-network inference accelerator in SystemVerilog that runs real MLPerf Tiny models in fp32, bf16 and int8.**

SIENNA multiplies matrices on a streaming systolic mesh, applies SELU, sigmoid or tanh on a bank of polynomial activation lanes, then pools and applies dropout. Every stage works on a different set of data at the same time. A Python host stack lowers models onto it through framework-neutral layer records, with importers for TFLite, ONNX and PyTorch quantized layers, so a whole ResNet-8 inference runs on the hardware in simulation.

| | |
|---|---|
| **7.43 TFLOPS** sustained | 64 × 64 mesh streaming matrix multiplies at 95.5% of peak, in fp32, bf16 and int8 |
| **14 µs** per ResNet-8 inference | CIFAR-10 image on the 64 × 64 mesh, every multiply-accumulate on the RTL |
| **within 1% or one output step** of the exact functions | SELU, sigmoid and tanh on every activation lane at the 1% tier, in every format, at N = 8 to 64 (measured worst: fp32 0.50%, bf16 0.77%, int8 SELU 0.77%; int8 sigmoid and tanh within one output step) |
| **58× fewer cycles** per activation | than the published TYTAN engine it grew from (747 → 12.8 cycles for tanh on one fp32 lane) |
| **up to 16× faster** on small layers | by packing many small matrix multiplies into one |

Times assume a 950 MHz clock ([why](#performance)).

---

## Architecture

![SIENNA dataflow](docs/sienna_dataflow.svg)

Work moves through SIENNA in *sets*: one N × N matrix multiplied by another, then the activation function on the result. What makes it fast, stage by stage:

- **Matrix engine** ([SystolicMesh](https://github.com/SoHam-56/SystolicMesh)). The mesh is a grid of small systolic arrays, and each one computes its own block of the result from start to finish, so they all finish together and nothing has to be combined afterwards. Inputs, running sums and results each have spare buffers, so the next set loads while the current one multiplies and the previous result is read out. Weights that a layer reuses stay in an on-chip cache instead of being sent again.
- **Activation** ([GPNAE](https://github.com/SoHam-56/GPNAE)). Sigmoid, tanh and SELU are usually computed from e^x, which takes a long series of terms and, for sigmoid and tanh, a slow division. SIENNA instead approximates each function directly with a few short polynomial pieces, which need only multiplies and adds, and outputs the function's limit where the true value is already within tolerance of it. A tanh takes about 13 cycles in fp32 instead of 747 in the published design. The coefficients are data in a small memory the host can rewrite between layers (`coef_ready_o` says when; a write while anything is in flight is refused), in three accuracy tiers (within 1%, 5% and 10%) that each layer picks from; the lower tiers take fewer cycles. Each lane's multiply-add unit works on many inputs in turn, so it starts a new operation every cycle, and 32 lanes run side by side.
- **Pipeline.** Up to 15 sets are in flight at once, each stage working on a different one, and each set can use a different activation function. Every boundary, from the host interface to the output, is the same *credit link*: the receiving side grants one credit per free buffer slot, and the sending side sends only while it holds one. Nothing can overrun or be dropped, and a slow consumer at the output simply holds the pipeline back, stage by stage, until it is ready. The host interface is hardware (a DMA engine, or the layer engine's input streams).
- **One top.** `sienna_top` is the only top module. `SET_PORT` chooses how a host drives it, at the wires; by design neither choice adds a register or moves a cycle:
  - `SET_PORT = 0`, the layer contract: a layer's configuration, then its rows on two streams, which the layer engine below turns into sets.
  - `SET_PORT = 1`, the set contract: one put per set with its settings, its bias and its rows (a DMA engine, or a testbench).
  - `COPIES` puts several pipelines behind one set-contract host, which sends sets to them in turn.
- **Layer engine.** On the layer contract, `sienna_sched` runs a whole network layer on its own. It splits the layer into sets, reuses weights, and adds the bias and any skip connection. It handles depthwise convolution and, in int8, the rescaling of each output channel. In int8 it also adds residual (skip) connections, where a layer's output is the sum of a convolution and an earlier tensor. Each output tile becomes two sets: the first holds the convolution's int8 result in a small on-chip buffer (N² bytes), the second brings the earlier tensor through the mesh unchanged (multiplied by the identity matrix) and adds the two with one int8 adder per activation lane. Measured in simulation, the add adds 6 cycles of latency to each beat of that second set (an add set finishes 6 cycles later than a plain one), and the set after it waits until the adders are empty. The host only streams the data in.
- **Packing.** A layer much smaller than the mesh would leave most of it idle. Several such layers are packed side by side into one matrix multiply, each in its own block, and run together. The hardware ignores everything outside each layer's block, so every result is identical to running the layers one at a time.
- **Number format chosen at build time.** One define makes the whole pipeline fp32, bf16 or int8 (int32 sums, then a per-channel rescale to int8 for any scale within the host encoder's domain): `make FMT=bf16` builds with `+define+FMT_BF16`, and no define builds fp32. Only the format package (`sienna_fmt_pkg`) and the arithmetic units read the define; every other module takes its widths, latencies and format-dependent steps from the package, and carries plain words. A build contains only its own format's multipliers, adders and activation lanes (`make check-hier` checks this). bf16 multiplies bf16 numbers but adds them up in fp32 and rounds each finished result to bf16 once, because summing in bf16 lets rounding errors pile up over long dot products. bf16 and int8 builds match bit-exact Python models of the hardware, and every int8 output is checked against the exact real-valued result.

---

## Performance

> All numbers are measured in cycle-accurate Verilator simulation of the RTL. Times assume a **950 MHz** clock, the frequency the published TYTAN activation engine reached in a 45 nm process ([paper](#publication)). Synthesis of the full SIENNA pipeline is planned.
>
> The throughput, model and set-timing figures were measured in the final sweep of 2026-10-25 (SIENNA 4ba233c and 59b8d6d), at every N and tile size. The packing speedups and the bf16 model accuracy were measured on 2026-10-05 to 2026-10-10 (main up to 9fa414e) and were not measured again.

**How a set is timed.** A set is one N × N by N × N matrix multiply (2N³ operations) followed by its activation, pooling and dropout. The testbench counts cycles from the host's start to the set's completion signal:

| N = 16, fp32, bias + ReLU set | Cycles | Time |
|---|---|---|
| First result (latency) | 99 | 104 ns |
| Each further set while streaming (throughput) | 19 | 20 ns |

### Throughput while streaming sets

Measured in Verilator simulation (`make perf-analysis`), 24 sets per run, with the activation applied to every set; the FLOPS count the matrix multiply only (2N³ per set). Each cell gives fp32 · bf16 · int8, in GFLOPS at N = 16 and TFLOPS at N = 32 and 64 (GOPS and TOPS for int8), then the time per set in µs at 950 MHz:

| Activation | N = 16 (GFLOPS) | N = 32 (TFLOPS) | N = 64 (TFLOPS) |
|---|---|---|---|
| ReLU / linear, no pooling | 410 · 410 · 410 | 1.78 · 1.78 · 1.73 | 7.43 · 7.43 · 7.43 |
| SELU | 45 · 86 · 77 | 0.14 · 0.21 · 0.18 | 1.11 · 1.66 · 1.40 |
| sigmoid | 42 · 38 · 71 | 0.13 · 0.13 · 0.16 | 0.95 · 0.80 · 1.29 |
| tanh | 45 · 86 · 71 | 0.14 · 0.21 · 0.16 | 1.11 · 1.66 · 1.29 |

| Time per set (µs) | N = 16 | N = 32 | N = 64 |
|---|---|---|---|
| ReLU / linear, no pooling | 0.020 · 0.020 · 0.020 | 0.037 · 0.037 · 0.038 | 0.071 · 0.071 · 0.071 |
| SELU | 0.183 · 0.096 · 0.106 | 0.473 · 0.317 · 0.374 | 0.473 · 0.317 · 0.374 |
| sigmoid | 0.196 · 0.217 · 0.115 | 0.514 · 0.516 · 0.407 | 0.553 · 0.656 · 0.407 |
| tanh | 0.183 · 0.096 · 0.115 | 0.473 · 0.317 · 0.407 | 0.473 · 0.317 · 0.407 |

N = 16 and 32 use T = 4 and 32 activation lanes; N = 64 uses 128 lanes and T = 2, 4 or 8, which give the same result. Peak is 486 GFLOPS at N = 16, 1.95 TFLOPS at N = 32 and 7.78 TFLOPS at N = 64. With ReLU the pipeline keeps pace with the host. Sets with SELU, sigmoid or tanh are limited by the activation lanes. Larger tiles are limited by the mesh instead, which needs T² cycles a set: at N = 64 a ReLU set takes 256 cycles at T = 16 (0.27 µs, 1.95 TFLOPS) and 4 096 at T = 64 (4.3 µs, 0.12 TFLOPS). Every N and tile size is in `perf_final.log` of the project report.

Against the tree before the restructure (main up to 9fa414e), at N = 32 and 64 the new activation lanes take 20% to 60% fewer cycles per set for SELU and tanh in fp32 and bf16 (N = 64, T = 2: tanh 1 060 → 449 cycles in fp32, 759 → 301 in bf16), and int8 takes 5% to 8% more (SELU 329 → 355, tanh 367 → 387). At N = 8 and 16: fp32 SELU and tanh take up to 34% fewer cycles (0% to 2% fewer at T = 16), bf16 SELU and tanh 42% to 52% fewer, and int8 up to 5.8% more. Sigmoid is slower there: 14% more in fp32 at N = 8, and in bf16 4.7% more at N = 16 and 1.94 times as many at N = 8. The earlier sweeps did not measure sigmoid at N = 32 or 64.

### MLPerf Tiny models

fp32, every multiply-accumulate on the RTL; the host only reshapes tensors and applies softmax. Both sizes run on the layer engine at T = 2 (32 lanes at N = 16, 128 at N = 64), measured in the final sweep:

| Model | Task | MACs | Latency, N = 16 | Latency, N = 64 | Throughput, N = 64 |
|---|---|---|---|---|---|
| ResNet-8 | image classification (CIFAR-10) | 12.5 M | 54.1 µs | 14.0 µs | 1.79 TFLOPS |
| DS-CNN | keyword spotting | 2.6 M | 31.4 µs | 8.4 µs | 623 GFLOPS |
| MobileNet | visual wake words (96 × 96) | 7.5 M | 102.7 µs | 39.2 µs | 382 GFLOPS |
| Autoencoder | anomaly detection, 40 slices | 10.6 M | 53.6 µs (18.5 µs for one slice) | 6.5 µs (0.16 µs per slice) | 3.25 TFLOPS |

In fp32 the hardware picks the same class as the floating-point model on every classifier and reproduces the anomaly score to four decimal places, at both sizes. At N = 64, bf16 builds take up to 0.8% fewer cycles than these. Larger tiles take longer: ResNet-8 at N = 64 takes 14.2 µs at T = 4, 14.7 µs at T = 8 and 21.0 µs at T = 16.

**bf16 accuracy.** The same models in bf16 on 32 × 32 and 64 × 64 meshes at every tile size, against the floating-point model (measured in Verilator simulation on 2026-10-10, before the restructure; the final sweep's model points pass the same gate at every N and tile size). The layer error is the largest difference from an exact float64 calculation on the same bf16 inputs, as a share of that layer's largest output; rounding a single number to bf16 can be off by up to 0.39%:

| Model | Agreement with the float model | Largest layer error |
|---|---|---|
| ResNet-8 | same class on every CIFAR-10 image (3 per run) | 0.34% |
| DS-CNN | same class on every input (3 per run) | 0.33% |
| MobileNet | same class on every input (3 per run) | 0.38% |
| Autoencoder | anomaly score within 0.23% on every input (4 per run) | 0.33% |

Summing in bf16 instead, the autoencoder's anomaly score came out 2.7 to 2.8 times too high.

### Packing small layers

The same small jobs, run one at a time and packed into shared sets (N = 32), with identical results (measured before the restructure; the final sweep checks that packed results are identical at every N and tile size, not the speedup):

| Jobs | Format | One at a time | Packed | Speedup |
|---|---|---|---|---|
| 16 jobs of 2 output columns | fp32 | 12.45 µs | 1.36 µs | **9.1×** |
| 16 jobs of 2 output columns | int8 | 8.27 µs | 0.95 µs | **8.7×** |
| 16 jobs of 2 columns, 1 024 rows each | bf16 | 22.6 µs | 1.41 µs | **16×** |

### Compared with other accelerators

**Matrix engine.** Multiply-accumulates per cycle do not depend on the clock, so they compare designs fairly. Most edge accelerators are integer-only; SIENNA builds in fp32, bf16 or int8 from the same source, one format per build.

| Design | Origin | MACs per cycle | Formats | Peak, as published |
|---|---|---|---|---|
| **SIENNA, N = 64** | this work, RTL | 4 096 | fp32, bf16, int8 | 7.78 TFLOPS at 950 MHz (assumed) |
| **SIENNA, N = 32** | this work, RTL | 1 024 | fp32, bf16, int8 | 1.95 TFLOPS at 950 MHz (assumed) |
| **SIENNA, N = 16** | this work, RTL | 256 | fp32, bf16, int8 | 486 GFLOPS at 950 MHz (assumed) |
| Google TPU v1 [1] | industry, 28 nm silicon | 65 536 | int8 (int16 at reduced rate) | 92 TOPS at 700 MHz |
| NVIDIA NVDLA, full configuration [2] | industry, open RTL | 2 048 int8, 1 024 int16 / fp16 | int8, int16, fp16 | not published |
| Arm Ethos-U65 [3] | industry, licensable IP | 256 or 512 | int8, int16 | 0.5–1 TOP/s at 1 GHz |
| Arm Ethos-U55 [3] | industry, licensable IP | 32 to 256 | int8, int16 | 64–512 GOP/s at 1 GHz |
| Gemmini, default [4] | academic, open RTL | 256 | int8 with int32 sums | not published |
| Eyeriss [5] | academic, 65 nm silicon | 168 | 16-bit fixed point | 33.6 GMAC/s at 200 MHz |

[1] Jouppi et al., ISCA 2017, [arXiv:1704.04760](https://arxiv.org/abs/1704.04760) · [2] [nvdla.org](http://nvdla.org/primer.html), [nv_full spec](https://github.com/nvdla/hw/blob/nvdlav1/spec/defs/nv_full.spec) · [3] Arm [Ethos-U55](https://armkeil.blob.core.windows.net/developer/Files/pdf/product-brief/arm-ethos-u55-product-brief.pdf) and [Ethos-U65](https://armkeil.blob.core.windows.net/developer/Files/pdf/arm-ethos-u65-product-brief.pdf) product briefs · [4] Genc et al., DAC 2021, [arXiv:1911.09925](https://arxiv.org/abs/1911.09925) · [5] Chen et al., JSSC 2017, [paper](https://www.rle.mit.edu/eems/wp-content/uploads/2016/11/eyeriss_jssc_2017.pdf)

**MLPerf Tiny.** Latency per inference, set against the MLPerf Tiny v1.4 closed-division results [6]:

| Benchmark | SIENNA, N = 64 | SIENNA, N = 16 | Qualcomm Sensing Hub | Renesas RA8P1 + Arm Ethos-U55 | Asygn NNPA_16x | STM32 Cortex-M7, 280 MHz |
|---|---|---|---|---|---|---|
| Image classification (ResNet-8) | 14.0 µs | 54.1 µs | 98.5 µs | 340 µs | 2.69 ms | 41.7 ms |
| Keyword spotting (DS-CNN) | 8.4 µs | 31.4 µs | 66.5 µs | 108 µs | 567 µs | 11.5 ms |
| Visual wake words (MobileNet) | 39.2 µs | 102.7 µs | 118 µs | 362 µs | 1.63 ms | 24.4 ms |
| Anomaly detection (autoencoder) | 6.5 µs | 18.6 µs | 69.0 µs | 132 µs | 57.6 µs | 1.17 ms |

These are not like-for-like:
- **MLPerf Tiny:** the entries are measured on production boards running int8 models, with the whole system in the loop.
- **SIENNA:** the times come from RTL simulation of the accelerator alone, running the fp32 models at an assumed 950 MHz.

Read them as an indication of where the architecture stands.

[6] [MLCommons MLPerf Tiny v1.4 results](https://mlcommons.org/benchmarks/inference-tiny/)

---

## Verification

- `make regression` runs SIENNA's tests (the pipeline, the models, packed layers, the GEMM sweep and the int8 TFLite layers) at every size: N = 8, 16, 32 and 64, every tile size each N allows, in fp32, bf16 and int8, then prints one table of verdicts. SystolicMesh, GPNAE and ArithmeticLibrary each have their own `make regression` for their parts.
- The pipeline regression runs 32 tests in fp32, 33 in bf16 and 50 in int8 (nine of them residual layers) at every tile size. They pass on 8 × 8 to 64 × 64 meshes at every tile size, in both mesh modes at N = 8 and 16 and built back to back at every N. Each test streams several sets, and every element of every stage is checked.
- The systolic mesh matches its bit-exact model and the true function in all three formats from 8 × 8 to 64 × 64 at every tile size, except the collapse-k 0 points listed below (SystolicMesh's `make regression`).
- **The final sweep (2026-10-25/26, every N, tile size and format) is not complete.** SIENNA passes 435 of its 447 points; 4 did not complete and 8 were not run (they need a larger machine and were ruled out for cost):
  - The int8 `residual` points pass at every N and tile size. Each case builds a deliberately wrong ADD (its first zero point dropped) and requires the ADD gate to reject it. The test moves the conv's zero point away from 0, with the same scales, until that wrong ADD lands at least two output steps from the right answer: at N = 32 (`tiles`, -2 → -5) and 64 (`deep_uncached`, -3 → -8) the first draws put it within one step. Those two cases keep 84.9% and 100% of their ADD outputs inside the clamp (84.8% and 100% as drawn); every other case is unchanged.
  - Not completed, and not diagnosed: `fp32/N64_T2/model-ck0`, `fp32/N64_T2/gemm-ck0` and `bf16/N64_T2/gemm-ck0`. Their builds stopped after 6 to 10 hours (peaks 443 GB and 262 GB). The suspects are a full disk quota and the large-code link, which then lacked `-Wl,--no-relax`. A link past 2 GB of code is still unproven: the largest collapse-k 0 binary built since (fp32, N = 64, T = 8, with `--no-relax`) is 1.42 GB, 1.10 GB of it code, and passes its GEMM sweep.
  - Not completed: `bf16/N64_T2/model-ck0`, on a full disk quota.
  - Not run, by a cost ruling (not a memory limit): the eight collapse-k 0 points at N = 64 with T = 4 (fp32 and bf16 `model-ck0` and `gemm-ck0`) or T = 2 and 4 (int8 `gemm-ck0` and `tflite-ck0`). They were sized for 512 and 768 GB machines, which were ruled out for cost. Their memory was not measured; by the measured doubling for each halving of T, T = 4 would need about 200 GB in fp32, 110 GB in bf16 and 40 GB in int8, and int8 T = 2 about 90 GB (estimates).
  - SystolicMesh passes 117 of 123. Its six collapse-k 0 points at N = 64, T = 2 and 4, were not run under the same cost ruling.
- The arithmetic models are also checked against references of their own (hand-derived corners, an integer round-to-nearest-even, float64), so a rounding bug shared by an RTL unit and its model still fails ArithmeticLibrary's and SystolicMesh's regressions.
- Every int8 output (layers, the rescale, the residual add and the activations) is checked against the exact real-valued result: it must lie within one output step or 1% of it (for activations, the tolerance of the lane's tier: 1%, 5% or 10%), including SAME-padded convolutions with per-channel scales and input zero points, and residual blocks converted from TensorFlow Lite and ONNX-style layers. Agreement with the TensorFlow Lite interpreter is reported beside it, not gated: 447 of the 198 656 layer outputs differ from it.
- A protocol checker on every credit link and assertions in every stage run in every simulation, and a firing assertion fails the run. Results stay identical with the output stalled at random, and with it held mid-stream until the host is blocked.
- After every reset and every coefficient load, the testbench reads each lane's coefficient memory and compares it, field by field, with the image the host meant to write; a difference fails the run and names the word (`make coef-fault` shows it failing on one corrupted word).
- `make check-fmt` fails if any SIENNA RTL file outside the format package's allowlist names a format, and `make check-hier` fails if a build's hierarchy holds another format's units, at both host contracts.
- The default point builds fp32 with no define and must give the same words and cycles as the fp32 point; `pipeline-multi` runs two pipelines behind one host (`COPIES = 2`) in every format.

---

## Getting started

You need Verilator 5 with a C++20 compiler (its `--timing` mode needs coroutines) and Python 3 with the packages in `requirements.txt` (NumPy, `tflite`, TensorFlow); "Standalone machine" below has the tested versions.

```bash
git clone --recursive https://github.com/SoHam-56/SIENNA.git
cd SIENNA
make help                              # every target and the current configuration

make regression SMALL=1                # every test at N = 8 and 16, the sizes a 16 GB machine can build
make regression                        # every test at N = 8 to 64 (days one at a time: use make -jN, below)
make pipeline FMT=bf16                 # the pipeline tests only (TEST=tanh narrows them)
make tflite FMT=int8                   # int8 TFLite layers, checked against the exact result
make residual FMT=int8                 # int8 residual layers (skip connections), every stage checked
make model FMT=fp32                    # MLPerf Tiny models end to end (models/, or MODEL_DIR=<dir>)
make perf-analysis FMT=bf16            # cycle-level throughput per stage
make multi FMT=int8                    # two pipelines behind one host (COPIES=2)
make check-fmt check-hier              # no format named outside the package; each build holds only its format's units
```

Activate the venv in the shell that runs make: some checks call `python3` directly, and without the venv that is the system's Python, which may lack NumPy.

`FMT` (`fp32`, `bf16`, `int8`; it becomes the build's define), `N` (matrix size, default 16), `TILE` (array size, default 4) and `LANES` (activation lanes, default 32) apply to every target.

How `make regression` works:
- It is a list of *points*. A point is one check at one configuration, e.g. `bf16/N8_T2/pipeline` (the pipeline tests in bf16 at N = 8, tile size 2); the checks are `pipeline`, `model`, `pack`, `gemm`, `tflite` and `residual`. At every N the pipeline also runs as `pipeline-b2b` (built with `-DBACK_TO_BACK`, which adds the back-to-back pass to every test), and at N = 8 and 16 as `pipeline-ck0` (the collapse-k 0 mesh) and `pipeline-ck0-b2b` (both). `default/N8_T2/pipeline` builds fp32 with no define and must equal `fp32/N8_T2/pipeline`, and `<format>/N8_T2/pipeline-multi` runs two pipelines behind one host. `python3 regression.py --action points` lists them.
- Each point is its own make target, so `make -j4 regression SMALL=1` runs four at a time, each simulating on a quarter of the cores; `make regression/bf16/N8_T2/pipeline` runs one alone.
- Pick N in `make -jN` for your memory; a bare `make -j` starts every point at once. Measured, the largest process of one point peaks at about 0.8 GB at N = 8, 1.5 GB at N = 16, 3.4 GB at N = 32 and 13 to 18 GB at N = 64. The collapse-k 0 points (`-ck0`) need far more at small tiles: about 52 GB at N = 32, T = 2, and at N = 64 about 96 GB at T = 8, 48 GB at T = 16 and 262 to 443 GB at T = 2 (bf16, fp32); T = 4 was not measured (about 200 GB in fp32 by estimate); each point also runs its C++ compiles side by side, which these peaks do not add up, so leave room beyond N times the peak. Measured on 8-core machines: every point one at a time on 16 GB machines (all of `SMALL=1`), and `make -j4 regression SMALL=1 FMT=int8` on one 128 GB machine; other `-jN` and memory combinations were not measured.
- Each point builds in a private copy of the working tree (its files that git does not ignore) and writes its log to `testbenches/results/regression/`. The run ends with one table there, `regression.log`. The pipeline point builds its testbench once for each group of tests that share a hardware configuration and runs the others on that build.
- A point passes only on its own verdict (every test passed, every output matched its reference). The regression fails if any point fails or did not run.
- `FMT=int8` narrows it to one format. `make regression-summary` prints the table again from the logs.
- The model points run the four MLPerf Tiny models in fp32 and bf16 from `models/` (or `MODEL_DIR=<dir>`); a model whose input file is absent is skipped by name and listed under the table ("Standalone machine" below).
- Run time of `SMALL=1` (205 points), measured on 2026-10-25: split by format and kind over 19 machines with 8 to 32 cores, each running `make -k -j8` over its points, the whole run took 84 minutes (its longest machine). At N = 32 and 64 (final sweep, one point or a few per machine) the longest points were fp32/N64_T64/model (22.4 hours, sharing an 8-core machine with two other points), fp32/N64_T64/pipeline (19.4 hours alone on a 32-core machine), fp32/N64_T64/pipeline-b2b (19.0 hours), bf16/N64_T64/model (14.6 hours) and bf16/N64_T64/pipeline and pipeline-b2b (13.5 hours each); 88 of the 110 N = 32 points took under an hour, and the longest, fp32/N32_T2/model-ck0, 4.4 hours.

Two Python scripts drive everything:
- **`regression.py`** proves the hardware right: stimulus, golden models and every check behind `make regression`.
- **`scripts/model_runner.py`** is the host software. It loads a `.tflite` model, lowers each layer to matrix multiplies, packs small ones, and runs them on the simulated hardware. It is laid out in layers (numerics, frontend, middle end, device protocol, backends), so a driver for real silicon replaces only the bottom of the file.

The MLPerf Tiny reference models are in `models/` (Apache 2.0, from MLCommons; `models/NOTICE`).

The layout: `src/` holds the RTL by function (`top`: sienna_top and its set-side bundle; `pipeline`: sienna_pipe and sienna_sched; `lanes`: the requantize and ADD lanes; `fifo`, `pool`, `dropout`) with the synthesis file list `src/sienna_rtl.f`; `testbenches/top/` the one top bench TB_sienna, whose MODE (`make TB_MODE=pipeline|layer|sets|copies`) picks the pipeline regression, the layer contract, the set contract or COPIES copies; `testbenches/units/` the unit benches; `testbenches/` also holds the stimulus and fixtures the benches read; `scripts/` model_runner.py and the hierarchy check.

---

## Publication

SIENNA's activation engine grew out of TYTAN:

> S. Pramanik, V. William, A. Raha, D. Das, A. Mukherjee and J. L. Paluh, "TYTAN: Taylor-series based Non-Linear Activation Engine for Deep Learning Accelerators," VDAT 2025, Springer. [Book](https://link.springer.com/book/9783032263049) · [arXiv:2512.23062](https://arxiv.org/abs/2512.23062)

The [GPNAE](https://github.com/SoHam-56/GPNAE) repository has the design as published (tag [`tytan_vdat2025`](https://github.com/SoHam-56/GPNAE/tree/tytan_vdat2025)) and what changed since.

## Standalone machine

Everything here runs on any Linux machine; nothing depends on the authors' servers.

**You need:** GNU make, git, a C++ compiler with C++20 coroutines (g++ 10 or later), Verilator 5.035 built with that compiler (the version every regression runs on), and Python 3 with the packages in `requirements.txt` (tested with Python 3.11). The released Verilator 5.052 is not supported yet: it runs SystolicMesh's regression and SIENNA's fp32 and int8 points, but stops on bf16 builds (gpnae_tail.sv), on GPNAE's lane bench (a zero-delay warning in TB_gpnae) and on the lint gates (its new warnings, and its removal of --xml-only, which check-hier reads). A Verilator configured with a g++ older than 10 omits the `-fcoroutines` its `--timing` builds need: put a `g++` wrapper that adds it first on `PATH`.

```bash
git clone --recursive https://github.com/SoHam-56/SIENNA.git && cd SIENNA
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # every make below runs in this shell
models/fetch_cifar10.sh          # optional: CIFAR-10 for the resnet8 model (a 163 MB download)
models/fetch_inputs.sh           # optional: the kws and ad01 models' input files (MLCommons Tiny, about 170 kB)
make -j2 regression SMALL=1
```

**Choosing N in `make -jN`:** the largest process of one point peaks at about 0.8 GB at N = 8 and 1.5 GB at N = 16 (above), and each point also runs its C++ compiles side by side, so leave room beyond N times the peak; one point at a time fits in 16 GB.

**What is skipped:** the model points run the MLPerf Tiny models in `models/` (the default `MODEL_DIR`). A model whose input file is not there is skipped by name inside its point and listed under the table as `skipped inside passing points: ...`: resnet8 without CIFAR-10 (`models/fetch_cifar10.sh` fetches it), kws without `marvin_617de221_0.wav` and ad01 without `normal_id_01_00000000_hist_librosa.bin` (`models/fetch_inputs.sh` fetches both). vww always runs. With `MODEL_DIR` pointing at a directory that holds no model, the model points show `SKIPPED (no MODEL_DIR)` and do not fail the run. `models/NOTICE` gives each file's source and license.

---

## Running with VCS

`make regression` runs on Verilator by default. `SIM=vcs` runs the same points, through the same gates, on Synopsys VCS, a 4-state simulator: it also models unknown (X) and floating (Z) bit values, which Verilator, a 2-state simulator, turns into 0 or 1.

**The authors have not run the VCS path yet** (no VCS license where it was written). The commands, gates and parsers are in place and were checked without VCS; your run is the first real one.

**You need:** `vcs` on your `PATH` with a working license, the C++ compiler VCS uses, GNU make, and Python 3 with `pip install -r requirements.txt`.

**Run it:**

```bash
make -j2 regression SIM=vcs SMALL=1
```

**Read the table** (also written to `testbenches/results/regression/regression.log`). Each row is one check: its name, its verdict, its time and its log. The last line is the overall verdict, for example `SIENNA REGRESSION PASS: 205 of 205 gated points passed`.

| Verdict | Meaning |
|---|---|
| `PASS` | the check met the same gate as on Verilator: every output against its reference and the true function |
| `FAIL (...)` | it did not; the bracket says why, the log (path on the row) says where |
| `N/A (...)` | not used here: every point runs on VCS |

**If a row fails,** send back `testbenches/results/regression/regression.log` and the log of each failing row. A line such as `FAIL: X in output word 37 of test <name>` means the design produced an unknown value where a number was expected: a real finding for the authors, not a set-up problem. A license or compile failure shows as `FAIL (exit N)` with VCS's message in the log.

**Options:** `TRACE=vcd` waveforms, `XPROP=1` X-propagation, `RANDINIT=1 SEED=<n>` random register power-up instead of X. Without them a VCS build is plain: no debug access and no waveform.

What maps to what:

| Verilator | VCS | Note |
|---|---|---|
| `--binary --timing --assert` | `vcs -full64 -sverilog -timescale=1ns/1ps` | VCS runs delays and assertions natively |
| `--top-module X` | `-top X` |  |
| `-I<dir>`, `+define+X`, `-DX` (also in `EXTRA_FLAGS`) | `+incdir+<dir>`, `+define+X` |  |
| `-G<name>=<value>` (sizes, `EXTRA_FLAGS`) | `-pvalue+<top>.<name>=<value>` | top-level parameters, as with `-G` |
| `--trace` (`TRACE=vcd`, `fst`) | `-debug_access+all +define+ENABLE_TRACE` (`TRACE=vcd`; VCS writes VCD only) | without it, `+vcs+dumpvarsoff`: no waveform |
| `--x-initial unique`, `+verilator+rand+reset+2` | `+vcs+initreg+random` (compile), `+vcs+initreg+<SEED>` (run): `RANDINIT=1 SEED=<n>` | VCS's default is X power-up, stricter than Verilator's zeros |
| (none) | `-xprop=tmerge`: `XPROP=1` | X-propagation through `if` and `case` |
| `+verilator+error+limit+N` | nothing | VCS carries on after `$error` by default |
| `-j`, `--build-jobs`, `--threads`, `SIM_THREADS` | `-j<cores>` (compile) | VCS simulates on one thread |
| `OPT_FAST=-O0`, `LARGE_CODE=1`, `USE_CCACHE` | not applicable | Verilator build workarounds |
| `/* verilator lint_off ... */`, `lint_waivers.vlt` | ignored | comments to VCS; the waiver file is Verilator's |
| `make sim`'s key from Verilator's dependency file | the same key over every source and include file | `make sim SIM=vcs` reuses a VCS build |
| `perf-analysis` (`VERILATOR_DIR`) | the same flow (`VCS_DIR`) | not a regression point |
| `check-hier`, `lint` | Verilator only | not regression points |

---

## Author

Soham Pramanik · [LinkedIn](https://www.linkedin.com/in/soham-pramanik-224004271/)
