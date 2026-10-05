# SIENNA

A parameterised neural-network accelerator in SystemVerilog, with the host software that runs real models on it.

A set of operands flows through four stages:
1. a systolic-mesh matrix multiply with bias;
2. a bank of polynomial activation lanes;
3. 2-D max pooling;
4. dropout.

Every stage builds in **fp32, bf16 or int8**. int8 uses int32 accumulation and TFLite requantization, and matches the TFLite interpreter bit for bit. The design streams sets back to back with all stages overlapped. It can schedule a whole network layer in hardware, and can pack several small jobs into one matrix multiply.

**Dependencies:** Verilator 5 · a C++20 compiler (`--timing` needs coroutines) · Python ≥ 3.10 · NumPy · the `tflite` Python package for the model flows

```bash
git clone --recursive git@github.com:SoHam-56/SIENNA.git
cd SIENNA && make help
```

---

## Architecture

```
            ┌──────────────┐  wide   ┌─────────────────────┐        ┌────────────┐      ┌─────────┐
 host ─────▶│ SystolicMesh │──read──▶│ activation lanes    │──────▶ │ Maxpool_2D │────▶ │ dropout │────▶ results
 A, B rows  │ N×N, T×T     │         │ NUM_LANES × GPNAE   │ buffer │ per lane   │      │ (LFSR)  │
 (credits)  │ arrays       │         │ (int8: requantize)  │        └────────────┘      └─────────┘
            └──────────────┘         └─────────────────────┘
```

| Stage | What it does |
|---|---|
| **Matrix multiply** ([SystolicMesh](https://github.com/SoHam-56/SystolicMesh)) | C = A × B + bias on an N×N mesh of T×T systolic arrays. Two staging banks, four partial-sum banks per PE and four result banks let host loads, compute and read-out overlap. Partial sums can stay in the PEs across sets, so a deep K accumulates without leaving the mesh. A weight cache holds B tiles that are reused across row tiles |
| **Activation** ([GPNAE](https://github.com/SoHam-56/GPNAE)) | `NUM_LANES` (default 32) `gpnae_poly` lanes evaluate SELU, sigmoid or tanh as fitted polynomials, with a tail unit for inputs past the fits; ReLU and linear bypass them. All lanes fill in parallel from the mesh's wide read port. In int8, `requant_lanes` first requantizes each lane's int32 sums with a per-channel multiplier, shift and zero point (TFLite's kernels), and the lanes run in fixed point |
| **Pooling** | A window dispatcher re-reads the activation buffer in pooling-window order and spreads windows across the `Maxpool_2D` lanes (`POOL_H` × `POOL_W`, stride, padding); a 1×1 window passes data straight through |
| **Dropout** | LFSR dropout, active in training mode; inference passes values through |

### Top levels

- **`src/sienna_top.sv`, the set pipeline.** The host loads one N×N set at a time and pulses start; a credit counter (`SETS_IN_FLIGHT`, default 15, every set the banks can hold) says when it may. Each set carries its own activation and settings, and results leave in issue order with a one-cycle completion pulse and the set's id.
- **`src/sienna_layer.sv`, the layer engine.** One network layer, C = A × B (+ bias, + residual), is scheduled entirely in hardware. The host writes a configuration and then streams A and B in a fixed order, and the engine makes every per-set decision itself:
  - row, column and depth tiles;
  - weight caching;
  - depthwise layers;
  - the int8 epilogue: per-column bias, multiplier and shift.
- **`src/sienna_multi.sv`.** `COPIES` independent pipelines behind one host port.

### Number formats

| Format | Operands and results | Sums | Reference the RTL is checked against |
|---|---|---|---|
| fp32 | IEEE binary32 | fp32 | NumPy float64, within tolerance |
| bf16 | bfloat16 | bf16 | a bit-exact model of every unit in bf16 |
| int8 | int8 codes with scale and zero point | int32, requantized to int8 | a bit-exact model of TFLite's int8 kernels; the TFLite interpreter |

`FMT=fp32|bf16|int8` selects the format for every build and run.

### Packing small jobs

Small layers waste most of an N×N mesh. A **packed set** puts up to N/b independent jobs on the diagonal of B in blocks of b columns. The PEs skip the products outside their own block. An 8-entry table gives each block its own activation and, in int8, its own requantization. `model_runner.pack_jobs` builds such a set and `unpack` splits the result.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/` | `sienna_top`, `sienna_layer`, `sienna_multi`, `requant_lanes`, `fwft` |
| `Maxpool/`, `Dropout/` | `Maxpool_2D`, `dropout` |
| `testbenches/` | `TB_sienna_top` (pipeline), `TB_model_run` (layer engine), `TB_sienna_model` (host-driven sets), `TB_sienna_multi`, unit benches; `tflite_int8/` holds the int8 TFLite test models |
| `SystolicMesh/`, `GPNAE/` | submodules, each with its own README, regression and [ArithmeticLibrary](https://github.com/SoHam-56/ArithmeticLibrary) submodule |
| `regression.py` | hardware sanity: golden models, stimulus, the pipeline regression and every hardware check |
| `model_runner.py` | the host software stack (below) |

---

## Running it

Everything runs from the repo root through the `Makefile`; `FMT`, `N` (16), `TILE` (4) and `LANES` (32) apply to every target.

| Command | What it runs |
|---|---|
| `make regression FMT=int8` | **One verdict.** It runs: the tool self-tests, the SystolicMesh and GPNAE regressions, the pipeline regression, packed layers, a GEMM sweep and, in int8, the TFLite layers. It prints a table and fails if any gated step fails |
| `make pipeline FMT=bf16 [TEST=tanh]` | the pipeline regression on `TB_sienna_top` (32 tests in fp32 and bf16, 41 in int8) |
| `make pack FMT=int8` | packed layers against each job run alone |
| `make gemm FMT=int8 [QUICK=1]` | GEMM shape sweep on the layer engine |
| `make tflite FMT=int8` | single-layer int8 TFLite models, bit for bit against the interpreter |
| `make model FMT=bf16 MODEL_DIR=<dir>` | MLPerf Tiny float models end to end |
| `make perf-analysis FMT=bf16` | cycle-level throughput and latency per stage |
| `make sm-verilator` / `make gpnae-verilator` | the submodules' own regressions in `FMT` |
| `make verilator [TRACE=fst]` / `make lint` | one build of `TB_sienna_top`, optionally with a waveform; lint only |

`PYTHON=<venv>/bin/python` selects the interpreter (the `tflite` and `model` flows need the `tflite` package).

---

## Host software stack

`model_runner.py` is the reference host software: a compiler and driver for SIENNA in one file. It is laid out as the layers a software team builds on:

| Section | Contents |
|---|---|
| 1. Numerics | format rounding, TFLite quantization and requantization, the int8 layer golden, TFLite's reference kernels |
| 2. Frontend | loads a `.tflite` and lowers each op to a job, Y = Σᵢ Xᵢ Wᵢ + b followed by an activation (conv via im2col, depthwise, residual add, average pool) |
| 3. Middle end | tiling into N×N sets or a layer's streams; packing small jobs |
| 4. Device protocol | the layer file `TB_model_run` reads (L configuration, Q requantize, P / E packing lines, rows of N words, int8 epilogue) and the result file it writes |
| 5. Device build | the compile-time package the simulators build against |
| 6. Backends | `RtlLayer` (layer engine), `RtlSets` (host-driven sets), `Emulator` (NumPy): `build()`, `run_job(job, tag)` |
| 7–8. Runtime, CLI | executes a model graph through a backend; `--action model`, `--action tflite [--pack]` |

A driver for real hardware replaces sections 5 and 6 and keeps everything above them.

```bash
python3 model_runner.py --model-dir <mlperf_tiny_models> --format bf16        # every MLPerf Tiny model on the layer engine
python3 model_runner.py --model-dir <dir> --models resnet8 --engine sets      # the host drives sienna_top set by set
python3 model_runner.py --action tflite                                       # int8 TFLite layers, bit for bit
python3 model_runner.py --action tflite --pack --n 32                         # several TFLite layers packed into one
```

---

## Verification

| Check | fp32 | bf16 | int8 |
|---|---|---|---|
| `make regression` (N = 16, T = 4), gated steps | 6 / 6 pass | 6 / 6 pass | 7 / 7 pass |
| Pipeline regression, N = 8, 16 (T = 2 … 16) and 32 | 27–32 tests per run, all pass | 27–32 tests per run, all pass | 39–41 tests per run, all pass |
| SystolicMesh, N = 8 / 16 / 32, every tile size, both collapse modes | bit-exact against its model | bit-exact | bit-exact |
| GPNAE lane | within 1% of the exact functions (worst 0.39%) | bit-exact against its model | bit-exact against its model |
| TFLite int8 layers (conv 3×3 SAME-padded with per-channel scales and input zero points, fully connected) | | | 5 / 5 models, 0 outputs differ from the interpreter |
| Packed TFLite layers (5 models in 2 packed sets, N = 32, T = 2 … 16) | | | 0 outputs differ |
| Lint (`make lint`) | clean | clean | clean |

Pipeline outputs are compared with each stage's golden values element by element. fp32 passes on tolerance against float64, while bf16 and int8 match their bit-exact models on every element. Assertions on every handshake run in every simulation, and a firing assertion fails the run.

In bf16, the regression also reports GPNAE's accuracy against the exact functions without gating it: sigmoid and tanh reach 7.2% and 8.6% against a 6.25% bound. The GPNAE README has the table.

---

## Performance

All numbers are Verilator simulation cycle counts on the RTL at `TILE = 4`, `LANES = 32`.

### Streaming sets (`make perf-analysis`, 24 sets per configuration, streaming host)

A set is one N×N×N matrix multiply followed by its activation, pooling and dropout. The figures below are the steady-state cycles between completed sets with sets streaming back to back (lower is better).

| Configuration | N=16 fp32 | N=16 bf16 | N=16 int8 | N=32 fp32 | N=32 bf16 | N=32 int8 | Bound by |
|---|---|---|---|---|---|---|---|
| bias + ReLU, no pooling | 19.0 | 19.0 | 19.0 | 36.0 | 36.0 | 36.5 | host load |
| ReLU, then max pooling | 21.0 | 21.0 | 21.0 | 63.0 | 63.0 | 63.0 | pooling |
| SELU | 214.8 | 157.0 | 97.0 | 562.2 | 488.0 | 328.0 | activation |
| sigmoid | 273.3 | 198.3 | 105.0 | 644.9 | 465.6 | 360.0 | activation |
| tanh | 263.0 | 189.7 | 108.0 | 623.2 | 530.0 | 366.0 | activation |
| tanh, inputs far outside the fitted range | 594.4 | 409.2 | 108.0 | 2 295.8 | 1 577.1 | 366.0 | activation |
| SELU over 3 accumulated depth passes, per pass | 91.0 | 66.0 | 33.0 | 269.1 | 204.9 | 110.0 | activation |
| packed set, mixed activations | 228.0 | 166.3 | 108.0 | 574.8 | 530.0 | 366.0 | activation |
| first set, bias + ReLU: latency | 100 | 95 | 79 | 164 | 159 | 143 | |

With a bypass activation the pipeline keeps up with the host:
- at N = 16, a bias + ReLU set completes every 19 cycles: 431 FLOP per cycle, 84% of the mesh's 256 MAC slots;
- at N = 32 it completes every 36 cycles: 1 820 FLOP per cycle, 89% of 1 024.

Polynomial activations are bound by the 32 lanes. They are fastest in int8, where the lanes run in fixed point and have no tail unit. Every configuration of the pipeline regression (32 in fp32 and bf16, 41 in int8) streams 24 sets in each run with every output checked.

### MLPerf Tiny models (`make model`, N = 16)

Every multiply-accumulate runs on the RTL; the host only reshapes and applies softmax.

| Model | MACs | fp32, host-driven sets: cycles | MACs / cycle | bf16, layer engine: cycles | Output vs the float model, fp32 / bf16 |
|---|---|---|---|---|---|
| resnet8 (CIFAR-10 image) | 12 505 728 | 51 396 | 243 | 51 959 | max \|Δ\| 1.0e−6 / 1.1e−2, same class |
| kws (speech clip) | 2 630 143 | 29 836 | 88 | 30 167 | max \|Δ\| 6.0e−8 / 2.5e−3, same class |
| vww (96×96 image) | 7 491 968 | 97 588 | 77 | 98 539 | max \|Δ\| 6.0e−8 / 7.0e−3, same class |
| ad01 (one spectrogram slice) | 264 192 | 17 640 | 15 | 17 715 | anomaly score 11.094 (float 11.094) / 30.84 |
| ad01 (40 slices as one batch) | 10 567 680 | 50 920 | 208 | 51 490 | anomaly score 9.709 (float 9.709) / 26.98 |

The mesh at N = 16 has 256 MAC slots per cycle, so resnet8 keeps 95% of them busy.
- **fp32:** every model matches the float model.
- **bf16:** the mesh sums in bf16, so error grows with the depth of a layer. ad01's first layer sums 640 products, which is why its anomaly score drifts.

The MLPerf Tiny models are not in the repo; point `MODEL_DIR` at a copy.

### GEMM shapes (`make gemm FMT=int8 QUICK=1`, N = 16)

| Shape (M × K × N) | Sets | Cycles | MACs / cycle |
|---|---|---|---|
| 1 × 16 × 16 | 1 | 98 | 2.6 |
| 1 × 64 × 256 | 64 | 1 121 | 14.6 |
| 64 × 48 × 40, bias + SELU | 36 | 1 327 | 92.6 |
| 64 × 48 × 40, bias + sigmoid | 36 | 1 423 | 86.4 |
| 64 × 48 × 40, bias + tanh | 36 | 1 459 | 84.2 |

All nine shapes of the sweep match the bit-exact model.

### Packing (`make pack`, T = 4)

Each row runs J small jobs (blocks of b columns) as one packed layer and again one job at a time. The results are identical, and these are the cycle counts:

| Format, N | b | Jobs | Packed | Alone | Speedup |
|---|---|---|---|---|---|
| fp32, 32 | 2  | 16 | 1 295 | 11 824 | 9.1× |
| bf16, 32 | 2  | 16 | 1 250 | 11 504 | 9.2× |
| int8, 32 | 2  | 16 | 903   | 7 854  | 8.7× |
| int8, 16 | 2  | 8  | 315   | 1 589  | 5.0× |
| int8, 16 | 8  | 2  | 315   | 445    | 1.4× |
| bf16, 32, linear, 1 024 rows | 2 | 16 | 1 342 | 21 472 | 16.0× |

The int8 TFLite layers packed at N = 32, T = 4 run 3 fully connected layers in one 178-cycle set, and a 3×3 conv plus a fully connected layer in 36 sets (1 487 cycles). All of them are bit-exact against the interpreter.
