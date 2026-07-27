# Cycle Estimator Design

## Overview

`tt-lang-sim-cycles` estimates hardware cycle counts for a tt-lang program from two inputs: a **hardware profile** (peak rates) and a **simulator trace** (`tt-lang-sim --trace`). It applies an **analytical ideal-peak model** — cycles are computed as work-counts ÷ hardware rates — and assumes the hardware runs at peak performance with no utilization derating.

The estimator is a trace **consumer**: it reads the JSONL trace file and never imports the simulator. The trace file is the only contract between the two, which is why the estimator can run wherever a trace can be copied, independent of the sim.

Quick start:

```
tt-lang-sim prog.py --cycles       # run + estimate in one step
tt-lang-sim-cycles trace.jsonl     # estimate from a saved trace
```

See [Command-Line Interface](#command-line-interface) for the full flag set.

---

## The Model

The trace supplies **work** (how many tiles each op computes or moves) and **structure** (which kernel runs on which node). It never supplies time — the simulator tick is a logical clock, not a duration (see [Design Rationale](#design-rationale--why-ideal-peak-not-fit-to-trace)).

Cycles come entirely from work ÷ rate.

**Per op.** Compute and movement each have a peak rate from the profile:

```
compute op:   cyc = tiles / R_compute(op_type, dtype)
movement op:  cyc = latency(locality) + (tiles × bytes_per_tile) / R_noc(locality)
```

**Per kernel.** The compute engine and the data-movement engine run concurrently, so the kernel time is the larger of the two serial paths, not their sum:

```
T_kernel = max( Σ cyc_compute , Σ cyc_movement )
```

**Per program.** The model is throughput-bound, with two levels of overlap:

- *Within a node* — the reader / compute / writer kernels run on that core's concurrent RISCs, so the node's time is the `max` of its kernels.
- *Across nodes* — distinct nodes are separate cores in parallel, so the program time is the `max` over nodes.
- *Aggregate DRAM ceiling* — every core draws DRAM from one shared GDDR6 controller, so the program is also bounded below by `total_dram_bytes / dram_aggregate_bw`. Only `dram`-locality movement counts (local/remote L1 never touch the controller). This is a static divide-by-peak (a real hardware ceiling), not a queuing/fairness model; it is disabled when `dram_aggregate_bw = 0`.

```
dram_floor = (Σ dram_bytes) / dram_aggregate_bw
T_program  = max( max_node( max_{k ∈ node} T_kernel(k) ), dram_floor )
```

The per-node NoC term (a single core's transfer/latency) and the aggregate DRAM floor (the shared controller) model different resources, so the program takes the `max` of both. The report records which one bound the program (`program_bound`: `per-node` | `aggregate-dram`) and the `dram_floor` value. Without the ceiling, a K-sweep matmul stays compute-bound at every K because each of N active cores is (incorrectly) given a private DRAM lane; the aggregate ceiling flips memory-heavy points to `aggregate-dram`.

Under ideal-peak with full pipelining, connected producer/consumer kernels overlap in steady state, so there is no serial sum along a dependency chain. The roofline **is** the estimate, not a lower bound. The model is deterministic from (profile, trace) and needs no measured-cycle labels.

**Fill/drain — reported, not folded into the bound.**
Pure throughput ignores a pipeline's fill (first item through read→compute→write) and drain (last item). A deterministic estimate of that overhead treats a node's kernels as stages with cycles `C_i` and `N` pipeline items (the write kernel's movement-op count; `N ≥ 1`):

```
node_time  = max_i(C_i) + (Σ_i C_i - max_i C_i) / N     # fill/drain-inclusive per node
fill_drain = max_node(node_time) − node_bound           # reported as `node_fill_drain`
T_program  = max( node_bound, dram_floor )              # the roofline — the reported estimate
```

`fill_drain` is **reported for information only — it is NOT added to `T_program`.** It's a crude, unprovable heuristic (assumes a read/compute/write stage shape, one item-count per node) that can *exceed* real per-node overhead: folding it in pushed the estimate *above* measured cycles on the reuse-matmul kernel at some sizes (device-confirmed, P100a), breaking `measured ≥ estimate`. Keeping `T_program` at the pure throughput roofline preserves that behavior. (Large `N` recovers pure throughput; `N = 1` is a serial sum.)

**Out of scope — the rigorous latency regime.**
Exact fill/drain and cross-node serialization from the real dependency DAG (`kernel_block.on`, dfb push/pop, pipe send/recv) are deferred; the correction above is a coarse per-node add-on, not a DAG traversal.

---

## Design Rationale — why ideal-peak, not fit-to-trace

The simulator tick is a **logical clock**: it increments by one per productive scheduler activation, measuring scheduling order rather than time (see `docs/TRACING.md`, *Logical Time*). It carries no wall-clock meaning and its value depends on the scheduler policy.

Two consequences shape the model:

- A tick duration cannot be multiplied by a rate to yield cycles. Any model fit to reconstruct tick durations predicts scheduling behavior, not hardware — reproducing a logical clock from its own sub-intervals is tautological.
- Physical quantities in the trace are the **work-counts** (tiles, and bytes derived from tiles), not the timing. The estimator therefore multiplies work by hardware rates and ignores tick durations entirely.

This is what lets the estimate be label-free and deterministic: given a profile and a trace, the answer is fixed, with no calibration step.

---

## Inputs

### Hardware profile

`HardwareProfile` (`cycles/types.py`) carries the rates that traces cannot provide:

| Field | Meaning |
|---|---|
| `compute_rate` | tiles/cycle by `(op_type, dtype)` |
| `compute_rate_default` | fallback tiles/cycle |
| `noc_bw` | bytes/cycle by locality (`local_l1` / `remote_l1` / `dram`) |
| `noc_latency` | fixed cycles per transfer, by locality |
| `dram_aggregate_bw` | shared GDDR6 ceiling, B/cyc (JSON stores `dram_aggregate_gbps`, ÷`clock_ghz` at load; `0` disables the ceiling) |
| `bytes_per_tile` | movement tile size; the dtype knob (bf16 2048 / fp32 4096 / bfp8 ~1088 B) |
| `clock_ghz` | GHz; ns reporting **and** the DRAM GB/s→B/cyc conversion at load |
| `dm_engines` | reserved for future overlap modelling |

Compute-rate lookup is tiered: exact `(op_type, dtype)`, then op-type-only `(op_type, "")`, then `compute_rate_default`. The op-type-only tier lets rates be keyed by op alone when the trace carries no dtype.

Built-in profiles are JSON files under `hw_profiles/` (one per board); `model.py` loads and resolves them (`types.py` holds only the dataclass schema). `--hw-profile <name | path.json>` selects one — a bundled name (full stem or board family, e.g. `wormhole` → `wormhole_n300`), or a path to a custom profile anywhere.

#### `wormhole_n300` provenance

All values are sourced from tt-metal and the Wormhole ISA docs:

| Field | Value | Source |
|---|---|---|
| `clock_ghz` | 1.0 | WH AICLK (tt-metal perf docs) |
| `bytes_per_tile` | 2048 | bf16 32×32 tile (32·32·2 B) |
| `dm_engines` | 2 | BRISC + NCRISC (METALIUM_GUIDE) |
| `noc_bw` / `noc_latency` | 25.3 B/cyc, 293 cyc | **measured**, tt-metal `noc_latencies.yaml` (64 KB / 2589 cyc asymptote; 293-cyc small-transfer floor) |
| `dram_aggregate_gbps` | 288 GB/s | 12 GDDR6 ch × 24 B/cyc @ 12 Gbps (tt-metal `Saturating_DRAM_bandwidth.md`). Stored as GB/s in the JSON; converted to B/cyc at load (÷ `clock_ghz`). It is the **spec upper bound** — a lower-bound model needs an *upper bound* on achievable BW, so the datasheet peak is used, not a measured figure (sustained is ~265, 92% of spec). |
| matmul rate (default) | 1/64 | `16 × fidelity` cyc/tile; tt-lang sets no MathFidelity → tt-metal default **HiFi4**, fixed. |
| matmul rate (fp32) | ≈1/68.5 | f32 args set `fp32_dest_acc_en` (`TTLSetComputeKernelConfig`) → ~7% slower. BH-calibrated. |
| SFPU default | 1/32 | 32 elem/clk ideal 1-instruction floor (SFPU spec) |

Known simplifications (see [Limitations](#limitations--deferred-work)): MathFidelity is fixed at HiFi4 (never set), so no fidelity swing; the one modelled dtype effect is fp32's `fp32_dest_acc_en` (~7%). `noc_bw` uses one measured asymptote for all localities (local L1 ≈ 2× remote, DRAM ≈ 24 B/cyc per channel); SFPU per-op cost (instruction count) is deferred.

The bundled `blackhole_p100a` profile mirrors this structure with Blackhole P100a values: 1.35 GHz, 448 GB/s aggregate DRAM (7/8 GDDR6), 60.9 B/cyc NoC.

### Simulator trace — the consumed contract

The estimator reads two event kinds and ignores all others:

| Event | Category | Fields read | Produces |
|---|---|---|---|
| `compute_op` | `compute` | `op_type`, `dtype`, `tiles` | one compute `OpWork` |
| `copy_end` | `copy` | `local_l1`, `remote_l1`, `dram` (tile counts) | one movement `OpWork` per non-zero locality |

`compute_op` is emitted once per math op. `copy_end` carries per-locality tile counts for Tensor↔Block copies; pipe- or block-only copies carry no locality fields and contribute no movement work.

The consumed set is declared as `parse.CONSUMED_EVENTS` and pinned against the producer's registry (`sim/trace.py`) by `test/sim/test_trace_contract.py`, so a producer-side rename fails a test rather than silently zeroing the estimate.

A trace without `compute_op` events — produced before the instrumentation, or with the `compute` category filtered out — parses as movement-only.

#### `compute_op` emission sites

`op_type` and `tiles` are known only at the op site, so each op-family emits at its own chokepoint:

| Site | Ops | `op_type` |
|---|---|---|
| `dfb.Block._binary_op` | `+ - * / //` | operator name (`add`/`sub`/`mul`/`truediv`/`floordiv`) |
| `dfb.matmul` | matmul | `matmul` (tiles = M·K·N) |
| `math._create_unary_op_wrapper` | `exp`, `rsqrt`, `sqrt`, `relu`, `sign`, … | op name |
| `math._apply_unary_with_params` | `relu_max`, `clamp`, `elu`, `leaky_relu`, … | `eltwise_unary` (generic) |
| `math._apply_binary_op` | `max`, `min`, `gt`, `lt`, `eq`, `ne` | `eltwise_binary` (generic) |
| `math._reduce_impl` | reduce sum/max | `reduce_sum` / `reduce_max` |

Each site also emits the declared `dtype` (via `dtype_name`), so compute-rate lookup uses the `(op_type, dtype)` tier and falls back to op-type-only when a profile has no dtype-specific row.

Not instrumented:
`block.broadcast` and `block.transpose` (layout ops — instrumented only if the model should charge for them).

---

## Output

The pipeline produces one canonical `CycleEstimate`; every view is a pure function of it (compute once, render many).

- **Summary** (default) — per-node roll-up: active nodes, per-node cycles, utilization, and a bound-class table (compute vs memory). `--include-zero-kernels` also lists idle nodes.
- **Detailed** (`--detailed`) — the full per-kernel table.
- **JSON** (`--json-out`) — self-describing (`tool`, `schema_version`, profile, and per-kernel work + cycles).
- **Re-render** (`--view-report REPORT.json`) — reload a saved JSON report and render it without re-running.

Example summary tail:

```
Type         Nodes    Avg Cycles           Max   Max node
..............................................................................
compute         48        54.61K        61.44K   node0
memory           0          0.00          0.00   -
------------------------------------------------------------------------------
DRAM (shared)
..............................................................................
  read           :  46.0 MB
  write          :  2.0 MB
  bandwidth      :  288 B/cyc   (288 GB/s @ 1.0 GHz)
  floor          :  174.76K
------------------------------------------------------------------------------
Program cycles :  174.76K   (aggregate-dram)
Per-node max   :  61.44K    (compute)
Active nodes   :  48 / 56   (8 idle)
```

`Nodes` counts nodes *bound* by that resource; a node is memory-bound when its movement path exceeds its compute path. `Per-node max` is the slowest node's cycles and its bound reason; `Program cycles` shows the final program time and which resource set it (`per-node` or `aggregate-dram`). The `DRAM (shared)` block renders only when the profile models an aggregate DRAM ceiling (`dram_aggregate_bw > 0`); profiles without one omit it entirely.

---

## Command-Line Interface

**Offline** — analyze a saved trace:

```
tt-lang-sim-cycles trace.jsonl
    [-d | --detailed]               # full per-kernel table
    [-p | --hw-profile NAME|FILE.json]     # built-in profile name or custom JSON
    [-o | --json-out OUT.json]      # write a self-describing report
    [-r | --view-report REPORT.json]       # reload + render a saved report
    [--include-zero-kernels]        # summary: also list idle nodes
```

**Inline** — run and estimate in one step:

```
tt-lang-sim prog.py --cycles [REPORT.json] [--hw-profile NAME|FILE.json] [--trace trace.jsonl]
```

Runs the program, then prints the estimate **summary** from the same in-memory trace (no file round-trip). Give a path after `--cycles` to also write the JSON report there (like `--trace`), and `--hw-profile` to target a specific part. For the per-kernel **detailed** view or to re-render a saved report, use `tt-lang-sim-cycles`.

---

## Module Layout

```
python/
├─ sim/                       simulator · PRODUCER
│  ├─ trace.py                defines the compute event + category (registry)
│  ├─ math.py, dfb.py         emit compute_op at op sites
│  ├─ copy.py                 emits copy_end with per-locality tile counts
│  └─ ttlang_sim.py           --trace / --cycles
│
└─ sim_stats/                 trace analysis · CONSUMER
   ├─ __main__.py             tt-lang-sim-stats (tensor/pipe/dfb tables)
   ├─ utils.py                shared trace + kernel-name helpers
   └─ cycles/                 the cycle estimator
      ├─ __main__.py          entry point: python -m sim_stats.cycles
      ├─ parse.py             trace → per-kernel work records; CONSUMED_EVENTS
      ├─ types.py             dataclasses (schema only)
      ├─ model.py             cycle math, per-node rollup, profile load/resolve, build_estimate
      ├─ report.py            summary / detailed / JSON / reload renderers
      ├─ cli.py               argument wiring
      └─ hw_profiles/         built-in profile data, one JSON per board (wormhole_n300, blackhole_p100a)
```

The only cross-package coupling is the trace itself: the sim (producer) defines the event schema and emits events; `cycles` (consumer) reads the file. `--cycles` adds one lazy, one-directional import (`sim` → `sim_stats`) purely for ergonomics; `sim_stats` is top-level in both the source and installed layouts, so that import is stable.

Today the estimator has its own runnable entry (`python -m sim_stats.cycles`, backed by `tt-lang-sim-cycles`), separate from the stats tool (`python -m sim_stats`). Unifying the two is an open question — see [Limitations & Deferred Work](#limitations--deferred-work).

---

## Validation

Under ideal-peak there are no per-kernel hardware labels, so the estimator is validated for correctness, behavior, and sensitivity — not fit to measured cycles.

- **Correctness** (regression fixtures): invariants — `2× tiles → 2× compute cycles`; `max(compute, movement) ≤ estimate ≤ compute + movement` (never additive); zero work → zero cycles; determinism. Plus hand-derived cross-checks on simple kernels.
- **Behavior**: per-kernel decomposition (compute vs movement, dominant term, bound class) across a work-count matrix (compute-bound / memory-bound / mixed / multi-node), small → large.
- **Sensitivity**: sweep the profile and confirm estimates and bound class shift sensibly.

Program-level accuracy is checked against device cycles on a matmul K-sweep (Wormhole N300, Blackhole P100a): `measured ≥ estimate` holds at every point. 

DRAM utilization is ~57–67% on Wormhole (residual consistent with an unmodelled NoC limit) and ~82–92% on Blackhole. dtype movement scaling and the fp32 `fp32_dest_acc_en` penalty (~7%) are device-confirmed. The residual is the utilization factor for later non-ideal modelling.

---

## Limitations & Deferred Work

- **Compute rates are partial** — the SFPU default is the ideal 1-instruction floor (32 elem/clk); real SFPU ops cost more, scaling with instruction count (kernel-dependent → profiling), and the SFPU unpack/pack-BW limit is not modelled. The matmul (FPU) rate is HiFi4 (64 cyc/tile) with an fp32 variant; per-fidelity variation is deferred (tt-lang fixes HiFi4).
- **Multicast movement is uncounted** — pipe copies (`Block→Pipe`, `Pipe→Block`) carry no locality fields, so a multicast fan-out contributes zero movement work. Kernels dominated by multicast (reuse/mcast matmul) are under-modelled on the movement path; pipe-copy byte-accounting is the next model factor.
- **dtype** — compute rate keys on `(op_type, dtype)`; the one modelled effect is fp32 matmul's `fp32_dest_acc_en` (~7%). bf16/bfp8 use the HiFi4 baseline (fidelity fixed). Movement byte-size follows `bytes_per_tile` (bf16 2048 / fp32 4096 / bfp8 ~1088).
- **`broadcast` / `transpose` are not charged** as compute.
- **Latency regime** — not in the bound. Fill/drain is reported as an informational delta only (see the model section); the rigorous version (exact fill/drain, cross-node serialization from the dependency DAG) is still deferred.
- **DRAM ceiling is the spec, not the achievable** — `dram_aggregate_gbps = 288` (spec) keeps the bound valid but loose (~265 is achievable). Tightening to ~265 risks breaking the `measured ≥ estimate` invariant; a per-workload utilization factor is the cleaner direction.
- **Behavioral-coverage and sensitivity sweeps**, and per-family / per-size reporting, are not yet built out.
- **Unified `sim_stats` entry (open — needs discussion)** — a `python -m sim_stats stats|cycles` subcommand dispatcher instead of two separate entries; restructures the stats tool, so its own change.
