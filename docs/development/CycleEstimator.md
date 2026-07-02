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

```
T_program = max_node( max_{k ∈ node} T_kernel(k) )
```

Under ideal-peak with full pipelining, connected producer/consumer kernels overlap in steady state, so there is no serial sum along a dependency chain. The roofline **is** the estimate, not a lower bound. The model is deterministic from (profile, trace) and needs no measured-cycle labels.

**Out of scope — the latency regime.**
Fill/drain latency for small workloads and explicit cross-node serialization are not modelled. They would require the dependency DAG (`kernel_block.on`, dfb push/pop, pipe send/recv); the current model is throughput-only.

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
| `bytes_per_tile` | movement tile size (provisional; bf16 = 2048 B) |
| `clock_ghz` | cycle↔ns reporting only; not used in the model |
| `dm_engines` | reserved for future overlap modelling |

Compute-rate lookup is tiered: exact `(op_type, dtype)`, then op-type-only `(op_type, "")`, then `compute_rate_default`. The op-type-only tier lets rates be keyed by op alone when the trace carries no dtype.

Built-in profiles live in `hardware_profile.py`, looked up by name; custom profiles load from JSON. `--hw-profile <name | path.json>` selects one.

Provenance:
the `wormhole_b0` **movement** rates are seeded from tt-metal NoC data (cited inline); **compute** rates are provisional pending arch/ISA references.

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

Not instrumented:
`block.broadcast` and `block.transpose` (layout ops — instrumented only if the model should charge for them). `dtype` is not currently emitted, so compute-rate lookup falls back to the op-type-only tier.

---

## Output

The pipeline produces one canonical `CycleEstimate`; every view is a pure function of it (compute once, render many).

- **Summary** (default) — per-node roll-up: active nodes, per-node cycles, utilization, and a bound-class table (compute vs memory). `--include-zero-kernels`
  also lists idle nodes.
- **Detailed** (`--detailed`) — the full per-kernel table.
- **JSON** (`--json-out`) — self-describing (`tool`, `schema_version`, profile, and er-kernel work + cycles).
- **Re-render** (`--view-report REPORT.json`) — reload a saved JSON report and render it without re-running.

Example summary tail:

```
Type         Nodes    Avg Cycles           Max   Max node
..............................................................................
compute          0          0.00          0.00   -
memory          32       4934.36       4934.36   node0
------------------------------------------------------------------------------
Program cycles : 4934.36
Active nodes   : 32 / 64  (32 idle)
Bottleneck     : 32 nodes @ 4934.36 (memory-bound)
```

`Nodes` counts nodes *bound* by that resource; a node is memory-bound when its movement path exceeds its compute path.

---

## Command-Line Interface

**Offline** — analyze a saved trace:

```
tt-lang-sim-cycles trace.jsonl
    [--hw-profile NAME|FILE.json]   # built-in profile name or custom JSON
    [--detailed]                    # full per-kernel table
    [--json-out OUT.json]           # write a self-describing report
    [--view-report REPORT.json]     # reload + render a saved report
    [--include-zero-kernels]        # summary: also list idle nodes
```

**Inline** — run and estimate in one step:

```
tt-lang-sim prog.py --cycles [--trace trace.jsonl]
```

Runs the program, then prints the summary from the same in-memory trace (no file round-trip). Combine with `--trace` to also save the trace. `--cycles` uses the default hardware profile; drop to `tt-lang-sim-cycles` for profile and report options.

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
      ├─ types.py             HardwareProfile, OpWork, KernelWork, KernelEstimate, CycleEstimate
      ├─ hardware_profile.py  built-in profile registry + JSON loader
      ├─ schedule.py          op / kernel / program cycle combiners
      ├─ model.py             build_estimate: work + profile → CycleEstimate
      ├─ report.py            summary / detailed / JSON / reload renderers
      └─ cli.py               argument wiring
```

The only cross-package coupling is the trace itself: the sim (producer) defines the event schema and emits events; `cycles` (consumer) reads the file. `--cycles` adds one lazy, one-directional import (`sim` → `sim_stats`) purely for ergonomics; `sim_stats` is top-level in both the source and installed layouts, so that import is stable.

Today the estimator has its own runnable entry (`python -m sim_stats.cycles`, backed by `tt-lang-sim-cycles`), separate from the stats tool (`python -m sim_stats`). Unifying the two is an open question — see [Limitations & Deferred Work](#limitations--deferred-work).

---

## Validation

Under ideal-peak there are no hardware labels, so the estimator is validated for correctness, behavior, and sensitivity — not accuracy.

- **Correctness** (regression fixtures): invariants — `2× tiles → 2× compute cycles`; `max(compute, movement) ≤ estimate ≤ compute + movement` (never additive); zero work → zero cycles; determinism. Plus hand-derived cross-checks on simple kernels.
- **Behavior**: per-kernel decomposition (compute vs movement, dominant term, bound class) across a work-count matrix (compute-bound / memory-bound / mixed / multi-node), small → large.
- **Sensitivity**: sweep the profile and confirm estimates and bound class shift sensibly.

Accuracy against profiled device cycles (`tt-metal` `ReadDeviceProfilerResults`, `PROFILER build`) is deferred until profiling data exists; the residual against ideal-peak is the utilization factor for later non-ideal modelling.

---

## Limitations & Deferred Work

- **Compute rates are provisional** — movement rates are seeded from tt-metal NoC data; compute rates await arch/ISA references.
- **`dtype` is not emitted**, so compute rates are keyed by `op_type` alone.
- **`broadcast` / `transpose` are not charged** as compute.
- **Latency regime** (fill/drain, cross-node serialization) is outside the current throughput-bound model; it needs the dependency DAG.
- **Behavioral-coverage and sensitivity sweeps**, and per-family / per-size reporting, are not yet built out.
- **Unified `sim_stats` entry (open — needs discussion)** — a `python -m sim_stats stats|cycles` subcommand dispatcher instead of two separate entries; restructures the stats tool, so its own change.
