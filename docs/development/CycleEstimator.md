# Cycle Estimator — v1.0 Restructure (Development Plan)

**Status:** v0.1 is being replaced, not extended. v0.1 predicts a *logical-time* signal, not
hardware cycles; no weight tuning fixes that. v1.0 re-bases the estimator on an analytical
peak-performance model. This document is the development reference and tracking checklist for
the v0.1 → v1.0 restructure. The self-check checklist is at the end.

For the current (v0.1) source layout see `python/sim_stats/cycle_tools/`.

---

## 1. Background — what v0.1 does

`tt-lang-sim-cycles` post-processes a `tt-lang-sim --trace` JSONL file and predicts each kernel's cycle count as an **additive sum of trace-derived terms**:

```
estimate = dfb_wait_dur + dfb_reserve_dur + copy_du(phase durations, from trace ticks)
         + roofline_base (max of compute / memory ceiling)
         + stall + sync + copy_overhead + blocked + launch (per-event overheads)
```

Pipeline: `parse_trace → extract_kernel_features → estimate_kernel_cycles → group_kernel_estimates → report`.
Constants (`flops_per_tile`, `peak_flops_per_cycle`, per-event costs, scale factors) live in `EstimatorConfig` and are hand-set placeholders.

---

## 2. Why restructure instead of patch

Three reasons — each a property of the design, not a tuning error:

1. **The prediction target is logical ticks, not cycles.**
  The simulator `tick` increments `+1` per scheduler activation that makes progress (`greenlet_scheduler.py`) — a fairness/ ordering counter, not time. `measured_cycles = kernel_end.tick − kernel_start.tick` counts scheduler turns.
  A "phase-only" estimate reconstructs it at ~0% WAPE, which only proves the model sums sub-intervals back into the whole — tautological, and meaningless as a hardware predictor.

2. **The combiner double-counts.**
  Roofline is added *on top of* phase durations that already cover the same work; when work overlaps, the sum over-predicts (up to ~30× on small traces). The correct operator is `max`, not `+` — a structural change.

3. **The constants have no hardware basis.**
  `flops_per_tile=2048`, `peak_flops_per_cycle=4096`, etc. are placeholders; the supporting machinery (`role_calibration_suggestions`, `ablation_metrics`) exists only to fit/diagnose the logical-tick target.

Target, combiner, and constants are all wrong. That is a restructure.

---

## 3. v1.0 Design — analytical peak model

**Goal:**
  Estimate real hardware cycles from *(a) hardware spec profile* and *(b) simulator trace*, assuming the hardware runs at ideal peak performance (no utilization derating).

### Inputs

- **Hardware spec profile** (`HardwareProfile`):
  Compute throughput (tiles/cycle by op-type +
  dtype), NoC bandwidth (bytes/cycle by locality) + per-transfer latency, DRAM bandwidth, engine count, clock.
- **Simulator trace**:
  Per-op work-counts (tiles, bytes, locality) and the dependency/overlap **structure** (which kernels block on which DFBs/pipes).

### Model

The trace supplies **structure**; it never supplies time. **Tick counts are never multiplied by a rate.** Cycles come from work ÷ rate:

$$\text{compute op:}\quad cyc = \frac{\text{tiles}}{R_{\text{compute}}(\text{op\_type},\,\text{dtype})}$$

$$\text{movement op:}\quad cyc = \text{latency} + \frac{\text{bytes}}{R_{\text{noc}}(\text{locality})}$$

Within a kernel, the compute engine and the data-movement engine run concurrently, so the kernel time is the larger of the two serial paths:

$$T_{\text{kernel}} = \max\!\big(\textstyle\sum cyc_{\text{compute}},\; \sum cyc_{\text{movement}}\big)$$

Across kernels, the program time is the critical path through the dependency DAG (edges from `kernel_block.on` + DFB push/pop + pipe send/recv), with each node weighted by `T_kernel`:

$$T_{\text{program}} = \text{critical-path}\big(\text{DAG},\; T_{\text{kernel}}\big)$$

Under ideal-peak, the roofline **is** the estimate (not a lower bound). The model is deterministic from (spec, trace) and needs **no measured-cycle labels** to build or run.

---

## 4. Architecture & Modules change plan

The one cross-package dependency is **new compute-op instrumentation in the simulator** — the trace does not currently record math ops at all (only `kernel_*`, `dfb_*`, `copy_*`, `operation_*`). Everything else is contained in `cycle_tools/`.

```
python/
├─ sim/                          ← simulator · PRODUCER
│  ├─ trace.py                   [CHANGE]  register new compute-op event + category (mechanism only)
│  ├─ math.py                    [ADD]     emit per-op trace event (op_type, dtype, tiles) at op sites
│  └─ greenlet_scheduler.py      [KEEP]
│
└─ sim_stats/                    ← post-processing · CONSUMER · project main dir
   ├─ __main__.py  (sim-stats)   [KEEP]
   ├─ utils.py                   [KEEP]
   ├─ cycle_estimator.py         [CHANGE]  compat shim + console entry; update re-exports
   └─ cycle_tools/               ← the cycle estimator
      ├─ parse.py                [CHANGE]  consume op events; demote tick-durations to diagnostics
      ├─ types.py                [CHANGE]  EstimatorConfig → HardwareProfile; add per-op records
      ├─ model.py                [REPLACE] additive sum → work÷rate + max / critical-path
      ├─ hardware.py             [ADD]     peak-rate spec table (Wormhole / Blackhole)
      ├─ schedule.py             [ADD]     overlap + critical-path combiner
      ├─ report.py               [TRIM]    drop ablation_metrics + role_calibration; keep per-family/size
      └─ cli.py                  [CHANGE]  drop tuning flags; add --hw-profile
```

| module | action | detail |
|---|---|---|
| `sim/math.py` + `trace.py` | add / change | `math.py`: emit a per-op event (`op_type`, `dtype`, tiles) at each op site. `trace.py`: register the event + category (mechanism only). |
| `parse.py` | change | build per-op work records and the dependency graph; keep `measured_cycles` and tick-durations **only as diagnostics**. |
| `types.py` | change | replace `EstimatorConfig` placeholders with `HardwareProfile`; add per-op record types. |
| `model.py` | replace | `work ÷ rate` per op; `max(compute, movement)` per kernel; remove additive sum, roofline-on-top, stall/sync/blocked terms and `mismatch_reason` escalation. |
| `schedule.py` | add | dependency-DAG critical-path combiner. |
| `hardware.py` | add | spec table for the target part. |
| `report.py` | trim | remove `ablation_metrics` + `role_calibration_suggestions`; per-kernel decomposition + per-family/size reporting. |
| `cli.py` | change | drop model-tuning flags; add `--hw-profile`. |
| `cycle_estimator.py` | change | update re-export list (drop `ablation_metrics`, `role_calibration_suggestions`, `mismatch_reason`); keep the `tt-lang-sim-cycles` entry. |

### Trace Instrumentation Detail

- `trace()` is the generic mechanism; instrumentation calls live at the behavior site, exactly as `copy.py`/`dfb.py` do today. The compute-op call therefore belongs in `math.py`, not `trace.py`. `trace.py` only learns the new event name + category.
- The change is **additive and non-breaking**:
  Not touch the scheduler tick, so existing events, tick progression, and kernel spans are unchanged; `tt-lang-sim-stats` ignores unknown events; old traces still parse (treated as "no compute term"). Put the new event under its own trace category so it is filterable and trace size stays controllable.

---

## 5. Removed in v1.0

- `EstimatorConfig` tuning knobs: `*_block_scale`, per-event cycle costs, `flops_per_tile`, `peak_flops_per_cycle`, `blocked_cycle_weight`.
- `report.ablation_metrics`, `report role_calibration_suggestions`.
- `model.mismatch_reason` escalation gate.
- The model-tuning CLI flags (already hidden in the current CLI).

---

## 6. Validation approach

Under ideal-peak there are **no hardware labels**, so v1.0 cannot be scored by accuracy. It is validated for correctness, behavior, and sensitivity; accuracy is deferred until profiling data exists.

- **Correctness (regression fixtures):** invariants — `2× tiles → 2× compute cycles`, `estimate ≥ roofline lower bound`, `max(compute, movement) ≤ estimate ≤ compute + movement`, zero work → zero cycles, determinism. Hand-derived cross-checks on simple kernels. Synthetic identifiability (recover known constants from generated labels).
- **Behavior:** per-kernel decomposition (compute vs movement, dominant term, bound class); coverage across a work-count matrix (compute-bound / memory-bound / mixed / multi-core / pipe),
  small → large.
- **Sensitivity:** sweep the hardware spec and confirm estimates/bound-class shift sensibly.
- **Deferred:** validate against profiled device cycles (tt-metal `ReadDeviceProfilerResults`, PROFILER build); the residual vs ideal-peak is the utilization factor for later non-ideal work.

---

## Open decisions

1. **Target part + spec source** — Wormhole or Blackhole? Peak rates from datasheet or known tt-metal constants? (blocks `hardware.py`)
2. **Trace instrumentation** — can the simulator emit `op_type` + `dtype` per compute op? (blocks the compute term; gates Phase 0 of validation)
3. **Overlap model** — confirm `max(compute, movement)` per kernel + critical-path across kernels.
4. **Scope line** — ideal-peak is the v1.0 deliverable; measured-cycle validation is out of scope until later.

---

## Implementation Checklist

### Trace instrumentation (`python/sim/`)
- [ ] Register a compute-op event name + category in `trace.py` (`_EVENT_CATEGORY`, `ALL_CATEGORIES`)
- [ ] Emit the event at op sites in `math.py` with `op_type`, `dtype`, `tiles` (and tile shape)
- [ ] Confirm instrumentation does not alter tick progression or existing events
- [ ] Confirm `tt-lang-sim-stats` output is unchanged (unknown event ignored)
- [ ] New event is under its own trace category (filterable; size control)

### Data model (`cycle_tools/types.py`)
- [ ] Add `HardwareProfile` (compute rates by op/dtype, NoC bw by locality, latency, clock, engines)
- [ ] Add per-op work-record type (kind, op_type, dtype, tiles, bytes, locality)
- [ ] Remove `EstimatorConfig` tuning knobs (scales, per-event costs, flops/bytes, blocked weight)
- [ ] Keep `measured_cycles` / tick fields only as diagnostics

### Parsing (`cycle_tools/parse.py`)
- [ ] Consume compute-op events → per-op work records
- [ ] Reconstruct dependency/overlap structure (`kernel_block.on`, dfb push/pop, pipe send/recv)
- [ ] Demote tick-duration extraction to diagnostics
- [ ] Handle old traces without op events gracefully (no compute term + clear note)

### Model & combiner (`cycle_tools/model.py`, `cycle_tools/schedule.py`)
- [ ] Per-op cost: compute = `tiles / rate`; movement = `latency + bytes / bw`
- [ ] Kernel cost: `max(compute_path, movement_path)`
- [ ] Program cost: critical path across kernels over the dependency DAG (`schedule.py`)
- [ ] Remove additive sum, roofline-on-top, stall/sync/blocked terms
- [ ] Remove `mismatch_reason` escalation logic

### Hardware profile (`cycle_tools/hardware.py`)
- [ ] Spec table for the chosen target part (pending decision #1)
- [ ] Each rate documents its source (datasheet vs known constant)

### Reporting (`cycle_tools/report.py`)
- [ ] Remove `ablation_metrics` and `role_calibration_suggestions`
- [ ] Per-kernel decomposition (compute vs movement, dominant term, bound class)
- [ ] Per-family + per-size reporting (never a single global number)
- [ ] Update `feature_provenance` to hardware-derived sources

### CLI & entry (`cycle_tools/cli.py`, `cycle_estimator.py`, `cycle_tools/__init__.py`)
- [ ] Drop model-tuning flags; add `--hw-profile`
- [ ] Update `cycle_estimator.py` re-exports (drop removed symbols)
- [ ] Update `cycle_tools/__init__.py` exports

### Validation
- [ ] Invariant tests (monotonicity, bounds, overlap, determinism, zero-work) as regression fixtures
- [ ] Hand-derived cross-checks on simple kernels
- [ ] Synthetic identifiability check
- [ ] Behavioral coverage across the work-count matrix
- [ ] Sensitivity sweep over the hardware spec
- [ ] (Deferred) hardware-profile validation

### Docs
- [ ] Replace remaining v0.1 user-guide content with the v1.0 model + CLI reference
- [ ] Document removed flags / breaking changes for users of `tt-lang-sim-cycles`
