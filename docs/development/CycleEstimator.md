# Cycle Estimator — v1.0 Restructure (Development Plan)

**Status:** the v0.1 additive / logical-tick model has been **removed**; the estimator is now the analytical ideal-peak model (work-counts ÷ hardware-profile rates). This document is the design reference + checklist; §1–2 record *why* v0.1 was replaced. The self-check checklist is at the end.

For the current (v0.1) source layout see `python/sim_stats/cycle_tools/`.

---

## 1. Background — what v0.1 does

`tt-lang-sim-cycles` post-processes a `tt-lang-sim --trace` JSONL file and predicts each kernel's cycle count as an **additive sum of trace-derived terms**:

```
estimate = dfb_wait_dur + dfb_reserve_dur + copy_dur         (phase durations, from trace ticks)
         + roofline_base                                     (max of compute / memory ceiling)
         + stall + sync + copy_overhead + blocked + launch   (per-event overheads)
```

Pipeline: `parse_trace → extract_kernel_features → estimate_kernel_cycles → group_kernel_estimates → report`.
Constants (`flops_per_tile`, `peak_flops_per_cycle`, per-event costs, scale factors) live in `EstimatorConfig` and are hand-set placeholders.

---

## 2. Why restructure instead of patch

Three reasons — each a property of the design, not a tuning error:

1. **The prediction target is logical ticks, not cycles.**
  The simulator `tick` increments `+1` per scheduler activation that makes progress (`greenlet_scheduler.py`) — a fairness/ordering counter, not time. `measured_cycles = kernel_end.tick − kernel_start.tick` counts scheduler turns.
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

Across kernels, under ideal-peak the program is **throughput-bound by its slowest kernel**: distinct nodes are separate cores running in parallel, and within a node the reader/compute/writer kernels share that core's concurrent RISCs. So the program time is the max over nodes of each node's max kernel:

$$T_{\text{program}} = \max_{\text{node}} \; \max_{k \in \text{node}} T_{\text{kernel}}(k)$$

Fill/drain latency and explicit cross-node serialization (the *latency* regime, needing the dependency DAG from `kernel_block.on` / dfb push-pop / pipe send-recv) are deferred.

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
      ├─ model.py                [REPLACE] additive sum → work÷rate + max (kernel & program)
      ├─ hardware_profile.py     [ADD]     peak-rate spec table (Wormhole / Blackhole), named registry
      ├─ schedule.py             [ADD]     overlap / throughput-bound combiner
      ├─ report.py               [TRIM]    drop ablation_metrics + role_calibration; keep per-family/size
      └─ cli.py                  [CHANGE]  drop tuning flags; add --hw-profile
```

| module | action | detail |
|---|---|---|
| `sim/math.py` + `trace.py` | add / change | `math.py`: emit a per-op event (`op_type`, `dtype`, tiles) at each op site. `trace.py`: register the event + category (mechanism only). |
| `parse.py` | change | build per-op work records and the dependency graph; keep `measured_cycles` and tick-durations **only as diagnostics**. |
| `types.py` | change | replace `EstimatorConfig` placeholders with `HardwareProfile`; add per-op record types. |
| `model.py` | replace | `work ÷ rate` per op; `max(compute, movement)` per kernel; remove additive sum, roofline-on-top, stall/sync/blocked terms and `mismatch_reason` escalation. |
| `schedule.py` | add | overlap / throughput-bound combiner (`max` within node and across parallel nodes). |
| `hardware_profile.py` | add | typed `HardwareProfile` registry of built-in parts (source of truth), looked up by name; plus `load_profile_json` / `resolve_profile` so `--hw-profile` accepts a built-in name or a custom `.json` path. |
| `report.py` | trim | remove `ablation_metrics` + `role_calibration_suggestions`; per-kernel decomposition + per-family/size reporting. |
| `cli.py` | change | drop model-tuning flags; add `--hw-profile`. |
| `cycle_estimator.py` | change | update re-export list (drop `ablation_metrics`, `role_calibration_suggestions`, `mismatch_reason`); keep the `tt-lang-sim-cycles` entry. |

### Trace Instrumentation Detail

- `trace()` is the generic mechanism; instrumentation calls live at the behavior site, exactly as `copy.py`/`dfb.py` do today. The compute-op call therefore belongs in `math.py`, not `trace.py`. `trace.py` only learns the new event name + category.
- The change is **additive and non-breaking**:
  Not touch the scheduler tick, so existing events, tick progression, and kernel spans are unchanged; `tt-lang-sim-stats` ignores unknown events; old traces still parse (treated as "no compute term"). Put the new event under its own trace category so it is filterable and trace size stays controllable.

#### `compute_op` coverage (current instrumentation)

`op_type`/`tiles` are only known at the op site, so each op-family emits at its own
chokepoint. Covered so far:

| site | ops | `op_type` |
|---|---|---|
| `dfb.Block._binary_op` | `+ - * / //` | operator name (add/sub/mul/truediv/floordiv) |
| `dfb.matmul` | matmul | `matmul` (tiles = M·K·N) |
| `math._create_unary_op_wrapper` | auto-gen unary (exp, rsqrt, sqrt, relu, sign, …) | op name |
| `math._apply_unary_with_params` | relu_max, clamp, elu, leaky_relu, celu, prelu, softplus, hardtanh, round, threshold | `eltwise_unary` (generic — no name threaded) |
| `math._apply_binary_op` | `max`, `min`, `gt`, `lt`, `eq`, `ne` | `eltwise_binary` (generic — no name threaded) |
| `math._reduce_impl` | reduce_sum, reduce_max | `reduce_sum` / `reduce_max` |

**Gaps (not yet emitting `compute_op`):**

| site | ops | note |
|---|---|---|
| `block.broadcast` | broadcast | fan-out/layout — decide whether it counts as compute |
| `block.transpose` | transpose | layout op — decide whether it counts as compute |
| tile-level / other | — | anything not routed through the sites above |

`_apply_unary_with_params` / `_apply_binary_op` use generic `op_type`s; thread a name from their callers if per-op compute rates are needed. Broadcast/transpose are layout ops — instrument them only if the compute model should charge for them.

---

## 5. Removed in v1.0

- `EstimatorConfig` tuning knobs: `*_block_scale`, per-event cycle costs, `flops_per_tile`, `peak_flops_per_cycle`, `blocked_cycle_weight`.
- `report.ablation_metrics`, `report role_calibration_suggestions`.
- `model.mismatch_reason` escalation gate.
- The model-tuning CLI flags (already hidden in the current CLI).

### Removal readiness & order

**Prerequisites (met):** peak produces complete compute + movement estimates, is reachable (`--model peak`), and has summary / detailed / JSON / view-report. The sim `compute_op` instrumentation exists in this branch, so both movement and compute flow end-to-end.

**Do atomically (or imports break):**
- remove the v0.1 model (`estimate_kernel_cycles`, `EstimatorConfig`, `mismatch_reason`) and report helpers (`ablation_metrics`, `role_calibration_suggestions`);
- slim `extract_kernel_features` / `KernelFeatures` down to the `measured_cycles` / `blocked_cycles` diagnostics, folded onto `KernelWork`;
- update `cycle_estimator.py` + `cycle_tools/__init__.py` re-exports in the same change (they still name the removed symbols);
- flip the default `--model` to peak; update tests.

**Accept before making peak the default:**
- compute rates are provisional placeholders (not hardware-validated);
- coverage gaps mean some ops emit no `compute_op` — `max`/`min`/compare (via `math._apply_binary_op`), `block.broadcast`/`transpose` — so those show 0 compute. Closing the `_apply_binary_op` gap first is cheap and avoids a silent under-count;
- keep the sim instrumentation and this removal on the same merge, so the peak default never ships without its producer.

---

## 6. Validation approach

Under ideal-peak there are **no hardware labels**, so v1.0 cannot be scored by accuracy. It is validated for correctness, behavior, and sensitivity; accuracy is deferred until profiling data exists.

- **Correctness (regression fixtures):** invariants — `2× tiles → 2× compute cycles`, `estimate ≥ roofline lower bound`, `max(compute, movement) ≤ estimate ≤ compute + movement` (never additive), zero work → zero cycles, determinism. Plus hand-derived cross-checks on simple kernels. (Synthetic identifiability does not apply under ideal-peak — there is no fitting step.)
- **Behavior:** per-kernel decomposition (compute vs movement, dominant term, bound class); coverage across a work-count matrix (compute-bound / memory-bound / mixed / multi-core / pipe), small → large.
- **Sensitivity:** sweep the hardware spec and confirm estimates/bound-class shift sensibly.
- **Deferred:** validate against profiled device cycles (tt-metal `ReadDeviceProfilerResults`, PROFILER build); the residual vs ideal-peak is the utilization factor for later non-ideal work.

---

## Open decisions

1. **Target part + spec source** — Wormhole or Blackhole? Peak rates from datasheet or known tt-metal constants? (blocks `hardware_profile.py`)
2. **Trace instrumentation** — can the simulator emit `op_type` + `dtype` per compute op? (blocks the compute term; gates Phase 0 of validation)
3. **Overlap model** — decided: `max(compute, movement)` per kernel; throughput-bound `max` across parallel nodes. Latency / critical-path regime (fill-drain, cross-node serialization) deferred.
4. **Scope line** — ideal-peak is the v1.0 deliverable; measured-cycle validation is out of scope until later.

---

## Implementation Checklist

### Trace instrumentation (`python/sim/`) — prototyped in a local branch, not merged
- [x] Register `compute_op` event + `compute` category in `trace.py`
- [x] Emit at main op sites — binary (`Block._binary_op`), matmul, unary, reduce (`op_type` + `tiles`)
- [x] New event under its own `compute` trace category (filterable)
- [ ] `dtype` not yet emitted (falls back to `compute_rate_default`)
- [ ] Coverage gaps: `block.broadcast`/`transpose`; `_apply_unary_with_params` / `_apply_binary_op` use generic labels — see coverage table above
- [ ] Review + merge into the shared simulator module

### Data model (`cycle_tools/types.py`)
- [x] Add `HardwareProfile` (compute rates by op/dtype, NoC bw by locality, latency, clock, engines)
- [x] Add per-op work-record type `OpWork` + `KernelWork` container (kind, op_type, dtype, tiles, locality)
- [x] Remove v0.1 types (`EstimatorConfig`, old `KernelEstimate`, `KernelGroupEstimate`, `KernelFeatures`); rename `PeakKernel`/`PeakResult` → `KernelEstimate`/`CycleEstimate`
- [x] Drop the logical-tick diagnostics (`measured_cycles`/`blocked_cycles`) — unused by the peak model

### Parsing (`cycle_tools/parse.py`)
- [x] Movement work records from `copy_end` localities (`extract_kernel_work`, alongside v0.1)
- [x] Handle traces without compute-op events gracefully (movement-only, empty compute path)
- [x] Consume `compute_op` events → compute work records (consumer done + tested against the contract; real traces await sim emission)
- [ ] Reconstruct dependency/overlap structure (`kernel_block.on`, dfb push/pop, pipe send/recv) — deferred; only needed for the latency regime
- [x] Remove `extract_kernel_features` (v0.1); tick-duration extraction dropped

### Model & combiner (`cycle_tools/model.py`, `cycle_tools/schedule.py`)
- [x] Per-op cost: compute = `tiles / rate`; movement = `latency + bytes / bw` (`schedule.op_cycles`)
- [x] Kernel cost: `max(compute_path, movement_path)` (`schedule.kernel_cycles`)
- [x] Program cost: throughput-bound `max` — parallel cores across nodes, concurrent RISCs within a node (`schedule.program_cycles`). DAG latency regime deferred.
- [x] Remove additive sum, roofline-on-top, stall/sync/blocked terms (whole v0.1 `estimate_kernel_cycles`)
- [x] Remove `mismatch_reason` escalation; rename `build_peak_result` → `build_estimate`

### Hardware profile (`cycle_tools/hardware_profile.py`)
- [x] Add `HardwareProfile` named registry (`get_profile`, scaffold, provisional rates)
- [~] Fill the spec table — `wormhole_b0` **movement** rates seeded from tt-metal `noc_latencies.yaml` + soc descriptor; **compute** rates still pending arch/ISA docs
- [x] Each rate documents its source (inline `# source:` citations to the tt-metal files)
- [x] Custom-profile file loader — `load_profile_json` / `resolve_profile`; `--hw-profile <name|path.json>`

### Reporting (`cycle_tools/report.py`)
- [x] `CycleEstimate` canonical intermediate — render + JSON are pure functions of it
- [x] Per-node **summary** (default, active nodes + utilization; `--include-zero-kernels` lists idle) + complete per-kernel **detailed** view (`--detailed`, all rows)
- [x] JSON export (`--json-out`) — full, self-describing (`tool`/`schema_version`/profile)
- [x] Reload + re-render a saved report (`--view-report`), with report-vs-trace validation
- [x] Remove `ablation_metrics`, `role_calibration_suggestions`, `feature_provenance`, `print_report`, `write_json_report` (v0.1); renderers renamed `print_summary`/`print_detailed`/`write_json`/`load_estimate`
- [ ] Per-family + per-size reporting (never a single global number)

### CLI & entry (`cycle_tools/cli.py`, `cycle_estimator.py`, `cycle_tools/__init__.py`)
- [x] `--hw-profile`, `--detailed`, `--json-out`, `--view-report`, `--include-zero-kernels`
- [x] Drop v0.1 tuning flags and `--model`; peak is the only model (default, no flag)
- [x] Trim `cycle_estimator.py` + `cycle_tools/__init__.py` re-exports to the current API

### Validation
- [x] Invariant tests (monotonicity, bounds, overlap=max-not-sum, determinism, zero-work) as regression fixtures
- [x] Hand-derived cross-checks on simple kernels
- [~] Synthetic identifiability check — N/A under ideal-peak (no fitting/calibration step to identify)
- [ ] Behavioral coverage across the work-count matrix
- [ ] Sensitivity sweep over the hardware spec
- [ ] (Deferred) hardware-profile validation

### Docs
- [ ] Replace remaining v0.1 user-guide content with the v1.0 model + CLI reference
- [ ] Document removed flags / breaking changes for users of `tt-lang-sim-cycles`
