# Cycle Estimator User Guide (v0.1)

## Overview

`tt-lang-sim-cycles` reads a simulator trace produced by `tt-lang-sim --trace` and estimates cycle counts for each kernel. The goal is to explain where cycles are spent — without running on real hardware — so bottlenecks can be identified and compared across design iterations.

The estimator uses a **phase-duration-first** model: the primary cost signals are the actual durations of blocking phases recorded in the trace (DFB wait/reserve stalls, copy transfer spans), not counts or abstract roofline projections. Roofline compute and memory ceilings are computed alongside the estimate and reported as secondary context for bound classification.

This document is a user guide. For the internal implementation layout, see the source files under `python/sim_stats/cycle_tools/`.

---

## Running the Estimator

Generate a trace with `tt-lang-sim`, then pass the trace file to `tt-lang-sim-cycles`:

```bash
tt-lang-sim examples/single_node_matmul.py --trace trace.jsonl
tt-lang-sim-cycles trace.jsonl
```

The estimator prints a per-kernel table to the terminal and exits. All model parameters have reasonable defaults; no configuration is required for a first run.

To also export a machine-readable JSON report:

```bash
tt-lang-sim-cycles trace.jsonl --json-out cycle_report.json
```

To override model parameters explicitly:

```bash
tt-lang-sim-cycles trace.jsonl \
  --flops-per-tile 2048 \
  --bytes-per-tile 2048 \
  --peak-flops-per-cycle 4096 \
  --memory-bytes-per-cycle 1024 \
  --dfb-wait-block-scale 1.0 \
  --dfb-reserve-block-scale 1.0 \
  --copy-duration-scale 1.0 \
  --mismatch-threshold-pct 20 \
  --json-out cycle_report.json
```

Run `tt-lang-sim-cycles --help` to see all available options with their defaults.

---

## What the Estimator Measures

The simulator records trace events with integer **tick** values. One tick is one scheduler activation: the scheduler increments the tick counter each time it hands control to a kernel. This is the same logical time used by the simulator's tracing system (see `docs/TRACING.md`).

The estimator reads three classes of timing signals from the trace.

### Kernel span

`kernel_start` and `kernel_end` bracket the total time a kernel was active, including any time it spent blocked:

$$T_{\text{measured}} = t_{\text{kernel\_end}} - t_{\text{kernel\_start}}$$

`kernel_block` and `kernel_unblock` mark spans when the kernel was waiting on a DFB slot and doing no useful work:

$$T_{\text{blocked}} = \sum (t_{\text{kernel\_unblock}} - t_{\text{kernel\_block}})$$

$$T_{\text{active}} = T_{\text{measured}} - T_{\text{blocked}}$$

`T_measured` is the ground-truth cost used for error calculation. It is read from the trace and never fed back into the prediction path.

### Phase-duration signals

These are the primary inputs to the v0.1 estimator. Each is the accumulated tick span of paired begin/end events within a kernel's lifetime:

$$D_{\text{wait}} = \sum (t_{\text{dfb\_wait\_end}} - t_{\text{dfb\_wait\_begin}})$$

$$D_{\text{reserve}} = \sum (t_{\text{dfb\_reserve\_end}} - t_{\text{dfb\_reserve\_begin}})$$

$$D_{\text{copy}} = \sum (t_{\text{copy\_end}} - t_{\text{copy\_start}})$$

$D_{\text{wait}}$ captures how long a kernel spent waiting to consume from a DFB (consumer stall). $D_{\text{reserve}}$ captures how long it spent waiting to write into a DFB (producer stall). $D_{\text{copy}}$ captures DMA transfer time.

### Event counters

In addition to durations, the estimator counts individual events: `dfb_wait_begin`, `dfb_reserve_begin`, `dfb_push`, `dfb_pop`, and `copy_end`. These feed the per-event overhead terms described in the next section.

---

## The Estimation Model

For each kernel the estimator computes an estimate $\hat{T}$ composed of three additive groups: phase-duration contributions, overhead terms, and a roofline base.

### Phase-duration contributions (primary)

Each phase-duration signal is scaled by a configurable coefficient:

$$P_{\text{wait}} = D_{\text{wait}} \times s_{\text{wait}}$$

$$P_{\text{reserve}} = D_{\text{reserve}} \times s_{\text{reserve}}$$

$$P_{\text{copy}} = D_{\text{copy}} \times s_{\text{copy}}$$

The scale factors $s_{\text{wait}}$, $s_{\text{reserve}}$, $s_{\text{copy}}$ default to 1.0. They can be tuned with a profiler or the calibration suggestions
printed at the end of the report (see **Reading the Output**).

### Overhead terms (secondary)

Per-event fixed costs account for DFB coordination overhead:

$$S_{\text{stall}} = n_{\text{wait}} \cdot c_{\text{wait}} + n_{\text{reserve}} \cdot c_{\text{reserve}}$$

$$S_{\text{sync}} = (n_{\text{push}} + n_{\text{pop}}) \cdot c_{\text{sync}}$$

$$S_{\text{copy}} = n_{\text{copy\_calls}} \cdot c_{\text{copy}}$$

where $c_{*}$ are the per-event cycle costs set in `EstimatorConfig`.

### Roofline base (secondary context)

The roofline terms provide a hardware-ceiling lower bound and serve as a sanity check, not as the dominant cost signal. Tile counts from the trace drive the
computation:

$$C_{\text{compute}} = \frac{(\text{wait\_tiles} + \text{reserve\_tiles}) \times F_{\text{tile}}}{R_{\text{flops}}}$$

$$C_{\text{memory}} = \frac{\text{memory\_tiles} \times B_{\text{tile}}}{R_{\text{bytes}}}$$

$$C_{\text{roofline}} = \max(C_{\text{compute}},\; C_{\text{memory}})$$

$F_{\text{tile}}$ and $B_{\text{tile}}$ are the assumed flops and bytes per tile.
$R_{\text{flops}}$ and $R_{\text{bytes}}$ are the hardware peak rates, set via `--peak-flops-per-cycle` and `--memory-bytes-per-cycle`.

### Final estimate

$$\boxed{\hat{T} = P_{\text{wait}} + P_{\text{reserve}} + P_{\text{copy}} + C_{\text{roofline}} + S_{\text{stall}} + S_{\text{sync}} + S_{\text{copy}} + c_{\text{launch}}}$$

The blocked-cycle term $T_{\text{blocked}} \times w_{\text{blocked}}$ is also available but is **disabled by default** ($w_{\text{blocked}} = 0$).
Enabling it leaks observed duration directly into the prediction and reduces explainability. It should only be used for diagnostic ablation.

---

## Reading the Output

Running `tt-lang-sim-cycles trace.jsonl` prints three sections.

### Per-kernel table

```
Kernel                       Role       Measured  Estimated     Err%     Eff   M-Eff       OI Bound
node0-compute                compute         288     300.00     4.17    0.01    0.01      inf compute-bound
node0-read                   read            191     181.00    -5.24    0.04    0.04     0.00 memory-bound
node0-write                  write           350     361.00     3.14    0.01    0.01     0.00 memory-bound
```

| Column | Meaning |
|--------|---------|
| `Measured` | Ground-truth ticks from the trace ($T_{\text{measured}}$) |
| `Estimated` | Model prediction ($\hat{T}$) |
| `Err%` | Signed error: positive means the model over-predicts |
| `Eff` | Roofline efficiency of the estimate: $\min(1, C_{\text{roofline}} / \hat{T})$ |
| `M-Eff` | Roofline efficiency of the measured cycles |
| `OI` | Operational intensity: $\text{flops} / \text{bytes\_moved}$ |
| `Bound` | `compute-bound` or `memory-bound` (ties fall to `memory-bound`) |

A negative `Err%` means the model under-predicted (estimated fewer cycles than observed). A low `Eff` value means the dominant cost is stall or sync overhead, not compute or memory throughput.

Below the table the report prints the overall Weighted Absolute Percentage Error (WAPE), which weights each kernel's error by its measured cycle count so heavier kernels influence the summary more than lightweight ones:

$$\text{WAPE} = \frac{\sum_{i} |\hat{T}_i - T_i|}{\sum_{i} T_i} \times 100 \quad (T_i > 0)$$

### Kernel-group totals

Kernels are grouped by node. The group estimate uses a heuristic critical-path aggregation that avoids naively summing fully overlappable read/compute/write phases:

$$\hat{T}_{\text{group}} = \max_{k \in \text{group}}(\hat{T}_k - S_{\text{sync},k}) + \max_{k \in \text{group}}(S_{\text{sync},k})$$

The group `Err%` is the signed error of the group estimate against the highest measured per-kernel cycle count within the group, which approximates the node's critical-path latency.

### Diagnostics section

The report ends with three diagnostic blocks:

- **Ablation diagnostics** show how much the blocked-cycle term contributes to the estimate. When `blocked_cycle_weight = 0` (the default), the full-model WAPE and the no-blocked-term WAPE are identical, confirming the estimate is driven entirely by trace event metadata:

  ```
  Ablation Diagnostics (internal)
  - Full model WAPE%: 4.48
  - No-blocked-term WAPE%: 4.48
  - Blocked term share of estimated cycles %: 0.00
  ```

- **Feature provenance** shows where each feature comes from — ground-truth trace, derived, or config — so the estimate remains fully auditable.

- **Calibration suggestions** print a recommended scale factor per role. These are derived from the ratio of measured cycles to the current non-blocked estimate:

  $$\hat{s}_{\text{role}} = \frac{\sum_{i \in \text{role}} T_i}{\sum_{i \in \text{role}} (\hat{T}_i - T_{\text{blocked},i} \cdot w_{\text{blocked}})}$$

  A value above 1.0 means the model under-predicts that role; below 1.0 means it over-predicts. Pass the suggested value to the corresponding `--*-block-scale` flag to improve accuracy for that role.

---

## Mismatch Analysis

For each kernel above the mismatch threshold (`--mismatch-threshold-pct`, default 20%), the estimator assigns a reason code and decides whether to recommend escalation to a lower-level model.

The reason codes and their priority order are:

| Reason | Condition | Action |
|--------|-----------|--------|
| `within-threshold` | $\text{abs\_err} \leq \tau$ | None required |
| `stall-dominated` | $T_{\text{blocked}} / T_{\text{measured}} \geq 0.30$ | Refine wait/sync model first; do not escalate yet |
| `blocked-term-dominated` | blocked term $/ \hat{T} \geq 0.50$ | Audit `blocked_cycle_weight`; do not escalate yet |
| `unknown-kernel-role` | role is `other` | Add semantic role tagging to the kernel |
| `no work signal in trace` | no tile counts in trace | Add op-level counters to the trace |
| `roofline-parameter mismatch` | none of the above | Tune roofline parameters or escalate |

The `needs_lower_level_model` flag is set **only** when both conditions hold:

$$\text{abs\_err} > \tau \quad \text{and} \quad \text{reason} = \texttt{roofline-parameter mismatch}$$

This gate prevents premature escalation to Tensix-level modeling when the root cause is a stall pattern or missing trace signal that can be resolved at the current modeling level.

---

## Configuration Reference

All parameters are set via CLI flags and stored in `EstimatorConfig`. The defaults are shown below.

| Flag | Default | Description |
|------|---------|-------------|
| `--flops-per-tile` | 2048 | Assumed flops per compute tile ($F_{\text{tile}}$) |
| `--bytes-per-tile` | 2048 | Assumed bytes per memory tile ($B_{\text{tile}}$) |
| `--peak-flops-per-cycle` | 4096 | Compute roofline peak ($R_{\text{flops}}$) |
| `--memory-bytes-per-cycle` | 1024 | Memory roofline peak ($R_{\text{bytes}}$) |
| `--dfb-wait-block-scale` | 1.0 | Scaling factor $s_{\text{wait}}$ |
| `--dfb-reserve-block-scale` | 1.0 | Scaling factor $s_{\text{reserve}}$ |
| `--copy-duration-scale` | 1.0 | Scaling factor $s_{\text{copy}}$ |
| `--wait-event-cycles` | 2.0 | Per-event cost $c_{\text{wait}}$ |
| `--reserve-event-cycles` | 2.0 | Per-event cost $c_{\text{reserve}}$ |
| `--sync-event-cycles` | 1.0 | Per-event cost $c_{\text{sync}}$ |
| `--copy-call-cycles` | 4.0 | Per-event cost $c_{\text{copy}}$ |
| `--blocked-cycle-weight` | 0.0 | Weight $w_{\text{blocked}}$ (keep at 0 to avoid leakage) |
| `--kernel-launch-cycles` | 0.0 | Fixed launch overhead $c_{\text{launch}}$ per kernel |
| `--mismatch-threshold-pct` | 20.0 | Error threshold $\tau$ for mismatch classification |
| `--json-out` | None | Path for the JSON report |
| `--include-zero-kernels` | False | Include kernels where both measured and estimated are zero |

---

## Implementation Layout

The estimator is organized as a package under `python/sim_stats/cycle_tools/`:

| File | Responsibility |
|------|----------------|
| `types.py` | Dataclass definitions (`TraceEvent`, `KernelFeatures`, `EstimatorConfig`, `KernelEstimate`, `KernelGroupEstimate`) |
| `parse.py` | Trace parsing and feature extraction |
| `model.py` | Estimation, grouping, and mismatch classification |
| `report.py` | Terminal and JSON reporting, diagnostics |
| `cli.py` | Argument parsing and pipeline wiring |

`python/sim_stats/cycle_estimator.py` is a thin compatibility shim that re-exports the public API and preserves the `tt-lang-sim-cycles` entrypoint. Shared trace helpers (JSONL reading, kernel name parsing) live in `python/sim_stats/utils.py` and are reused by both `tt-lang-sim-cycles` and `tt-lang-sim-stats`.

---

## Known Limitations

1. Regression fixtures are not yet formalized. Model outputs should be locked against a reference trace to detect regressions across estimator changes.
2. Phase-duration scale factors default to 1.0 and have not been hardware-calibrated. Use the calibration suggestions in the report and verify against real device profiles when available.
3. Read kernels tend to show slightly higher error than compute and write kernels. Tuning `--dfb-reserve-block-scale` is the first step to address this.
4. The group critical-path aggregation is heuristic. A dependency-graph-aware critical path would improve multi-role overlap accuracy in future iterations.

## Suggested Next Steps

1. Add fixed-trace regression tests that lock the output schema and WAPE summary.
2. Collect hardware profiles to calibrate `flops_per_tile`, `bytes_per_tile`, and peak rates for specific device configurations.
3. Write a calibration guide showing how to use `--json-out` output and the calibration suggestions to iteratively tune scale factors.
4. Decide on the timeline for slimming or removing the compatibility shim (`sim_stats/cycle_estimator.py`).
