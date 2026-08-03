# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Greenfield CuTeDSL flash-attention kernels (forward + backward) specialized for
Wan2.1 video-diffusion **training** on H100/sm90. Not a wrapper around
FlashAttention — new source, hard-specialized to one workload: bf16, head_dim
128, MHA, non-causal, no mask/dropout/softcap, batch 1, `S ∈ {32760, 75600}`
self-attention, `S×512` (or `S×257` I2V) cross-attention, heads ∈ {12, 40}.

FlashAttention-3 and -4 are installed **only as measurement baselines** — no
kernel code is imported from them.

Consumer: [wan-attn](https://github.com/Quarky93/wan-attn) dispatches to these
kernels through its `wan_attention` custom op.

## Environment

There is no venv in this repo. Everything runs from the sibling project's:

```bash
/workspace/wan-attn/.venv/bin/python        # torch 2.12.0+cu130, cutlass-dsl 4.6.0
```

wan-flash is installed editable into it, so edits here are live there.

## Commands

```bash
# fast battery (~25 s) — run before every commit
/workspace/wan-attn/.venv/bin/python -m pytest tests/ -q

# add true Wan shapes (32760 / 75600); ~35 s, run before shipping a kernel change
WAN_FLASH_SLOW_TESTS=1 /workspace/wan-attn/.venv/bin/python -m pytest tests/ -q

# one test
/workspace/wan-attn/.venv/bin/python -m pytest tests/test_bwd.py -q -k alt_configs

# baseline comparison (fa3 / fa4 / wan) across shapes
/workspace/wan-attn/.venv/bin/python bench/bench.py --impl wan fa3 fa4 \
    --shapes all --modes fwd bwd

# THE measurement tool for any default change (see "Measurement protocol")
/workspace/wan-attn/.venv/bin/python tools/ab_feature.py bwd cluster_n 2 4
```

## Architecture

**Three kernels, one host driver.** `interface.py` owns torch↔cute conversion
and a compile cache keyed by `(shape, features, scheduler, nsplit, cluster_n)`;
everything is static per compile — shapes are baked in, so a new shape means a
new compile (warm one untimed call before benching).

- `fwd_sm90.py` — single warp-specialized kernel. 384 threads: 1 producer
  warpgroup (warp 0 issues TMA, `setmaxregister` 24) + 2 consumer MMA
  warpgroups (240). K/V TMA pipelines, separate Q pipeline, KV blocks iterated
  **descending** so the ragged tail block merges into the `is_first` online-
  softmax step. exp2-domain softmax with the scale folded into `scale_log2`.
- `bwd_sm90.py` — three-kernel chain: preprocess (`D = rowsum(O∘dO)`, lse→log2,
  zero dQaccum) → main (dK/dV-stationary, warp-specialized, 80×128 tiles,
  SdP/dQ swapAB, dK/dV register-resident) → postprocess (fp32 dQaccum →
  scale → bf16). Warp 1 of the producer group is a dedicated dQ store warp
  (`cp.reduce.async.bulk.add.f32`), ping-ponging `sdQaccum` with the MMA
  warpgroups through named barriers.
- `features.py` — every technique choice is a flag with a **measured** default.
  `docs/FEATURES.md` records the verdict and numbers for each, including
  rejected ones.

**Wan-specific designs that differ from FA3/FA4** (don't "fix" these back):
2-CTA cluster with K/V multicast in fwd and Q/dO multicast in bwd (FA4's sm90
leaves multicast unused); split-M in bwd for the tiny-KV cross-attention shapes
(fixes an occupancy pathology both upstreams have); per-shape `scheduler="auto"`
policy resolved in `interface.py`.

## Measurement protocol (non-negotiable)

Margins here are 1–3% and run-to-run spread is ~0.5%, so **an unpaired
single-run comparison cannot decide anything.** Use `tools/ab_feature.py`,
which does paired ABBA rounds in one process with burn-in and reports
median + IQR. Every default in `features.py` was set this way; a change that
cannot show a paired win does not land.

The kernels run at a 698 W power wall, so wins usually appear as **clock**
(NVML MHz) rather than idle time — removing work (memory traffic, ALU issue
slots) converts to frequency, not visibly to occupancy.

## Correctness gates

`tests/oracle.py` is a chunked-fp32 reference (a naive one cannot fit at
S=75,600). The pass rule is the FA test-suite convention: **candidate error
≤ 2× FA3's own error** vs the oracle, per tensor (o, dq, dk, dv separately).
Useful canaries when changing the backward: dk/dv are bitwise-deterministic at
`nsplit=1`, so `torch.equal` against the pre-change build is a fast exactness
check; dq is atomically accumulated and therefore nondeterministic by design.

## CuTeDSL footguns this kernel has hit

`docs/CUTEDSL_COOKBOOK.md` is the full list. The expensive ones:

- **Loop-carried state**: objects mutated only inside a `self.`-method called
  from a `cutlass.range` body are NOT loop-carried. Return and reassign
  (`q_state, kv_state = self._consumer_tile(...)`).
- **`TensorSSA.to(bf16)`** can emit one scalar `cvt` per element. Use the
  packed `cvt.rn.bf16x2.f32` idiom (`_cvt_bf16_frag`) on any hot path.
- **Cluster exit safety**: a CTA that exits while a peer may still target its
  mbarriers faults (flaky CUDA 719). Every multicast pipeline needs
  `producer_tail` before exit; phantom tail CTAs do full compute and skip only
  stores.
- **Pipelines have two states** (producer and consumer). Changing stage count
  on one side only deadlocks — and only at ≥2 tiles per slot, so small smoke
  shapes pass. Always smoke a multi-tile persistent config.
- **The MLIR wgmma pass** fences every wgmma in the function if any wgmma group
  crosses a loop back-edge, or if any inline-PTX asm block appears. Both were
  measured; see `docs/ROADMAP.md` before attempting software pipelining.
- Spill check is cheap and worth doing after register-pressure changes:
  `cuobjdump -res-usage` on the cubin (`CUTE_DSL_KEEP=cubin`) — `STACK:0` is
  required; ptxas budgets 168 regs/thread here, not the 240 `setmaxnreg` cap.

## Before proposing an optimization

Read `docs/FEATURES.md` (rejected ideas with numbers) and `docs/ROADMAP.md`
(remaining headroom, what blocks each item). Several plausible ideas —
cluster-pair dQ reduction over DSM, `cluster_n=4`, dV-GEMM hoisting, software-
pipelined loop rotation, inline-PTX wgmma, double-buffered sQ — were built,
measured, and rejected; their code is on branches `dq-dsm-experiment`,
`phase3-rotation-experiment`, `phase3-ptx-wgmma`.

The forward is at the practical floor (86–90% tensor-busy, per-clock parity
with FA3's C++ kernel). The backward's ~20% tensor-idle pocket is real but
locked behind the compiler pass above.
