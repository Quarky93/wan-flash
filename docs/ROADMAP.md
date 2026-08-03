# Performance roadmap (2026-07-31 investigation)

Evidence base: SASS dissection of the running cubins + nsys timelines + NVML
power/clock sampling + targeted A/B experiments (ncu is hard-blocked on this
host: RmProfilingAdminOnly=1 and no CAP_PERFMON in the container — restoring
counter access is an infra ask below). Full findings in the session archive;
key facts:

- **Forward is at the wall.** 86–90% tensor-core-busy at the 698 W power cap;
  at h40/75600 our per-clock throughput is IDENTICAL to FA3's (3694 vs 3696
  FLOP/SM/clk) — the entire 2.3% lead is clock bought by multicast's energy
  savings. Zero spills, zero wave-quantization loss (measured flat on
  exact-fill A/B), consumer loop at the algorithmic instruction floor.
  Further forward wins must REMOVE ENERGY, not instructions.
- **Backward holds the real pocket:** 765–1010 idle tensor cycles per tile
  (TC-busy 75.9–80.5% vs ideal) — every sm90 implementation (ours, FA3, FA4)
  is stuck in the same band and ours is already best. Plus a quantified
  dQaccum L2-RMW energy tax (full-removal ablation: −9.9% time, +127 MHz).
- **Cross-bwd:** 38.5% of the chain is preprocess/postprocess/fills, not math.
- fp8/precision backward: confirmed nonexistent (2026 releases audited). RoPE
  fusion into the kernel: measured-out (best variant +0.14% step; Q+K variant
  net NEGATIVE). Cross-attn redesign: bounded at ~0.3% step. max_offset:
  an FP8 saturation trick, identically 0 at bf16 — do not re-propose.

## Phase 1 — sure wins (~2 days, low risk) — **DONE 2026-07-31**

| item | expected | measured outcome |
|---|---|---|
| Pinned-memory H2D fix in trainer data path | −0.4% step @480p | **KEPT: −0.27%** synthetic, neutral real-data; batches bitwise-identical |
| 4-wide FADD tree for row_sum (mirrors the measured fmax win) | +0.3–1.0% fwd | **REJECTED: 0.3–0.9% slower** all self shapes (ABBA A/B; row_sum is off the exp2 critical path — see FEATURES.md) |
| batch=2 instead of 2× grad-accum (**recipe-permitting**) | −1–1.5% step/sample | **EXCLUDED by user constraint (no VRAM increases).** Measured for the record: 480p −0.6…−1.0%/sample (+4.2 GiB); 720p neutral (+13.3 GiB) |
| cvt hygiene + cross zero_init tidy | ~0, bit-neutral | **AUDIT CLEAN** — narrowing cvts already packed; fills ≈14 µs (0.6% of chain) |

## Phase 2 — the dQ campaign (~1–1.5 weeks, medium risk)

1. **Cluster-pair dQ reduction over DSM** — **REJECTED 2026-07-31 after a
   1-day spike** (full evidence in FEATURES.md; code on branch
   `dq-dsm-experiment`). The premise silently assumed an fp32 DSM reduce;
   sm90's is integer-only (ptxas-probed), smem has no room for copy
   staging, and every feasible ld+add+st combine placement (store warp
   0.22×, 24-reg crew 0.73–0.78×, MMA WGs 0.58–0.61×) costs more than the
   measured +2.4–3.5% upper bound (handshake+half-flush with adds
   skipped). An sm100 technique; do not re-attempt on sm90.
2. **cluster_n=4** — **REJECTED 2026-07-31**: implemented (one-line assert
   widening — the machinery generalized), exact incl. multi-phantom tails,
   but 4.7–6.4% SLOWER than cluster_n=2 at all self shapes (4-way lockstep
   jitter coupling + quarter-tile multicast boxes). Expected +1.5–2.5%;
   the second traffic halving does not pay on sm90.

**Phase 2 closed 2026-07-31: both items refuted by measurement.** The bwd
dQaccum L2-RMW pocket (+127 MHz if removed) is real but unreachable on
sm90 — it needs sm100's fp32 DSM reduce or bigger smem. Remaining live
items: Phase 2.5 (double-buffered sQ, fwd @32760 only) and Phase 3.

## Phase 3 — the bubble (timeboxed 2 weeks, exploratory, high risk)

Attack the ~20% tensor-idle dS critical path (S/dP wait → exp+dS → cvt →
STSM → fence → 3 dependent WGMMAs): deepen dS staging / start the next
m-block's S-GEMM before dQ drains. Full closure is +19–24% bwd kernel;
power-elasticity (measured 0.65) and history cap the realistic yield at
**+3–6% bwd**. smem is exhausted (227.5/228 KB) so staging must trade against
the K/V piggyback — prototype-first with a hard go/no-go gate at day 5.
This is the largest remaining pocket on sm90, and it is unmined by FA3/FA4
too — landing it would be a genuinely novel result.

**Probe #1 done 2026-08-01 (dV-GEMM hoist): REJECTED, −15.5% uniformly** —
a WGMMA issue mid-pointwise drags a `wgmma.fence` that serializes the ALU
section around it (FEATURES.md). Constraint learned: the pocket cannot be
mined by fine-grained GEMM reordering; the viable shape is a coarse-grain
loop rotation (next block's S-GEMM issued inside the previous block's
epilogue region, GEMM bursts kept contiguous) — the full Phase-3 build,
with acc_S double-buffering as its register-budget crux.

**Full build done 2026-08-01: NO-GO on sm90 + CuTeDSL** (branch
`phase3-rotation-experiment`, correct at all shapes). The rotation was
built and iterated through four measured failure modes — branch-in-body
(0.35×), duplicated gemm call-sites with preallocated accumulators
(fence+DEPBAR around EVERY HGMMA: 68 vs base's 5), register spills
(STACK:160 → fixed), and finally the root blocker: **the CuTeDSL wgmma
pipeliner falls back to per-instruction fencing for the whole function
whenever a wgmma group is left pending across the scf.for back-edge**
(diagnostic: adding one retire-wait before the back-edge takes fences
34 → 5). A pending cross-iteration group IS the rotation's mechanism, so
the technique is inexpressible without full serialization. With all
groups retired per block, the best reachable reorder measures
0.93–0.98× — base's structure is already optimal in that class. Paths
that could reopen this: an upstream CuTeDSL change (allow pending wgmma
groups across loop back-edges), or hand-written CUDA/PTX for the
consumer loop. **The dS-bubble pocket survives as unmined on sm90 — now
with the precise reason no one has mined it.**

**Inline-PTX escape hatch tested 2026-08-03: DOES NOT WORK** (branch
`phase3-ptx-wgmma`). Hypothesis: issue the boundary-crossing GEMM as
inline PTX so the MLIR pass never sees the pending group. Measured: one
PTX wgmma anywhere in the function makes the pass bail to
per-instruction fencing globally — in the BASE loop shape (5 fences
normally) it jumps to **34**, timing 0.80–0.85×. The pass treats an
opaque asm block as unknown wgmma state and fences every DSL wgmma it
owns, so PTX+DSL mixing is strictly worse than either alone. Supporting
findings: `inline_ptx`'s `read_write_args` silently discards updated
values (accumulators need an explicit 64-wide mov ladder per GEMM), and
`MakeGMMASmemDescOp` exists in the dialect but has no smem_desc→i64
cast, so every descriptor must be hand-encoded. The only surviving
variant is an ALL-PTX consumer loop (all 5 GEMMs, ss-mode double
descriptors, mov ladders throughout) — i.e. the CUDA rewrite scoped to
one function, carrying a mov-ladder tax of the same order as the
+0.5–1% step prize. **Not recommended.**

Optional Phase 2.5: double-buffered sQ — **REJECTED 2026-08-01: measured
FLAT** (the shipped early-Q-release already covers the tile boundary;
FEATURES.md has the numbers + a producer/consumer PipelineState deadlock
lesson).

Cross-bwd chain fusion (fold pre/post into main; −28–35% of the cross chain,
1.65× → ~2.3× FA3 headline) — schedule when the headline ratio matters more
than the ~0.3% step it buys.

## Explicit rejections (do not revisit without new evidence)

RoPE-in-kernel fusion · cross-attn multi-head-per-CTA redesign · CUDA graphs
(0.2–0.6% for medium-high silent-bug risk) · layer-level PDL (<0.05%) ·
tile_m 192 (forces overlap off: −14%) · max_offset · TMA-store elision ·
sm100 two-kernel bwd split (−40% GEMM work) · any fp8 backward (nonexistent).

## Infra ask (0.5 d of host-admin time, not ours)

Set `RmProfilingAdminOnly=0` or grant CAP_PERFMON to the container →
restores ncu/CUPTI. Unlocks true warp-stall attribution for Phase 3 and
locked-clock A/Bs (separating IPC from energy wins in one run).

## Expected totals if all phases land

Attention −2 to −6% (backward-weighted; more at 720p), training step
−1 to −2.5%, plus the recipe-conditional batch-2 win. Stated plainly: the
forward is finished, the remaining pool is backward-shaped, and Phase 3 is
the only item that could still move the needle by more than a point or two.
