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
| batch=2 instead of 2× grad-accum (**recipe-permitting**) | −1–1.5% step/sample | **480p: −0.6…−1.0%/sample (+4.2 GiB); 720p: neutral (+13.3 GiB).** Report-only; defaults unchanged |
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

Optional Phase 2.5: double-buffered sQ for the S=32760 shapes (+1.5–2.5% fwd
there, ~0 at 75600; 2–4 d; must re-verify the zero-quantization behavior).

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
