# FEATURES — kernel A/B verdicts (fwd + bwd)

All numbers: H100 SXM (132 SMs), bf16, head_dim 128, `triton.testing.do_bench`
(warmup 200 ms, rep 1000 ms), TFLOP/s = `4*b*h*sq*skv*d / t`. One clean run of
`python -m bench.bench --impl fa3 fa4 wan --shapes all` (2026-07-30, defaults =
cluster (2,1) K/V multicast + scheduler auto + packed bf16 cvt + rescale_skip
8.0 + overlap + 128x128 + 2 stages):

| shape | FA3 (C++) | FA4 (CuTeDSL) | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|---|---|
| self h12 S=32760  | 9.658 ms / 682.8  | 10.045 ms / 656.6 | 9.496 ms / 694.5  | **1.017x** | **1.058x** |
| self h12 S=75600  | 51.136 ms / 686.7 | 52.849 ms / 664.4 | 50.324 ms / 697.8 | **1.016x** | **1.050x** |
| self h40 S=32760  | 31.859 ms / 689.9 | 33.222 ms / 661.6 | 31.266 ms / 703.0 | **1.019x** | **1.063x** |
| self h40 S=75600  | 172.416 ms / 678.9 | 176.509 ms / 663.1 | 169.211 ms / 691.7 | **1.019x** | **1.043x** |
| cross h12 32760x512 | 0.189 ms / 544.0 | 0.223 ms / 462.7 | 0.189 ms / 544.4 | **1.001x** | **1.176x** |
| cross h12 75600x512 | 0.444 ms / 536.1 | 0.521 ms / 456.5 | 0.434 ms / 547.9 | **1.022x** | **1.200x** |
| cross h40 32760x512 | 0.640 ms / 536.4 | 0.763 ms / 450.1 | 0.628 ms / 546.7 | **1.019x** | **1.215x** |
| cross h40 75600x512 | 1.513 ms / 523.8 | 1.752 ms / 452.5 | 1.456 ms / 544.3 | **1.039x** | **1.203x** |

Run-to-run noise ~±0.5-1% (steady-state interleaved medians agree within
±0.3%: self ratios 1.008-1.024 across three protocols). Session 2026-07-30
closed the last ~3% to FA3 with three levers, in order of discovery:
cluster multicast (+2-3%), packed `cvt.rn.bf16x2.f32` for P (+2-3%), and a
per-shape scheduler policy (+1.5% at h40/75600 only). Pre-session numbers are
in the git history of this file.

## Verdict table (defaults in wan_flash/features.py)

| feature | default | verdict | numbers (h12, TFLOP/s) |
|---|---|---|---|
| `intra_wg_overlap` | **True** | KEEP. Single biggest win. QK(i+1)+PV(i) in flight while softmax(i+1) runs (`wait_group(1)`), plus consumer-WG ping-pong named barriers. | self 32760: 562.4 → 641.3 (+14%); self 75600: 570.9 → 645.6 (+13%) |
| `cluster_mn` | **(2, 1)** | KEEP. 2-CTA cluster = two m-blocks of the same head; K/V TMA multicast (each CTA loads half the box, hardware fans out; `CopyBulkTensorTileG2SMulticastOp`, consumer arrive count × 2, `make_layout_image_mask`). Halves K/V L2/HBM traffic — at the 698 W power wall that buys ~45 MHz of clock (1457 → 1503 MHz measured at h40 32760). The C++ FA3 hopper trick that FA4 sm90 leaves unused ("No mcast for now"). Grid pads to whole clusters; a phantom tail CTA (odd m_blocks) recomputes its peer's m-block and skips stores. | on top of packed-cvt build, steady-state medians: single sched 0.992/0.992/0.979/0.997 → 1.014/1.020/1.008/1.024 vs FA3 (h12-32760/h12-75600/h40-32760/h40-75600); persistent sched 1.001/1.004/0.991/1.007 → 1.023/1.023/1.020/1.006 |
| packed P cvt | (always on) | KEEP. `convert_P` fp32→bf16 via `cvt.rn.bf16x2.f32` pairs instead of TensorSSA `.to(bf16)` (which emits scalar cvts here — same trap the bwd hit, cookbook 4.3). Runs once per KV block on the softmax critical path: the single biggest lever of the 2026-07-30 session. Epilogue O cvt also packed (neutral, kept for consistency). | single+cluster: 672.7 → 688.0 @32760 (+2.3%), 686.9 → 703.0 @75600 (+2.3%); identical bits (both round-nearest) |
| `scheduler` | **"auto"** | Per-shape policy, resolved in interface.py: **persistent** everywhere except units-per-cluster-slot ≥ 128 AND n_blocks ≥ 16 → **single** (among Wan shapes: only self h40@75600). Persistent chains every cluster pair for the whole kernel (~170 ms at h40/75600) — accumulated pair-coupling drift there measures 1.5% slower than single (pairs retire per tile); at ≤78-unit chains persistent wins ~1% (prologue amortization + separate-sO O-drain overlap). | h40 75600 single 167.5 ms (698.7) vs persistent 170.1 ms (688.0), 3× reproduced; all other self shapes persistent wins by 0.5-1.2%; cross: persistent+cluster 1.029/1.042 vs FA3 at h40 |
| `tile_n` | **128** | KEEP 128. Re-swept under cluster multicast: 120 → 659.0/662.6/668.4/660.3; 144 → 632.9-641.8; 160 → 636.4-653.1 (all four self shapes, vs 664-679 at 128). FA3's 176 config also loses in our architecture (single+cluster probe @32760: 651.3/659.1 vs 674.6/683.3). | pre-cluster sweep in git history |
| `rescale_skip_threshold` | **8.0** | KEEP. Port of FA4's sm100-only trick to sm90 (not in FA4 sm90): if the running max moved < 8 log2-units, keep the stale max ⇒ acc_scale = 1.0, and skip the whole O-rescale FMA loop when the warp agrees (`vote.any`). Mathematically exact (max-shift invariance); P ≤ 2^8, far inside bf16/fp32 range. | on single sched: 638.8 → 652.0 (+2.1%), 645.2 → 656.7 (+1.8%). on persistent: 652.1 → 663.8 (+1.8%), 664.7 → 666.8 (+0.3%) |
| `num_stages` | **2** | KEEP 2. Re-tested under single+cluster (no sO, smem headroom exists): 3 stages = 671.2/687.0/680.2/678.2 vs 673.9/684.6/683.9/678.7 — noise-level mixed. | also pre-cluster: +0.3% @single |
| `mma_pv_is_rs` | True | Only RS implemented (P never leaves registers; FA4 default at hd128). SS path not built — FA4's own data says RS wins at 128×128. | — |
| fmax reduce | 4-wide tree | KEEP. `TensorSSA.reduce(MAX)` emits a serial FMAX chain; hand 4-wide tree (FA3/FA4 idiom) gives ILP 4. | self 75600: 658.7 → 667.9 (+1.4%); 32760 flat |
| mainloop `unroll` | 1 | KEEP 1. unroll=2 helped 32760 (+0.8%) but catastrophically regressed 75600 (661 → 440, reproducible; pathological codegen). Not robust. | — |

## Correctness gates (all green)

- `tests/test_fwd.py` fast battery (6 shapes incl. ragged 960/3000, odd
  m_blocks 1152 = cluster phantom tile, cross 512/257): PASS with the
  cluster+auto defaults.
- Alt-config battery (`test_fwd_alt_configs`): scheduler ∈ {single,
  persistent} × cluster ∈ {(1,1),(2,1)} at 1152 + 3000 (covers the auto
  policy's "single" branch and both phantom paths): PASS.
- `WAN_FLASH_SLOW_TESTS=1` true shapes self h12 32760 + 75600 (75600 has odd
  m_blocks = 591 → persistent phantom path at true shape): PASS.
- Feature matrix re-verified vs the chunked-fp32 oracle for: tile_n ∈ {120, 128, 144, 160, 176, 192}, rescale_skip ∈ {0, 8}, scheduler ∈ {single, persistent}, overlap ∈ {on, off} — rel_l2 ≈ 2.2e-3 (bf16 floor, same as FA3), LSE max err ≤ 1.5e-6.
- Packed cvt is bit-identical to `.to(bf16)` (both cvt.rn); vs-FA3 max|dO|
  unchanged at 9.8e-4-3.9e-3 across the battery.
- Output contract: o bf16 (b,s,h,d); lse fp32 (b,h,s), natural log,
  `lse = (row_max*scale_log2 + log2(row_sum)) * ln2`, m-tail predicated.

## Notes / non-defaults worth knowing

- Cluster mechanics: K/V pipelines get `cta_layout_vmnk=(1,2,1,1)` and a
  consumer group of `2 * mma_warps` (each consumer warp's lanes 0-1 signal
  the empty barrier of BOTH CTAs, `mbarrier_arrive` with peer rank); tx_count
  stays the full tile (two half-box TMAs land per CTA per stage).
  `pipeline_init_arrive/wait` become cluster_arrive/cluster_wait. Both CTAs
  of a cluster must run the SAME tile count in lockstep — that's why the
  phantom tail tile does full compute and only skips stores.
- Power wall context: all impls run pegged at 698 W on this box. Wins come
  from removing work (multicast: memory system; packed cvt: ALU issue slots),
  which converts to clock headroom, not idle time.
- FA3's kBlockN=176 + our architecture measured slower than 128 (see tile_n
  row) — its C++ advantage was fully explained by the scalar-cvt bug + no
  multicast; after both fixes we lead FA3 at every Wan shape.
- The DSL loop-transform gotcha that cost the most debugging time: objects
  mutated only inside a `self.`-method called from a `cutlass.range` body are
  NOT loop-carried (the region analyzer tracks assignments and direct method
  receivers, and explicitly skips `self`). Pipeline states must be returned
  and reassigned in the loop body (`q_state, kv_state = self._consumer_tile(...)`).
- Second DSL codegen trap (now hit twice, fwd + bwd): TensorSSA `.to(bf16)`
  on accumulator fragments can emit one scalar `cvt` per element. Always use
  the packed `cvt.rn.bf16x2.f32` inline-asm idiom on hot paths.

---

# BACKWARD (wan_flash/bwd_sm90.py, 2026-07-29)

Three-kernel chain per docs/BWD_STUDY.md: preprocess (D=rowsum(O*dO),
lse*log2e with +inf pad sentinel, dQaccum zero) -> main (dK/dV-stationary,
384 threads: producer warp 0 TMA, warp 1 dQ `cp.reduce.async.bulk.add.f32`,
2 MMA WGs x 5 WGMMAs; tile 80x128, stages 2/2/2, SdP_swapAB + dQ_swapAB,
AtomLayout(1,2,1) => dK/dV RS from registers) -> postprocess (dQaccum fp32
fragment-order -> *scale -> bf16). Inputs (q,k,v,o,lse,do), lse NATURAL log
(b,h,s) fp32 = the FA3/FA4 contract (drop-in for FA3-fwd hybrids, tested).

One clean run, `bench.bench --impl fa3 fa4 wan --shapes all --modes bwd`
(2026-07-29, final defaults; bwd TFLOP/s = 2.5 * 4*b*h*sq*skv*d / t;
identical o/lse/do from FA3's fwd fed to all three):

| shape | FA3 raw bwd | FA4 raw bwd | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|---|---|
| self h12 S=32760  | 27.512 ms / 599.2 | 26.644 ms / 618.7 | 26.687 ms / 617.7 | **1.031x** | 0.998x |
| self h12 S=75600  | 144.050 ms / 609.4 | 139.518 ms / 629.2 | 138.986 ms / 631.6 | **1.036x** | **1.004x** |
| self h40 S=32760  | 90.670 ms / 606.0 | 87.963 ms / 624.7 | 88.390 ms / 621.7 | **1.026x** | 0.995x |
| self h40 S=75600  | 491.066 ms / 595.9 | 467.720 ms / 625.6 | 476.565 ms / 614.0 | **1.030x** | 0.981x |
| cross h12 32760x512 | 1.179 ms / 218.5 | 1.148 ms / 224.4 | 0.714 ms / 360.8 | **1.651x** | **1.608x** |
| cross h12 75600x512 | 2.700 ms / 220.2 | 2.625 ms / 226.5 | 1.591 ms / 373.8 | **1.698x** | **1.650x** |
| cross h40 32760x512 | 2.852 ms / 301.1 | 2.821 ms / 304.4 | 2.294 ms / 374.4 | **1.243x** | **1.230x** |
| cross h40 75600x512 | 6.555 ms / 302.3 | 6.488 ms / 305.5 | 5.197 ms / 381.4 | **1.261x** | **1.248x** |

Run-to-run noise ~±0.5-1% on self shapes; bench order is (fa3, fa4, wan)
per shape, wan measured last (warmest GPU).

## Verdict table (defaults in wan_flash/features.py BwdFeatures)

| feature | default | verdict | numbers (main kernel, h12 S=32760 unless noted) |
|---|---|---|---|
| packed `cvt.rn.bf16x2.f32` for P/dS | **on** | KEEP. TensorSSA `.to(bf16)` emits one scalar cvt per element at our 128x80 accum shape (FA4's known trap, cookbook 4.3). Biggest single win. | 29.38 -> 24.06 ms main kernel (+18%); wall 31.5 -> 27.0 ms |
| PDL chain (pre -> main) | **on** | KEEP. `use_pdl=True` on preprocess+main; preprocess waits before O/dO, signals after loads; main's producer waits between load_Q(0) and the first load_LSE, so K/Q(0) TMA overlaps the preprocess tail. | ~0.3-0.9 ms of launch overlap per call |
| K-tail mask form | select, every iter | KEEP FA4's unconditional predicated selects. A runtime `if is_tail_cta:` skip-branch around them measured 5% SLOWER (scf.if region blocks scheduling). Compiled out when skv % 128 == 0 (cross 512). | branch: 27.08 ms vs select: 25.82 ms wall |
| K/V tx piggyback | **on** | KEEP. K/V ride the first Q/dO stage mbarriers via raw `mbarrier_expect_tx` (stock pipeline + 2 PTX lines instead of FA4's pipeline subclass); first S GEMM waits Q+K only, V lands with dO(0). | h40 75600 main 501.8 -> 487.2 ms; neutral at h12 |
| `nsplit` (split-M, cross) | **0 = auto** | KEEP. Grid (n_blocks*nsplit, h, b); dK/dV via fp32 gmem bulk-reduce-add + generalized postprocess. Auto: 1 if wave eff >= 0.85 else smallest split with eff >= 0.95 (h12 cross -> 16ish, h40 -> 4). THE cross-attn fix: 48/160 CTAs on 132 SMs was 36%/61% occupancy. | cross h12 32760: 1.137 -> 0.726 ms (+57%); h40: 2.85 -> 2.29 ms (+25%) |
| `tile_m` | **80** | KEEP 80 (FA3/FA4 non-causal config, dQ_swapAB). 64 (causal-config variant, no dQ_swapAB) compiles+passes but was not faster where tried. | — |
| `num_stages` | **2** | KEEP. 1 passes the battery (PdS single-buffer barrier path exercised) but starves the Q/dO prefetch. | — |
| dQ accumulation | fp32 bulk atomics | FA3/FA4 default; dq nondeterministic run-to-run (~1e-4 rel), dk/dv bitwise-deterministic at nsplit=1. nsplit>1 makes dk/dv atomic-order nondeterministic too (cross only; documented in features.py). | self-repro: dk/dv max|diff| 0.0; dq 1.2e-4 |

## Correctness gates (all green)

- `tests/test_bwd.py` fast battery (5 shapes incl. ragged 960/3000 and cross
  512/257, both exercising the K-tail + split-M paths): PASS.
- FA3-fwd hybrid (our bwd on FA3's o/lse): PASS at <= 2x FA3's own error.
- autograd chain `wan_flash_attn` vs fp32 SDPA: PASS (grad rel_l2 < 6e-3).
- `WAN_FLASH_SLOW_TESTS=1` true shapes self h12 32760 + 75600: PASS.
- Errors vs the chunked-fp32 oracle: rel_l2 ~2.3e-3 on dq/dk/dv (bf16 floor,
  same as FA3's raw bwd on identical inputs); dk/dv agree with FA3 to ~1e-4
  (1-ulp level), dq differs only by atomic order.
- Alt configs re-verified vs oracle: tile_m=64, num_stages=1, batch=2,
  nsplit auto>1 on cross shapes.

## Known gap / notes

- **self h40 S=75600 is ~2% behind FA4 wall (3.7% on the main kernel,
  469.8 vs 452.9 ms interleaved) while ~3% ahead of FA3.** The gap appears
  only at that shape and grows with h at fixed S=75600 (h12 +0.4%, h20 +4.1%,
  h30 +6.3%, h40 +9.7% before the fixes below); power sampling showed our
  kernel drawing ~+40-60W at similar clocks there, pointing at an L2/drift
  interaction of the dQaccum reduce traffic at high wave counts rather than
  per-iteration instruction count. The KV tx piggyback + dropping the
  mask-skip branch recovered ~2/3 of the original gap; the remainder ships
  as-is.
- The dQaccum gmem element order is the dQ WGMMA accumulator fragment order;
  main kernel and postprocess must be built with the same (tile_m, num_wg,
  AtomLayoutMdQ, dQ_swapAB) — enforced by constructing both from the same
  feature set in interface.py. Same coupling for dK/dV accum under split-M
  (AtomLayoutNdKV, dKV_swapAB).
- Loop-carry footgun (same as fwd): consumer pipeline state is returned and
  reassigned from `_mma_m_block`; `dKV_accumulate` is reassigned directly in
  the range body. Neither is mutated through `self`.
