# FEATURES — kernel A/B verdicts (fwd + bwd)

All numbers: H100 SXM (132 SMs), bf16, head_dim 128, `triton.testing.do_bench`
(warmup 200 ms, rep 1000 ms), TFLOP/s = `4*b*h*sq*skv*d / t`. One clean run of
`python -m bench.bench --impl fa3 fa4 wan --shapes all` (2026-07-29, defaults =
persistent + rescale_skip 8.0 + overlap + 128x128 + 2 stages):

| shape | FA3 (C++) | FA4 (CuTeDSL) | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|---|---|
| self h12 S=32760  | 9.673 ms / 681.7  | 10.045 ms / 656.4  | 9.954 ms / 662.5  | 0.972x | 1.009x |
| self h12 S=75600  | 51.133 ms / 686.7 | 52.869 ms / 664.2  | 52.654 ms / 666.9 | 0.971x | 1.004x |
| self h40 S=32760  | 31.964 ms / 687.6 | 33.233 ms / 661.4  | 33.236 ms / 661.3 | 0.962x | 1.000x |
| self h40 S=75600  | 172.100 ms / 680.1 | 176.299 ms / 663.9 | 178.111 ms / 657.2 | 0.966x | 0.990x |
| cross h12 32760x512 | 0.190 ms / 541.5 | 0.223 ms / 462.3 | 0.193 ms / 532.8 | 0.984x | **1.153x** |
| cross h12 75600x512 | 0.445 ms / 534.5 | 0.521 ms / 456.4 | 0.443 ms / 536.3 | **1.003x** | **1.175x** |
| cross h40 32760x512 | 0.640 ms / 536.8 | 0.762 ms / 451.0 | 0.644 ms / 533.5 | 0.994x | **1.183x** |
| cross h40 75600x512 | 1.513 ms / 524.0 | 1.751 ms / 452.7 | 1.496 ms / 529.8 | **1.011x** | **1.170x** |

Run-to-run noise ~±0.5-1%; self-attn h12 wan numbers ranged 660.6-664.4
(32760) and 658.7-667.9 (75600) across sessions.

## Verdict table (defaults in wan_flash/features.py)

| feature | default | verdict | numbers (h12, TFLOP/s) |
|---|---|---|---|
| `intra_wg_overlap` | **True** | KEEP. Single biggest win. QK(i+1)+PV(i) in flight while softmax(i+1) runs (`wait_group(1)`), plus consumer-WG ping-pong named barriers. | self 32760: 562.4 → 641.3 (+14%); self 75600: 570.9 → 645.6 (+13%) |
| `rescale_skip_threshold` | **8.0** | KEEP. Port of FA4's sm100-only trick to sm90 (not in FA4 sm90): if the running max moved < 8 log2-units, keep the stale max ⇒ acc_scale = 1.0, and skip the whole O-rescale FMA loop when the warp agrees (`vote.any`). Mathematically exact (max-shift invariance); P ≤ 2^8, far inside bf16/fp32 range. | on single sched: 638.8 → 652.0 (+2.1%), 645.2 → 656.7 (+1.8%). on persistent: 652.1 → 663.8 (+1.8%), 664.7 → 666.8 (+0.3%) |
| `scheduler` | **"persistent"** | KEEP. min(tiles, 132) CTAs; per-CTA loop over (head, m_block); separate sO buffer + BAR_O_FREE handoff so the next tile's mainloop overlaps the previous tile's O TMA drain; Q pipeline cycled per tile. | self 32760: 652.0 → 664.4 (+1.9%); self 75600: ~flat; **cross 32760: 464.5 → 532.0 (+14.5%); cross 75600: 464.2 → 538.6 (+16%)** — cross-attn is prologue-bound (4 KV blocks), exactly as predicted in docs/SPECIALIZATION.md F1 |
| `tile_n` | **128** | KEEP 128. 120 (tail-free for both Wan S: 273/630 exact blocks; PV-K padded 120→128 with zeroed V rows + P slots) loses at 32760 and only edges out at 75600. 144/160/176/192 all lose. | on persistent+skip base: 120 → 634.6 (−4.4%) @32760, 672.3 (+0.8%) @75600. 144/160/176/192 @ single+skip: 637.7/638.2/642.1/645.8 vs 654.9 |
| `num_stages` | **2** | KEEP 2. 3 is noise-level (+0.3%) and burns the last smem headroom (224 KB), incompatible with the persistent sO buffer (needs 192 KB total). | single+skip: 657.0 vs 654.9 @32760; 660.5 vs 658.1 @75600 |
| `mma_pv_is_rs` | True | Only RS implemented (P never leaves registers; FA4 default at hd128). SS path not built — FA4's own data says RS wins at 128×128. | — |
| fmax reduce | 4-wide tree | KEEP. `TensorSSA.reduce(MAX)` emits a serial FMAX chain; hand 4-wide tree (FA3/FA4 idiom) gives ILP 4. | self 75600: 658.7 → 667.9 (+1.4%); 32760 flat |
| mainloop `unroll` | 1 | KEEP 1. unroll=2 helped 32760 (+0.8%) but catastrophically regressed 75600 (661 → 440, reproducible; pathological codegen). Not robust. | — |

## Correctness gates (all green)

- `tests/test_fwd.py` fast battery (5 shapes incl. ragged 960/3000, cross 512/257): PASS.
- `WAN_FLASH_SLOW_TESTS=1` true shapes self h12 32760 + 75600: PASS.
- Feature matrix re-verified vs the chunked-fp32 oracle for: tile_n ∈ {120, 128, 144, 160, 176, 192}, rescale_skip ∈ {0, 8}, scheduler ∈ {single, persistent}, overlap ∈ {on, off} — rel_l2 ≈ 2.2e-3 (bf16 floor, same as FA3), LSE max err ≤ 1.5e-6.
- Output contract: o bf16 (b,s,h,d); lse fp32 (b,h,s), natural log,
  `lse = (row_max*scale_log2 + log2(row_sum)) * ln2`, m-tail predicated.

## Notes / non-defaults worth knowing

- `tile_n=120` at S=75600 with skip is the only config that beat the default
  (672.3 vs 666.8); a per-shape override table could pick it up, but +0.8%
  was judged not worth the per-shape special case yet.
- A 2-CTA cluster with K/V TMA multicast (FA4 sm90 leaves it unused) is the
  main unexploited lever left; K/V is largely L2-resident at these shapes so
  the expected win is small.
- The DSL loop-transform gotcha that cost the most debugging time: objects
  mutated only inside a `self.`-method called from a `cutlass.range` body are
  NOT loop-carried (the region analyzer tracks assignments and direct method
  receivers, and explicitly skips `self`). Pipeline states must be returned
  and reassigned in the loop body (`q_state, kv_state = self._consumer_tile(...)`).

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
