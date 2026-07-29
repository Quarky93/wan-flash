# FEATURES — forward kernel A/B verdicts

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
