**Confirmed Bugs**
**Critical**
1. Megatron domain/structure sidecar embeddings are gradient-dead at init.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/features/domain/embedding.py:56`, `:57` and `/Volumes/external/sources/cppmega/cppmega/features/structure/embedding.py:68`, `:69` zero both the table and projection, then `/Volumes/external/sources/cppmega/cppmega/megatron/custom_embedding.py:185` and `:204` add those residuals. The existing test masks this by manually filling projection at `/Volumes/external/sources/cppmega/tests/test_structure_embedding.py:20`. Failure: CE backprop gives zero grad to both factors, so sidecars never learn. Test/fix: one-step grad test for default init; keep residual output zero by zeroing only table and leaving projection nonzero, matching MLX `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/structure_embedding.py:73`.

**High**
2. Generation defaults enable graph-routed dense attention but provide no graph tensors.  
Evidence: `/Volumes/external/sources/cppmega/scripts/nebius_h200_megatron_cpp_generation_eval.py:641` enables graph/dense bias by default, while `:391` installs only zero scalar structure fields and `:530` repeats that before decode. Graph bias requires route tensors at `/Volumes/external/sources/cppmega/cppmega/megatron/graph_route_attention_bias_patch.py:60`, `:233`. Failure: graph-routed checkpoints can crash before logits in default generation. Test/fix: default graph routes off for prompt-only generation, or forward explicit zero-count `graph_*` tensors and test the worker path.

3. Non-absorbed Megatron sparse DSA gathers invalid top-k indices directly.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/megatron/dsa_sparse_attention.py:92`, `:103` gather with raw `topk_indices`; the absorbed path clamps/masks invalid indices at `/Volumes/external/sources/cppmega/cppmega/megatron/dsa_sparse_absorbed.py:60`. Failure: `-1`/out-of-range sentinels can crash or corrupt sparse selection. Test/fix: tiny `topk_indices=[-1,0]` and `[sk,0]` tests; copy absorbed invalid-mask/clamp behavior.

4. Graph sidecar edges are silently truncated.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/megatron/structure_dataset_patch.py:410`, `:418` cap rows; graph tensors use this at `:518`. Live launch caps are `/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/container_run.sh:12`, `:13`. Failure: high-degree graph routes are dropped with no receipt, so attention paths and gradients use partial graph evidence. Test/fix: overflow fixture with cap 1 and two edges; fail closed or emit dropped-edge counters.

5. FastMTP reintroduces unsafe Liger FLCE `reduction="none"`.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/megatron/fastmtp_layer.py:93` defaults Liger on and `:108` calls FLCE with `"none"`; `/Volumes/external/sources/cppmega/cppmega/megatron/mtp_liger_ce.py:157` warns that mode can corrupt gradients. Failure: FastMTP can bypass the safer patched MTP loss. Test/fix: FastMTP Liger/reference gradient parity under masked loss; route through safe scalar reduction.

**Medium**
6. Structure IDs are silently clamped, unlike domain IDs.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/features/structure/embedding.py:113` clamps; domain rejects invalid IDs at `/Volumes/external/sources/cppmega/cppmega/features/domain/embedding.py:91`. Failure: corrupt sidecars train boundary buckets. Test/fix: negative/over-max tests per component; raise like domain.

7. Structure-enabled Megatron can silently skip missing sidecars.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/features/structure/embedding.py:100` returns a scalar when no inputs exist; `/Volumes/external/sources/cppmega/cppmega/megatron/custom_embedding.py:214` only adds if ndim matches. Failure: structure-enabled training can become token-only. Test/fix: fail closed unless an explicit allow-missing flag is set.

8. DSA fused indexer ignores caller masks.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/megatron/dsa_indexer_fused_patch.py:488` accepts kwargs but ignores `mask`; `/Volumes/external/sources/cppmega/cppmega/megatron/index_cache_patch.py:148` recomputes with `mask=None`. Failure: padded/masked keys can occupy top-k slots. Test/fix: masked high-score key must never be selected; apply mask before top-k or raise on unsupported mask.

9. `Mamba3NoConvMixer` drops advertised trap/RoPE inputs.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/megatron/noconv_mamba_mixer.py:807` parses `trap`/`angles`, but `:842` passes `trap=None, angles=None` into scan. Failure: that variant is not testing full Mamba3 behavior. Test/fix: output must change when trap/angles change; pass them through or rename as disabled variant.

**Design Gaps**
**High**
10. MLX graph sidecars do not reach normal model/loss calls.  
Evidence: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/megatron_indexed.py:395` can build graph packets, but `:457` returns ordinary LM batches; `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/batch.py:163` forwards only structure/platform. The model consumes graph bias only if `block_bias` is supplied at `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/models/dense_cpp_lm.py:347`. Failure: sidecars can load but not affect logits, attention, or gradients. Test/fix: end-to-end sidecar batch where changing graph edges changes loss and graph/indexer grads.

11. MLX domain sidecars are config-visible but not wired.  
Evidence: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/megatron_indexed.py:25` side-channel keys omit token domain/role/confidence; `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/batch.py:163` cannot forward them. Failure: domain route IDs have no default gradient path. Test/fix: add batch/model fields and one-step domain embedding grad test.

**Medium**
12. MTP objectives stop gradients into main decoder/head.  
Evidence: Megatron detaches at `/Volumes/external/sources/cppmega/cppmega/megatron/fastmtp_layer.py:307`, `:319`; MLX stops gradients at `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/mtp.py:268`, `:438`. Failure: MTP trains side modules, not backbone/head. Test/fix: gradient ownership test; config-gate if intentional.

13. IFIM is a proxy squared-logit penalty, not true IFIM/domain-conditioned loss.  
Evidence: `/Volumes/external/sources/cppmega.mlx/cppmega_v4/buildspec/api.py:226`; spec name at `/Volumes/external/sources/cppmega.mlx/cppmega_v4/buildspec/loss_spec.py:181`. Failure: configs can claim Fisher/diagnostic behavior without those inputs. Test/fix: rename proxy or implement estimator and sidecar sensitivity test.

**Low**
14. Dense Megatron graph bias is wired, but long-context memory guard is weak.  
Evidence: `/Volumes/external/sources/cppmega/cppmega/megatron/graph_route_attention_bias_patch.py:99` defaults max seq to 16384; live TE used `b1ss` bias at `/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/stage_5_seq_16384_gbs_8_mbs_2.log:1219`. Failure: O(B*S^2) fixed bias is fragile and has no trainable graph-edge gradient. Test/fix: bytes-based cap and sparse route for long context.

**Stale Docs / Artifacts**
- `/Volumes/external/sources/cppmega/outputs/megatron_ready/h200_fp8_recompute_debug_1024_train.json:7` is stale: current code requires loss/doc/structure and broader graph sidecars at `/Volumes/external/sources/cppmega/cppmega/megatron/structure_dataset_patch.py:612`, `:639`, `:255`.
- `/Volumes/external/sources/cppmega.mlx/configs/stage_domain_routed_foundation.yaml:28` claims domain side embeddings; current MLX batch/model path does not pass them.
- `/Volumes/external/sources/cppmega.mlx/configs/stage_domain_routed_foundation.yaml:67` claims diagnostic-conditioned IFIM; current implementation is the proxy above.

Read-only status: changed files none, simplifications made none, tests run none. I did not start/stop jobs or touch running data processes. Live artifact trace: the H200 graph-route run did map graph sidecars and inject dense TE attention bias, but MLX default training does not currently propagate graph sidecars into attention or gradients. Remaining risk: no runtime tests were run in this pass.  
