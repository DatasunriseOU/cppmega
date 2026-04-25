# Deprecated Path Gates (2026-04-25)

Cppmega now treats old runtime paths as fail-closed. A deprecated path must not
run by accident: it raises before training unless the caller sets a long
explicit acknowledgement env var. With the acknowledgement, the path prints a
`DEPRECATED` warning to stderr.

## Runtime Gates Added

| Old path | Required acknowledgement | Replacement |
| --- | --- | --- |
| MXFP8 BF16 backward bridge: `CPPMEGA_TE_MXFP8_DGRAD_BF16=1`, `CPPMEGA_TE_MXFP8_WGRAD_BF16=1`, `CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK=1`, `NVTE_BACKWARD_OVERRIDE=dequantized/high_precision` | `CPPMEGA_I_UNDERSTAND_MXFP8_BF16_BACKWARD_BRIDGE_IS_DEPRECATED_AND_SLOW=1` | `CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER=1`, fallback disabled |
| Main-head native-Hopper CE bypass: `CPPMEGA_PREFER_NATIVE_HOPPER_CE=1` | `CPPMEGA_I_UNDERSTAND_PREFER_NATIVE_HOPPER_CE_IS_DEPRECATED=1` | `CPPMEGA_LINEAR_CE_KERNEL=auto` |
| Main-head nonfused Liger CE: `CPPMEGA_MAIN_HEAD_LIGER_NONFUSED=1` | `CPPMEGA_I_UNDERSTAND_MAIN_HEAD_LIGER_NONFUSED_IS_DEPRECATED_AND_MATERIALIZES_LOGITS=1` | `CPPMEGA_LINEAR_CE_KERNEL=auto` or `cce` |
| MTP Liger CE: `CPPMEGA_MTP_CE_KERNEL=liger` | `CPPMEGA_I_UNDERSTAND_MTP_LIGER_CE_IS_DEPRECATED=1` | `CPPMEGA_MTP_CE_KERNEL=native` |
| MTP CE disabled: `CPPMEGA_MTP_CE_KERNEL=none/off/0/false` | `CPPMEGA_I_UNDERSTAND_MTP_CE_DISABLED_IS_DEPRECATED=1` | `CPPMEGA_MTP_CE_KERNEL=native` |
| DSA gather-scatter sparse attention: `CPPMEGA_DSA_SPARSE_MODE=gather_scatter/pytorch` | `CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW=1` | `CPPMEGA_DSA_SPARSE_MODE=tilelang` |
| TileLang SparseMLA patch failure falling back to gather-scatter | `CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW=1` | Fix the TileLang/SparseMLA import/kernel failure |
| FP8 activation legacy packer: `CPPMEGA_FP8_ACTIVATION_BACKEND=legacy/triton/torch`, or CUDA FP8 activation compression with TE unavailable | `CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_IS_DEPRECATED_AND_SYNCY=1` | `CPPMEGA_FP8_ACTIVATION_BACKEND=te` |
| `cppmega.megatron.mamba3_author_spec` | `CPPMEGA_I_UNDERSTAND_MAMBA3_AUTHOR_SPEC_IS_DEPRECATED=1` | `cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec` |
| `nam56r_mamba3_te_pretrain()` old nheads=112 recipe | `CPPMEGA_I_UNDERSTAND_NAM56R_MAMBA3_TE_PRETRAIN_USES_DEPRECATED_NHEADS112=1` | `nam56r_mamba3_native_pretrain()` |
| `--use-legacy-models` through cppmega GPT builder | `CPPMEGA_I_UNDERSTAND_MEGATRON_LEGACY_GPT_MODEL_IS_DEPRECATED=1` | Megatron Core GPT with `--use-mcore-models` |

## Deprecated Shell Launchers

The old H200 wrapper scripts that target `h200_legacy` or historical sweep
contracts now source `scripts/lib/deprecated_guard.sh`. They require:

```bash
CPPMEGA_I_UNDERSTAND_LEGACY_H200_SCRIPTS_ARE_DEPRECATED=1
```

Without that env var, they exit before doing SSH/gcloud work. With it, they
print a `DEPRECATED` warning. Use the current local GB10 launcher or the
bench3/europe in-place remote-body scripts instead.

Currently gated:

- `scripts/remote_sync_h200.sh`
- `scripts/remote_smoke_h200.sh`
- `scripts/remote_smoke_h200_dsa.sh`
- `scripts/remote_smoke_h200_nam56r_mixed_a.sh`
- `scripts/remote_train_h200_nam56r_lite.sh`
- `scripts/remote_train_h200_nam56r_full.sh`
- `scripts/remote_train_h200_nam56r_noconv.sh`
- `scripts/remote_train_h200_nam56r_noconv.sh-e`
- `scripts/remote_train_h200_nam56r_noconv_nsys.sh-e`
- `scripts/remote_setup_h200.sh`
- `scripts/remote_sweep_h200_dsa_production.sh`
- `scripts/remote_smoke_h200_structure_poly.sh`
- `scripts/remote_smoke_h200_structure_ingress.sh`
- `scripts/remote_smoke_h200_ngram_hash_poly.sh`

## Policy

New compatibility bridges should use
`cppmega.megatron.deprecated_paths.require_deprecated_ack()` once they are no
longer the intended path. Do not add silent fallback from the current path to an
old path; if a fallback is genuinely needed for diagnostics, make it explicit,
count/log it, and require an acknowledgement env var.
