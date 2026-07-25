"""TE <-> FA4 parity test v3: correct FA4 beta23 score_mod with vector extraction."""
import math, sys, json, traceback
import torch

B, S, H, D = 2, 128, 8, 64
BETA = 2.0
NUM_CHUNKS = 4
CHUNK_SIZE = S // NUM_CHUNKS
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
PARITY_THRESHOLD = 0.1
MAX_RARE = 1  # no rare edges in this test

results = {}
def report(key, val):
    results[key] = val
    print(f"  {key}: {val}")

print(f"Config: B={B}, S={S}, H={H}, D={D}, beta={BETA}, chunks={NUM_CHUNKS}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

# Build chunk structure
call_edges = [(0, 2), (1, 3), (0, 1)]
chunk_bias = torch.zeros(B, NUM_CHUNKS + 1, NUM_CHUNKS + 1, device=DEVICE, dtype=torch.float32)
for b in range(B):
    for (src, dst) in call_edges:
        chunk_bias[b, src, dst] += BETA

token_to_chunk = torch.zeros(B, S, dtype=torch.int32, device=DEVICE)
for b in range(B):
    for c in range(NUM_CHUNKS):
        token_to_chunk[b, c * CHUNK_SIZE:(c + 1) * CHUNK_SIZE] = c

# Dense bias [B, 1, S, S]
dense_bias = torch.zeros(B, 1, S, S, device=DEVICE, dtype=torch.float32)
for b in range(B):
    for qi in range(S):
        qc = int(token_to_chunk[b, qi].item())
        for ki in range(S):
            kc = int(token_to_chunk[b, ki].item())
            dense_bias[b, 0, qi, ki] = chunk_bias[b, qc, kc]

print(f"Dense bias: nonzero={dense_bias.count_nonzero().item()}, max={dense_bias.max().item():.4f}")

# Q, K, V
torch.manual_seed(42)
q = torch.randn(B, S, H, D, device=DEVICE, dtype=DTYPE)
k = torch.randn(B, S, H, D, device=DEVICE, dtype=DTYPE)
v = torch.randn(B, S, H, D, device=DEVICE, dtype=DTYPE)
scale = 1.0 / math.sqrt(D)

# REFERENCE: Manual PyTorch (fp32 computation, cast back to bf16)
print("\n--- Reference: Manual PyTorch (fp32) ---")
q_t = q.transpose(1, 2).float()
k_t = k.transpose(1, 2).float()
v_t = v.transpose(1, 2).float()
scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
scores = scores + dense_bias.float()
causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=DEVICE), diagonal=1)
scores = scores + causal_mask
attn_weights = torch.softmax(scores, dim=-1)
ref_manual = torch.matmul(attn_weights, v_t).transpose(1, 2).to(DTYPE)
print(f"  norm={ref_manual.float().norm().item():.4f}")

# REFERENCE 2: TE
print("\n--- Reference: TE DotProductAttention ---")
te_out = None
te_api_used = "none"
try:
    import transformer_engine.pytorch as te
    te_attn = te.DotProductAttention(
        num_attention_heads=H, kv_channels=D, attention_dropout=0.0,
        qkv_format="bshd", attn_mask_type="causal",
    ).to(DEVICE)
    te_out = te_attn(q.clone(), k.clone(), v.clone(),
        attention_mask=None, qkv_format="bshd",
        max_seqlen_q=S, max_seqlen_kv=S,
        core_attention_bias_type="post_scale_bias",
        core_attention_bias=dense_bias)
    te_api_used = "te.DotProductAttention(post_scale_bias)"
    diff = (te_out.float() - ref_manual.float()).abs()
    print(f"  TE vs Manual: max_diff={diff.max().item():.6f}")
except ImportError:
    print("  TE not installed, using manual reference as primary")
    te_out = ref_manual
    te_api_used = "manual_pytorch_fp32 (TE not installed)"
except Exception as e:
    print(f"  TE error: {e}")
    te_out = ref_manual
    te_api_used = "manual_pytorch_fp32 (TE error)"

# TEST: FA4 with score_mod (production-matching signature)
print("\n--- Test: FA4 flash_attn_func with score_mod ---")
fa4_out = None
try:
    from flash_attn.cute.interface import flash_attn_func

    c_plus_1 = NUM_CHUNKS + 1
    chunk_bias_flat = chunk_bias.reshape(B, -1).contiguous()  # [B, (C+1)*(C+1)]
    rare_q = torch.zeros(B, MAX_RARE, dtype=torch.int32, device=DEVICE)
    rare_k = torch.zeros(B, MAX_RARE, dtype=torch.int32, device=DEVICE)
    rare_w = torch.zeros(B, MAX_RARE, dtype=torch.float32, device=DEVICE)

    aux_tensors = [token_to_chunk, token_to_chunk, chunk_bias_flat, rare_q, rare_k, rare_w]

    # Production-matching score_mod: keyword-only ABI, vector[0] extraction
    def graph_score_mod(score, batch_idx, head_idx, *, q_idx, kv_idx, seqlen_info, aux_tensors):
        token_to_chunk_q = aux_tensors[0]
        token_to_chunk_k = aux_tensors[1]
        chunk_bias_flat_t = aux_tensors[2]
        rare_q_t = aux_tensors[3]
        rare_k_t = aux_tensors[4]
        rare_w_t = aux_tensors[5]

        # Extract scalars from vector<1xi32> (CuTe DSL requirement)
        b = batch_idx[0]
        qi = q_idx[0]
        ki = kv_idx[0]

        # Chunk-pair gather via flat indexing
        qc = token_to_chunk_q[b, qi]
        kc = token_to_chunk_k[b, ki]
        flat_idx = qc * c_plus_1 + kc
        bias_val = chunk_bias_flat_t[b, flat_idx]
        out = score + bias_val

        # Rare token-edge overlay: bounded scan
        for i in range(MAX_RARE):
            q_match = rare_q_t[b, i] == qi
            k_match = rare_k_t[b, i] == ki
            out = out + q_match * k_match * rare_w_t[b, i]

        return out

    fa4_out = flash_attn_func(
        q=q, k=k, v=v,
        softmax_scale=scale, causal=True,
        score_mod=graph_score_mod,
        aux_tensors=aux_tensors,
    )
    if isinstance(fa4_out, tuple):
        fa4_out = fa4_out[0]
    print(f"  FA4 norm={fa4_out.float().norm().item():.4f}")
except Exception as e:
    print(f"  FA4 FAILED: {e}")
    traceback.print_exc()

# COMPARISON
print("\n" + "=" * 60)
print("PARITY RESULTS")
print("=" * 60)
report("te_api_used", te_api_used)

if fa4_out is not None:
    diff_fa4_ref = (fa4_out.float() - ref_manual.float()).abs()
    max_diff = diff_fa4_ref.max().item()
    mean_diff = diff_fa4_ref.mean().item()
    report("fa4_vs_ref_max_diff", f"{max_diff:.6f}")
    report("fa4_vs_ref_mean_diff", f"{mean_diff:.6f}")
    report("threshold", PARITY_THRESHOLD)
    report("max_diff", f"{max_diff:.6f}")
    report("mean_diff", f"{mean_diff:.6f}")
    passed = max_diff < PARITY_THRESHOLD
    report("verdict", "PASS" if passed else "FAIL")
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"  max_diff={max_diff:.6f} (threshold={PARITY_THRESHOLD})")
    print(f"  mean_diff={mean_diff:.6f}")
else:
    report("verdict", "FAIL (FA4 did not produce output)")
    print("  VERDICT: FAIL (FA4 did not produce output)")

with open("/home/dave/parity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to ~/parity_results.json")
