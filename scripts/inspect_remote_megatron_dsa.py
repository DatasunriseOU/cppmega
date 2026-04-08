from pathlib import Path


FILES = [
    "/mnt/data/megatron-lm/megatron/training/arguments.py",
    "/mnt/data/megatron-lm/gpt_builders.py",
    "/mnt/data/megatron-lm/model_provider.py",
    "/mnt/data/megatron-lm/pretrain_gpt.py",
    "/mnt/data/megatron-lm/megatron/core/transformer/transformer_config.py",
]

NEEDLES = [
    "apply_rope_fusion",
    "experimental_attention_variant",
    "core_transformer_config_from_args",
    "MLATransformerConfig",
    "multi_latent_attention",
]


def main() -> None:
    for file in FILES:
        p = Path(file)
        print(f"===== {file} =====")
        text = p.read_text()
        for needle in NEEDLES:
            if needle not in text:
                continue
            idx = text.index(needle)
            start = max(0, idx - 500)
            end = min(len(text), idx + 1800)
            print(text[start:end])
            print("---")


if __name__ == "__main__":
    main()
