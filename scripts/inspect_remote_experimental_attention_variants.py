from pathlib import Path


def main() -> None:
    p = Path(
        "/mnt/data/megatron-lm/megatron/core/models/gpt/experimental_attention_variant_module_specs.py"
    )
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if any(
            needle in line
            for needle in (
                "_get_backend_spec_provider",
                "TESpecProvider",
                "KitchenSpecProvider",
                "HAVE_TE",
                "HAVE_KITCHEN",
            )
        ):
            start = max(1, i - 10)
            end = min(len(lines), i + 25)
            print(f"--- {start}:{end} ---")
            for j in range(start, end + 1):
                print(f"{j}: {lines[j - 1]}")


if __name__ == "__main__":
    main()
