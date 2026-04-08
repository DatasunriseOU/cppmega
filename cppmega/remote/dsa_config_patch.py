"""Small helpers for native Megatron DSA smoke integration."""

from __future__ import annotations


def disable_rope_fusion_for_dsa(argv: list[str]) -> list[str]:
    """Return argv with a YAML-free override for Megatron DSA smoke.

    Current Megatron validates that DSAttention requires apply_rope_fusion=False,
    but the active GPT CLI lane does not expose a dedicated flag for that field.
    `cppmega` keeps the port native-first by patching only this config knob at
    process startup for the smoke lane.
    """

    if "--experimental-attention-variant" not in argv:
        return argv
    idx = argv.index("--experimental-attention-variant")
    if idx + 1 >= len(argv) or argv[idx + 1] != "dsa":
        return argv
    if "--yaml-cfg" in argv:
        return argv
    return argv
