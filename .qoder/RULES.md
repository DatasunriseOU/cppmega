# RULE: NEVER DISABLE FEATURES TO WORKAROUND BUGS

If something is broken, FIX IT until it works. NEVER disable a feature flag,
env var, or code path as a workaround. The correct response to a bug is to
fix the bug, not to turn off the feature that exposes it.

This applies to:
- CPPMEGA_GRAPH_ROUTES_ENABLED
- CPPMEGA_DSA_PATCH_ENABLED
- CPPMEGA_STRUCTURE_ENABLED
- Any other feature flag or code path

If the converter doesn't produce the right output, FIX THE CONVERTER.
If the patch expects something the data doesn't have, FIX THE DATA PIPELINE.
Never set a flag to "0" to make an error go away.
