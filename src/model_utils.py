"""Small model-loading helpers shared by training, eval, and the agent.

The headline feature is `pick_attn_impl()`: Flash Attention 2 is great when it's
installed, but building it is slow and version-fragile, and it isn't available on
every box (Windows, no-nvcc images, consumer GPUs). Rather than hard-code
`attn_implementation="flash_attention_2"` and risk a crash on load, we detect it
and fall back to PyTorch's built-in `sdpa` (scaled-dot-product attention), which
is fast and ships with torch.

This keeps the same code runnable on an H100/A100 job (flash-attn) and on a local
3080 smoke test (sdpa) with no edits.
"""

import os


def pick_attn_impl() -> str:
    """Return the best available attention implementation.

    Override with ATTN_IMPL=flash_attention_2|sdpa|eager if you want to force one.
    Otherwise: flash_attention_2 if the `flash_attn` package imports, else sdpa.
    """
    forced = os.environ.get("ATTN_IMPL")
    if forced:
        return forced
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"
