"""Adapt a single-channel (m7) nnU-Net checkpoint to the 2-channel iterative refiner (input = [CT, prev]).

nnU-Net's ``-pretrained_weights`` loader shape-matches and SKIPS the first conv when in-channels differ (1 vs 2)
→ the stem would be randomly initialised, throwing away the m7 warm-start where it matters most. Instead we
expand the checkpoint ONCE on disk: pad the stem conv weight ``[out,1,k,k,k] -> [out,2,k,k,k]`` with channel 0 =
m7's CT weights and channel 1 = zeros, so the first forward is IDENTICAL to m7 and training learns to use
``prev``. Then ``-pretrained_weights <expanded>.pth`` matches and loads fully.

Pure torch; no nnunet import. Handles both a raw ``state_dict`` and the nnU-Net checkpoint wrapper
(``{"network_weights": ...}``).
"""
from __future__ import annotations

import torch


def find_input_conv_keys(state_dict, in_channels=1):
    """Return ALL stem input-conv weight keys — every 5-D conv weight whose in-channel dim equals
    ``in_channels``. In a ResEnc-UNet only the input stem has ``in_channels`` inputs (every other conv sees
    >= base features), so this is exactly the stem conv(s). Crucially there can be SEVERAL aliased keys for the
    same stem conv (e.g. dynamic_network_architectures saves both ``convs.0.conv.weight`` AND
    ``convs.0.all_modules.0.weight``); ALL must be expanded or nnU-Net's shape-check fails on the one it loads by."""
    hits = [k for k, v in state_dict.items()
            if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 5 and v.shape[1] == in_channels]
    if not hits:
        raise KeyError(f"no 5-D conv weight with in_channels={in_channels} found (keys like *.stem*.weight)")
    return sorted(hits, key=lambda k: list(state_dict).index(k))


def find_input_conv_key(state_dict, in_channels=1):
    """Back-compat: first stem input-conv key (prefer :func:`find_input_conv_keys`, which returns all)."""
    return find_input_conv_keys(state_dict, in_channels)[0]


def expand_first_conv(state_dict, new_in=2, old_in=1, init="zero"):
    """In place: expand EVERY stem input conv from ``old_in`` to ``new_in`` input channels (all aliased keys).
    channels 0..old_in-1 = the original weights; the new channel(s) are ``zero`` (default; first forward ==
    original) or ``copy`` (copy of channel 0, rescaled). Returns the list of modified keys."""
    if init not in ("zero", "copy"):
        raise ValueError(f"init must be 'zero' or 'copy', got {init!r}")
    keys = find_input_conv_keys(state_dict, old_in)
    for key in keys:
        w = state_dict[key]
        new_w = torch.zeros((w.shape[0], new_in, *w.shape[2:]), dtype=w.dtype, device=w.device)
        new_w[:, :old_in] = w
        if init == "copy":
            scale = old_in / float(new_in)
            for c in range(old_in, new_in):
                new_w[:, c] = w[:, c % old_in] * scale
            new_w[:, :old_in] = w * scale
        state_dict[key] = new_w
    return keys


def expand_checkpoint(in_path, out_path, new_in=2, old_in=1, init="zero"):
    """Load an nnU-Net checkpoint (or raw state_dict), expand its stem conv to ``new_in`` channels, save to
    ``out_path`` preserving the ``network_weights`` wrapper. Returns the expanded conv key."""
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "network_weights" in ckpt:
        sd = ckpt["network_weights"]
        keys = expand_first_conv(sd, new_in=new_in, old_in=old_in, init=init)
        ckpt["network_weights"] = sd
    else:
        keys = expand_first_conv(ckpt, new_in=new_in, old_in=old_in, init=init)
    torch.save(ckpt, out_path)
    print(f"[expand-ckpt] {in_path} -> {out_path}: {old_in}->{new_in} ch (init={init}); stems: {keys}", flush=True)
    return keys
