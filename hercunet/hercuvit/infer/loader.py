"""Load a trained HercuViT for inference.

A HercuViT checkpoint (written by :class:`hercunet.hercuvit.train.trainer.HercuvitTrainer`) stores the trained
weights **and** the Dinovol ``model_cfg``, so the architecture is rebuilt from the cfg (no Dinovol checkpoint
needed at inference — :func:`hercunet.hercuvit.netbuild.build_from_cfg`) and the trained weights loaded over it.
``--model`` is a local folder or ``.pth``; ``--model-hf`` snapshot-downloads a repo (reusing the same mechanism
as ``hercunet infer``).

Returns ``(net, recipe, model_cfg)`` — the HercuViT analogue of ``load_affinity_net``'s tuple (HercuViT is not an
nnU-Net, so there is no plans/label manager; the recipe carries the working scale and class count instead).
"""
from __future__ import annotations

import os
from dataclasses import replace

import torch

from .. import netbuild
from ..train.recipe import HERCUVIT, _TUPLE_FIELDS
from ..train.recipe import HercuvitRecipe  # noqa: F401 (re-export for callers)


def _recipe_from_dict(d: dict) -> "HercuvitRecipe":
    """Rebuild a :class:`HercuvitRecipe` from a saved ``asdict`` (coercing the tuple fields back from lists)."""
    from dataclasses import fields
    valid = {f.name for f in fields(HercuvitRecipe)}
    over = {k: (tuple(v) if k in _TUPLE_FIELDS and v is not None else v)
            for k, v in (d or {}).items() if k in valid}
    return replace(HERCUVIT, **over)


def _resolve_ckpt(model=None, model_hf=None, ckpt="checkpoint_best.pth") -> str:
    """Return a local path to a HercuViT ``.pth``. ``model`` = a ``.pth`` or a folder containing one; ``model_hf`` =
    a repo to snapshot-download. Prefers ``checkpoint_best.pth`` then ``checkpoint_final.pth`` then any ``.pth``."""
    if model and model_hf:
        raise SystemExit("hercuvit infer: pass --model OR --model-hf, not both.")
    root = model
    if model_hf:
        from huggingface_hub import snapshot_download
        print(f"[hercuvit] downloading model from HF {model_hf} …", flush=True)
        root = snapshot_download(model_hf)
    if not root:
        raise SystemExit("hercuvit infer: give a model — --model <dir|.pth> or --model-hf <repo>.")
    if os.path.isfile(root):
        return root
    if not os.path.isdir(root):
        raise SystemExit(f"hercuvit infer: --model {root} is neither a file nor a directory.")
    for name in (ckpt, "checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"):
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    pth = sorted(f for f in os.listdir(root) if f.endswith(".pth"))
    if not pth:
        raise SystemExit(f"hercuvit infer: no .pth checkpoint found under {root}.")
    return os.path.join(root, pth[0])


def load_hercuvit(model=None, *, model_hf=None, ckpt="checkpoint_best.pth", device="cuda",
                  attention_backend=None, with_instance=False):
    """Build the trained HercuViT net and return ``(net, recipe, model_cfg)``.

    ``attention_backend`` overrides the recipe's stored backend (default: the recipe value; ``auto`` is safe on
    any GPU). ``with_instance`` builds the instance head too (default False — v1 output is the semantic surface).
    """
    path = _resolve_ckpt(model, model_hf, ckpt)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    recipe = _recipe_from_dict(ck.get("recipe", {}))
    model_cfg = ck["model_cfg"]
    backend = attention_backend or recipe.attention_backend

    net = netbuild.build_from_cfg(
        model_cfg, device=device, attention_backend=backend, arch_override=ck.get("arch_override") or {},
        global_blocks=tuple(ck.get("global_blocks", recipe.global_blocks)),
        window_size=tuple(ck.get("window_size", recipe.window_size)),
        n_sem_classes=recipe.n_sem_classes, mask_dim=recipe.mask_dim, n_queries=recipe.n_queries,
        query_dim=recipe.query_dim, instance_layers=recipe.instance_layers,
        deep_supervision=recipe.deep_supervision, with_instance=with_instance,
        # the query-head architecture must match what was trained, or the state_dict mismatches (§13).
        add_null_query=recipe.add_null_query, learned_queries=recipe.learned_queries,
        fps_queries=recipe.fps_queries, cross_attn=recipe.cross_attn,
    )
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    # When loading a full (instance) checkpoint into a semantic-only net, the instance_head.* keys are expected to
    # be "unexpected" — that is fine. A MISSING non-instance key is a real error.
    missing = [m for m in missing if not m.startswith("instance_head.")]
    unexpected = [u for u in unexpected if not u.startswith("instance_head.")]
    if missing or unexpected:
        raise RuntimeError(f"hercuvit load not clean: missing={missing[:5]} unexpected={unexpected[:5]}")
    net.eval().to(device)
    print(f"[hercuvit] loaded {path} (backend={backend}, with_instance={with_instance})", flush=True)
    return net, recipe, model_cfg
