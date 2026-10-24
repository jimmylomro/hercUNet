"""Shared warm-start checkpoint resolution (local path or HuggingFace download) — used by the nnU-Net
refiner (``hercunet train fit``) and HercuViT training."""
from __future__ import annotations


def resolve_pretrained(pretrained=None, pretrained_hf=None, hf_file=None):
    """Return a LOCAL checkpoint path for the warm-start. ``pretrained`` is a local ``.pth``; ``pretrained_hf``
    is a HuggingFace repo id to download the checkpoint from (default filename auto-resolved, or ``hf_file``).
    The download uses the shared HF cache, so pulling the m7 model right after ``preprocess`` (which already
    ``snapshot_download``-ed the same repo for its plans) is a cache hit. Exactly one source may be given."""
    if pretrained and pretrained_hf:
        raise SystemExit("hercunet train fit: pass --pretrained (a local .pth) OR --pretrained-hf (download), "
                         "not both.")
    if not pretrained_hf:
        return pretrained
    from huggingface_hub import hf_hub_download, list_repo_files
    if hf_file:
        return hf_hub_download(pretrained_hf, hf_file)
    files = list_repo_files(pretrained_hf)
    for want in ("checkpoint_best.pth", "checkpoint_final.pth"):     # m7 warm-start prefers checkpoint_best
        hit = sorted(f for f in files if f.rsplit("/", 1)[-1] == want)
        if hit:
            print(f"[fit] warm-start: downloading {hit[0]} from HF {pretrained_hf}", flush=True)
            return hf_hub_download(pretrained_hf, hit[0])
    pth = sorted(f for f in files if f.endswith(".pth"))
    if not pth:
        raise SystemExit(f"hercunet train fit: no .pth checkpoint found in HF repo {pretrained_hf}; "
                         f"pass --pretrained-hf-file <name>.")
    print(f"[fit] warm-start: downloading {pth[0]} from HF {pretrained_hf}", flush=True)
    return hf_hub_download(pretrained_hf, pth[0])
