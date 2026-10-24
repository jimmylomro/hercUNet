"""``NnunetJacobiModel`` — the production nnU-Net / Jacobi surface model as an :class:`~hercunet.infer.model.InferenceModel`.

This wraps the EXACT behaviour the engine used to hard-code: an 8-ch ``[CT, prev, orient(6)]`` (or 2-ch ``[CT, prev]``
plain) nnU-Net whose orientation channels are recomputed in-forward (so TTA stays correct), double-buffered Jacobi
passes with a prev-surface feedback channel, and an optional affinity head emitted on the final pass. All the
streaming / tiling / blend / claim-queue / leader-finalise / S3 machinery lives in the engine; this class supplies
only load + per-window norm + input stack + forward + output specs.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .. import nnunet_infer as NI
from ..model import AuxSpec, InferenceModel, OutputSpec

# HercUNet v0 affinity offsets (recipe aff_offsets=(1,3,9) → z,y,x), the channel order of aff_head. Recorded in the
# affinity zarr attrs; the band solve maps channel c → (dz,dy,dx) through this. (Moved here from jacobi_refine.)
V0_AFF_OFFSETS = [(1, 0, 0), (3, 0, 0), (9, 0, 0),
                  (0, 1, 0), (0, 3, 0), (0, 9, 0),
                  (0, 0, 1), (0, 0, 3), (0, 0, 9)]


def _resolve_model_dir(model=None, model_hf=None) -> str:
    """A LOCAL nnU-Net model folder (plans.json + fold_0/checkpoint_*.pth). ``model`` local dir OR ``model_hf`` repo."""
    if model and model_hf:
        raise SystemExit("infer: pass --model (a local folder) OR --model-hf (download), not both.")
    if model:
        if not os.path.isdir(model):
            raise SystemExit(f"infer: --model {model} is not a directory.")
        return model
    if not model_hf:
        raise SystemExit("infer: give a model — --model <folder> or --model-hf <repo>.")
    from huggingface_hub import snapshot_download
    print(f"[infer] downloading model from HF {model_hf} …", flush=True)
    path = snapshot_download(model_hf)
    print(f"[infer] model at {path}", flush=True)
    return path


class NnunetJacobiModel(InferenceModel):
    name = "nnunet-jacobi"

    def __init__(self, net, p_size, ctnorm, *, affinity, n_aff, keep_affinity, passes, overlap):
        self.net = net
        self._p = int(p_size)
        self.ctnorm = ctnorm                       # (lo, hi, mean, std) percentile fingerprint
        self.affinity = bool(affinity)             # 8-ch model (orient recomputed in-forward)
        self.n_aff = n_aff                         # affinity-head channel count, or None
        self.keep_affinity = bool(keep_affinity)   # emit the affinity zarr on the final pass
        self._passes = int(passes)
        self._overlap = float(overlap)

    # --- construction --------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, *, model=None, model_hf=None, ckpt="checkpoint_best.pth", device="cuda",
             affinity=True, keep_affinity=False, n_orient=6, passes=3, overlap=0.25, compile=False, **_):
        model_dir = _resolve_model_dir(model, model_hf)
        n_aff = None
        if affinity and keep_affinity:
            net, cfg, _lm, _pm, _dsj, num_in, n_aff = NI.load_affinity_net_with_head(model_dir, ckpt, device, n_orient=n_orient)
            assert num_in == 2 + n_orient, f"expected {2 + n_orient}-ch affinity model, got {num_in}"
        elif affinity:
            net, cfg, _lm, _pm, _dsj, num_in = NI.load_affinity_net(model_dir, ckpt, device, n_orient=n_orient)
            assert num_in == 2 + n_orient, f"expected {2 + n_orient}-ch affinity model, got {num_in}"
        else:
            if keep_affinity:
                raise SystemExit("infer --keep-affinity: needs the affinity model, not --plain ([CT, prev]).")
            net, cfg, _lm, _pm, _dsj, num_in = NI.load_net(model_dir, ckpt, device, deep_supervision=False, force_in_channels=2)
            assert num_in == 2, f"expected a 2-channel [CT, prev] model, got num_input_channels={num_in}"
        if compile:
            import torch
            print("[infer] torch.compile(net, max-autotune-no-cudagraphs) — one-time warmup on first windows …", flush=True)
            net = torch.compile(net, mode="max-autotune-no-cudagraphs")
        P = int(cfg.patch_size[0])
        assert list(cfg.patch_size) == [P, P, P], cfg.patch_size
        if keep_affinity:
            assert len(V0_AFF_OFFSETS) == n_aff, f"offset table ({len(V0_AFF_OFFSETS)}) != checkpoint n_aff ({n_aff})"
        return cls(net, P, NI.ct_norm_params(model_dir), affinity=affinity, n_aff=n_aff,
                   keep_affinity=keep_affinity, passes=passes, overlap=overlap)

    # --- static config -------------------------------------------------------------------------------------------
    @property
    def patch_size(self) -> int:
        return self._p

    @property
    def passes(self) -> int:
        return self._passes

    @property
    def uses_prev(self) -> bool:
        return self._passes > 1

    @property
    def overlap(self) -> float:
        return self._overlap

    @property
    def train_voxel_band(self):
        return (7.5, 10.0)         # HercUNet v0 native-coarse domain; warn outside (10 tolerates a 2.4µm scroll → L2)

    @property
    def aux(self) -> Optional[AuxSpec]:
        if self.keep_affinity:
            return AuxSpec(name="aff", channels=self.n_aff, offsets=V0_AFF_OFFSETS, final_pass_only=True)
        return None

    # --- per-window hooks ----------------------------------------------------------------------------------------
    def norm_ct(self, sub: np.ndarray) -> np.ndarray:
        lo, hi, mean, std = self.ctnorm
        return NI.norm_ct(sub, lo, hi, mean, std)

    def base_input(self, ct_norm: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        P = self._p
        pv = np.zeros((P, P, P), np.float32) if prev is None else prev
        return np.stack([ct_norm, pv]).astype(np.float32)        # [2,P,P,P] = [CT, prev]

    def forward(self, xb_base):
        """``[B,2,P,P,P]`` [CT,prev] → ``(seg_logits[B,C,P,P,P], aff[B,n_aff,P,P,P]|None)``. Orientation is
        recomputed here from THIS batch's CT (``xb_base[:,0:1]``) and concatenated (8-ch) so TTA is correct."""
        import torch
        if self.affinity:
            from ...losses.affinity import ct_orientation
            orient = ct_orientation(xb_base[:, 0:1], sigma_grad=1.0, sigma_tensor=3.0)
            xin = torch.cat([xb_base, orient.to(xb_base.dtype)], dim=1)
        else:
            xin = xb_base
        out = self.net(xin)
        if isinstance(out, (list, tuple)):
            if len(out) == 2:                                    # AffinityHeadNet -> (seg, aff)
                seg, aff = out
                if isinstance(seg, (list, tuple)):
                    seg = seg[0]
                return seg, aff
            return out[0], None
        return out, None
