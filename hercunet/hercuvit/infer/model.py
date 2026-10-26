"""``HercuvitSurfaceModel`` — the v2 decoupled HercuViT surface net as an :class:`~hercunet.infer.model.InferenceModel`,
so it runs on the SAME engine as the nnU-Net/Jacobi model (streaming, tiling, Gaussian blend, claim-queue, leader
finalise + S3). Single-pass (no Jacobi prev feedback), 1-channel Dinovol-normalised CT, surface-only output.

The blend is on the SURFACE (``sem_logits[:,1]-sem_logits[:,0]`` → Gaussian-blend → sigmoid). Per-sheet masks are not
blended; when mesh output is requested (``with_meshes``) the instance head is loaded on and each window's sheet
*meshes* are extracted from the SAME forward (no second encoder pass) via :func:`hercunet.meshes.extract.extract_window_sheets`
and handed to the engine per window (``emits_meshes`` / :meth:`forward_meshes`) to write into the per-worker .asv.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...infer.model import InferenceModel, OutputSpec


class HercuvitSurfaceModel(InferenceModel):
    name = "hercuvit-surface"

    def __init__(self, net, p_size, *, overlap=0.25, with_meshes=False, mesh_thresh=0.4, mesh_sigma=3.0,
                 mesh_fit_iters=40, mesh_fit_lr=1.0, mesh_fit_lam=0.3, mesh_min_pts=400, mesh_step_um=30.0,
                 mesh_mass_frac=0.10, mesh_conf_thr=0.70, mesh_vox_frac=0.001, mesh_merge_thr=1.0 / 3):
        self.net = net
        self._p = int(p_size)
        self._overlap = float(overlap)
        self.emits_meshes = bool(with_meshes)
        self._mesh = dict(thresh=float(mesh_thresh), sigma=float(mesh_sigma), iters=int(mesh_fit_iters),
                          lr=float(mesh_fit_lr), lam=float(mesh_fit_lam), min_pts=int(mesh_min_pts),
                          step_um=float(mesh_step_um), mass_frac=float(mesh_mass_frac),
                          conf_thr=float(mesh_conf_thr), vox_frac=float(mesh_vox_frac),
                          merge_thr=float(mesh_merge_thr))

    @classmethod
    def load(cls, *, model=None, model_hf=None, ckpt="checkpoint_best.pth", device="cuda",
             attention_backend=None, overlap=0.25, with_meshes=False, mesh_thresh=0.4, mesh_sigma=3.0,
             mesh_fit_iters=40, mesh_fit_lr=1.0, mesh_fit_lam=0.3, mesh_min_pts=400, mesh_step_um=30.0,
             mesh_mass_frac=0.10, mesh_conf_thr=0.70, mesh_vox_frac=0.001, mesh_merge_thr=1.0 / 3, **_):
        from .loader import load_hercuvit
        net, recipe, _cfg = load_hercuvit(model, model_hf=model_hf, ckpt=ckpt, device=device,
                                          attention_backend=attention_backend, with_instance=bool(with_meshes))
        return cls(net, int(recipe.window), overlap=overlap, with_meshes=with_meshes, mesh_thresh=mesh_thresh,
                   mesh_sigma=mesh_sigma, mesh_fit_iters=mesh_fit_iters, mesh_fit_lr=mesh_fit_lr,
                   mesh_fit_lam=mesh_fit_lam, mesh_min_pts=mesh_min_pts, mesh_step_um=mesh_step_um,
                   mesh_mass_frac=mesh_mass_frac, mesh_conf_thr=mesh_conf_thr, mesh_vox_frac=mesh_vox_frac,
                   mesh_merge_thr=mesh_merge_thr)

    @property
    def patch_size(self) -> int:
        return self._p

    @property
    def passes(self) -> int:
        return 1                              # single-pass: no Jacobi prev feedback

    @property
    def overlap(self) -> float:
        return self._overlap

    @property
    def output(self) -> OutputSpec:
        return OutputSpec(name="hercuvit-sem")

    @property
    def train_voxel_band(self):
        return (7.5, 10.0)                    # Dinovol ~9.6 µm working scale

    def norm_ct(self, sub: np.ndarray) -> np.ndarray:
        from ..data.normalize import dinovol_normalise
        return dinovol_normalise(sub)

    def base_input(self, ct_norm: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
        return ct_norm[None].astype(np.float32)   # [1,P,P,P] — single-channel; prev ignored (single-pass)

    def forward(self, xb_base):
        """``[B,1,P,P,P]`` → ``(sem_logits[B,2,P,P,P], None)`` — the decoupled surface head; no aux."""
        out = self.net(xb_base, run_instance=False)
        return out["sem_logits"][0], None

    def forward_meshes(self, xb_base, voxel_um):
        """ONE ``run_instance`` forward → ``(sem_logits[B,2,P,P,P], meshes)`` where ``meshes`` is a list of length B,
        each the window's ``{sheet_id: mesh}`` dict (verts in window-local voxel z,y,x). ``voxel_um`` is the engine's
        working-level voxel size (so the fit's 30 µm grid maps to the right pixel step). No second encoder pass."""
        import torch
        from ..model.decoder_common import mask_logits_from_embed
        from ...meshes.extract import extract_window_sheets, prune_sheet_masks, merge_confused_sheets
        out = self.net(xb_base, run_instance=True)
        sem = out["sem_logits"]                                               # deep-sup LIST: [0] = full-res
        sem = (sem[0] if isinstance(sem, (list, tuple)) else sem)            # [B,2,P,P,P]
        mf = out["mask_features"]                                             # tensor [B,C,P,P,P] (or list)
        mf = (mf[0] if isinstance(mf, (list, tuple)) else mf).float()        # [B,C,P,P,P]
        me = out["instance"]["layers"][-1]["mask_embed"].float()             # [B,Q,C]
        meshes = []
        for b in range(xb_base.shape[0]):
            ml = mask_logits_from_embed(me[b:b + 1], mf[b:b + 1])[0]                        # (Q,P,P,P)
            owner = ml.argmax(0)                                                            # (P,P,P) ids
            memb = torch.exp(ml.amax(0) - torch.logsumexp(ml, 0))                           # (P,P,P) own-membership
            gate = torch.softmax(sem[b].float(), 0)[1] > self._mesh["thresh"]              # (P,P,P) bool
            # merge over-split pairs (symmetric cross-confidence > 1/3), then prune one-sided fragments
            owner = merge_confused_sheets(owner, gate, ml, thresh=self._mesh["merge_thr"])
            gate = prune_sheet_masks(owner, gate, memb, mass_frac=self._mesh["mass_frac"],
                                     conf_thr=self._mesh["conf_thr"])
            min_pts = max(self._mesh["min_pts"], int(self._mesh["vox_frac"] * owner.numel()))
            meshes.append(extract_window_sheets(
                owner, gate, voxel_um=float(voxel_um), sigma=self._mesh["sigma"], min_pts=min_pts,
                step_um=self._mesh["step_um"], iters=self._mesh["iters"], lr=self._mesh["lr"], lam=self._mesh["lam"],
                device=owner.device.type))
        return sem, meshes
