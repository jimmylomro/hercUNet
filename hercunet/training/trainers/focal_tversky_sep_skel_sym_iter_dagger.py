"""ONLINE-DAgger iterative refiner trainer — extends the _Iter trainer with a self-conditioning ``train_step``.

The refiner ``D(CT, prev) -> surface`` must repair its OWN output at inference (prev = its previous-pass softmax),
but the _Iter trainer only ever feeds a LABEL-derived prev (complete sheets + octant/fragment corruption) — so it
never learns to un-thin / re-connect its own incomplete predictions (memory gradphi-iterative-refiner
"RECALIBRATION"). Online DAgger fixes the train/inference prev mismatch by construction:

  Each train step, with prob ``dagger_prob``: run K in [1, ``dagger_maxk``] warm-up forwards under ``no_grad``,
  feeding the model's OWN surface-prob output back as ``prev`` each time (exactly the inference recurrence), then
  ONE graded forward on that self-generated prev, loss on it. K=0 (prob 1-``dagger_prob``) is a normal supervised
  step (keeps the cold/bootstrap + label-refine maps sharp).

Why this shape:
  * Warm-up passes are ``no_grad`` → hold NO activations → step memory ≈ the single-pass run (fits the 5090), and
    K>1 teaches repair at DEEPER iteration depths for only K extra cheap forwards (no BPTT — prev is detached,
    matching "apply the loss on the second pass").
  * ``prev0`` (the warm-up's starting prev) still comes from ComposePrevTransform (bootstrap / octant "+"-seam /
    fragment), so warm-up starts from DIVERSE states (cold, label, cut) → the self-outputs it must repair are
    varied. Octant seams are ALSO what teach neighbour-context healing (the ½-window shift is inference-time
    SPATIAL; in training each sample is one window). DAgger adds SELF-repair — complementary, keep both on.
  * Network stays in train mode during warm-up: nnU-Net ResEncUNet uses InstanceNorm (no running stats) and no
    dropout, so train vs eval forward is identical here — no toggling needed.

AMP / grad-scaler / grad-clip(12) / DS-list loss handling copied VERBATIM from nnUNetTrainer.train_step.
``validation_step`` is inherited unchanged (single pass on the deterministic composed prev — a stable proxy; the
real self-repair evaluation is inference).

Knobs (``dagger_prob`` / ``dagger_maxk``, and the ``prev_*`` composition params) come from the training recipe
(:mod:`hercunet.training.recipe`); the run301 defaults are ``dagger_prob=0.5``, ``dagger_maxk=2``.
"""
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from hercunet.training.trainers.focal_tversky_sep_skel_sym_iter import (
    nnUNetTrainer_FocalTverskySepSkelSym_Iter_500epochs)

__all__ = ["nnUNetTrainer_FocalTverskySepSkelSym_IterDagger_500epochs"]


class nnUNetTrainer_FocalTverskySepSkelSym_IterDagger_500epochs(
        nnUNetTrainer_FocalTverskySepSkelSym_Iter_500epochs):

    def _dagger_params(self):
        from hercunet.training.recipe import active
        r = getattr(self, "recipe", None) or active()        # AffinityMalis sets self.recipe; else the active recipe
        return r.dagger_prob, r.dagger_maxk

    def _ac(self):
        return autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # --- online DAgger: with prob p, replace prev by the model's own output after K no_grad self-passes ---
        p_dagger, maxk = self._dagger_params()
        if maxk > 0 and float(torch.rand(1).item()) < p_dagger:
            K = int(torch.randint(1, maxk + 1, (1,)).item())
            ct, prev = data[:, 0:1], data[:, 1:2]
            with torch.no_grad(), self._ac():
                for _ in range(K):
                    out = self.network(torch.cat([ct, prev], dim=1))
                    logits = out[0] if isinstance(out, (list, tuple)) else out
                    prev = torch.softmax(logits.float(), dim=1)[:, 1:2].to(data.dtype)
            data = torch.cat([ct, prev.detach()], dim=1)               # graded pass conditions on OWN output

        with self._ac():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
