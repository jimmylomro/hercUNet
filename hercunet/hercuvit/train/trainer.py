"""HercuViT standalone training loop (NOT nnU-Net; ``docs/hercuvit.md`` §7).

HercuViT is not an nnU-Net, so it has its own PyTorch loop (per the owner's "whole new tooling, in parallel"
directive) that reuses hercunet machinery around it: the data layer (dataset), the HF pull + warm-start pattern
(encoder loaded by ``build_hercuvit``), and the single-line ``[ep N] loss …`` logging. bf16 autocast, AdamW on
``net.param_groups``, grad-clip, warmup + constant (or cosine) LR.

Loss: PLAIN Mask2Former on THIN labels every epoch (Hungarian + per-query sigmoid mask BCE+Dice + token
objectness against each sheet's thin medial surface), with the OPTIONAL SymSkel (``w_symskel>0``) and AffMALIS
(``w_malis>0``) add-ons. Schedule: **LP-FT** (Kumar et al., ICLR 2022) — the transformer is FROZEN for the first
``freeze_warmup_epochs`` (train the head on the distilled encoder's features so the random head's early gradients
do not distort them), then fine-tuned at ``encoder_lr_mult`` × LR for the rest. The optimiser is rebuilt on the
freeze→unfreeze transition (the trainable set changes). Checkpoints save model + optimiser + epoch + the Dinovol
``model_cfg`` (so inference rebuilds the architecture without the Dinovol ckpt).
"""
from __future__ import annotations

import os
import time

import torch

from . import losses as L


# ANSI colours, matching the hercunet trainer's per-term log line.
_C = dict(tot="\033[1;97m", sem="\033[36m", inst="\033[35m", dim="\033[2m", r="\033[0m")


class HercuvitTrainer:
    def __init__(self, net, recipe, out_dir, *, model_cfg=None, device="cuda") -> None:
        self.net = net
        self.recipe = recipe
        self.out_dir = out_dir
        self.model_cfg = model_cfg
        self.device = torch.device(device)
        os.makedirs(out_dir, exist_ok=True)
        self.num_epochs = int(recipe.epochs)
        self.start_epoch = 0
        self.global_step = 0
        self._frozen = None
        self.optim = None

    # ---- LP-FT freeze-warmup ----------------------------------------------------------------------
    def _frozen_at(self, ep: int) -> bool:
        """True while the ENCODER (stem + transformer) is frozen — the first ``freeze_warmup_epochs`` (train the head
        on the encoder's features), then it is fine-tuned at ``encoder_lr_mult`` × LR."""
        return ep < int(self.recipe.freeze_warmup_epochs)

    def _set_frozen(self, frozen: bool) -> None:
        if frozen == self._frozen:
            return
        self.net.set_encoder_trainable(not frozen)              # freeze the WHOLE encoder (stem + transformer)
        self._build_optimiser()                                 # the trainable set changed → rebuild the optimiser
        self._frozen = frozen
        print(f"[hercuvit] encoder {'FROZEN (head warmup)' if frozen else 'trainable @ encoder_lr_mult × LR'}",
              flush=True)

    def _build_optimiser(self) -> None:
        groups = self.net.param_groups(base_lr=self.recipe.base_lr,
                                       encoder_lr_mult=self.recipe.encoder_lr_mult)
        for g in groups:                                        # remember the base LR to scale by the schedule
            g["base_lr"] = g["lr"]
        self.optim = torch.optim.AdamW(groups, lr=self.recipe.base_lr, weight_decay=self.recipe.weight_decay)

    # ---- LR schedule ------------------------------------------------------------------------------
    def _lr_scale(self) -> float:
        import math
        w = max(1, int(self.recipe.warmup_steps))
        if self.global_step < w:
            return (self.global_step + 1) / w
        if self.recipe.lr_schedule == "cosine":
            total = max(1, self.num_epochs * max(1, self._steps_per_epoch))
            prog = min(1.0, (self.global_step - w) / max(1, total - w))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return 1.0

    def _apply_lr(self) -> float:
        s = self._lr_scale()
        for g in self.optim.param_groups:
            g["lr"] = g["base_lr"] * s
        return self.optim.param_groups[0]["lr"]

    # ---- one epoch --------------------------------------------------------------------------------
    def _run_epoch(self, loader, epoch: int) -> None:
        self.net.train()
        acc = {k: 0.0 for k in ("loss", "class", "bce", "dice", "obj", "dn", "symskel", "malis", "surf")}
        n = 0
        amp_dtype = torch.bfloat16
        t0 = time.time()
        for bi, batch in enumerate(loader):
            ct = batch["ct"].to(self.device, non_blocking=True)
            sem = batch["sem"].to(self.device, non_blocking=True)
            owner = batch["owner"].to(self.device, non_blocking=True)
            has_id = batch["has_identity"]
            owner_thin = owner * (sem == 1).to(owner.dtype)    # THIN GT: sheet id kept ONLY on its medial surface

            lr = self._apply_lr()
            self.optim.zero_grad(set_to_none=True)
            surface_only = bool(self.recipe.surface_only)             # design-B stage 1: train ONLY the surface head
            run_instance = not surface_only
            dn_owner = owner_thin if (self.recipe.denoise and run_instance) else None  # denoise group (§13.2), train-only
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                out = self.net(ct, dn_owner=dn_owner, run_instance=run_instance)
                # SURFACE head (author-style, direct): the thin medial-surface stack (fg-weighted CE + Focal-Tversky +
                # clDice skel-recall + separation) on owner_thin's medial (sem==1), deep-supervised. w_surface>0 or
                # surface_only wires it (otherwise the surface head is unsupervised, the legacy instance-only default).
                st = None
                loss = None
                if self.recipe.w_surface > 0 or surface_only:
                    st = L.semantic_loss(out["sem_logits"], sem, self.recipe)
                    w_surf = self.recipe.w_surface if self.recipe.w_surface > 0 else 1.0
                    loss = w_surf * st["sem"]
                it = skt = mt = None
                if run_instance:
                    inst = out["instance"]
                    # PLAIN Mask2Former on THIN labels — Hungarian + per-query mask BCE+Dice (+token objectness) against
                    # each sheet's THIN medial surface (owner & sem==1), ignore-masked. SymSkel / AffMALIS are optional
                    # add-ons (w_symskel / w_malis > 0). In design B the readout uses the surface-GATED mask features.
                    it = L.instance_loss(inst, out["mask_features"], owner_thin, has_id, self.recipe,
                                         ignore=(sem == 2), seed=self.global_step)
                    loss = it["inst"] if loss is None else loss + it["inst"]
                    if self.recipe.w_symskel > 0 and inst is not None:
                        sk, skt = L.m2f_symskel_mask_loss(inst, out["mask_features"], sem, owner, has_id, self.recipe,
                                                          seed=self.global_step, use_softmax=False)
                        loss = loss + self.recipe.w_symskel * sk
                    if self.recipe.w_malis > 0 and inst is not None:
                        ml, mt = L.affinity_malis_loss(inst, out["mask_features"], owner, has_id, self.recipe)
                        loss = loss + self.recipe.w_malis * ml
            loss_f = loss.float()
            loss_f.backward()
            if self.recipe.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.recipe.grad_clip)
            self.optim.step()
            self.global_step += 1

            n += 1
            acc["loss"] += float(loss_f.detach())
            if st is not None:
                acc["surf"] += float(st["sem"])
            if it is not None:
                acc["class"] += float(it["class"]); acc["bce"] += float(it["mask_bce"])
                acc["dice"] += float(it["mask_dice"]); acc["obj"] += float(it["obj"])
                acc["dn"] += float(it.get("dn", 0.0))
            if skt is not None:
                acc["symskel"] += float(skt["tversky"])
            if mt is not None:
                acc["malis"] += float(mt["malis"])

        m = {k: v / max(n, 1) for k, v in acc.items()}
        dt = time.time() - t0
        print(
            f"{_C['tot']}[ep {epoch}] loss {m['loss']:.4f}{_C['r']} | "
            f"{_C['sem']}surf {m['surf']:.3f}{_C['r']} | "
            f"{_C['inst']}inst [class {m['class']:.3f} bce {m['bce']:.3f} dice {m['dice']:.3f} obj {m['obj']:.3f} dn {m['dn']:.3f}]{_C['r']} | "
            f"{_C['sem']}symskel {m['symskel']:.3f} malis {m['malis']:.4f}{_C['r']} "
            f"{_C['dim']}| lr {self.optim.param_groups[0]['lr']:.2e} {'frozen' if self._frozen else 'ft'} {dt:.0f}s{_C['r']}",
            flush=True,
        )

    # ---- validation (per-sheet coverage, §13.5) ---------------------------------------------------
    def validate(self, val_loader, epoch: int) -> float:
        """Per-sheet coverage on the held-out windows (the metric that matters, not mask-Dice). Logs a ``[val epN]``
        line and returns the primary score (IoU>0.5 capture rate) used for best-checkpoint selection."""
        from .metrics import coverage_metrics
        mw = int(self.recipe.val_windows) or 10 ** 9                     # 0 => all val windows
        m = coverage_metrics(self.net, val_loader, self.device, max_windows=mw,
                             mask_mode=getattr(self.recipe, "mask_mode", "sigmoid"))
        print(
            f"{_C['sem']}[val ep {epoch}] coverage: IoU mean {m['iou_mean']:.3f} med {m['iou_median']:.3f} "
            f"p90 {m['iou_p90']:.3f} | captured >0.1 {m['cap010']*100:.0f}% >0.5 {m['cap050']*100:.0f}% | "
            f"fire {m['fire_mean']:.1f}/{self.recipe.n_queries} [{m['fire_min']}-{m['fire_max']}] "
            f"({m['sheets']} sheets / {m['windows']} win){_C['r']}",
            flush=True,
        )
        return float(m["cap050"])

    # ---- drive ------------------------------------------------------------------------------------
    def run(self, loader, val_loader=None) -> None:
        self._steps_per_epoch = len(loader) if hasattr(loader, "__len__") else 1
        print(f"[recipe] {self.recipe.name}: {self.recipe.summary()}", flush=True)
        self.net.to(self.device)
        vi = int(getattr(self.recipe, "val_interval", 0))
        best = -1.0
        for epoch in range(self.start_epoch, self.num_epochs):
            self._set_frozen(self._frozen_at(epoch))
            self._run_epoch(loader, epoch)
            self.save_checkpoint(epoch, "checkpoint_latest.pth")
            # coverage validation is instance-specific; in surface-only training the instance head is untrained, so
            # skip it (the surface loss trend + checkpoint_latest are the signal). Re-enabled once instance is on.
            if (val_loader is not None and vi > 0 and not bool(getattr(self.recipe, "surface_only", False))
                    and ((epoch + 1) % vi == 0 or epoch == self.num_epochs - 1)):
                score = self.validate(val_loader, epoch)
                if score > best:                                        # best-by-coverage checkpoint
                    best = score
                    self.save_checkpoint(epoch, "checkpoint_best.pth")
                    print(f"[hercuvit] new best coverage (IoU>0.5 {score*100:.0f}%) @ ep {epoch} → checkpoint_best.pth",
                          flush=True)
        self.save_checkpoint(self.num_epochs - 1, "checkpoint_final.pth")
        print("[hercuvit] training done", flush=True)

    # ---- checkpoints ------------------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, name: str) -> None:
        from dataclasses import asdict
        path = os.path.join(self.out_dir, name)
        torch.save({
            "model": self.net.state_dict(),
            "model_cfg": self.model_cfg,
            "recipe": asdict(self.recipe),
            "arch_override": getattr(self.net, "_hercuvit_arch", {}),   # encoder arch assembled from the recipe (for infer)
            "variant": self.recipe.variant,
            # keep the `full` variant's "all" sentinel a string (list("all") would mangle it to ['a','l','l']).
            "global_blocks": self.recipe.global_blocks if isinstance(self.recipe.global_blocks, str)
                             else list(self.recipe.global_blocks),
            "window_size": list(self.recipe.window_size),
            "epoch": epoch,
            "global_step": self.global_step,
            "optim": self.optim.state_dict() if self.optim is not None else None,
            "frozen": self._frozen,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.start_epoch = int(ck.get("epoch", -1)) + 1
        self.global_step = int(ck.get("global_step", 0))
        # rebuild the optimiser for the resumed freeze state, then load its state
        self._set_frozen(self._frozen_at(self.start_epoch))
        if ck.get("optim") is not None:
            try:
                self.optim.load_state_dict(ck["optim"])
            except (ValueError, KeyError):
                print("[hercuvit] optimiser state did not match the resumed freeze state — continuing fresh optim", flush=True)
        print(f"[hercuvit] resumed from {path} at epoch {self.start_epoch}", flush=True)
