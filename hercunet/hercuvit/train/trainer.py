"""HercuViT standalone training loop (NOT nnU-Net; ``docs/hercuvit.md`` §7).

HercuViT is not an nnU-Net, so it has its own PyTorch loop (per the owner's "whole new tooling, in parallel"
directive) that reuses hercunet machinery around it: the data layer (dataset), the HF pull + warm-start pattern
(encoder loaded by ``build_hercuvit``), and the single-line ``[ep N] loss …`` logging. bf16 autocast, AdamW on
``net.param_groups``, grad-clip, warmup + constant (or cosine) LR.

Staged schedule (§7), one ``run()``:
  1. **Semantic** (epochs ``[0, semantic_epochs)``) — transformer frozen, stem + pixel decoder + semantic head
     at full LR; Dice + CE deep supervision.
  2. **Unfreeze** (next ``unfreeze_epochs``) — all trainable, transformer at ``transformer_lr_mult`` × LR, windows
     ON (always on — the retrofit is structural); still semantic-only.
  3. **Instance** (remaining ``instance_epochs``) — add the Mask2Former instance losses; joint, encoder low LR.

The optimiser is rebuilt at each stage transition (the trainable-parameter set changes). Checkpoints save model +
optimiser + epoch + the Dinovol ``model_cfg`` (so inference rebuilds the architecture without the Dinovol ckpt).
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
        self.num_epochs = int(recipe.semantic_epochs + recipe.unfreeze_epochs + recipe.instance_epochs)
        self.start_epoch = 0
        self.global_step = 0
        self._stage = None
        self.optim = None

    # ---- staged schedule --------------------------------------------------------------------------
    def stage_for_epoch(self, ep: int) -> int:
        r = self.recipe
        if ep < r.semantic_epochs:
            return 1
        if ep < r.semantic_epochs + r.unfreeze_epochs:
            return 2
        return 3

    def _configure_stage(self, stage: int) -> None:
        if stage == self._stage:
            return
        self.net.set_transformer_trainable(stage >= 2)          # stage 1 freezes the transformer
        self._build_optimiser()
        self._stage = stage
        print(f"[stage {stage}] transformer_trainable={stage >= 2} instance_loss={stage >= 3}", flush=True)

    def _build_optimiser(self) -> None:
        groups = self.net.param_groups(base_lr=self.recipe.base_lr,
                                       transformer_lr_mult=self.recipe.transformer_lr_mult)
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
    def _run_epoch(self, loader, epoch: int, stage: int) -> None:
        self.net.train()
        acc = {k: 0.0 for k in ("loss", "class", "bce", "dice", "obj", "symskel", "malis")}
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
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                out = self.net(ct)
                inst = out["instance"]
                # DEFAULT: PLAIN Mask2Former on THIN labels — Hungarian + per-query sigmoid mask BCE+Dice (+token
                # objectness) against each sheet's THIN medial surface (owner & sem==1), ignore-masked. Each matched
                # mask IS a thin sheet and its query index IS the membership; the readout max_q[P(sheet_q)·σ(mask_q)]
                # is thin for free. On the overfit windows this matched the full old-HercUNet machinery, so it's the
                # baseline. SymSkel (per-query thinness) and AffMALIS (separation) are OPTIONAL add-ons — enabled by
                # recipe w_symskel / w_malis > 0 — for A/B at corpus scale, where near-identical compressed sheets
                # actually stress separation. (The intermediate node/edge ownership head is gone.)
                it = L.instance_loss(inst, out["mask_features"], owner_thin, has_id, self.recipe,
                                     ignore=(sem == 2), seed=self.global_step)
                loss = it["inst"]
                skt = mt = None
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
            acc["class"] += float(it["class"]); acc["bce"] += float(it["mask_bce"])
            acc["dice"] += float(it["mask_dice"]); acc["obj"] += float(it["obj"])
            if skt is not None:
                acc["symskel"] += float(skt["tversky"])
            if mt is not None:
                acc["malis"] += float(mt["malis"])

        m = {k: v / max(n, 1) for k, v in acc.items()}
        dt = time.time() - t0
        print(
            f"{_C['tot']}[ep {epoch}] loss {m['loss']:.4f}{_C['r']} | "
            f"{_C['inst']}inst [class {m['class']:.3f} bce {m['bce']:.3f} dice {m['dice']:.3f} obj {m['obj']:.3f}]{_C['r']} | "
            f"{_C['sem']}symskel {m['symskel']:.3f} malis {m['malis']:.4f}{_C['r']} "
            f"{_C['dim']}| lr {self.optim.param_groups[0]['lr']:.2e} stage {stage} {dt:.0f}s{_C['r']}",
            flush=True,
        )

    # ---- drive ------------------------------------------------------------------------------------
    def run(self, loader) -> None:
        self._steps_per_epoch = len(loader) if hasattr(loader, "__len__") else 1
        print(f"[recipe] {self.recipe.name}: {self.recipe.summary()}", flush=True)
        self.net.to(self.device)
        for epoch in range(self.start_epoch, self.num_epochs):
            stage = self.stage_for_epoch(epoch)
            self._configure_stage(stage)
            self._run_epoch(loader, epoch, stage)
            self.save_checkpoint(epoch, "checkpoint_latest.pth")
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
            "arch_override": getattr(self.net, "_hercuvit_arch", {}),   # resolved VARIANT_ARCH + tweak (for infer)
            "variant": self.recipe.variant,
            "global_blocks": list(self.recipe.global_blocks),
            "window_size": list(self.recipe.window_size),
            "epoch": epoch,
            "global_step": self.global_step,
            "optim": self.optim.state_dict() if self.optim is not None else None,
            "stage": self._stage,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.start_epoch = int(ck.get("epoch", -1)) + 1
        self.global_step = int(ck.get("global_step", 0))
        # rebuild the optimiser for the resumed stage, then load its state
        stage = self.stage_for_epoch(self.start_epoch)
        self._configure_stage(stage)
        if ck.get("optim") is not None:
            try:
                self.optim.load_state_dict(ck["optim"])
            except (ValueError, KeyError):
                print("[hercuvit] optimiser state did not match the resumed stage — continuing fresh optim", flush=True)
        print(f"[hercuvit] resumed from {path} at epoch {self.start_epoch}", flush=True)
