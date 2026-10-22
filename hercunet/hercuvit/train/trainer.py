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
import torch.nn.functional as F

from . import losses as L


# ANSI colours, matching the hercunet trainer's per-term log line.
_C = dict(tot="\033[1;97m", sem="\033[36m", inst="\033[35m", dim="\033[2m", r="\033[0m")


@torch.no_grad()
def _fps_sheet_coverage(query_idx, owner_thin, has_id, *, side: int = 16):
    k = max(1, owner_thin.shape[-1] // int(side))
    """FPS seed coverage over GT sheets: fraction of GT sheets that have >=1 FPS seed landing in their 1/8 cell.

    ``query_idx`` (B, 1+Q) are the FPS-selected 1/8 token indices (null = -1), ``owner_thin`` (B,D,H,W) the thin GT
    sheet ids. For each seed, its 8³ cell's GT sheet ids count as "covered". Returns ``(covered, total)`` summed
    over the batch (micro-average numerator/denominator), so the epoch coverage is Σcovered / Σsheets. Cheap:
    ~64 small ``unique`` calls per item, detached — a diagnostic of whether the query init even reaches every sheet."""
    B = owner_thin.shape[0]
    covered_tot = 0
    sheet_tot = 0
    qi = query_idx.detach()
    for b in range(B):
        if not bool(has_id[b]):
            continue
        ob = owner_thin[b]
        ids = torch.unique(ob); ids = ids[ids > 0]
        if ids.numel() == 0:
            continue
        seeds = qi[b]; seeds = seeds[seeds >= 0].tolist()             # drop the null query (-1); -> python ints
        covered: set[int] = set()
        for s in seeds:
            tz, ty, tx = s // (side * side), (s // side) % side, s % side
            blk = ob[tz * k:(tz + 1) * k, ty * k:(ty + 1) * k, tx * k:(tx + 1) * k]
            u = torch.unique(blk); u = u[u > 0]
            if u.numel():
                covered.update(int(x) for x in u.tolist())
        covered_tot += len(set(int(x) for x in ids.tolist()) & covered)
        sheet_tot += int(ids.numel())
    return covered_tot, sheet_tot


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
        self._dec_frozen = None
        self.optim = None

    # ---- LP-FT freeze-warmup ----------------------------------------------------------------------
    def _frozen_at(self, ep: int) -> bool:
        """True while the ENCODER (stem + transformer) is frozen — the first ``freeze_warmup_epochs`` (train the head
        on the encoder's features), then it is fine-tuned at ``encoder_lr_mult`` × LR."""
        return ep < int(self.recipe.freeze_warmup_epochs)

    def _decoder_frozen_at(self, ep: int) -> bool:
        """True while the warm-started decoder TRUNK + surface head are frozen — the first ``freeze_decoder_epochs``
        (instance-head warmup against a fixed surface gate); the mask head + instance head always train."""
        return ep < int(getattr(self.recipe, "freeze_decoder_epochs", 0) or 0)

    def _apply_freezing(self, ep: int) -> None:
        """Apply the encoder (LP-FT) and decoder-trunk (instance-warmup) freeze states for epoch ``ep``; rebuild the
        optimiser once if the trainable set changed."""
        enc_f, dec_f = self._frozen_at(ep), self._decoder_frozen_at(ep)
        changed = False
        if enc_f != self._frozen:
            self.net.set_encoder_trainable(not enc_f)           # freeze the WHOLE encoder (stem + transformer)
            self._frozen = enc_f; changed = True
        if dec_f != self._dec_frozen:
            self.net.set_decoder_trunk_trainable(not dec_f)     # freeze neck + upsample tail + surface head
            self._dec_frozen = dec_f; changed = True
        if changed:
            self._build_optimiser()                             # the trainable set changed → rebuild the optimiser
            print(f"[hercuvit] encoder {'FROZEN' if enc_f else 'trainable@×LR'} | decoder-trunk "
                  f"{'FROZEN (instance warmup)' if dec_f else 'trainable'}", flush=True)

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
        # accumulate on-GPU as tensors (one .item() per epoch at print time, NOT ~8 float() syncs/step — §profiling)
        acc = {k: torch.zeros((), device=self.device) for k in ("loss", "class", "own", "dn", "symskel", "malis", "surf", "fire", "share")}
        n_share = 0                                                    # count of steps that reported surf_share (subspace split)
        pdc = {k: torch.zeros((), device=self.device) for k in ("tp", "fp", "fn")}  # surface pseudo-Dice (nnU-Net-style)
        own = {"correct": torch.zeros((), device=self.device), "fg": torch.zeros((), device=self.device)}
        fpsc = {"cov": 0.0, "n": 0.0}                                  # FPS seed coverage (diagnostic; sampled every 25 steps)
        n = 0
        nan_skips = 0
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
                surface_in_play = (self.recipe.w_surface > 0 or surface_only)
                do_surface = surface_in_play and not bool(self._dec_frozen)  # surface head frozen in warmup → no loss
                st = None
                loss = None
                if do_surface:
                    st = L.semantic_loss(out["sem_logits"], sem, self.recipe)
                    w_surf = self.recipe.w_surface if self.recipe.w_surface > 0 else 1.0
                    loss = w_surf * st["sem"]
                it = skt = mt = None
                if run_instance:
                    inst = out["instance"]
                    # PLAIN Mask2Former on THIN labels — Hungarian + per-query mask BCE+Dice (+token objectness) against
                    # each sheet's THIN medial surface (owner & sem==1), ignore-masked. SymSkel / AffMALIS are optional
                    # add-ons (w_symskel / w_malis > 0). In design B the readout uses the surface-GATED mask features.
                    # design C: mask_features live at 1/4, so match the thin owner + ignore mask to that resolution
                    mf = out["mask_features"]
                    if mf.shape[-1] != owner_thin.shape[-1]:
                        r = owner_thin.shape[-1] // mf.shape[-1]
                        owner_i = F.max_pool3d(owner_thin.float()[:, None], r, r)[:, 0].long()
                        ignore_i = F.max_pool3d((sem == 2).float()[:, None], r, r)[:, 0] > 0.5
                    else:
                        owner_i, ignore_i = owner_thin, (sem == 2)
                    it = L.instance_loss(inst, mf, owner_i, has_id, self.recipe,
                                         ignore=ignore_i, seed=self.global_step)
                    loss = it["inst"] if loss is None else loss + it["inst"]
                    if self.recipe.w_symskel > 0 and inst is not None:
                        sk, skt = L.m2f_symskel_mask_loss(inst, out["mask_features"], sem, owner, has_id, self.recipe,
                                                          seed=self.global_step, use_softmax=False)
                        loss = loss + self.recipe.w_symskel * sk
                    if self.recipe.w_malis > 0 and inst is not None:
                        ml, mt = L.affinity_malis_loss(inst, out["mask_features"], owner, has_id, self.recipe)
                        loss = loss + self.recipe.w_malis * ml
            loss_f = loss.float()
            if not torch.isfinite(loss_f):                             # NaN/Inf guard: drop the batch, never step on it
                self.optim.zero_grad(set_to_none=True)
                nan_skips += 1; self.global_step += 1
                continue
            loss_f.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.recipe.grad_clip) \
                if self.recipe.grad_clip else None
            if gnorm is not None and not torch.isfinite(gnorm):        # non-finite grads → skip (don't poison weights)
                self.optim.zero_grad(set_to_none=True)
                nan_skips += 1; self.global_step += 1
                continue
            self.optim.step()
            self.global_step += 1

            n += 1
            acc["loss"] += loss_f.detach()
            if "surf_share" in out:
                acc["share"] += out["surf_share"].detach(); n_share += 1
            if st is not None:
                acc["surf"] += st["sem"].detach()
            if surface_in_play:
                # online surface pseudo-Dice (nnU-Net convention): accumulate TP/FP/FN over the epoch from the
                # full-res surface argmax vs the thin medial (sem==1), ignoring sem==2 → one epoch micro-Dice.
                # Logged even while the surface head is frozen (monitors that the warm-started head stays good).
                with torch.no_grad():
                    pr = out["sem_logits"][0].detach().float().argmax(1)   # (B,D,H,W) predicted class
                    vmask = sem != 2
                    pm = (pr == 1) & vmask
                    tm = (sem == 1) & vmask
                    pdc["tp"] += (pm & tm).sum(); pdc["fp"] += (pm & ~tm).sum(); pdc["fn"] += (~pm & tm).sum()
            if it is not None:
                acc["class"] += it["class"]; acc["own"] += it["mask_bce"]
                acc["dn"] += it["dn"]; acc["fire"] += it["n_fire"]
                own["correct"] += it["own_correct"]; own["fg"] += it["own_fg"]
                inst_out = out.get("instance")
                if inst_out is not None and inst_out.get("query_idx") is not None and (bi % 25 == 0):
                    c, ntot = _fps_sheet_coverage(inst_out["query_idx"], owner_thin, has_id,
                                                  side=int(inst_out.get("seed_grid", 16)))
                    fpsc["cov"] += c; fpsc["n"] += ntot
            if skt is not None:
                acc["symskel"] += float(skt["tversky"])
            if mt is not None:
                acc["malis"] += float(mt["malis"])

        m = {k: float(v) / max(n, 1) for k, v in acc.items()}         # tensor → float: ~8 syncs ONCE per epoch
        share_val = float(acc["share"]) / max(n_share, 1)             # mean ‖f[0:surf_dim]‖² (subspace norm share)
        pdice = float(2 * pdc["tp"] / (2 * pdc["tp"] + pdc["fp"] + pdc["fn"]).clamp_min(1.0))  # epoch micro pseudo-Dice
        own_acc = float(own["correct"]) / max(float(own["fg"]), 1.0)   # frac of sheet voxels owned by their matched query
        fps_cov = fpsc["cov"] / max(fpsc["n"], 1.0)                    # frac of GT sheets reached by >=1 FPS seed
        dt = time.time() - t0
        run_instance = not bool(self.recipe.surface_only)
        share_str = f" share {share_val:.3f}" if n_share > 0 else ""   # subspace norm share (≈0.19; →1 = collapse)
        parts = [f"{_C['tot']}[ep {epoch}] loss {m['loss']:.4f}{_C['r']}",
                 f"{_C['sem']}surf {m['surf']:.3f} pdice {pdice:.4f}{share_str}{_C['r']}"]
        if run_instance:
            # own-acc = ownership quality (↑ = masks carve the surface into their matched sheets); fire = queries used
            parts.append(f"{_C['inst']}inst [class {m['class']:.3f} own-ce {m['own']:.3f} dn {m['dn']:.3f}] "
                         f"own-acc {own_acc:.3f} fire {m['fire']:.1f}/{int(self.recipe.n_queries)} "
                         f"fps-cov {fps_cov:.2f}{_C['r']}")
            extra = []
            if self.recipe.w_symskel > 0: extra.append(f"symskel {m['symskel']:.3f}")
            if self.recipe.w_malis > 0: extra.append(f"malis {m['malis']:.4f}")
            if extra:
                parts.append(f"{_C['sem']}{' '.join(extra)}{_C['r']}")
        tail = (f"{_C['dim']}lr {self.optim.param_groups[0]['lr']:.2e} {'frozen' if self._frozen else 'ft'}"
                f"{f' skip {nan_skips}' if nan_skips else ''} {dt:.0f}s{_C['r']}")
        print(" | ".join(parts) + " | " + tail, flush=True)

    # ---- validation (per-sheet coverage, §13.5) ---------------------------------------------------
    def validate(self, val_loader, epoch: int) -> float:
        """Validation on the held-out fold. In **surface-only** training the instance head is untrained, so score the
        surface head (foreground Dice + tolerant surface Dice); otherwise score per-sheet instance coverage (§13.5).
        Logs a ``[val epN]`` line and returns the primary score for best-checkpoint selection."""
        mw = int(self.recipe.val_windows) or 10 ** 9                     # 0 => all val windows
        if bool(getattr(self.recipe, "surface_only", False)):
            from .metrics import surface_metrics
            m = surface_metrics(self.net, val_loader, self.device, max_windows=mw)
            print(
                f"{_C['sem']}[val ep {epoch}] surface: fg-Dice {m['fg_dice']:.3f} | tol±1 Dice {m['sdice_tol1']:.3f} "
                f"(P {m['precision']:.3f} R {m['recall']:.3f}) | {m['nfg']}/{m['windows']} win w/ surface{_C['r']}",
                flush=True,
            )
            return float(m["sdice_tol1"])
        from .metrics import coverage_metrics
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
            self._apply_freezing(epoch)
            self._run_epoch(loader, epoch)
            self.save_checkpoint(epoch, "checkpoint_latest.pth")
            # validation: surface-head Dice in surface-only mode, instance coverage otherwise (validate() branches).
            if (val_loader is not None and vi > 0
                    and ((epoch + 1) % vi == 0 or epoch == self.num_epochs - 1)):
                score = self.validate(val_loader, epoch)
                if score > best:                                        # best-by-validation-score checkpoint
                    best = score
                    metric = "surface tol±1 Dice" if bool(getattr(self.recipe, "surface_only", False)) else "coverage IoU>0.5"
                    self.save_checkpoint(epoch, "checkpoint_best.pth")
                    print(f"[hercuvit] new best {metric} ({score:.3f}) @ ep {epoch} → checkpoint_best.pth", flush=True)
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
        self._apply_freezing(self.start_epoch)
        if ck.get("optim") is not None:
            try:
                self.optim.load_state_dict(ck["optim"])
            except (ValueError, KeyError):
                print("[hercuvit] optimiser state did not match the resumed freeze state — continuing fresh optim", flush=True)
        print(f"[hercuvit] resumed from {path} at epoch {self.start_epoch}", flush=True)
