"""``hercuvit stitcher train`` — fit the cross-window stitching decoder on ``labels create`` samples (docs §7).

A small standalone PyTorch loop in the HercuViT style (not nnU-Net): AdamW on two param groups (no weight-decay on
LayerNorm / biases / ``s`` / ``b`` / ``E_slot``, §7.8), warmup → cosine (or constant) LR, bf16 autocast for the
decoder body with the cosine affinity head + BCE loss in float32, a one-line ``[ep N]`` audit log, and
``checkpoint_{latest,best,final}.pth`` carrying the resolved :class:`StitchConfig` so inference rebuilds the exact
decoder. Validation (every ``val_interval`` epochs) is edge **average precision** on a held-out fold (§7.9).
"""
from __future__ import annotations

import os
import time

import torch

from .config import StitchConfig, CONFIG_FIELDS, config_from_dict
from .loss import stitch_loss

_C = dict(tot="\033[1;97m", val="\033[36m", dim="\033[2m", r="\033[0m")


class StitchTrainer:
    def __init__(self, model, cfg: StitchConfig, out_dir: str, *, device="cuda") -> None:
        self.model = model
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = torch.device(device)
        os.makedirs(out_dir, exist_ok=True)
        self.start_epoch = 0
        self.global_step = 0
        self._steps_per_epoch = 1
        self._build_optimiser()

    def _build_optimiser(self) -> None:
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim <= 1 or "slot" in name) else decay).append(p)   # no WD on LN/bias/s/b/E_slot (§7.8)
        self.optim = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=self.cfg.base_lr)

    def _lr_scale(self) -> float:
        import math
        w = max(1, int(self.cfg.warmup_steps))
        if self.global_step < w:
            return (self.global_step + 1) / w
        if self.cfg.lr_schedule == "cosine":
            total = max(1, self.cfg.epochs * max(1, self._steps_per_epoch))
            prog = min(1.0, (self.global_step - w) / max(1, total - w))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return 1.0

    # ---- one epoch -------------------------------------------------------------------------------
    def _run_epoch(self, loader, epoch: int) -> None:
        self.model.train()
        acc = torch.zeros((), device=self.device)
        n = 0
        pw_last = 0.0
        facc = cacc = 0.0                                          # focal / contrastive running sums (for the log)
        t0 = time.time()
        for batch in loader:
            Xs = batch["Xs"].to(self.device, non_blocking=True)
            Ps = batch["Ps"].to(self.device, non_blocking=True)
            slot_s = batch["slot_s"].to(self.device, non_blocking=True)
            valid_s = batch["valid_s"].to(self.device, non_blocking=True)
            Xq = batch["Xq"].to(self.device, non_blocking=True)
            slot_q = batch["slot_q"].to(self.device, non_blocking=True)
            conf_q = batch["conf_q"].to(self.device, non_blocking=True)
            valid_q = batch["valid_q"].to(self.device, non_blocking=True)
            Y = batch["Y"].to(self.device, non_blocking=True)

            for g in self.optim.param_groups:
                g["lr"] = self.cfg.base_lr * self._lr_scale()
            self.optim.zero_grad(set_to_none=True)
            # decoder body in bf16; the affinity head + BCE run in float32 (stitch_loss upcasts the logits).
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                out = self.model(Xs, Ps, slot_s, valid_s, Xq, slot_q, valid_q)
            loss, st = stitch_loss(out, Y, valid_q, conf_q, pos_weight=self.cfg.pos_weight,
                                   gamma=self.cfg.focal_gamma, tau=self.cfg.contrast_tau,
                                   lambda_contrast=self.cfg.contrast_weight)
            if not torch.isfinite(loss):
                self.optim.zero_grad(set_to_none=True); self.global_step += 1; continue
            loss.backward()
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
            self.global_step += 1
            acc += loss.detach(); n += 1; pw_last = st["pos_weight"]
            facc += st["focal"]; cacc += st["contrast"]

        m = float(acc) / max(n, 1)
        dt = time.time() - t0
        print(f"{_C['tot']}[ep {epoch}] loss {m:.4f}{_C['r']} | "
              f"{_C['dim']}focal {facc / max(n, 1):.3f} contrast {cacc / max(n, 1):.3f} pos_w {pw_last:.1f} "
              f"lr {self.optim.param_groups[0]['lr']:.2e} {dt:.0f}s{_C['r']}", flush=True)

    # ---- validation: edge average precision on the held-out fold (§7.9) --------------------------
    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> float:
        self.model.eval()
        scores, labels = [], []
        vloss, nb = 0.0, 0
        for batch in val_loader:
            mv = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = self.model(mv["Xs"], mv["Ps"], mv["slot_s"], mv["valid_s"], mv["Xq"], mv["slot_q"], mv["valid_q"])
            loss, _ = stitch_loss(out, mv["Y"], mv["valid_q"], mv["conf_q"], pos_weight=self.cfg.pos_weight,
                                  gamma=self.cfg.focal_gamma, tau=self.cfg.contrast_tau,
                                  lambda_contrast=self.cfg.contrast_weight)       # SAME loss on val → overfit gap
            vloss += float(loss); nb += 1
            logits = out[-1]["logit"]
            Y, vq = mv["Y"], mv["valid_q"]
            B, M, _ = Y.shape
            vpair = vq[:, :, None] & vq[:, None, :]
            eye = torch.eye(M, dtype=torch.bool, device=self.device)[None]
            valid = vpair & ~eye & (Y >= 0)
            scores.append(logits[valid].float().cpu()); labels.append((Y == 1)[valid].cpu())
        if not scores:
            return 0.0
        sc = torch.cat(scores).numpy(); la = torch.cat(labels).numpy().astype(int)
        vl = vloss / max(nb, 1)
        if la.sum() == 0 or la.sum() == len(la):
            return 0.0
        from sklearn.metrics import average_precision_score
        ap = float(average_precision_score(la, sc))
        pos_frac = la.mean()
        print(f"{_C['val']}[val ep {epoch}] loss {vl:.4f}  edge AP {ap:.4f} "
              f"(over {len(la)} pairs, {pos_frac*100:.1f}% positive){_C['r']}", flush=True)
        return ap

    # ---- drive -----------------------------------------------------------------------------------
    def run(self, loader, val_loader=None) -> None:
        self._steps_per_epoch = len(loader) if hasattr(loader, "__len__") else 1
        print(f"[stitcher] config: {self.cfg.summary()}", flush=True)
        self.model.to(self.device)
        vi = int(self.cfg.val_interval)
        best = -1.0
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self._run_epoch(loader, epoch)
            self.save_checkpoint(epoch, "checkpoint_latest.pth")
            if val_loader is not None and vi > 0 and ((epoch + 1) % vi == 0 or epoch == self.cfg.epochs - 1):
                score = self.validate(val_loader, epoch)
                if score > best:
                    best = score
                    self.save_checkpoint(epoch, "checkpoint_best.pth")
                    print(f"[stitcher] new best edge AP ({score:.4f}) @ ep {epoch} → checkpoint_best.pth", flush=True)
        self.save_checkpoint(self.cfg.epochs - 1, "checkpoint_final.pth")
        print("[stitcher] training done", flush=True)

    # ---- checkpoints -----------------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, name: str) -> None:
        from dataclasses import asdict
        torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg),
                    "epoch": epoch, "global_step": self.global_step,
                    "optim": self.optim.state_dict()}, os.path.join(self.out_dir, name))

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.start_epoch = int(ck.get("epoch", -1)) + 1
        self.global_step = int(ck.get("global_step", 0))
        if ck.get("optim") is not None:
            self.optim.load_state_dict(ck["optim"])
        print(f"[stitcher] resumed from {path} at epoch {self.start_epoch}", flush=True)


def _resolve_samples(samples=None, samples_hf=None) -> str:
    """A LOCAL dir of ``stitch_*.npz``. ``--samples`` is a local dir; ``--samples-hf`` snapshot-downloads an HF
    dataset of samples (cached + idempotent — same mechanism the training corpus uses)."""
    if samples and samples_hf:
        raise SystemExit("hercuvit stitcher train: pass --samples (a local dir) OR --samples-hf (download), not both.")
    if samples:
        return samples
    if not samples_hf:
        raise SystemExit("hercuvit stitcher train: give samples — --samples <dir> or --samples-hf <repo> "
                         "(the output of `hercuvit labels create`).")
    from huggingface_hub import snapshot_download
    print(f"[stitcher] resolving samples from HF dataset {samples_hf} (cached if already pulled) …", flush=True)
    path = snapshot_download(samples_hf, repo_type="dataset", max_workers=16)
    print(f"[stitcher] samples at {path}", flush=True)
    return path


def train_run(a) -> None:
    """``hercuvit stitcher train`` handler: resolve samples, build the decoder, fit."""
    from torch.utils.data import DataLoader

    from .dataset import StitchSampleDataset, stitch_collate, find_samples, split_samples
    from .model import StitchDecoder

    cfg = config_from_dict({k: getattr(a, k) for k in CONFIG_FIELDS if getattr(a, k, None) is not None})
    samples_dir = _resolve_samples(getattr(a, "samples", None), getattr(a, "samples_hf", None))
    files = find_samples(samples_dir)
    tr_files, va_files = split_samples(files, fold=cfg.val_fold) if cfg.val_interval > 0 else (files, [])

    label_mode = getattr(a, "label_mode", "v2") or "v2"          # v2 = the default (mark certain negatives, not ignore)
    tr_ds = StitchSampleDataset(tr_files, pca_dim=cfg.pca_dim, label_mode=label_mode)
    loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=a.num_workers,
                        collate_fn=stitch_collate, drop_last=True, pin_memory=(a.device != "cpu"))
    val_loader = None
    if va_files:
        val_ds = StitchSampleDataset(va_files, pca_dim=cfg.pca_dim)   # val stays RAW so edge AP is comparable across modes
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=max(1, a.num_workers // 2), collate_fn=stitch_collate)
    print(f"[stitcher] {samples_dir}: {len(tr_ds)} train + {len(va_files)} val (fold {cfg.val_fold}); "
          f"batch {cfg.batch_size}; workers {a.num_workers}; device {a.device}; label_mode {label_mode} "
          f"(val=raw)", flush=True)

    model = StitchDecoder(cfg)
    trainer = StitchTrainer(model, cfg, a.out, device=a.device)
    if getattr(a, "continue_", False):
        latest = os.path.join(a.out, "checkpoint_latest.pth")
        if os.path.isfile(latest):
            trainer.load_checkpoint(latest)
        else:
            print("[stitcher] --continue: no checkpoint_latest.pth found, starting fresh", flush=True)
    trainer.run(loader, val_loader=val_loader)
