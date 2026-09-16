"""v3 AFFINITY + MALIS iterative refiner — **the HercUNet run301 / v0 trainer**.

Extends the DAgger iterative refiner with (a) a CT **orientation field** input (6 sign-free structure-tensor
channels, recomputed each step from the augmented CT — v3 §2.3), (b) an **affinity output head** (v3 §2.2),
and (c) **constrained MALIS** over the per-sheet instance GT (v3 §4.1), pass/epoch-**ramped** (v3 §5.1). The v2
surface head + SepSkelSym loss are kept (detection substrate). Iteration/DAgger/prev-corruption unchanged.

Network input at forward = ``[CT(1), prev_crest(1), orientation(6)] = 8`` channels; outputs = ``(seg_ds, aff)``.

**Owner instance GT** rides as a SECOND segmentation channel (nearest-interp, threaded through nnU-Net's
augmentation exactly like the surface seg) via ``_OwnerDataLoader`` + ``get_dataloaders`` (the owner-aware
loader pattern proven in ``nnUNetTrainer_MultiTaskSurface``). Prepare it with ``hercunet train export-owner``
(writes ``<case>_owner.b2nd`` at preprocessed resolution). ``train_step`` splits the 2-channel seg target:
channel 0 = surface (→ SepSkelSym), channel 1 = owner id (→ constrained MALIS at full res).

All hyperparameters (MALIS weight/ramp/crop, affinity offsets, material-growth, DAgger, prev composition)
come from the **training recipe** :mod:`hercunet.training.recipe` — the run301 / v0 values are the defaults, so
a bare ``hercunet train fit`` reproduces the recipe with no env vars, and the trainer logs the resolved recipe
at the start of each run. Warm-start from the m7 checkpoint with
``hercunet train fit --pretrained <m7 checkpoint_best.pth>`` — the trainer's :meth:`load_pretrained` detects the
m7-like single-channel stem and expands it (see below).
"""
import os

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from hercunet.training.trainers.focal_tversky_sep_skel_sym_iter_dagger import (
    nnUNetTrainer_FocalTverskySepSkelSym_IterDagger_500epochs)
from hercunet.training.affinity import AffinityHeadNet, ct_orientation, material_grow_loss, material_prob
from hercunet.training.malis import ConstrainedMalisLoss
from hercunet.training.recipe import active as active_recipe

__all__ = ["nnUNetTrainer_AffinityMalis_IterDagger_500epochs"]

_N_ORIENT = 6


class _OwnerDataLoader(nnUNetDataLoader):
    """Stock loader + loads ``<case>_owner.b2nd`` and threads it as a SECOND segmentation channel (nearest interp,
    so it is spatially augmented exactly like the surface seg). Full-res owner survives as channel 1 of the DS[0]
    target; the trainer splits it off for MALIS."""

    def _load_owner(self, identifier):
        import blosc2
        from batchgenerators.utilities.file_and_folder_operations import join
        path = join(self._data.source_folder, "owner_b2nd", identifier + ".b2nd")
        if not os.path.exists(path):                                  # m7 MALIS-off cases have no owner -> all-zero
            return None
        return blosc2.open(urlpath=path, mode="r", dparams={"nthreads": 1}, **self._data.mmap_kwargs)

    def generate_train_batch(self):
        selected = self.get_indices()
        data_all = seg_all = None
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for j, i in enumerate(selected):
                force_fg = self.get_do_oversample(j)
                data, seg, seg_prev, properties = self._data.load_case(i)
                owner = self._load_owner(i)
                shape = data.shape[1:]
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
                bbox = [[l, u] for l, u in zip(bbox_lbs, bbox_ubs)]
                data_c = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                seg_c = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                if owner is None:                                       # m7 (MALIS-off) case: no ownersTr -> zero
                    own_c = torch.zeros((1, *data_c.shape[1:]), dtype=torch.int16)  # owner=0 -> MALIS silent
                else:
                    own_c = torch.from_numpy(crop_and_pad_nd(owner, bbox, 0, cast_cropped_to=np.int16)).to(torch.int16)
                if seg_prev is not None:
                    sp = torch.from_numpy(crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    seg_c = torch.cat((seg_c, sp[None]), dim=0)
                seg_c = torch.cat((seg_c, own_c), dim=0)                # <-- owner as an extra seg channel (nearest)
                if self.patch_size_was_2d:
                    data_c, seg_c = data_c[:, 0], seg_c[:, 0]
                if self.transforms is not None:
                    tr = self.transforms(**{"image": data_c, "segmentation": seg_c})
                    data_s, seg_s = tr["image"], tr["segmentation"]
                else:
                    data_s, seg_s = data_c, seg_c
                if data_all is None:
                    data_all = torch.empty((self.batch_size, *data_s.shape), dtype=torch.float32)
                data_all[j] = data_s
                if isinstance(seg_s, list):
                    if seg_all is None:
                        seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_s]
                    for k, s in enumerate(seg_s):
                        seg_all[k][j] = s
                else:
                    if seg_all is None:
                        seg_all = torch.empty((self.batch_size, *seg_s.shape), dtype=seg_s.dtype)
                    seg_all[j] = seg_s
        return {"data": data_all, "target": seg_all, "keys": selected}


class nnUNetTrainer_AffinityMalis_IterDagger_500epochs(
        nnUNetTrainer_FocalTverskySepSkelSym_IterDagger_500epochs):

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # ALL hyperparameters come from the active training recipe (run301 unless --recipe overrode it;
        # see hercunet.training.recipe). The resolved recipe is logged in initialize() for per-run audit.
        self.recipe = active_recipe()
        self.offsets = self.recipe.offsets()
        self.malis = ConstrainedMalisLoss(offsets=self.offsets, crop=self.recipe.malis_crop)
        self.malis.last_grow = self.malis.last_sep = 0.0
        # CT-material growth/air-suppression (v3 §4.3). tau/scale in NORMALIZED-CT space; run301: tau=0, s=0.4.
        self.mat_tau = self.recipe.mat_tau
        self.mat_scale = self.recipe.mat_scale
        # validation cross-section viz (safety check vs "imaginary sheets"). OFF in run301.
        self.val_viz = self.recipe.val_viz
        self.val_viz_images = self.recipe.val_viz_images
        self.val_viz_slices = self.recipe.val_viz_slices
        self._val_viz_n = 0
        self._vgrp = {}                                              # per-group fg-Dice on the VAL set
        self._rng = np.random.default_rng(0)
        self._reset_terms()

    # ---- per-term logging: surface vs MALIS, split into grow(attractive)/sep(repulsive) ----
    def _reset_terms(self):
        self._n = 0; self._s_surf = self._s_mal = self._s_grow = self._s_sep = self._s_w = 0.0
        self._s_mgrow = self._s_mair = self._s_mw = 0.0
        self._grp = {}                                                # name -> [weighted_loss_sum, count]

    def _acc(self, surf, mal, grow, sep, w, mgrow=0.0, mair=0.0, mw=0.0):
        self._n += 1
        self._s_surf += surf; self._s_mal += mal; self._s_grow += grow; self._s_sep += sep; self._s_w += w
        self._s_mgrow += mgrow; self._s_mair += mair; self._s_mw += mw

    def _group_dice(self, seg_out, surf_target, prev, keys, grp):
        """Accumulate per-item foreground soft-Dice into ``grp`` (name -> [sum, count]), bucketed by regime
        (cold=0'd-prev/bootstrap vs warm=prev-present) and by source (our/m7/other, from the case-id prefix).
        Used for BOTH train and val (separate grp dicts) so we can see whether each regime/dataset is actually
        LEARNING. Dice is smoothed -> NaN-free even on empty-foreground items (self.loss is not). Ignore(2)
        voxels excluded."""
        out = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
        with torch.no_grad():
            B = out.shape[0]
            prob = torch.softmax(out.float(), dim=1)[:, 1]            # (B,Z,Y,X) fg prob
            tgt = surf_target[0][:, 0]                                # (B,Z,Y,X) {0,1,2}
            valid = (tgt != 2).float()
            p = (prob * valid).reshape(B, -1)
            g = ((tgt == 1).float() * valid).reshape(B, -1)
            dice = (2 * (p * g).sum(1) + 1.0) / (p.sum(1) + g.sum(1) + 1.0)   # per-item, smoothed
            cold = (prev.reshape(B, -1).abs().amax(dim=1) < 1e-6).tolist()
            d = dice.tolist()
        nkeys = len(keys) if keys is not None else 0
        def add(name, v):
            s = grp.setdefault(name, [0.0, 0]); s[0] += v; s[1] += 1
        for b in range(B):
            add("cold" if cold[b] else "warm", d[b])
            k = str(keys[b]) if b < nkeys else ""
            add("m7" if k.startswith("m7") else ("our" if k.startswith("our") else "other"), d[b])

    def _print_grp(self, tag, grp):
        if not grp:
            return
        C = {"dim": "\033[2m", "r": "\033[0m"}
        order = ["cold", "warm", "our", "m7", "other"]
        parts = " ".join(f"{k} {grp[k][0] / max(grp[k][1], 1):.3f}{C['dim']}(n{grp[k][1]}){C['r']}"
                         for k in order if k in grp)
        print(f"{C['dim']}[ep {self.current_epoch}] {tag} fg-Dice by-group (↑better):{C['r']} {parts}", flush=True)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._reset_terms()

    def on_train_epoch_end(self, train_outputs):
        n = max(self._n, 1)
        surf, mal = self._s_surf / n, self._s_mal / n
        grow, sep, w = self._s_grow / n, self._s_sep / n, self._s_w / n
        mgrow, mair, mw = self._s_mgrow / n, self._s_mair / n, self._s_mw / n
        C = dict(tot="\033[1;97m", surf="\033[36m", mal="\033[35m", grow="\033[32m", sep="\033[31m",
                 mat="\033[33m", air="\033[34m", dim="\033[2m", r="\033[0m")
        print(f"{C['tot']}[ep {self.current_epoch}] loss {surf + w * mal + mw * (mgrow + mair):.4f}{C['r']} | "
              f"{C['surf']}surf {surf:.4f}{C['r']} | "
              f"{C['mal']}malis {mal:.4f}{C['r']} {C['dim']}(w{w:.2f}){C['r']} "
              f"[{C['grow']}grow {grow:.4f}{C['r']} {C['sep']}sep {sep:.4f}{C['r']}] | "
              f"{C['mat']}mat {mgrow + mair:.4f}{C['r']} {C['dim']}(w{mw:.2f}){C['r']} "
              f"[{C['mat']}mgrow {mgrow:.4f}{C['r']} {C['air']}air {mair:.4f}{C['r']}]", flush=True)
        self._print_grp("train", getattr(self, "_grp", None))       # per regime (cold/warm) + source (our/m7)
        super().on_train_epoch_end(train_outputs)

    def build_network_architecture(self, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        base = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, 2 + _N_ORIENT, num_output_channels, enable_deep_supervision)
        return AffinityHeadNet(base, n_aff=len(self.offsets))

    def initialize(self):
        super().initialize()
        self.print_to_log_file(f"[recipe] {self.recipe.name}: {self.recipe.summary()}")   # per-run audit

    def set_deep_supervision_enabled(self, enabled: bool):
        net = self.network.module if self.is_ddp else self.network
        net.base.decoder.deep_supervision = enabled

    def load_pretrained(self, ckpt_path):
        """Smart warm-start for ``hercunet train fit --pretrained``. Auto-detects the checkpoint shape and does
        the right thing PER PARAMETER — no env var, no ablation-vs-recipe split:

          * **m7-like** (a single-channel-stem base network): its keys map onto our ``base.*`` sub-net, and the
            8-channel input stem is expanded from m7's 1-channel weights (channel 0 = m7's CT weights, the
            ``prev`` + orientation channels zero-init) so the first forward is identical to m7 and training
            learns to use the extra channels. The affinity head initialises fresh.
          * **HercUNet-like** (a full AffinityHeadNet checkpoint, e.g. an earlier run): every matching key loads
            directly (equivalent to a warm-start from our own architecture; use ``--continue`` to resume a run).

        Matching is by parameter: try a direct key match, then the ``base.``-stripped m7 mapping; load on exact
        shape, stem-expand a ``[out,1,k,k,k]``→``[out,C,k,k,k]`` conv, else keep the fresh init. Reports the tally
        so a wrong/garbage checkpoint (mostly "fresh") is obvious in the log."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pre = sd.get("network_weights", sd)
        net = self.network.module if self.is_ddp else self.network
        tgt = net.state_dict(); merged = {}; loaded = stem = kept = 0
        for k, v in tgt.items():
            cands = [k] + ([k[5:]] if k.startswith("base.") else [])   # direct (HercUNet) then m7-base mapping
            p = next((pre[c] for c in cands if c in pre), None)
            if p is not None:
                if p.shape == v.shape:
                    merged[k] = p; loaded += 1; continue
                if v.dim() == 5 and p.dim() == 5 and p.shape[1] == 1 and v.shape[0] == p.shape[0]:
                    w = v.clone().zero_(); w[:, 0:1] = p; merged[k] = w; stem += 1; continue  # 1→C stem expand
            merged[k] = v; kept += 1
        net.load_state_dict(merged, strict=True)
        self.print_to_log_file(
            f"[warm-start] {os.path.basename(ckpt_path)}: {loaded} matched, {stem} stem-expanded "
            f"({'m7-like' if stem else 'no stem expansion'}), {kept} fresh")

    # ---- data: swap in the owner-aware loader (structure copied from MultiTaskSurface, class swapped) ----
    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        ds_scales = self._get_deep_supervision_scales()
        rot, dummy2d, init_ps, mirror = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_tf = self.get_training_transforms(
            patch_size, rot, ds_scales, mirror, dummy2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_tf = self.get_validation_transforms(
            ds_scales, is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        ds_tr, ds_val = self.get_tr_and_val_datasets()
        dl_tr = _OwnerDataLoader(ds_tr, self.batch_size, init_ps, patch_size, self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_tf,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = _OwnerDataLoader(ds_val, self.batch_size, patch_size, patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_tf,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        nproc = get_allowed_n_proc_DA()
        if nproc == 0:
            gtr = SingleThreadedAugmenter(dl_tr, None); gval = SingleThreadedAugmenter(dl_val, None)
        else:
            gtr = NonDetMultiThreadedAugmenter(dl_tr, None, nproc, max(6, nproc // 2), None,
                                               self.device.type == "cuda", 0.002)
            gval = NonDetMultiThreadedAugmenter(dl_val, None, max(1, nproc // 2), max(3, nproc // 4), None,
                                                self.device.type == "cuda", 0.002)
        _ = next(gtr); _ = next(gval)
        return gtr, gval

    def _malis_weight(self):
        warm = max(1, self.recipe.malis_warm_epochs)
        return self.recipe.malis_w * min(1.0, (self.current_epoch + 1) / warm)

    def _mat_weight(self):
        warm = max(1, self.recipe.mat_warm_epochs)
        return self.recipe.mat_w * min(1.0, (self.current_epoch + 1) / warm)

    def _orient(self, ct):
        with torch.no_grad():
            return ct_orientation(ct, sigma_grad=1.0, sigma_tensor=3.0).to(ct.dtype)

    # ---- validation cross-section viz: CT ("the sheet") with prev / output / label / material-gate each
    #      alpha-blended in ITS OWN column (one thing per panel), N z-cross-section slices per window. This is
    #      the safety check that growth stays ON MATERIAL and does not hallucinate sheets in air. ----
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_viz_n = 0
        self._vgrp = {}

    def on_validation_epoch_end(self, val_outputs):
        self._print_grp("val", getattr(self, "_vgrp", None))         # our/m7 + cold/warm val Dice: is it learning?
        super().on_validation_epoch_end(val_outputs)

    def _save_val_viz(self, ct, prev, out_prob, label, mat, case_id):
        """ct/prev/out_prob/label/mat: (Z,Y,X) numpy (label may hold 0/1/ignore). One JPEG, slices along Y."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = os.path.join(self.output_folder, "val_viz", f"ep{self.current_epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)
        Z, Y, X = ct.shape
        ns = min(self.val_viz_slices, Y)
        ys = np.linspace(Y * 0.1, Y * 0.9, ns).astype(int)
        lo, hi = np.percentile(ct, [2, 98])
        base = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)          # CT display normalization
        lab = (label == 1).astype(np.float32)                       # surface-GT (ignore/​bg -> 0)
        cols = [("CT", None, None), ("prev", prev, "spring"), ("output", out_prob, "hot"),
                ("label", lab, "winter"), ("gate m(CT)", mat, "cool")]
        fig, ax = plt.subplots(ns, len(cols), figsize=(2.2 * len(cols), 2.2 * ns), squeeze=False)
        for r, y in enumerate(ys):
            b = base[:, y, :]                                       # (Z,X) cross-section
            for c, (name, ov, cm) in enumerate(cols):
                a = ax[r][c]; a.imshow(b, cmap="gray", vmin=0, vmax=1, aspect="auto")
                if ov is not None:
                    o = np.clip(ov[:, y, :], 0, 1)
                    a.imshow(o, cmap=cm, vmin=0, vmax=1, alpha=(o * 0.75), aspect="auto")
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(name, fontsize=9)
                if c == 0:
                    a.set_ylabel(f"y={y}", fontsize=7)
        fig.suptitle(f"ep{self.current_epoch}  {case_id}  (tau={self.mat_tau} s={self.mat_scale})", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        p = os.path.join(out_dir, f"{case_id}.jpg")
        fig.savefig(p, dpi=85, format="jpg"); plt.close(fig)
        return p

    def _maybe_val_viz(self, ct, prev, seg_out, surf_target, keys):
        # VAL-sourced (fixed val windows, comparable across epochs; val has no augmentation by design).
        if not self.val_viz or self._val_viz_n >= self.val_viz_images:
            return
        out = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
        prob = torch.softmax(out.float(), dim=1)[:, 1]                  # (B,Z,Y,X) surface prob
        lab = surf_target[0][:, 0]                                      # (B,Z,Y,X)
        mat = material_prob(ct[:, 0].float(), self.mat_tau, self.mat_scale)  # (B,Z,Y,X)
        cn = ct[:, 0].float().cpu().numpy(); pn = prev[:, 0].float().cpu().numpy()
        on = prob.cpu().numpy(); ln = lab.cpu().numpy(); mn = mat.cpu().numpy()
        nkeys = len(keys) if keys is not None else 0
        for b in range(cn.shape[0]):
            if self._val_viz_n >= self.val_viz_images:
                break
            cid = str(keys[b]) if b < nkeys else f"val{self._val_viz_n}"
            self._save_val_viz(cn[b], pn[b], on[b], ln[b], mn[b], cid)
            self._val_viz_n += 1

    def validation_step(self, batch: dict) -> dict:
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else [target.to(self.device, non_blocking=True)]
        surf_target = [t[:, 0:1] for t in target]                    # split off owner (ch1); keep surface (ch0)
        ct, prev = data[:, 0:1], data[:, 1:2]
        orient = self._orient(ct)
        with torch.no_grad(), self._ac():
            seg_out, aff = self.network(torch.cat([ct, prev, orient], dim=1))
            del aff
            l = self.loss(seg_out, surf_target)
        self._maybe_val_viz(ct, prev, seg_out, surf_target, batch.get("keys"))
        self._group_dice(seg_out, surf_target, prev, batch.get("keys"), self._vgrp)   # val per-group fg-Dice
        output = seg_out[0] if self.enable_deep_supervision else seg_out
        tgt = surf_target[0]
        axes = [0] + list(range(2, output.ndim))
        oseg = output.argmax(1)[:, None]
        pred = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
        pred.scatter_(1, oseg, 1)
        if self.label_manager.has_ignore_label:
            mask = (tgt != self.label_manager.ignore_label).float()
            tgt = tgt.clone(); tgt[tgt == self.label_manager.ignore_label] = 0
        else:
            mask = None
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred, tgt, axes=axes, mask=mask)
        return {"loss": l.detach().cpu().numpy(),
                "tp_hard": tp.detach().cpu().numpy()[1:], "fp_hard": fp.detach().cpu().numpy()[1:],
                "fn_hard": fn.detach().cpu().numpy()[1:]}

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else [target.to(self.device, non_blocking=True)]
        # split the 2-channel seg target: [surface, owner]
        surf_target = [t[:, 0:1] for t in target]                    # SepSkelSym expects (B,1,...)
        owner = target[0][:, 1].long()                                # full-res instance GT for MALIS

        self.optimizer.zero_grad(set_to_none=True)
        ct, prev = data[:, 0:1], data[:, 1:2]
        orient = self._orient(ct)

        p_dagger, maxk = self._dagger_params()
        if maxk > 0 and float(torch.rand(1).item()) < p_dagger:
            K = int(torch.randint(1, maxk + 1, (1,)).item())
            with torch.no_grad(), self._ac():
                for _ in range(K):
                    out = self.network(torch.cat([ct, prev, orient], dim=1))
                    seg = out[0] if isinstance(out, tuple) else out
                    seg = seg[0] if isinstance(seg, (list, tuple)) else seg
                    prev = torch.softmax(seg.float(), dim=1)[:, 1:2].to(data.dtype)
            prev = prev.detach()

        with self._ac():
            seg_out, aff = self.network(torch.cat([ct, prev, orient], dim=1))
            l_surf = self.loss(seg_out, surf_target)
            l = l_surf
            wm = self._malis_weight()
            l_mal = mgrow = mair = 0.0
            if wm > 0:
                l_mal_t = self.malis(aff, owner, rng=self._rng)       # malis casts to fp32 on CPU internally
                l = l + wm * l_mal_t
                l_mal = float(l_mal_t.detach())
            wt = self._mat_weight()
            if wt > 0:                                                # CT-material grow/air-suppress (v3 §4.3)
                l_mat, g_mat, a_mat = material_grow_loss(              # memory-lean: no fp32 aff copy
                    aff, ct, orient, self.offsets, tau=self.mat_tau, scale=self.mat_scale)
                l = l + wt * l_mat
                mgrow, mair = float(g_mat), float(a_mat)
            self._acc(float(l_surf.detach()), l_mal, self.malis.last_grow if wm > 0 else 0.0,
                      self.malis.last_sep if wm > 0 else 0.0, wm, mgrow, mair, wt)
        self._group_dice(seg_out, surf_target, prev, batch.get("keys"), self._grp)   # train per-group fg-Dice

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
