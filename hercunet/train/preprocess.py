"""``hercunet train preprocess`` — fingerprint -> transplant m7's ResEncUNetL plans -> preprocess.

Wraps three stock nnU-Net 2.8.1 library functions in-process (no CLI, no copy-paste):

    nnunetv2.experiment_planning.plan_and_preprocess_api.extract_fingerprints
    nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets.move_plans_between_datasets
    nnunetv2.experiment_planning.plan_and_preprocess_api.preprocess

Mirrors ``v3_chain.sh`` STEP3: seed a ``Dataset100`` "plans source" from the published m7
checkpoint, transplant its plans onto each target dataset (so we train in m7's exact geometry),
then preprocess that configuration. Requires the nnU-Net env roots (``nnUNet_raw`` /
``nnUNet_preprocessed`` / ``nnUNet_results``) to be set, exactly as the CLI path does.
"""
from __future__ import annotations

import glob
import os
import shutil

M7_REPO = "scrollprize/surface_m7_nnunet"
PLANS_ID = "nnUNetResEncUNetLPlans"
PLANS_SOURCE_ID = 100
PLANS_SOURCE_NAME = "Dataset100_VesuviusSurface"


def setup_m7_plans_source(m7_repo: str = M7_REPO, plans_id: str = PLANS_ID) -> str:
    """Create ``Dataset100`` under ``$nnUNet_preprocessed`` holding m7's plans + dataset.json +
    fingerprint, so ``move_plans_between_datasets`` can transplant them. Idempotent."""
    from huggingface_hub import snapshot_download
    from nnunetv2.paths import nnUNet_preprocessed

    m7 = snapshot_download(m7_repo)
    d100 = os.path.join(nnUNet_preprocessed, PLANS_SOURCE_NAME)
    os.makedirs(d100, exist_ok=True)
    shutil.copyfile(os.path.join(m7, "plans.json"), os.path.join(d100, plans_id + ".json"))
    for fn in ("dataset.json", "dataset_fingerprint.json"):
        shutil.copyfile(os.path.join(m7, fn), os.path.join(d100, fn))
    print(f"[preprocess] m7 plans source -> {d100}")
    return d100


def patch_plans_prev_channels(plans_path: str, n_total: int, prev_norm: str = "NoNormalization") -> None:
    """Grow every configuration's ``normalization_schemes`` / ``use_mask_for_norm`` to ``n_total``
    input channels: channel 0 keeps its CT scheme, channels 1..n-1 (the ∇φ prev / candidate crests)
    get ``NoNormalization`` (already in [0,1]; the -1 sentinel must survive). Idempotent — only
    extends a length-1 list. Ported verbatim from the research pipeline (stdlib only)."""
    import json
    with open(plans_path) as f:
        plans = json.load(f)
    for cfg in plans.get("configurations", {}).values():
        ns = cfg.get("normalization_schemes")
        if ns is not None and len(ns) == 1:
            cfg["normalization_schemes"] = [ns[0]] + [prev_norm] * (n_total - 1)
        um = cfg.get("use_mask_for_norm")
        if um is not None and len(um) == 1:
            cfg["use_mask_for_norm"] = [bool(um[0])] + [False] * (n_total - 1)
    with open(plans_path, "w") as f:
        json.dump(plans, f, indent=4)


def preprocess_dataset(dataset_id: int, *, configuration: str = "3d_fullres",
                       plans_id: str = PLANS_ID, num_processes: int = 8, max_candidates: int = 4,
                       setup_plans: bool = True, m7_repo: str = M7_REPO) -> None:
    """Fingerprint -> transplant m7 plans -> patch to ``1+max_candidates`` input channels ->
    preprocess ``dataset_id`` in ``configuration`` (the HercUNet iterative refiner recipe)."""
    from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess
    from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import (
        move_plans_between_datasets)

    if setup_plans:
        setup_m7_plans_source(m7_repo=m7_repo, plans_id=plans_id)

    name = maybe_convert_to_dataset_name(dataset_id)
    print(f"[preprocess] {name}: extract_fingerprint")
    extract_fingerprints([dataset_id], num_processes=num_processes, check_dataset_integrity=False)

    print(f"[preprocess] {name}: move plans {PLANS_SOURCE_ID} -> {dataset_id} ({plans_id})")
    move_plans_between_datasets(PLANS_SOURCE_ID, dataset_id, plans_id, plans_id)

    pre_dir = os.path.join(nnUNet_preprocessed, name)
    os.makedirs(pre_dir, exist_ok=True)
    if max_candidates > 0:                              # HercUNet: CT + K candidate/prev crests
        n_total = 1 + max_candidates
        print(f"[preprocess] {name}: patch plans -> {n_total} input channels")
        patch_plans_prev_channels(os.path.join(pre_dir, plans_id + ".json"), n_total)

    # nnU-Net's preprocess reads dataset.json from the PREPROCESSED folder; carry it from raw.
    raw_dir = _dataset_dir(nnUNet_raw, name)
    shutil.copyfile(os.path.join(raw_dir, "dataset.json"), os.path.join(pre_dir, "dataset.json"))

    print(f"[preprocess] {name}: preprocess {configuration}")
    preprocess([dataset_id], plans_identifier=plans_id, configurations=(configuration,),
               num_processes=num_processes)
    print(f"[preprocess] {name}: done")


def _dataset_dir(root: str, name: str) -> str:
    exact = os.path.join(root, name)
    if os.path.isdir(exact):
        return exact
    # tolerate id-only ("Dataset221") -> resolve the DatasetXXX_* folder
    hits = glob.glob(os.path.join(root, name.split("_")[0] + "*"))
    if not hits:
        raise FileNotFoundError(f"no dataset folder for {name!r} under {root}")
    return hits[0]
