"""``hercunet train fit`` — launch a HercUNet trainer by DIRECT INSTANTIATION.

A faithful reduction of nnunetv2 2.8.1 ``run/run_training.py::run_training``, with the
string->class discovery (``recursive_find_trainer_class_by_name``) replaced by importing the
class. So trainers live in ``hercunet.training.trainers`` and NOTHING is copied into the
nnunetv2 package, no ``nnUNet_extTrainer`` env var, no fork.

Trainer selection (``--trainer``): a short alias from ``ALIASES`` below, or an explicit
``module:ClassName`` dotted path (handy before the alias target exists / for one-offs).
"""
from __future__ import annotations

import importlib
import os

# Short alias -> "module:ClassName". HercUNet is the AffinityMalis iterative-DAgger refiner (the
# run301 / v0 model). The target module lands as the trainer chain is ported into
# hercunet.training.trainers; until then, pass the class explicitly with --trainer module:ClassName.
ALIASES = {
    "hercunet": "hercunet.training.trainers.affinity_malis_iter_dagger:"
                "nnUNetTrainer_AffinityMalis_IterDagger_500epochs",
}


def resolve_trainer(spec: str):
    """``alias`` | ``module:ClassName`` -> the trainer class (imported, not discovered)."""
    target = ALIASES.get(spec, spec)
    if ":" not in target:
        raise SystemExit(
            f"hercunet train fit: unknown trainer {spec!r}. Use one of {sorted(ALIASES)} "
            f"or an explicit 'module:ClassName'.")
    mod_name, cls_name = target.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    if not (isinstance(cls, type) and issubclass(cls, nnUNetTrainer)):
        raise SystemExit(f"hercunet train fit: {target} is not an nnUNetTrainer subclass")
    return cls


def build_trainer(trainer_cls, dataset, configuration, fold, plans_identifier, device,
                  continue_training: bool = False):
    """= nnU-Net get_trainer_from_args, minus the string discovery (we hold the class)."""
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset))
    plans = load_json(join(base, plans_identifier + ".json"))
    plans["continue_training"] = continue_training
    dataset_json = load_json(join(base, "dataset.json"))
    return trainer_cls(plans=plans, configuration=configuration, fold=fold,
                       dataset_json=dataset_json, device=device)


def _train_one(trainer_cls, dataset, configuration, fold, plans_identifier, device,
               pretrained, continue_training, epochs=None):
    import torch

    trainer = build_trainer(trainer_cls, dataset, configuration, fold, plans_identifier, device,
                            continue_training)
    if epochs:                                          # training-length override (e.g. --epochs 1 for a smoke)
        trainer.num_epochs = int(epochs)
    if continue_training:
        _maybe_continue(trainer)
    elif pretrained:
        if not trainer.was_initialized:
            trainer.initialize()
        # Prefer the trainer's own smart warm-start (the HercUNet trainer stem-expands an m7-like checkpoint;
        # detected from the checkpoint itself, no env var). Fall back to nnU-Net's loader for stock trainers.
        if hasattr(trainer, "load_pretrained"):
            trainer.load_pretrained(pretrained)
        else:
            from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
            load_pretrained_weights(trainer.network, pretrained, verbose=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    trainer.run_training()
    return trainer


def _maybe_continue(trainer):
    from batchgenerators.utilities.file_and_folder_operations import isfile, join
    for name in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"):
        ckpt = join(trainer.output_folder, name)
        if isfile(ckpt):
            trainer.load_checkpoint(ckpt)
            return
    print("[fit] --continue: no checkpoint found, starting fresh")


def _ddp_worker(rank, world_size, trainer_spec, dataset, configuration, fold, plans_identifier,
                pretrained, continue_training, recipe_path, epochs):
    import torch
    import torch.distributed as dist
    from hercunet.training import recipe as _recipe
    _recipe.set_active(_recipe.load(recipe_path))       # each spawned worker re-installs the active recipe
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(torch.device("cuda", rank))
    try:
        _train_one(resolve_trainer(trainer_spec), dataset, configuration, fold, plans_identifier,
                   torch.device("cuda", rank), pretrained, continue_training, epochs)
    finally:
        dist.destroy_process_group()


def fit(trainer_spec: str, dataset, *, configuration: str = "3d_fullres", fold=0,
        plans_identifier: str = "nnUNetResEncUNetLPlans", pretrained: str | None = None,
        num_gpus: int = 1, device: str = "cuda", continue_training: bool = False,
        recipe_path: str | None = None, epochs: int | None = None):
    """Train ``trainer_spec`` on ``dataset``. Single-GPU or DDP (``num_gpus > 1``). ``recipe_path`` (a
    ``--recipe`` file) overrides the run301 recipe; ``epochs`` overrides the training length."""
    import torch
    from hercunet.training import recipe as _recipe
    _recipe.set_active(_recipe.load(recipe_path))       # run301 by default; a variant if --recipe given
    fold = fold if fold == "all" else int(fold)         # argparse hands a str; nnU-Net indexes splits by int
    if num_gpus > 1:
        if device != "cuda":
            raise SystemExit("hercunet train fit: DDP (--num-gpus > 1) requires --device cuda")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        import torch.multiprocessing as mp
        mp.spawn(_ddp_worker,
                 args=(num_gpus, trainer_spec, dataset, configuration, fold, plans_identifier,
                       pretrained, continue_training, recipe_path, epochs),
                 nprocs=num_gpus, join=True)
    else:
        _train_one(resolve_trainer(trainer_spec), dataset, configuration, fold, plans_identifier,
                   torch.device(device), pretrained, continue_training, epochs)


def _free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port
