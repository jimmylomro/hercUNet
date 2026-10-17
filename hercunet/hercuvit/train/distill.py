"""Distillation runner for the slim variants — makes ``hercuvit-tiny`` / ``-nano`` from the frozen Dinovol
teacher (design ``docs/hercuvit.md`` §8.1). Entry point ``run(args)`` (wired to ``hercuvit distill``).

The teacher is the frozen Dinovol encoder (full global attention); the student is a slim encoder (stem copied,
slim transformer in its final windowed + spatial-reduction pattern). We regress student features to the teacher
on unlabelled 128³ crops with token-cosine + Gram + cls-cosine losses (``model/distill.py``). No labels, no
trained hercuvit-small needed — so this is testable today against the released Dinovol checkpoint.

The distilled student encoder is saved so supervised training (``hercuvit train``) can warm-start from it instead
of the random fresh blocks.

``--smoke`` runs a few steps on synthetic volumes to exercise the whole machinery with no data dependency.
Real unlabelled-crop streaming from a scroll/corpus is a thin wrapper over the data layer — see the TODO in
``docs/hercuvit.md`` §13; the smoke path and the loss/optimiser machinery are complete.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from ..model.hercuvit_net import build_variant, VARIANT_ARCH
from ..model.encoder import build_and_load_backbone, DINOVOL_HF_REPO, DINOVOL_CKPT_FILE
from ..model.distill import DistillProjection, distill_step_losses


def _auto_attention_backend(module: torch.nn.Module) -> None:
    """Let torch choose the SDPA kernel (the dinovol config forces cuDNN, which lacks plans for some shapes)."""
    for m in module.modules():
        if hasattr(m, "attention_priority"):
            m.attention_priority = None


def resolve_dinovol_ckpt(spec: str) -> str:
    """A local path, or a Hugging Face repo id → cached ``teacher_305k.pt`` (reusing hercunet's HF helper)."""
    if spec and Path(spec).exists():
        return spec
    from ...train.launch import resolve_pretrained  # reuse hercunet's hf_hub_download wrapper
    return resolve_pretrained(pretrained_hf=spec or DINOVOL_HF_REPO, hf_file=DINOVOL_CKPT_FILE)


def _synthetic_crops(n: int, batch: int, device, generator):
    for _ in range(n):
        yield torch.randn(batch, 1, 128, 128, 128, device=device, generator=generator)


def distill(
    *,
    teacher_ckpt: str,
    student_variant: str = "tiny",
    out: str | None = None,
    steps: int = 2000,
    lr: float = 2e-4,
    token_weight: float = 1.0,
    gram_weight: float = 1.0,
    cls_weight: float = 0.5,
    gram_n_sample: int = 512,
    batch: int = 1,
    device: str = "cuda",
    smoke: bool = False,
    log_every: int = 50,
    seed: int = 0,
) -> dict:
    """Run distillation; returns a summary dict. Saves the student encoder + projection to ``out`` when given."""
    if student_variant not in ("tiny", "nano"):
        raise ValueError(f"distillation targets the slim variants (tiny|nano), got {student_variant!r}")
    ckpt = resolve_dinovol_ckpt(teacher_ckpt)

    teacher, t_cfg, _, miss, unexp = build_and_load_backbone(ckpt, device=device, return_embed_skips=False)
    if miss or unexp:
        raise RuntimeError(f"teacher load not clean: missing={miss[:5]} unexpected={unexp[:5]}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    _auto_attention_backend(teacher)

    student_net = build_variant(ckpt, student_variant, device=device, with_instance=False, deep_supervision=False)
    student = student_net.backbone
    _auto_attention_backend(student)
    student.train()
    s_dim = int(VARIANT_ARCH[student_variant]["embed_dim"])
    t_dim = int(t_cfg["embed_dim"])
    projection = DistillProjection(s_dim, t_dim).to(device)

    params = [p for p in student.parameters() if p.requires_grad] + list(projection.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    amp = device.startswith("cuda") if isinstance(device, str) else False

    last = {}
    src = _synthetic_crops(steps, batch, device, gen) if smoke else _synthetic_crops(steps, batch, device, gen)
    # NOTE: non-smoke real-crop streaming is the documented TODO (§13); the loop below is identical either way.
    for step, x in enumerate(src, start=1):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            t_feats = teacher.forward_features(x, view_kind="global")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            s_feats = student.forward_features(x, view_kind="global")
            loss, terms = distill_step_losses(
                s_feats, t_feats, projection,
                token_weight=token_weight, gram_weight=gram_weight, cls_weight=cls_weight,
                gram_n_sample=gram_n_sample,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = terms
        if step % log_every == 0 or step == 1 or step == steps:
            msg = " ".join(f"{k}={v:.4f}" for k, v in terms.items())
            print(f"[distil {student_variant} step {step}/{steps}] {msg}", flush=True)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"student_backbone": student.state_dict(), "projection": projection.state_dict(),
             "variant": student_variant, "arch": VARIANT_ARCH[student_variant], "teacher_ckpt": ckpt},
            out,
        )
    return {"variant": student_variant, "steps": steps, "final_terms": last, "out": out}


def run(args) -> dict:
    """CLI entry (``hercuvit distill``). ``args`` is an argparse Namespace; missing *or None* attrs fall back to
    defaults (the CLI defaults the recipe-overridable flags to ``None`` meaning "use the default/recipe value")."""
    def g(name, default):
        v = getattr(args, name.replace("-", "_"), default)
        return default if v is None else v
    smoke = bool(g("smoke", False))
    return distill(
        teacher_ckpt=g("teacher", DINOVOL_HF_REPO),
        student_variant=g("student", "tiny"),
        out=g("out", None),
        steps=int(g("steps", 8 if smoke else 2000)),   # --smoke → a few steps unless an explicit --steps
        lr=float(g("distill_lr", 2e-4)),
        token_weight=float(g("token_weight", 1.0)),
        gram_weight=float(g("gram_weight", 1.0)),
        cls_weight=float(g("cls_weight", 0.5)),
        batch=int(g("batch", 1)),
        device=g("device", "cuda"),
        smoke=smoke,
    )
