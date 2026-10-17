"""Distillation losses for the slim variants (hercuvit-tiny / -nano) — design ``docs/hercuvit.md`` §8.1.

The student encoder (slim transformer, stem copied from the teacher) regresses to the frozen Dinovol teacher on
unlabelled 128³ crops. Three feature-space terms, each width-agnostic or projected so a narrower student (nano,
embed 192) can still be compared to the 288-d teacher:

- **token cosine** — per-patch cosine between the student's tokens (projected to the teacher width) and the
  teacher's tokens at the same positions;
- **Gram** (DINOv3 Gram anchoring) — match the student's pairwise patch-similarity matrix to the teacher's; this
  is dimensionless, so it works across widths and preserves the patch-relationship structure sheet-ID needs;
- **cls cosine** — same on the cls token (projected).

No labels. The teacher runs full global attention; the student runs its *final* attention pattern (windowed +
spatial-reduction) during distillation so it learns to compensate (design §8.1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DistillProjection(nn.Module):
    """Projects student features to the teacher width for the cosine terms (identity when widths match)."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Identity() if student_dim == teacher_dim else nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def token_cosine_loss(student_proj: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
    """1 − cosine between projected student tokens and teacher tokens. Both ``(B, G, C)``."""
    return (1.0 - F.cosine_similarity(student_proj.float(), teacher_tokens.float(), dim=-1)).mean()


def gram_loss(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor, *, n_sample: int = 512) -> torch.Tensor:
    """MSE between the student's and teacher's pairwise patch-cosine (Gram) matrices.

    Width-agnostic (the Gram matrices are ``(B, n, n)`` regardless of channel count). ``n_sample`` random tokens
    are used per step to keep the ``n×n`` matrix cheap (full 4096² would be ~67 MB/matrix)."""
    B, G, _ = student_tokens.shape
    n = min(int(n_sample), G)
    idx = torch.randperm(G, device=student_tokens.device)[:n]
    s = F.normalize(student_tokens[:, idx].float(), dim=-1)
    t = F.normalize(teacher_tokens[:, idx].float(), dim=-1)
    gs = s @ s.transpose(1, 2)
    gt = t @ t.transpose(1, 2)
    return F.mse_loss(gs, gt)


def cls_cosine_loss(student_cls_proj: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
    """1 − cosine on the cls token. Both ``(B, C)`` (or ``(B, 1, C)``)."""
    s = student_cls_proj.float().flatten(1)
    t = teacher_cls.float().flatten(1)
    return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()


def distill_step_losses(
    student_feats: dict,
    teacher_feats: dict,
    projection: DistillProjection,
    *,
    token_weight: float = 1.0,
    gram_weight: float = 1.0,
    cls_weight: float = 0.5,
    gram_n_sample: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Combine the three terms from student/teacher ``forward_features`` dicts. Returns ``(loss, terms)``.

    Uses ``x_norm_patchtokens`` for token/Gram; ``x_norm_clstoken`` for the cls term when both expose it.
    """
    s_tok = student_feats["x_norm_patchtokens"]
    t_tok = teacher_feats["x_norm_patchtokens"]
    s_proj = projection(s_tok)
    terms: dict[str, float | torch.Tensor] = {}
    loss = s_tok.new_zeros(())

    l_tok = token_cosine_loss(s_proj, t_tok)
    loss = loss + token_weight * l_tok
    terms["token"] = float(l_tok.detach())

    l_gram = gram_loss(s_tok, t_tok, n_sample=gram_n_sample)
    loss = loss + gram_weight * l_gram
    terms["gram"] = float(l_gram.detach())

    s_cls = student_feats.get("x_norm_clstoken")
    t_cls = teacher_feats.get("x_norm_clstoken")
    if cls_weight > 0 and s_cls is not None and t_cls is not None:
        l_cls = cls_cosine_loss(projection(s_cls), t_cls)
        loss = loss + cls_weight * l_cls
        terms["cls"] = float(l_cls.detach())

    terms["total"] = float(loss.detach())
    return loss, terms
