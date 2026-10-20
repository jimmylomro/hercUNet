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


# --------------------------------------------------------------------------- intermediate-block (hint) distillation
class HintProjections(nn.Module):
    """One student→teacher-width projection per hint layer (identity when widths match), for intermediate-feature
    (FitNets-style) distillation. Separate from :class:`DistillProjection` so each depth gets its own map."""

    def __init__(self, n_hints: int, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Identity() if student_dim == teacher_dim else nn.Linear(student_dim, teacher_dim)
            for _ in range(int(n_hints))
        ])

    def forward(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return self.projs[i](x)


def hint_block_pairs(teacher_depth: int, student_depth: int, fractions) -> list:
    """Depth-aligned hint layers: for each relative depth ``f`` in ``fractions`` (0,1), the ``(teacher_block,
    student_block)`` at the SAME fractional depth. Our transformer is single-scale (every block at 1/8), so hints
    align by DEPTH, not resolution — see ``docs/dinovol_sheet_architecture.md`` §13.7. ``f=1.0`` (the final layer)
    is already supervised by the ``x_norm_patchtokens`` term, so pass intermediate fractions only (e.g. 0.33, 0.66).
    De-duplicated, sorted by teacher depth."""
    pairs = {}
    for f in fractions:
        ti = max(0, min(int(teacher_depth) - 1, round(float(f) * teacher_depth) - 1))
        si = max(0, min(int(student_depth) - 1, round(float(f) * student_depth) - 1))
        pairs[ti] = si                                                   # de-dup on teacher block
    return sorted(pairs.items())


def hint_token_loss(student_hints, teacher_hints, projections: HintProjections, num_prefix: int) -> torch.Tensor:
    """Mean per-hint token-cosine between the student's and teacher's intermediate PATCH tokens (prefix stripped),
    the student projected to the teacher width. ``*_hints`` are lists of ``(B, prefix+N, C)`` block outputs."""
    if not teacher_hints:
        return teacher_hints[0].new_zeros(()) if teacher_hints else torch.zeros(())
    total = 0.0
    for r, (s, t) in enumerate(zip(student_hints, teacher_hints)):
        total = total + token_cosine_loss(projections(r, s[:, num_prefix:]), t[:, num_prefix:])
    return total / len(teacher_hints)
