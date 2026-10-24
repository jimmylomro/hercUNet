"""HercuViT distillation — corpus-mode dispatch, scroll resolution, variant guard (CPU, no ckpt / no data).

The heavy end-to-end (build small/tiny/nano + the distilled-encoder byte-identical round-trip through
``build_hercuvit(distilled_encoder=…)``) needs the 480 MB Dinovol ckpt, so it is validated on a GPU pod /
locally, not in CI."""
import pytest

from hercunet.hercuvit.train import distill
from hercunet.hercuvit.train.recipe import VARIANTS


def test_resolve_scrolls_all_and_comma():
    from hercunet.hercunet.labels.m7mine import SCROLLS
    assert distill._resolve_scrolls(None) == list(SCROLLS)        # default = all mined scrolls
    assert distill._resolve_scrolls("all") == list(SCROLLS)
    assert distill._resolve_scrolls("PHerc1447, PHerc0800") == ["PHerc1447", "PHerc0800"]


def test_corpus_mode_requires_its_source():
    # each mode fails fast (SystemExit) when its source flag is missing; unknown mode too.
    with pytest.raises(SystemExit):
        list(distill._corpus_crops("herculabels", 1, 1, "cpu", corpus=None))
    with pytest.raises(SystemExit):
        list(distill._corpus_crops("m7", 1, 1, "cpu", manifest=None))
    with pytest.raises(SystemExit):
        list(distill._corpus_crops("bogus", 1, 1, "cpu"))


def test_distill_supports_all_variants_and_rejects_unknown():
    assert {"small", "tiny", "nano"}.issubset(set(VARIANTS))       # the distillable students (full = raw teacher, not distilled)
    with pytest.raises(ValueError):                                 # guard fires before any ckpt load
        distill.distill(teacher_ckpt="x", student_variant="bogus", steps=1, device="cpu")
