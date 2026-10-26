"""Tests for the confidence-weighted fragment prune (hercunet.meshes.extract.prune_sheet_masks)."""
import numpy as np
import torch

from hercunet.meshes.extract import prune_sheet_masks


def _window():
    """owner/gate/memb for a window with: a big confident sheet (1), a small UNCONFIDENT fragment (2 -> drop),
    and a small CONFIDENT thin sheet (3 -> keep via the conf guard). Background (0) is ungated."""
    owner = np.zeros(2000, np.int64)
    memb = np.zeros(2000, np.float32)
    gate = np.zeros(2000, bool)
    owner[:1000] = 1; memb[:1000] = 0.90; gate[:1000] = True        # big + confident  -> keep (large self-mass)
    owner[1000:1100] = 2; memb[1000:1100] = 0.40; gate[1000:1100] = True  # small + unconfident -> DROP
    owner[1100:1200] = 3; memb[1100:1200] = 0.85; gate[1100:1200] = True  # small + confident   -> keep (guard)
    return owner, gate, memb


def test_drops_small_unconfident_keeps_the_rest():
    owner, gate, memb = _window()
    g2 = prune_sheet_masks(owner, gate, memb, mass_frac=0.10, conf_thr=0.70).cpu().numpy()
    assert not g2[(owner == 2)].any()                 # fragment cleared
    assert g2[(owner == 1)].all()                     # big sheet kept
    assert g2[(owner == 3)].all()                     # confident thin sheet kept
    assert int(g2.sum()) == 1100                      # only sheet 2's 100 voxels removed


def test_confident_fragment_survives():
    """A small sheet below the mass cut but ABOVE the confidence threshold is not dropped."""
    owner = np.array([1] * 1000 + [2] * 50, np.int64)
    memb = np.array([0.9] * 1000 + [0.8] * 50, np.float32)
    gate = np.ones(1050, bool)
    g2 = prune_sheet_masks(owner, gate, memb, mass_frac=0.10, conf_thr=0.70).cpu().numpy()
    assert g2.all()                                   # conf 0.8 >= 0.70 -> kept despite tiny mass


def test_large_low_conf_sheet_survives():
    """A large sheet (above the mass cut) is kept even if its mean confidence is low."""
    owner = np.array([1] * 1000 + [2] * 900, np.int64)
    memb = np.array([0.9] * 1000 + [0.45] * 900, np.float32)   # sheet 2: big but low-conf
    gate = np.ones(1900, bool)
    g2 = prune_sheet_masks(owner, gate, memb, mass_frac=0.10, conf_thr=0.70).cpu().numpy()
    assert g2.all()                                   # self-mass 405 >= 0.10*900 -> not "small" -> kept


def test_empty_gate_noop():
    owner = np.zeros(100, np.int64)
    gate = np.zeros(100, bool)
    memb = np.zeros(100, np.float32)
    g2 = prune_sheet_masks(owner, gate, memb)
    assert not torch.as_tensor(g2).any()
