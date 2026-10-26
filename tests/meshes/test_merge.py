"""Tests for the symmetric-confusion merge (hercunet.meshes.extract.merge_confused_sheets)."""
import numpy as np
import torch

from hercunet.meshes.extract import merge_confused_sheets

_L = np.log  # readable logit construction


def _logits(rows):
    """rows: list of (count, [softmax target per query]) -> mask_logits [Q, N] whose softmax matches the targets."""
    cols = []
    for n, probs in rows:
        cols.append(np.tile(np.array([_L(max(p, 1e-6)) for p in probs], np.float32)[:, None], (1, n)))
    return torch.as_tensor(np.concatenate(cols, axis=1))        # [Q, N]


def test_merges_symmetric_pair():
    """Sheets 1 and 2 mutually confuse (~0.4 each way, > 1/3); sheet 3 is confident. 1&2 merge into one."""
    Q = 4
    ml = _logits([(100, [0.075, 0.45, 0.40, 0.075]),      # owner 1, leaks 0.40 to query 2
                  (100, [0.075, 0.40, 0.45, 0.075]),      # owner 2, leaks 0.40 to query 1  -> mutual
                  (100, [0.03, 0.03, 0.04, 0.90])])       # owner 3, confident
    owner = ml.argmax(0)                                   # [300]
    gate = torch.ones(300, dtype=torch.bool)
    o2 = merge_confused_sheets(owner, gate, ml, thresh=1.0 / 3).cpu().numpy()
    assert set(np.unique(o2)) == {1, 3}                    # query 2 folded into 1 (tie -> larger id... equal mass)
    assert (o2 == 3).sum() == 100                          # sheet 3 untouched
    assert (o2 != 3).sum() == 200                          # 1 and 2 are now one sheet


def test_asymmetric_does_not_merge():
    """Sheet 1 leaks 0.40 to query 2, but sheet 2 is confident (only 0.05 back) -> not mutual -> no merge."""
    ml = _logits([(100, [0.075, 0.45, 0.40, 0.075]),      # 1 -> 2 is 0.40
                  (100, [0.025, 0.05, 0.90, 0.025]),      # 2 -> 1 is 0.05  (one-sided)
                  (100, [0.03, 0.03, 0.04, 0.90])])
    owner = ml.argmax(0)
    gate = torch.ones(300, dtype=torch.bool)
    o2 = merge_confused_sheets(owner, gate, ml, thresh=1.0 / 3).cpu().numpy()
    assert set(np.unique(o2)) == {1, 2, 3}                 # nothing merged


def test_single_sheet_noop():
    ml = _logits([(50, [0.02, 0.94, 0.02, 0.02])])
    owner = ml.argmax(0)
    gate = torch.ones(50, dtype=torch.bool)
    o2 = merge_confused_sheets(owner, gate, ml, thresh=1.0 / 3)
    assert set(torch.unique(o2).tolist()) == {1}
