"""Interactive label-editing operations for the viewer's split / merge / delete controls.

These live in the LABEL package — not the viewer — so an edit runs the SAME clustering + mesh code the
corpus pipeline uses (forced borderline split → relabel → medial-mesh refit). The viewer only supplies
the selected sheet id and drives its loading overlay; every cluster decision is made here.
"""

from __future__ import annotations

import numpy as np

from hercunet.labels.augment import augment as aug
from hercunet.labels.mesh.medial import fit_sheet_meshes


def build_edit_context(C, args, device):
    """Bundle the per-window arrays an interactive edit needs, from a fresh pipeline context ``C``. Only
    the SPARSE meshlet arrays + provenance — no dense ``bced``/``ff`` (the mesh refit ignores them; the CT
    background reaches the viewer separately via the ``brick`` payload). The mutable labelling
    (``plab``/``ulab``) is copied so edits don't disturb the base sample. Passed in-process to the viewer
    and back into the edit/save functions below."""
    shape = tuple(int(s) for s in np.asarray(C["bced"]).shape)
    ctx = dict(E=np.asarray(C["E"], np.float32), sid=np.asarray(C["sid"]),
               pts=np.asarray(C["pts"], np.float32), vu=float(C["vu"]), shape=shape,
               normals=np.asarray(C["normals"], np.float32), jac=np.asarray(C["jac"], np.float32),
               device=device, plab=np.asarray(C["plab"]).copy(), org=C["org"],
               prop_dist=np.asarray(C["prop_dist"], np.float32),
               scroll=args.scroll, coords=args.coords, level=args.level, run_id=args.run_id,
               conf_ds=args.conf_ds, gpu=bool(args.gpu))
    ctx["ulab"] = aug.per_unit_labels(ctx["sid"], ctx["plab"], n_units=len(ctx["E"]))
    return ctx


def edit_context_from_window(meta, meshlet, device):
    """Build the SAME edit context from a corpus window loaded off disk (``meta`` + ``meshlet`` cloud) —
    the ``edit`` command path, no pipeline re-run. Frames come back f16 from the bundle; cast to f32."""
    coords = ",".join(str(int(c)) for c in meta["coords"])
    ctx = dict(E=np.asarray(meshlet["E"], np.float32), sid=np.asarray(meshlet["sid"]),
               pts=np.asarray(meshlet["pts"], np.float32), vu=float(meshlet["vu"]),
               shape=tuple(int(s) for s in meshlet["shape"]),
               normals=np.asarray(meshlet["normals"], np.float32), jac=np.asarray(meshlet["jac"], np.float32),
               device=device, plab=np.asarray(meshlet["plab"]).copy(), org=meta.get("org"),
               prop_dist=np.asarray(meshlet["prop_dist"], np.float32),
               scroll=meta["scroll"], coords=coords, level=meta["level"], run_id=meta.get("run_id"),
               conf_ds=int(meta.get("conf_ds", 4)), gpu=bool(device == "cuda"))
    ctx["ulab"] = (np.asarray(meshlet["ulab"]) if "ulab" in meshlet
                   else aug.per_unit_labels(ctx["sid"], ctx["plab"], n_units=len(ctx["E"])))
    return ctx


def meshlet_from_ctx(ctx):
    """The meshlet cloud to persist for an edited window — the sparse arrays only (frames downcast to f16
    by the bundle writer). Inverse of :func:`edit_context_from_window`."""
    return dict(pts=ctx["pts"], sid=ctx["sid"], plab=ctx["plab"], normals=ctx["normals"], jac=ctx["jac"],
                prop_dist=ctx["prop_dist"], E=ctx["E"], ulab=ctx["ulab"], vu=ctx["vu"], shape=ctx["shape"])


def _edited_quality(base_quality, meshes):
    """Quality for an edited window (option (c)): the human vouched for it, so untouched sheet ids keep
    their base score and any NEW id (from a split/merge) is stamped ``"manual"``; the window score is the
    min over the numeric (auto) sheets, or ``"manual"`` if none remain."""
    base = base_quality.get("per_sheet", {}) if isinstance(base_quality, dict) else {}
    per = {str(int(c)): base.get(str(int(c)), "manual") for c in meshes}
    numeric = [v for v in per.values() if isinstance(v, (int, float))]
    return dict(window=(round(min(numeric), 4) if numeric else "manual"),
                method="mesh_field_congruence_v1+manual", threshold_ref=0.80, per_sheet=per)


def save_edited_window(ctx, meshes, base_meta):
    """Assemble the EDITED window for the corpus from ``ctx`` (the edited labelling) + the current
    ``meshes``, carrying provenance from ``base_meta``. Returns ``(meta, meshes, meshlet)`` — the caller
    (``Corpus.write_window``) persists it; no confidence is computed here (recomputed at export, and the
    delete low-confidence is implicit in ``plab == -1``). Marks the window ``edited``."""
    meta = {**base_meta, "edited": True, "kind": "base", "n_sheets": len(meshes),
            "quality": _edited_quality(base_meta.get("quality"), meshes)}
    return meta, meshes, meshlet_from_ctx(ctx)


def split_sheet(ctx, cid, progress=None):
    """Force-split sheet ``cid`` at its weakest embedding seam, refit the two halves' meshes, and UPDATE
    ``ctx`` in place (``plab``, ``ulab``) so subsequent edits compose. Returns
    ``(new_plab, {id: mesh, ...}, (kept_id, new_id))`` or ``None`` if the sheet can't be split.

    ID STABILITY (unlike the corpus split-augmentation's :func:`augment.relabel_split`, which mints two
    fresh ids and retires ``cid``): for interactive editing we KEEP ``cid`` for one half and create only
    ONE new id (``max(plab)+1``) for the split-off half. No other sheet's id ever changes, so undo (a
    whole-state restore) and re-edits stay unambiguous. Weakest-point cut = :func:`augment.forced_split`;
    mesh refit = :func:`mesh.medial.fit_sheet_meshes`. The optional ``progress`` hook lets the viewer
    update its overlay before the (slower) mesh refit."""
    cid = int(cid)
    res = aug.forced_split(ctx["E"], ctx["ulab"], cid)
    if res is None:
        return None
    unit_side, _sep = res                                             # bool over cid's units (ascending id); True = split-off
    plab, sid = np.asarray(ctx["plab"]), np.asarray(ctx["sid"])
    units = np.where(aug.per_unit_labels(sid, plab, n_units=len(ctx["E"])) == cid)[0]
    side_of_unit = {int(u): bool(s) for u, s in zip(units, unit_side)}
    new_id = int(plab.max()) + 1                                      # the ONLY new id; cid is kept for the other half
    new_plab = plab.copy()
    mask = new_plab == cid
    new_plab[mask] = np.array([new_id if side_of_unit.get(int(u), False) else cid for u in sid[mask]],
                              dtype=new_plab.dtype)
    if progress is not None:
        progress("split_status", "updating sheet meshes…")
    pl = np.where(np.isin(new_plab, (cid, new_id)), new_plab, -1)     # refit only the two halves
    meshes = fit_sheet_meshes(ctx["pts"], pl, None, None, ctx["vu"],
                              normals=ctx["normals"], jac=ctx["jac"], step_um=30.0,
                              device=ctx.get("device", "cpu"), verbose=False)
    ctx["plab"] = new_plab
    ctx["ulab"] = aug.per_unit_labels(ctx["sid"], new_plab, n_units=len(ctx["E"]))
    return new_plab, meshes, (cid, new_id)


def merge_sheets(ctx, cid_a, cid_b, progress=None):
    """Force-merge sheets ``cid_a`` and ``cid_b`` into one — keeping the id of the LARGER sheet (by point
    count), refit its mesh, UPDATE ``ctx`` in place. Returns ``(new_plab, {keep_id: mesh}, keep_id,
    drop_id)``. (Just relabels the smaller sheet's points to the larger's id — same as the corpus
    merge-augmentation's :func:`augment.relabel_merge_set`, then a medial refit of the union.)"""
    a, b = int(cid_a), int(cid_b)
    plab = np.asarray(ctx["plab"])
    keep, drop = (a, b) if int((plab == a).sum()) >= int((plab == b).sum()) else (b, a)
    new_plab = plab.copy()
    new_plab[new_plab == drop] = keep
    if progress is not None:
        progress("split_status", "updating sheet meshes…")
    pl = np.where(new_plab == keep, keep, -1)                         # refit only the merged sheet
    meshes = fit_sheet_meshes(ctx["pts"], pl, None, None, ctx["vu"],
                              normals=ctx["normals"], jac=ctx["jac"], step_um=30.0,
                              device=ctx.get("device", "cpu"), verbose=False)
    ctx["plab"] = new_plab
    ctx["ulab"] = aug.per_unit_labels(ctx["sid"], new_plab, n_units=len(ctx["E"]))
    return new_plab, meshes, keep, drop


def delete_sheets(ctx, cids, progress=None):
    """Delete sheets ``cids`` by marking their points as noise (label -1) AND flagging their footprint as
    a LOW-CONFIDENCE region (we deliberately removed structure there → that whole region is now
    untrustworthy). UPDATE ``ctx`` in place. Returns ``(new_plab, deleted_ids, region)`` where ``region``
    is a boolean volume (block shape) covering the deleted sheets' footprint. No mesh refit — noise has no
    sheet mesh."""
    from scipy import ndimage as ndi
    ids = [int(c) for c in cids]
    plab = np.asarray(ctx["plab"])
    shape = tuple(int(s) for s in ctx["shape"])
    sel = np.isin(plab, ids)
    p = np.round(np.asarray(ctx["pts"])[sel]).astype(int)             # the deleted sheets' points
    for a in range(3):
        p[:, a] = p[:, a].clip(0, shape[a] - 1)
    footprint = np.zeros(shape, bool)
    if len(p):
        footprint[p[:, 0], p[:, 1], p[:, 2]] = True
    region = ndi.binary_dilation(footprint, iterations=4)            # thicken the point samples into the region
    new_plab = plab.copy()
    for c in ids:
        new_plab[new_plab == c] = -1
    ctx["plab"] = new_plab
    ctx["ulab"] = aug.per_unit_labels(ctx["sid"], new_plab, n_units=len(ctx["E"]))
    return new_plab, ids, region
