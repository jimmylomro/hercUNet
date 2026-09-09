"""Window visualisation — the ``--visualise image`` cross-section montage.

Renders a grid of z cross-sections through a generated window: each panel is the cleaned CT slice in
grayscale with the streamlet sample points near that depth scattered on top, coloured by their probeom
cluster. This is the ``image`` mode of ``hercunet labels create --visualise``; the ``interactive`` live
viewer is a separate, not-yet-implemented mode.

Reproduces the research ``cluster25d`` volume montage for the point-cloud pipeline (one 8-D vector per
meshlet; points sample the meshlets), reusing the graph-coloured palette from ``common.scales`` so two
touching sheets can never share a colour.
"""

from __future__ import annotations

import os

import numpy as np


def render_window_montage(bced, pts, plab, org, voxel_um, out_dir, *,
                          scroll, level, coords, n_panels=16, band_px=1):
    """Write a cross-section montage PNG of ``pts`` coloured by ``plab`` over the ``bced`` CT slices.

    ``bced`` [Z,Y,X] cleaned density; ``pts`` [N,3] point cloud in (z,y,x); ``plab`` [N] cluster label
    (-1 = noise, not drawn); ``org`` window centre (zc,yc,xc); ``voxel_um`` scale; ``out_dir`` output
    directory. Returns the written path. ``n_panels`` z-slices are spread over the labelled point
    extent; each panel scatters points within ``band_px`` of that slice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hercunet.labels.common.scales import cluster_colors

    bced = np.asarray(bced, np.float32)
    Z = bced.shape[0]
    pts = np.asarray(pts, np.float32)
    plab = np.asarray(plab)
    zp = np.round(pts[:, 0]).astype(int)
    keep = plab >= 0

    # panels span the depth range that actually carries clustered points (fall back to full window)
    if keep.any():
        z0, z1 = int(zp[keep].min()), int(zp[keep].max())
    else:
        z0, z1 = 3, Z - 4
    z0, z1 = max(0, z0), min(Z - 1, max(z0 + 1, z1))
    shown = np.unique(np.linspace(z0, z1, n_panels).round().astype(int))

    cols = cluster_colors(plab)                                    # {label: RGBA}, noise -1 → grey
    ncol = 4 if len(shown) <= 16 else 5
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, z in zip(axes, shown):
        sl = bced[z]
        lo, hi = np.percentile(sl, [1, 99])
        ax.imshow(np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray")
        m = keep & (np.abs(zp - z) <= band_px)
        if m.any():
            c = [cols[int(l)] for l in plab[m]]
            ax.scatter(pts[m, 2], pts[m, 1], c=c, s=3, linewidths=0)
        nseen = len({int(l) for l in plab[m]}) if m.any() else 0
        ax.set_title(f"z={z} — {nseen} sheets", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(shown):]:
        ax.axis("off")

    nclust = len({int(l) for l in plab if l >= 0})
    fig.suptitle(f"{scroll} L{level} {coords} — 2.5-D clusters ({nclust} sheets); "
                 f"streamlet points coloured by cluster", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    os.makedirs(out_dir, exist_ok=True)
    zc, yc, xc = org
    path = os.path.join(out_dir, f"cluster_montage_s{scroll}_L{level}_z{zc}_y{yc}_x{xc}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"→ montage {path}  ({len(shown)} cross-sections, {nclust} sheets)", flush=True)
    return path
