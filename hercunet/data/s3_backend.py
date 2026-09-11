"""Backend over the Vesuvius Challenge S3 open-data bucket.

The bucket is the authoritative inventory: it lists every scanned scroll by PHerc id (~45),
each with a ``volumes/`` folder (raw µCT) and often a ``segments/`` folder (unwrapped
surface volumes). This backend enumerates all of them and opens volumes/segments via the
robust :class:`ZarrSegment` reader (vesuvius's own opener fails on these HTTP zarr groups).

Naming: the five worked scrolls map onto friendly names (Paris4 → Scroll 1, …); the rest
are shown by prettified PHerc id. Nothing is filtered out — a scroll with no segments, or a
store that fails to open, is still listed and reports its state rather than disappearing.
"""

from __future__ import annotations

import re

from ..config import Config
from .backend import SegmentSource
from .segment import SegmentVolume
from .types import ScrollInfo, SegmentInfo
from .zarr_reader import ZarrSegment, parse_voxel_um

_BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

_PHERC_TO_SCROLL = {
    "PHercParis4": "1",
    "PHercParis3": "2",
    "PHerc0332": "3",
    "PHerc1667": "4",
    "PHerc0172": "5",
}

# Triage categories for the scroll list (curated from challenge status — easily edited).
#   "read"          -> text recovered (green)
#   "ink"           -> extensive published ink labels, not fully read (blue)
#   "first_letters" -> the 13 Grand-Prize / First-Letters-eligible scrolls, text not yet
#                      found (faint red) — these are the ones to focus on
#   ""              -> everything else (fragments, variants) — no tint
_READ_SCROLLS = {"1", "4"}
_INK_LABEL_SCROLLS = {"5"}
# The 2027 Grand Prize / First Letters eligible scrolls (scrollprize.org/prizes), by PHerc id.
_FIRST_LETTERS_SCROLLS = {
    "PHerc0125", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268", "PHerc0358",
    "PHerc0800", "PHerc0813", "PHerc0826", "PHerc1203", "PHerc1218", "PHerc1447", "PHerc1545",
}


def _scroll_category(sid: str, pherc: str) -> str:
    if sid in _READ_SCROLLS:
        return "read"
    if sid in _INK_LABEL_SCROLLS:
        return "ink"
    if pherc in _FIRST_LETTERS_SCROLLS:
        return "first_letters"
    return ""

# Curated (status, notes) keyed by canonical scroll id, from the project brief.
_SCROLL_FACTS = {
    "1": ("read", "First scroll read — 2023 Grand Prize; ink became directly visible in "
                  "June 2026 high-res rescans. 250 published segments."),
    "4": ("read", "First scroll unwrapped and read end-to-end (June 2026) — a Stoic ethics "
                  "treatise referencing Aristocreon."),
    "5": ("partially read", "Unusually legible (likely lead-rich). First title from a "
                            "still-rolled scroll: Philodemus, On Vices, Book 1."),
    "2": ("unread", "Scanned; not yet read."),
    "3": ("unread", "Scanned; not yet read."),
}


class S3Backend(SegmentSource):
    def __init__(self, config: Config):
        self._config = config
        self._scrolls: list[ScrollInfo] | None = None
        self._aligned_cache: dict = {}  # pherc -> (volume_url, voxel) | None
        self.refresh_note = ""

    # ---- listing --------------------------------------------------------------------
    def _list_prefixes(self, prefix: str = "") -> list[str]:
        """Immediate sub-prefixes (folders) under ``prefix``. Retries transient failures so a
        one-off network hiccup isn't mistaken for 'no data' (which surfaced as spurious
        'no zarr found' errors when opening segments)."""
        import requests

        url = f"{_BUCKET}/?list-type=2&delimiter=/&prefix={prefix}"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    return re.findall(r"<Prefix>([^<]+)</Prefix>", r.text)
            except requests.RequestException:
                pass
        return []

    def list_scrolls(self) -> list[ScrollInfo]:
        if self._scrolls is None:
            self._scrolls = self._build_scrolls()
        return self._scrolls

    def _finest_voxel(self, pherc: str) -> float | None:
        """Finest (smallest) voxel size in µm among this scroll's volumes, parsed from their names.
        Drives the resolution shown in the list — the ceiling on what any method can resolve here."""
        try:
            vols = self._list_prefixes(f"{pherc}/volumes/")
        except Exception:
            return None
        voxs = [v for v in (parse_voxel_um(u) for u in vols) if v]
        return min(voxs) if voxs else None

    def _build_scrolls(self) -> list[ScrollInfo]:
        scrolls: list[ScrollInfo] = []
        phercs = [pfx.rstrip("/").split("/")[-1] for pfx in self._list_prefixes("")]
        phercs = [p for p in phercs if p.startswith("PHerc")]
        # finest scan resolution per scroll, in parallel — one S3 listing each, so the list build
        # stays quick; shown in the name so the resolution ceiling is visible before opening.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as ex:
            finest = dict(zip(phercs, ex.map(self._finest_voxel, phercs)))
        for pherc in phercs:
            sid = _PHERC_TO_SCROLL.get(pherc, pherc)
            category = _scroll_category(sid, pherc)
            status, notes = _SCROLL_FACTS.get(sid, (None, None))
            if category == "first_letters":
                status = status or "unread — First Letters open"
                notes = notes or (
                    f"🎯 First Letters target ($50k): {_prettify(pherc)}, "
                    "Grand-Prize-eligible, text not yet found."
                )
            v = finest.get(pherc)
            base = _scroll_name(pherc, sid)
            scrolls.append(
                ScrollInfo(
                    scroll_id=sid,
                    name=f"{base} ({v:g}µm)" if v else base,
                    status=status,
                    notes=notes or f"Raw scan ({_prettify(pherc)}). Open to browse the volume.",
                    category=category,
                    resolution_um=v,
                    extra={"pherc": pherc},
                )
            )
        from .fragments import fragment_catalog
        for row in fragment_catalog():
            scrolls.append(ScrollInfo(
                scroll_id=row["id"],
                name=row["name"],
                resolution_um=row["voxel_um"],
                status="labelled fragment",
                notes="Detached fragment with IR-derived ink GROUND TRUTH: exposed-surface CT "
                      "render + human ink mask. Open, then toggle the ink overlay to see the "
                      "crackle where the label says ink.",
                category="ink",
                extra={"fragment": row},
            ))
        scrolls.sort(key=_sort_key)
        self.refresh_note = f"{len(scrolls)} scrolls + fragments (S3 / dl.ash2txt)"
        return scrolls

    def list_segments(self, scroll: ScrollInfo) -> list[SegmentInfo]:
        if scroll.extra.get("fragment") is not None:
            return []  # a fragment IS the surface; its ink label loads as the overlay
        pherc = scroll.extra["pherc"]
        prefixes = [
            p for p in self._list_prefixes(f"{pherc}/segments/")
            if p.rstrip("/").split("/")[-1] not in (pherc, "segments")
        ]
        # Whether a segment is *viewable* is authoritative only from the data: does it have a
        # rendered surface-volume zarr? (The id — even "auto_grown" — does NOT tell us; some
        # auto-grown segments are rendered, many are mesh-only.) Probe concurrently so the
        # extra check per segment doesn't serialise into a slow listing.
        status = self._probe_renderable(prefixes)
        segments = [
            SegmentInfo(
                segment_id=p.rstrip("/").split("/")[-1],
                scroll=scroll,
                resolution_um=scroll.resolution_um or 7.91,
                extra={
                    "seg_prefix": f"{_BUCKET}/{p}",
                    "renderable": status.get(p, (True, False))[0],
                    "has_ink": status.get(p, (True, False))[1],
                },
            )
            for p in prefixes
        ]
        segments.sort(key=lambda s: s.segment_id)
        return segments

    def _probe_renderable(self, prefixes: list[str]) -> dict[str, tuple[bool, bool]]:
        """Map each segment prefix -> (is viewable, has an ink-detection prediction), in parallel."""
        if not prefixes:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        def status(p: str) -> tuple[bool, bool]:
            root = p.rstrip("/")
            subs = {x.rstrip("/").split("/")[-1] for x in self._list_prefixes(root + "/")}
            renderable = bool(
                self._zarr_dirs_under(root + "/surface-volumes/")
                or self._zarr_dirs_under(root + "/")
            )
            return renderable, ("ink-detection" in subs)

        with ThreadPoolExecutor(max_workers=min(16, len(prefixes))) as ex:
            return dict(zip(prefixes, ex.map(status, prefixes)))

    def see_ink(self, segment: SegmentInfo, cache_dir: str | None = None):
        """Open the finest scroll volume that contains this segment, and its ink prediction
        mapped into that volume's voxels. Returns ``(volume, points)`` or None. The volume and
        the points share the same (aligned) coordinate frame, so the overlay lands on the sheet.
        """
        import os

        import numpy as np

        from .ink_projection import choose_aligned_scan, project_segment_ink

        seg_prefix = segment.extra["seg_prefix"]
        best = choose_aligned_scan(seg_prefix)
        if best is None:
            return None
        scan_id, vol_url, voxel = best

        pts, cache_file = None, None
        if cache_dir:
            try:
                d = os.path.join(str(cache_dir), "projections")
                os.makedirs(d, exist_ok=True)
                cache_file = os.path.join(d, f"{segment.segment_id}_{scan_id}_v2.npy")
                if os.path.exists(cache_file):
                    pts = np.load(cache_file)
            except OSError:
                cache_file = None
        if pts is None:
            pts = project_segment_ink(seg_prefix, scan_id)
            if pts is not None and cache_file:
                try:
                    np.save(cache_file, pts)
                except OSError:
                    pass
        if pts is None:
            return None
        volume = ZarrSegment(vol_url, voxel, voxel_override=voxel)
        return volume, pts

    def project_fragment_ink(self, scroll: ScrollInfo, cache_dir: str | None = None):
        """Back-project a labelled fragment's IR ink mask into its raw-CT voxels (via the
        fragment's ``result.ppm``). Returns (N,4) [z,y,x,ink] in the CT voxel grid, or None.
        Cached to disk, so the heavy 2.5 GB ppm range-read happens once per fragment."""
        frag = scroll.extra.get("fragment")
        if frag is None:
            return None
        from .fragments import project_fragment_ink

        return project_fragment_ink(frag, cache_dir)

    def load_fragment_ir(self, scroll: ScrollInfo):
        """The fragment's rendered IR scan (ink-visible ground truth), or None for non-fragments."""
        frag = scroll.extra.get("fragment")
        if frag is None:
            return None
        from .fragments import load_fragment_ir

        return load_fragment_ir(frag)

    # ---- opening --------------------------------------------------------------------
    def open_scroll_volume(self, scroll: ScrollInfo) -> SegmentVolume:
        frag = scroll.extra.get("fragment")
        if frag is not None:
            from .fragments import open_fragment_ct
            return open_fragment_ct(frag)
        pherc = scroll.extra["pherc"]
        # Prefer the finest volume that actually CONTAINS this scroll's segments, so the raw
        # view, ink back-projection, and the depth-link line all share one aligned frame.
        aligned = self._aligned_display_volume(pherc)
        if aligned is not None:
            url, voxel = aligned
            try:
                return ZarrSegment(url, voxel, voxel_override=voxel)
            except Exception:
                pass
        candidates = self._zarr_dirs_under(f"{pherc}/volumes/")
        if not candidates:
            raise ValueError(f"no volume zarr found for {pherc}")
        return _open_first(candidates)

    def _aligned_display_volume(self, pherc: str):
        """(volume_url, voxel_um) of the finest volume containing a segment, or None. Memoised."""
        if pherc in self._aligned_cache:
            return self._aligned_cache[pherc]
        result = self._compute_aligned_display_volume(pherc)
        self._aligned_cache[pherc] = result
        return result

    def _compute_aligned_display_volume(self, pherc: str):
        from .ink_projection import choose_aligned_scan

        prefixes = [
            p for p in self._list_prefixes(f"{pherc}/segments/")
            if p.rstrip("/").split("/")[-1] not in (pherc, "segments")
        ]
        for p in prefixes[:6]:  # first segment whose mesh fits a volume wins
            try:
                best = choose_aligned_scan(f"{_BUCKET}/{p}")
            except Exception:
                best = None
            if best is not None:
                return best[1], best[2]
        return None

    def load_segment_xyz(self, segment: SegmentInfo, scan_id: str):
        """(x, y, z) per-surface-pixel scroll-voxel maps for this segment, cached to disk so a
        re-open (and the depth-line + endpoint markers, which share it) is instant."""
        import os

        import numpy as np

        from .ink_projection import load_xyz

        cache_file = None
        cache_dir = self._config.disk_cache_dir
        if cache_dir:
            try:
                d = os.path.join(str(cache_dir), "meshes")
                os.makedirs(d, exist_ok=True)
                cache_file = os.path.join(d, f"{segment.segment_id}_{scan_id}_xyz.npz")
                if os.path.exists(cache_file):
                    dat = np.load(cache_file)
                    return dat["x"], dat["y"], dat["z"]
            except OSError:
                cache_file = None
        xyz = load_xyz(segment.extra["seg_prefix"], scan_id)
        if xyz is not None and cache_file:
            try:
                tmp = f"{cache_file}.{os.getpid()}.tmp.npz"  # .npz so np.savez won't re-append
                np.savez(tmp, x=xyz[0], y=xyz[1], z=xyz[2])
                os.replace(tmp, cache_file)
            except OSError:
                pass
        return xyz

    def load_segment_zmap(self, segment: SegmentInfo, scan_id: str):
        """Per-surface-pixel scroll-z map for the depth-link line (the z of the shared xyz)."""
        xyz = self.load_segment_xyz(segment, scan_id)
        return None if xyz is None else xyz[2]

    def open_segment(self, segment: SegmentInfo) -> SegmentVolume:
        prefix = segment.extra["seg_prefix"].rstrip("/")
        seg_root = prefix[len(_BUCKET) + 1:] + "/"
        candidates = self._zarr_dirs_under(seg_root + "surface-volumes/")
        if not candidates:
            # some segments store the surface volume directly, no surface-volumes/ subdir
            candidates = self._zarr_dirs_under(seg_root)
        if not candidates:
            subdirs = {p.rstrip("/").split("/")[-1] for p in self._list_prefixes(seg_root)}
            if "mesh" in subdirs and "surface-volumes" not in subdirs:
                raise ValueError(
                    f"segment {segment.segment_id} is an auto-grown mesh only — no rendered "
                    "surface volume exists yet, so there is nothing to view. (These early "
                    "automated traces on First-Letters scrolls still need rendering.)"
                )
            raise ValueError(f"no surface-volume zarr found for {segment.segment_id}")
        return _open_first(candidates)

    def load_ink_overlay(self, segment: SegmentInfo, volume_url: str | None = None):
        """Ink-detection model prediction for this segment, as a grayscale preview raster.

        Uses the small ``ink-detection/downsampled/*-ds8.jpg`` preview (~2 MB) so the whole
        surface loads at once; picks the one rendered from the opened surface volume. These
        are MODEL PREDICTIONS, not human labels — the read scrolls carry them for nearly
        every segment; the open-data bucket has no human/IR ink masks for scroll segments."""
        import io

        import numpy as np
        from PIL import Image

        extra = getattr(segment, "extra", {}) or {}
        frag = extra.get("fragment")
        if frag is not None:  # a labelled fragment carries a real human/IR ink mask
            from .fragments import load_fragment_ink
            return load_fragment_ink(frag)
        if "seg_prefix" not in extra:
            return None  # e.g. a raw scroll volume has no surface-level ink raster
        seg_root = extra["seg_prefix"][len(_BUCKET) + 1:].rstrip("/")
        previews = [
            k for k in self._list_keys(seg_root + "/ink-detection/downsampled/")
            if k.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not previews:
            return None
        choice = _match_ink(previews, volume_url) or previews[0]
        import requests

        try:
            data = requests.get(f"{_BUCKET}/{choice}", timeout=90).content
            img = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        except Exception as exc:
            raise ValueError(f"ink preview load failed: {type(exc).__name__}: {exc}")
        return img, choice.rsplit("/", 1)[-1]

    def project_ink(self, segment: SegmentInfo, scan_id: str, cache_dir: str | None = None):
        """Back-project a segment's ink prediction into native volume voxels via its .tifxyz
        mesh for ``scan_id`` (the displayed volume's scan). Returns (N,4) [z,y,x,ink] or None.
        The (heavy) result is cached to disk so re-projecting a segment is instant."""
        import os

        import numpy as np

        from .ink_projection import project_segment_ink

        seg_prefix = segment.extra["seg_prefix"]
        cache_file = None
        if cache_dir:
            d = os.path.join(str(cache_dir), "projections")
            try:
                os.makedirs(d, exist_ok=True)
                cache_file = os.path.join(d, f"{segment.segment_id}_{scan_id}_v2.npy")
                if os.path.exists(cache_file):
                    return np.load(cache_file)
            except OSError:
                cache_file = None
        pts = project_segment_ink(seg_prefix, scan_id)
        if pts is not None and cache_file:
            try:
                np.save(cache_file, pts)
            except OSError:
                pass
        return pts

    def _zarr_dirs_under(self, prefix: str) -> list[str]:
        return [
            f"{_BUCKET}/{p}"
            for p in self._list_prefixes(prefix)
            if p.rstrip("/").endswith(".zarr")
        ]

    def _list_keys(self, prefix: str) -> list[str]:
        """Object keys (files) directly under ``prefix``. Retries transient failures."""
        import requests

        url = f"{_BUCKET}/?list-type=2&prefix={prefix}"
        for _ in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    return re.findall(r"<Key>([^<]+)</Key>", r.text)
            except requests.RequestException:
                pass
        return []


# --------------------------------------------------------------------------- helpers --
def _open_first(candidates: list[str]) -> SegmentVolume:
    """Try candidate zarr URLs in preference order; return the first that opens. Transient HTTPS hiccups during
    the metadata open are absorbed inside ``discover_levels`` (``_http_json``/``_http_ok`` retry with backoff),
    so a ValueError here means the candidate genuinely has no readable arrays — move to the next one."""
    errors = []
    for url in _ordered(candidates):
        try:
            return ZarrSegment(url, parse_voxel_um(url))
        except Exception as exc:  # try the next candidate rather than hiding the scroll
            errors.append(f"{url.rsplit('/', 1)[-1] or url}: {type(exc).__name__}: {str(exc)[:80]}")
    raise ValueError("no zarr opened — " + "; ".join(errors[:3]))


def _match_ink(previews: list[str], volume_url: str | None) -> str | None:
    """Pick the ink preview rendered from ``volume_url`` (same ``volume-<ts>`` token)."""
    if not volume_url:
        return None
    base = volume_url.rstrip("/").rsplit("/", 1)[-1].lower().replace(".zarr", "")
    m = re.search(r"volume-(\d+)", base)
    token = m.group(1) if m else None
    for p in previews:
        pl = p.lower()
        if token and token in pl:
            return p
        if base and base in pl:
            return p
    return None


def _ordered(candidates: list[str]) -> list[str]:
    def key(u: str):
        name = u.rstrip("/").split("/")[-1].lower()
        preferred = 0 if ("masked" in name or "standardized" in name) else 1
        return (preferred, parse_voxel_um(name) or 1e9)

    return sorted(candidates, key=key)


def _prettify(pherc: str) -> str:
    body = pherc[len("PHerc"):] if pherc.startswith("PHerc") else pherc
    body = re.sub(r"([A-Za-z])(\d)", r"\1 \2", body)   # Paris4 -> Paris 4
    body = re.sub(r"(\d)([A-Za-z])", r"\1 \2", body)   # 1667Cr -> 1667 Cr
    body = re.sub(r"^0+(\d)", r"\1", body)             # 0172 -> 172
    return "PHerc " + body.strip()


def _scroll_name(pherc: str, sid: str) -> str:
    pretty = _prettify(pherc)
    if sid in {"1", "2", "3", "4", "5"}:
        return f"Scroll {sid} ({pretty})"
    return pretty


def _sort_key(s: ScrollInfo):
    if s.scroll_id in {"1", "2", "3", "4", "5"}:
        return (0, int(s.scroll_id), "")
    return (1, 0, s.extra.get("pherc", ""))
