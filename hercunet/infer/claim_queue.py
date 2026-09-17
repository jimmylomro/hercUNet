"""Elastic multi-instance claim-queue on a SHARED directory.

Run the same command on any number of workers (extra ones can join anytime, on separate pods sharing a network
volume, or as separate processes on one box — one per GPU). An atomic ``O_EXCL`` ``<id>.claim`` file gives each
work item to exactly ONE worker; an ``<id>.done`` file is the resume-safe completion marker. A worker that
crashes mid-item leaves a ``.claim`` with no ``.done`` — :func:`reclaim_orphans` re-queues those. Callers must
keep work items DISJOINT in their output so the parallel writes never race (e.g. stride-P tiles / blocks).

Used by the iterative Jacobi refiner (:mod:`hercunet.infer.jacobi_refine`) for BOTH modes: single-instance
multi-GPU (each local GPU is one worker, donedir on the local FS) and multi-instance (workers across pods sharing
a network volume). The primitives here are the single source of truth for the claim/done protocol.
"""
from __future__ import annotations

import os
import random
import time


def try_claim(donedir, name) -> bool:
    """Atomically claim ``name`` (``O_EXCL`` create of ``<name>.claim``). Returns True iff THIS process won it
    (False if another worker already holds the claim). Also usable for one-off leader election (``_finalize``)."""
    try:
        os.close(os.open(os.path.join(donedir, f"{name}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def mark_done(donedir, name) -> None:
    open(os.path.join(donedir, f"{name}.done"), "w").close()


def is_done(donedir, name) -> bool:
    return os.path.exists(os.path.join(donedir, f"{name}.done"))


def is_claimed(donedir, name) -> bool:
    return os.path.exists(os.path.join(donedir, f"{name}.claim"))


def count_done(donedir, int_only=True) -> int:
    """Number of completion markers. ``int_only`` counts only integer-named items (tiles/blocks), so it ignores
    named markers like ``_complete``/``_created`` used for coordination."""
    try:
        fs = os.listdir(donedir)
    except FileNotFoundError:
        return 0
    return sum(1 for f in fs if f.endswith(".done") and (not int_only or f[:-5].isdigit()))


def reclaim_orphans(donedir) -> int:
    """Delete ``.claim`` files that have no matching ``.done`` (left by crashed workers) so their items re-queue.
    Returns how many were reclaimed."""
    n = 0
    try:
        fs = os.listdir(donedir)
    except FileNotFoundError:
        return 0
    for f in fs:
        if f.endswith(".claim") and not os.path.exists(os.path.join(donedir, f[:-6] + ".done")):
            try:
                os.remove(os.path.join(donedir, f))
                n += 1
            except OSError:
                pass
    return n


def wait_for_path(path, what="", poll=5.0, timeout=None) -> None:
    """Block until ``path`` exists (e.g. a leader's ``_complete`` marker)."""
    t0 = time.time()
    while not os.path.exists(path):
        if timeout and time.time() - t0 > timeout:
            raise TimeoutError(f"waited {timeout:.0f}s for {what or path}")
        time.sleep(poll)


def wait_all_done(donedir, n_items, what="", int_only=True, poll=5.0, timeout=None, log_every=30.0) -> None:
    """BARRIER: block until every work item has a ``.done`` (across all workers). ``n_items`` is the total the
    caller expects (deterministic from the tiling). ``log_every`` (seconds; 0 to silence) prints the GLOBAL
    ``done/total`` so a worker waiting here still reports overall progress."""
    t0 = last = time.time()
    d = count_done(donedir, int_only=int_only)
    while d < n_items:
        if timeout and time.time() - t0 > timeout:
            raise TimeoutError(f"barrier {what}: {d}/{n_items} after {timeout:.0f}s")
        if log_every and time.time() - last >= log_every:
            print(f"[barrier{' ' + what if what else ''}] GLOBAL {d}/{n_items} done, waiting…", flush=True)
            last = time.time()
        time.sleep(poll)
        d = count_done(donedir, int_only=int_only)


def elect_leader(donedir, tag, my_id, settle=4.0):
    """Pick exactly ONE leader deterministically — robust where ``O_EXCL`` is NOT atomic under burst contention
    (e.g. a shared network volume when every worker clears a barrier at the same instant, so several would each
    "win" a ``try_claim`` and collide). Each worker drops a unique ``_<tag>_vote_<my_id>`` file, waits ``settle``
    for the others' votes to land, then the lexicographically-smallest vote wins. Returns True for one worker.
    ``my_id`` must be unique per worker (e.g. "host:pid"). Votes from a PRIOR run/restart are ignored (only those
    written within ~3x``settle`` of this worker's vote count), so a dead worker's stale vote can't "win"."""
    pfx = f"_{tag}_vote_"
    vote = f"{pfx}{my_id}"
    vp = os.path.join(donedir, vote)
    open(vp, "w").close()
    t = os.path.getmtime(vp)
    time.sleep(settle)
    fresh = []
    for f in os.listdir(donedir):
        if f.startswith(pfx):
            try:
                if abs(os.path.getmtime(os.path.join(donedir, f)) - t) < settle * 3:   # same election window
                    fresh.append(f)
            except OSError:
                pass
    fresh.sort()
    return bool(fresh) and fresh[0] == vote


def claim_chunks(donedir, ids, chunk=1, reclaim=False, shuffle=True, id_str=str):
    """Yield lists of freshly-claimed ids (up to ``chunk`` each), until this worker can claim no more (all items
    are either done or held by other workers). The caller processes each returned id and calls :func:`mark_done`.
    Dynamic work-stealing: a faster worker simply comes back and claims more. ``shuffle`` (per-pid) de-syncs
    workers to cut claim collisions; ``reclaim`` first re-queues orphaned claims from crashed workers."""
    os.makedirs(donedir, exist_ok=True)
    if reclaim:
        reclaim_orphans(donedir)
    order = list(ids)
    if shuffle:
        random.Random(os.getpid()).shuffle(order)
    while True:
        won = []
        for i in order:
            if len(won) >= chunk:
                break
            nm = id_str(i)
            if is_done(donedir, nm):
                continue
            if try_claim(donedir, nm):
                won.append(i)
        if not won:                                                   # nothing left this worker can claim
            return
        yield won
