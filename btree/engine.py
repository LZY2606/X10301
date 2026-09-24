"""B-tree engine with page-image WAL, torn-write crash simulation and idempotent recovery.

Every fact the review archive needs is emitted through ``emit(kind, payload)``:
logical ops, page pre-images, slot directories, free space, sibling links,
page-id allocations and log sequence numbers.
"""
import hashlib
from bisect import bisect_left, bisect_right

from .page import (LEAF_FLAG, Page, PageOverflow, PageCorrupt,
                   leaf_rec, leaf_value, internal_rec, internal_child)

DEFAULT_RULES = {
    "split_basis": "byte",          # split decided by byte space, never by key count
    "separator_leaf": "copy-up",    # leaf split copies separator into parent
    "separator_internal": "promote",  # internal split promotes (moves) the middle key
    "duplicate_policy": "reject",   # reject | overwrite | append
    "empty_leaf": "unlink",         # unlink | keep empty leaves in the sibling chain
}


class CrashError(Exception):
    def __init__(self, lsn, page_id, bytes_written):
        super().__init__(f"crash during write of page {page_id} at lsn {lsn} "
                         f"after {bytes_written} bytes")
        self.lsn = lsn
        self.page_id = page_id
        self.bytes_written = bytes_written


class SplitFailure(Exception):
    pass


def _hx(b: bytes) -> str:
    return b.hex()


class Engine:
    def __init__(self, page_size: int, rules: dict = None, emit=None):
        self.page_size = page_size
        self.rules = dict(DEFAULT_RULES, **(rules or {}))
        self.emit = emit or (lambda kind, payload: None)
        self.pages = {}          # page_id -> Page (buffer pool)
        self.disk = {}           # page_id -> bytes (simulated durable storage)
        self.log = []            # WAL records
        self.lsn = 0
        self.next_page_id = 1
        self.root_id = None
        self.op_seq = 0
        self.dirty = {}            # page_id -> Page, flushed at op end
        self.pending_crash = None   # bytes to write before tearing
        self.crashed = False

    # ------------------------------------------------------------------
    def alloc_page(self, flags) -> Page:
        page = Page(self.page_size, self.next_page_id, flags)
        self.emit("page_id_alloc", {"page_id": page.page_id, "flags": flags,
                                    "allocator_high_water": self.next_page_id})
        self.next_page_id += 1
        self.pages[page.page_id] = page
        return page

    def log_root(self):
        self.lsn += 1
        self.log.append({"lsn": self.lsn, "kind": "root", "root_id": self.root_id})

    def mark_dirty(self, page: Page):
        self.dirty[page.page_id] = page

    def flush(self):
        """WAL protocol: append every dirty page image to the log first,
        then flush pages to durable storage. A crash tears the flush, never
        the log, so recovery can always restore a consistent state."""
        pending = list(self.dirty.values())
        self.dirty.clear()
        images = []
        for page in pending:
            self.lsn += 1
            page.lsn = self.lsn
            page.set_crc()
            before = self.disk.get(page.page_id, bytes(self.page_size))
            after = bytes(page.buf)
            self.log.append({"lsn": self.lsn, "kind": "page",
                             "page_id": page.page_id,
                             "before": before.hex(), "after": after.hex()})
            self.emit("page_pre_image", {"page_id": page.page_id, "lsn": self.lsn,
                                         "hex": before.hex()})
            self.emit("log_record", {"lsn": self.lsn, "kind": "page",
                                     "page_id": page.page_id,
                                     "before_len": len(before), "after_len": len(after)})
            self.emit("page_image", self._image_payload(page))
            images.append((page.page_id, before, after))
        for page_id, before, after in images:
            if self.pending_crash is not None:
                k = self.pending_crash
                self.pending_crash = None
                self.disk[page_id] = after[:k] + before[k:]
                self.crashed = True
                self.emit("crash", {"at_lsn": self.lsn, "page_id": page_id,
                                    "bytes_written": k, "page_size": self.page_size})
                raise CrashError(self.lsn, page_id, k)
            self.disk[page_id] = after

    def _image_payload(self, page: Page):
        recs = page.records()
        payload = {
            "page_id": page.page_id,
            "lsn": page.lsn,
            "flags": page.flags,
            "is_leaf": page.is_leaf,
            "hex": bytes(page.buf).hex(),
            "slot_directory": page.slot_directory(),
            "free_space": page.free_space,
            "free_start": page.free_start,
            "free_end": page.free_end,
            "right_sibling": page.right_sibling,
            "keys": [_hx(k) for k, _ in recs],
        }
        if page.is_leaf:
            payload["values"] = [_hx(leaf_value(r)) for _, r in recs]
        else:
            payload["leftmost_child"] = page.leftmost_child
            payload["children"] = [page.leftmost_child] + [internal_child(r) for _, r in recs]
        return payload

    # ------------------------------------------------------------------
    def descend(self, key: bytes):
        path = []
        pid = self.root_id
        while True:
            page = self.pages[pid]
            path.append(pid)
            if page.is_leaf:
                return path
            recs = page.records()
            keys = [k for k, _ in recs]
            i = bisect_right(keys, key)
            pid = page.leftmost_child if i == 0 else internal_child(recs[i - 1][1])

    # ------------------------------------------------------------------
    def insert(self, key: bytes, value: bytes):
        if self.root_id is None:
            root = self.alloc_page(LEAF_FLAG)
            self.root_id = root.page_id
            self.log_root()
            root.rebuild([(key, leaf_rec(key, value))], 0)
            self.emit("root_change", {"old_root": None, "new_root": root.page_id,
                                      "reason": "root_create"})
            self.mark_dirty(root)
            return {"status": "ok", "outcome": "inserted"}
        path = self.descend(key)
        leaf = self.pages[path[-1]]
        recs = leaf.records()
        keys = [k for k, _ in recs]
        idx = bisect_left(keys, key)
        dup = idx < len(keys) and keys[idx] == key
        policy = self.rules["duplicate_policy"]
        outcome = "inserted"
        if dup:
            if policy == "reject":
                self.emit("duplicate_key", {"key": _hx(key), "policy": policy,
                                            "outcome": "rejected"})
                return {"status": "duplicate_rejected"}
            if policy == "overwrite":
                recs[idx] = (key, leaf_rec(key, value))
                outcome = "overwritten"
            else:  # append
                j = idx
                while j < len(keys) and keys[j] == key:
                    j += 1
                recs.insert(j, (key, leaf_rec(key, value)))
                outcome = "appended"
            self.emit("duplicate_key", {"key": _hx(key), "policy": policy,
                                        "outcome": outcome})
        else:
            recs.insert(idx, (key, leaf_rec(key, value)))
        self._put_leaf(path, leaf, recs)
        return {"status": "ok", "outcome": outcome}

    def _put_leaf(self, path, leaf, recs):
        try:
            leaf.rebuild(recs, 0)
        except PageOverflow:
            self._split_leaf(path, leaf, recs)
            return
        self.mark_dirty(leaf)

    def _split_leaf(self, path, leaf, recs):
        basis = self.rules["split_basis"]
        if basis == "count":
            mid = len(recs) // 2
            left, right = recs[:mid], recs[mid:]
        else:  # byte basis: balance used bytes, never key count
            sizes = [len(r) for _, r in recs]
            total = sum(sizes)
            cum, i = 0, 0
            while i < len(recs) - 1 and cum + sizes[i] < total / 2:
                cum += sizes[i]
                i += 1
            i = max(1, min(i, len(recs) - 1))
            left, right = recs[:i], recs[i:]
        new = self.alloc_page(LEAF_FLAG)
        new.right_sibling = leaf.right_sibling
        leaf.right_sibling = new.page_id
        for part, chunk in ((leaf, left), (new, right)):
            try:
                part.rebuild(chunk, 0)
            except PageOverflow as exc:
                raise SplitFailure(
                    f"split_basis={basis} produced a half that does not fit: {exc}")
        self.mark_dirty(leaf)
        self.mark_dirty(new)
        sep = right[0][0]
        if self.rules["separator_leaf"] == "promote":
            # wrong for leaves: removes the separator key from the leaf level
            right = right[1:]
            new.rebuild(right, 0)
            self.mark_dirty(new)
        self.emit("page_split", {
            "page_id": leaf.page_id, "new_page_id": new.page_id, "level": "leaf",
            "basis": basis,
            "left_bytes": leaf.needed_bytes(left), "right_bytes": new.needed_bytes(right),
            "capacity_bytes": self.page_size,
            "left_keys": len(left), "right_keys": len(right),
            "separator": _hx(sep), "separator_rule": self.rules["separator_leaf"]})
        self._insert_into_parent(path, sep, new.page_id)

    def _insert_into_parent(self, path, sep, right_id):
        if len(path) == 1:
            old_root = path[0]
            new_root = self.alloc_page(0)  # internal
            new_root.leftmost_child = old_root
            new_root.rebuild([(sep, internal_rec(sep, right_id))], 0)
            self.root_id = new_root.page_id
            self.log_root()
            self.mark_dirty(new_root)
            self.emit("root_change", {"old_root": old_root, "new_root": new_root.page_id,
                                      "reason": "root_split"})
            return
        parent = self.pages[path[-2]]
        recs = parent.records()
        keys = [k for k, _ in recs]
        idx = bisect_right(keys, sep)
        recs.insert(idx, (sep, internal_rec(sep, right_id)))
        try:
            parent.rebuild(recs, 0)
        except PageOverflow:
            self._split_internal(path[:-1], parent, recs)
            return
        self.mark_dirty(parent)

    def _split_internal(self, path, page, recs):
        mid = len(recs) // 2
        promoted = recs[mid][0]
        if self.rules["separator_internal"] == "copy-up":
            # wrong for internal pages: keeps the separator in the right page too
            left, right = recs[:mid], recs[mid:]
        else:
            left, right = recs[:mid], recs[mid + 1:]
        new = self.alloc_page(0)
        if self.rules["separator_internal"] == "copy-up":
            new.leftmost_child = internal_child(recs[mid - 1][1]) if mid > 0 \
                else page.leftmost_child
        else:
            new.leftmost_child = internal_child(recs[mid][1])
        page.rebuild(left, 0)
        new.rebuild(right, 0)
        self.mark_dirty(page)
        self.mark_dirty(new)
        self.emit("page_split", {
            "page_id": page.page_id, "new_page_id": new.page_id, "level": "internal",
            "basis": self.rules["split_basis"],
            "left_bytes": page.needed_bytes(left), "right_bytes": new.needed_bytes(right),
            "capacity_bytes": self.page_size,
            "left_keys": len(left), "right_keys": len(right),
            "separator": _hx(promoted), "separator_rule": self.rules["separator_internal"]})
        self._insert_into_parent(path, promoted, new.page_id)

    # ------------------------------------------------------------------
    def delete(self, key: bytes):
        if self.root_id is None:
            return {"status": "missing"}
        path = self.descend(key)
        leaf = self.pages[path[-1]]
        recs = leaf.records()
        keys = [k for k, _ in recs]
        idx = bisect_left(keys, key)
        if idx >= len(keys) or keys[idx] != key:
            self.emit("delete_missing", {"key": _hx(key)})
            return {"status": "missing"}
        recs.pop(idx)
        leaf.rebuild(recs, 0)
        self.mark_dirty(leaf)
        if len(recs) == 0 and len(path) > 1:
            if self.rules.get("empty_leaf", "unlink") == "unlink":
                self._unlink_empty_leaf(path, leaf)
            else:
                self.emit("empty_leaf_kept", {"page_id": leaf.page_id})
        return {"status": "ok"}

    def _unlink_empty_leaf(self, path, leaf):
        parent = self.pages[path[-2]]
        precs = parent.records()
        # unlink from the leaf sibling chain: find the predecessor by walking
        # the chain from the tree's leftmost leaf
        leftmost = self.pages[self.root_id]
        while not leftmost.is_leaf:
            leftmost = self.pages[leftmost.leftmost_child]
        pred, page, guard = None, leftmost, 0
        while page.page_id != leaf.page_id and guard < 100000:
            nxt = page.right_sibling
            if nxt == leaf.page_id:
                pred = page
                break
            page = self.pages.get(nxt)
            if page is None:
                break
            guard += 1
        if pred is not None:
            pred.right_sibling = leaf.right_sibling
            self.mark_dirty(pred)
        # remove the child pointer from the parent
        ci = next((i for i, (_, r) in enumerate(precs)
                   if internal_child(r) == leaf.page_id), None)
        if ci is None:
            # leaf was the parent's leftmost child: the next child takes over
            if not precs:
                self.emit("empty_leaf_kept", {"page_id": leaf.page_id,
                                              "reason": "parent_has_no_separator"})
                return
            parent.leftmost_child = internal_child(precs[0][1])
            precs.pop(0)
        else:
            precs.pop(ci)
        parent.rebuild(precs, 0)
        self.mark_dirty(parent)
        del self.pages[leaf.page_id]
        self.dirty.pop(leaf.page_id, None)
        self.emit("page_freed", {"page_id": leaf.page_id,
                                 "reason": "empty_after_delete",
                                 "unlinked_from_sibling_chain": True})
        if len(precs) == 0 and parent.page_id == self.root_id:
            old = self.root_id
            self.root_id = parent.leftmost_child
            self.log_root()
            self.emit("root_change", {"old_root": old, "new_root": self.root_id,
                                      "reason": "root_shrink"})

    # ------------------------------------------------------------------
    def run_script(self, script):
        """script: list of ops; crash/recover are ops too. Stops at crash."""
        results = []
        for op in script:
            kind = op["op"]
            self.op_seq += 1
            self.emit("logical_op", {"seq": self.op_seq, "op": kind,
                                     "key": op.get("key"), "value": op.get("value"),
                                     "source_id": op.get("id")})
            if self.crashed and kind != "recover":
                continue  # system is down until recovery runs
            try:
                if kind == "insert":
                    results.append(self.insert(bytes.fromhex(op["key"]),
                                               bytes.fromhex(op["value"])))
                elif kind == "delete":
                    results.append(self.delete(bytes.fromhex(op["key"])))
                elif kind == "crash":
                    self.pending_crash = op["bytes"]
                elif kind == "recover":
                    self.recover(passes=op.get("passes", 2),
                                 committed_hash=op.get("committed_hash"),
                                 naive=op.get("naive", False))
                else:
                    raise ValueError(f"unknown op {kind}")
                if kind in ("insert", "delete"):
                    self.flush()
            except CrashError:
                continue
        return results

    # ------------------------------------------------------------------
    def state_hash(self):
        reachable, _ = self.reachable_pages()
        h = hashlib.sha256()
        for pid in reachable:
            h.update(pid.to_bytes(4, "little"))
            h.update(self.disk[pid])
        return h.hexdigest()

    def reachable_pages(self):
        """BFS from root over durable pages; returns (ids, problems)."""
        problems = []
        if self.root_id is None or self.root_id not in self.disk:
            return [], problems
        seen, queue = [], [self.root_id]
        visited = set()
        while queue:
            pid = queue.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            raw = self.disk.get(pid)
            if raw is None:
                problems.append({"page_id": pid, "problem": "missing_page"})
                continue
            try:
                page = Page(self.page_size, buf=raw)
                errs = page.validate()
            except PageCorrupt as exc:
                errs = [str(exc)]
            if errs:
                problems.append({"page_id": pid, "problem": "reachable_half_page",
                                 "errors": errs})
                continue
            seen.append(pid)
            if page.is_leaf:
                continue
            queue.append(page.leftmost_child)
            for _, rec in page.records():
                queue.append(internal_child(rec))
        return seen, problems

    def recover(self, passes=2, committed_hash=None, naive=False):
        """Redo page images from the WAL. LSN guard makes replay idempotent."""
        hashes = []
        for p in range(passes):
            applied = 0
            records = sorted(self.log, key=lambda r: r["lsn"])
            if naive:
                records = list(reversed(records))  # wrong order, no LSN guard
            for rec in records:
                if rec["kind"] == "root":
                    self.root_id = rec["root_id"]
                    continue
                raw = self.disk.get(rec["page_id"])
                cur_lsn = -1
                if raw is not None:
                    try:
                        pg = Page(self.page_size, buf=raw)
                        if pg.magic_ok() and pg.crc_ok():
                            cur_lsn = pg.lsn
                    except PageCorrupt:
                        cur_lsn = -1
                if naive or rec["lsn"] > cur_lsn:
                    self.disk[rec["page_id"]] = bytes.fromhex(rec["after"])
                    applied += 1
            self._reload_buffer()
            hashes.append(self.state_hash())
            self.emit("recovery", {"pass": p + 1, "applied": applied,
                                   "state_hash": hashes[-1], "naive": naive})
        idempotent = len(set(hashes)) == 1
        matches_committed = committed_hash is None or hashes[-1] == committed_hash
        self.emit("judgment", {
            "rule": "replay_idempotent",
            "status": "pass" if (idempotent and matches_committed) else "fail",
            "details": {"passes": passes, "hashes": hashes,
                        "committed_hash": committed_hash,
                        "idempotent": idempotent,
                        "matches_committed": matches_committed}})
        reachable, problems = self.reachable_pages()
        self.emit("judgment", {
            "rule": "crash_no_half_page",
            "status": "pass" if not problems else "fail",
            "details": {"root_id": self.root_id, "reachable_pages": reachable,
                        "problems": problems}})

    def _reload_buffer(self):
        pages = {}
        for pid, raw in self.disk.items():
            try:
                page = Page(self.page_size, buf=raw)
                if page.magic_ok() and page.crc_ok():
                    pages[pid] = page
            except PageCorrupt:
                continue
        # keep only pages reachable from the root: freed pages may still have
        # stale images on disk and must not re-enter the buffer pool
        self.pages = {}
        queue = [self.root_id] if self.root_id in pages else []
        while queue:
            pid = queue.pop()
            if pid in self.pages or pid not in pages:
                continue
            page = pages[pid]
            self.pages[pid] = page
            if not page.is_leaf:
                queue.append(page.leftmost_child)
                for _, rec in page.records():
                    queue.append(internal_child(rec))
