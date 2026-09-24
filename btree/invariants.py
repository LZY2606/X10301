"""Structural invariant checks. Each check emits a machine-readable judgment event."""
from .page import Page, PageCorrupt, internal_child


def _min_key(pages, page_size, pid, _guard=0):
    page = pages.get(pid)
    if page is None or _guard > 10000:
        return None
    while not page.is_leaf:
        page = pages.get(page.leftmost_child)
        if page is None:
            return None
    recs = page.records()
    return recs[0][0] if recs else None


def _reachable_pages(engine):
    if engine.root_id is None or engine.root_id not in engine.pages:
        return {}
    seen, queue = {}, [engine.root_id]
    while queue:
        pid = queue.pop()
        if pid in seen or pid not in engine.pages:
            continue
        page = engine.pages[pid]
        seen[pid] = page
        if not page.is_leaf:
            queue.append(page.leftmost_child)
            for _, rec in page.records():
                queue.append(internal_child(rec))
    return seen


def check_all(engine, events):
    """Run every invariant over the reachable in-memory state."""
    judgments = []

    def judge(rule, status, details):
        judgments.append({"rule": rule, "status": status, "details": details})

    pages = _reachable_pages(engine)

    # keys deleted by logical ops: separators pointing at them are legitimately
    # stale and must not be flagged
    deleted_keys = set()
    for ev in events:
        if ev.get("kind") == "logical_op" and ev["payload"].get("op") == "delete":
            try:
                deleted_keys.add(bytes.fromhex(ev["payload"]["key"]))
            except (TypeError, ValueError):
                pass

    # per-page structural checks
    page_problems = {}
    for pid, page in pages.items():
        errs = page.validate()
        if errs:
            page_problems[pid] = errs
    judge("page_structure", "pass" if not page_problems else "fail",
          {"problems": page_problems})

    # separator consistency
    leaf_bad, internal_bad = [], []
    leaf_keys = set()
    for pid, page in pages.items():
        if page.is_leaf:
            for k, _ in page.records():
                leaf_keys.add(k)
    for pid, page in pages.items():
        if page.is_leaf:
            continue
        for sep, rec in page.records():
            child = pages.get(internal_child(rec))
            if child is None:
                continue
            if child.is_leaf:
                # copy-up: the separator must exist at the leaf level (unless
                # it was later deleted by a logical delete)
                if sep not in leaf_keys and sep not in deleted_keys:
                    leaf_bad.append({"page_id": pid, "separator": sep.hex(),
                                     "problem": "separator_missing_from_leaves"})
            else:
                # promote: separator must not remain among the child internal
                # page's own keys, and must not exceed the subtree minimum
                child_keys = [k for k, _ in child.records()]
                mn = _min_key(pages, engine.page_size, child.page_id)
                if sep in child_keys or (mn is not None and sep > mn):
                    internal_bad.append({"page_id": pid, "separator": sep.hex(),
                                         "child_min": mn.hex() if mn else None,
                                         "still_in_child": sep in child_keys})
    judge("separator_leaf_copy_up", "pass" if not leaf_bad else "fail",
          {"violations": leaf_bad})
    judge("separator_internal_promote", "pass" if not internal_bad else "fail",
          {"violations": internal_bad})

    # leaf sibling chain covers exactly the reachable leaves, in key order
    chain, chain_problem = [], None
    if engine.root_id is not None and engine.root_id in pages:
        page = pages[engine.root_id]
        while not page.is_leaf:
            page = pages[page.leftmost_child]
        guard = 0
        while page is not None and guard < 100000:
            chain.append(page.page_id)
            nxt = page.right_sibling
            page = pages.get(nxt) if nxt else None
            guard += 1
        if guard >= 100000:
            chain_problem = "sibling_chain_cycle"
    leaves_sorted = sorted(p for p, pg in pages.items() if pg.is_leaf)
    ok = chain_problem is None and sorted(chain) == leaves_sorted and len(chain) == len(set(chain))
    judge("sibling_chain", "pass" if ok else "fail",
          {"chain": chain, "leaves": leaves_sorted, "problem": chain_problem})

    # split events must be byte-based and both halves must fit
    split_bad = []
    for ev in events:
        if ev.get("kind") != "page_split":
            continue
        p = ev["payload"]
        if p["basis"] != "byte":
            split_bad.append({"page_id": p["page_id"], "basis": p["basis"]})
        if p["left_bytes"] > p["capacity_bytes"] - 32 or p["right_bytes"] > p["capacity_bytes"] - 32:
            split_bad.append({"page_id": p["page_id"], "overflow": True})
    judge("split_basis_byte", "pass" if not split_bad else "fail",
          {"violations": split_bad})

    # no reachable empty leaf (empty pages must be unlinked after delete)
    empty = []
    if engine.root_id is not None and engine.root_id in pages:
        root = pages[engine.root_id]
        for pid, page in pages.items():
            if page.is_leaf and page.slot_count == 0 and pid != root.page_id:
                empty.append(pid)
    judge("empty_page_unlinked", "pass" if not empty else "fail",
          {"empty_reachable_leaves": empty})
    return judgments
