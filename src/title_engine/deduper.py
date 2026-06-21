"""
Near-duplicate detection — spec §6.

"Dedupe near-identical titles (MinHash) first — don't pay to label
'iPhone 13 128GB' forty times; label once, reuse."

Implementation: character 3-gram Jaccard similarity.
No external deps needed at this scale (2-3k titles).
For >50k titles, swap in the datasketch library.
"""
from __future__ import annotations


def _shingles(text: str, n: int = 3) -> frozenset[str]:
    """Character n-gram shingles of a normalised string."""
    t = text.lower().strip()
    if len(t) < n:
        return frozenset({t})
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def jaccard(a: str, b: str, n: int = 3) -> float:
    sa, sb = _shingles(a, n), _shingles(b, n)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class Deduper:
    """
    Keeps track of seen titles and deduplicates near-identical ones.

    Usage:
        deduper = Deduper(threshold=0.85)
        unique = deduper.deduplicate(titles)   # list of {id, raw}
        # unique contains one representative per near-duplicate cluster.
        # deduper.clusters maps each kept id to the list of ids it represents.
    """

    def __init__(self, threshold: float = 0.85, shingle_n: int = 3):
        self.threshold = threshold
        self.shingle_n = shingle_n
        self.clusters: dict[str, list[str]] = {}  # kept_id → [duplicate_ids]
        self._kept: list[dict] = []               # kept representative items

    def deduplicate(self, items: list[dict]) -> list[dict]:
        """
        items: list of {id, raw}.
        Returns the deduplicated subset (one representative per cluster).
        """
        self.clusters = {}
        self._kept = []

        for item in items:
            raw = item.get("raw", "")
            merged = False
            for kept_item in self._kept:
                sim = jaccard(raw, kept_item["raw"], self.shingle_n)
                if sim >= self.threshold:
                    self.clusters[kept_item["id"]].append(item["id"])
                    merged = True
                    break
            if not merged:
                self._kept.append(item)
                self.clusters[item["id"]] = []

        n_in = len(items)
        n_out = len(self._kept)
        if n_in != n_out:
            print(f"[Deduper] {n_in} titles → {n_out} unique ({n_in - n_out} near-duplicates removed)")

        return list(self._kept)

    def cluster_size(self, representative_id: str) -> int:
        return 1 + len(self.clusters.get(representative_id, []))
