"""
heuristic_defence.py
====================
Heuristic-only defence (no cryptographic provenance) — the IMPOSSIBILITY baseline.

Implements three common heuristic approaches proposed in the literature:
  1. Keyword blacklist — reject memories containing suspicious imperative phrases
  2. Perplexity proxy  — reject memories with anomalously low entropy (overly
                         assertive sentences)
  3. Semantic anomaly  — reject memories whose embedding is far from the centroid
                         of trusted memories

These are then used in the evaluation to demonstrate that:
  (a) they fail against the HeuristicBypassAttack (crafted to evade them)
  (b) they never reach Component 1's 0% ASR even on naive attacks
  (c) therefore, provenance is NECESSARY, not just sufficient

This is the empirical backing for the impossibility claim in the paper.
"""

from __future__ import annotations
import re
import numpy as np


# ── 1. Keyword blacklist ─────────────────────────────────────────────────────
_BLACKLIST_PATTERNS = [
    r"\bignore\b.*\bprevious\b",
    r"\boverride\b",
    r"\bsystem\s+context\b",
    r"\binstruction\b.*\bfollow\b",
    r"\bdisregard\b",
    r"\bnew\s+directive\b",
    r"\bmalicious\b",
    r"\bpoison\b",
    r"\binjection\b",
]
_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in _BLACKLIST_PATTERNS]


def keyword_filter(content: str) -> bool:
    """Return True (allow) if no blacklisted pattern matches."""
    return not any(rx.search(content) for rx in _BLACKLIST_RE)


# ── 2. Perplexity proxy (sentence-length entropy heuristic) ─────────────────
def entropy_proxy(content: str) -> float:
    """
    Proxy for perplexity: average word length / sentence count.
    Very short, assertive sentences typical of injected commands have low entropy.
    Higher score = more 'natural', lower score = more suspicious.
    """
    sentences = re.split(r"[.!?]", content)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg_len = np.mean([len(s.split()) for s in sentences])
    return float(avg_len)


def entropy_filter(content: str, min_entropy: float = 8.0) -> bool:
    """Return True (allow) if entropy proxy exceeds threshold."""
    return entropy_proxy(content) >= min_entropy


# ── 3. Semantic anomaly (embedding distance from trusted centroid) ───────────
class SemanticAnomalyFilter:
    """
    Computes the embedding of the trusted memory centroid and rejects
    new memories whose cosine similarity to the centroid falls below threshold.
    Rationale: legitimate memories cluster around the system's usage patterns;
    adversarial memories about random factual claims are outliers.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.centroid  = None

    def update_centroid(self, store) -> None:
        """Recompute centroid from all verified memories in the store."""
        from agent.memory_store import _get_model
        rows = store.conn.execute(
            "SELECT content FROM memories WHERE hmac_tag IS NOT NULL"
        ).fetchall()
        if not rows:
            self.centroid = None
            return
        model  = _get_model()
        vecs   = model.encode([r[0] for r in rows], normalize_embeddings=True)
        self.centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid /= norm

    def passes(self, content: str) -> bool:
        """Return True (allow) if the memory is close enough to the centroid."""
        if self.centroid is None:
            return True   # no centroid yet: allow everything
        from agent.memory_store import _get_model
        model = _get_model()
        v     = model.encode([content], normalize_embeddings=True)[0]
        sim   = float(np.dot(v, self.centroid))
        return sim >= self.threshold


# ── Combined heuristic defence wrapper ──────────────────────────────────────
class HeuristicDefence:
    """
    Applies all three heuristics as a filter at retrieve time (no provenance).
    Returns only memories that pass ALL active heuristics.
    Used to demonstrate the impossibility of heuristic-only certified defence.
    """

    def __init__(
        self,
        use_keyword  : bool = True,
        use_entropy  : bool = True,
        entropy_min  : float = 8.0,
        use_semantic : bool = True,
        semantic_thresh: float = 0.05,
    ):
        self.use_keyword   = use_keyword
        self.use_entropy   = use_entropy
        self.entropy_min   = entropy_min
        self.use_semantic  = use_semantic
        self.sem_filter    = SemanticAnomalyFilter(semantic_thresh) if use_semantic else None

    def filter(self, memories: list[dict], store=None) -> list[dict]:
        """Return the subset of memories that pass all heuristics."""
        if self.use_semantic and self.sem_filter and store:
            self.sem_filter.update_centroid(store)

        allowed = []
        for m in memories:
            content = m["content"]
            if self.use_keyword and not keyword_filter(content):
                continue
            if self.use_entropy and not entropy_filter(content, self.entropy_min):
                continue
            if self.use_semantic and self.sem_filter and not self.sem_filter.passes(content):
                continue
            allowed.append(m)
        return allowed
