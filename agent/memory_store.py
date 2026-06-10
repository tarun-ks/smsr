"""
memory_store.py
===============
Persistent multi-session memory store backed by SQLite (metadata) +
FAISS (vector search) + sentence-transformers (embeddings).

Each memory entry carries:
  id         — auto-increment rowid
  content    — the text of the memory
  session_id — which session wrote it
  timestamp  — ISO timestamp at write time
  hmac_tag   — HMAC-SHA256 tag if written via trusted path, else NULL
"""

import sqlite3
import time
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = None

def _get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


class MemoryStore:
    """
    Multi-session persistent memory store.
    Supports two write paths:
      - trusted_write()  : tags the entry with an HMAC (legitimate agent path)
      - inject()         : writes WITHOUT a tag (simulates attacker direct injection)
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self.dim = 384
        self._build_faiss_index()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT    NOT NULL,
                session_id TEXT    NOT NULL,
                timestamp  REAL    NOT NULL,
                hmac_tag   TEXT,
                embedding  BLOB    NOT NULL
            )
        """)
        self.conn.commit()

    def _build_faiss_index(self):
        """Rebuild in-memory FAISS index from all rows in DB."""
        self.index = faiss.IndexFlatIP(self.dim)   # inner product = cosine on unit vecs
        self._id_map = []                          # faiss row → db id
        rows = self.conn.execute(
            "SELECT id, embedding FROM memories ORDER BY id"
        ).fetchall()
        if rows:
            vecs = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            self.index.add(vecs)
            self._id_map = [r[0] for r in rows]

    def _embed(self, text: str) -> np.ndarray:
        model = _get_model()
        v = model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        return v

    def _insert(self, content: str, session_id: str, hmac_tag=None) -> int:
        emb = self._embed(content)
        ts  = time.time()
        cur = self.conn.execute(
            "INSERT INTO memories (content, session_id, timestamp, hmac_tag, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (content, session_id, ts, hmac_tag, emb.tobytes())
        )
        self.conn.commit()
        rowid = cur.lastrowid
        self.index.add(emb.reshape(1, -1))
        self._id_map.append(rowid)
        return rowid

    def trusted_write(self, content: str, session_id: str, hmac_tag: str) -> int:
        """Legitimate write path — includes HMAC provenance tag."""
        return self._insert(content, session_id, hmac_tag=hmac_tag)

    def inject(self, content: str, session_id: str = "attacker") -> int:
        """
        Attacker injection path — no HMAC tag.
        Simulates direct DB write (SQL injection, misconfigured permissions, etc.)
        """
        return self._insert(content, session_id, hmac_tag=None)

    def retrieve(self, query: str, k: int = 5,
                 verified_only: bool = False,
                 candidate_ids: list = None) -> list[dict]:
        """
        Return top-k memories by cosine similarity.
        verified_only: filter to HMAC-tagged entries only.
        candidate_ids: restrict search to specific DB ids (used by SMSR).
        """
        if self.index.ntotal == 0:
            return []

        q_emb = self._embed(query).reshape(1, -1)
        # search over all or over a candidate subset
        if candidate_ids is None:
            search_k = min(k * 10, self.index.ntotal)  # over-fetch for filtering
            scores, indices = self.index.search(q_emb, search_k)
            db_ids = [self._id_map[i] for i in indices[0] if i >= 0]
            score_map = {self._id_map[i]: float(scores[0][j])
                         for j, i in enumerate(indices[0]) if i >= 0}
        else:
            # build a temporary sub-index from candidate_ids
            rows = self.conn.execute(
                f"SELECT id, embedding FROM memories WHERE id IN ({','.join('?'*len(candidate_ids))})",
                candidate_ids
            ).fetchall()
            if not rows:
                return []
            sub_vecs = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            sub_idx  = faiss.IndexFlatIP(self.dim)
            sub_idx.add(sub_vecs)
            sub_scores, sub_positions = sub_idx.search(q_emb, min(k, len(rows)))
            db_ids   = [rows[p][0] for p in sub_positions[0] if p >= 0]
            score_map = {rows[p][0]: float(sub_scores[0][j])
                         for j, p in enumerate(sub_positions[0]) if p >= 0}

        if not db_ids:
            return []

        placeholders = ','.join('?' * len(db_ids))
        filter_clause = "AND hmac_tag IS NOT NULL" if verified_only else ""
        rows = self.conn.execute(
            f"SELECT id, content, session_id, timestamp, hmac_tag "
            f"FROM memories WHERE id IN ({placeholders}) {filter_clause}",
            db_ids
        ).fetchall()

        results = []
        for r in rows:
            results.append({
                "id"        : r[0],
                "content"   : r[1],
                "session_id": r[2],
                "timestamp" : r[3],
                "hmac_tag"  : r[4],
                "score"     : score_map.get(r[0], 0.0),
                "verified"  : r[4] is not None,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def count(self, verified_only: bool = False) -> int:
        if verified_only:
            return self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE hmac_tag IS NOT NULL"
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def reset(self):
        self.conn.execute("DELETE FROM memories")
        self.conn.commit()
        self._build_faiss_index()
