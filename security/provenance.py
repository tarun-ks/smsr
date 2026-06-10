"""
provenance.py
=============
HMAC-SHA256 provenance tagging for memory entries.

K_server is a server-side secret never exposed to agents or users.
The tag binds content + session_id + timestamp, preventing an
adversary from forging a valid tag without K_server.

Security: forging one tag requires breaking HMAC-SHA256, i.e.
finding a collision in SHA-256 or guessing the 256-bit key.
"""

import hmac
import hashlib
import os
import secrets


class HMACProvenance:
    """
    Signs and verifies memory entries.
    In production: K_server lives in an HSM / secrets manager.
    For experiments: generated fresh each run (stored as instance attribute).
    """

    def __init__(self, key: bytes = None):
        # 256-bit server key — never shared with agents or users
        self.key = key if key is not None else secrets.token_bytes(32)

    def sign(self, content: str, session_id: str, timestamp: float) -> str:
        """Return hex HMAC tag for a memory entry."""
        msg = f"{content}||{session_id}||{timestamp:.6f}".encode("utf-8")
        return hmac.new(self.key, msg, hashlib.sha256).hexdigest()

    def verify(self, content: str, session_id: str, timestamp: float, tag: str) -> bool:
        """Return True iff tag is the correct HMAC for this entry."""
        expected = self.sign(content, session_id, timestamp)
        return hmac.compare_digest(expected, tag)

    def verify_entry(self, entry: dict) -> bool:
        """Verify a memory dict (as returned by MemoryStore.retrieve)."""
        if not entry.get("hmac_tag"):
            return False
        return self.verify(
            entry["content"],
            entry["session_id"],
            entry["timestamp"],
            entry["hmac_tag"],
        )
