"""Phase C.2 — TopicGuard: embedding-based topic scope validator.

Uses OpenAI text-embedding-3-small to pre-compute allowed-topic vectors
once at construction time, then classifies each query by cosine similarity.

Pipeline
--------
  1. __init__(allowed_topics): embed each anchor string → store unit vectors
  2. check(text) → (is_allowed: bool, reason: str)
       embed query → cosine_sim vs every anchor vector → max_sim
       if max_sim >= threshold (default 0.6) → allowed
       else → off-topic with closest topic name + score
  3. refuse_message(reason) → polite bilingual refusal string

Bilingual anchors
-----------------
Default anchors include both English and Vietnamese phrases so that
queries in either language reach the 0.6 cosine threshold.
The refuse message always displays Vietnamese topic names for UX consistency.
"""
from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Default bilingual topic anchors ──────────────────────────────────────────
# Each semantic cluster has both an English and a Vietnamese phrase so that
# queries in either language land within cosine 0.6 of at least one anchor.
_DEFAULT_ANCHORS: list[str] = [
    # EN — personal data protection
    "personal data protection law and regulations",
    # VN — bảo vệ dữ liệu cá nhân
    "bảo vệ dữ liệu cá nhân Nghị định 13 2023",
    # EN — data privacy
    "data privacy rights and regulations",
    # VN — quyền riêng tư dữ liệu
    "quyền riêng tư dữ liệu cá nhân quy định pháp luật",
    # EN — consent & data processing
    "consent for processing personal data",
    # VN — đồng ý xử lý dữ liệu
    "sự đồng ý xử lý dữ liệu cá nhân",
    # EN — data controller
    "data controller and data processor responsibilities",
    # VN — bên kiểm soát dữ liệu
    "trách nhiệm bên kiểm soát và bên xử lý dữ liệu",
    # EN — VAT
    "VAT value added tax rate and rules",
    # VN — thuế GTGT
    "thuế giá trị gia tăng GTGT mức thuế suất",
    # EN — tax filing
    "tax declaration and filing procedures",
    # VN — khai thuế
    "kê khai thuế GTGT nộp tờ khai thuế",
]

# Human-readable cluster labels (used in refuse messages)
_DISPLAY_TOPICS_VN = [
    "bảo vệ dữ liệu cá nhân",
    "quy định quyền riêng tư dữ liệu",
    "sự đồng ý và xử lý dữ liệu",
    "trách nhiệm bên kiểm soát dữ liệu",
    "thuế giá trị gia tăng (GTGT)",
    "khai báo và nộp thuế",
]


class TopicGuard:
    """Embedding-based topic scope validator."""

    DEFAULT_THRESHOLD = 0.6

    def __init__(
        self,
        allowed_topics: Sequence[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.threshold = threshold
        self._model = model

        anchors = list(allowed_topics) if allowed_topics is not None else _DEFAULT_ANCHORS
        self.allowed_topics = anchors

        self._client = self._make_client()
        self._topic_vecs: list[np.ndarray] = [self._embed(t) for t in anchors]

    # ── OpenAI client ─────────────────────────────────────────────────────────
    @staticmethod
    def _make_client():
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        vec = np.array(resp.data[0].embedding, dtype=float)
        return vec / (np.linalg.norm(vec) + 1e-12)

    # ── Public API ────────────────────────────────────────────────────────────
    def check(self, text: str) -> tuple[bool, str]:
        """Return (is_allowed, reason_string).

        is_allowed=True  when max cosine similarity >= self.threshold.
        is_allowed=False when off-topic; reason names the closest anchor + score.
        """
        if not text.strip():
            return False, "Empty query — no topic to match."

        q_vec = self._embed(text)
        sims = [_cosine(q_vec, tv) for tv in self._topic_vecs]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        best_anchor = self.allowed_topics[best_idx]

        if best_sim >= self.threshold:
            return True, (
                f"On-topic: '{best_anchor}' (similarity={best_sim:.3f})"
            )
        return False, (
            f"Off-topic: closest match is '{best_anchor}' "
            f"(similarity={best_sim:.3f}, threshold={self.threshold})"
        )

    def refuse_message(self, reason: str) -> str:
        """Return a polite bilingual refusal explaining the scope limit."""
        m = re.search(r"closest match is '([^']+)'", reason)
        closest_raw = m.group(1) if m else ""
        # map anchor to a human-readable VN label if possible
        closest_vn = closest_raw  # fallback: show raw anchor
        for anchor, display in zip(_DEFAULT_ANCHORS, _DISPLAY_TOPICS_VN * 2):
            if anchor == closest_raw:
                closest_vn = display
                break

        ms = re.search(r"similarity=([\d.]+)", reason)
        sim_str = ms.group(1) if ms else "—"

        topics_vn = ", ".join(_DISPLAY_TOPICS_VN)
        return (
            f"Tôi chỉ hỗ trợ câu hỏi về: {topics_vn}.\n"
            f"Câu hỏi của bạn có vẻ liên quan đến '{closest_vn}' "
            f"(similarity {sim_str}).\n"
            f"Bạn có thể đặt lại câu hỏi cụ thể hơn về {topics_vn} không?"
        )
