"""Phase C.5 — GuardedRAGPipeline: full-stack guardrail integration.

Layer execution order
---------------------
  L1  InputGuard (PII) + TopicGuard  ─── asyncio.gather (parallel)
  L2  RAG pipeline (sync, thread-wrapped)
  L3  OutputGuard (Llama Guard via Groq) — async
  L4  Audit log  — asyncio.create_task (fire-and-forget, never awaited)

Early-exit on L1 block (topic refused or hard PII) or L3 block
(unsafe response).  L2 and L3 are only reached for allowed queries.

Usage
-----
  from full_pipeline import GuardedRAGPipeline

  async def main():
      pipeline = GuardedRAGPipeline()          # uses mock RAG by default
      answer, timings = await pipeline.guarded_query("Thuế GTGT là bao nhiêu?")
      print(answer, timings)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

_root = Path(__file__).resolve().parents[1]

# ── High-confidence PII types that hard-block the query ───────────────────────
# PERSON / LOCATION / DATE_TIME have too many false positives on Vietnamese text
# and are excluded from blocking decisions.
_BLOCKING_PII = frozenset({
    "CCCD", "PHONE_VN", "TAX_CODE_VN",
    "EMAIL", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "CRYPTO", "SSN",
})


# ── Mock RAG function (used when the real Day18 pipeline is not provided) ─────
def _mock_rag_sync(query: str) -> str:
    """Simulate extractive RAG: 40-120 ms, deterministic per query."""
    import random
    rng = random.Random(hash(query) & 0xFFFF)
    time.sleep(rng.uniform(0.04, 0.12))
    snippet = query[:40].rstrip()
    return (
        f"Theo quy định hiện hành, {snippet}... "
        f"[Mock answer — replace with real RAG pipeline]"
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────
class GuardedRAGPipeline:
    """Four-layer guardrailed RAG pipeline.

    Parameters
    ----------
    rag_fn          Synchronous callable (query: str) → str.
                    Defaults to a mock that simulates ~40-120 ms extractive RAG.
    groq_api_key    Groq key for OutputGuard; falls back to GROQ_API_KEY env.
    openai_api_key  OpenAI key for TopicGuard; falls back to OPENAI_API_KEY env.
    """

    _AUDIT_FILE = _root / "phase-c" / "audit_log.jsonl"

    def __init__(
        self,
        rag_fn: Optional[Callable[[str], str]] = None,
        groq_api_key:   Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        import sys
        sys.path.insert(0, str(_root / "phase-c"))

        from input_guard  import InputGuard
        from topic_guard  import TopicGuard
        from output_guard import OutputGuardAPI

        self.input_guard  = InputGuard()
        self.topic_guard  = TopicGuard()
        self.output_guard = OutputGuardAPI(api_key=groq_api_key)
        self.rag_fn       = rag_fn or _mock_rag_sync

    # ── L1: Input validation (parallel) ──────────────────────────────────────
    async def _run_l1(
        self, user_input: str
    ) -> tuple[str, list[str], bool, str, float, float, float]:
        """Run InputGuard and TopicGuard in parallel.

        Returns
        -------
        sanitized, blocking_pii, topic_ok, topic_reason,
        l1_wall_ms, pii_ms, topic_ms
        """
        pii_elapsed:   list[float] = []
        topic_elapsed: list[float] = []

        async def _timed_pii():
            t = perf_counter()
            result = await asyncio.to_thread(self.input_guard.sanitize, user_input)
            pii_elapsed.append((perf_counter() - t) * 1000)
            return result

        async def _timed_topic():
            t = perf_counter()
            result = await asyncio.to_thread(self.topic_guard.check, user_input)
            topic_elapsed.append((perf_counter() - t) * 1000)
            return result

        wall_t0 = perf_counter()
        (sanitized, _pii_latency), (topic_ok, topic_reason) = await asyncio.gather(
            _timed_pii(), _timed_topic()
        )
        l1_wall_ms = (perf_counter() - wall_t0) * 1000

        all_pii      = self.input_guard.parse_redacted_types(sanitized)
        blocking_pii = [t for t in all_pii if t in _BLOCKING_PII]

        return (
            sanitized, blocking_pii, topic_ok, topic_reason,
            l1_wall_ms,
            pii_elapsed[0] if pii_elapsed else 0.0,
            topic_elapsed[0] if topic_elapsed else 0.0,
        )

    # ── L2: RAG (sync wrapped in thread) ─────────────────────────────────────
    async def _run_l2(self, sanitized: str) -> tuple[str, float]:
        t0 = perf_counter()
        answer = await asyncio.to_thread(self.rag_fn, sanitized)
        return answer, (perf_counter() - t0) * 1000

    # ── L3: Output safety (async Groq call) ───────────────────────────────────
    async def _run_l3(self, user_input: str, answer: str) -> tuple[bool, str, float]:
        t0 = perf_counter()
        is_safe, raw, _ = await self.output_guard.check_async(user_input, answer)
        return is_safe, raw, (perf_counter() - t0) * 1000

    # ── L4: Audit log (fire-and-forget) ──────────────────────────────────────
    async def _audit_task(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"

        def _write() -> None:
            self._AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with self._AUDIT_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line)

        await asyncio.to_thread(_write)

    def _fire_audit(self, entry: dict) -> None:
        """Schedule audit write as a background task (never awaited)."""
        asyncio.create_task(self._audit_task(entry))

    # ── Refuse message ────────────────────────────────────────────────────────
    @staticmethod
    def refuse_response(reason: str) -> str:
        return (
            "Xin lỗi, tôi không thể xử lý yêu cầu này.\n"
            f"Lý do: {reason}\n"
            "Vui lòng đặt câu hỏi liên quan đến bảo vệ dữ liệu cá nhân "
            "hoặc thuế giá trị gia tăng (GTGT)."
        )

    # ── Main entry point ──────────────────────────────────────────────────────
    async def guarded_query(
        self, user_input: str
    ) -> tuple[str, dict[str, float]]:
        """Run a query through all four guard layers.

        Returns
        -------
        answer    : the final answer (or a refusal message)
        timings   : {"L1": ms, "L2": ms, "L3": ms, "total": ms,
                     "pii_ms": ms, "topic_ms": ms, "blocked_at": str|None}
        """
        t_total = perf_counter()
        timings: dict[str, float | str | None] = {
            "L1": 0.0, "L2": 0.0, "L3": 0.0, "total": 0.0,
            "pii_ms": 0.0, "topic_ms": 0.0, "blocked_at": None,
        }

        # ── L1 ────────────────────────────────────────────────────────────────
        (
            sanitized, blocking_pii, topic_ok, topic_reason,
            l1_ms, pii_ms, topic_ms,
        ) = await self._run_l1(user_input)

        timings["L1"]       = l1_ms
        timings["pii_ms"]   = pii_ms
        timings["topic_ms"] = topic_ms

        if blocking_pii:
            reason = f"PII detected: {blocking_pii}"
            timings["blocked_at"] = "L1-PII"
            timings["total"]      = (perf_counter() - t_total) * 1000
            answer = self.refuse_response(reason)
            self._fire_audit({
                "ts": datetime.now(timezone.utc).isoformat(),
                "input": user_input[:120], "blocked": "L1-PII",
                "reason": reason, **{k: timings[k] for k in ("L1","L2","L3","total")},
            })
            return answer, timings

        if not topic_ok:
            timings["blocked_at"] = "L1-Topic"
            timings["total"]      = (perf_counter() - t_total) * 1000
            answer = self.refuse_response(topic_reason)
            self._fire_audit({
                "ts": datetime.now(timezone.utc).isoformat(),
                "input": user_input[:120], "blocked": "L1-Topic",
                "reason": topic_reason, **{k: timings[k] for k in ("L1","L2","L3","total")},
            })
            return answer, timings

        # ── L2 ────────────────────────────────────────────────────────────────
        answer, l2_ms = await self._run_l2(sanitized)
        timings["L2"] = l2_ms

        # ── L3 ────────────────────────────────────────────────────────────────
        is_safe, raw_verdict, l3_ms = await self._run_l3(user_input, answer)
        timings["L3"] = l3_ms

        if not is_safe:
            timings["blocked_at"] = "L3"
            answer = self.refuse_response(f"unsafe output detected: {raw_verdict.strip()}")

        timings["total"] = (perf_counter() - t_total) * 1000
        self._fire_audit({
            "ts": datetime.now(timezone.utc).isoformat(),
            "input": user_input[:120], "blocked": timings["blocked_at"],
            "answer": answer[:120],
            **{k: timings[k] for k in ("L1","L2","L3","total")},
        })
        return answer, timings
