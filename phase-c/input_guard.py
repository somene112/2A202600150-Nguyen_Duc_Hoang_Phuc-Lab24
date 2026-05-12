"""Phase C.1 — InputGuard: VN regex + Presidio PII redaction.

Pipeline (two-stage, applied in order):
  Stage 1 — Vietnamese regex patterns (no external model, ~0ms)
    CCCD        12 consecutive digits
    PHONE_VN    Vietnamese mobile: +84… or 0(3x/5x/7x/8x/9x)…
    TAX_CODE_VN 10-digit MST, optional -3 suffix
    EMAIL       RFC-5321-ish regex (universal, but applied here before Presidio)

  Stage 2 — Presidio NLP (language="en", en_core_web_lg, lazy-loaded once)
    Covers: PERSON, ORGANIZATION, PHONE_NUMBER, EMAIL_ADDRESS,
            LOCATION, CREDIT_CARD, CRYPTO, DATE_TIME, NRP, …

Both stages replace detected spans with <ENTITY_TYPE> tokens.
Subsequent calls after the first are fast because the Presidio engine
is cached as a class-level singleton.
"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

# ── VN regex patterns (ordered: most specific first) ──────────────────────────
# CCCD: exactly 12 digits, not part of a longer number
_RE_CCCD = re.compile(r"(?<!\d)\d{12}(?!\d)")

# Vietnamese mobile prefixes (Viettel 03x/08x/09x, Mobifone 07x/08x/09x,
# Vinaphone 08x/09x, etc.).  Matches +84XXXXXXXXX or 0XXXXXXXXX.
_RE_PHONE_VN = re.compile(
    r"(?:\+84|(?<!\d)0)"               # country code or leading 0
    r"(?:3[2-9]|5[25689]|7[06789]|8[1-689]|9[0-9])"  # mobile prefix pair
    r"\d{7}"                            # remaining 7 digits
    r"(?!\d)"
)

# Tax identification number: 10 digits, optional -NNN branch suffix
# Applied AFTER phone so phone numbers are already replaced.
_RE_TAX_CODE = re.compile(r"(?<!\d)\d{10}(?:-\d{3})?(?!\d)")

# Email (RFC-5321 simplified)
_RE_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"          # negative lookbehind
    r"[A-Za-z0-9._%+\-]{1,64}"
    r"@"
    r"[A-Za-z0-9.\-]+"
    r"\."
    r"[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+\-@])"          # negative lookahead
)

_VN_STAGES: list[tuple[re.Pattern, str]] = [
    (_RE_CCCD,     "CCCD"),
    (_RE_PHONE_VN, "PHONE_VN"),
    (_RE_TAX_CODE, "TAX_CODE_VN"),
    (_RE_EMAIL,    "EMAIL"),
]

# Minimum Presidio confidence score; lower = more recalls, more false positives
_PRESIDIO_SCORE_THRESHOLD = 0.35


class InputGuard:
    """Two-stage PII sanitizer (VN regex → Presidio NLP)."""

    _analyzer:   "AnalyzerEngine | None"  = None
    _anonymizer: "AnonymizerEngine | None" = None

    # ── Lazy loader (class-level singleton) ───────────────────────────────────
    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._analyzer is not None:
            return
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        cls._analyzer   = AnalyzerEngine()
        cls._anonymizer = AnonymizerEngine()

    # ── Stage 1: VN regex ─────────────────────────────────────────────────────
    @staticmethod
    def _redact_vn(text: str) -> str:
        for pattern, label in _VN_STAGES:
            text = pattern.sub(f"<{label}>", text)
        return text

    # ── Stage 2: Presidio NLP ─────────────────────────────────────────────────
    @classmethod
    def _redact_presidio(cls, text: str) -> str:
        if not text.strip():
            return text
        cls._ensure_loaded()

        results = cls._analyzer.analyze(          # type: ignore[union-attr]
            text=text,
            language="en",
            score_threshold=_PRESIDIO_SCORE_THRESHOLD,
        )
        if not results:
            return text

        from presidio_anonymizer.entities import OperatorConfig
        operators = {
            r.entity_type: OperatorConfig("replace", {"new_value": f"<{r.entity_type}>"})
            for r in results
        }
        return cls._anonymizer.anonymize(          # type: ignore[union-attr]
            text=text,
            analyzer_results=results,
            operators=operators,
        ).text

    # ── Public API ────────────────────────────────────────────────────────────
    def sanitize(self, text: str) -> tuple[str, float]:
        """Redact PII and return (sanitized_text, latency_ms).

        Applies VN regex first, then Presidio NLP on the partially
        redacted text so the two stages never double-label a span.
        """
        t0 = time.perf_counter()
        out = self._redact_vn(text)
        out = self._redact_presidio(out)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return out, latency_ms

    @staticmethod
    def parse_redacted_types(sanitized: str) -> list[str]:
        """Extract unique entity-type labels from <TYPE> tokens in sanitized text."""
        return list(dict.fromkeys(re.findall(r"<([A-Z_]+)>", sanitized)))

    # ── Warm-up helper ────────────────────────────────────────────────────────
    @classmethod
    def warm_up(cls) -> None:
        """Pre-load the Presidio NLP model (call once before timing tests)."""
        cls._ensure_loaded()
        cls._analyzer.analyze("Warm-up pass.", language="en")  # type: ignore[union-attr]
