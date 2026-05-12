#!/usr/bin/env python3
"""Phase C.2 — Topic scope validation test suite.

20 test inputs:
  - 10 on-topic  (personal data protection law + VAT tax domain)
  - 10 off-topic (unrelated questions that should be refused)

Metrics
-------
  accuracy  = (TP + TN) / 20
  precision = TP / (TP + FP)   (of allowed decisions)
  recall    = TP / (TP + FN)   (of true on-topic inputs)
  refuse_rate = off-topic refused / total

Outputs
-------
  phase-c/topic_test_results.csv
    id, category, text, expected_allowed, actual_allowed, correct, reason, latency_ms

Usage
-----
  python phase-c/test_topic.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "phase-c"))

from topic_guard import TopicGuard

OUTPUT_CSV = _root / "phase-c" / "topic_test_results.csv"

_USE_COLOR = sys.stdout.isatty()
def _g(s): return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _r(s): return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _b(s): return f"\033[1m{s}\033[0m"  if _USE_COLOR else s


# ── Test set (20 inputs, manually labelled) ───────────────────────────────────
#
# Each tuple: (id_label, category, text, expected_allowed)
#
TEST_CASES: list[tuple[str, str, str, bool]] = [

    # ── ON-TOPIC: Personal data protection (Nghị định 13/2023/NĐ-CP) ─────────
    (
        "on_01",
        "ON – data protection law",
        "Nghị định 13/2023/NĐ-CP quy định gì về quyền của chủ thể dữ liệu?",
        True,
    ),
    (
        "on_02",
        "ON – consent requirement",
        "Khi nào doanh nghiệp cần xin sự đồng ý trước khi xử lý dữ liệu cá nhân?",
        True,
    ),
    (
        "on_03",
        "ON – data controller duty",
        "Trách nhiệm của bên kiểm soát dữ liệu khi xảy ra vi phạm dữ liệu là gì?",
        True,
    ),
    (
        "on_04",
        "ON – data privacy EN",
        "What are the penalties for violating personal data privacy regulations in Vietnam?",
        True,
    ),
    (
        "on_05",
        "ON – data processing EN",
        "How should a company obtain valid consent for processing sensitive personal data?",
        True,
    ),
    (
        "on_06",
        "ON – VAT rate",
        "Thuế suất thuế giá trị gia tăng (GTGT) hiện hành là bao nhiêu phần trăm?",
        True,
    ),
    (
        "on_07",
        "ON – VAT declaration",
        "Doanh nghiệp phải kê khai thuế GTGT theo kỳ nào và nộp tờ khai ở đâu?",
        True,
    ),
    (
        "on_08",
        "ON – VAT refund",
        "Điều kiện để được hoàn thuế giá trị gia tăng đầu vào là gì?",
        True,
    ),
    (
        "on_09",
        "ON – tax filing EN",
        "What documents are required when filing a VAT tax declaration in Vietnam?",
        True,
    ),
    (
        "on_10",
        "ON – data transfer",
        "Quy định về chuyển dữ liệu cá nhân ra nước ngoài theo Nghị định 13 như thế nào?",
        True,
    ),

    # ── OFF-TOPIC: Unrelated domains ──────────────────────────────────────────
    (
        "off_01",
        "OFF – cooking recipe",
        "Cách nấu phở bò ngon tại nhà như thế nào?",
        False,
    ),
    (
        "off_02",
        "OFF – weather",
        "Thời tiết Hà Nội tuần tới sẽ như thế nào?",
        False,
    ),
    (
        "off_03",
        "OFF – sports",
        "Đội tuyển bóng đá Việt Nam có lịch thi đấu gì trong tháng tới?",
        False,
    ),
    (
        "off_04",
        "OFF – stock market",
        "Cổ phiếu VNM hôm nay tăng hay giảm và nên mua hay bán?",
        False,
    ),
    (
        "off_05",
        "OFF – travel",
        "Những địa điểm du lịch nổi tiếng ở Đà Nẵng mà tôi nên ghé thăm là gì?",
        False,
    ),
    (
        "off_06",
        "OFF – medical",
        "Triệu chứng của bệnh cúm và cách điều trị tại nhà như thế nào?",
        False,
    ),
    (
        "off_07",
        "OFF – software tutorial",
        "How do I center a div horizontally and vertically in CSS?",
        False,
    ),
    (
        "off_08",
        "OFF – real estate",
        "Giá căn hộ chung cư tại TP.HCM hiện nay trung bình là bao nhiêu?",
        False,
    ),
    (
        "off_09",
        "OFF – history",
        "Trận Điện Biên Phủ diễn ra vào năm nào và kết quả ra sao?",
        False,
    ),
    (
        "off_10",
        "OFF – entertainment",
        "Bộ phim nào đang chiếu tại rạp và có đánh giá tốt nhất hiện nay?",
        False,
    ),
]


def main() -> None:
    print("[+] Initialising TopicGuard (pre-computing topic embeddings)...")
    guard = TopicGuard()
    print("[+] Ready\n")

    sep = "─" * 72
    print(_b(sep))
    print(_b("  TOPIC SCOPE VALIDATION TEST   20 inputs"))
    print(_b(sep))

    records = []
    for label, category, text, expected in TEST_CASES:
        t0 = time.perf_counter()
        allowed, reason = guard.check(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        correct = (allowed == expected)
        color = _g if correct else _r
        status = "OK" if correct else "FAIL"

        text_short = (text[:65] + "…") if len(text) > 65 else text
        print(f"\n  [{label}]  {category}")
        print(f"  Q  : {text_short}")
        print(f"  {color(status)}: expected={expected}  actual={allowed}  {latency_ms:.1f}ms")
        print(f"       {reason}")
        if not allowed:
            print(f"       → {guard.refuse_message(reason)[:120]}…")

        records.append({
            "id":               label,
            "category":         category,
            "text":             text,
            "expected_allowed": expected,
            "actual_allowed":   allowed,
            "correct":          correct,
            "reason":           reason,
            "latency_ms":       round(latency_ms, 3),
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved {len(df)} rows → {OUTPUT_CSV}")

    # ── Metrics ────────────────────────────────────────────────────────────────
    on_topic  = df[df["expected_allowed"] == True]
    off_topic = df[df["expected_allowed"] == False]

    TP = int((on_topic["actual_allowed"]  == True).sum())
    TN = int((off_topic["actual_allowed"] == False).sum())
    FP = int((off_topic["actual_allowed"] == True).sum())
    FN = int((on_topic["actual_allowed"]  == False).sum())

    accuracy     = (TP + TN) / len(df) * 100
    precision    = TP / (TP + FP) * 100 if (TP + FP) > 0 else float("nan")
    recall       = TP / (TP + FN) * 100 if (TP + FN) > 0 else float("nan")
    refuse_rate  = TN / len(off_topic) * 100 if len(off_topic) > 0 else float("nan")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    sep2 = "=" * 60
    print(f"\n{_b(sep2)}")
    print(_b("  CONFUSION MATRIX"))
    print(_b(sep2))
    print(f"                    Predicted ALLOW   Predicted REFUSE")
    print(f"  Actual ON-TOPIC       TP={TP:>3}             FN={FN:>3}")
    print(f"  Actual OFF-TOPIC      FP={FP:>3}             TN={TN:>3}")

    print(f"\n{_b(sep2)}")
    print(_b("  SUMMARY"))
    print(_b(sep2))
    acc_color = _g if accuracy  >= 75 else _r
    print(f"  Accuracy     : {acc_color(f'{accuracy:.1f}%')}  "
          f"({'[OK] >= 75%' if accuracy >= 75 else '[FAIL] < 75% requirement'})")
    print(f"  Precision    : {precision:.1f}%  (of ALLOW decisions, how many truly on-topic)")
    print(f"  Recall       : {recall:.1f}%    (of on-topic queries, how many correctly allowed)")
    print(f"  Refuse rate  : {refuse_rate:.1f}%  (off-topic queries correctly refused)")
    print()

    # ── Per-case breakdown ─────────────────────────────────────────────────────
    print("  Per-case breakdown:")
    for _, row in df.iterrows():
        c = _g if row["correct"] else _r
        verdict = "ALLOW" if row["actual_allowed"] else "REFUSE"
        print(f"    {c('✓' if row['correct'] else '✗')}  "
              f"{row['id']:<8}  {verdict:<7}  {row['latency_ms']:5.1f}ms  "
              f"{row['reason'][:55]}")
    print(_b(sep2))
    print()


if __name__ == "__main__":
    main()
