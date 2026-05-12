#!/usr/bin/env python3
"""Phase C.5 — Latency benchmark for GuardedRAGPipeline.

Corpus
------
  53  real questions from phase-a/testset_v1.csv
  47  synthetic domain queries (data protection + VAT)
  ─── 100 total

Two runs per query
------------------
  Guarded  : L1 + L2 + L3   (full pipeline)
  Baseline : L2 only        (raw RAG, no guards)

Parallelism verification
------------------------
  L1 is confirmed truly parallel when:
    wall_time < InputGuard_ms + TopicGuard_ms
  (i.e. wall time is less than the sum of sequential components)

Outputs
-------
  phase-c/latency_benchmark.csv
  phase-c/latency_summary.json

Targets
-------
  L1  P95 < 50 ms   (excellent < 30 ms)
  L3  P95 < 100 ms  (excellent < 50 ms)

Usage
-----
  python phase-c/benchmark_latency.py
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from time import perf_counter

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "phase-c"))

from full_pipeline import GuardedRAGPipeline

OUTPUT_CSV  = _root / "phase-c" / "latency_benchmark.csv"
OUTPUT_JSON = _root / "phase-c" / "latency_summary.json"

_USE_COLOR = sys.stdout.isatty()
def _g(s): return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _r(s): return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _b(s): return f"\033[1m{s}\033[0m"  if _USE_COLOR else s
def _y(s): return f"\033[93m{s}\033[0m" if _USE_COLOR else s


# ── 47 synthetic domain queries ───────────────────────────────────────────────
_SYNTHETIC: list[str] = [
    # Data protection — Vietnamese (15)
    "Nghị định 13/2023 quy định quyền nào cho chủ thể dữ liệu?",
    "Bên kiểm soát dữ liệu phải lưu giữ hồ sơ xử lý dữ liệu trong bao lâu?",
    "Khi nào cần lập đánh giá tác động xử lý dữ liệu cá nhân?",
    "Dữ liệu cá nhân nhạy cảm bao gồm những loại nào theo Nghị định 13?",
    "Điều kiện để bên xử lý dữ liệu hoạt động hợp pháp là gì?",
    "Biện pháp bảo vệ dữ liệu cá nhân theo mặc định là gì?",
    "Xử lý dữ liệu cá nhân của trẻ em cần điều kiện đặc biệt nào?",
    "Thủ tục đăng ký hoạt động xử lý dữ liệu cá nhân ra nước ngoài như thế nào?",
    "Mức phạt vi phạm quy định bảo vệ dữ liệu cá nhân là bao nhiêu?",
    "Quyền từ chối cung cấp dữ liệu cá nhân được áp dụng trong trường hợp nào?",
    "Bên xử lý dữ liệu có thể ủy quyền cho bên thứ ba không?",
    "Yêu cầu bảo mật kỹ thuật đối với hệ thống lưu trữ dữ liệu cá nhân là gì?",
    "Cách thức thực hiện quyền xóa dữ liệu theo Nghị định 13 như thế nào?",
    "Thông báo vi phạm dữ liệu cá nhân cần nộp cho cơ quan nào?",
    "Hợp đồng xử lý dữ liệu giữa bên kiểm soát và bên xử lý cần nội dung gì?",

    # Data protection — English (10)
    "What legal bases allow processing of personal data under Decree 13?",
    "How must consent be obtained for processing sensitive personal data?",
    "What records must a data controller maintain under Vietnamese law?",
    "When is a data protection impact assessment mandatory?",
    "What are the cross-border data transfer requirements under Decree 13?",
    "How long must consent records be retained under Vietnamese personal data law?",
    "What are the penalties for unauthorised disclosure of personal data?",
    "What information must be included in a privacy notice under Decree 13?",
    "How should a data subject's access request be handled?",
    "What security standards apply to personal data processing systems?",

    # VAT tax — Vietnamese (12)
    "Thuế GTGT đầu ra được tính như thế nào trên hóa đơn bán hàng?",
    "Doanh nghiệp xuất khẩu hàng hóa áp dụng thuế suất GTGT bao nhiêu?",
    "Điều kiện khấu trừ thuế GTGT đầu vào là gì?",
    "Hàng hóa, dịch vụ không chịu thuế GTGT bao gồm những loại nào?",
    "Tờ khai thuế GTGT mẫu 01/GTGT cần kê khai những chỉ tiêu nào?",
    "Thời hạn nộp thuế GTGT đối với doanh nghiệp khai thuế theo quý là bao nhiêu?",
    "Điều kiện hoàn thuế GTGT đối với dự án đầu tư mới như thế nào?",
    "Phương pháp khấu trừ thuế GTGT áp dụng cho đối tượng nào?",
    "Hóa đơn điện tử cần có những thông tin bắt buộc nào để được khấu trừ GTGT?",
    "Dịch vụ phần mềm cung cấp qua internet chịu thuế GTGT bao nhiêu phần trăm?",
    "Cách xác định giá tính thuế GTGT đối với hàng nhập khẩu như thế nào?",
    "Doanh nghiệp mới thành lập có được khai thuế GTGT theo quý không?",

    # VAT tax — English (10)
    "What is the reduced VAT rate for essential goods in Vietnam?",
    "Which services are exempt from Vietnamese VAT?",
    "How are VAT credits carried forward when input exceeds output?",
    "What documentation is required to claim an input VAT deduction?",
    "How does Vietnam VAT apply to digital services from foreign providers?",
    "What are the VAT registration thresholds for small businesses?",
    "How is VAT calculated for mixed-use goods and services?",
    "What penalties apply for late VAT return filing in Vietnam?",
    "Can a VAT refund be claimed within the same tax period?",
    "What is the VAT treatment for goods sold at zero-rated exports?",
]


# ── Load queries ──────────────────────────────────────────────────────────────
def load_queries() -> list[str]:
    testset_path = _root / "phase-a" / "testset_v1.csv"
    real: list[str] = []
    if testset_path.exists():
        df = pd.read_csv(testset_path, encoding="utf-8-sig")
        real = df["question"].dropna().tolist()
        print(f"[+] Loaded {len(real)} questions from testset_v1.csv")
    else:
        print("[!] testset_v1.csv not found — using synthetic only")

    combined = real + _SYNTHETIC
    # Deduplicate while preserving order
    seen: set[str] = set()
    queries: list[str] = []
    for q in combined:
        if q not in seen:
            seen.add(q)
            queries.append(q)
    print(f"[+] Total unique queries: {len(queries)}")
    return queries[:110]  # cap at 110


# ── Parallelism verifier ──────────────────────────────────────────────────────
async def verify_l1_parallel(pipeline: GuardedRAGPipeline) -> dict:
    """Run a single query and measure InputGuard vs TopicGuard wall time."""
    probe = "Khi nào doanh nghiệp cần xin sự đồng ý trước khi xử lý dữ liệu cá nhân?"

    pii_ms_list:   list[float] = []
    topic_ms_list: list[float] = []

    async def _timed_pii():
        t = perf_counter()
        await asyncio.to_thread(pipeline.input_guard.sanitize, probe)
        pii_ms_list.append((perf_counter() - t) * 1000)

    async def _timed_topic():
        t = perf_counter()
        await asyncio.to_thread(pipeline.topic_guard.check, probe)
        topic_ms_list.append((perf_counter() - t) * 1000)

    wall_start = perf_counter()
    await asyncio.gather(_timed_pii(), _timed_topic())
    wall_ms    = (perf_counter() - wall_start) * 1000

    pii_ms   = pii_ms_list[0]
    topic_ms = topic_ms_list[0]
    seq_ms   = pii_ms + topic_ms
    # True parallel: wall_time should be < sequential sum
    # True parallel: wall < sequential sum (even 1ms faster confirms it).
    # When one task dominates (topic~400ms vs pii~10ms) the theoretical speedup
    # is only 1.02×, so the absolute gap matters more than the ratio.
    is_parallel = wall_ms < seq_ms - 1.0  # at least 1ms faster than sequential

    return {
        "wall_ms":      round(wall_ms, 1),
        "pii_ms":       round(pii_ms, 1),
        "topic_ms":     round(topic_ms, 1),
        "sequential_ms": round(seq_ms, 1),
        "is_parallel":  is_parallel,
        "speedup":      round(seq_ms / max(wall_ms, 0.1), 2),
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────
async def run_all(
    pipeline: GuardedRAGPipeline,
    queries: list[str],
    rl_delay_s: float = 1.5,
) -> list[dict]:
    """Sequential execution with rate-limit pause after each Groq (L3) call.

    Groq free tier allows ~30 req/min for llama-3.3-70b-versatile.
    Sequential + 1.5s delay keeps us well under ~24 req/min.
    """
    results: list[dict] = []
    n = len(queries)
    print(f"[+] Running guarded pipeline on {n} queries (sequential, "
          f"rl_delay={rl_delay_s}s after L3 calls)…")

    for qid, q in enumerate(queries):
        answer, t = await pipeline.guarded_query(q)
        results.append({
            "query_id":   qid,
            "query":      q[:80],
            "L1_ms":      round(t["L1"],    2),
            "L2_ms":      round(t["L2"],    2),
            "L3_ms":      round(t["L3"],    2),
            "total_ms":   round(t["total"], 2),
            "blocked_at": t.get("blocked_at"),
        })
        if (qid + 1) % 10 == 0:
            print(f"    … {qid + 1}/{n}")
        # Only pace when L3 was reached (Groq API was called)
        if t.get("blocked_at") is None:
            await asyncio.sleep(rl_delay_s)

    # Let fire-and-forget audit tasks finish
    await asyncio.sleep(0.3)
    return results


async def run_baseline(pipeline: GuardedRAGPipeline, queries: list[str]) -> list[float]:
    sem = asyncio.Semaphore(10)  # RAG is local, can run more concurrently

    async def one_baseline(q: str) -> float:
        async with sem:
            t0 = perf_counter()
            await asyncio.to_thread(pipeline.rag_fn, q)
            return (perf_counter() - t0) * 1000

    print(f"[+] Running baseline (L2 only) on {len(queries)} queries…")
    return list(await asyncio.gather(*[one_baseline(q) for q in queries]))


# ── Percentile helper ─────────────────────────────────────────────────────────
def pct(arr: list[float], label: str = "") -> dict:
    a = np.array([v for v in arr if v > 0])   # exclude 0 (blocked / skipped)
    if len(a) == 0:
        return {"P50": 0, "P95": 0, "P99": 0, "n": 0, "label": label}
    return {
        "P50": round(float(np.percentile(a, 50)), 1),
        "P95": round(float(np.percentile(a, 95)), 1),
        "P99": round(float(np.percentile(a, 99)), 1),
        "n":   len(a),
        "label": label,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    print(_b("=" * 70))
    print(_b("  LATENCY BENCHMARK — GuardedRAGPipeline"))
    print(_b("=" * 70))

    # ── Init pipeline ──────────────────────────────────────────────────────────
    print("\n[+] Initialising pipeline (loading NLP models)…")
    pipeline = GuardedRAGPipeline()
    pipeline.input_guard.warm_up()
    print("[+] Ready")

    # ── Verify L1 parallelism ─────────────────────────────────────────────────
    print("\n[+] Verifying L1 parallelism…")
    par = await verify_l1_parallel(pipeline)
    color = _g if par["is_parallel"] else _r
    print(
        f"    InputGuard={par['pii_ms']}ms  "
        f"TopicGuard={par['topic_ms']}ms  "
        f"Wall={par['wall_ms']}ms  "
        f"Sequential={par['sequential_ms']}ms"
    )
    verdict = "PARALLEL ✓" if par["is_parallel"] else "SEQUENTIAL ✗ — fix asyncio.gather"
    print(f"    {color(verdict)}  speedup={par['speedup']}×")

    if not par["is_parallel"]:
        # Should not happen — this means asyncio.to_thread isn't running concurrently.
        # Potential fix: ensure the thread pool has enough workers.
        import concurrent.futures
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        loop.set_default_executor(executor)
        print("    [!] Increased thread pool to 16 workers — re-running verify…")
        par = await verify_l1_parallel(pipeline)
        print(f"    Wall={par['wall_ms']}ms  "
              f"is_parallel={par['is_parallel']}  speedup={par['speedup']}×")

    # ── Load queries ──────────────────────────────────────────────────────────
    queries = load_queries()

    # ── Guarded run ───────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    guarded = await run_all(pipeline, queries)
    print(f"    Done in {time.perf_counter()-t_start:.1f}s")

    # ── Baseline run ──────────────────────────────────────────────────────────
    baseline_times = await run_baseline(pipeline, queries)
    print(f"    Baseline done")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(guarded)
    df["baseline_ms"] = [round(b, 2) for b in baseline_times]
    df["overhead_ms"] = (df["total_ms"] - df["baseline_ms"]).round(2)

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[+] Saved {len(df)} rows → {OUTPUT_CSV}")

    # ── Compute stats ─────────────────────────────────────────────────────────
    non_blocked = df[df["blocked_at"].isna()]

    stats = {
        "L1":       pct(df["L1_ms"].tolist(),           "L1 (all)"),
        "L2":       pct(non_blocked["L2_ms"].tolist(),  "L2 (non-blocked)"),
        "L3":       pct(non_blocked["L3_ms"].tolist(),  "L3 (non-blocked)"),
        "total":    pct(df["total_ms"].tolist(),         "Total (all)"),
        "baseline": pct(df["baseline_ms"].tolist(),      "Baseline (L2 only)"),
        "overhead": pct(df["overhead_ms"].tolist(),      "Overhead (guarded-baseline)"),
        "parallelism": par,
        "meta": {
            "n_queries":    len(df),
            "n_blocked_L1": int(df["blocked_at"].str.startswith("L1", na=False).sum()),
            "n_blocked_L3": int((df["blocked_at"] == "L3").sum()),
            "n_passed":     int(non_blocked.shape[0]),
        },
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved summary → {OUTPUT_JSON}")

    # ── Print comparison table ────────────────────────────────────────────────
    targets = {"L1": (50, 30), "L3": (100, 50)}

    def _row(name: str, s: dict, target_p95: int | None = None) -> str:
        p95_c = s["P95"]
        if target_p95:
            ok, excellent = targets.get(name, (None, None)) if name in targets else (None, None)
            if ok is None:
                ok, excellent = target_p95, target_p95 // 2
            color = _g if p95_c < excellent else (_y if p95_c < ok else _r)
            badge = (f"[EXCELLENT <{excellent}ms]" if p95_c < excellent
                     else f"[OK <{ok}ms]"           if p95_c < ok
                     else f"[FAIL ≥{ok}ms]")
        else:
            color = lambda x: x
            badge = ""
        return (f"  {name:<12} P50={s['P50']:>6.0f}ms  P95={color(f'{p95_c:>6.0f}ms')}  "
                f"P99={s['P99']:>6.0f}ms  n={s['n']:>3}  {badge}")

    sep = "─" * 72
    print(f"\n{_b('=' * 72)}")
    print(_b("  LATENCY SUMMARY"))
    print(_b("=" * 72))
    print(f"{_b(sep)}")
    print(_row("L1",       stats["L1"],       50))
    print(_row("L2 (RAG)", stats["L2"],       None))
    print(_row("L3",       stats["L3"],       100))
    print(_row("Total",    stats["total"],    None))
    print(f"{_b(sep)}")
    print(_row("Baseline", stats["baseline"], None))
    print(_row("Overhead", stats["overhead"], None))
    print(_b(sep))
    print()
    print(f"  Queries total    : {stats['meta']['n_queries']}")
    print(f"  Blocked at L1    : {stats['meta']['n_blocked_L1']}")
    print(f"  Blocked at L3    : {stats['meta']['n_blocked_L3']}")
    print(f"  Passed all layers: {stats['meta']['n_passed']}")
    print()
    print(f"  L1 parallelism: {_g('CONFIRMED') if par['is_parallel'] else _r('NOT CONFIRMED')}")
    print(f"    wall={par['wall_ms']}ms < sequential={par['sequential_ms']}ms "
          f"(speedup {par['speedup']}×)")
    print(_b("=" * 72))
    print()


if __name__ == "__main__":
    asyncio.run(main())
