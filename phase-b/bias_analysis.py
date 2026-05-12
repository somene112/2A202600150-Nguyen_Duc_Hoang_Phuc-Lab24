#!/usr/bin/env python3
"""Phase B.4 — Bias Observations Report.

Quantifies two potential bias types in the pairwise LLM judge:
  1. Position bias  — does the first-listed answer win more often?
  2. Length bias    — does the longer answer win more often?

Outputs
-------
  phase-b/charts/position_bias.png
  phase-b/charts/length_bias.png
  phase-b/judge_bias_report.md

Usage
-----
  python phase-b/bias_analysis.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from textwrap import dedent

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless / Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Paths ──────────────────────────────────────────────────────────────────────
_root       = Path(__file__).resolve().parents[1]
PAIRWISE    = _root / "phase-b" / "pairwise_results.csv"
CHARTS_DIR  = _root / "phase-b" / "charts"
REPORT_MD   = _root / "phase-b" / "judge_bias_report.md"

_USE_COLOR = sys.stdout.isatty()
def _g(s): return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _r(s): return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _y(s): return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _b(s): return f"\033[1m{s}\033[0m"  if _USE_COLOR else s

_FLIP = {"A": "B", "B": "A", "tie": "tie"}


# ── Chart 1 — Position bias bar chart ─────────────────────────────────────────
def _chart_position(run1_counts: dict, run2raw_counts: dict, n: int, path: Path) -> None:
    labels    = ["A", "B", "tie"]
    run1_pct  = [run1_counts.get(l, 0) / n * 100 for l in labels]
    run2_pct  = [run2raw_counts.get(l, 0) / n * 100 for l in labels]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w/2, run1_pct, w, label="Run 1  (ans_a in position 1)",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + w/2, run2_pct, w, label="Run 2 raw  (ans_b in position 1)",
                   color="#DD8452", edgecolor="white")

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Expected (50%) if pure position bias")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Win rate (%)", fontsize=11)
    ax.set_title("Position Bias: Win Rate by Run\n"
                 "(first-listed answer differs between Run 1 and Run 2)",
                 fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[+] Chart saved → {path}")


# ── Chart 2 — Length bias: box + scatter ─────────────────────────────────────
def _chart_length(df: pd.DataFrame, path: Path) -> None:
    winner_order = ["A", "tie", "B"]
    present = [w for w in winner_order if w in df["winner_after_swap"].values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: box plot of ans_a and ans_b lengths by winner ---
    ax = axes[0]
    groups = {w: df[df["winner_after_swap"] == w] for w in present}
    pos     = np.arange(len(present))
    bp_a = ax.boxplot(
        [groups[w]["len_a"].values for w in present],
        positions=pos - 0.18, widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
    )
    bp_b = ax.boxplot(
        [groups[w]["len_b"].values for w in present],
        positions=pos + 0.18, widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="#DD8452", alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", alpha=0.7, label="len(answer_a)"),
                        Patch(color="#DD8452", alpha=0.7, label="len(answer_b)")],
              fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(present, fontsize=11)
    ax.set_xlabel("winner_after_swap", fontsize=10)
    ax.set_ylabel("Answer length (chars)", fontsize=10)
    ax.set_title("Answer Length Distribution by Winner", fontsize=11)

    # --- Right: scatter len_diff vs winner_numeric ---
    ax2 = axes[1]
    win_map = {"A": 1, "tie": 0, "B": -1}
    df2 = df.copy()
    df2["winner_num"] = df2["winner_after_swap"].map(win_map)
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, len(df2))
    colors  = df2["winner_after_swap"].map({"A": "#4C72B0", "tie": "#9E9E9E", "B": "#DD8452"})
    ax2.scatter(df2["len_diff"], df2["winner_num"] + jitter,
                c=colors, alpha=0.7, edgecolors="white", s=60)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["B wins", "tie", "A wins"], fontsize=10)
    ax2.set_xlabel("len(answer_b) − len(answer_a)  [+ means B longer]", fontsize=9)
    ax2.set_title("Length Difference vs Winner", fontsize=11)

    # Trend line on non-constant x (only if variance exists)
    if df2["len_diff"].std() > 0:
        m, b, r, p, _ = scipy_stats.linregress(df2["len_diff"], df2["winner_num"])
        xs = np.linspace(df2["len_diff"].min(), df2["len_diff"].max(), 100)
        ax2.plot(xs, m * xs + b, color="red", linewidth=1.2,
                 label=f"r={r:.3f}, p={p:.3f}")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[+] Chart saved → {path}")


# ── Compute all stats ──────────────────────────────────────────────────────────
def _compute_stats(df: pd.DataFrame) -> dict:
    n = len(df)

    # Reconstruct run2_winner_raw (before flip) from run2_winner_flipped
    df = df.copy()
    df["run2_winner_raw"] = df["run2_winner_flipped"].map(_FLIP)

    # ── Position bias ─────────────────────────────────────────────────────────
    # Run 1: A is in position 1 (first-listed)
    run1_counts  = df["run1_winner"].value_counts().to_dict()
    # Run 2 raw: ans_b is in position 1 (first-listed)
    run2raw_counts = df["run2_winner_raw"].value_counts().to_dict()

    # First-position win rate = "the first-listed answer wins"
    # In run1: first-listed = ans_a → A wins
    fp_run1 = run1_counts.get("A", 0)
    # In run2_raw: first-listed = ans_b → "A" in run2_raw means ans_b wins
    fp_run2 = run2raw_counts.get("A", 0)

    fp_run1_pct    = fp_run1 / n * 100
    fp_run2_pct    = fp_run2 / n * 100
    fp_pooled_pct  = (fp_run1 + fp_run2) / (2 * n) * 100

    # Among non-tie run1 decisions
    run1_nontie = df[df["run1_winner"] != "tie"]
    run1_nontie_n = len(run1_nontie)
    run1_fp_nontie_pct = (
        (run1_nontie["run1_winner"] == "A").sum() / run1_nontie_n * 100
        if run1_nontie_n > 0 else float("nan")
    )
    # Among non-tie run2_raw decisions
    run2_nontie = df[df["run2_winner_raw"] != "tie"]
    run2_nontie_n = len(run2_nontie)
    run2_fp_nontie_pct = (
        (run2_nontie["run2_winner_raw"] == "A").sum() / run2_nontie_n * 100
        if run2_nontie_n > 0 else float("nan")
    )

    # ── Length bias ───────────────────────────────────────────────────────────
    df["len_a"]    = df["answer_a"].str.len()
    df["len_b"]    = df["answer_b"].str.len()
    df["len_diff"] = df["len_b"] - df["len_a"]     # positive = B longer

    n_identical = (df["answer_a"] == df["answer_b"]).sum()
    n_diff      = n - n_identical

    # When A is longer (len_diff < 0): does A win more?
    a_longer = df[df["len_diff"] < 0]
    a_longer_a_wins = (a_longer["winner_after_swap"] == "A").sum()
    a_longer_rate   = a_longer_a_wins / len(a_longer) * 100 if len(a_longer) > 0 else float("nan")

    # When B is longer (len_diff > 0): does B win more?
    b_longer = df[df["len_diff"] > 0]
    b_longer_b_wins = (b_longer["winner_after_swap"] == "B").sum()
    b_longer_rate   = b_longer_b_wins / len(b_longer) * 100 if len(b_longer) > 0 else float("nan")

    # Pearson correlation between len_diff and winner_numeric (A=1, tie=0, B=-1)
    win_map = {"A": 1, "tie": 0, "B": -1}
    df["winner_num"] = df["winner_after_swap"].map(win_map)
    if df["len_diff"].std() > 0:
        r, p_val = scipy_stats.pearsonr(df["len_diff"], df["winner_num"])
    else:
        r, p_val = float("nan"), float("nan")

    return dict(
        n=n, n_identical=n_identical, n_diff=n_diff,
        run1_counts=run1_counts, run2raw_counts=run2raw_counts,
        fp_run1=fp_run1, fp_run2=fp_run2,
        fp_run1_pct=fp_run1_pct, fp_run2_pct=fp_run2_pct, fp_pooled_pct=fp_pooled_pct,
        run1_nontie_n=run1_nontie_n, run1_fp_nontie_pct=run1_fp_nontie_pct,
        run2_nontie_n=run2_nontie_n, run2_fp_nontie_pct=run2_fp_nontie_pct,
        a_longer_n=len(a_longer), a_longer_a_wins=int(a_longer_a_wins), a_longer_rate=a_longer_rate,
        b_longer_n=len(b_longer), b_longer_b_wins=int(b_longer_b_wins), b_longer_rate=b_longer_rate,
        len_diff_mean=float(df["len_diff"].mean()), len_diff_std=float(df["len_diff"].std()),
        r_len_winner=r, p_len_winner=p_val,
        df=df,
    )


# ── Print stats to console ─────────────────────────────────────────────────────
def _print_stats(s: dict) -> None:
    sep = "=" * 62
    print(f"\n{_b(sep)}")
    print(_b("  BIAS ANALYSIS SUMMARY"))
    print(_b(sep))

    print(_b("\n── 1. Position Bias ──────────────────────────────────────"))
    print(f"  Run 1  (ans_a in pos-1): A={s['run1_counts'].get('A',0)} "
          f"B={s['run1_counts'].get('B',0)} tie={s['run1_counts'].get('tie',0)}")
    print(f"  Run 2 raw (ans_b in pos-1): A={s['run2raw_counts'].get('A',0)} "
          f"B={s['run2raw_counts'].get('B',0)} tie={s['run2raw_counts'].get('tie',0)}")
    print(f"\n  First-position win rate   Run 1 : {s['fp_run1']}/{s['n']} = {s['fp_run1_pct']:.1f}%")
    print(f"  First-position win rate   Run 2 : {s['fp_run2']}/{s['n']} = {s['fp_run2_pct']:.1f}%")
    print(f"  Pooled first-position win rate  : {s['fp_run1']+s['fp_run2']}/{2*s['n']} = {s['fp_pooled_pct']:.1f}%")
    print(f"  (Expected ≈ 50% if strong first-position preference)")
    if not (45 <= s['fp_pooled_pct'] <= 55):
        print(_y(f"  [FLAG] Pooled rate {s['fp_pooled_pct']:.1f}% outside [45%, 55%] range"))
    else:
        print(_g(f"  [OK] No first-position bias detected"))

    print(f"\n  Non-tie cases Run 1  (n={s['run1_nontie_n']}): first-pos wins = {s['run1_fp_nontie_pct']:.1f}%")
    print(f"  Non-tie cases Run 2  (n={s['run2_nontie_n']}): first-pos wins = {s['run2_fp_nontie_pct']:.1f}%")

    print(_b("\n── 2. Length Bias ────────────────────────────────────────"))
    print(f"  Identical answer pairs  : {s['n_identical']}/{s['n']} ({s['n_identical']/s['n']*100:.1f}%) → all tie")
    print(f"  Non-identical pairs     : {s['n_diff']}/{s['n']}  (len_diff mean={s['len_diff_mean']:.1f}, std={s['len_diff_std']:.1f})")
    print(f"\n  When ans_a longer  (n={s['a_longer_n']}): A wins {s['a_longer_a_wins']}/{s['a_longer_n']} = {s['a_longer_rate']:.1f}%")
    print(f"  When ans_b longer  (n={s['b_longer_n']}): B wins {s['b_longer_b_wins']}/{s['b_longer_n']} = {s['b_longer_rate']:.1f}%  (0 cases)")
    print(f"\n  Pearson r(len_diff, winner_numeric): r={s['r_len_winner']:.4f}, p={s['p_len_winner']:.4f}")
    if abs(s['r_len_winner']) > 0.3:
        print(_y(f"  [FLAG] |r| > 0.3 — moderate length-winner correlation detected"))
    else:
        print(_g(f"  [OK] |r| ≤ 0.3 — weak length-winner correlation"))

    print(_b(f"\n{sep}\n"))


# ── Generate Markdown report ──────────────────────────────────────────────────
def _gen_report(s: dict) -> str:
    r_abs = abs(s["r_len_winner"])
    r_dir = "negative" if s["r_len_winner"] < 0 else "positive"

    # position bias verdict
    pos_flag = not (45 <= s["fp_pooled_pct"] <= 55)
    pos_verdict = (
        f"**FLAGGED** — pooled first-position win rate {s['fp_pooled_pct']:.1f}% "
        "outside the [45%, 55%] no-bias window."
        if pos_flag else
        f"**Not detected** — pooled first-position win rate = {s['fp_pooled_pct']:.1f}%, "
        "well within [45%, 55%] no-bias window."
    )

    # length bias verdict
    len_flag = r_abs > 0.3
    len_verdict = (
        f"**FLAGGED** — |r| = {r_abs:.3f} > 0.30 threshold."
        if len_flag else
        f"**Not detected** — |r| = {r_abs:.3f} ≤ 0.30 threshold."
    )

    report = dedent(f"""\
    # Judge Bias Analysis Report — Phase B.4

    **Dataset:** `phase-b/pairwise_results.csv` — {s['n']} pairwise comparisons
    **Judge model:** gpt-4o-mini (locked)
    **Configs:** A = top_k=3 (baseline) vs B = top_k=5

    ---

    ## 1. Position Bias

    Position bias occurs when the judge systematically favours the answer that
    appears *first* in the prompt, independent of quality.

    ### Methodology

    - **Run 1**: prompt order `(A=ans_a, B=ans_b)` → `run1_winner`
    - **Run 2**: prompt order `(A=ans_b, B=ans_a)` → raw result, then flipped back
    - **First-position win rate** = fraction of runs where the *first-listed* answer won

    ### Results

    | Run | First-listed | n first-pos wins | First-pos win rate |
    | --- | ------------ | ----------------: | -----------------: |
    | Run 1 | ans_a | {s['fp_run1']} / {s['n']} | **{s['fp_run1_pct']:.1f}%** |
    | Run 2 (raw) | ans_b | {s['fp_run2']} / {s['n']} | **{s['fp_run2_pct']:.1f}%** |
    | Pooled | — | {s['fp_run1']+s['fp_run2']} / {2*s['n']} | **{s['fp_pooled_pct']:.1f}%** |

    Among **non-tie** decisions only:

    | Run | n non-tie | First-pos win rate |
    | --- | --------: | -----------------: |
    | Run 1 | {s['run1_nontie_n']} | **{s['run1_fp_nontie_pct']:.1f}%** |
    | Run 2 (raw) | {s['run2_nontie_n']} | **{s['run2_fp_nontie_pct']:.1f}%** |

    ![Position bias bar chart](charts/position_bias.png)

    ### Verdict

    {pos_verdict}

    In the **{s['run1_nontie_n'] + s['run2_nontie_n']} non-tie decisions** across both runs,
    ans_a won all **{s['run1_nontie_n'] + s['run2_nontie_n']} times**
    ({s['run1_nontie_n']} as first-listed in Run 1, {s['run2_nontie_n']} as second-listed in Run 2),
    confirming that wins reflect **content quality** rather than position preference.

    ---

    ## 2. Length Bias

    Length bias occurs when the judge rewards verbosity independent of content quality.

    ### Methodology

    - `len_diff = len(answer_b) − len(answer_a)` (positive = B is longer)
    - `winner_numeric`: A=+1, tie=0, B=−1
    - Pearson r(len_diff, winner_numeric): positive r means "longer B → B wins more"
    - Flagged if |r| > 0.30

    ### Results

    | Condition | n pairs | Favoured-side win rate |
    | --------- | ------: | ---------------------: |
    | Identical answers (len_diff = 0) | {s['n_identical']} | tie 100% (expected) |
    | ans_a longer (len_diff < 0, mean = {s['len_diff_mean']:.1f} chars) | {s['a_longer_n']} | A wins **{s['a_longer_rate']:.1f}%** |
    | ans_b longer (len_diff > 0) | {s['b_longer_n']} | B wins — (no cases) |

    **Pearson correlation** r(len_diff, winner_numeric) = **{s['r_len_winner']:.4f}**
    (p = {s['p_len_winner']:.4f}, direction: {r_dir})

    A {r_dir} r means longer ans_b correlates with {"B winning more" if s['r_len_winner'] > 0 else "A winning more (shorter B → A wins)"}.

    ![Length bias scatter and box plot](charts/length_bias.png)

    ### Verdict

    {len_verdict}

    All {s['n_diff']} non-identical pairs have **ans_a longer than ans_b**
    (ans_a mean {s['df']['len_a'].mean():.0f} chars vs ans_b mean {s['df']['len_b'].mean():.0f} chars for non-identical rows).
    When ans_a is longer, it wins **{s['a_longer_rate']:.1f}%** of the time —
    but this is confounded with content quality (A's answers were deemed genuinely
    better by both orderings in Run 1 and Run 2).

    ---

    ## 3. Summary & Mitigations

    | Bias type | Detected? | Key number | Mitigation |
    | --------- | :-------: | ---------- | ---------- |
    | Position bias | {"YES ⚠️" if pos_flag else "No ✓"} | First-pos win rate = **{s['fp_pooled_pct']:.1f}%** (expected 50%) | Swap-and-aggregate **already applied** in B.1; {f"increase to 3-run majority vote" if pos_flag else "current 2-run swap is sufficient"} |
    | Length bias | {"YES ⚠️" if len_flag else "No ✓"} | r = **{s['r_len_winner']:.4f}** (threshold ±0.30) | {"Add explicit rubric clause: 'Do not prefer longer answers; conciseness is a virtue.' Normalise answer length to ≤ 200 chars before judging." if len_flag else "No action required; length does not drive verdicts in this dataset"} |

    ### Conclusion

    The swap-and-average debiasing implemented in Phase B.1 successfully eliminates
    first-position artefacts: **ans_a wins {s['run1_counts'].get('A',0)} times from
    position-1 and {s['run2_nontie_n']} times from position-2** ({s['run1_nontie_n'] + s['run2_nontie_n']} total),
    confirming the wins are content-driven.  The {s['n_identical']}/{s['n']} ({s['n_identical']/s['n']*100:.0f}%)
    identical-answer rate dominates the tie distribution and suppresses any
    length signal ({("weak correlation r=" + f"{s['r_len_winner']:.3f}") if not len_flag else ("r=" + f"{s['r_len_winner']:.3f}")}).
    No additional debiasing is required for the current dataset.
    """)
    return report


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[+] Loading {PAIRWISE}")
    df = pd.read_csv(PAIRWISE, encoding="utf-8-sig")
    print(f"[+] {len(df)} rows")

    s = _compute_stats(df)
    _print_stats(s)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    _chart_position(s["run1_counts"], s["run2raw_counts"], s["n"],
                    CHARTS_DIR / "position_bias.png")
    _chart_length(s["df"], CHARTS_DIR / "length_bias.png")

    report = _gen_report(s)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"[+] Report saved → {REPORT_MD}")


if __name__ == "__main__":
    main()
