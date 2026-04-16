from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RETRIEVERS = ["minilm", "bm25", "hybrid"]
LABELS     = {"minilm": "MiniLM\n(dense)", "bm25": "BM25\n(sparse)", "hybrid": "Hybrid"}
COLORS     = {"baseline": "#1D9E75", "attack": "#D85A30", "defense": "#534AB7"}

def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"  [missing] {path}"); return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _judge(path: Path) -> Optional[Dict]:
    d = _load(path)
    return d.get("summary") if d else None

def _bar_group(ax, groups, series: Dict[str, List], ylabel: str, title: str) -> None:
    n, ns = len(groups), len(series)
    w     = 0.22
    offs  = np.linspace(-w*(ns-1)/2, w*(ns-1)/2, ns)
    x     = np.arange(n)
    for off, (label, vals) in zip(offs, series.items()):
        safe = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x+off, safe, w, label=label, color=COLORS.get(label,"#888"), alpha=0.85)
        for b, v, orig in zip(bars, safe, vals):
            if orig is not None:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LABELS.get(g,g) for g in groups], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--judge-dir",   type=Path, default=Path("results/judge"))
    parser.add_argument("--output-dir",  type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recall5:    Dict[str, List] = {c: [] for c in ["baseline","attack","defense"]}
    spoof_win:  List = []
    correctness: Dict[str, List] = {c: [] for c in ["baseline","attack","defense"]}
    misleading:  Dict[str, List] = {c: [] for c in ["attack","defense"]}

    for ret in RETRIEVERS:
        for cond in ["baseline","attack","defense"]:
            m = _load(args.results_dir / f"{ret}_{cond}_metrics.json")
            recall5[cond].append(m.get("recall@5") if m else None)
            s = _judge(args.judge_dir / f"{ret}_{cond}_judged.json")
            correctness[cond].append(s.get("top1_answer_correctness") if s else None)
        atk = _load(args.results_dir / f"{ret}_attack_metrics.json")
        spoof_win.append(atk.get("top1_spoof_win_rate") if atk else None)
        for cond in ["attack","defense"]:
            s = _judge(args.judge_dir / f"{ret}_{cond}_judged.json")
            misleading[cond].append(s.get("top1_misleading_rate") if s else None)

    # Figure 1: Recall@5
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar_group(ax, RETRIEVERS, recall5, "Recall@5", "Recall@5 — baseline vs attack vs defense")
    fig.tight_layout(); fig.savefig(args.output_dir/"recall_at_5.png", dpi=150); plt.close(fig)
    print(f"Saved: {args.output_dir}/recall_at_5.png")

    # Figure 2: Spoof win rate
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(RETRIEVERS))
    safe = [v if v is not None else 0.0 for v in spoof_win]
    bars = ax.bar(x, safe, color=COLORS["attack"], alpha=0.85, width=0.5)
    for b, v, orig in zip(bars, safe, spoof_win):
        if orig is not None:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[r] for r in RETRIEVERS], fontsize=9)
    ax.set_ylabel("Top-1 spoof win rate"); ax.set_title("Top-1 spoof win rate under attack")
    ax.set_ylim(0, 1.1); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(args.output_dir/"top1_spoof_win_rate.png", dpi=150); plt.close(fig)
    print(f"Saved: {args.output_dir}/top1_spoof_win_rate.png")

    # Figure 3: LLM judge (if available)
    if any(v is not None for vals in correctness.values() for v in vals):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _bar_group(axes[0], RETRIEVERS, correctness, "Answer correctness", "LLM Judge: answer correctness")
        _bar_group(axes[1], RETRIEVERS, misleading,  "Misleading rate",    "LLM Judge: misleading rate")
        fig.tight_layout(); fig.savefig(args.output_dir/"llm_judge.png", dpi=150); plt.close(fig)
        print(f"Saved: {args.output_dir}/llm_judge.png")

    # Summary table
    fmt = lambda v: f"{v:.3f}" if v is not None else "  N/A"
    print("\n" + "="*68)
    print(f"{'Retriever':<18} {'R@5 base':>9} {'R@5 atk':>9} {'R@5 def':>9} {'SpWin':>7}")
    print("-"*68)
    for i, ret in enumerate(RETRIEVERS):
        print(f"{LABELS[ret].replace(chr(10),' '):<18} {fmt(recall5['baseline'][i]):>9} "
              f"{fmt(recall5['attack'][i]):>9} {fmt(recall5['defense'][i]):>9} {fmt(spoof_win[i]):>7}")
    print("="*68)

if __name__ == "__main__":
    main()
