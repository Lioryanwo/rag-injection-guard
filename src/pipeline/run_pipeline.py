from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASELINE_TOP_K = 5
ATTACK_RETRIEVAL_TOP_K = 20
FINAL_KEEP_TOP_K = 5
LLM_JUDGE_TOP_K = 1
LLM_JUDGE_MAX_QUERIES = 100

ATTACK_MAX_QUERIES = 300
ATTACK_MODE = "strong"   # light / medium / strong
ATTACK_CANDIDATES_PER_STYLE = 3
ATTACK_KEEP_PER_STYLE = 1
ATTACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFENSE_THRESHOLD = 0.30
DEFENSE_MODE = "cross_encoder"
DEFENSE_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFENSE_SEMANTIC_WEIGHT = 0.75
DEFENSE_RETRIEVAL_WEIGHT = 0.25
DEFENSE_LEXICAL_PENALTY_WEIGHT = 0.15
DEFENSE_BATCH_SIZE = 16


def run(cmd: list) -> None:
    print("\n" + "=" * 72)
    print("RUN:", " ".join(str(c) for c in cmd))
    print("=" * 72)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def eval_ret(py, results, output):
    run([
        py, "-m", "src.evaluation.evaluate_retrieval",
        "--results-path", results,
        "--output-path", output,
    ])


def maybe_llm(py, results, output, top_k=LLM_JUDGE_TOP_K):
    run([
        py, "-m", "src.evaluation.evaluate_llm",
        "--results-path", results,
        "--output-path", output,
        "--top-k", str(top_k),
        "--max-queries", str(LLM_JUDGE_MAX_QUERIES),
    ])


def main() -> None:
    py = sys.executable

    # 1. Corpus
    run([
        py, "-m", "src.corpus.create_corpus",
        "--train-size", "5000",
        "--validation-size", "1000",
    ])
    run([py, "-m", "src.corpus.chunking"])

    # 2. Baseline indexes
    for idx, model in [
        ("indexes/minilm_base", "sentence-transformers/all-MiniLM-L6-v2"),
        ("indexes/bge_m3_base", "BAAI/bge-m3"),
    ]:
        run([
            py, "-m", "src.corpus.build_index",
            "--chunks-path", "data/processed/corpus_chunks.jsonl",
            "--index-dir", idx,
            "--model-name", model,
        ])

    # 3. Retrieval eval queries
    run([py, "-m", "src.corpus.build_retrieval_eval_queries"])

    # 4. Baseline retrieval
    for name, module, extra in [
        (
            "minilm_baseline",
            "src.retrieval.baseline_retrieval",
            [
                "--index-dir", "indexes/minilm_base",
                "--model-name", "sentence-transformers/all-MiniLM-L6-v2",
            ],
        ),
        (
            "bge_baseline",
            "src.retrieval.baseline_retrieval",
            [
                "--index-dir", "indexes/bge_m3_base",
                "--model-name", "BAAI/bge-m3",
            ],
        ),
        (
            "bm25_baseline",
            "src.retrieval.bm25_retrieval",
            ["--index-dir", "indexes/bge_m3_base"],
        ),
        (
            "hybrid_baseline",
            "src.retrieval.hybrid_retrieval",
            [
                "--index-dir", "indexes/bge_m3_base",
                "--model-name", "BAAI/bge-m3",
                "--alpha", "0.6",
                "--candidate-k", "50",
            ],
        ),
    ]:
        run([
            py, "-m", module,
            "--queries-path", "data/processed/val_queries.jsonl",
            "--top-k", str(BASELINE_TOP_K),
            "--output-path", f"results/retrieval/{name}_results.json",
            *extra,
        ])
        eval_ret(py, f"results/retrieval/{name}_results.json", f"results/retrieval/{name}_metrics.json")
        maybe_llm(py, f"results/retrieval/{name}_results.json", f"results/judge/{name}_judged.json")

    # 5. Generate + inject embedding-optimized attacks
    attack_cmd = [
        py, "-m", "src.attack.generate_attacks",
        "--queries-path", "data/processed/val_queries.jsonl",
        "--output-path", "data/processed/spoof_chunks.jsonl",
        "--candidates-per-style", str(ATTACK_CANDIDATES_PER_STYLE),
        "--keep-per-style", str(ATTACK_KEEP_PER_STYLE),
        "--embedding-model", ATTACK_EMBEDDING_MODEL,
        "--max-queries", str(ATTACK_MAX_QUERIES),
        "--attack-mode", ATTACK_MODE,
        "--use-llm",
        "--temperature", "0.9",
    ]
    run(attack_cmd)

    run([
        py, "-m", "src.attack.inject_attacks",
        "--real-chunks", "data/processed/corpus_chunks.jsonl",
        "--spoof-chunks", "data/processed/spoof_chunks.jsonl",
        "--output-path", "data/processed/augmented_chunks.jsonl",
    ])

    # 6. Attacked indexes
    for idx, model in [
        ("indexes/minilm_attack", "sentence-transformers/all-MiniLM-L6-v2"),
        ("indexes/bge_m3_attack", "BAAI/bge-m3"),
    ]:
        run([
            py, "-m", "src.corpus.build_index",
            "--chunks-path", "data/processed/augmented_chunks.jsonl",
            "--index-dir", idx,
            "--model-name", model,
        ])

    # 7. Retrieval under attack: retrieve top-20 so defense can rerank candidates.
    for name, module, extra in [
        (
            "minilm_attack",
            "src.retrieval.baseline_retrieval",
            [
                "--index-dir", "indexes/minilm_attack",
                "--model-name", "sentence-transformers/all-MiniLM-L6-v2",
            ],
        ),
        (
            "bm25_attack",
            "src.retrieval.bm25_retrieval",
            ["--index-dir", "indexes/bge_m3_attack"],
        ),
        (
            "hybrid_attack",
            "src.retrieval.hybrid_retrieval",
            [
                "--index-dir", "indexes/bge_m3_attack",
                "--model-name", "BAAI/bge-m3",
                "--alpha", "0.6",
                "--candidate-k", "50",
            ],
        ),
    ]:
        run([
            py, "-m", module,
            "--queries-path", "data/processed/val_queries.jsonl",
            "--top-k", str(ATTACK_RETRIEVAL_TOP_K),
            "--output-path", f"results/retrieval/{name}_results.json",
            *extra,
        ])
        eval_ret(py, f"results/retrieval/{name}_results.json", f"results/retrieval/{name}_metrics.json")
        maybe_llm(py, f"results/retrieval/{name}_results.json", f"results/judge/{name}_judged.json")

    # 8. Semantic defense: Cross-Encoder reranks 20 candidates and keeps final top-5.
    for prefix in ["minilm", "bm25", "hybrid"]:
        run([
            py, "-m", "src.defense.defense_filter",
            "--input-path", f"results/retrieval/{prefix}_attack_results.json",
            "--queries-path", "data/processed/val_queries.jsonl",
            "--output-path", f"results/retrieval/{prefix}_defense_results.json",
            "--keep-top-k", str(FINAL_KEEP_TOP_K),
            "--suspicion-threshold", str(DEFENSE_THRESHOLD),
            "--defense-mode", DEFENSE_MODE,
            "--cross-encoder-model", DEFENSE_CROSS_ENCODER_MODEL,
            "--semantic-weight", str(DEFENSE_SEMANTIC_WEIGHT),
            "--retrieval-weight", str(DEFENSE_RETRIEVAL_WEIGHT),
            "--lexical-penalty-weight", str(DEFENSE_LEXICAL_PENALTY_WEIGHT),
            "--batch-size", str(DEFENSE_BATCH_SIZE),
        ])
        eval_ret(py, f"results/retrieval/{prefix}_defense_results.json", f"results/retrieval/{prefix}_defense_metrics.json")
        maybe_llm(py, f"results/retrieval/{prefix}_defense_results.json", f"results/judge/{prefix}_defense_judged.json")

    # 9. Attack-level metrics
    run([
        py, "-m", "src.attack.evaluate_attack",
        "--baseline-results", "results/retrieval/minilm_baseline_results.json",
        "--attack-results", "results/retrieval/minilm_attack_results.json",
        "--queries", "data/processed/retrieval_eval_queries.json",
        "--spoof-chunks", "data/processed/spoof_chunks.jsonl",
        "--top-k", str(FINAL_KEEP_TOP_K),
        "--output", "results/retrieval/attack_metrics.json",
    ])

    # 10. Plot
    run([
        py, "-m", "src.evaluation.plot_results",
        "--output-dir", "results/figures",
    ])

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE — results in results/ and results/figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
