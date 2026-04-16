from __future__ import annotations
from typing import Dict

def _norm(text: str, allowed: set, default: str) -> str:
    cleaned = text.strip().upper()
    for label in allowed:
        if label in cleaned: return label
    return default

class LLMJudge:
    def __init__(self, llm_client):
        self.client = llm_client

    def judge_chunk(self, question: str, chunk_text: str, gold_answer: str) -> Dict:
        support = _norm(self.client.generate(
            f"Question: {question}\nContext: {chunk_text}\nGold answer: {gold_answer}\n"
            f"Does the context SUPPORT the gold answer?\nAnswer: SUPPORTS / PARTIAL / UNSUPPORTED"),
            {"SUPPORTS","PARTIAL","UNSUPPORTED"}, "UNSUPPORTED")

        misleading = _norm(self.client.generate(
            f"Question: {question}\nContext: {chunk_text}\nGold answer: {gold_answer}\n"
            f"Is the context MISLEADING?\nAnswer: MISLEADING / NOT_MISLEADING"),
            {"MISLEADING","NOT_MISLEADING"}, "NOT_MISLEADING")

        predicted = self.client.generate(
            f"Answer briefly using only the context.\nQuestion: {question}\nContext: {chunk_text}")

        correctness = _norm(self.client.generate(
            f"Question: {question}\nGold: {gold_answer}\nPredicted: {predicted}\n"
            f"Is the predicted answer correct?\nAnswer: CORRECT / PARTIAL / INCORRECT"),
            {"CORRECT","PARTIAL","INCORRECT"}, "INCORRECT")

        return {"support": support, "misleading": misleading,
                "predicted_answer": predicted, "answer_correctness": correctness}
