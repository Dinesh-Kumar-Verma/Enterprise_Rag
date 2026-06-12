"""
RAGAS-based evaluation: faithfulness, answer relevancy, context recall.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_recall,
        context_precision,
        faithfulness,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available. Install with: pip install ragas datasets")


class RAGASEvaluator:
    """
    Evaluates RAG outputs using RAGAS metrics.
    Requires ground-truth answers for full evaluation.
    """

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not installed"}

        data: dict[str, list[Any]] = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            metrics.append(context_recall)

        try:
            result = evaluate(dataset, metrics=metrics)
            scores = {
                "faithfulness": round(float(result["faithfulness"]), 4),
                "answer_relevancy": round(float(result["answer_relevancy"]), 4),
                "context_precision": round(float(result["context_precision"]), 4),
            }
            if ground_truth:
                scores["context_recall"] = round(float(result["context_recall"]), 4)
            return scores
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_batch(
        self, test_cases: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Evaluate a batch of test cases.
        Each test_case: {query, answer, contexts, ground_truth (optional)}
        """
        if not RAGAS_AVAILABLE:
            return {"error": "RAGAS not installed"}

        data: dict[str, list] = {
            "question": [],
            "answer": [],
            "contexts": [],
        }
        has_ground_truth = all("ground_truth" in tc for tc in test_cases)
        if has_ground_truth:
            data["ground_truth"] = []

        for tc in test_cases:
            data["question"].append(tc["query"])
            data["answer"].append(tc["answer"])
            data["contexts"].append(tc["contexts"])
            if has_ground_truth:
                data["ground_truth"].append(tc["ground_truth"])

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            metrics.append(context_recall)

        try:
            result = evaluate(dataset, metrics=metrics)
            return {k: round(float(v), 4) for k, v in result.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            return {"error": str(e)}
