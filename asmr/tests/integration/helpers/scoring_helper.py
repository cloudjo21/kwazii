"""
Scoring helpers for integration tests
"""

import logging
from typing import Any, Union

from asmr.index.models import FieldBasedRanking

logger = logging.getLogger(__name__)


def collect_field_scores(
    field_results: Union[list[tuple[int, float]], FieldBasedRanking]
) -> dict[int, float]:
    """
    Extract document scores from field retrieval results.
    
    Args:
        field_results: List of (doc_id, score) tuples from retriever
    
    Returns:
        Dictionary mapping document IDs to scores
    """
    doc_scores = {}

    for doc_id, score in field_results:
        if doc_id not in doc_scores:
            doc_scores[doc_id] = 0.0
        doc_scores[doc_id] += score

    return doc_scores


def aggregate_and_report_top_docs(
    field_document_scores: dict[str, dict[str, float]],
    sample_documents: list[dict[str, Any]],
    field_names: list[str],
    top_k: int = 4,
    title: str = "Top Documents by Aggregated Score"
) -> list[tuple[str, float]]:
    """
    Aggregate scores across fields and report top documents with breakdown.
    
    Args:
        field_document_scores: Dictionary mapping field names to document scores
        sample_documents: List of document dictionaries
        field_names: List of field names for score breakdown
        top_k: Number of top documents to report
        title: Title for the report
    
    Returns:
        List of (doc_id, total_score) tuples for top documents
    """
    # Aggregate scores across all fields
    aggregated_scores = {}
    for field_name, doc_scores in field_document_scores.items():
        for doc_id, score in doc_scores.items():
            if doc_id not in aggregated_scores:
                aggregated_scores[doc_id] = 0.0
            aggregated_scores[doc_id] += score

    if not aggregated_scores:
        logger.warning("No aggregated scores to report")
        return []

    # Get top documents by aggregated score
    top_docs = sorted(aggregated_scores.items(),
                      key=lambda x: x[1],
                      reverse=True)[:top_k]

    print(f"\n📊 {title}:")
    print(f"{'Rank':<4} {'Doc ID':<6} {'Title':<40} {'Total Score':<12}")
    print("-" * 70)

    for rank, (doc_id, total_score) in enumerate(top_docs, 1):
        # Get document title
        doc_title = sample_documents[rank].get('title', f'Document {doc_id}')
        title_truncated = doc_title[:37] + "..." if len(
            doc_title) > 40 else doc_title

        print(
            f"{rank:<4} {doc_id:<6} {title_truncated:<40} {total_score:<12.4f}"
        )

        # Show field-wise score breakdown
        field_breakdown = []
        for field_name in field_names:
            if field_name in field_document_scores and doc_id in field_document_scores[
                    field_name]:
                score = field_document_scores[field_name][doc_id]
                field_breakdown.append(f"{field_name}: {score:.4f}")

        if field_breakdown:
            logger.info(
                f"  Doc {doc_id} breakdown: {', '.join(field_breakdown)}")

    return top_docs


def log_field_results(field_name: str,
                      field_results: FieldBasedRanking,
                      sample_documents: list[dict[str, Any]],
                      top_k: int = 5) -> dict[str, float]:
    """
    Log field retrieval results with document titles and collect scores.
    
    Args:
        field_name: Name of the field being processed
        field_results: List of (doc_id, score) tuples from retriever
        sample_documents: List of document dictionaries
        top_k: Number of top results to display
    
    Returns:
        Dictionary mapping document IDs to scores
    """
    print(f"{field_name}: {len(field_results)} results")

    doc_scores = {}

    if field_results:
        print(f"  Top results [{field_name}]:")
        for i, (doc_id, score) in enumerate(field_results[:top_k]):
            doc_title = sample_documents[i].get('title', f'Document {doc_id}')
            print(f"    {i+1}. Doc {doc_id}: {doc_title} (score: {score:.4f})")

            # Collect scores for aggregation
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
            doc_scores[doc_id] += score

    return doc_scores
