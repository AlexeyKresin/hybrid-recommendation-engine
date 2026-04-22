"""
Module: hybrid_recommender.py

Description:
This module combines collaborative filtering and content-based scores
into a hybrid recommendation model.

Core Idea:
- Normalize scores from different models to a common scale
- Combine them using a weighted sum

Formula:
    hybrid_score = alpha * CF_score + (1 - alpha) * CB_score

Where:
    alpha = 1.0 → fully collaborative
    alpha = 0.0 → fully content-based
"""

def min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    """
    Normalize scores to the range [0, 1] using min-max scaling.

    Args:
        scores (dict): {item_id: score}

    Returns:
        dict: {item_id: normalized_score}
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    # Handle edge case: all scores are identical
    if max_score == min_score:
        return {movie_id: 1.0 for movie_id in scores}

    return {
        movie_id: (score - min_score) / (max_score - min_score)
        for movie_id, score in scores.items()
    }


def combine_scores(
    collaborative_scores: dict[int, float],
    content_scores: dict[int, float],
    alpha: float = 0.7
) -> dict[int, float]:
    """
    Combine collaborative and content-based scores into hybrid scores.

    Steps:
    1. Normalize both score sets to [0, 1]
    2. Take union of all items
    3. Compute weighted sum

    Args:
        collaborative_scores (dict): CF scores {item_id: score}
        content_scores (dict): CB scores {item_id: score}
        alpha (float): Weight for collaborative component (0–1)

    Returns:
        dict: {item_id: hybrid_score}
    """

    # Normalize scores to make them comparable
    collaborative_scores = min_max_normalize(collaborative_scores)
    content_scores = min_max_normalize(content_scores)

    # Union of all candidate items
    all_movie_ids = set(collaborative_scores.keys()) | set(content_scores.keys())

    hybrid_scores = {}

    for movie_id in all_movie_ids:
        cf_score = collaborative_scores.get(movie_id, 0.0)
        cb_score = content_scores.get(movie_id, 0.0)

        # Weighted combination
        hybrid_scores[movie_id] = alpha * cf_score + (1 - alpha) * cb_score

    return hybrid_scores