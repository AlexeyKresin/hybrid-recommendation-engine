"""
hybrid_recommender.py

Purpose:
--------
Combines Collaborative Filtering and Content-Based Filtering into one
hybrid recommendation score.

Core Idea:
----------
CF and content-based models produce scores on different scales.
Before combining them, both score sets are normalized to [0, 1].

Formula:
--------
hybrid_score = alpha * CF_score + (1 - alpha) * Content_score

Alpha Meaning:
--------------
alpha = 1.0 -> fully collaborative filtering
alpha = 0.0 -> fully content-based filtering
alpha = 0.8 -> 80% CF, 20% content
"""


def min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    """
    Normalize scores to the range [0, 1].

    This makes CF and content scores comparable before combining.
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    # If all scores are equal, assign every item the same normalized value.
    if max_score == min_score:
        return {movie_id: 1.0 for movie_id in scores}

    return {
        movie_id: (score - min_score) / (max_score - min_score)
        for movie_id, score in scores.items()
    }


def combine_scores(
    collaborative_scores: dict[int, float],
    content_scores: dict[int, float],
    alpha: float = 0.8,
) -> dict[int, float]:
    """
    Combine CF and content-based scores.

    Steps:
    ------
    1. Normalize CF scores to [0, 1]
    2. Normalize content scores to [0, 1]
    3. Take the union of candidate movies
    4. Compute weighted hybrid score

    Returns:
    --------
    dict:
        {movie_id: hybrid_score}
    """
    collaborative_scores = min_max_normalize(collaborative_scores)
    content_scores = min_max_normalize(content_scores)

    all_movie_ids = set(collaborative_scores.keys()) | set(content_scores.keys())

    hybrid_scores = {}

    for movie_id in all_movie_ids:
        cf_score = collaborative_scores.get(movie_id, 0.0)
        content_score = content_scores.get(movie_id, 0.0)

        hybrid_scores[movie_id] = (
            alpha * cf_score
            + (1 - alpha) * content_score
        )

    return hybrid_scores