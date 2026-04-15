def min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

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
    collaborative_scores = min_max_normalize(collaborative_scores)
    content_scores = min_max_normalize(content_scores)

    all_movie_ids = set(collaborative_scores.keys()) | set(content_scores.keys())

    hybrid_scores = {}
    for movie_id in all_movie_ids:
        cf_score = collaborative_scores.get(movie_id, 0.0)
        cb_score = content_scores.get(movie_id, 0.0)

        hybrid_scores[movie_id] = alpha * cf_score + (1 - alpha) * cb_score

    return hybrid_scores