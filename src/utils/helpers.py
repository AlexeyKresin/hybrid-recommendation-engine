import pandas as pd


def format_recommendations(
    recommendations: list[tuple[int, float]],
    movies_df: pd.DataFrame,
    score_column: str = "score",
    clip_ratings: bool = False
) -> pd.DataFrame:
    """
    Convert [(movie_id, score), ...] into a readable dataframe with titles.
    """
    movie_lookup = movies_df.set_index("movie_id")["title"].to_dict()
    rows = []

    for movie_id, score in recommendations:
        if clip_ratings:
            score = min(max(float(score), 1.0), 5.0)

        rows.append({
            "movie_id": movie_id,
            "title": movie_lookup.get(movie_id, f"Movie {movie_id}"),
            score_column: round(float(score), 3)
        })

    return pd.DataFrame(rows)