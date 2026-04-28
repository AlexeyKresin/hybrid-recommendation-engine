"""
helpers.py

Utility functions for formatting and displaying recommendations.
"""

import pandas as pd


def format_recommendations(
    recommendations: list[tuple[int, float]],
    movies_df: pd.DataFrame,
    score_column: str = "score",
    clip_ratings: bool = False,
) -> pd.DataFrame:
    """
    Convert recommendation output into a readable DataFrame.

    Input:
    ------
    recommendations:
        List of (movie_id, score) tuples

    movies_df:
        DataFrame containing movie metadata (must include title)

    score_column:
        Name of the score column (e.g., 'score', 'predicted_score', etc.)

    clip_ratings:
        If True, clip scores to valid rating range [1, 5]

    Output:
    -------
    DataFrame with columns:
        - movie_id
        - title
        - score_column
    """

    # Create lookup: movie_id -> title
    movie_lookup = movies_df.set_index("movie_id")["title"].to_dict()

    rows = []

    for movie_id, score in recommendations:

        # Optional: clip predicted ratings to valid range
        if clip_ratings:
            score = min(max(float(score), 1.0), 5.0)

        rows.append({
            "movie_id": movie_id,
            "title": movie_lookup.get(movie_id, f"Movie {movie_id}"),
            score_column: round(float(score), 3),
        })

    return pd.DataFrame(rows)