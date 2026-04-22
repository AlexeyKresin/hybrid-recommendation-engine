"""
Module: collaborative_filtering.py

Description:
This module implements user-based collaborative filtering.

Main Components:
- Normalization (mean-centering per user)
- User similarity computation (cosine similarity)
- Ranking-based recommendation (implicit scoring)
- Rating prediction (explicit prediction)
- Top-N recommendation extraction

Key Concepts:
- Similar users influence recommendations
- Ratings are normalized to remove user bias
- Predictions use weighted averages of neighbors
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def normalize_user_item_matrix(user_item_matrix: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Mean-center each user's ratings.

    This removes user bias (e.g., users who rate consistently high or low).

    Args:
        user_item_matrix (DataFrame): User-item ratings matrix

    Returns:
        tuple:
            - user_means (Series): Mean rating per user
            - normalized_matrix (DataFrame): Mean-centered ratings
    """
    user_means = user_item_matrix.mean(axis=1)

    # Subtract each user's mean from their ratings
    normalized_matrix = user_item_matrix.sub(user_means, axis=0)

    return user_means, normalized_matrix


def compute_user_similarity(normalized_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between users.

    Missing values are replaced with 0 after normalization.

    Args:
        normalized_matrix (DataFrame): Mean-centered user-item matrix

    Returns:
        DataFrame: User-user similarity matrix
    """
    # Fill missing values with 0 (neutral after normalization)
    filled_matrix = normalized_matrix.fillna(0)

    similarity = cosine_similarity(filled_matrix)

    return pd.DataFrame(
        similarity,
        index=normalized_matrix.index,
        columns=normalized_matrix.index
    )


def predict_scores_ranking(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    k: int = 20
) -> dict[int, float]:
    """
    Compute ranking-based recommendation scores.

    Higher score = more relevant item.
    NOTE: These are NOT actual predicted ratings.

    Args:
        user_id (int): Target user
        user_item_matrix (DataFrame): User-item matrix
        user_similarity_df (DataFrame): User similarity matrix
        k (int): Number of nearest neighbors

    Returns:
        dict: {item_id: score}
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix")

    user_ratings = user_item_matrix.loc[user_id]

    # Get similarity scores for the user (exclude self)
    similarities = user_similarity_df.loc[user_id].drop(user_id)

    # Keep only positively correlated users
    similarities = similarities[similarities > 0]

    # Select top-k most similar users
    top_k_users = similarities.sort_values(ascending=False).head(k)

    scores: dict[int, float] = {}

    for movie_id in user_item_matrix.columns:

        # Only score items the user has not rated
        if pd.isna(user_ratings[movie_id]):
            score = 0.0

            # Weighted sum of neighbors' ratings
            for other_user, sim in top_k_users.items():
                other_rating = user_item_matrix.loc[other_user, movie_id]

                if not pd.isna(other_rating):
                    score += sim * other_rating

            scores[movie_id] = score

    return scores


def predict_ratings_top_k(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 20,
    min_neighbors: int = 3
) -> dict[int, float]:
    """
    Predict actual ratings using collaborative filtering.

    Formula:
        r̂_ui = μ_u + (Σ(sim(u,v) * (r_vi - μ_v))) / Σ|sim(u,v)|

    Args:
        user_id (int): Target user
        user_item_matrix (DataFrame): User-item matrix
        user_similarity_df (DataFrame): User similarity matrix
        user_means (Series): Mean ratings per user
        k (int): Number of neighbors
        min_neighbors (int): Minimum required neighbors for prediction

    Returns:
        dict: {item_id: predicted_rating}
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix")

    user_ratings = user_item_matrix.loc[user_id]

    similarities = user_similarity_df.loc[user_id].drop(user_id)
    top_k_users = similarities.sort_values(ascending=False).head(k)

    predictions: dict[int, float] = {}

    for movie_id in user_item_matrix.columns:

        # Predict only for unseen items
        if pd.isna(user_ratings[movie_id]):

            numerator = 0.0
            denominator = 0.0
            neighbor_count = 0

            for other_user, sim in top_k_users.items():
                other_rating = user_item_matrix.loc[other_user, movie_id]

                if not pd.isna(other_rating):
                    numerator += sim * (other_rating - user_means[other_user])
                    denominator += abs(sim)
                    neighbor_count += 1

            # Skip if not enough neighbors or invalid denominator
            if denominator == 0 or neighbor_count < min_neighbors:
                continue

            predicted_rating = user_means[user_id] + (numerator / denominator)

            # Clip predictions to MovieLens rating range
            predicted_rating = min(5.0, max(1.0, predicted_rating))

            predictions[movie_id] = predicted_rating

    return predictions


def recommend_top_n(
    scores: dict[int, float],
    top_n: int = 10
) -> list[tuple[int, float]]:
    """
    Return top-N items based on scores.

    Args:
        scores (dict): {item_id: score}
        top_n (int): Number of items to return

    Returns:
        list of tuples: [(item_id, score), ...]
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]