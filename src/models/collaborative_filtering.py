"""
collaborative_filtering.py

Purpose:
--------
Implements User-Based Collaborative Filtering (CF).

Core Idea:
----------
Users with similar preferences will rate items similarly.

Pipeline:
---------
1. Normalize ratings (mean-centering per user)
2. Compute similarity between users (cosine similarity)
3. Use top-K similar users to:
   - Generate ranking scores (for recommendations)
   - Predict ratings (for RMSE)

Includes:
---------
- Fast vectorized ranking (used in evaluation)
- Rating prediction (top-K)
- Single rating prediction (optimized for RMSE)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Step 1: Normalize User Ratings
# --------------------------------------------------

def normalize_user_item_matrix(
    user_item_matrix: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Normalize ratings by subtracting each user's mean.

    Why:
    ----
    Removes user bias (e.g., some users rate higher than others).
    This makes similarity more meaningful.

    Returns:
    --------
    user_means : average rating per user
    normalized_matrix : mean-centered ratings
    """
    user_means = user_item_matrix.mean(axis=1)
    normalized_matrix = user_item_matrix.sub(user_means, axis=0)

    return user_means, normalized_matrix


# --------------------------------------------------
# Step 2: Compute User Similarity
# --------------------------------------------------

def compute_user_similarity(normalized_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between users.

    Why:
    ----
    Measures how similar users are based on rating patterns.

    Missing values are treated as 0.
    """
    filled_matrix = normalized_matrix.fillna(0)

    similarity = cosine_similarity(filled_matrix)

    return pd.DataFrame(
        similarity,
        index=normalized_matrix.index,
        columns=normalized_matrix.index,
    )


# --------------------------------------------------
# Step 3: Fast Ranking (Main Recommendation Function)
# --------------------------------------------------

def predict_scores_ranking_fast(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    k: int = 20,
) -> dict[int, float]:
    """
    Compute recommendation scores for all unseen items.

    This is the main CF function used for:
    - Precision@K evaluation
    - Hybrid model

    Key Idea:
    ---------
    Score(item) = sum(similarity * neighbor_rating)

    Optimization:
    -------------
    Fully vectorized → much faster than nested loops.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found")

    user_ratings = user_item_matrix.loc[user_id]

    similarities = user_similarity_df.loc[user_id].drop(user_id)

    # Use only positively similar users
    similarities = similarities[similarities > 0]

    if similarities.empty:
        return {}

    # Select top-K neighbors
    top_k_users = similarities.sort_values(ascending=False).head(k)

    neighbor_ratings = user_item_matrix.loc[top_k_users.index]

    sims = top_k_users.to_numpy().reshape(-1, 1)

    # Weighted sum of neighbor ratings
    scores_array = (neighbor_ratings.fillna(0).to_numpy() * sims).sum(axis=0)

    unseen_mask = user_ratings.isna().to_numpy()
    movie_ids = user_item_matrix.columns.to_numpy()

    scores = {
        int(movie_id): float(score)
        for movie_id, score, unseen in zip(movie_ids, scores_array, unseen_mask)
        if unseen
    }

    return scores


# --------------------------------------------------
# Step 4: Predict Ratings (Top-K)
# --------------------------------------------------

def predict_ratings_top_k(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 20,
    min_neighbors: int = 3,
) -> dict[int, float]:
    """
    Predict actual ratings for unseen items.

    Why:
    ----
    Used for displaying predicted ratings (not for evaluation).

    Formula:
    --------
    r_hat = user_mean + weighted deviation from neighbors
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found")

    user_ratings = user_item_matrix.loc[user_id]

    similarities = user_similarity_df.loc[user_id].drop(user_id)
    top_k_users = similarities.sort_values(ascending=False).head(k)

    predictions = {}

    for movie_id in user_item_matrix.columns:
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

            if denominator == 0 or neighbor_count < min_neighbors:
                continue

            predicted_rating = user_means[user_id] + numerator / denominator

            # Clip to valid range
            predictions[movie_id] = min(5.0, max(1.0, predicted_rating))

    return predictions


# --------------------------------------------------
# Step 5: Fast Single Prediction (RMSE)
# --------------------------------------------------

def predict_single_rating(
    user_id: int,
    movie_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 20,
) -> float:
    """
    Predict rating for ONE (user, movie) pair.

    Why:
    ----
    Used for RMSE evaluation (fast and efficient).

    Avoids computing full recommendation lists.
    """
    if user_id not in user_item_matrix.index:
        return np.nan

    if movie_id not in user_item_matrix.columns:
        return user_means.get(user_id, np.nan)

    similarities = user_similarity_df.loc[user_id].drop(user_id)
    top_k_users = similarities.sort_values(ascending=False).head(k)

    numerator = 0.0
    denominator = 0.0

    for other_user, sim in top_k_users.items():
        other_rating = user_item_matrix.loc[other_user, movie_id]

        if not pd.isna(other_rating):
            numerator += sim * (other_rating - user_means[other_user])
            denominator += abs(sim)

    if denominator == 0:
        return user_means[user_id]

    predicted_rating = user_means[user_id] + numerator / denominator

    return min(5.0, max(1.0, predicted_rating))


# --------------------------------------------------
# Utility: Top-N Recommendation Extraction
# --------------------------------------------------

def recommend_top_n(
    scores: dict[int, float],
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """
    Return top-N items sorted by score.
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]