"""
metrics.py

Purpose:
--------
Provides evaluation utilities for recommendation systems.

Metrics:
--------
- Precision@K: evaluates ranking quality
- RMSE: evaluates rating prediction accuracy

Key Idea:
---------
Precision@K answers: "Did we recommend relevant items?"
RMSE answers: "How accurate are predicted ratings?"
"""

import numpy as np
import pandas as pd


# --------------------------------------------------
# Train / Test Split
# --------------------------------------------------

def train_test_split_per_user(
    ratings: pd.DataFrame,
    test_size: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets per user.

    Strategy:
    ---------
    - Sort each user's interactions by timestamp.
    - Last `test_size` interactions go to the test set.
    - Remaining interactions go to the training set.

    This simulates predicting future user behavior.
    """
    train_list = []
    test_list = []

    for _, user_ratings in ratings.groupby("user_id"):
        user_ratings = user_ratings.sort_values("timestamp")

        if len(user_ratings) <= test_size:
            train_list.append(user_ratings)
            continue

        test = user_ratings.tail(test_size)
        train = user_ratings.iloc[:-test_size]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = (
        pd.concat(test_list, ignore_index=True)
        if test_list
        else pd.DataFrame(columns=ratings.columns)
    )

    return train_df, test_df


# --------------------------------------------------
# Precision@K
# --------------------------------------------------

def precision_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Formula:
    --------
    Precision@K = (# relevant items in top-K) / K
    """
    if not relevant_items:
        return 0.0

    recommended_items = recommended_items[:k]

    hits = sum(
        1 for item in recommended_items
        if item in relevant_items
    )

    return hits / k


def evaluate_precision_at_k(
    model_func,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    k: int = 10,
    model_kwargs: dict | None = None,
) -> float:
    """
    Evaluate mean Precision@K across all users.

    Workflow:
    ---------
    For each test user:
    1. Treat test movies with rating >= 4 as relevant.
    2. Generate recommendation scores.
    3. Rank movies by score.
    4. Compute Precision@K.
    """
    precisions = []
    model_kwargs = model_kwargs or {}

    for user_id in test_df["user_id"].unique():
        if user_id not in user_item_matrix.index:
            continue

        user_test = test_df[test_df["user_id"] == user_id]

        relevant_items = set(
            user_test[user_test["rating"] >= 4]["movie_id"]
        )

        if not relevant_items:
            continue

        scores = model_func(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
            **model_kwargs,
        )

        ranked_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        recommended_items = [
            movie_id for movie_id, _ in ranked_items[:k]
        ]

        precisions.append(
            precision_at_k(
                recommended_items=recommended_items,
                relevant_items=relevant_items,
                k=k,
            )
        )

    return float(np.mean(precisions)) if precisions else 0.0


# --------------------------------------------------
# RMSE
# --------------------------------------------------

def rmse(
    predictions: list[float],
    actuals: list[float],
) -> float:
    """
    Compute Root Mean Squared Error.

    Formula:
    --------
    RMSE = sqrt(mean((prediction - actual)^2))
    """
    if not predictions:
        return 0.0

    predictions_array = np.array(predictions)
    actuals_array = np.array(actuals)

    return float(
        np.sqrt(
            np.mean((predictions_array - actuals_array) ** 2)
        )
    )