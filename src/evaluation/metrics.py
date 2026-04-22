"""
Module: metrics.py

Description:
This module provides evaluation utilities for recommendation systems.

Metrics Implemented:
- Precision@K (ranking quality)
- RMSE (rating prediction accuracy)

Additional Utilities:
- Train/test split per user (time-aware split)

Key Idea:
- Precision@K evaluates ranking relevance
- RMSE evaluates numerical prediction accuracy
"""

import numpy as np
import pandas as pd


def train_test_split_per_user(ratings: pd.DataFrame, test_size: int = 5):
    """
    Split dataset into train and test sets per user.

    Strategy:
    - Sort interactions by timestamp
    - Use last `test_size` items as test set
    - Remaining items go to training set

    Args:
        ratings (DataFrame): Full ratings dataset
        test_size (int): Number of test interactions per user

    Returns:
        tuple:
            - train_df (DataFrame)
            - test_df (DataFrame)
    """
    train_list = []
    test_list = []

    for _, group in ratings.groupby("user_id"):

        # Sort chronologically (important for realistic evaluation)
        group = group.sort_values("timestamp")

        # If user has too few interactions → keep all in train
        if len(group) <= test_size:
            train_list.append(group)
            continue

        # Split: last interactions = test
        test = group.tail(test_size)
        train = group.iloc[:-test_size]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame()

    return train_df, test_df


def precision_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int
) -> float:
    """
    Compute Precision@K.

    Formula:
        Precision@K = (# of relevant items in top-K) / K

    Args:
        recommended_items (list): Ranked list of recommended items
        relevant_items (set): Ground truth relevant items
        k (int): Cutoff rank

    Returns:
        float: Precision@K score
    """
    if not relevant_items:
        return 0.0

    recommended_items = recommended_items[:k]

    hits = sum(1 for item in recommended_items if item in relevant_items)

    return hits / k


def evaluate_precision_at_k(
    model_func,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item_matrix,
    user_similarity_df,
    k: int = 10,
    model_kwargs: dict | None = None,
) -> float:
    """
    Evaluate average Precision@K across all users.

    Workflow:
    - For each user:
        1. Get relevant items (rating >= 4)
        2. Generate recommendations
        3. Compute Precision@K
    - Return mean over users

    Args:
        model_func (callable): Function that generates scores
        train_df (DataFrame): Training data
        test_df (DataFrame): Test data
        user_item_matrix (DataFrame): User-item matrix
        user_similarity_df (DataFrame): User similarity matrix
        k (int): Top-K cutoff
        model_kwargs (dict): Additional model parameters

    Returns:
        float: Mean Precision@K
    """
    precisions = []
    model_kwargs = model_kwargs or {}

    for user_id in test_df["user_id"].unique():

        # Skip users not present in training matrix
        if user_id not in user_item_matrix.index:
            continue

        test_user = test_df[test_df["user_id"] == user_id]

        # Relevant items = high ratings (>= 4)
        relevant_items = set(
            test_user[test_user["rating"] >= 4]["movie_id"]
        )

        if not relevant_items:
            continue

        # Generate recommendation scores
        scores = model_func(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
            **model_kwargs,
        )

        # Rank items by score
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended = [movie_id for movie_id, _ in ranked_items[:k]]

        # Compute Precision@K
        p_at_k = precision_at_k(recommended, relevant_items, k)
        precisions.append(p_at_k)

    return float(np.mean(precisions)) if precisions else 0.0


def rmse(predictions: list[float], actuals: list[float]) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Formula:
        RMSE = sqrt(mean((prediction - actual)^2))

    Args:
        predictions (list): Predicted ratings
        actuals (list): Ground truth ratings

    Returns:
        float: RMSE value
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def evaluate_rmse(
    predict_func,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item_matrix,
    user_similarity_df,
    user_means,
    model_kwargs: dict | None = None,
) -> float:
    """
    Evaluate RMSE across all users.

    Workflow:
    - For each user:
        1. Predict ratings for unseen items
        2. Compare with actual ratings in test set
    - Aggregate predictions across all users
    - Compute RMSE

    Args:
        predict_func (callable): Function that predicts ratings
        train_df (DataFrame): Training data
        test_df (DataFrame): Test data
        user_item_matrix (DataFrame): User-item matrix
        user_similarity_df (DataFrame): User similarity matrix
        user_means (Series): Mean rating per user
        model_kwargs (dict): Additional parameters

    Returns:
        float: RMSE score
    """
    preds = []
    actuals = []
    model_kwargs = model_kwargs or {}

    for user_id in test_df["user_id"].unique():

        if user_id not in user_item_matrix.index:
            continue

        user_test = test_df[test_df["user_id"] == user_id]

        # Generate predictions for this user
        scores = predict_func(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
            user_means=user_means,
            **model_kwargs,
        )

        for _, row in user_test.iterrows():
            movie_id = row["movie_id"]
            actual_rating = row["rating"]

            # Only evaluate items we predicted
            if movie_id in scores:
                preds.append(scores[movie_id])
                actuals.append(actual_rating)

    if not preds:
        return 0.0

    return rmse(preds, actuals)