"""
Module: compare_models.py

Description:
This module compares the performance of multiple recommendation approaches
on the same train/test split.

Compared models:
- Collaborative Filtering (ranking + rating prediction)
- Hybrid Recommender with multiple alpha values

Evaluation Metrics:
- Precision@5
- Precision@10
- RMSE (for rating prediction models only)

Purpose:
- Provide a side-by-side summary of model quality
- Help analyze the effect of hybrid weighting
"""

import pandas as pd

from src.data_processing.data_loader import build_user_item_matrix
from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking,
    predict_ratings_top_k,
)
from src.models.content_based_filtering import recommend_content_based
from src.models.hybrid_recommender import combine_scores
from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
    evaluate_rmse,
)


def hybrid_model_func(
    user_id: int,
    user_item_matrix,
    user_similarity_df,
    ratings_df,
    movies_df,
    alpha: float,
    k: int = 20,
) -> dict[int, float]:
    """
    Generate hybrid recommendation scores for a single user.

    Args:
        user_id (int): Target user ID
        user_item_matrix (DataFrame): User-item training matrix
        user_similarity_df (DataFrame): User-user similarity matrix
        ratings_df (DataFrame): Training ratings data
        movies_df (DataFrame): Movie metadata
        alpha (float): Hybrid weight for collaborative filtering
        k (int): Number of nearest neighbors for collaborative filtering

    Returns:
        dict: {movie_id: hybrid_score}
    """
    collaborative_scores = predict_scores_ranking(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
        k=k,
    )

    content_scores = recommend_content_based(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        top_n=1000,      # large candidate pool for evaluation
        min_rating=4.0,  # treat ratings >= 4 as liked items
    )

    return combine_scores(
        collaborative_scores=collaborative_scores,
        content_scores=content_scores,
        alpha=alpha,
    )


def content_model_func(
    user_id: int,
    ratings_df,
    movies_df,
) -> dict[int, float]:
    """
    Generate content-based recommendation scores for a single user.

    Args:
        user_id (int): Target user ID
        ratings_df (DataFrame): Ratings data
        movies_df (DataFrame): Movie metadata

    Returns:
        dict: {movie_id: content_score}
    """
    return recommend_content_based(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        top_n=1000,
        min_rating=4.0,
    )


def compare_models(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Compare recommendation models on a shared train/test split.

    Workflow:
    1. Split ratings into train/test per user
    2. Build collaborative filtering artifacts
    3. Evaluate collaborative filtering performance
    4. Evaluate hybrid models with different alpha values
    5. Return a summary table

    Args:
        ratings (DataFrame): Full ratings dataset
        movies (DataFrame): Movies metadata

    Returns:
        DataFrame: Comparison table with model names and evaluation metrics
    """
    # Create train/test split
    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    # Build collaborative filtering inputs
    train_matrix = build_user_item_matrix(train_df)
    user_means, normalized_matrix = normalize_user_item_matrix(train_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    results = []

    # -----------------------------------
    # Collaborative Filtering Evaluation
    # -----------------------------------
    cf_precision_5 = evaluate_precision_at_k(
        model_func=predict_scores_ranking,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        k=5,
        model_kwargs={"k": 20},
    )

    cf_precision_10 = evaluate_precision_at_k(
        model_func=predict_scores_ranking,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        k=10,
        model_kwargs={"k": 20},
    )

    cf_rmse = evaluate_rmse(
        predict_func=predict_ratings_top_k,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        user_means=user_means,
        model_kwargs={"k": 20, "min_neighbors": 3},
    )

    results.append({
        "model": "CF Ranking / Predicted",
        "precision@5": cf_precision_5,
        "precision@10": cf_precision_10,
        "rmse": cf_rmse,
    })

    # -----------------------------------
    # Hybrid Model Evaluation
    # -----------------------------------
    for alpha in [0.3, 0.5, 0.7]:
        hybrid_precision_5_scores = []
        hybrid_precision_10_scores = []

        for user_id in test_df["user_id"].unique():
            if user_id not in train_matrix.index:
                continue

            # Relevant items = test movies rated 4 or higher
            test_user = test_df[test_df["user_id"] == user_id]
            relevant_items = set(test_user[test_user["rating"] >= 4]["movie_id"])

            if not relevant_items:
                continue

            scores = hybrid_model_func(
                user_id=user_id,
                user_item_matrix=train_matrix,
                user_similarity_df=user_similarity_df,
                ratings_df=train_df,
                movies_df=movies,
                alpha=alpha,
                k=20,
            )

            ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            recommended_5 = [movie_id for movie_id, _ in ranked_items[:5]]
            recommended_10 = [movie_id for movie_id, _ in ranked_items[:10]]

            hits_5 = sum(1 for item in recommended_5 if item in relevant_items) / 5
            hits_10 = sum(1 for item in recommended_10 if item in relevant_items) / 10

            hybrid_precision_5_scores.append(hits_5)
            hybrid_precision_10_scores.append(hits_10)

        results.append({
            "model": f"Hybrid alpha={alpha}",
            "precision@5": (
                sum(hybrid_precision_5_scores) / len(hybrid_precision_5_scores)
                if hybrid_precision_5_scores else 0.0
            ),
            "precision@10": (
                sum(hybrid_precision_10_scores) / len(hybrid_precision_10_scores)
                if hybrid_precision_10_scores else 0.0
            ),
            "rmse": None,  # hybrid model is currently evaluated only as a ranker
        })

    return pd.DataFrame(results)