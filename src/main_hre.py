"""
Main entry point for Hybrid Recommendation System.

This script performs three main tasks:
1. Demo recommendations for selected users
2. Model evaluation (Precision@K and RMSE)
3. Model comparison (CF vs Content vs Hybrid with alpha tuning)

The system combines:
- Collaborative Filtering (user-based)
- Content-Based Filtering (movie features)
- Hybrid model (weighted combination)

Author: Your Name
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ----------------------------
# Data Loading
# ----------------------------
from src.data_processing.data_loader import (
    load_ratings,
    load_movies,
    build_user_item_matrix,
)

# ----------------------------
# Models
# ----------------------------
from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking_fast,   # CF ranking (vectorized)
    predict_ratings_top_k,         # CF predicted ratings
    predict_single_rating,         # CF single rating (RMSE)
    recommend_top_n,
)

from src.models.content_based_filtering import recommend_content_based
from src.models.hybrid_recommender import combine_scores

# ----------------------------
# Evaluation
# ----------------------------
from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
)

from src.evaluation.compare_models import compare_models

# ----------------------------
# Utilities
# ----------------------------
from src.utils.helpers import format_recommendations


# ----------------------------
# Execution Flags
# ----------------------------
RUN_DEMO_RECOMMENDATIONS = True
RUN_FULL_EVALUATION = True
RUN_MODEL_COMPARISON = True


# ----------------------------
# Helper: Print Section Header
# ----------------------------
def print_section(title: str) -> None:
    """Prints a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------
# Demo: Recommendations per User
# ----------------------------
def run_recommender_for_user(
    user_id: int,
    ratings,
    movies,
    user_item_matrix,
    user_means,
    user_similarity_df,
    top_n: int = 5,
    k: int = 20,
    alpha: float = 0.8,
) -> None:
    """
    Generate and display recommendations for a single user.

    Shows:
    - CF ranking scores
    - CF predicted ratings
    - Content-based scores
    - Hybrid scores
    """

    # --- Collaborative Filtering (ranking-based)
    ranking_scores = predict_scores_ranking_fast(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
        k=k,
    )

    # --- Collaborative Filtering (predicted ratings)
    predicted_scores = predict_ratings_top_k(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
        user_means=user_means,
        k=k,
        min_neighbors=3,
    )

    # --- Content-Based Filtering
    content_scores = recommend_content_based(
        user_id=user_id,
        ratings_df=ratings,
        movies_df=movies,
        top_n=top_n,
        min_rating=3.0,  # lower threshold for better signal
    )

    # --- Hybrid Model (combine CF + Content)
    hybrid_scores = combine_scores(
        collaborative_scores=ranking_scores,
        content_scores=content_scores,
        alpha=alpha,
    )

    # --- Display results
    recommendation_outputs = [
        ("Top recommendations by ranking score", ranking_scores, "score"),
        ("Top recommendations by predicted score", predicted_scores, "predicted_score"),
        ("Top recommendations by content-based score", content_scores, "content_score"),
        ("Top recommendations by hybrid score", hybrid_scores, "hybrid_score"),
    ]

    print_section(f"Recommendations for user {user_id}")

    for title, scores, score_column in recommendation_outputs:
        top_items = recommend_top_n(scores, top_n=top_n)

        formatted = format_recommendations(
            top_items,
            movies,
            score_column=score_column,
            clip_ratings=False,
        )

        print(f"\n{title}:\n")
        print(formatted.to_string(index=False))


# ----------------------------
# Fast RMSE Evaluation
# ----------------------------
def evaluate_rmse_fast(
    test_df,
    train_matrix,
    user_similarity_df,
    user_means,
    k: int = 20,
) -> float:
    """
    Compute RMSE using fast single-rating prediction.

    Only predicts ratings for test pairs (efficient).
    """
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        predicted_rating = predict_single_rating(
            user_id=row["user_id"],
            movie_id=row["movie_id"],
            user_item_matrix=train_matrix,
            user_similarity_df=user_similarity_df,
            user_means=user_means,
            k=k,
        )

        if not pd.isna(predicted_rating):
            y_true.append(row["rating"])
            y_pred.append(predicted_rating)

    return np.sqrt(mean_squared_error(y_true, y_pred))


# ----------------------------
# Demo Execution
# ----------------------------
def run_demo_recommendations(
    ratings,
    movies,
    user_item_matrix,
    user_means,
    user_similarity_df,
) -> None:
    """Run demo recommendations for selected users."""
    demo_start = time.time()

    test_user_ids = [1, 10, 25, 50, 100]

    for user_id in test_user_ids:
        run_recommender_for_user(
            user_id=user_id,
            ratings=ratings,
            movies=movies,
            user_item_matrix=user_item_matrix,
            user_means=user_means,
            user_similarity_df=user_similarity_df,
            top_n=5,
            k=20,
            alpha=0.8,
        )

    print(f"\nDemo recommendation time: {time.time() - demo_start:.2f} seconds")


# ----------------------------
# Full Evaluation
# ----------------------------
def run_full_evaluation(ratings) -> None:
    """Evaluate model using Precision@K and RMSE."""
    print_section("EVALUATION")

    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    train_matrix = build_user_item_matrix(train_df)

    train_user_means, train_normalized_matrix = normalize_user_item_matrix(train_matrix)
    train_user_similarity_df = compute_user_similarity(train_normalized_matrix)

    # --- Precision@K
    precision = evaluate_precision_at_k(
        model_func=predict_scores_ranking_fast,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=train_user_similarity_df,
        k=10,
        model_kwargs={"k": 20},
    )

    print(f"\nPrecision@10: {precision:.4f}")

    # --- RMSE
    rmse_score = evaluate_rmse_fast(
        test_df=test_df,
        train_matrix=train_matrix,
        user_similarity_df=train_user_similarity_df,
        user_means=train_user_means,
        k=20,
    )

    print(f"RMSE: {rmse_score:.4f}")


# ----------------------------
# Model Comparison
# ----------------------------
def run_model_comparison(ratings, movies) -> None:
    """
    Compare CF, Content, and Hybrid models.

    Includes:
    - Alpha tuning
    - Cold-start simulation (handled internally)
    """
    print_section("MODEL COMPARISON")

    comparison_df = compare_models(
        ratings,
        movies,
        alphas=[i / 20 for i in range(21)],  # alpha sweep
        eval_user_limit=None,
    )

    print(comparison_df.to_string(index=False))


# ----------------------------
# Main Execution
# ----------------------------
def main() -> None:
    """Main execution pipeline."""

    total_start = time.time()

    # --- Load data
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "movielens"

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    # --- Build matrices
    user_item_matrix = build_user_item_matrix(ratings)
    user_means, normalized_matrix = normalize_user_item_matrix(user_item_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    # --- Run components
    if RUN_DEMO_RECOMMENDATIONS:
        run_demo_recommendations(
            ratings,
            movies,
            user_item_matrix,
            user_means,
            user_similarity_df,
        )

    if RUN_FULL_EVALUATION:
        run_full_evaluation(ratings)

    if RUN_MODEL_COMPARISON:
        run_model_comparison(ratings, movies)

    # --- Final summary
    print_section("FINAL SUMMARY")
    print("Hybrid recommendation system completed successfully.")
    print("Includes CF, Content, Hybrid, RMSE, Precision@K, and cold-start analysis.")

    print("\n" + "=" * 60)
    print(f"TOTAL RUNTIME: {time.time() - total_start:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()