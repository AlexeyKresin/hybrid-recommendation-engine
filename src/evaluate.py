"""
Module: evaluate.py

Description:
This script evaluates the performance of the recommendation system using
multiple metrics and compares different models.

Responsibilities:
- Load dataset (ratings + movies)
- Split data into train/test sets
- Build user-item matrix
- Compute similarity between users
- Evaluate:
    • Precision@K (ranking quality)
    • RMSE (rating prediction accuracy)
- Compare multiple models
- Measure execution time for performance analysis

How to run:
    python -m src.evaluate
    OR
    PYTHONPATH=. python src/evaluate.py
"""

from pathlib import Path
import time

# Data loading
from src.data_processing.data_loader import (
    load_ratings,
    load_movies,
    build_user_item_matrix,
)

# Collaborative filtering models
from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking,
    predict_ratings_top_k,
)

# Evaluation metrics
from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
    evaluate_rmse,
)

# Model comparison utility
from src.evaluation.compare_models import compare_models


def main() -> None:
    """
    Main evaluation pipeline.

    Steps:
    1. Load data
    2. Split into train/test
    3. Build user-item matrix
    4. Compute similarity
    5. Evaluate ranking (Precision@K)
    6. Evaluate rating prediction (RMSE)
    7. Compare models
    8. Report execution times
    """

    # Start total execution timer
    total_start = time.time()

    # Resolve project root and dataset path
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "movielens"

    # Load datasets
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Split dataset into train and test per user
    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    # Build user-item interaction matrix
    train_matrix = build_user_item_matrix(train_df)

    # Normalize ratings (mean-centering per user)
    user_means, normalized_matrix = normalize_user_item_matrix(train_matrix)

    # Compute similarity between users
    user_similarity_df = compute_user_similarity(normalized_matrix)

    # -------------------------------
    # Precision@K Evaluation (Ranking)
    # -------------------------------
    start = time.time()

    precision = evaluate_precision_at_k(
        model_func=predict_scores_ranking,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        k=10,  # evaluate top-10 recommendations
        model_kwargs={"k": 20},  # number of neighbors
    )

    print(f"Precision@10: {precision:.4f}")
    print(f"Precision evaluation time: {time.time() - start:.2f} seconds")

    # -------------------------------
    # RMSE Evaluation (Rating Accuracy)
    # -------------------------------
    start = time.time()

    rmse_score = evaluate_rmse(
        predict_func=predict_ratings_top_k,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        user_means=user_means,
        model_kwargs={
            "k": 20,              # neighbors
            "min_neighbors": 3,   # minimum required neighbors
        },
    )

    print(f"RMSE: {rmse_score:.4f}")
    print(f"RMSE evaluation time: {time.time() - start:.2f} seconds")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # -------------------------------
    # Compare multiple models
    # -------------------------------
    start = time.time()

    comparison_df = compare_models(ratings, movies)

    print(comparison_df.to_string(index=False))
    print(f"Model comparison time: {time.time() - start:.2f} seconds")

    # -------------------------------
    # Total execution time
    # -------------------------------
    print("\n" + "=" * 60)
    print(f"TOTAL EVALUATION TIME: {time.time() - total_start:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()