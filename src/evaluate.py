"""
Module: evaluate.py

Optimized evaluation script.

Changes:
- Precision@K still uses ranking scores.
- RMSE now uses predict_single_rating() instead of predicting all unseen movies.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data_processing.data_loader import (
    load_ratings,
    load_movies,
    build_user_item_matrix,
)

from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking,
    predict_single_rating,
)

from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
)

from src.evaluation.compare_models import compare_models


def main() -> None:
    total_start = time.time()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "movielens"

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    train_matrix = build_user_item_matrix(train_df)

    user_means, normalized_matrix = normalize_user_item_matrix(train_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    # -------------------------------
    # Precision@K Evaluation
    # -------------------------------
    start = time.time()

    precision = evaluate_precision_at_k(
        model_func=predict_scores_ranking,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        k=10,
        model_kwargs={"k": 20},
    )

    print(f"Precision@10: {precision:.4f}")
    print(f"Precision evaluation time: {time.time() - start:.2f} seconds")

    # -------------------------------
    # Fast RMSE Evaluation
    # -------------------------------
    start = time.time()

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        pred = predict_single_rating(
            user_id=row["user_id"],
            movie_id=row["movie_id"],
            user_item_matrix=train_matrix,
            user_similarity_df=user_similarity_df,
            user_means=user_means,
            k=20,
        )

        if not pd.isna(pred):
            y_true.append(row["rating"])
            y_pred.append(pred)

    rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"RMSE: {rmse_score:.4f}")
    print(f"RMSE evaluation time: {time.time() - start:.2f} seconds")
    print(f"RMSE evaluated on {len(y_true)} test ratings")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    start = time.time()

    comparison_df = compare_models(ratings, movies)

    print(comparison_df.to_string(index=False))
    print(f"Model comparison time: {time.time() - start:.2f} seconds")

    print("\n" + "=" * 60)
    print(f"TOTAL EVALUATION TIME: {time.time() - total_start:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()