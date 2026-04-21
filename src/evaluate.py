from pathlib import Path
import time

from src.data_processing.data_loader import (
    load_ratings,
    load_movies,
    build_user_item_matrix,
)
from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking,
    predict_ratings_top_k,
)
from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
    evaluate_rmse,
)
from src.evaluation.compare_models import compare_models


def main() -> None:
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

    start = time.time()
    rmse_score = evaluate_rmse(
        predict_func=predict_ratings_top_k,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        user_means=user_means,
        model_kwargs={"k": 20, "min_neighbors": 3},
    )
    print(f"RMSE: {rmse_score:.4f}")
    print(f"RMSE evaluation time: {time.time() - start:.2f} seconds")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    start = time.time()
    #comparison_df = compare_models(ratings, movies)
    print(comparison_df.to_string(index=False))
    print(f"Model comparison time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()