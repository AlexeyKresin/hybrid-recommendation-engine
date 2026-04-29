"""
Main entry point for Hybrid Recommendation System.

Runs:
1. Demo recommendations
2. Evaluation
3. Model comparison / cold-start experiments
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
    predict_scores_ranking_fast,
    predict_single_rating,
    recommend_top_n,
)

from src.models.content_based_filtering import recommend_content_based
from src.models.hybrid_recommender import combine_scores

from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
)

from src.evaluation.compare_models import compare_models
from src.evaluation.cold_start_experiments import run_extra_cold_start_experiments
from src.utils.helpers import format_recommendations

from src.utils.report_printer import (
    print_report_header,
    print_metric_section,
    print_fixed_vs_adaptive_section,
    print_final_insights,
)


RUN_DEMO_RECOMMENDATIONS = True
RUN_FULL_EVALUATION = True
RUN_MODEL_COMPARISON = True


def print_section(title: str) -> None:
    print(f"\n{'=' * 22} {title} {'=' * 22}")


def print_user_profile_summary(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    user_ratings = ratings[ratings["user_id"] == user_id]
    merged = user_ratings.merge(movies, on="movie_id", how="left")

    liked_movies = (
        merged[merged["rating"] >= 4]
        .sort_values("rating", ascending=False)
        .head(3)
    )

    print("\nUser Profile")
    print("-" * 60)
    print(f"User ID        : {user_id}")
    print(f"Ratings Count  : {len(user_ratings)}")
    print(f"Average Rating : {user_ratings['rating'].mean():.2f}")

    print("\nLiked Movies")
    print("-" * 60)

    for _, row in liked_movies.iterrows():
        stars = "★" * int(row["rating"])
        print(f"{row['title']}  {stars}")


def run_recommender_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    top_n: int = 5,
    k: int = 20,
    alpha: float = 0.8,
) -> None:
    cf_scores = predict_scores_ranking_fast(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
        k=k,
    )

    content_scores = recommend_content_based(
        user_id=user_id,
        ratings_df=ratings,
        movies_df=movies,
        top_n=1000,
        min_rating=3.0,
    )

    hybrid_scores = combine_scores(
        collaborative_scores=cf_scores,
        content_scores=content_scores,
        alpha=alpha,
    )

    print_section(f"USER {user_id}")
    print_user_profile_summary(user_id, ratings, movies)

    top_hybrid = recommend_top_n(hybrid_scores, top_n=top_n)

    formatted = format_recommendations(
        top_hybrid,
        movies,
        score_column="Hybrid Score",
        clip_ratings=False,
    )

    print("\nTop Hybrid Recommendations")
    print("-" * 60)
    print(formatted.to_string(index=False))

    print("\nWhy these recommendations?")
    print("These movies combine signals from similar users and similar movie features.")


def evaluate_rmse_fast(
    test_df: pd.DataFrame,
    train_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 20,
) -> float:
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

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_demo_recommendations(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
) -> None:
    demo_start = time.time()
    demo_users = [1, 50, 100]

    print_section("DEMO RECOMMENDATIONS")

    for user_id in demo_users:
        run_recommender_for_user(
            user_id=user_id,
            ratings=ratings,
            movies=movies,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
            top_n=5,
            k=20,
            alpha=0.8,
        )

    print(f"\nDemo Runtime: {time.time() - demo_start:.2f} seconds")


def run_full_evaluation(ratings: pd.DataFrame) -> None:
    #print_section("EVALUATION")

    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    train_matrix = build_user_item_matrix(train_df)
    train_user_means, train_normalized_matrix = normalize_user_item_matrix(train_matrix)
    train_user_similarity_df = compute_user_similarity(train_normalized_matrix)

    precision = evaluate_precision_at_k(
        model_func=predict_scores_ranking_fast,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=train_user_similarity_df,
        k=10,
        model_kwargs={"k": 20},
    )

    rmse_score = evaluate_rmse_fast(
        test_df=test_df,
        train_matrix=train_matrix,
        user_similarity_df=train_user_similarity_df,
        user_means=train_user_means,
        k=20,
    )

    print_metric_section(precision, rmse_score)


def run_model_comparison(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    print_section("COLD-START MODEL COMPARISON")

    comparison_df = compare_models(
        ratings,
        movies,
        alphas=[i / 20 for i in range(21)],
        eval_user_limit=None,
    )

    best_row = comparison_df.loc[comparison_df["precision@10"].idxmax()]

    cf_row = comparison_df[
        comparison_df["model"] == "Collaborative Filtering"
    ].iloc[0]

    content_row = comparison_df[
        comparison_df["model"] == "Content-Based"
    ].iloc[0]

    adaptive_row = comparison_df[
        comparison_df["model"] == "Adaptive Hybrid"
    ].iloc[0]

    summary_df = pd.DataFrame([
        {
            "Model": "Collaborative Filtering",
            "Precision@5": cf_row["precision@5"],
            "Precision@10": cf_row["precision@10"],
        },
        {
            "Model": "Content-Based",
            "Precision@5": content_row["precision@5"],
            "Precision@10": content_row["precision@10"],
        },
        {
            "Model": f"Best Hybrid ({best_row['model']})",
            "Precision@5": best_row["precision@5"],
            "Precision@10": best_row["precision@10"],
        },
        {
            "Model": "Adaptive Hybrid",
            "Precision@5": adaptive_row["precision@5"],
            "Precision@10": adaptive_row["precision@10"],
        },
    ])

    print_fixed_vs_adaptive_section(comparison_df)

    print("\n1. New User Cold Start")
    print("-" * 60)
    print("Scenario: Each user has only 5 ratings.")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest Model")
    print("-" * 60)
    print(f"Model        : {best_row['model']}")
    print(f"Precision@5 : {best_row['precision@5']:.4f}")
    print(f"Precision@10: {best_row['precision@10']:.4f}")

    if "avg_alpha" in adaptive_row.index and pd.notna(adaptive_row["avg_alpha"]):
        print(f"Adaptive Avg Alpha: {adaptive_row['avg_alpha']:.4f}")

    run_extra_cold_start_experiments(ratings, movies)


def main() -> None:
    total_start = time.time()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "movielens"

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    user_item_matrix = build_user_item_matrix(ratings)
    _, normalized_matrix = normalize_user_item_matrix(user_item_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    if RUN_DEMO_RECOMMENDATIONS:
        run_demo_recommendations(
            ratings=ratings,
            movies=movies,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
        )

    if RUN_FULL_EVALUATION:
        run_full_evaluation(ratings)

    if RUN_MODEL_COMPARISON:
        run_model_comparison(ratings, movies)

    print_section("FINAL SUMMARY")
    print("Hybrid recommendation system completed successfully.")
    print("CF, Content-Based Filtering, Hybrid Scoring, RMSE, Precision@K,")
    print("alpha tuning, adaptive alpha, and all three cold-start scenarios are included.")

    print(f"\nTotal Runtime: {time.time() - total_start:.2f} seconds")

    print_final_insights()


if __name__ == "__main__":
    main()