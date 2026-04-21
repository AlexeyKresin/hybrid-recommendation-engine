from pathlib import Path

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
    recommend_top_n,
)
from src.evaluation.metrics import (
    train_test_split_per_user,
    evaluate_precision_at_k,
    evaluate_rmse,
)
from src.models.content_based_filtering import recommend_content_based
from src.models.hybrid_recommender import combine_scores
from src.utils.helpers import format_recommendations
from src.evaluation.compare_models import compare_models


def run_recommender_for_user(
    user_id: int,
    ratings,
    movies,
    user_item_matrix,
    user_means,
    user_similarity_df,
    top_n: int = 5,
    k: int = 20,
    alpha: float = 0.5,
) -> None:
    ranking_scores = predict_scores_ranking(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
        k=k,
    )
    top_ranked = recommend_top_n(ranking_scores, top_n=top_n)
    formatted_ranked = format_recommendations(
        top_ranked,
        movies,
        score_column="score",
        clip_ratings=False,
    )

    predicted_scores = predict_ratings_top_k(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        user_similarity_df=user_similarity_df,
        user_means=user_means,
        k=k,
        min_neighbors=3,
    )
    top_predicted = recommend_top_n(predicted_scores, top_n=top_n)
    formatted_predicted = format_recommendations(
        top_predicted,
        movies,
        score_column="predicted_score",
        clip_ratings=False,
    )

    content_scores = recommend_content_based(
        user_id=user_id,
        ratings_df=ratings,
        movies_df=movies,
        top_n=top_n,
        min_rating=4.0,
    )
    top_content = recommend_top_n(content_scores, top_n=top_n)
    formatted_content = format_recommendations(
        top_content,
        movies,
        score_column="content_score",
        clip_ratings=False,
    )

    hybrid_scores = combine_scores(
        collaborative_scores=ranking_scores,
        content_scores=content_scores,
        alpha=alpha,
    )
    top_hybrid = recommend_top_n(hybrid_scores, top_n=top_n)
    formatted_hybrid = format_recommendations(
        top_hybrid,
        movies,
        score_column="hybrid_score",
        clip_ratings=False,
    )

    print(f"\n{'=' * 60}")
    print(f"Recommendations for user {user_id}")
    print(f"{'=' * 60}")

    print("\nTop recommendations by ranking score:\n")
    print(formatted_ranked.to_string(index=False))

    print("\nTop recommendations by predicted score (raw collaborative filtering output):\n")
    print(formatted_predicted.to_string(index=False))

    print("\nTop recommendations by content-based score:\n")
    print(formatted_content.to_string(index=False))

    print("\nTop recommendations by hybrid score:\n")
    print(formatted_hybrid.to_string(index=False))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "movielens"

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    user_item_matrix = build_user_item_matrix(ratings)

    user_means, normalized_matrix = normalize_user_item_matrix(user_item_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    test_user_ids = [1, 10, 25, 50, 100]
    top_n = 5
    k = 20
    alpha = 0.5

    for user_id in test_user_ids:
        run_recommender_for_user(
            user_id=user_id,
            ratings=ratings,
            movies=movies,
            user_item_matrix=user_item_matrix,
            user_means=user_means,
            user_similarity_df=user_similarity_df,
            top_n=top_n,
            k=k,
            alpha=alpha,
        )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    train_matrix = build_user_item_matrix(train_df)
    user_means, normalized_matrix = normalize_user_item_matrix(train_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    precision = evaluate_precision_at_k(
        model_func=predict_scores_ranking,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        k=10,
        model_kwargs={"k": 20},
    )

    rmse_score = evaluate_rmse(
        predict_func=predict_ratings_top_k,
        train_df=train_df,
        test_df=test_df,
        user_item_matrix=train_matrix,
        user_similarity_df=user_similarity_df,
        user_means=user_means,
        model_kwargs={"k": 20, "min_neighbors": 3},
    )

    print(f"\nPrecision@10: {precision:.4f}")
    print(f"RMSE: {rmse_score:.4f}")
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison_df = compare_models(ratings, movies)
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()