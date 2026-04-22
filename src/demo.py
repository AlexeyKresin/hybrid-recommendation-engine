"""
Module: demo.py

Description:
This script demonstrates the recommendation system by generating
recommendations for selected users using multiple approaches:

- Collaborative Filtering (ranking-based)
- Collaborative Filtering (rating prediction)
- Content-Based Filtering
- Hybrid Recommendation (combination of both)

Responsibilities:
- Load dataset
- Build user-item matrix
- Compute similarity
- Generate recommendations for sample users
- Display formatted results

How to run:
    python -m src.demo
    OR
    PYTHONPATH=. python src/demo.py
"""

from pathlib import Path

# Data loading
from src.data_processing.data_loader import (
    load_ratings,
    load_movies,
    build_user_item_matrix,
)

# Collaborative filtering
from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking,
    predict_ratings_top_k,
    recommend_top_n,
)

# Content-based model
from src.models.content_based_filtering import recommend_content_based

# Hybrid model
from src.models.hybrid_recommender import combine_scores

# Utilities
from src.utils.helpers import format_recommendations


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
    """
    Generates and prints recommendations for a single user using multiple models.

    Args:
        user_id (int): Target user ID
        ratings (DataFrame): Ratings dataset
        movies (DataFrame): Movies metadata
        user_item_matrix (DataFrame): User-item interaction matrix
        user_means (Series): Mean rating per user
        user_similarity_df (DataFrame): User similarity matrix
        top_n (int): Number of recommendations to display
        k (int): Number of neighbors for collaborative filtering
        alpha (float): Weight for hybrid model (0 = content, 1 = collaborative)

    Output:
        Prints formatted recommendation tables for each model
    """

    # -------------------------------
    # 1. Ranking-based CF
    # -------------------------------
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

    # -------------------------------
    # 2. Rating prediction CF
    # -------------------------------
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

    # -------------------------------
    # 3. Content-based filtering
    # -------------------------------
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

    # -------------------------------
    # 4. Hybrid model
    # -------------------------------
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

    # -------------------------------
    # Output results
    # -------------------------------
    print(f"\n{'=' * 60}")
    print(f"Recommendations for user {user_id}")
    print(f"{'=' * 60}")

    print("\nTop recommendations by ranking score:\n")
    print(formatted_ranked.to_string(index=False))

    print("\nTop recommendations by predicted score:\n")
    print(formatted_predicted.to_string(index=False))

    print("\nTop recommendations by content-based score:\n")
    print(formatted_content.to_string(index=False))

    print("\nTop recommendations by hybrid score:\n")
    print(formatted_hybrid.to_string(index=False))


def main() -> None:
    """
    Entry point for the demo script.

    Steps:
    1. Load dataset
    2. Build user-item matrix
    3. Compute similarity
    4. Run recommender for sample users
    """

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "movielens"

    # Load data
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    # Build user-item matrix
    user_item_matrix = build_user_item_matrix(ratings)

    # Normalize + compute similarity
    user_means, normalized_matrix = normalize_user_item_matrix(user_item_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    # Example users for demo
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
            alpha=0.5,
        )


if __name__ == "__main__":
    main()