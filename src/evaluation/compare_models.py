"""
compare_models.py

Compares:
- Collaborative Filtering
- Content-Based Filtering
- Fixed-alpha Hybrid model
- Adaptive-alpha Hybrid model

Supports:
- Normal model comparison
- New User Cold Start simulation
"""

import time
import pandas as pd

from src.data_processing.data_loader import build_user_item_matrix
from src.models.collaborative_filtering import (
    normalize_user_item_matrix,
    compute_user_similarity,
    predict_scores_ranking_fast,
)
from src.models.content_based_filtering import (
    recommend_content_based,
    build_movie_feature_matrix,
)
from src.models.hybrid_recommender import combine_scores, combine_scores_adaptive
from src.evaluation.metrics import (
    train_test_split_per_user,
    precision_at_k,
)


def _get_relevant_items(
    test_df: pd.DataFrame,
    user_id: int,
    min_rating: float = 4.0,
) -> set[int]:
    """Return movies rated highly by a user in the test set."""
    user_test = test_df[test_df["user_id"] == user_id]
    return set(user_test[user_test["rating"] >= min_rating]["movie_id"])


def _rank_items(scores: dict[int, float], k: int) -> list[int]:
    """Return top-k movie IDs sorted by score."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in ranked[:k]]


def _mean(values: list[float]) -> float:
    """Safely compute mean."""
    return sum(values) / len(values) if values else 0.0


def simulate_cold_start_train(
    train_df: pd.DataFrame,
    ratings_per_user: int = 5,
) -> pd.DataFrame:
    """
    Simulate new-user cold start by limiting each user
    to a small number of training ratings.
    """
    return (
        train_df
        .sample(frac=1, random_state=42)
        .groupby("user_id", as_index=False, group_keys=False)
        .head(ratings_per_user)
        .reset_index(drop=True)
    )


def compare_models(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    alphas: list[float] | None = None,
    k_neighbors: int = 20,
    eval_user_limit: int | None = None,
    cold_start: bool = True,
    ratings_per_user: int = 5,
) -> pd.DataFrame:
    """
    Compare CF, Content-Based, Fixed Hybrid, and Adaptive Hybrid recommenders.

    If cold_start=True, each user is limited to ratings_per_user
    training ratings to simulate the New User Cold Start scenario.
    """

    start = time.time()

    if alphas is None:
        alphas = [0.3, 0.5, 0.7]

    print("Step 1: Splitting data...")
    train_df, test_df = train_test_split_per_user(ratings, test_size=5)
    print(f"Completed in {time.time() - start:.2f}s")

    if cold_start:
        print(
            f"Step 2: Simulating new-user cold start "
            f"({ratings_per_user} ratings/user)..."
        )
        train_df = simulate_cold_start_train(
            train_df,
            ratings_per_user=ratings_per_user,
        )

    print("Step 3: Building user-item matrix...")
    train_matrix = build_user_item_matrix(train_df)

    print("Step 4: Computing user similarity...")
    _, normalized_matrix = normalize_user_item_matrix(train_matrix)
    user_similarity_df = compute_user_similarity(normalized_matrix)

    print("Step 5: Building movie feature matrix...")
    movie_feature_matrix = build_movie_feature_matrix(movies)

    eval_users = [
        user_id
        for user_id in test_df["user_id"].unique()
        if user_id in train_matrix.index
    ]

    if eval_user_limit is not None:
        eval_users = eval_users[:eval_user_limit]

    print(f"Step 6: Processing {len(eval_users)} users...")

    cf_scores_cache = {}
    content_scores_cache = {}
    relevant_items_cache = {}

    for i, user_id in enumerate(eval_users, start=1):
        if i == 1 or i % 50 == 0:
            print(f"Processing user {i}/{len(eval_users)}")

        relevant_items = _get_relevant_items(test_df, user_id)

        if not relevant_items:
            continue

        relevant_items_cache[user_id] = relevant_items

        cf_scores_cache[user_id] = predict_scores_ranking_fast(
            user_id=user_id,
            user_item_matrix=train_matrix,
            user_similarity_df=user_similarity_df,
            k=k_neighbors,
        )

        content_scores_cache[user_id] = recommend_content_based(
            user_id=user_id,
            ratings_df=train_df,
            movies_df=movies,
            movie_feature_matrix=movie_feature_matrix,
            top_n=1000,
            min_rating=3.0,
        )

    print(f"User scoring completed in {time.time() - start:.2f}s")

    evaluated_users = list(relevant_items_cache.keys())

    if not evaluated_users:
        return pd.DataFrame(
            columns=["model", "precision@5", "precision@10", "rmse", "avg_alpha"]
        )

    results = []

    print("Step 7: Evaluating Collaborative Filtering...")

    cf_precision_5 = []
    cf_precision_10 = []

    for user_id in evaluated_users:
        relevant_items = relevant_items_cache[user_id]
        cf_scores = cf_scores_cache[user_id]

        cf_precision_5.append(
            precision_at_k(_rank_items(cf_scores, 5), relevant_items, 5)
        )
        cf_precision_10.append(
            precision_at_k(_rank_items(cf_scores, 10), relevant_items, 10)
        )

    results.append({
        "model": "Collaborative Filtering",
        "precision@5": _mean(cf_precision_5),
        "precision@10": _mean(cf_precision_10),
        "rmse": None,
        "avg_alpha": None,
    })

    print("Step 8: Evaluating Content-Based...")

    content_precision_5 = []
    content_precision_10 = []

    for user_id in evaluated_users:
        relevant_items = relevant_items_cache[user_id]
        content_scores = content_scores_cache[user_id]

        content_precision_5.append(
            precision_at_k(_rank_items(content_scores, 5), relevant_items, 5)
        )
        content_precision_10.append(
            precision_at_k(_rank_items(content_scores, 10), relevant_items, 10)
        )

    results.append({
        "model": "Content-Based",
        "precision@5": _mean(content_precision_5),
        "precision@10": _mean(content_precision_10),
        "rmse": None,
        "avg_alpha": None,
    })

    print("Step 9: Evaluating Fixed Hybrid models...")

    for alpha in alphas:
        hybrid_precision_5 = []
        hybrid_precision_10 = []

        for user_id in evaluated_users:
            relevant_items = relevant_items_cache[user_id]

            hybrid_scores = combine_scores(
                collaborative_scores=cf_scores_cache[user_id],
                content_scores=content_scores_cache[user_id],
                alpha=alpha,
            )

            hybrid_precision_5.append(
                precision_at_k(_rank_items(hybrid_scores, 5), relevant_items, 5)
            )
            hybrid_precision_10.append(
                precision_at_k(_rank_items(hybrid_scores, 10), relevant_items, 10)
            )

        results.append({
            "model": f"Hybrid alpha={alpha:.2f}",
            "precision@5": _mean(hybrid_precision_5),
            "precision@10": _mean(hybrid_precision_10),
            "rmse": None,
            "avg_alpha": alpha,
        })

    print("Step 10: Evaluating Adaptive Hybrid model...")

    adaptive_precision_5 = []
    adaptive_precision_10 = []
    adaptive_alphas = []

    for user_id in evaluated_users:
        relevant_items = relevant_items_cache[user_id]

        adaptive_scores, user_alpha = combine_scores_adaptive(
            user_id=user_id,
            ratings_df=train_df,
            collaborative_scores=cf_scores_cache[user_id],
            content_scores=content_scores_cache[user_id],
            min_alpha=0.6,
            max_alpha=0.9,
            max_ratings=50,
        )

        adaptive_alphas.append(user_alpha)

        adaptive_precision_5.append(
            precision_at_k(_rank_items(adaptive_scores, 5), relevant_items, 5)
        )
        adaptive_precision_10.append(
            precision_at_k(_rank_items(adaptive_scores, 10), relevant_items, 10)
        )

    results.append({
        "model": "Adaptive Hybrid",
        "precision@5": _mean(adaptive_precision_5),
        "precision@10": _mean(adaptive_precision_10),
        "rmse": None,
        "avg_alpha": _mean(adaptive_alphas),
    })

    print(f"Total runtime: {time.time() - start:.2f}s")

    return pd.DataFrame(results)