"""
compare_models.py

Purpose:
--------
Efficiently compare multiple recommendation models:
- Collaborative Filtering (CF)
- Content-Based Filtering
- Hybrid model (CF + Content)

Key Features:
-------------
- Caches CF and Content scores per user (no recomputation)
- Supports alpha tuning for hybrid model
- Supports cold-start simulation
- Evaluates using Precision@5 and Precision@10

Workflow:
---------
1. Split data into train/test
2. (Optional) simulate cold-start by limiting ratings per user
3. Build matrices and similarity models
4. Precompute scores for all users
5. Evaluate CF, Content, and Hybrid models
"""

import pandas as pd
import time

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
from src.models.hybrid_recommender import combine_scores
from src.evaluation.metrics import (
    train_test_split_per_user,
    precision_at_k,
)


# ----------------------------
# Helper Functions
# ----------------------------

def _get_relevant_items(test_df: pd.DataFrame, user_id: int, min_rating: float = 4.0) -> set[int]:
    """
    Get relevant (liked) items for evaluation.

    Only items with rating >= min_rating are considered relevant.
    """
    user_test = test_df[test_df["user_id"] == user_id]
    return set(user_test[user_test["rating"] >= min_rating]["movie_id"])


def _rank_items(scores: dict[int, float], k: int) -> list[int]:
    """Return top-k movie IDs sorted by score."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in ranked[:k]]


def simulate_cold_start_train(train_df, ratings_per_user: int = 5):
    """
    Simulate cold-start scenario:
    Limit each user to a small number of ratings.

    This reduces available user history and tests model robustness.
    """
    return (
        train_df
        .sample(frac=1, random_state=42)
        .groupby("user_id", as_index=False, group_keys=False)
        .head(ratings_per_user)
        .reset_index(drop=True)
    )


# ----------------------------
# Main Comparison Function
# ----------------------------

def compare_models(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    alphas: list[float] | None = None,
    k_neighbors: int = 20,
    eval_user_limit: int | None = None,
) -> pd.DataFrame:

    start = time.time()

    # ----------------------------
    # Step 1: Train/Test Split
    # ----------------------------
    print("Step 1: Splitting data...")

    if alphas is None:
        alphas = [0.3, 0.5, 0.7]

    train_df, test_df = train_test_split_per_user(ratings, test_size=5)

    print(f"Completed in {time.time() - start:.2f}s")

    # ----------------------------
    # Step 2: Cold-start Simulation (optional)
    # ----------------------------
    USE_COLD_START = True

    if USE_COLD_START:
        print("=== COLD START EXPERIMENT (5 ratings/user) ===")
        train_df = simulate_cold_start_train(train_df, ratings_per_user=5)

    # ----------------------------
    # Step 3: Build User-Item Matrix
    # ----------------------------
    print("Step 2: Building user-item matrix...")
    train_matrix = build_user_item_matrix(train_df)

    # ----------------------------
    # Step 4: Normalize + Similarity
    # ----------------------------
    print("Step 3: Normalizing matrix...")
    user_means, normalized_matrix = normalize_user_item_matrix(train_matrix)

    print("Step 4: Computing user similarity...")
    user_similarity_df = compute_user_similarity(normalized_matrix)

    # ----------------------------
    # Step 5: Content Features
    # ----------------------------
    print("Step 5: Building movie feature matrix...")
    movie_feature_matrix = build_movie_feature_matrix(movies)

    # ----------------------------
    # Step 6: Select Evaluation Users
    # ----------------------------
    eval_users = [
        user_id
        for user_id in test_df["user_id"].unique()
        if user_id in train_matrix.index
    ]

    if eval_user_limit is not None:
        eval_users = eval_users[:eval_user_limit]

    print(f"Step 6: Processing {len(eval_users)} users...")

    # ----------------------------
    # Step 7: Precompute Scores (KEY OPTIMIZATION)
    # ----------------------------
    cf_scores_cache = {}
    content_scores_cache = {}
    relevant_items_cache = {}

    for i, user_id in enumerate(eval_users):
        if i % 50 == 0:
            print(f"Processing user {i+1}/{len(eval_users)}")

        relevant_items = _get_relevant_items(test_df, user_id)
        if not relevant_items:
            continue

        relevant_items_cache[user_id] = relevant_items

        # --- CF scores
        cf_scores_cache[user_id] = predict_scores_ranking_fast(
            user_id=user_id,
            user_item_matrix=train_matrix,
            user_similarity_df=user_similarity_df,
            k=k_neighbors,
        )

        # --- Content scores
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
        return pd.DataFrame(columns=["model", "precision@5", "precision@10", "rmse"])

    results = []

    # ----------------------------
    # Step 8: Evaluate CF
    # ----------------------------
    print("Step 8: Evaluating Collaborative Filtering...")

    cf_precision_5 = []
    cf_precision_10 = []

    for user_id in evaluated_users:
        relevant_items = relevant_items_cache[user_id]
        cf_scores = cf_scores_cache[user_id]

        cf_precision_5.append(precision_at_k(_rank_items(cf_scores, 5), relevant_items, 5))
        cf_precision_10.append(precision_at_k(_rank_items(cf_scores, 10), relevant_items, 10))

    results.append({
        "model": "Collaborative Filtering",
        "precision@5": sum(cf_precision_5) / len(cf_precision_5),
        "precision@10": sum(cf_precision_10) / len(cf_precision_10),
        "rmse": None,
    })

    # ----------------------------
    # Step 9: Evaluate Content
    # ----------------------------
    print("Step 9: Evaluating Content-Based...")

    content_precision_5 = []
    content_precision_10 = []

    for user_id in evaluated_users:
        relevant_items = relevant_items_cache[user_id]
        content_scores = content_scores_cache[user_id]

        content_precision_5.append(precision_at_k(_rank_items(content_scores, 5), relevant_items, 5))
        content_precision_10.append(precision_at_k(_rank_items(content_scores, 10), relevant_items, 10))

    results.append({
        "model": "Content-Based",
        "precision@5": sum(content_precision_5) / len(content_precision_5),
        "precision@10": sum(content_precision_10) / len(content_precision_10),
        "rmse": None,
    })

    # ----------------------------
    # Step 10: Evaluate Hybrid (alpha tuning)
    # ----------------------------
    print("Step 10: Evaluating Hybrid models...")

    for alpha in alphas:
        print(f"Evaluating alpha={alpha:.2f}")

        hybrid_precision_5 = []
        hybrid_precision_10 = []

        for user_id in evaluated_users:
            relevant_items = relevant_items_cache[user_id]

            hybrid_scores = combine_scores(
                collaborative_scores=cf_scores_cache[user_id],
                content_scores=content_scores_cache[user_id],
                alpha=alpha,
            )

            hybrid_precision_5.append(precision_at_k(_rank_items(hybrid_scores, 5), relevant_items, 5))
            hybrid_precision_10.append(precision_at_k(_rank_items(hybrid_scores, 10), relevant_items, 10))

        results.append({
            "model": f"Hybrid alpha={alpha}",
            "precision@5": sum(hybrid_precision_5) / len(hybrid_precision_5),
            "precision@10": sum(hybrid_precision_10) / len(hybrid_precision_10),
            "rmse": None,
        })

    print(f"Total runtime: {time.time() - start:.2f}s")

    return pd.DataFrame(results)