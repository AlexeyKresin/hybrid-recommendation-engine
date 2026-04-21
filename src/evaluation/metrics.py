import numpy as np
import pandas as pd


def train_test_split_per_user(ratings: pd.DataFrame, test_size: int = 5):
    train_list = []
    test_list = []

    for _, group in ratings.groupby("user_id"):
        group = group.sort_values("timestamp")

        if len(group) <= test_size:
            train_list.append(group)
            continue

        test = group.tail(test_size)
        train = group.iloc[:-test_size]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame()

    return train_df, test_df


def precision_at_k(
    recommended_items: list[int],
    relevant_items: set[int],
    k: int
) -> float:
    if not relevant_items:
        return 0.0

    recommended_items = recommended_items[:k]
    hits = sum(1 for item in recommended_items if item in relevant_items)

    return hits / k


def evaluate_precision_at_k(
    model_func,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item_matrix,
    user_similarity_df,
    k: int = 10,
    model_kwargs: dict | None = None,
) -> float:
    precisions = []
    model_kwargs = model_kwargs or {}

    for user_id in test_df["user_id"].unique():
        if user_id not in user_item_matrix.index:
            continue

        test_user = test_df[test_df["user_id"] == user_id]

        relevant_items = set(
            test_user[test_user["rating"] >= 4]["movie_id"]
        )

        if not relevant_items:
            continue

        scores = model_func(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
            **model_kwargs,
        )

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended = [movie_id for movie_id, _ in ranked_items[:k]]

        p_at_k = precision_at_k(recommended, relevant_items, k)
        precisions.append(p_at_k)

    return float(np.mean(precisions)) if precisions else 0.0


def rmse(predictions: list[float], actuals: list[float]) -> float:
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def evaluate_rmse(
    predict_func,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_item_matrix,
    user_similarity_df,
    user_means,
    model_kwargs: dict | None = None,
) -> float:
    preds = []
    actuals = []
    model_kwargs = model_kwargs or {}

    for user_id in test_df["user_id"].unique():
        if user_id not in user_item_matrix.index:
            continue

        user_test = test_df[test_df["user_id"] == user_id]

        scores = predict_func(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            user_similarity_df=user_similarity_df,
            user_means=user_means,
            **model_kwargs,
        )

        for _, row in user_test.iterrows():
            movie_id = row["movie_id"]
            actual_rating = row["rating"]

            if movie_id in scores:
                preds.append(scores[movie_id])
                actuals.append(actual_rating)

    if not preds:
        return 0.0

    return rmse(preds, actuals)