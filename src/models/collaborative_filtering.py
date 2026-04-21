import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def normalize_user_item_matrix(user_item_matrix: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Mean-center each user's ratings.
    """
    user_means = user_item_matrix.mean(axis=1)
    normalized_matrix = user_item_matrix.sub(user_means, axis=0)
    return user_means, normalized_matrix


def compute_user_similarity(normalized_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between users.
    Missing values are filled with 0 after normalization.
    """
    filled_matrix = normalized_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix)

    return pd.DataFrame(
        similarity,
        index=normalized_matrix.index,
        columns=normalized_matrix.index
    )


def predict_scores_ranking(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    k: int = 20
) -> dict[int, float]:
    """
    Ranking-style score:
    Higher score means more recommendable.
    This is not a true predicted rating.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix")

    user_ratings = user_item_matrix.loc[user_id]
    similarities = user_similarity_df.loc[user_id].drop(user_id)

    #KEEP ONLY POSITEVELY CORRELATED USERS:
    similarities = similarities[similarities > 0]
    top_k_users = similarities.sort_values(ascending=False).head(k)

    scores: dict[int, float] = {}

    for movie_id in user_item_matrix.columns:
        if pd.isna(user_ratings[movie_id]):
            score = 0.0

            for other_user, sim in top_k_users.items():
                other_rating = user_item_matrix.loc[other_user, movie_id]
                if not pd.isna(other_rating):
                    score += sim * other_rating

            scores[movie_id] = score

    return scores


def predict_ratings_top_k(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    user_means: pd.Series,
    k: int = 20,
    min_neighbors: int = 3
) -> dict[int, float]:
    """
    Predict actual ratings using weighted sum of top-k similar users.
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix")

    user_ratings = user_item_matrix.loc[user_id]
    similarities = user_similarity_df.loc[user_id].drop(user_id)
    top_k_users = similarities.sort_values(ascending=False).head(k)

    predictions: dict[int, float] = {}

    for movie_id in user_item_matrix.columns:
        if pd.isna(user_ratings[movie_id]):
            numerator = 0.0
            denominator = 0.0
            neighbor_count = 0

            for other_user, sim in top_k_users.items():
                other_rating = user_item_matrix.loc[other_user, movie_id]

                if not pd.isna(other_rating):
                    numerator += sim * (other_rating - user_means[other_user])
                    denominator += abs(sim)
                    neighbor_count += 1

            # onlt keep predictions where we have at least min_neighbors contributing

            if denominator == 0 or neighbor_count < min_neighbors:
               continue

            predicted_rating = user_means[user_id] + (numerator / denominator)

            # Clip to MovieLens range
            predicted_rating = min(5.0, max(1.0, predicted_rating))

            predictions[movie_id] = predicted_rating

    return predictions


def recommend_top_n(
    scores: dict[int, float],
    top_n: int = 10
) -> list[tuple[int, float]]:
    """
    Sort scores descending and return top N.
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]