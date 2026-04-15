import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Childrens",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def build_movie_feature_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    feature_matrix = movies_df[["movie_id"] + GENRE_COLUMNS].copy()
    feature_matrix = feature_matrix.set_index("movie_id")
    return feature_matrix


def compute_movie_similarity(movie_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(movie_feature_matrix)

    return pd.DataFrame(
        similarity,
        index=movie_feature_matrix.index,
        columns=movie_feature_matrix.index,
    )


def build_user_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movie_feature_matrix: pd.DataFrame,
    min_rating: float = 4.0,
) -> pd.Series:
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    liked_movies = user_ratings[user_ratings["rating"] >= min_rating]["movie_id"]

    liked_features = movie_feature_matrix.loc[
        movie_feature_matrix.index.intersection(liked_movies)
    ]

    if liked_features.empty:
        return pd.Series(0.0, index=movie_feature_matrix.columns)

    return liked_features.mean(axis=0)


def recommend_content_based(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_n: int = 10,
    min_rating: float = 4.0,
) -> dict[int, float]:
    movie_feature_matrix = build_movie_feature_matrix(movies_df)
    user_profile = build_user_profile(
        user_id=user_id,
        ratings_df=ratings_df,
        movie_feature_matrix=movie_feature_matrix,
        min_rating=min_rating,
    )

    if user_profile.sum() == 0:
        return {}

    seen_movies = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist())

    scores = {}
    for movie_id, features in movie_feature_matrix.iterrows():
        if movie_id in seen_movies:
            continue

        score = cosine_similarity(
            [user_profile.values],
            [features.values]
        )[0][0]

        scores[movie_id] = float(score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(ranked)