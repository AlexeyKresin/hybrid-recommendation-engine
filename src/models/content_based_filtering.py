"""
Module: content_based_filtering.py

Description:
This module implements a content-based recommendation system using movie genres.

Core Idea:
- Each movie is represented as a feature vector (genres)
- Each user is represented as a profile (average of liked movies)
- Recommendations are generated based on similarity between user profile and movies

Key Steps:
1. Build movie feature matrix
2. Build user profile from liked items
3. Compute similarity between user profile and unseen items
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Genre feature columns (MovieLens dataset)
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
    """
    Create a movie feature matrix based on genre indicators.

    Args:
        movies_df (DataFrame): Movies dataset with genre columns

    Returns:
        DataFrame: Matrix where rows = movies, columns = genres
    """
    feature_matrix = movies_df[["movie_id"] + GENRE_COLUMNS].copy()
    feature_matrix = feature_matrix.set_index("movie_id")

    return feature_matrix


def compute_movie_similarity(movie_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cosine similarity between movies based on genre features.

    Args:
        movie_feature_matrix (DataFrame): Movie feature matrix

    Returns:
        DataFrame: Movie-to-movie similarity matrix
    """
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
    """
    Build a user profile based on liked movies.

    The profile is the average feature vector of movies the user rated above a threshold.

    Args:
        user_id (int): Target user
        ratings_df (DataFrame): Ratings dataset
        movie_feature_matrix (DataFrame): Movie feature matrix
        min_rating (float): Threshold for "liked" movies

    Returns:
        Series: User preference vector over genres
    """
    # Get user's ratings
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]

    # Select movies the user liked
    liked_movies = user_ratings[user_ratings["rating"] >= min_rating]["movie_id"]

    # Get feature vectors of liked movies
    liked_features = movie_feature_matrix.loc[
        movie_feature_matrix.index.intersection(liked_movies)
    ]

    # If no liked movies → return zero vector
    if liked_features.empty:
        return pd.Series(0.0, index=movie_feature_matrix.columns)

    # User profile = average feature vector
    return liked_features.mean(axis=0)


def recommend_content_based(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_n: int = 10,
    min_rating: float = 4.0,
) -> dict[int, float]:
    """
    Generate content-based recommendations for a user.

    Args:
        user_id (int): Target user
        ratings_df (DataFrame): Ratings dataset
        movies_df (DataFrame): Movies dataset
        top_n (int): Number of recommendations
        min_rating (float): Threshold for liked movies

    Returns:
        dict: {movie_id: similarity_score}
    """
    # Build feature matrix
    movie_feature_matrix = build_movie_feature_matrix(movies_df)

    # Build user profile
    user_profile = build_user_profile(
        user_id=user_id,
        ratings_df=ratings_df,
        movie_feature_matrix=movie_feature_matrix,
        min_rating=min_rating,
    )

    # If user has no preferences → return empty
    if user_profile.sum() == 0:
        return {}

    # Movies already seen by user
    seen_movies = set(
        ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist()
    )

    scores = {}

    # Compute similarity between user profile and each unseen movie
    for movie_id, features in movie_feature_matrix.iterrows():

        if movie_id in seen_movies:
            continue

        score = cosine_similarity(
            [user_profile.values],
            [features.values]
        )[0][0]

        scores[movie_id] = float(score)

    # Sort and return top-N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return dict(ranked)