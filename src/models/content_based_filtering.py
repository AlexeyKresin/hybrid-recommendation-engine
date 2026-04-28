"""
content_based_filtering.py

Purpose:
--------
Implements Content-Based Filtering for movie recommendations.

Core Idea:
----------
Recommend movies that are similar to movies the user already liked.

Movie Features:
---------------
- Genre indicators
- TF-IDF title features
- Normalized release year

Workflow:
---------
1. Build a movie feature matrix
2. Build a user profile from movies the user liked
3. Compute cosine similarity between the user profile and all movies
4. Recommend the most similar unseen movies
"""

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
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


def extract_year(title: str) -> int | None:
    """
    Extract release year from a movie title.

    Example:
    --------
    "Toy Story (1995)" -> 1995
    """
    match = re.search(r"\((\d{4})\)", str(title))
    return int(match.group(1)) if match else None


def build_movie_feature_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature vectors for all movies.

    Features include:
    - Genre columns
    - TF-IDF title representation
    - Normalized release year
    """
    movies = movies_df.copy()

    # Genre features: binary indicators such as Action, Comedy, Drama
    genre_matrix = csr_matrix(movies[GENRE_COLUMNS].values)

    # Title features: TF-IDF captures useful words from movie titles
    title_text = movies["title"].fillna("")
    tfidf = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=500,
    )
    title_matrix = tfidf.fit_transform(title_text)

    # Year feature: normalize release year to range [0, 1]
    movies["year"] = movies["title"].apply(extract_year)
    movies["year"] = movies["year"].fillna(movies["year"].median())

    year_values = movies["year"].values.reshape(-1, 1)
    year_range = year_values.max() - year_values.min()

    if year_range == 0:
        normalized_year = np.zeros_like(year_values)
    else:
        normalized_year = (year_values - year_values.min()) / year_range

    year_matrix = csr_matrix(normalized_year)

    # Combine all content features into one movie representation
    combined_matrix = hstack(
        [
            genre_matrix,
            title_matrix,
            year_matrix,
        ]
    )

    return pd.DataFrame(
        combined_matrix.toarray(),
        index=movies["movie_id"],
    )


def compute_movie_similarity(movie_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute movie-to-movie cosine similarity.

    This is useful for inspecting which movies are similar to each other.
    """
    similarity = cosine_similarity(movie_feature_matrix.values)

    return pd.DataFrame(
        similarity,
        index=movie_feature_matrix.index,
        columns=movie_feature_matrix.index,
    )


def build_user_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movie_feature_matrix: pd.DataFrame,
    min_rating: float = 3.0,
) -> pd.Series:
    """
    Build a content profile for one user.

    The profile is the average feature vector of movies the user liked.
    Movies with rating >= min_rating are treated as positive examples.

    The final profile is normalized so cosine similarity focuses on direction.
    """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]

    liked_movies = user_ratings[user_ratings["rating"] >= min_rating]["movie_id"]
    liked_movie_ids = movie_feature_matrix.index.intersection(liked_movies)

    if liked_movie_ids.empty:
        return pd.Series(0.0, index=movie_feature_matrix.columns)

    liked_features = movie_feature_matrix.loc[liked_movie_ids]

    user_profile = liked_features.mean(axis=0)

    profile_norm = np.linalg.norm(user_profile)

    if profile_norm > 0:
        user_profile = user_profile / profile_norm

    return user_profile


def recommend_content_based(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame | None = None,
    movie_feature_matrix: pd.DataFrame | None = None,
    top_n: int = 10,
    min_rating: float = 3.0,
) -> dict[int, float]:
    """
    Generate content-based recommendation scores for a user.

    Returns:
    --------
    dict:
        {movie_id: content_similarity_score}
    """
    if movie_feature_matrix is None:
        if movies_df is None:
            raise ValueError("Either movies_df or movie_feature_matrix must be provided")

        movie_feature_matrix = build_movie_feature_matrix(movies_df)

    user_profile = build_user_profile(
        user_id=user_id,
        ratings_df=ratings_df,
        movie_feature_matrix=movie_feature_matrix,
        min_rating=min_rating,
    )

    if user_profile.sum() == 0:
        return {}

    seen_movies = set(
        ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"]
    )

    user_vector = user_profile.values.reshape(1, -1)

    similarity_scores = cosine_similarity(
        user_vector,
        movie_feature_matrix.values,
    )[0]

    # Return only movies the user has not already rated
    scores = {
        movie_id: float(score)
        for movie_id, score in zip(movie_feature_matrix.index, similarity_scores)
        if movie_id not in seen_movies
    }

    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return dict(top_scores)