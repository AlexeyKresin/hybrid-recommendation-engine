from pathlib import Path
import pandas as pd


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    """
    Load MovieLens ratings data.
    """
    data_dir = Path(data_dir)
    ratings_path = data_dir / "u.data"

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    return ratings


def load_movies(data_dir: str | Path) -> pd.DataFrame:
    """
    Load MovieLens movie metadata.
    """
    data_dir = Path(data_dir)
    movies_path = data_dir / "u.item"

    movie_cols = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=movie_cols
    )
    return movies


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-item matrix with users as rows and movies as columns.
    """
    return ratings.pivot(index="user_id", columns="movie_id", values="rating")