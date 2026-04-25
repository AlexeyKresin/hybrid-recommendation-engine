from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


def load_movies(data_dir: Path) -> pd.DataFrame:
    columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url", *GENRE_COLUMNS
    ]

    return pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=columns,
        encoding="latin-1",
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw" / "movielens"

    movies = load_movies(data_dir)
    movie_features = movies[GENRE_COLUMNS]

    example_title = "Star Wars (1977)"
    example_movie = movies[movies["title"] == example_title].iloc[0]

    example_vector = movie_features.loc[example_movie.name].values.reshape(1, -1)
    similarities = cosine_similarity(example_vector, movie_features)[0]

    movies["similarity_to_reference"] = similarities

    similar_movies = (
        movies[movies["title"] != example_title]
        .sort_values("similarity_to_reference", ascending=False)
        .head(10)
    )

    different_movies = (
        movies[movies["title"] != example_title]
        .sort_values("similarity_to_reference", ascending=True)
        .head(10)
    )

    print("\nReference movie:")
    print(example_title)

    print("\nMost similar movies:")
    print(similar_movies[["movie_id", "title", "similarity_to_reference"]])

    print("\nMost different movies:")
    print(different_movies[["movie_id", "title", "similarity_to_reference"]])


if __name__ == "__main__":
    main()