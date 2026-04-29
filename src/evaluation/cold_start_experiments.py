import pandas as pd


def run_new_item_cold_start(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """
    Simulates new item cold start by selecting popular movies
    and pretending they have no training ratings.
    """

    print("\n2. New Item Cold Start")
    print("-" * 60)
    print("Scenario: Selected movies have no training ratings.")
    print("Strategy: Content-based recommendation using metadata.")

    movie_stats = (
        ratings.groupby("movie_id")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
        .sort_values("rating_count", ascending=False)
    )

    cold_movies = movie_stats.head(10).merge(movies, on="movie_id", how="left")

    print(f"Cold-start movies tested: {len(cold_movies)}\n")
    print(f"{'Movie ID':<10} {'Title':<45} {'Old Rating Count':>18}")
    print("-" * 80)

    for _, row in cold_movies.iterrows():
        title = row["title"][:42] + "..." if len(row["title"]) > 45 else row["title"]
        print(f"{row['movie_id']:<10} {title:<45} {int(row['rating_count']):>18}")

    print("\nObservation:")
    print("CF struggles because these items have no rating history.")
    print("Content-based filtering works using metadata (title, genre, year).")


def run_new_system_cold_start(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """
    Simulates new system cold start (no user history).
    Uses popularity-based fallback.
    """

    print("\n3. New System Cold Start")
    print("-" * 60)
    print("Scenario: No user history available.")
    print("Strategy: Popularity-based fallback.")

    popularity_df = (
        ratings.groupby("movie_id")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )

    popularity_df = popularity_df[popularity_df["rating_count"] >= 50]

    popularity_df = popularity_df.merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )

    popularity_df = popularity_df.sort_values(
        by=["avg_rating", "rating_count"],
        ascending=False
    ).head(10)

    print()
    print(f"{'Rank':<6} {'Title':<45} {'Avg Rating':>12} {'Rating Count':>14}")
    print("-" * 85)

    for rank, (_, row) in enumerate(popularity_df.iterrows(), start=1):
        title = row["title"][:42] + "..." if len(row["title"]) > 45 else row["title"]
        print(
            f"{rank:<6} "
            f"{title:<45} "
            f"{row['avg_rating']:>12.2f} "
            f"{int(row['rating_count']):>14}"
        )


def run_cold_start_summary() -> None:
    """
    Prints final insight summary.
    """

    print("\nKey Insight")
    print("-" * 60)
    print("Hybrid helps when user history is limited.")
    print("Content-based filtering helps when item metadata is available.")
    print("Popularity fallback is needed when no personalization data exists.")


def run_extra_cold_start_experiments(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """
    Wrapper to run all additional cold-start experiments.
    """

    run_new_item_cold_start(ratings, movies)
    run_new_system_cold_start(ratings, movies)
    run_cold_start_summary()