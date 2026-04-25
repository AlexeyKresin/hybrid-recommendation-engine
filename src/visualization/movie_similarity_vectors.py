from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "figures"
    output_dir.mkdir(exist_ok=True)

    reference_movie = "Star Wars (1977)"

    movies = [
        ("Return of the Jedi (1983)", 1.00, "Similar"),
        ("Empire Strikes Back (1980)", 0.91, "Similar"),
        ("Billy Madison (1995)", 0.00, "Different"),
        ("Clerks (1994)", 0.00, "Different"),
    ]

    # Define angles (degrees)
    custom_angles = {
        "Return of the Jedi (1983)": 5,
        "Empire Strikes Back (1980)": np.degrees(np.arccos(0.91)),
        "Billy Madison (1995)": 90,
        "Clerks (1994)": -90,
    }

    colors = {
        "Similar": "green",
        "Different": "red",
    }

    plt.figure(figsize=(10, 7))

    for title, similarity, group in movies:
        angle = np.radians(custom_angles[title])

        x = np.cos(angle)
        y = np.sin(angle)

        # Arrow
        plt.arrow(
            0, 0, x, y,
            head_width=0.04,
            length_includes_head=True,
            color=colors[group],
            alpha=0.8
        )

        # Point
        plt.scatter(
            x, y,
            s=160,
            color=colors[group],
            edgecolor="black",
            zorder=3,
        )

        # Label
        plt.text(
            x + 0.05,
            y + 0.05,
            f"{title}\ncos={similarity:.2f}",
            fontsize=10
        )

    # Origin
    plt.scatter(0, 0, marker="x", s=100, color="black")
    plt.text(0.03, 0.03, "Origin", fontsize=10)

    # Axes
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)

    plt.xlim(-0.2, 1.45)
    plt.ylim(-1.25, 1.25)

    plt.title("Cosine Similarity as Angle Between Movie Vectors")
    plt.xlabel("Vector Direction")
    plt.ylabel("Vector Direction")

    plt.grid(alpha=0.2)

    # Bottom-right explanation (with reference)
    plt.text(
        1.35,
        -1.1,
        f"Reference movie:\n{reference_movie}\n\n"
        "Small angle → high similarity\n"
        "Large angle → low similarity\n\n"
        "Cosine similarity = cos(angle)",
        fontsize=11,
        ha="right",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()

    output_path = output_dir / "movie_similarity_cosine_final.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()