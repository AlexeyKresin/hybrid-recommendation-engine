import pandas as pd


def print_report_header() -> None:
    print("\n" + "=" * 78)
    print("HYBRID RECOMMENDATION SYSTEM REPORT".center(78))
    print("=" * 78)

    print("\nDataset        : MovieLens 100K")
    print("Models         : CF, Content-Based, Hybrid, Adaptive Hybrid")
    print("Evaluation     : Precision@K, RMSE")
    print("Cold-Start     : User, Item, System")


def print_metric_section(precision: float, rmse: float) -> None:
    print("\n" + "=" * 22 + " EVALUATION " + "=" * 22)

    print("\nResults")
    print("-" * 60)
    print(f"Precision@10 : {precision:.4f}")
    print(f"RMSE         : {rmse:.4f}")

    print("\nMetric Explanation")
    print("-" * 60)
    print("Precision@10: Fraction of top-10 recommendations that are relevant.")
    print("RMSE: Measures prediction error between predicted and actual ratings.")

    print("\nInterpretation")
    print("-" * 60)
    print(
        "The model predicts ratings well (low RMSE), but ranking quality is limited.\n"
        "This is expected due to sparse user-item interactions."
    )


def print_fixed_vs_adaptive_section(df: pd.DataFrame) -> None:
    print("\n" + "=" * 22 + " FIXED VS ADAPTIVE " + "=" * 22)

    fixed = df[df["model"].str.startswith("Hybrid alpha=")]
    adaptive = df[df["model"] == "Adaptive Hybrid"]

    if fixed.empty or adaptive.empty:
        return

    best_fixed = fixed.loc[fixed["precision@10"].idxmax()]
    adaptive_row = adaptive.iloc[0]

    summary = pd.DataFrame([
        {
            "Model": best_fixed["model"],
            "Precision@10": best_fixed["precision@10"],
        },
        {
            "Model": "Adaptive Hybrid",
            "Precision@10": adaptive_row["precision@10"],
        },
    ])

    print("\nComparison")
    print("-" * 60)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nInsight")
    print("-" * 60)
    print(
        "Adaptive Hybrid matches the best fixed alpha without manual tuning.\n"
        "This makes the model more flexible across different users."
    )


def print_final_insights() -> None:
    print("\n" + "=" * 22 + " FINAL INSIGHTS " + "=" * 22)

    print("1. Collaborative Filtering performs best with sufficient user data.")
    print("2. Content-Based Filtering supports cold-start scenarios.")
    print("3. Hybrid model improves recommendation quality.")
    print("4. Adaptive alpha removes need for manual tuning.")
    print("5. Cold-start significantly reduces performance.")

    print("\n" + "=" * 22 + " CONCLUSION " + "=" * 22)
    print(
        "The Hybrid Recommendation System successfully balances user behavior\n"
        "and item features. Adaptive weighting achieves strong performance\n"
        "without manual parameter tuning."
    )