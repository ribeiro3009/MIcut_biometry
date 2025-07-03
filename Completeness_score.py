import polars as pl
import os
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
INPUT_CSV = os.path.join("output", "full_analysis.csv")
OUTPUT_CSV = os.path.join("output", "completeness_scores.csv")

# Define which columns are features and their weights.
# Higher weight means more importance. We can start with equal weights.
# We assume higher values are better. If lower is better (e.g., noise), we'll invert the score.
FEATURE_WEIGHTS = {
    "NFIQ2": 1.0,
    "minutiae_count": 1.0,
    "singularities_count": 1.0,
    "cluster_count": -1.0,  # Lower is better, so we give it a negative weight to invert it
    "solidity": 1.0,
    "coverage": 1.0,
    "sharpness": 1.0,
    "orientation_std": -1.0, # Lower is better
    "contrast": 1.0,
    "bg_noise": -1.0, # Lower is better
    "ridge_frequency": 1.0
}

def calculate_completeness_score():
    """
    Reads the consolidated feature data, normalizes the features, and calculates
    a weighted completeness score for each fingerprint.
    """
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at {INPUT_CSV}")
        print("Please run the main pipeline first.")
        return

    print(f"Reading data from {INPUT_CSV}...")
    df = pl.read_csv(INPUT_CSV)

    feature_cols = list(FEATURE_WEIGHTS.keys())
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input CSV: {missing_cols}")
        return

    # --- Feature Normalization ---
    print("Normalizing features...")
    scaler = MinMaxScaler()
    
    # Polars requires a 2D array for the scaler, so we convert and then put it back
    features_df = df.select(feature_cols)
    normalized_features = scaler.fit_transform(features_df.to_numpy())
    
    # Create a new DataFrame with normalized features
    normalized_df = pl.DataFrame(normalized_features, schema=feature_cols)

    # --- Score Calculation ---
    print("Calculating completeness score...")
    score_expr = pl.lit(0.0)
    total_weight = 0.0

    for col, weight in FEATURE_WEIGHTS.items():
        # If weight is negative, it means lower values are better.
        # We invert the normalized score (1 - score) for these features.
        if weight < 0:
            score_expr += (1 - pl.col(col)) * abs(weight)
        else:
            score_expr += pl.col(col) * weight
        total_weight += abs(weight)

    # Calculate the weighted average
    normalized_df = normalized_df.with_columns(
        ((score_expr / total_weight) * 100).alias("completeness_score")
    )

    # --- Final DataFrame ---
    # Select original filename and the new score
    final_df = pl.concat([
        df.select("filename"),
        normalized_df.select("completeness_score")
    ], how="horizontal")

    final_df.write_csv(OUTPUT_CSV)
    print(f"\nSuccessfully calculated scores. Results saved to {OUTPUT_CSV}")

def main():
    calculate_completeness_score()

if __name__ == "__main__":
    main()
