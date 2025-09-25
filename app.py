import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Team name normalization map
# -------------------------------------------------------------------
TEAM_NAME_MAP = {
    "OAK": "LV",     # Oakland Raiders -> Las Vegas Raiders
    "SD": "LAC",     # San Diego Chargers -> Los Angeles Chargers
    "STL": "LA",     # St. Louis Rams -> Los Angeles Rams
    "WSH": "WAS",    # Washington generic -> Washington Football/Commanders
    "WFT": "WAS",    # Washington Football Team -> Washington
    "LA": "LAR",     # Kaggle sometimes uses "LA" -> normalize to "LAR"
}

def normalize_team_names(df, cols):
    """Apply TEAM_NAME_MAP to given columns in DataFrame."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df

# -------------------------------------------------------------------
# Load historical data
# -------------------------------------------------------------------
def load_historical_schedules(schedule_path="nfl.historical.schedules.csv"):
    schedules = pd.read_csv(schedule_path)

    # Normalize names
    schedules = normalize_team_names(schedules, ["home_team", "away_team"])

    return schedules


def load_historical_spreads(spread_path="nfl.kaggle.spreads.data.csv"):
    spreads = pd.read_csv(spread_path)

    # Rename columns to match schedule expectations
    spreads = spreads.rename(
        columns={
            "score_home": "home_score",
            "score_away": "away_score",
            "spread_favorite": "spread"
        }
    )

    # Normalize names
    spreads = normalize_team_names(
        spreads, ["team_home", "team_away", "team_favorite_id"]
    )

    return spreads

# -------------------------------------------------------------------
# Merge & diagnostics
# -------------------------------------------------------------------
def merge_historical_data(schedules, spreads):
    """Merge schedules and spreads with safety checks and diagnostics."""

    merged = pd.merge(
        schedules,
        spreads,
        left_on=["season", "week", "home_team", "away_team"],
        right_on=["schedule_season", "schedule_week", "team_home", "team_away"],
        how="left",
        suffixes=("_schedule", "_spread"),
    )

    # Track missing critical fields
    critical_cols = ["spread", "home_score", "away_score"]
    present_cols = [c for c in critical_cols if c in merged.columns]

    missing_rows = merged[merged[present_cols].isna().any(axis=1)]

    if not missing_rows.empty:
        logging.warning(
            f"⚠️ Found {len(missing_rows)} rows with missing critical fields."
        )
        logging.info(
            "Examples of missing rows:\n%s", missing_rows.head().to_string()
        )

    return merged, missing_rows


def summarize_missing_data(missing_rows, total_rows):
    """Summarize missing vs available data for reliability check."""
    if total_rows == 0:
        return pd.DataFrame()

    summary = {
        "Total Games": total_rows,
        "Games with Missing Data": len(missing_rows),
        "Percent Missing": round(100 * len(missing_rows) / total_rows, 2),
    }
    return pd.DataFrame([summary])

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
def create_features(df):
    df = df.copy()

    # Drop games with missing spread or scores
    df = df.dropna(subset=["spread", "home_score", "away_score"])

    df["point_diff"] = df["home_score"] - df["away_score"]
    df["cover"] = np.where(df["point_diff"] + df["spread"] > 0, 1, 0)

    X = df[["spread"]]
    y = df["cover"]

    return X, y

# -------------------------------------------------------------------
# Train model
# -------------------------------------------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logging.info(f"Model trained. Test Accuracy: {acc:.3f}")
    return model

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    schedules = load_historical_schedules()
    spreads = load_historical_spreads()

    merged, missing_rows = merge_historical_data(schedules, spreads)

    summary_table = summarize_missing_data(missing_rows, len(merged))
    print("\n=== Missing Data Summary ===")
    print(summary_table.to_string(index=False))

    X, y = create_features(merged)
    if len(X) > 0 and len(y.unique()) > 1:
        model = train_model(X, y)
    else:
        logging.error("Not enough data to train model.")
