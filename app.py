# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

# -------------------------
# Styling + small UX touches
# -------------------------
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0C0C0C; color: #FFFFFF; }
    h1,h2,h3 { color: #241773; }
    .stButton>button { background-color: #241773; color: white; border-radius: 8px; border: 1px solid #9E7C0C; }
    input { background-color: #1C1C1C !important; color: white !important; }
    .dataframe td, .dataframe th { color: white; }
    </style>
""", unsafe_allow_html=True)

def enable_enter_to_tab():
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === "Enter") {
                const inputs = Array.from(document.querySelectorAll('input, textarea, select'));
                const index = inputs.indexOf(document.activeElement);
                if (index > -1) {
                    e.preventDefault();
                    const next = inputs[index + 1];
                    if (next) next.focus();
                }
            }
        });
        </script>
    """, unsafe_allow_html=True)

enable_enter_to_tab()

# -------------------------
# Utility / Normalization
# -------------------------
def normalize_team_names(df, team_cols):
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df

def normalize_schedule_team_abbreviations(schedule_df):
    if 'season' in schedule_df.columns:
        mask = schedule_df['season'].fillna(0).astype(int) >= 2020
        schedule_df.loc[mask & (schedule_df.get('home_team') == 'OAK'), 'home_team'] = 'LV'
        schedule_df.loc[mask & (schedule_df.get('away_team') == 'OAK'), 'away_team'] = 'LV'
    return schedule_df

def normalize_spreads_team_abbreviations(spreads_df):
    if 'season' in spreads_df.columns:
        mask = spreads_df['season'].fillna(0).astype(int) < 2020
        spreads_df.loc[mask & (spreads_df.get('home_team') == 'LV'), 'home_team'] = 'OAK'
        spreads_df.loc[mask & (spreads_df.get('away_team') == 'LV'), 'away_team'] = 'OAK'
    return spreads_df

# -------------------------
# Load & normalize Kaggle spreads CSV
# -------------------------
def load_historical_spreads(spread_path: str = "nfl.kaggle.spreads.data.csv"):
    """
    Reads Kaggle CSV and returns a DataFrame with standardized column names:
      - season, week, home_team, away_team, spread, home_score, away_score
    If the file is not found at spread_path, returns None.
    """
    if not os.path.exists(spread_path):
        st.warning(f"Spreads CSV not found at '{spread_path}'. You can upload it below.")
        uploaded = st.file_uploader("Upload Kaggle spreads CSV (optional)", type=["csv"])
        if uploaded is None:
            return None
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(spread_path)

    # Show raw columns for debugging
    st.subheader("Kaggle spreads CSV - sample and schema")
    st.write("Columns detected:", list(df.columns))
    st.dataframe(df.head(5))
    st.write(df.dtypes)

    # Lowercase & strip headers for robust matching
    df.columns = df.columns.str.strip().str.lower()

    # Map Kaggle columns to our standardized names
    rename_map = {
        'schedule_season': 'season',
        'schedule_week'  : 'week',
        'team_home'      : 'home_team',
        'team_away'      : 'away_team',
        'spread_favorite': 'spread',
        'score_home'     : 'home_score',
        'score_away'     : 'away_score',
        'schedule_playoff': 'schedule_playoff',
        'schedule_date'  : 'gameday'
    }

    # Only rename those which exist
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=cols_to_rename)

    # Filter to regular season only if we have schedule_playoff present
    if 'schedule_playoff' in df.columns:
        try:
            df = df[df['schedule_playoff'] == False]
        except Exception:
            # if values not strictly boolean, try to coerce to boolean-ish
            df = df[~df['schedule_playoff'].astype(str).str.lower().isin(['true','1','t','y','yes'])]

    # Normalize teams & types
    df = normalize_team_names(df, ['home_team', 'away_team'])
    df = normalize_spreads_team_abbreviations(df)

    # Coerce numeric season/week
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce').astype('Int64')

    # Keep only columns we need if present
    desired = ['season', 'week', 'home_team', 'away_team', 'spread', 'home_score', 'away_score', 'gameday']
    existing = [c for c in desired if c in df.columns]
    df = df[existing].copy()

    return df

# -------------------------
# Merge historical schedules and spreads with robust diagnostics
# -------------------------
def merge_historical_data(schedules_df: pd.DataFrame, spreads_df: pd.DataFrame):
    # Normalize schedule
    schedules_df = normalize_team_names(schedules_df, ['home_team', 'away_team'])
    schedules_df = normalize_schedule_team_abbreviations(schedules_df)
    if 'season' in schedules_df.columns:
        schedules_df['season'] = pd.to_numeric(schedules_df['season'], errors='coerce').astype('Int64')
    if 'week' in schedules_df.columns:
        schedules_df['week'] = pd.to_numeric(schedules_df['week'], errors='coerce').astype('Int64')

    # Ensure spreads_df present and normalized
    if spreads_df is None:
        st.error("Spreads DataFrame is None (CSV not loaded). Cannot merge historical data.")
        return schedules_df.assign(spread=np.nan, home_score=np.nan, away_score=np.nan)

    spreads_df = normalize_team_names(spreads_df, ['home_team', 'away_team'])
    spreads_df = normalize_spreads_team_abbreviations(spreads_df)
    if 'season' in spreads_df.columns:
        spreads_df['season'] = pd.to_numeric(spreads_df['season'], errors='coerce').astype('Int64')
    if 'week' in spreads_df.columns:
        spreads_df['week'] = pd.to_numeric(spreads_df['week'], errors='coerce').astype('Int64')

    # Merge on canonical keys
    on_keys = ['season', 'week', 'home_team', 'away_team']
    merged = pd.merge(schedules_df, spreads_df, on=on_keys, how='left', suffixes=('', '_spread'))

    # Standard required columns in pipeline
    required_cols = ['spread', 'home_score', 'away_score']

    # If required columns are absent after merge, add them as NaN (defensive)
    missing_cols_post_merge = [c for c in required_cols if c not in merged.columns]
    if missing_cols_post_merge:
        st.warning(f"The following expected columns were not found in merged data and will be added as NaN: {missing_cols_post_merge}")
        for c in missing_cols_post_merge:
            merged[c] = np.nan

    # Now detect rows with any missing required data
    missing_rows_mask = merged[required_cols].isna().any(axis=1)
    missing_rows = merged[missing_rows_mask].copy()

    # Diagnostic display (detailed)
    if not missing_rows.empty:
        st.subheader("⚠️ Missing historical values (spread or scores)")
        display_cols = [col for col in ['season','week','home_team','away_team'] + required_cols if col in merged.columns]
        st.dataframe(missing_rows[display_cols].reset_index(drop=True))

        missing_counts = missing_rows[required_cols].isna().sum()
        total_games = len(merged)
        missing_fraction = len(missing_rows) / total_games if total_games > 0 else 0.0

        # Summary table
        summary_table = pd.DataFrame({
            'Total Games': [total_games],
            'Games Missing Spread': [int(missing_counts.get('spread', 0))],
            'Games Missing Home Score': [int(missing_counts.get('home_score', 0))],
            'Games Missing Away Score': [int(missing_counts.get('away_score', 0))],
            'Fraction Missing': [missing_fraction]
        })
        st.subheader("📊 Missing Data Summary")
        st.dataframe(summary_table)

        # Estimate reliability impact (simple heuristic)
        # If many training rows have no spread/scores, we will have fewer labeled games for training
        estimated_labeled_games = total_games - int(missing_counts.get('spread', 0)) - int(missing_counts.get('home_score', 0)) - int(missing_counts.get('away_score', 0))
        st.write(f"Estimated labeled games available for training (heuristic): {max(0, estimated_labeled_games)} / {total_games}")

    # Attempt swapped-team merge: sometimes spreads are recorded with home/away reversed
    if not missing_rows.empty and 'home_team' in spreads_df.columns and 'away_team' in spreads_df.columns:
        swapped = spreads_df.rename(columns={
            'home_team': 'away_team',
            'away_team': 'home_team',
            'home_score': 'away_score',
            'away_score': 'home_score'
        })
        swapped = swapped[[c for c in ['season','week','home_team','away_team'] + required_cols if c in swapped.columns]]

        # Merge swapped onto just the missing rows (dropping required cols to prevent conflicts)
        missing_keys = missing_rows[['season','week','home_team','away_team']].drop_duplicates()
        swapped_merge = pd.merge(
            missing_keys,
            swapped,
            on=['season','week','home_team','away_team'],
            how='left',
            suffixes=('','_swapped')
        )

        # For each required column, fill into merged where merged is na and swapped_merge has a value
        for col in required_cols:
            if col in swapped_merge.columns:
                # Build index alignment
                condition = merged[col].isna()
                # Create mapping key to values from swapped_merge
                key_cols = ['season','week','home_team','away_team']
                temp = swapped_merge.set_index(key_cols)[col]
                # align by index via merging keys temporarily
                merged_keys = merged[key_cols].astype(object)
                # produce series of candidate fills
                candidate = merged_keys.merge(temp.reset_index().rename(columns={col: f"cand_{col}"}),
                                             on=key_cols, how='left')[f"cand_{col}"]
                # Assign where merged[col] isna and candidate notna
                mask_assign = merged[col].isna() & candidate.notna()
                merged.loc[mask_assign, col] = candidate[mask_assign].values

    return merged

# -------------------------
# Feature engineering + labels
# -------------------------
def compute_rest_days(schedule_df):
    def days_since_last_game(df, team_col):
        last_game_dates = {}
        rest_days = []
        # sort by season, week and gameday if available
        sort_cols = ['season', 'week']
        if 'gameday' in df.columns:
            sort_cols = ['season', 'week', 'gameday']
        for idx, row in df.sort_values(sort_cols).iterrows():
            team = row.get(team_col)
            game_date = None
            if 'gameday' in row and pd.notna(row['gameday']):
                try:
                    game_date = pd.to_datetime(row['gameday'])
                except Exception:
                    game_date = None
            if team in last_game_dates and game_date is not None:
                delta = (game_date - last_game_dates[team]).days
            else:
                delta = np.nan
            rest_days.append(delta)
            if game_date is not None:
                last_game_dates[team] = game_date
        return rest_days

    df = schedule_df.copy()
    df = df.sort_values(['season','week'])
    df['home_rest'] = days_since_last_game(df, 'home_team')
    df['away_rest'] = days_since_last_game(df, 'away_team')
    df['rest_diff'] = df['home_rest'] - df['away_rest']
    return df[['season','week','home_team','away_team','rest_diff']]

def get_roster_injury_impact(years, max_week=None):
    # Placeholder: returns zeros; replace with real data source if available
    teams = sorted(['LV','BUF','CHI','DAL'])
    max_week = max_week or 17
    rows = []
    for y in years:
        for w in range(1, max_week + 1):
            for t in teams:
                rows.append({'team': t, 'season': y, 'week': w, 'injury_impact': 0})
    return pd.DataFrame(rows)

def get_yearly_stats_by_weeks(years, max_week=None):
    # Placeholder: synthetic stats for demonstration
    teams = sorted(['LV','BUF','CHI','DAL'])
    max_week = max_week or 17
    rows = []
    for y in years:
        for w in range(1, max_week + 1):
            for t in teams:
                rows.append({
                    'season': y,
                    'week': w,
                    'team': t,
                    'epa_offense': np.random.uniform(-1, 1),
                    'epa_defense': np.random.uniform(-1, 1),
                    'snap_counts': np.random.randint(50, 100)
                })
    return pd.DataFrame(rows)

def prepare_game_features(schedule_df, team_stats_df, rest_days_df, injury_df):
    df = schedule_df.copy()

    # Ensure canonical keys exist for merges
    for c in ['season','week','home_team','away_team']:
        if c not in df.columns:
            df[c] = np.nan

    if rest_days_df is not None and not rest_days_df.empty:
        df = df.merge(rest_days_df, on=['season','week','home_team','away_team'], how='left')
    else:
        df['rest_diff'] = 0

    if injury_df is not None and not injury_df.empty:
        home_inj = injury_df.rename(columns={'team': 'home_team', 'injury_impact': 'injury_home'})
        away_inj = injury_df.rename(columns={'team': 'away_team', 'injury_impact': 'injury_away'})
        df = df.merge(home_inj[['season','week','home_team','injury_home']], on=['season','week','home_team'], how='left')
        df = df.merge(away_inj[['season','week','away_team','injury_away']], on=['season','week','away_team'], how='left')
        df['injury_home'] = df['injury_home'].fillna(0)
        df['injury_away'] = df['injury_away'].fillna(0)
        df['injury_diff'] = df['injury_home'] - df['injury_away']
    else:
        df['injury_diff'] = 0

    # Synthetic feature placeholders (replace with real features as needed)
    np.random.seed(0)
    df['epa_off_diff'] = np.random.normal(0, 1, len(df))
    df['epa_def_diff'] = np.random.normal(0, 1, len(df))
    df['snap_diff_offense'] = np.random.randint(-10, 10, len(df))
    df['snap_diff_defense'] = np.random.randint(-10, 10, len(df))
    df['home_advantage'] = 1

    if 'spread' not in df.columns:
        df['spread'] = np.nan

    return df

def add_ats_cover_label(df):
    def row_cover(r):
        if pd.isna(r.get('spread')) or pd.isna(r.get('home_score')) or pd.isna(r.get('away_score')):
            return np.nan
        margin = r['home_score'] - r['away_score']
        # home covers if margin > spread (typical decimal spread)
        return int(margin > r['spread'])
    df['ats_cover'] = df.apply(row_cover, axis=1)
    return df

# -------------------------
# Model training + prediction
# -------------------------
def train_model(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['ats_cover'].astype(int).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_scaled, y)
    return model, scaler

def predict_week(model, scaler, df, feature_cols):
    X_pred = df[feature_cols].copy()
    X_scaled = scaler.transform(X_pred)
    df = df.copy()
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)
    df['ats_prob'] = probs
    df['predicted_cover'] = preds

    # Insert probabilities next to team names
    if 'home_team' in df.columns:
        home_idx = df.columns.get_loc('home_team')
        df.insert(home_idx + 1, 'home_cover_prob', df['ats_prob'])
    else:
        df['home_cover_prob'] = df['ats_prob']

    if 'away_team' in df.columns:
        # recalc index in case columns changed
        away_idx = df.columns.get_loc('away_team')
        df.insert(away_idx + 1, 'away_cover_prob', 1 - df['ats_prob'])
    else:
        df['away_cover_prob'] = 1 - df['ats_prob']

    # Ensure game_id exists for ordering; if not, create a surrogate
    if 'game_id' not in df.columns:
        df['game_id'] = df.apply(lambda r: f"{r.get('season','NA')}-{r.get('week','NA')}-{r.get('home_team','NA')}-vs-{r.get('away_team','NA')}", axis=1)

    desired_order = [c for c in ['game_id', 'home_team', 'home_cover_prob', 'away_team', 'away_cover_prob', 'spread'] if c in df.columns]
    other_cols = [c for c in df.columns if c not in desired_order]
    df = df[desired_order + other_cols]

    return df, df[df['predicted_cover'] == 1]

# -------------------------
# Streamlit app flow
# -------------------------
st.title("🏈 NFL ATS Prediction App — Regenerated with Robust CSV handling & Diagnostics")

season_default = datetime.now().year
season = st.number_input("Season (year)", min_value=1990, max_value=2100, value=season_default, step=1)
week = st.number_input("Week to predict", min_value=1, max_value=18, value=1, step=1)

# Buttons and session state
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = None

st.markdown("### 1) Load schedule for the target week")
if st.button("Load Schedule from nfl_data_py"):
    schedules = nfl.import_schedules([season])
    schedules = schedules[(schedules['season'] == int(season)) & (schedules['week'] == int(week)) & (schedules.get('game_type') == 'REG')]
    schedules = normalize_team_names(schedules, ['home_team', 'away_team'])
    schedules = normalize_schedule_team_abbreviations(schedules)
    # ensure gameday exists if present in schedule data
    if 'schedule_date' in schedules.columns:
        schedules = schedules.rename(columns={'schedule_date': 'gameday'})
    schedules['spread'] = np.nan
    st.session_state.schedule_df = schedules
    st.success("Schedule loaded into session state.")

if st.session_state.schedule_df is not None:
    st.subheader("Schedule (edit spreads below)")
    schedule_df = st.session_state.schedule_df.copy()
    # allow manual input of spreads for the loaded schedule
    spreads_in = []
    for idx, row in schedule_df.reset_index(drop=True).iterrows():
        label = f"{row.get('away_team','')} @ {row.get('home_team','')} (W{row.get('week','')})"
        default = "" if pd.isna(row.get('spread')) else str(row.get('spread'))
        s = st.text_input(label, value=default, key=f"spread_{idx}")
        try:
            val = float(s) if s.strip() != "" else np.nan
        except Exception:
            val = np.nan
        spreads_in.append(val)
    schedule_df['spread'] = spreads_in
    st.session_state.schedule_df = schedule_df

    st.markdown("---")
    st.markdown("### 2) Load historical spreads CSV (Kaggle)")
    spread_path_input = st.text_input("Path to Kaggle spreads CSV", value="nfl.kaggle.spreads.data.csv")
    load_csv_btn = st.button("Load CSV and prepare training data")

    if load_csv_btn:
        historical_spreads = load_historical_spreads(spread_path_input)
        if historical_spreads is None:
            st.error("No spreads CSV provided. Please upload or provide correct path.")
        else:
            st.success("Spreads CSV loaded and normalized.")

            # Load historical schedules for training (2019 .. season-1)
            years = list(range(2019, int(season)))
            if len(years) == 0:
                st.error("No historical years to train on. Choose a season > 2019.")
            else:
                st.info("Importing historical schedules (this may take a moment)...")
                historical_schedules = nfl.import_schedules(years)
                historical_schedules = historical_schedules[historical_schedules.get('game_type') == 'REG']
                historical_schedules = normalize_team_names(historical_schedules, ['home_team', 'away_team'])
                historical_schedules = normalize_schedule_team_abbreviations(historical_schedules)

                # Merge with robust diagnostics
                merged_train_data = merge_historical_data(historical_schedules, historical_spreads)

                # Show a quick sanity check
                st.subheader("Merged training data (sample)")
                st.dataframe(merged_train_data.head(10))

                # Drop incomplete game results necessary for label creation
                before_drop = len(merged_train_data)
                merged_train_data = merged_train_data.dropna(subset=['spread', 'home_score', 'away_score'])
                after_drop = len(merged_train_data)
                st.write(f"Dropped {before_drop - after_drop} incomplete rows; {after_drop} rows remain for training.")

                # Feature engineering and labels
                rest_days_train = compute_rest_days(merged_train_data)
                team_stats_train = get_yearly_stats_by_weeks(years)
                injury_train = get_roster_injury_impact(years)
                train_games = prepare_game_features(merged_train_data, team_stats_train, rest_days_train, injury_train)
                train_games = add_ats_cover_label(train_games)
                train_games = train_games.dropna(subset=['ats_cover'])
                st.write(f"Final training rows with labels: {len(train_games)}")

                if len(train_games) < 10:
                    st.warning("Very few training rows available. Model quality will be poor. Consider obtaining a more complete spreads CSV or widening historical years.")

                # Define features for model
                feature_cols = ['epa_off_diff','epa_def_diff','snap_diff_offense','snap_diff_defense','rest_diff','injury_diff','home_advantage','spread']
                # Ensure feature columns exist in train_games; if not, add placeholder zeros
                for c in feature_cols:
                    if c not in train_games.columns:
                        train_games[c] = 0

                # Train model
                try:
                    model, scaler = train_model(train_games, feature_cols)
                    st.success("Model trained.")
                except Exception as e:
                    st.error(f"Model training failed: {e}")
                    st.stop()

                # Prepare prediction data
                pred_schedule = st.session_state.schedule_df.copy()
                # ensure season/week columns exist for pred
                pred_schedule['season'] = pred_schedule.get('season', season)
                pred_schedule['week'] = pred_schedule.get('week', week)
                rest_days_pred = compute_rest_days(pred_schedule)
                team_stats_pred = get_yearly_stats_by_weeks([season])
                injury_pred = get_roster_injury_impact([season])
                pred_games = prepare_game_features(pred_schedule, team_stats_pred, rest_days_pred, injury_pred)

                # Ensure feature columns present
                for c in feature_cols:
                    if c not in pred_games.columns:
                        pred_games[c] = 0

                # Predict
                try:
                    predictions, upsets = predict_week(model, scaler, pred_games, feature_cols)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                st.subheader("Predictions for selected week")
                st.dataframe(predictions[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])

                if not upsets.empty:
                    st.subheader("Predicted Upset Covers (away covers)")
                    st.dataframe(upsets[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])

                # Save results
                out_file = f"nfl_ats_predictions_week_{int(week)}.xlsx"
                try:
                    predictions.to_excel(out_file, index=False)
                    st.success(f"Predictions saved to {out_file}")
                except Exception as e:
                    st.warning(f"Could not save to file: {e}")

# End of file
