# app.py
import os
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# -------------------------
# Basic page config + styling
# -------------------------
st.set_page_config(page_title="NFL ATS Predictor", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #0C0C0C; color: #FFFFFF; }
h1,h2,h3 { color: #241773; }
.dataframe td, .dataframe th { color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Team acronym normalization map (hard-coded)
# -------------------------
TEAM_NAME_MAP = {
    "OAK": "LV",     # Oakland -> Las Vegas
    "SD": "LAC",     # San Diego Chargers -> LA Chargers
    "SDG": "LAC",
    "STL": "LAR",    # St. Louis Rams -> LA Rams (we use LAR)
    "WSH": "WAS",    # Washington variants -> WAS
    "WFT": "WAS",
    "LA": "LAR",     # ambiguous LA -> LAR (Rams) usually in some datasets
    # Add more if you encounter other legacy codes
}

def apply_team_map(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper().replace(TEAM_NAME_MAP)
    return df

# -------------------------
# Load Kaggle CSV (single-file source)
# -------------------------
def load_kaggle_spreads(path="nfl.kaggle.spreads.data.csv"):
    """
    Read the single Kaggle CSV and standardize column names.
    Returns DataFrame or None if no file and no upload.
    """
    df = None
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        st.warning(f"File not found at '{path}'. You may upload the Kaggle CSV below.")
        uploaded = st.file_uploader("Upload Kaggle spreads CSV (single file with schedule+spreads)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            return None

    # Normalize header names (strip, lower)
    df.columns = df.columns.str.strip().str.lower()

    # Show raw columns & a preview for debugging
    st.subheader("Kaggle CSV: Columns detected and preview")
    st.write(list(df.columns))
    st.dataframe(df.head(5))

    # Standardize column names we expect (only rename if present)
    rename_map = {
        'schedule_date'     : 'gameday',
        'schedule_season'   : 'season',
        'schedule_week'     : 'week',
        'team_home'         : 'home_team',
        'team_away'         : 'away_team',
        'score_home'        : 'home_score',
        'score_away'        : 'away_score',
        'spread_favorite'   : 'spread',
        'team_favorite_id'  : 'favorite_team_id',
        'schedule_playoff'  : 'schedule_playoff',
        'over_under_line'   : 'over_under'
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Coerce season/week numeric (defensive)
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce').astype('Int64')

    # Apply team name normalization (Kaggle may have OAK etc.)
    df = apply_team_map(df, ['home_team', 'away_team', 'favorite_team_id'])

    # Ensure we have columns we need; keep relevant subset if present
    keep_cols = [c for c in ['gameday','season','week','home_team','away_team','spread','home_score','away_score','schedule_playoff'] if c in df.columns]
    df = df[keep_cols].copy()

    return df

# -------------------------
# Build derived schedule and spreads views (both from same Kaggle DF)
# -------------------------
def extract_schedule_and_spreads_from_kaggle(kaggle_df):
    """
    From the single Kaggle dataframe, build:
      - schedules_df: rows with season/week/home_team/away_team/gameday
      - spreads_df: rows with season/week/home_team/away_team/spread/home_score/away_score
    """
    if kaggle_df is None:
        return None, None

    # schedules: dedupe by season/week/home/away/gameday
    sched_cols = [c for c in ['season','week','home_team','away_team','gameday'] if c in kaggle_df.columns]
    schedules_df = kaggle_df[sched_cols].drop_duplicates().reset_index(drop=True)

    # spreads: select spread/result columns
    spreads_cols = [c for c in ['season','week','home_team','away_team','spread','home_score','away_score'] if c in kaggle_df.columns]
    spreads_df = kaggle_df[spreads_cols].drop_duplicates().reset_index(drop=True)

    # Defensive: Ensure required columns exist in both; if missing, add NaN
    for df in (schedules_df, spreads_df):
        for c in ['season','week','home_team','away_team']:
            if c not in df.columns:
                df[c] = np.nan

    # Normalize types
    schedules_df['season'] = pd.to_numeric(schedules_df['season'], errors='coerce').astype('Int64')
    schedules_df['week'] = pd.to_numeric(schedules_df['week'], errors='coerce').astype('Int64')
    spreads_df['season'] = pd.to_numeric(spreads_df['season'], errors='coerce').astype('Int64')
    spreads_df['week'] = pd.to_numeric(spreads_df['week'], errors='coerce').astype('Int64')

    return schedules_df, spreads_df

# -------------------------
# Merge with diagnostics + swapped-team attempt
# -------------------------
def merge_historical_data(schedules_df, spreads_df):
    """
    Merge schedules_df with spreads_df on season/week/home_team/away_team.
    Show diagnostics of missing spreads/scores and attempt swapped-team fill.
    """
    # Defensive: if spreads_df is None, create empty with required cols
    if spreads_df is None:
        spreads_df = pd.DataFrame(columns=['season','week','home_team','away_team','spread','home_score','away_score'])

    merged = pd.merge(schedules_df, spreads_df,
                      on=['season','week','home_team','away_team'],
                      how='left',
                      suffixes=('','_sp'))

    # Make sure standardized columns exist
    for c in ['spread','home_score','away_score']:
        if c not in merged.columns:
            merged[c] = np.nan

    # Identify rows missing any critical element (spread or scores)
    missing_mask = merged[['spread','home_score','away_score']].isna().any(axis=1)
    missing_rows = merged[missing_mask].copy()

    # Diagnostic summary table
    total_games = len(merged)
    missing_counts = missing_rows[['spread','home_score','away_score']].isna().sum().to_dict()
    missing_fraction = len(missing_rows) / total_games if total_games > 0 else 0.0

    st.subheader("📋 Missing Data Diagnostics (historical)")
    st.write(f"Total historical games (from Kaggle): **{total_games}**")
    diag_df = pd.DataFrame([{
        'Games Missing Spread': int(missing_counts.get('spread', 0)),
        'Games Missing Home Score': int(missing_counts.get('home_score', 0)),
        'Games Missing Away Score': int(missing_counts.get('away_score', 0)),
        'Fraction Missing': f"{missing_fraction:.2%}"
    }])
    st.dataframe(diag_df)

    if not missing_rows.empty:
        # Show first 30 missing examples with key columns
        display_cols = [c for c in ['season','week','home_team','away_team','spread','home_score','away_score'] if c in merged.columns]
        st.subheader("Examples of games with missing spread/scores")
        st.dataframe(missing_rows[display_cols].head(50))

        # Attempt swapped-team merge: maybe spread recorded with reversed home/away
        swapped = spreads_df.rename(columns={
            'home_team': 'away_team',
            'away_team': 'home_team',
            'home_score': 'away_score',
            'away_score': 'home_score'
        })
        # keep relevant cols
        keep_cols = [c for c in ['season','week','home_team','away_team','spread','home_score','away_score'] if c in swapped.columns]
        swapped = swapped[keep_cols]

        # Merge swapped onto missing rows
        keys = ['season','week','home_team','away_team']
        missing_keys = missing_rows[keys].drop_duplicates()
        swapped_merge = pd.merge(missing_keys, swapped, on=keys, how='left', suffixes=('','_swapped'))

        # Fill values where merged has NA
        for col in ['spread','home_score','away_score']:
            if col in swapped_merge.columns:
                # create mapping by keys
                map_df = swapped_merge.set_index(keys)[col]
                # use apply row-wise fill
                def fill_row(r):
                    if pd.isna(r[col]):
                        key = (r['season'], r['week'], r['home_team'], r['away_team'])
                        return map_df.get(key, np.nan) if key in map_df.index else np.nan
                    return r[col]
                # Efficient approach: build a Series of candidate fills aligned to merged
                merged_keys = merged[keys].astype(object)
                candidate = merged_keys.merge(map_df.reset_index().rename(columns={col: f"cand_{col}"}),
                                              on=keys, how='left')[f"cand_{col}"]
                mask_assign = merged[col].isna() & candidate.notna()
                merged.loc[mask_assign, col] = candidate[mask_assign].values

        # Recompute missing counts after swap attempt
        missing_mask2 = merged[['spread','home_score','away_score']].isna().any(axis=1)
        missing_rows_after = merged[missing_mask2].copy()
        st.write(f"After attempting swapped-team fill, still missing: {len(missing_rows_after)} games.")
        if not missing_rows_after.empty:
            st.dataframe(missing_rows_after[display_cols].head(50))

    return merged

# -------------------------
# Feature engineering, labels, training, prediction
# -------------------------
def compute_rest_days(schedule_df):
    def days_since_last_game(df, team_col):
        last_game = {}
        rest_days = []
        for idx, row in df.sort_values(['season','week']).iterrows():
            team = row.get(team_col)
            game_date = None
            if 'gameday' in row and pd.notna(row['gameday']):
                try:
                    game_date = pd.to_datetime(row['gameday'])
                except Exception:
                    game_date = None
            if team in last_game and game_date is not None:
                rest = (game_date - last_game[team]).days
            else:
                rest = np.nan
            rest_days.append(rest)
            if game_date is not None:
                last_game[team] = game_date
        return rest_days

    sdf = schedule_df.copy()
    sdf = sdf.sort_values(['season','week'])
    sdf['home_rest'] = days_since_last_game(sdf, 'home_team')
    sdf['away_rest'] = days_since_last_game(sdf, 'away_team')
    sdf['rest_diff'] = sdf['home_rest'] - sdf['away_rest']
    return sdf[['season','week','home_team','away_team','rest_diff']]

def get_roster_injury_impact(years, max_week=17):
    # placeholder zeros - replace with real API later
    teams = sorted(['LV','BUF','CHI','DAL'])
    rows = []
    for y in years:
        for w in range(1, max_week+1):
            for t in teams:
                rows.append({'team': t, 'season': y, 'week': w, 'injury_impact': 0})
    return pd.DataFrame(rows)

def get_yearly_stats_by_weeks(years, max_week=17):
    # placeholder synthetic stats
    teams = sorted(['LV','BUF','CHI','DAL'])
    rows = []
    for y in years:
        for w in range(1, max_week+1):
            for t in teams:
                rows.append({
                    'season': y,
                    'week': w,
                    'team': t,
                    'epa_offense': np.random.uniform(-1,1),
                    'epa_defense': np.random.uniform(-1,1),
                    'snap_counts': np.random.randint(50,100)
                })
    return pd.DataFrame(rows)

def prepare_game_features(schedule_df, team_stats_df, rest_days_df, injury_df):
    df = schedule_df.copy()
    # merge rest days if available
    if rest_days_df is not None and not rest_days_df.empty:
        df = df.merge(rest_days_df, on=['season','week','home_team','away_team'], how='left')
    else:
        df['rest_diff'] = 0

    # injuries
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

    # synthetic features
    np.random.seed(0)
    df['epa_off_diff'] = np.random.normal(0,1,len(df))
    df['epa_def_diff'] = np.random.normal(0,1,len(df))
    df['snap_diff_offense'] = np.random.randint(-10,10,len(df))
    df['snap_diff_defense'] = np.random.randint(-10,10,len(df))
    df['home_advantage'] = 1

    if 'spread' not in df.columns:
        df['spread'] = np.nan

    return df

def add_ats_cover_label(df):
    def cover_row(r):
        if pd.isna(r.get('spread')) or pd.isna(r.get('home_score')) or pd.isna(r.get('away_score')):
            return np.nan
        margin = r['home_score'] - r['away_score']
        return int(margin > r['spread'])
    df['ats_cover'] = df.apply(cover_row, axis=1)
    return df

def train_model(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['ats_cover'].astype(int).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_scaled, y)
    return model, scaler

def predict_week(model, scaler, df, feature_cols):
    Xp = df[feature_cols].copy()
    Xp_scaled = scaler.transform(Xp)
    probs = model.predict_proba(Xp_scaled)[:,1]
    preds = model.predict(Xp_scaled)
    out = df.copy()
    out['ats_prob'] = probs
    out['predicted_cover'] = preds
    # insert probabilities next to teams
    if 'home_team' in out.columns:
        hi = out.columns.get_loc('home_team')
        out.insert(hi+1, 'home_cover_prob', out['ats_prob'])
    else:
        out['home_cover_prob'] = out['ats_prob']
    if 'away_team' in out.columns:
        ai = out.columns.get_loc('away_team')
        out.insert(ai+1, 'away_cover_prob', 1 - out['ats_prob'])
    else:
        out['away_cover_prob'] = 1 - out['ats_prob']
    # game_id
    if 'game_id' not in out.columns:
        out['game_id'] = out.apply(lambda r: f"{r.get('season','NA')}-{r.get('week','NA')}-{r.get('home_team','NA')}-vs-{r.get('away_team','NA')}", axis=1)
    desired = [c for c in ['game_id','home_team','home_cover_prob','away_team','away_cover_prob','spread'] if c in out.columns]
    other = [c for c in out.columns if c not in desired]
    out = out[desired + other]
    return out, out[out['predicted_cover'] == 1]

# -------------------------
# Streamlit App UI Flow
# -------------------------
st.title("🏈 NFL ATS Predictor (Kaggle single-file source)")

season_default = datetime.now().year
season = st.number_input("Season (target)", min_value=1990, max_value=2100, value=season_default, step=1)
week = st.number_input("Week (target)", min_value=1, max_value=18, value=1, step=1)

# Session state schedule
if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = None

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1) Load schedule for target week (nfl_data_py)")
    if st.button("Load schedule from nfl_data_py"):
        schedules = nfl.import_schedules([int(season)])
        schedules = schedules[(schedules['season'] == int(season)) & (schedules['week'] == int(week)) & (schedules['game_type'] == 'REG')]
        # rename schedule_date if present
        if 'schedule_date' in schedules.columns:
            schedules = schedules.rename(columns={'schedule_date':'gameday'})
        schedules = apply_team_map(schedules, ['home_team','away_team'])
        # ensure spread column exists for manual entry
        schedules['spread'] = np.nan
        st.session_state.schedule_df = schedules
        st.success("Schedule loaded into session state.")

    if st.session_state.schedule_df is not None:
        st.subheader("Schedule (enter spreads below)")
        schedule_df = st.session_state.schedule_df.copy().reset_index(drop=True)
        user_spreads = []
        for idx, row in schedule_df.iterrows():
            label = f"{row.get('away_team','')} @ {row.get('home_team','')} (W{row.get('week','')})"
            default = "" if pd.isna(row.get('spread')) else str(row.get('spread'))
            val = st.text_input(label, value=default, key=f"spread_{idx}")
            try:
                v = float(val) if val.strip() != "" else np.nan
            except Exception:
                v = np.nan
            user_spreads.append(v)
        schedule_df['spread'] = user_spreads
        st.session_state.schedule_df = schedule_df

with col2:
    st.subheader("2) Load Kaggle CSV (single file)")
    kaggle_path = st.text_input("Kaggle CSV path", value="nfl.kaggle.spreads.data.csv")
    if st.button("Load Kaggle CSV"):
        kaggle_df = load_kaggle_spreads(kaggle_path)
        if kaggle_df is not None:
            st.session_state.kaggle_df = kaggle_df
            st.success("Kaggle CSV loaded and normalized.")

# If kaggle loaded, prepare training and show diagnostics
if "kaggle_df" in st.session_state:
    kag_df = st.session_state.kaggle_df
    schedules_df, spreads_df = extract_schedule_and_spreads_from_kaggle(kag_df)
    merged_train = merge_historical_data(schedules_df, spreads_df)
    st.subheader("Merged training sample")
    st.dataframe(merged_train.head(10))

    # Drop rows with no spread or no scores for label creation
    before = len(merged_train)
    merged_train = merged_train.dropna(subset=['spread','home_score','away_score'])
    after = len(merged_train)
    st.write(f"Dropped {before-after} incomplete historical rows. {after} remain for training.")

    # Feature engineering + training
    rest_train = compute_rest_days(merged_train)
    team_stats_train = get_yearly_stats_by_weeks(list(range(2019, int(season))))
    injury_train = get_roster_injury_impact(list(range(2019, int(season))))
    train_games = prepare_game_features(merged_train, team_stats_train, rest_train, injury_train)
    train_games = add_ats_cover_label(train_games)
    train_games = train_games.dropna(subset=['ats_cover'])
    st.write(f"Training rows with labels: {len(train_games)}")

    if len(train_games) < 10:
        st.warning("Not enough labeled training rows. Model quality will be poor. Consider using a more complete Kaggle CSV or expanding years.")

    feature_cols = ['epa_off_diff','epa_def_diff','snap_diff_offense','snap_diff_defense','rest_diff','injury_diff','home_advantage','spread']
    for c in feature_cols:
        if c not in train_games.columns:
            train_games[c] = 0

    try:
        model, scaler = train_model(train_games, feature_cols)
        st.success("Model trained on historical data.")
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # Prepare prediction set from session schedule (or if none, build from nfl_data_py for target week)
    if st.session_state.schedule_df is None:
        st.warning("No schedule loaded for prediction. Click 'Load schedule from nfl_data_py' to fetch the target week schedule.")
    else:
        pred_schedule = st.session_state.schedule_df.copy()
        # ensure canonical keys exist
        if 'season' not in pred_schedule.columns:
            pred_schedule['season'] = season
        if 'week' not in pred_schedule.columns:
            pred_schedule['week'] = week

        rest_pred = compute_rest_days(pred_schedule)
        team_stats_pred = get_yearly_stats_by_weeks([int(season)])
        injury_pred = get_roster_injury_impact([int(season)])
        pred_games = prepare_game_features(pred_schedule, team_stats_pred, rest_pred, injury_pred)

        for c in feature_cols:
            if c not in pred_games.columns:
                pred_games[c] = 0

        try:
            predictions, upsets = predict_week(model, scaler, pred_games, feature_cols)
            st.subheader("Predictions for selected week")
            st.dataframe(predictions[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])
            if not upsets.empty:
                st.subheader("Upset Covers Predicted")
                st.dataframe(upsets[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])
            out_file = f"nfl_ats_predictions_week_{int(week)}.xlsx"
            predictions.to_excel(out_file, index=False)
            st.success(f"Saved predictions to {out_file}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Load a Kaggle CSV to proceed. Use the upload box if the file isn't present at the default path.")

# End
