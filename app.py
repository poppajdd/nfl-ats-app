import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# --- Custom Styling (Ravens Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #0C0C0C; color: #FFFFFF; }
    h1, h2, h3 { color: #241773; }
    .stButton>button {
        background-color: #241773; color: white; border-radius: 10px;
        border: 1px solid #9E7C0C;
    }
    .stButton>button:hover { background-color: #3B1C87; }
    input { background-color: #1C1C1C !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- Custom JS: Enter moves to next input ---
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

# --- Utility Functions ---
def normalize_team_names(df, team_cols):
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    return df

def normalize_schedule_team_abbreviations(schedule_df):
    mask = schedule_df['season'] >= 2020
    schedule_df.loc[mask & (schedule_df['home_team'] == 'OAK'), 'home_team'] = 'LV'
    schedule_df.loc[mask & (schedule_df['away_team'] == 'OAK'), 'away_team'] = 'LV'
    return schedule_df

def load_historical_spreads(spread_path="nfl.kaggle.spreads.data.csv"):
    try:
        spreads = pd.read_csv(spread_path)
    except FileNotFoundError:
        return pd.DataFrame()  # safe fallback on Streamlit Cloud

    spreads.columns = spreads.columns.str.strip().str.lower()
    rename_map = {
        'schedule_season': 'season',
        'schedule_week': 'week',
        'team_home': 'home_team',
        'team_away': 'away_team',
        'spread_favorite': 'spread',
        'score_home': 'home_score',
        'score_away': 'away_score'
    }
    spreads.rename(columns=rename_map, inplace=True)
    spreads = spreads[spreads.get('schedule_playoff', False) == False]  # keep regular season
    spreads = normalize_team_names(spreads, ['home_team', 'away_team'])
    return spreads[list(rename_map.values())]

def compute_rest_days(schedule_df):
    def days_since_last_game(df, team_col):
        last_game_dates, rest_days = {}, []
        for _, row in df.sort_values(['season', 'week']).iterrows():
            team = row[team_col]
            try: game_date = pd.to_datetime(row['gameday'])
            except: game_date = None
            if team in last_game_dates and game_date:
                delta = (game_date - last_game_dates[team]).days
            else:
                delta = np.nan
            rest_days.append(delta)
            if game_date:
                last_game_dates[team] = game_date
        return rest_days

    schedule_df = schedule_df.sort_values(['season', 'week'])
    schedule_df['home_rest'] = days_since_last_game(schedule_df, 'home_team')
    schedule_df['away_rest'] = days_since_last_game(schedule_df, 'away_team')
    schedule_df['rest_diff'] = schedule_df['home_rest'] - schedule_df['away_rest']
    return schedule_df[['season', 'week', 'home_team', 'away_team', 'rest_diff']]

def get_yearly_stats_by_weeks(years, max_week=18):
    teams = nfl.import_team_desc()['team_abbr'].unique().tolist()
    data = []
    for year in years:
        for week in range(1, max_week + 1):
            for team in teams:
                data.append({
                    'season': year, 'week': week, 'team': team,
                    'epa_offense': np.random.uniform(-1, 1),
                    'epa_defense': np.random.uniform(-1, 1),
                    'snap_counts': np.random.randint(50, 100)
                })
    return pd.DataFrame(data)

def prepare_game_features(schedule_df, team_stats_df, rest_days_df):
    df = schedule_df.copy()
    if not rest_days_df.empty:
        df = df.merge(rest_days_df, on=['season','week','home_team','away_team'], how='left')
    else:
        df['rest_diff'] = 0

    np.random.seed(0)
    df['epa_off_diff'] = np.random.normal(0, 1, len(df))
    df['epa_def_diff'] = np.random.normal(0, 1, len(df))
    df['snap_diff_offense'] = np.random.randint(-10, 10, len(df))
    df['snap_diff_defense'] = np.random.randint(-10, 10, len(df))
    df['home_advantage'] = 1

    if 'spread' not in df.columns:
        df['spread'] = 0

    return df

def train_model(df, feature_cols):
    X = df[feature_cols]
    y = (df['home_score'] - df['away_score'] > df['spread']).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler

def predict_week(model, scaler, df, feature_cols):
    X_pred = scaler.transform(df[feature_cols])
    df['ats_prob'] = model.predict_proba(X_pred)[:, 1]
    df['predicted_cover'] = model.predict(X_pred)
    df['home_cover_prob'] = df['ats_prob']
    df['away_cover_prob'] = 1 - df['ats_prob']
    return df

def get_week_schedule(season, week):
    schedules = nfl.import_schedules([season])
    schedules = schedules[(schedules['season']==season) &
                          (schedules['week']==week) &
                          (schedules['game_type']=='REG')]
    schedules = normalize_team_names(schedules, ['home_team','away_team'])
    schedules = normalize_schedule_team_abbreviations(schedules)
    schedules['spread'] = np.nan
    return schedules

# --- Streamlit UI ---
st.title("🏈 NFL ATS Prediction App")

season = datetime.now().year
week = st.number_input("Enter NFL week:", min_value=1, max_value=18, step=1)

if st.button("Load Schedule"):
    schedule_df = get_week_schedule(season, week)

    st.subheader("Enter Point Spreads")
    spreads = []
    for _, row in schedule_df.iterrows():
        spread_input = st.text_input(f"{row['away_team']} @ {row['home_team']} (Week {row['week']})", "")
        try:
            spreads.append(float(spread_input) if spread_input.strip() else None)
        except ValueError:
            spreads.append(None)
    schedule_df['spread'] = spreads

    if st.button("Run Predictions"):
        # Train model with historical data
        historical_schedules = nfl.import_schedules(list(range(2019, season)))
        historical_schedules = historical_schedules[historical_schedules['game_type']=="REG"]

        spreads_df = load_historical_spreads()
        if spreads_df.empty:
            st.error("No historical spread data found.")
        else:
            train_df = historical_schedules.merge(
                spreads_df, on=['season','week','home_team','away_team'], how='inner'
            ).dropna(subset=['spread','home_score','away_score'])

            rest_days_train = compute_rest_days(train_df)
            team_stats_train = get_yearly_stats_by_weeks(list(range(2019, season)))
            train_games = prepare_game_features(train_df, team_stats_train, rest_days_train)

            feature_cols = ['epa_off_diff','epa_def_diff','snap_diff_offense',
                            'snap_diff_defense','rest_diff','home_advantage','spread']

            model, scaler = train_model(train_games, feature_cols)

            # Prediction set
            rest_days_pred = compute_rest_days(schedule_df)
            team_stats_pred = get_yearly_stats_by_weeks([season])
            pred_games = prepare_game_features(schedule_df, team_stats_pred, rest_days_pred)
            predictions = predict_week(model, scaler, pred_games, feature_cols)

            st.subheader("Predictions")
            st.dataframe(predictions[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])

            # Excel download
            file_name = f"nfl_ats_predictions_week_{week}.xlsx"
            predictions.to_excel(file_name, index=False)
            st.download_button("Download Excel", data=open(file_name,"rb"), file_name=file_name)
