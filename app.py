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
    .stButton>button { background-color: #241773; color: white; border-radius: 10px; border: 1px solid #9E7C0C; }
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

def normalize_spreads_team_abbreviations(spreads_df):
    mask = spreads_df['season'] < 2020
    spreads_df.loc[mask & (spreads_df['home_team'] == 'LV'), 'home_team'] = 'OAK'
    spreads_df.loc[mask & (spreads_df['away_team'] == 'LV'), 'away_team'] = 'OAK'
    return spreads_df

def load_historical_spreads(spread_path):
    spreads = pd.read_csv(spread_path)
    spreads.columns = spreads.columns.str.strip().str.lower()
    rename_map = {
        'schedule_season': 'season',
        'schedule_week': 'week',
        'team_home': 'home_team',
        'team_away': 'away_team',
        'spread_favorite': 'spread',
        'score_home': 'home_score',
        'score_away': 'away_score',
        'schedule_playoff': 'schedule_playoff'
    }
    spreads.rename(columns=rename_map, inplace=True)
    spreads = spreads[spreads['schedule_playoff'] == False]
    spreads = normalize_team_names(spreads, ['home_team', 'away_team'])
    spreads['season'] = pd.to_numeric(spreads['season'], errors='coerce').astype('Int64')
    spreads['week'] = pd.to_numeric(spreads['week'], errors='coerce').astype('Int64')
    spreads = normalize_spreads_team_abbreviations(spreads)
    spreads = spreads[list(rename_map.values())]
    return spreads

def merge_historical_data(schedules_df, spreads_df):
    schedules_df = normalize_team_names(schedules_df, ['home_team', 'away_team'])
    schedules_df = normalize_schedule_team_abbreviations(schedules_df)
    schedules_df['season'] = pd.to_numeric(schedules_df['season'], errors='coerce').astype('Int64')
    schedules_df['week'] = pd.to_numeric(schedules_df['week'], errors='coerce').astype('Int64')

    spreads_df = normalize_team_names(spreads_df, ['home_team', 'away_team'])
    spreads_df['season'] = pd.to_numeric(spreads_df['season'], errors='coerce').astype('Int64')
    spreads_df['week'] = pd.to_numeric(spreads_df['week'], errors='coerce').astype('Int64')

    merged = pd.merge(schedules_df, spreads_df,
                      on=['season', 'week', 'home_team', 'away_team'],
                      how='left')

    # --- Missing data diagnostics ---
    missing_rows = merged[merged[['spread', 'home_score', 'away_score']].isna().any(axis=1)]
    if not missing_rows.empty:
        st.subheader("⚠️ Missing Historical Data Detected")
        st.write("Some games are missing spreads or scores:")
        st.dataframe(missing_rows[['season','week','home_team','away_team','spread','home_score','away_score']])
        missing_counts = missing_rows[['spread','home_score','away_score']].isna().sum()
        st.write("Count of missing values:", missing_counts)
        missing_fraction = len(missing_rows)/len(merged)
        st.write(f"Fraction of games with missing data: {missing_fraction:.2%}")

        # --- Summary table for reliability ---
        summary_table = pd.DataFrame({
            'Total Games': [len(merged)],
            'Games Missing Spread': [missing_counts['spread']],
            'Games Missing Home Score': [missing_counts['home_score']],
            'Games Missing Away Score': [missing_counts['away_score']],
            'Fraction Missing': [missing_fraction]
        })
        st.subheader("📊 Missing Data Summary Table")
        st.dataframe(summary_table)

    if not missing_rows.empty:
        swapped_spreads = spreads_df.rename(columns={
            'home_team': 'away_team',
            'away_team': 'home_team',
            'home_score': 'away_score',
            'away_score': 'home_score'
        })[['season','week','home_team','away_team','spread','home_score','away_score']]

        swapped_merge = pd.merge(
            missing_rows.drop(columns=['spread','home_score','away_score'], errors='ignore'),
            swapped_spreads,
            on=['season','week','home_team','away_team'],
            how='left'
        )

        for col in ['spread','home_score','away_score']:
            if col in swapped_merge.columns:
                merged.loc[merged['spread'].isna(), col] = swapped_merge[col].values

    return merged

def compute_rest_days(schedule_df):
    def days_since_last_game(df, team_col):
        last_game_dates = {}
        rest_days = []
        for idx, row in df.sort_values(['season','week']).iterrows():
            team = row[team_col]
            try: game_date = pd.to_datetime(row['gameday'])
            except: game_date = None
            delta = (game_date - last_game_dates[team]).days if team in last_game_dates and game_date else np.nan
            rest_days.append(delta)
            if game_date: last_game_dates[team] = game_date
        return rest_days

    schedule_df = schedule_df.sort_values(['season','week'])
    schedule_df['home_rest'] = days_since_last_game(schedule_df,'home_team')
    schedule_df['away_rest'] = days_since_last_game(schedule_df,'away_team')
    schedule_df['rest_diff'] = schedule_df['home_rest'] - schedule_df['away_rest']
    return schedule_df[['season','week','home_team','away_team','rest_diff']]

def get_roster_injury_impact(years, max_week=None):
    teams = ['LV','BUF','CHI','DAL']
    max_week = max_week or 17
    data = []
    for year in years:
        for week in range(1,max_week+1):
            for team in teams:
                data.append({'team':team,'season':year,'week':week,'injury_impact':0})
    return pd.DataFrame(data)

def get_yearly_stats_by_weeks(years,max_week=None):
    teams = ['LV','BUF','CHI','DAL']
    max_week = max_week or 17
    data = []
    for year in years:
        for week in range(1,max_week+1):
            for team in teams:
                data.append({
                    'season': year,
                    'week': week,
                    'team': team,
                    'epa_offense': np.random.uniform(-1,1),
                    'epa_defense': np.random.uniform(-1,1),
                    'snap_counts': np.random.randint(50,100)
                })
    return pd.DataFrame(data)

def prepare_game_features(schedule_df, team_stats_df, rest_days_df, injury_df):
    df = schedule_df.copy()
    if not rest_days_df.empty:
        df = df.merge(rest_days_df, on=['season','week','home_team','away_team'], how='left')
    else:
        df['rest_diff'] = 0

    if not injury_df.empty:
        home_inj = injury_df.rename(columns={'team':'home_team','injury_impact':'injury_home'})
        away_inj = injury_df.rename(columns={'team':'away_team','injury_impact':'injury_away'})
        df = df.merge(home_inj[['season','week','home_team','injury_home']], on=['season','week','home_team'], how='left')
        df = df.merge(away_inj[['season','week','away_team','injury_away']], on=['season','week','away_team'], how='left')
        df['injury_home'].fillna(0,inplace=True)
        df['injury_away'].fillna(0,inplace=True)
        df['injury_diff'] = df['injury_home'] - df['injury_away']
    else:
        df['injury_diff'] = 0

    np.random.seed(0)
    df['epa_off_diff'] = np.random.normal(0,1,len(df))
    df['epa_def_diff'] = np.random.normal(0,1,len(df))
    df['snap_diff_offense'] = np.random.randint(-10,10,len(df))
    df['snap_diff_defense'] = np.random.randint(-10,10,len(df))
    df['home_advantage'] = 1

    if 'spread' not in df.columns: df['spread'] = 0
    return df

def add_ats_cover_label(df):
    def cover(row):
        if pd.isna(row['spread']) or pd.isna(row['home_score']) or pd.isna(row['away_score']): return np.nan
        margin = row['home_score'] - row['away_score']
        return int(margin > row['spread'])
    df['ats_cover'] = df.apply(cover, axis=1)
    return df

def train_model(df, feature_cols):
    X = df[feature_cols]
    y = df['ats_cover']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_scaled, y)
    return model, scaler

def predict_week(model, scaler, df, feature_cols):
    X_pred = df[feature_cols]
    X_pred = scaler.transform(X_pred)
    df['ats_prob'] = model.predict_proba(X_pred)[:,1]
    df['predicted_cover'] = model.predict(X_pred)

    home_team_idx = df.columns.get_loc('home_team')
    df.insert(home_team_idx+1,'home_cover_prob',df['ats_prob'])
    away_team_idx = df.columns.get_loc('away_team')
    df.insert(away_team_idx+1,'away_cover_prob',1-df['ats_prob'])

    desired_order = ['game_id','home_team','home_cover_prob','away_team','away_cover_prob','spread']
    other_cols = [col for col in df.columns if col not in desired_order]
    df = df[desired_order + other_cols]

    return df, df[df['predicted_cover']==1]

def get_week_schedule(season, week):
    schedules = nfl.import_schedules([season])
    schedules = schedules[(schedules['season']==season) & (schedules['week']==week) & (schedules['game_type']=='REG')]
    schedules = normalize_team_names(schedules,['home_team','away_team'])
    schedules = normalize_schedule_team_abbreviations(schedules)
    schedules['spread'] = np.nan
    return schedules

# --- Streamlit App ---
st.title("🏈 NFL ATS Prediction App with Missing Data Summary")

season = datetime.now().year
week = st.number_input("Enter NFL week to predict:", min_value=1, max_value=18, step=1)

if "schedule_df" not in st.session_state: st.session_state.schedule_df = None

if st.button("Load Schedule"):
    st.session_state.schedule_df = get_week_schedule(season, week)

if st.session_state.schedule_df is not None:
    schedule_df = st.session_state.schedule_df.copy()

    st.subheader("Enter Point Spreads")
    spreads = []
    for idx, row in schedule_df.iterrows():
        spread_input = st.text_input(
            f"{row['away_team']} @ {row['home_team']} (Week {row['week']})",
            value="", key=f"spread_{idx}"
        )
        try: spread_value = float(spread_input) if spread_input.strip() != "" else None
        except ValueError: spread_value = None
        spreads.append(spread_value)
    schedule_df['spread'] = spreads
    st.session_state.schedule_df = schedule_df

    if st.button("Run Predictions"):
        historical_schedules = nfl.import_schedules(list(range(2019, season)))
        historical_schedules = historical_schedules[historical_schedules['game_type']=='REG']
        historical_spreads = load_historical_spreads('nfl.kaggle.spreads.data.csv')
        merged_train_data = merge_historical_data(historical_schedules, historical_spreads)
        merged_train_data = merged_train_data.dropna(subset=['spread','home_score','away_score'])

        rest_days_train = compute_rest_days(merged_train_data)
        team_stats_train = get_yearly_stats_by_weeks(list(range(2019, season)))
        injury_train = get_roster_injury_impact(list(range(2019, season)))

        train_games = prepare_game_features(merged_train_data, team_stats_train, rest_days_train, injury_train)
        train_games = add_ats_cover_label(train_games)
        train_games.dropna(subset=['ats_cover'], inplace=True)

        feature_cols = ['epa_off_diff','epa_def_diff','snap_diff_offense','snap_diff_defense','rest_diff','injury_diff','home_advantage','spread']
        model, scaler = train_model(train_games, feature_cols)

        rest_days_pred = compute_rest_days(schedule_df)
        team_stats_pred = get_yearly_stats_by_weeks([season])
        injury_pred = get_roster_injury_impact([season])

        pred_games = prepare_game_features(schedule_df, team_stats_pred, rest_days_pred, injury_pred)
        predictions, upsets = predict_week(model, scaler, pred_games, feature_cols)

        st.subheader("Predictions")
        st.dataframe(predictions[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])

        if not upsets.empty:
            st.subheader("Upset Covers Predicted")
            st.dataframe(upsets[['home_team','home_cover_prob','away_team','away_cover_prob','spread']])

        predictions.to_excel(f"nfl_ats_predictions_week_{week}.xlsx", index=False)
        st.success(f"Results saved to nfl_ats_predictions_week_{week}.xlsx")
