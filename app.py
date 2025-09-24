import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# --- Styling ---
st.markdown("""
<style>
.stApp { background-color: #0C0C0C; color: #FFFFFF; }
h1, h2, h3 { color: #241773; }
.stButton>button { background-color: #241773; color: white; border-radius: 10px; border: 1px solid #9E7C0C; }
.stButton>button:hover { background-color: #3B1C87; }
input { background-color: #1C1C1C !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- Custom JS for Enter Key ---
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
    schedules_df = normalize_team_names(schedules_df, ['home_team','away_team'])
    schedules_df = normalize_schedule_team_abbreviations(schedules_df)
    schedules_df['season'] = pd.to_numeric(schedules_df['season'], errors='coerce').astype('Int64')
    schedules_df['week'] = pd.to_numeric(schedules_df['week'], errors='coerce').astype('Int64')

    spreads_df = normalize_team_names(spreads_df, ['home_team','away_team'])
    spreads_df['season'] = pd.to_numeric(spreads_df['season'], errors='coerce').astype('Int64')
    spreads_df['week'] = pd.to_numeric(spreads_df['week'], errors='coerce').astype('Int64')

    merged = pd.merge(schedules_df, spreads_df,
                      on=['season','week','home_team','away_team'],
                      how='left')

    # --- Safe missing data detection ---
    expected_cols = ['spread','home_score','away_score']
    existing_cols = [c for c in expected_cols if c in merged.columns]
    missing_rows = merged[merged[existing_cols].isna().any(axis=1)] if existing_cols else pd.DataFrame()

    if not missing_rows.empty:
        st.subheader("⚠️ Missing Historical Data Detected")
        st.write("Some games are missing spreads or scores:")
        st.dataframe(missing_rows[['season','week','home_team','away_team'] + existing_cols])
        missing_counts = missing_rows[existing_cols].isna().sum()
        missing_fraction = len(missing_rows)/len(merged)
        st.write("Count of missing values:", missing_counts)
        st.write(f"Fraction of games with missing data: {missing_fraction:.2%}")

        # Summary table
        summary_table = pd.DataFrame({
            'Total Games':[len(merged)],
            'Games Missing Spread':[missing_counts.get('spread',0)],
            'Games Missing Home Score':[missing_counts.get('home_score',0)],
            'Games Missing Away Score':[missing_counts.get('away_score',0)],
            'Fraction Missing':[missing_fraction]
        })
        st.subheader("📊 Missing Data Summary Table")
        st.dataframe(summary_table)

        # Attempt swapped merge safely
        swapped_spreads = spreads_df.rename(columns={
            'home_team':'away_team','away_team':'home_team',
            'home_score':'away_score','away_score':'home_score'
        })[['season','week','home_team','away_team'] + existing_cols]

        swapped_merge = pd.merge(
            missing_rows.drop(columns=existing_cols, errors='ignore'),
            swapped_spreads,
            on=['season','week','home_team','away_team'],
            how='left'
        )

        for col in existing_cols:
            if col in swapped_merge.columns:
                merged.loc[merged[col].isna(), col] = swapped_merge[col].values

    return merged

# --- Rest of functions remain the same (compute_rest_days, get_yearly_stats_by_weeks, prepare_game_features, etc.) ---
# --- Key update: safe dropna before training ---
def safe_dropna(df, subset_cols):
    existing_cols = [c for c in subset_cols if c in df.columns]
    missing_cols = set(subset_cols) - set(existing_cols)
    if missing_cols:
        st.warning(f"Columns missing in dropna and skipped: {missing_cols}")
    if existing_cols:
        df = df.dropna(subset=existing_cols)
    return df

# --- Streamlit App ---
st.title("🏈 NFL ATS Prediction App with Missing Data Summary")

season = datetime.now().year
week = st.number_input("Enter NFL week to predict:",min_value=1,max_value=18,step=1)

if "schedule_df" not in st.session_state: st.session_state.schedule_df=None

if st.button("Load Schedule"):
    st.session_state.schedule_df = get_week_schedule(season,week)

if st.session_state.schedule_df is not None:
    schedule_df = st.session_state.schedule_df.copy()

    st.subheader("Enter Point Spreads")
    spreads=[]
    for idx,row in schedule_df.iterrows():
        spread_input = st.text_input(f"{row['away_team']} @ {row['home_team']} (Week {row['week']})",value="",key=f"spread_{idx}")
        try: spread_value = float(spread_input) if spread_input.strip()!="" else None
        except: spread_value=None
        spreads.append(spread_value)
    schedule_df['spread']=spreads
    st.session_state.schedule_df = schedule_df

    if st.button("Run Predictions"):
        historical_schedules = nfl.import_schedules(list(range(2019,season)))
        historical_schedules = historical_schedules[historical_schedules['game_type']=='REG']
        historical_spreads = load_historical_spreads('nfl.kaggle.spreads.data.csv')
        merged_train_data = merge_historical_data(historical_schedules,historical_spreads)

        # --- Safe dropna before training ---
        merged_train_data = safe_dropna(merged_train_data,['spread','home_score','away_score'])

        # Continue with rest of pipeline (rest_days_train, team_stats_train, injury_train, train_model, etc.)
