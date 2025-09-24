# 🏈 NFL ATS Prediction App

A Streamlit app that predicts NFL teams against the spread (ATS).

## Features
- Load current season schedules via [nfl_data_py](https://github.com/nflverse/nfl_data_py).
- Manually enter betting spreads per game.
- Generate predictions using an **XGBoost** model.
- Download results as Excel.

## How to Run Locally
```bash
# Clone repo
git clone https://github.com/your-username/nfl-ats-app.git
cd nfl-ats-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
