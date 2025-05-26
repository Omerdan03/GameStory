# Game Story - Applied Analytics & Data Science Take-Home Challenge

## Overview

This repository contains the solution for the Game Story Applied Analytics & Data Science Take-Home Challenge. The goal of this challenge was to provide actionable insights from anonymized game data and build a predictive model for forecasting player deposits.

The data consists of four CSV files:
- `players.csv`: Contains player demographics and acquisition details.
- `sessions.csv`: Logs of player gameplay sessions.
- `transactions.csv`: Includes details of deposits, withdrawals, entry fees, and rewards.
- `tournaments.csv`: Contains metadata about tournaments.

The challenge is split into two parts:
1. **Part A**: Business analysis and insights.
2. **Part B**: Predictive modeling for deposit forecasting.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Part A: Insights and KPIs](#part-a-insights-and-kpis)
3. [Part B: Predictive Model](#part-b-predictive-model)
4. [Environment Setup](#environment-setup)
5. [Model Card](#model-card)
6. [File Structure](#file-structure)

---

## Quick Start

To run this project, you need to have Python and the required libraries installed. Follow these steps to get started:

1. **Clone this repository** or download the zip file.
2. **Set up the environment**:
   - If using `conda`, create a new environment:
     ```bash
     conda env create -f environment.yml
     conda activate game_story_env
     ```
   - If using `pip`, install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
3. **Run the Notebooks and Prediction Script**:
   - **Part A**: Open and run `Part_A/Part_A.ipynb` to compute key metrics such as Day-1 retention, 7-day cumulative ARPU, and 30-day churn rate.
   - **Part B**: Open and run `Part_B/Part_B.ipynb` to follow the workflow for training a deposit forecasting model. This includes feature engineering, model evaluation, and feature importance analysis.  
     This notebook produces a `model.pkl` file that can be used for prediction on new data.
   - **Run Prediction Script**: Once the model is trained, you can generate predictions by running the script below:

     ```bash
     python Part_B/main.py \
       --data_path ../data/ \
       --model_path ./Part_B/model.pkl \
       --output_path ./scores.csv
     ```

     - `--data_path`: Path to the folder containing `players.csv`, `sessions.csv`, and `transactions.csv`.
     - `--model_path`: Path to the trained `model.pkl` file.
     - `--output_path`: Where to save the predictions (default is `scores.csv`).

---

## Part A: Insights and KPIs

In this part, we focus on calculating key performance indicators (KPIs) and providing business insights. The KPIs calculated are:
- **Day-1 Retention**: Percentage of players who return to the game after their first session.
- **7-Day Cumulative ARPU**: Average revenue per user over the first 7 days of gameplay.
- **30-Day Churn Rate**: Percentage of players who are inactive after 30 days.

The insights derived from the analysis include actionable business takeaways such as:
- Identification of cohorts with the most significant opportunities for retention improvement.
- Key areas for monetization growth in Q3.

The results are detailed in the `Part_A.ipynb` notebook and the `Insight_memo.pdf`.

---

## Part B: Predictive Model

The goal of this part is to build a model to forecast the expected deposit amount for the next 30 days per player.

### Regression Model – XGBoost for Deposit Forecasting

**Key Features Used:**  
- `avg_session_length`, `active_days`, `total_deposit`, `deposit_count`, `avg_deposit`

**Performance Metrics:**  
- **R² score:** -0.147  
- **RMSE:** $41  
- **Error Pattern:** Low prediction variance, similar to baseline model

**SHAP Insights:**  
- High `avg_deposit` and `active_days` → increase predicted value  
- High `total_deposit` and `num_sessions` → decrease prediction (saturation/churn risk)

---

### Clustering Model – KMeans Behavioral Segmentation

**Key Features Used:**  
- Scaled features: `avg_session_length`, `active_days`, `total_deposit`, `deposit_count`, etc.

**Performance Metrics:**  
- **R² score:** 0.002  
- **RMSE:** $39  
- **Error Pattern:** Low prediction variance, behaved like baseline model

**Insights:**  
- Average deposit per cluster showed little variance  
- Indicates clustering does not meaningfully separate high/low depositors

---

### Time Series Model (ARIMA) – Not Applicable

**Reason:**  
Although ARIMA models are suited for sequential forecasting, they are not applicable here due to insufficient data per player. Most users lack a long enough deposit history to fit and validate time series models such as ARIMA or ARIMAX.

