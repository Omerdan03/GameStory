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
3. **Run the Notebooks**:
   - Part A: Open and run `Part_A/Part_A.ipynb` to compute key metrics like Day-1 retention, 7-day cumulative ARPU, and 30-day churn rate.
   - Part B: Open and run `Part_B/Part_B.ipynb` to build the deposit forecasting model, generate predictions, and analyze feature importance.

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

The goal of this part is to build a model to forecast the expected deposit amount for the next 30 days per player. The process includes:
- **Data Loading & Feature Engineering**: Preprocessing the player and transaction data to create meaningful features.
- **Model Comparison**: Testing at least two predictive models (e.g., Random Forest, XGBoost) to forecast deposit amounts.
- **Feature Importance**: Visualizing which features contribute most to the model’s predictions.

The final output includes:
- A `scores.csv` file containing player IDs and their predicted deposit amounts for the next 30 days.
- A **model card** explaining the model's performance, metrics, and next steps.