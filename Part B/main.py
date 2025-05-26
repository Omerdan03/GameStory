import argparse
import pickle
from pathlib import Path

import pandas as pd


def load_model(model_path):
    """Load the pre-trained model from a pickle file."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def read_data(data_folder):
    """Read all input CSVs into pandas DataFrames."""
    players_df = pd.read_csv(data_folder / "players.csv")
    sessions_df = pd.read_csv(data_folder / "sessions.csv")
    transactions_df = pd.read_csv(data_folder / "transactions.csv")
    return players_df, sessions_df, transactions_df


def preprocess_sessions(sessions_df):
    """Preprocess sessions to extract session-based features."""
    sessions_df['start_ts'] = pd.to_datetime(sessions_df['start_ts'])
    sessions_df['end_ts'] = pd.to_datetime(sessions_df['end_ts'])

    return sessions_df.groupby('player_id').agg(
        num_sessions=('session_id', 'count'),
        avg_session_length=(
            'end_ts', lambda x: (x - sessions_df.loc[x.index, 'start_ts']).dt.total_seconds().mean()
        )
    ).reset_index()


def preprocess_transactions(transactions_df):
    """Preprocess transactions to extract financial features."""
    transactions_df['txn_ts'] = pd.to_datetime(transactions_df['txn_ts'])

    agg = transactions_df.groupby('player_id').agg(
        total_deposit=('amount', 'sum'),
        avg_deposit=('amount', 'mean'),
        deposit_count=('amount', 'count'),
        first_deposit=('txn_ts', 'min'),
        last_deposit=('txn_ts', 'max')
    ).reset_index()

    agg['active_days'] = (agg['last_deposit'] - agg['first_deposit']).dt.days + 1
    return agg


def generate_features(players_df, sessions_df, transactions_df):
    """Merge and clean all features into a single DataFrame."""
    session_features = preprocess_sessions(sessions_df)
    transaction_features = preprocess_transactions(transactions_df)

    features = players_df.copy()
    features = features.merge(transaction_features, on='player_id', how='left')
    features = features.merge(session_features, on='player_id', how='left')

    return features


def transform_features(features_df, model):
    """Apply encoding and scaling to features."""
    X = features_df.drop(columns=['player_id', 'signup_date', 'first_deposit', 'last_deposit'], errors='ignore').copy()

    for col, encoder in model['encoders'].items():
        encoded = encoder.transform(X[[col]])
        encoded_df = pd.DataFrame(
            encoded,
            index=X.index,
            columns=encoder.get_feature_names_out([col])
        )
        X = X.drop(columns=col)
        X = pd.concat([X, encoded_df], axis=1)

    X_scaled = model['scaler'].transform(X.dropna())
    return X, X_scaled


def predict_clusters(X_scaled, model):
    """Assign cluster labels and map to average deposit predictions."""
    cluster_labels = model['cluster'].predict(X_scaled)
    cluster_avg_map = dict(zip(
        model['cluster_avg']['cluster'],
        model['cluster_avg']['avg_30_day_deposit']
    ))
    return pd.Series(cluster_labels).map(cluster_avg_map)


def main(data_path, model_path, output_path):
    model = load_model(model_path)
    players_df, sessions_df, transactions_df = read_data(data_path)
    features_df = generate_features(players_df, sessions_df, transactions_df)

    results = features_df[['player_id']].copy()

    X, X_scaled = transform_features(features_df, model)

    predicted_deposits = predict_clusters(X_scaled, model)

    results.loc[predicted_deposits.index, 'prediction'] = predicted_deposits
    results['prediction'] = results['prediction'].fillna(0)

    results.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deposit prediction model")
    parser.add_argument("--data_path", type=Path, default=Path("../data/"), help="Path to data folder")
    parser.add_argument("--model_path", type=Path, default=Path.cwd() / "model.pkl", help="Path to pickled model file")
    parser.add_argument("--output_path", type=Path, default=Path("scores.csv"), help="Path to save predictions CSV")

    args = parser.parse_args()
    main(args.data_path, args.model_path, args.output_path)
