import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from fnn_network import FootballPredictor
from get_data_and_preprocess import get_data_and_preprocess
from predict import predict_matches
 

def backtest(model, test_df):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            elo = torch.tensor([[
                row["ELO_goal_home"],
                row["ELO_goal_away"],
                row["ELO_attack_home"],
                row["ELO_attack_away"],
                row["ELO_defense_home"],
                row["ELO_defense_away"],
                row["ELO_home"],
                row["ELO_away"],
                row["ELO_home_specific"],
                row["ELO_away_specific"]
            ]], dtype=torch.float32) / 400.0

            running = torch.tensor([[
    row["Winning_streak_home"],
    row["Winning_streak_away"],

    row["Not_losing_streak_home"],
    row["Not_losing_streak_away"],

    row["Rolling_goals_scored_home"],
    row["Rolling_goals_conceded_home"],

    row["Rolling_goals_scored_away"],
    row["Rolling_goals_conceded_away"],

    row["Rolling_goal_difference_home"],
    row["Rolling_goal_difference_away"],

    row["Form_points_home"],
    row["Form_points_away"],

    row["Clean_sheets_home"],
    row["Clean_sheets_away"]
]], dtype=torch.float32)

            home = torch.tensor([row["home_id"]], dtype=torch.long)
            away = torch.tensor([row["away_id"]], dtype=torch.long)
            tournament = torch.tensor(
                [row["tournament_id"]],
                dtype=torch.long
            )

            probs = torch.softmax(
                model(home, away, tournament, elo, running),
                dim=1
            )

            predictions.append(probs.numpy())
            actual.append(int(row["Result"]))

    predictions = np.array(predictions)
    actual = np.array(actual)

    return predictions, actual

def evaluate_predictions(predictions, actual):
    predictions = np.clip(
        predictions,
        1e-15,
        1 - 1e-15
    )

    predictions = predictions / predictions.sum(axis=1, keepdims=True)

    logloss = -np.mean(
        np.log(
            predictions[
                np.arange(len(actual)),
                actual
            ]
        )
    )

    predicted_class = np.argmax(
        predictions,
        axis=1
    )

    accuracy = np.mean(
        predicted_class == actual
    )

    one_hot = np.eye(3)[actual]

    brier = np.mean(
        np.sum(
            (predictions - one_hot) ** 2,
            axis=1
        )
    )

    confidence = predictions.max(axis=1)
    correct = (predicted_class == actual).astype(float)

    bins = np.linspace(0, 1, 11)
    calibration = []

    for i in range(10):
        if i == 9:
            mask = (
                (confidence >= bins[i]) &
                (confidence <= bins[i + 1])
            )
        else:
            mask = (
                (confidence >= bins[i]) &
                (confidence < bins[i + 1])
            )

        if mask.sum() > 0:
            calibration.append({
                "bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}",
                "predicted": confidence[mask].mean(),
                "actual": correct[mask].mean(),
                "count": int(mask.sum())
            })

    calibration = pd.DataFrame(calibration)

    print(f"\nMatches:  {len(actual)}")
    print(f"Log loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Brier:    {brier:.4f}")
    print("\nCalibration:")
    print(calibration.to_string(index=False))

    return {
        "logloss": logloss,
        "accuracy": accuracy,
        "brier": brier,
        "calibration": calibration
    }

train_df, test_df = get_data_and_preprocess()
num_teams = int(max(train_df["home_id"].max(), train_df["away_id"].max()) + 1)
num_tournaments = int(train_df["tournament_id"].max() + 1)
model = FootballPredictor(num_teams=num_teams, num_tournaments=num_tournaments)
model.load_state_dict(torch.load("fnn_model.pth"))

predictions, actual = backtest(model, test_df)

results = evaluate_predictions(
    predictions,
    actual
)

backtest_df = test_df[
    ["date", "home_team", "away_team", "tournament",
     "home_score", "away_score", "Result"]
].copy()

backtest_df["home_win_probability"] = predictions[:, 0]
backtest_df["draw_probability"] = predictions[:, 1]
backtest_df["away_win_probability"] = predictions[:, 2]
backtest_df["prediction"] = np.argmax(predictions, axis=1)

backtest_df["prediction"] = backtest_df["prediction"].map({
    0: "Home",
    1: "Draw",
    2: "Away"
})

print("\nBacktest:")
print(backtest_df.head(20).to_string(index=False))

matches = [
    ("Northern Ireland", "Austria", "UEFA Nations League"),
    ("Germany", "Netherlands", "UEFA Nations League"),
    ("France", "Italy", "UEFA Nations League")
]

future_predictions = predict_matches(matches)

print("\nPredictions:")
print(future_predictions.to_string(index=False))
