from get_data_and_preprocess import get_data_and_preprocess
from fnn_network import FootballPredictor
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

train_df, _ = get_data_and_preprocess()

train_teams = pd.concat([train_df["home_team"], train_df["away_team"]]).unique()
train_tournaments = train_df["tournament"].unique()

team_id = {team: i for i, team in enumerate(train_teams)}
tournament_id = {tournament: i for i, tournament in enumerate(train_tournaments)}

num_teams = len(team_id)
num_tournaments = len(tournament_id)
model = FootballPredictor(num_teams=num_teams, num_tournaments=num_tournaments)

try:
    model.load_state_dict(torch.load("fnn_model.pth"))
    print("Successfully loaded trained parameters!")
except FileNotFoundError:
    print("Weights file not found. Running with baseline model.")

def get_latest_team_state(team, df):
    home_matches = df[df["home_team"] == team]
    away_matches = df[df["away_team"] == team]

    candidates = []

    if len(home_matches) > 0:
        candidates.append(("home", home_matches.iloc[-1]))

    if len(away_matches) > 0:
        candidates.append(("away", away_matches.iloc[-1]))

    if not candidates:
        raise ValueError(f"No historical matches found for {team}")

    role, row = max(
        candidates,
        key=lambda x: x[1]["date"]
    )

    if role == "home":
        suffix = "home"
    else:
        suffix = "away"

    return {
        "ELO_goal": row[f"ELO_goal_{suffix}"],
        "ELO_attack": row[f"ELO_attack_{suffix}"],
        "ELO_defense": row[f"ELO_defense_{suffix}"],
        "ELO": row[f"ELO_{suffix}"],
        "ELO_specific": row[f"ELO_{suffix}_specific"],

        "Winning_streak": row[f"Winning_streak_{suffix}"],
        "Not_losing_streak": row[f"Not_losing_streak_{suffix}"],

        "Rolling_goals_scored": row[
            f"Rolling_goals_scored_{suffix}"
        ],

        "Rolling_goals_conceded": row[
            f"Rolling_goals_conceded_{suffix}"
        ],

        "Rolling_goal_difference": row[
            f"Rolling_goal_difference_{suffix}"
        ],

        "Form_points": row[
            f"Form_points_{suffix}"
        ],

        "Clean_sheets": row[
            f"Clean_sheets_{suffix}"
        ],
    }

def predict_matches(matches):
    model.eval()
    predictions = []

    with torch.no_grad():
        for home_team, away_team, competition in matches:
            if home_team not in team_id:
                raise ValueError(f"Unknown home team: {home_team}")
            if away_team not in team_id:
                raise ValueError(f"Unknown away team: {away_team}")
            if competition not in tournament_id:
                raise ValueError(f"Unknown competition: {competition}")

            h = get_latest_team_state(home_team, train_df)
            a = get_latest_team_state(away_team, train_df)

            elo = torch.tensor([[
                h["ELO_goal"],
                a["ELO_goal"],
                h["ELO_attack"],
                a["ELO_attack"],
                h["ELO_defense"],
                a["ELO_defense"],
                h["ELO"],
                a["ELO"],
                h["ELO_specific"],
                a["ELO_specific"]
            ]], dtype=torch.float32) / 400.0

            running = torch.tensor([[
                h["Winning_streak"],
                a["Winning_streak"],
            
                h["Not_losing_streak"],
                a["Not_losing_streak"],
            
                h["Rolling_goals_scored"],
                h["Rolling_goals_conceded"],
            
                a["Rolling_goals_scored"],
                a["Rolling_goals_conceded"],
            
                h["Rolling_goal_difference"],
                a["Rolling_goal_difference"],
            
                h["Form_points"],
                a["Form_points"],
            
                h["Clean_sheets"],
                a["Clean_sheets"]
            ]], dtype=torch.float32)

            home = torch.tensor([team_id[home_team]], dtype=torch.long)
            away = torch.tensor([team_id[away_team]], dtype=torch.long)
            tournament = torch.tensor([tournament_id[competition]], dtype=torch.long)
            
            probs = torch.softmax(
                model(home, away, tournament, elo, running),
                dim=1
            )[0]

            predictions.append({
                "home_team": home_team,
                "away_team": away_team,
                "competition": competition,
                "home_win_probability": probs[0].item(),
                "draw_probability": probs[1].item(),
                "away_win_probability": probs[2].item(),
                "prediction": ["Home", "Draw", "Away"][probs.argmax().item()]

            })

    return pd.DataFrame(predictions)
if __name__ == "__main__":
    upcoming_fixtures = [
        ("Argentina", "Brazil", "Official-Match"),
        ("France", "England", "Official-Match")
    ]
    res = predict_matches(upcoming_fixtures)
    print(res)
