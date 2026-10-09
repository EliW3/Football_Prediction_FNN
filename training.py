from get_data_and_preprocess.py import get_data_and_preprocess
import pandas as pd
from fnn_network.py import FootballPredictor

def train(TEST_START, TEST_END)
  elo_columns = [
      "ELO_goal_home",
      "ELO_goal_away",
      "ELO_attack_home",
      "ELO_attack_away",
      "ELO_defense_home",
      "ELO_defense_away",
      "ELO_home",
      "ELO_away",
      "ELO_home_specific",
      "ELO_away_specific"
  ]
  running_columns = [
      "Winning_streak_home",
      "Winning_streak_away",
      "Not_losing_streak_home",
      "Not_losing_streak_away",
      "Rolling_goals_scored_home",
      "Rolling_goals_conceded_home",
      "Rolling_goals_scored_away",
      "Rolling_goals_conceded_away",
      "Rolling_goal_difference_home",
      "Rolling_goal_difference_away",
      "Form_points_home",
      "Form_points_away",
      "Clean_sheets_home",
      "Clean_sheets_away"
  ]
  train_df = df[df["date"] < f"{TEST_START}-01-01"].copy()
  test_df = df[
      (df["date"] >= f"{TEST_START}-01-01") &
      (df["date"] < f"{TEST_END}-01-01")
  ].copy()
  
  X_home = torch.tensor(train_df["home_id"].values, dtype=torch.long)
  X_away = torch.tensor(train_df["away_id"].values, dtype=torch.long)
  X_tournament = torch.tensor(train_df["tournament_id"].values, dtype=torch.long)
  
  X_elo = torch.tensor(
      train_df[elo_columns].values,
      dtype=torch.float32
  ) / 400.0
  
  X_running = torch.tensor(
      train_df[running_columns].values,
      dtype=torch.float32
  )
  
  y = torch.tensor(
      train_df["Result"].values,
      dtype=torch.long
  )
  
  dataset = TensorDataset(
      X_home,
      X_away,
      X_tournament,
      X_elo,
      X_running,
      y
  )
  
  loader = DataLoader(
      dataset,
      batch_size=256,
      shuffle=True
  )
  
  model = FootballPredictor()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_func = nn.CrossEntropyLoss()
  
  for epoch in range(30):
      model.train()
      total_loss = 0
  
      for home, away, tournament, elo, running, target in loader:
          logits = model(
              home,
              away,
              tournament,
              elo,
              running
          )
  
          loss = loss_func(logits, target)
  
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  
          total_loss += loss.item()
  
      print(
          f"Epoch {epoch + 1:02d} "
          f"Loss: {total_loss / len(loader):.4f}"
      )
