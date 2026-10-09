import torch
import torch.nn as nn

class FootballPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.85)

        self.team_embedding = nn.Embedding(len(train_teams), 8)
        self.team_home_specific_embedding = nn.Embedding(len(train_teams), 8)
        self.team_away_specific_embedding = nn.Embedding(len(train_teams), 8)
        self.competition_embedding = nn.Embedding(len(train_tournaments), 4)

        self.fc = nn.Sequential(
            nn.Linear(60, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 3)
        )

    def forward(self, home, away, tournament, elo, running):
        h = self.team_embedding(home)
        a = self.team_embedding(away)
        hs = self.team_home_specific_embedding(home)
        aw = self.team_away_specific_embedding(away)
        c = self.competition_embedding(tournament)

        x = torch.cat([h, a, hs, aw, c, elo, running], dim=1)
        return self.fc(x) / self.temperature
