## Feature Engineering & Columns

The model utilizes three main categories of engineered features to capture team strength, current form, historical momentum, and team identities:

### 1. Elo Rating Metrics (`elo_columns`)
These features track long-term team strength and specialized tactical ratings derived from an Elo-based calculation system.

| Column Name | Description |
| :--- | :--- |
| `ELO_goal_home` | Baseline goal-based Elo rating for the home team. |
| `ELO_goal_away` | Baseline goal-based Elo rating for the away team. |
| `ELO_attack_home` | Offensive strength rating for the home team. |
| `ELO_attack_away` | Offensive strength rating for the away team. |
| `ELO_defense_home` | Defensive resilience rating for the home team. |
| `ELO_defense_away` | Defensive resilience rating for the away team. |
| `ELO_home` | Overall general Elo rating for the home team. |
| `ELO_away` | Overall general Elo rating for the away team. |
| `ELO_home_specific` | Location-specific Elo rating for the home team playing at home. |
| `ELO_away_specific` | Location-specific Elo rating for the away team playing away. |

### 2. Form & Running Metrics (`running_columns`)
These features capture short-term momentum, rolling performance windows, and recent defensive/offensive consistency.

| Column Name | Description |
| :--- | :--- |
| `Winning_streak_home` | Number of consecutive wins for the home team leading into the match. |
| `Winning_streak_away` | Number of consecutive wins for the away team leading into the match. |
| `Not_losing_streak_home` | Unbeaten streak count (wins + draws) for the home team. |
| `Not_losing_streak_away` | Unbeaten streak count (wins + draws) for the away team. |
| `Rolling_goals_scored_home` | Total or average goals scored by the home team over a recent rolling window. |
| `Rolling_goals_conceded_home` | Total or average goals conceded by the home team over a recent rolling window. |
| `Rolling_goals_scored_away` | Total or average goals scored by the away team over a recent rolling window. |
| `Rolling_goals_conceded_away` | Total or average goals conceded by the away team over a recent rolling window. |
| `Rolling_goal_difference_home` | Goal difference tracking over a recent rolling window for the home team. |
| `Rolling_goal_difference_away` | Goal difference tracking over a recent rolling window for the away team. |
| `Form_points_home` | Weighted points accumulated by the home team over recent matches (e.g., last 5 games). |
| `Form_points_away` | Weighted points accumulated by the away team over recent matches (e.g., last 5 games). |
| `Clean_sheets_home` | Total clean sheets kept by the home team over the rolling window. |
| `Clean_sheets_away` | Total clean sheets kept by the away team over the rolling window. |

### 3. Team Entity Embeddings
High-dimensional categorical representations processed by the neural network to learn hidden team characteristics and handle different tournament contexts.

| Feature / Embedding Type | Description |
| :--- | :--- |
| `General Team-Specific Embedding` | Learns latent features of a team's overall identity, historical tier, and playstyle regardless of venue. |
| `Home Team-Specific Embedding` | Captures unique structural characteristics, tactical variations, and psychological factors when playing in front of a home crowd. |
| `Away Team-Specific Embedding` | Captures a team's tactical profile, defensive setups, and historical resilience when playing in hostile environments. |
| `Neutral Competition Embedding` | Adjusts team representations and mitigates standard home-field advantage metrics when matches are played at neutral venues (e.g., tournament finals, international group stages). |
