import pandas as pd
from collections import defaultdict
import numpy as np

def get_data_and_preprocess(TRAIN_START=2015, TEST_START=2023, TEST_END=2026, URL="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"):
    URL = URL
    
    K = 24
    K_GOALS = 10
    
    df = pd.read_csv(URL)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    current_elo = defaultdict(lambda: 1500.0)
    home_elo = defaultdict(lambda: 1500.0)
    away_elo = defaultdict(lambda: 1500.0)
    goal_home_elo = defaultdict(lambda: 1500.0)
    goal_away_elo = defaultdict(lambda: 1500.0)
    attack_elo = defaultdict(lambda: 1500.0)
    defense_elo = defaultdict(lambda: 1500.0)
    
    win_streak = defaultdict(int)
    not_losing_streak = defaultdict(int)
    goals_scored_history = defaultdict(list)
    goals_conceded_history = defaultdict(list)
    form_history = defaultdict(list)
    clean_sheet_history = defaultdict(list)
    
    features = []
    
    for row in df.itertuples():
        h, a = row.home_team, row.away_team
        hs, aws = row.home_score, row.away_score
        date = row.date
    
        rh, ra = current_elo[h], current_elo[a]
        rhs, ras = home_elo[h], away_elo[a]
        rgh, rga = goal_home_elo[h], goal_away_elo[a]
        rah, raa = attack_elo[h], attack_elo[a]
        rdh, rda = defense_elo[h], defense_elo[a]
    
        e_home = 1 / (1 + 10 ** ((ra - rh) / 400))
        e_home_spec = 1 / (1 + 10 ** ((ras - rhs) / 400))
    
        exp_home_goals = 1.4 * 10 ** ((rah - rda) / 400)
        exp_away_goals = 1.4 * 10 ** ((raa - rdh) / 400)
    
        gs_h = goals_scored_history[h][-5:]
        gc_h = goals_conceded_history[h][-5:]
        gs_a = goals_scored_history[a][-5:]
        gc_a = goals_conceded_history[a][-5:]
    
        pts_h = form_history[h][-5:]
        pts_a = form_history[a][-5:]
        cs_h = clean_sheet_history[h][-5:]
        cs_a = clean_sheet_history[a][-5:]
    
        features.append({
            "date": date,
            "home_team": h,
            "away_team": a,
            "tournament": row.tournament,
            "home_score": hs,
            "away_score": aws,
            "Result": 0 if hs > aws else 1 if hs == aws else 2,
    
            "ELO_home": rh,
            "ELO_away": ra,
            "ELO_home_specific": rhs,
            "ELO_away_specific": ras,
            "ELO_goal_home": rgh,
            "ELO_goal_away": rga,
            "ELO_attack_home": rah,
            "ELO_attack_away": raa,
            "ELO_defense_home": rdh,
            "ELO_defense_away": rda,
    
            "Winning_streak_home": win_streak[h],
            "Winning_streak_away": win_streak[a],
            "Not_losing_streak_home": not_losing_streak[h],
            "Not_losing_streak_away": not_losing_streak[a],
    
            "Rolling_goals_scored_home": np.mean(gs_h) if gs_h else np.nan,
            "Rolling_goals_conceded_home": np.mean(gc_h) if gc_h else np.nan,
            "Rolling_goals_scored_away": np.mean(gs_a) if gs_a else np.nan,
            "Rolling_goals_conceded_away": np.mean(gc_a) if gc_a else np.nan,
    
            "Rolling_goal_difference_home": np.sum(gs_h) - np.sum(gc_h) if gs_h else np.nan,
            "Rolling_goal_difference_away": np.sum(gs_a) - np.sum(gc_a) if gs_a else np.nan,
    
            "Form_points_home": np.sum(pts_h) if pts_h else np.nan,
            "Form_points_away": np.sum(pts_a) if pts_a else np.nan,
    
            "Clean_sheets_home": np.sum(cs_h) if cs_h else np.nan,
            "Clean_sheets_away": np.sum(cs_a) if cs_a else np.nan,
        })
    
        gd = hs - aws
        mult = 1.0 if gd == 0 else np.log(abs(gd) + 1) * 1.75 / (1.75 + 0.00175 * abs(rh - ra))
    
        if gd > 0:
            sh, sa = 1.0, 0.0
        elif gd < 0:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5
    
        new_rh = rh + K * (sh - e_home)
        new_ra = ra + K * (sa - (1 - e_home))
    
        new_rhs = rhs + K * (sh - e_home_spec)
        new_ras = ras + K * (sa - (1 - e_home_spec))
    
        new_rgh = rgh + K * mult * (sh - e_home)
        new_rga = rga + K * mult * (sa - (1 - e_home))
    
        diff_h = hs - exp_home_goals
        diff_a = aws - exp_away_goals
    
        new_rah = rah + K_GOALS * diff_h
        new_rda = rda - K_GOALS * diff_h
        new_raa = raa + K_GOALS * diff_a
        new_rdh = rdh - K_GOALS * diff_a
    
        current_elo[h], current_elo[a] = new_rh, new_ra
        goal_home_elo[h], goal_away_elo[a] = new_rgh, new_rga
        attack_elo[h], attack_elo[a] = new_rah, new_raa
        defense_elo[h], defense_elo[a] = new_rdh, new_rda
    
        if not getattr(row, "neutral", False):
            home_elo[h] = new_rhs
            away_elo[a] = new_ras
    
        goals_scored_history[h].append(hs)
        goals_conceded_history[h].append(aws)
        goals_scored_history[a].append(aws)
        goals_conceded_history[a].append(hs)
    
        clean_sheet_history[h].append(int(aws == 0))
        clean_sheet_history[a].append(int(hs == 0))
    
        if gd > 0:
            form_history[h].append(3)
            form_history[a].append(0)
            win_streak[h] += 1
            win_streak[a] = 0
            not_losing_streak[h] += 1
            not_losing_streak[a] = 0
        elif gd == 0:
            form_history[h].append(1)
            form_history[a].append(1)
            win_streak[h] = 0
            win_streak[a] = 0
            not_losing_streak[h] += 1
            not_losing_streak[a] += 1
        else:
            form_history[h].append(0)
            form_history[a].append(3)
            win_streak[h] = 0
            win_streak[a] += 1
            not_losing_streak[h] = 0
            not_losing_streak[a] += 1
    
    df = pd.DataFrame(features)
    
    df = df[
        (df["date"] >= f"{TRAIN_START}-01-01") &
        (df["date"] < f"{TEST_END}-12-31")
    ].reset_index(drop=True)
    
    df = df.dropna().reset_index(drop=True)
    
    train_df = df[df["date"] < f"{TEST_START}-01-01"].copy()
    
    test_df = df[
        (df["date"] >= f"{TEST_START}-01-01") &
        (df["date"] < f"{TEST_END}-01-01")
    ].copy()
    
    
    train_teams = pd.concat([
        train_df["home_team"],
        train_df["away_team"]
    ]).unique()
    
    train_tournaments = train_df["tournament"].unique()
    
    team_id = {
        team: i
        for i, team in enumerate(train_teams)
    }
    
    tournament_id = {
        tournament: i
        for i, tournament in enumerate(train_tournaments)
    }
    
    df["home_id"] = df["home_team"].map(team_id)
    df["away_id"] = df["away_team"].map(team_id)
    df["tournament_id"] = df["tournament"].map(tournament_id)
    df = df.dropna().reset_index(drop=True)
    
    return train_df, test_df
