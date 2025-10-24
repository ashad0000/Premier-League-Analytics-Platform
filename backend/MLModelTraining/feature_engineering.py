# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, data_dir='data/files/StandardizedSeasonMatches'):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()

    def load_all_seasons(self):
            """Load all season match data"""
            # Fix: remove the duplicate 'StandardizedSeasonMatches' in the path
            season_files = sorted(self.data_dir.glob('EPLS*.csv'))

            if not season_files:
                raise FileNotFoundError(
                    f"No season files found in {self.data_dir}. "
                    f"Expected files matching pattern 'EPLS*.csv'"
                )

            all_matches = []
            print(f"Found {len(season_files)} season files")

            for file in season_files:
                df = pd.read_csv(file)
                season = file.stem.replace('EPLS', '')
                df['Season'] = season
                all_matches.append(df)
                print(f"  Loaded {file.name}: {len(df)} matches")

            return pd.concat(all_matches, ignore_index=True)
    def create_current_season_features(self, df):
        """Feature engineering for current season prediction model"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

        features_df = pd.DataFrame()

        # Team-specific rolling features
        for team_col, opp_col, location in [('HomeTeam', 'AwayTeam', 'home'),
                                            ('AwayTeam', 'HomeTeam', 'away')]:

            # Goals scored/conceded rolling averages (optimal windows: 5, 10 matches)
            for window in [5, 10]:
                if location == 'home':
                    goals_scored = df.groupby(team_col)['FTHG'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    goals_conceded = df.groupby(team_col)['FTAG'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                else:
                    goals_scored = df.groupby(team_col)['FTAG'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    goals_conceded = df.groupby(team_col)['FTHG'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )

                features_df[f'{location}_goals_scored_L{window}'] = goals_scored
                features_df[f'{location}_goals_conceded_L{window}'] = goals_conceded
                features_df[f'{location}_goal_diff_L{window}'] = goals_scored - goals_conceded

            # Shots rolling averages
            for window in [5, 10]:
                if location == 'home':
                    features_df[f'{location}_shots_L{window}'] = df.groupby(team_col)['HS'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    features_df[f'{location}_shots_on_target_L{window}'] = df.groupby(team_col)['HST'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                else:
                    features_df[f'{location}_shots_L{window}'] = df.groupby(team_col)['AS'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    features_df[f'{location}_shots_on_target_L{window}'] = df.groupby(team_col)['AST'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )

            # Corners rolling averages
            for window in [5, 10]:
                if location == 'home':
                    features_df[f'{location}_corners_L{window}'] = df.groupby(team_col)['HC'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                else:
                    features_df[f'{location}_corners_L{window}'] = df.groupby(team_col)['AC'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )

            # Fouls and cards rolling averages
            for window in [5, 10]:
                if location == 'home':
                    features_df[f'{location}_fouls_L{window}'] = df.groupby(team_col)['HF'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    features_df[f'{location}_yellow_cards_L{window}'] = df.groupby(team_col)['HY'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                else:
                    features_df[f'{location}_fouls_L{window}'] = df.groupby(team_col)['AF'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    features_df[f'{location}_yellow_cards_L{window}'] = df.groupby(team_col)['AY'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )

        # Win/loss streaks
        df['HomeWin'] = (df['FTR'] == 'H').astype(int)
        df['AwayWin'] = (df['FTR'] == 'A').astype(int)
        df['Draw'] = (df['FTR'] == 'D').astype(int)

        features_df['home_win_streak'] = df.groupby('HomeTeam')['HomeWin'].transform(
            lambda x: x.rolling(5, min_periods=1).sum().shift(1)
        )
        features_df['away_win_streak'] = df.groupby('AwayTeam')['AwayWin'].transform(
            lambda x: x.rolling(5, min_periods=1).sum().shift(1)
        )

        # Form (points in last N matches)
        def calculate_form(group, window=5):
            points = []
            for _, row in group.iterrows():
                if row['FTR'] == 'H':
                    points.append(3)
                elif row['FTR'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
            return pd.Series(points, index=group.index).rolling(window, min_periods=1).sum().shift(1)

        features_df['home_form_L5'] = df.groupby('HomeTeam').apply(calculate_form).reset_index(level=0, drop=True)
        features_df['away_form_L5'] = df.groupby('AwayTeam').apply(calculate_form).reset_index(level=0, drop=True)

        # Head-to-head features
        df['MatchPair'] = df.apply(lambda x: tuple(sorted([x['HomeTeam'], x['AwayTeam']])), axis=1)
        features_df['h2h_home_wins'] = df.groupby('MatchPair')['HomeWin'].transform(
            lambda x: x.expanding().sum().shift(1)
        )
        features_df['h2h_away_wins'] = df.groupby('MatchPair')['AwayWin'].transform(
            lambda x: x.expanding().sum().shift(1)
        )

        # Interaction terms for logistic regression
        features_df['goal_diff_interaction'] = (
                features_df['home_goal_diff_L5'] * features_df['away_goal_diff_L5']
        )
        features_df['form_interaction'] = (
                features_df['home_form_L5'] * features_df['away_form_L5']
        )
        features_df['shots_efficiency_home'] = (
                features_df['home_shots_on_target_L5'] / (features_df['home_shots_L5'] + 1)
        )
        features_df['shots_efficiency_away'] = (
                features_df['away_shots_on_target_L5'] / (features_df['away_shots_L5'] + 1)
        )

        # Add target and metadata
        features_df['FTR'] = df['FTR']
        features_df['Season'] = df['Season']
        features_df['HomeTeam'] = df['HomeTeam']
        features_df['AwayTeam'] = df['AwayTeam']
        features_df['Date'] = df['Date']

        return features_df.dropna()

    def create_historical_features(self, df):
        """Feature engineering for cross-era historical comparison model"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

        features_df = pd.DataFrame()

        # Era-normalized metrics (normalize by season averages)
        season_stats = df.groupby('Season').agg({
            'FTHG': 'mean',
            'FTAG': 'mean',
            'HS': 'mean',
            'AS': 'mean',
            'HST': 'mean',
            'AST': 'mean'
        }).add_suffix('_season_avg')

        df = df.merge(season_stats, left_on='Season', right_index=True)

        # Normalized strength metrics
        features_df['home_scoring_strength'] = df['FTHG'] / (df['FTHG_season_avg'] + 0.01)
        features_df['away_scoring_strength'] = df['FTAG'] / (df['FTAG_season_avg'] + 0.01)
        features_df['home_shot_efficiency'] = df['HST'] / (df['HS'] + 1)
        features_df['away_shot_efficiency'] = df['AST'] / (df['AS'] + 1)

        # Relative performance indicators (vs contemporary teams)
        for team_col, location in [('HomeTeam', 'home'), ('AwayTeam', 'away')]:
            # Season-relative win rate
            season_team_stats = df.groupby(['Season', team_col]).agg({
                'FTR': lambda x: (x == location[0].upper()).sum() / len(x)
            }).reset_index()
            season_team_stats.columns = ['Season', team_col, f'{location}_win_rate']

            df = df.merge(season_team_stats, on=['Season', team_col], how='left')
            features_df[f'{location}_relative_win_rate'] = df[f'{location}_win_rate']

            # Points per game in season
            def calc_ppg(group):
                ppg = []
                total_points = 0
                matches = 0
                for _, row in group.iterrows():
                    if matches > 0:
                        ppg.append(total_points / matches)
                    else:
                        ppg.append(0)

                    if location == 'home' and row['FTR'] == 'H':
                        total_points += 3
                    elif location == 'away' and row['FTR'] == 'A':
                        total_points += 3
                    elif row['FTR'] == 'D':
                        total_points += 1
                    matches += 1

                return pd.Series(ppg, index=group.index)

            features_df[f'{location}_ppg'] = df.groupby(['Season', team_col]).apply(calc_ppg).reset_index(level=[0,1], drop=True)

        # Era-agnostic team strength (ELO-style rating)
        team_ratings = {}
        initial_rating = 1500
        k_factor = 32

        home_ratings = []
        away_ratings = []

        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            if home_team not in team_ratings:
                team_ratings[home_team] = initial_rating
            if away_team not in team_ratings:
                team_ratings[away_team] = initial_rating

            home_ratings.append(team_ratings[home_team])
            away_ratings.append(team_ratings[away_team])

            # Update ratings based on result
            expected_home = 1 / (1 + 10 ** ((team_ratings[away_team] - team_ratings[home_team]) / 400))

            if row['FTR'] == 'H':
                actual_home = 1
            elif row['FTR'] == 'D':
                actual_home = 0.5
            else:
                actual_home = 0

            team_ratings[home_team] += k_factor * (actual_home - expected_home)
            team_ratings[away_team] += k_factor * ((1 - actual_home) - (1 - expected_home))

        features_df['home_elo_rating'] = home_ratings
        features_df['away_elo_rating'] = away_ratings
        features_df['elo_diff'] = features_df['home_elo_rating'] - features_df['away_elo_rating']

        # Style compatibility metrics (attacking vs defensive profiles)
        features_df['home_attack_style'] = df['HS'] / (df['HS_season_avg'] + 0.01)
        features_df['away_attack_style'] = df['AS'] / (df['AS_season_avg'] + 0.01)
        features_df['style_compatibility'] = features_df['home_attack_style'] * features_df['away_attack_style']

        # Consistency metrics (variance in performance)
        for team_col, location in [('HomeTeam', 'home'), ('AwayTeam', 'away')]:
            if location == 'home':
                features_df[f'{location}_performance_variance'] = df.groupby(team_col)['FTHG'].transform(
                    lambda x: x.rolling(10, min_periods=3).std().shift(1)
                )
            else:
                features_df[f'{location}_performance_variance'] = df.groupby(team_col)['FTAG'].transform(
                    lambda x: x.rolling(10, min_periods=3).std().shift(1)
                )

        # Add target and metadata
        features_df['FTR'] = df['FTR']
        features_df['Season'] = df['Season']
        features_df['HomeTeam'] = df['HomeTeam']
        features_df['AwayTeam'] = df['AwayTeam']
        features_df['Date'] = df['Date']

        return features_df.dropna()

def save_processed_data(self):
    """Process and save both feature sets"""
    print("Loading all season data...")
    all_matches = self.load_all_seasons()

    print("Creating current season features...")
    current_season_features = self.create_current_season_features(all_matches)

    # Fix: use Path object consistently
    output_dir = Path('data/files/MLData')
    output_dir.mkdir(parents=True, exist_ok=True)

    current_season_features.to_csv(output_dir / 'current_season_features.csv', index=False)
    print(f"Saved current season features: {len(current_season_features)} rows, {len(current_season_features.columns)} features")

    print("Creating historical comparison features...")
    historical_features = self.create_historical_features(all_matches)
    historical_features.to_csv(output_dir / 'historical_features.csv', index=False)
    print(f"Saved historical features: {len(historical_features)} rows, {len(historical_features.columns)} features")

    return current_season_features, historical_features
if __name__ == "__main__":
    engineer = FeatureEngineer()
    current_features, historical_features = engineer.save_processed_data()
    print("\nFeature engineering complete!")
