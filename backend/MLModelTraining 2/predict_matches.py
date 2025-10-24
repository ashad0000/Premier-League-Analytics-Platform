# backend/MLModelTraining/predict_matches.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class MatchPredictor:
    def __init__(self, data_dir='data/files'):
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / 'MLModelTraining' / 'models'
        self.data_ml_dir = self.data_dir / 'MLData'
        self.load_models()
        self.load_team_data()

    def load_models(self):
        """Load trained models and scalers"""
        print("Loading models...")

        # Current season model
        self.current_model = joblib.load(self.model_dir / 'current_season_ensemble.pkl')
        self.current_scaler = joblib.load(self.model_dir / 'current_season_scaler.pkl')
        self.current_features = joblib.load(self.model_dir / 'current_season_features.pkl')

        # Historical model
        self.historical_model = joblib.load(self.model_dir / 'historical_ensemble.pkl')
        self.historical_scaler = joblib.load(self.model_dir / 'historical_scaler.pkl')
        self.historical_features = joblib.load(self.model_dir / 'historical_features.pkl')

        # Team encoder
        self.team_encoder = joblib.load(self.data_ml_dir / 'team_encoder.pkl')

        print("✓ Models loaded successfully")

    def load_team_data(self):
        """Load preprocessed feature data"""
        # Load current season data (202526)
        self.current_season_data = pd.read_csv(self.data_ml_dir / 'current_season_202526.csv')

        # Load team mapping
        self.team_mapping = pd.read_csv(self.data_ml_dir / 'team_mapping.csv')

        # Load historical training data for reference
        self.historical_data = pd.read_csv(self.data_ml_dir / 'historical_training.csv')

        # Get all unique teams - filter out NaN/float values
        all_classes = self.team_encoder.classes_
        # Keep only string values (actual team names)
        self.all_teams = sorted([str(team) for team in all_classes if isinstance(team, str)])

        print(f"✓ Loaded data for {len(self.all_teams)} teams")

    def get_latest_team_stats(self, team, is_home=True):
        """Get most recent statistics for a team from current season data"""
        # Get team's encoded ID
        team_encoded = self.team_encoder.transform([team])[0]
        
        # Filter mapping for this team's matches
        if is_home:
            team_matches = self.team_mapping[
                (self.team_mapping['HomeTeam'] == team) & 
                (self.team_mapping['Season'] == '202526')
            ]
            prefix = 'home_'
        else:
            team_matches = self.team_mapping[
                (self.team_mapping['AwayTeam'] == team) & 
                (self.team_mapping['Season'] == '202526')
            ]
            prefix = 'away_'
        
        if len(team_matches) == 0:
            # Return zero features if no recent matches
            return {feat: 0.0 for feat in self.current_features if feat.startswith(prefix)}
        
        # Get the index of the most recent match
        latest_idx = team_matches.index[-1]
        
        # Extract features from current season data
        features = {}
        for feat in self.current_features:
            if feat.startswith(prefix):
                if feat in self.current_season_data.columns:
                    # Get the corresponding row in current_season_data
                    if latest_idx < len(self.current_season_data):
                        features[feat] = self.current_season_data.loc[latest_idx, feat]
                    else:
                        features[feat] = 0.0
                else:
                    features[feat] = 0.0
        
        return features

    def predict_match(self, home_team, away_team, season='202526'):
        """Predict match outcome for any season"""
        print(f"\n{'='*70}")
        print(f"=== Match Prediction ===")
        print(f"Home: {home_team}")
        print(f"Away: {away_team}")
        print(f"Season: {season}")
        print(f"{'='*70}")
        
        # Validate teams
        if home_team not in self.all_teams:
            print(f"Error: '{home_team}' not found. Available teams:")
            for team in self.all_teams:
                print(f"  - {team}")
            return None
        
        if away_team not in self.all_teams:
            print(f"Error: '{away_team}' not found. Available teams:")
            for team in self.all_teams:
                print(f"  - {team}")
            return None
        
        # Use current season model if season is 202526
        if season == '202526':
            return self._predict_current_season(home_team, away_team)
        else:
            return self._predict_historical(home_team, away_team, season)

    def _predict_current_season(self, home_team, away_team):
        """Predict using current season model"""
        # Get features for both teams
        home_features = self.get_latest_team_stats(home_team, is_home=True)
        away_features = self.get_latest_team_stats(away_team, is_home=False)
        
        # Combine features in correct order
        feature_vector = []
        for feat in self.current_features:
            if feat.startswith('home_'):
                feature_vector.append(home_features.get(feat, 0.0))
            elif feat.startswith('away_'):
                feature_vector.append(away_features.get(feat, 0.0))
            else:
                # For non-location features, use average or zero
                feature_vector.append(0.0)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        # Scale and predict
        feature_vector_scaled = self.current_scaler.transform(feature_vector)
        probabilities = self.current_model.predict_proba(feature_vector_scaled)[0]
        prediction = self.current_model.predict(feature_vector_scaled)[0]
        
        # Map prediction (0=Away, 1=Draw, 2=Home)
        outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        print(f"\n--- Prediction Results ---")
        print(f"Most Likely Outcome: {outcome_map[prediction]}")
        print(f"\nProbabilities:")
        print(f"  Home Win ({home_team}): {probabilities[2]*100:.2f}%")
        print(f"  Draw: {probabilities[1]*100:.2f}%")
        print(f"  Away Win ({away_team}): {probabilities[0]*100:.2f}%")
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': outcome_map[prediction],
            'probabilities': {
                'home_win': float(probabilities[2]),
                'draw': float(probabilities[1]),
                'away_win': float(probabilities[0])
            }
        }

    def _predict_historical(self, home_team, away_team, season):
        """Predict using historical model for past seasons"""
        # For historical predictions, use averaged features from historical data
        # This is simplified since we removed identifiers from training
        
        # Create a feature vector with default historical values
        feature_vector = np.zeros(len(self.historical_features))
        
        # Fill with mean values from historical data as baseline
        for i, feat in enumerate(self.historical_features):
            if feat in self.historical_data.columns:
                feature_vector[i] = self.historical_data[feat].mean()
        
        feature_vector = feature_vector.reshape(1, -1)
        
        # Handle NaN
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        # Scale and predict
        feature_vector_scaled = self.historical_scaler.transform(feature_vector)
        probabilities = self.historical_model.predict_proba(feature_vector_scaled)[0]
        prediction = self.historical_model.predict(feature_vector_scaled)[0]
        
        outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        print(f"\n--- Historical Prediction Results ---")
        print(f"Note: Using historical model with averaged features")
        print(f"Most Likely Outcome: {outcome_map[prediction]}")
        print(f"\nProbabilities:")
        print(f"  Home Win ({home_team}): {probabilities[2]*100:.2f}%")
        print(f"  Draw: {probabilities[1]*100:.2f}%")
        print(f"  Away Win ({away_team}): {probabilities[0]*100:.2f}%")
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'season': season,
            'prediction': outcome_map[prediction],
            'probabilities': {
                'home_win': float(probabilities[2]),
                'draw': float(probabilities[1]),
                'away_win': float(probabilities[0])
            }
        }

    def interactive_prediction(self):
        """Interactive command-line interface"""
        print("\n" + "="*70)
        print("Premier League Match Prediction System")
        print("="*70)
        print(f"\nAvailable teams ({len(self.all_teams)}):")
        for i, team in enumerate(self.all_teams, 1):
            print(f"{i:2d}. {team}")
        
        while True:
            print("\n" + "-"*70)
            
            season = input("\nEnter season (e.g., 202526) [default: 202526]: ").strip() or '202526'
            home_team = input("Enter home team name: ").strip()
            away_team = input("Enter away team name: ").strip()
            
            if not home_team or not away_team:
                print("Error: Both team names are required")
                continue
            
            self.predict_match(home_team, away_team, season)
            
            continue_choice = input("\nPredict another match? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nThank you for using the prediction system!")
                break

if __name__ == "__main__":
    predictor = MatchPredictor()
    predictor.interactive_prediction()