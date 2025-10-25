# backend/MLModelTraining/train_models.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import warnings
import json
warnings.filterwarnings('ignore')

class EnsembleModelTrainer:
    def __init__(self, data_dir='data/files/MLData'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path('data/files/MLModelTraining/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_accuracy = 0.90
        self.min_f1_score = 0.90
        self.max_retrain_attempts = 5

    def load_features(self, feature_type):
        """Load pre-processed features (already cleaned of identifiers)"""
        data_path = self.data_dir / f'{feature_type}_training.csv'
        df = pd.read_csv(data_path)
        print(f"Loaded {feature_type} data: {len(df)} samples, {len(df.columns)} columns")
        return df

    def prepare_data_splits(self, df, test_size=0.15, val_size=0.15):
        """70-15-15 split with no identifiers"""
        # Separate features and target
        X = df.drop(columns=['FTR'])
        y = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})

        # Verify no identifiers
        identifier_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'TeamID']
        for col in identifier_cols:
            if col in X.columns:
                raise ValueError(f"Identifier column '{col}' found in features! Remove from feature engineering.")

        print(f"Feature columns ({len(X.columns)}): {list(X.columns[:5])}...")

        # Handle NaN
        X = X.fillna(0)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # 70-30 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # 15-15 split from temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        print(f"Split - Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, X.columns.tolist()

    def train_with_validation(self, X_train, X_val, y_train, y_val):
        """Train until requirements met"""
        best_accuracy = 0
        best_f1 = 0
        best_ensemble = None
        best_params = {}

        for attempt in range(self.max_retrain_attempts):
            print(f"\n--- Attempt {attempt + 1}/{self.max_retrain_attempts} ---")

            # Train LR
            lr_configs = [
                {'C': 0.01, 'max_iter': 2000, 'class_weight': 'balanced'},
                {'C': 0.1, 'max_iter': 2000, 'class_weight': 'balanced'},
                {'C': 1.0, 'max_iter': 2000, 'class_weight': None},
                {'C': 10.0, 'max_iter': 2000, 'class_weight': None}
            ]

            best_lr = None
            best_lr_score = 0
            for config in lr_configs:
                lr = LogisticRegression(random_state=42 + attempt, **config)
                lr.fit(X_train, y_train)
                score = lr.score(X_val, y_val)
                if score > best_lr_score:
                    best_lr_score = score
                    best_lr = lr
            print(f"Best LR val accuracy: {best_lr_score:.4f}")

            # Train KNN
            knn_configs = [
                {'n_neighbors': 5, 'weights': 'uniform'},
                {'n_neighbors': 7, 'weights': 'distance'},
                {'n_neighbors': 11, 'weights': 'distance'},
                {'n_neighbors': 15, 'weights': 'distance'}
            ]

            best_knn = None
            best_knn_score = 0
            for config in knn_configs:
                knn = KNeighborsClassifier(**config)
                knn.fit(X_train, y_train)
                score = knn.score(X_val, y_val)
                if score > best_knn_score:
                    best_knn_score = score
                    best_knn = knn
            print(f"Best KNN val accuracy: {best_knn_score:.4f}")

            # Train SVM
            svm_configs = [
                {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 10.0, 'kernel': 'rbf', 'gamma': 'auto'},
                {'C': 1.0, 'kernel': 'poly', 'degree': 3}
            ]

            best_svm = None
            best_svm_score = 0
            for config in svm_configs:
                svm = SVC(probability=True, random_state=42 + attempt, class_weight='balanced', **config)
                svm.fit(X_train, y_train)
                score = svm.score(X_val, y_val)
                if score > best_svm_score:
                    best_svm_score = score
                    best_svm = svm
            print(f"Best SVM val accuracy: {best_svm_score:.4f}")

            # Ensemble
            ensemble = VotingClassifier(
                estimators=[('lr', best_lr), ('knn', best_knn), ('svm', best_svm)],
                voting='soft'
            )
            ensemble.fit(X_train, y_train)

            val_pred = ensemble.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')

            print(f"Ensemble - Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}")

            if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_f1 > best_f1):
                best_accuracy = val_accuracy
                best_f1 = val_f1
                best_ensemble = ensemble
                best_params = {
                    'lr_params': best_lr.get_params(),
                    'knn_params': best_knn.get_params(),
                    'svm_params': best_svm.get_params(),
                    'val_accuracy': float(val_accuracy),
                    'val_f1_score': float(val_f1),
                    'attempt': attempt + 1
                }

            if val_accuracy >= self.min_accuracy and val_f1 >= self.min_f1_score:
                print(f"\n✓ Requirements met!")
                break

        if best_accuracy < self.min_accuracy or best_f1 < self.min_f1_score:
            print(f"\n⚠ Warning: Best model below requirements")
            print(f"  Acc: {best_accuracy:.4f} (need {self.min_accuracy})")
            print(f"  F1: {best_f1:.4f} (need {self.min_f1_score})")

        return best_ensemble, best_params

    def train_model(self, model_name):
        """Train a single model type"""
        print(f"\n{'='*70}")
        print(f"=== Training {model_name.title()} Model ===")
        print(f"{'='*70}")

        df = self.load_features(model_name)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = self.prepare_data_splits(df)

        ensemble, params = self.train_with_validation(X_train, X_val, y_train, y_val)

        # Test evaluation
        print("\n--- Test Set Evaluation ---")
        test_pred = ensemble.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print("\n" + classification_report(y_test, test_pred, target_names=['Away', 'Draw', 'Home']))

        # Save
        model_path = self.model_dir / f'{model_name}_ensemble.pkl'
        scaler_path = self.model_dir / f'{model_name}_scaler.pkl'
        features_path = self.model_dir / f'{model_name}_features.pkl'
        params_path = self.model_dir / f'{model_name}_params.json'

        joblib.dump(ensemble, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path)

        params['test_accuracy'] = float(test_acc)
        params['test_f1_score'] = float(test_f1)
        params['num_features'] = len(features)

        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"\n✓ Saved to {self.model_dir}/")

    def train_all_models(self):
        """Train both models"""
        self.train_model('current_season')
        self.train_model('historical')

        print(f"\n{'='*70}")
        print("✓ All models trained successfully!")
        print(f"{'='*70}")

if __name__ == "__main__":
    trainer = EnsembleModelTrainer()
    trainer.train_all_models()