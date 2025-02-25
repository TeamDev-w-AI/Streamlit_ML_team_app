# models/rf_xgb.py

# Core data science libraries
import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Visualization libraries
import plotly.graph_objects as go

# Streamlit for web interface
import streamlit as st


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.training_summary = {}

    def create_features(self, data, seq_length):
        df = pd.DataFrame(data)
        for i in range(1, seq_length + 1):
            df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
        return df.dropna()

    def prepare_data(self, data, seq_length):
        features = self.create_features(data, seq_length)
        X = features.drop(features.columns[0], axis=1)
        y = features.iloc[:, 0]
        return X, y

    def train(self, X, y, **kwargs):
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self._update_training_summary(X_scaled, y)
            return self.model
        except Exception as e:
            st.error(f"Random Forest training error: {str(e)}")
            return None

    def predict(self, X, return_conf_int=True, alpha=0.05):
        try:
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)

            if return_conf_int:
                # Calculate prediction intervals using bootstrapping
                predictions = np.array([tree.predict(X_scaled)
                                        for tree in self.model.estimators_])
                lower = np.percentile(predictions, alpha / 2 * 100, axis=0)
                upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)
                return pred, (lower, upper)
            return pred
        except Exception as e:
            st.error(f"Random Forest prediction error: {str(e)}")
            return None

    def _update_training_summary(self, X, y):
        predictions = self.model.predict(X)
        self.training_summary = {
            'feature_importance': pd.Series(
                self.model.feature_importances_,
                index=range(X.shape[1])
            ),
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None,
            'residuals': y - predictions
        }


class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        self.scaler = StandardScaler()
        self.training_summary = {}

    def create_features(self, data, seq_length):
        df = pd.DataFrame(data)
        for i in range(1, seq_length + 1):
            df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
        return df.dropna()

    def prepare_data(self, data, seq_length):
        features = self.create_features(data, seq_length)
        X = features.drop(features.columns[0], axis=1)
        y = features.iloc[:, 0]
        return X, y

    def train(self, X, y, eval_set=None, **kwargs):
        try:
            X_scaled = self.scaler.fit_transform(X)
            eval_set_scaled = [(self.scaler.transform(eval_set[0]), eval_set[1])] if eval_set else None

            self.model.fit(
                X_scaled, y,
                eval_set=eval_set_scaled,
                early_stopping_rounds=10,
                verbose=False
            )
            self._update_training_summary(X_scaled, y)
            return self.model
        except Exception as e:
            st.error(f"XGBoost training error: {str(e)}")
            return None

    def predict(self, X, return_conf_int=True, alpha=0.05):
        try:
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)

            if return_conf_int:
                # Use quantile regression for confidence intervals
                lower = self.model.predict(X_scaled, ntree_limit=self.model.best_ntree_limit)
                upper = lower + 2 * np.std(self.training_summary['residuals'])
                return pred, (lower, upper)
            return pred
        except Exception as e:
            st.error(f"XGBoost prediction error: {str(e)}")
            return None

    def _update_training_summary(self, X, y):
        predictions = self.model.predict(X)
        self.training_summary = {
            'feature_importance': pd.Series(
                self.model.feature_importances_,
                index=range(X.shape[1])
            ),
            'best_score': self.model.best_score if hasattr(self.model, 'best_score') else None,
            'residuals': y - predictions
        }


def plot_feature_importance(model, feature_names=None):
    importance = model.training_summary['feature_importance']
    if feature_names:
        importance.index = feature_names

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance.index,
        x=importance.values,
        orientation='h'
    ))

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        template='plotly_white'
    )

    return fig