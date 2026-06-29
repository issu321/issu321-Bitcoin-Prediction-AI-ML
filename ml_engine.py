import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class BitcoinPredictionEngine:
    """
    Advanced Bitcoin Price Prediction Engine using Machine Learning
    Features: Technical Indicators, Feature Engineering, Multiple Models
    """

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.feature_columns = []
        self.target_column = 'Close'
        self.metrics = {}
        self.is_trained = False

    def generate_sample_data(self, days=365):
        """Generate realistic Bitcoin price data for demonstration"""
        np.random.seed(42)

        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq='D')

        # Base price with trend and volatility
        base_price = 45000
        trend = np.linspace(0, 15000, days)

        # Add cyclical components (weekly, monthly patterns)
        weekly_cycle = 2000 * np.sin(2 * np.pi * np.arange(days) / 7)
        monthly_cycle = 3000 * np.sin(2 * np.pi * np.arange(days) / 30)

        # Add random walk
        random_walk = np.cumsum(np.random.normal(0, 800, days))

        # Add volatility clusters
        volatility = np.random.normal(0, 1500, days)

        close_prices = base_price + trend + weekly_cycle + monthly_cycle + random_walk + volatility
        close_prices = np.maximum(close_prices, 10000)  # Minimum price floor

        # Generate OHLCV data
        data = pd.DataFrame({
            'Date': dates,
            'Open': close_prices * (1 + np.random.normal(0, 0.01, days)),
            'High': close_prices * (1 + np.abs(np.random.normal(0, 0.03, days))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0, 0.03, days))),
            'Close': close_prices,
            'Volume': np.random.randint(10000000, 50000000, days) + np.random.normal(0, 5000000, days)
        })

        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1) * 1.01)
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1) * 0.99)
        data['Volume'] = np.maximum(data['Volume'], 1000000)

        return data

    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Abs'] = df['Price_Change'].abs()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100

        # Moving Averages
        for window in [5, 10, 20, 50, 100]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()

        # Lag features
        for lag in [1, 3, 5, 7]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Close_Skew_{window}'] = df['Close'].rolling(window=window).skew()

        # Time features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['DaysFromStart'] = (df['Date'] - df['Date'].min()).dt.days

        return df

    def prepare_features(self, df):
        """Prepare feature matrix and target vector"""
        df = self.calculate_technical_indicators(df)

        # Select feature columns (exclude non-numeric and target)
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        # Drop rows with NaN values
        df_clean = df.dropna()

        X = df_clean[self.feature_columns]
        y = df_clean['Close']

        return X, y, df_clean

    def train_model(self, df, test_size=0.2):
        """Train the prediction model"""
        X, y, df_clean = self.prepare_features(df)

        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=100, gamma='scale')
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        self.metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(self.feature_columns)
        }

        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.metrics['feature_importance'] = importance_df.to_dict('records')

        self.is_trained = True
        self.last_prices = y_test.values
        self.last_predictions = y_test_pred
        self.test_dates = df_clean['Date'].iloc[split_idx:].values

        return self.metrics

    def predict_future(self, df, days=7):
        """Predict future prices"""
        if not self.is_trained:
            self.train_model(df)

        df = self.calculate_technical_indicators(df)
        df_clean = df.dropna()

        # Get last row for prediction
        last_row = df_clean.iloc[-1:].copy()
        predictions = []
        prediction_dates = []

        current_date = pd.to_datetime(last_row['Date'].values[0])
        current_close = last_row['Close'].values[0]

        for i in range(days):
            # Update date
            current_date += timedelta(days=1)
            prediction_dates.append(current_date)

            # Prepare features for prediction
            features = last_row[self.feature_columns].values
            features_scaled = self.scaler.transform(features)

            # Predict
            pred_price = self.model.predict(features_scaled)[0]
            predictions.append(pred_price)

            # Update last row for next prediction (simple approach)
            current_close = pred_price

        return {
            'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
            'predictions': [round(p, 2) for p in predictions],
            'current_price': round(df_clean['Close'].iloc[-1], 2),
            'confidence_range': [round(p * 0.95, 2) for p in predictions],
            'confidence_range_high': [round(p * 1.05, 2) for p in predictions]
        }

    def generate_charts(self, df, predictions=None, save_path=None):
        """Generate comprehensive analysis charts"""
        df = self.calculate_technical_indicators(df)
        df_clean = df.dropna()

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Bitcoin Price Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Price Chart with Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(df_clean['Date'], df_clean['Close'], label='Close Price', color='#00d4ff', linewidth=1.5)
        ax1.plot(df_clean['Date'], df_clean['MA_20'], label='MA 20', color='#ff6b6b', alpha=0.8)
        ax1.plot(df_clean['Date'], df_clean['MA_50'], label='MA 50', color='#51cf66', alpha=0.8)
        if predictions:
            pred_dates = pd.to_datetime(predictions['dates'])
            ax1.plot(pred_dates, predictions['predictions'], 'r--', label='Predicted', linewidth=2)
            ax1.fill_between(pred_dates, predictions['confidence_range'], 
                           predictions['confidence_range_high'], alpha=0.2, color='red')
        ax1.set_title('Price Trend & Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Volume
        ax2 = axes[0, 1]
        colors = ['#51cf66' if c >= o else '#ff6b6b' 
                  for c, o in zip(df_clean['Close'], df_clean['Open'])]
        ax2.bar(df_clean['Date'], df_clean['Volume'], color=colors, alpha=0.7, width=1)
        ax2.set_title('Trading Volume')
        ax2.grid(True, alpha=0.3)

        # 3. RSI
        ax3 = axes[1, 0]
        ax3.plot(df_clean['Date'], df_clean['RSI'], color='#ffd43b', linewidth=1.5)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.fill_between(df_clean['Date'], 30, 70, alpha=0.1, color='gray')
        ax3.set_title('RSI (Relative Strength Index)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. MACD
        ax4 = axes[1, 1]
        ax4.plot(df_clean['Date'], df_clean['MACD'], label='MACD', color='#00d4ff', linewidth=1.5)
        ax4.plot(df_clean['Date'], df_clean['MACD_Signal'], label='Signal', color='#ff6b6b', linewidth=1.5)
        colors_macd = ['#51cf66' if val >= 0 else '#ff6b6b' for val in df_clean['MACD_Histogram']]
        ax4.bar(df_clean['Date'], df_clean['MACD_Histogram'], color=colors_macd, alpha=0.7, width=1)
        ax4.set_title('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Bollinger Bands
        ax5 = axes[2, 0]
        ax5.plot(df_clean['Date'], df_clean['Close'], label='Close', color='#00d4ff', linewidth=1.5)
        ax5.plot(df_clean['Date'], df_clean['BB_Upper'], label='Upper Band', color='#ff6b6b', alpha=0.7)
        ax5.plot(df_clean['Date'], df_clean['BB_Lower'], label='Lower Band', color='#51cf66', alpha=0.7)
        ax5.fill_between(df_clean['Date'], df_clean['BB_Lower'], df_clean['BB_Upper'], alpha=0.1, color='gray')
        ax5.set_title('Bollinger Bands')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Feature Importance (if available)
        ax6 = axes[2, 1]
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            ax6.barh(importance_df['feature'], importance_df['importance'], color='#00d4ff')
            ax6.set_title('Top 10 Feature Importance')
        else:
            ax6.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                    ha='center', va='center', fontsize=14)
            ax6.set_title('Feature Importance')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close()
            return save_path
        else:
            plt.savefig(f'{project_dir}/static/images/analysis_chart.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close()
            return f'{project_dir}/static/images/analysis_chart.png'

    def analyze_csv(self, file_path):
        """Analyze uploaded CSV file"""
        try:
            df = pd.read_csv(file_path)

            analysis = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': {},
                'correlation_matrix': {},
                'sample_data': df.head(10).to_dict('records')
            }

            # Numeric summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                analysis['numeric_summary'][col] = {
                    'mean': round(float(df[col].mean()), 4),
                    'std': round(float(df[col].std()), 4),
                    'min': round(float(df[col].min()), 4),
                    'max': round(float(df[col].max()), 4),
                    'median': round(float(df[col].median()), 4)
                }

            # Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                analysis['correlation_matrix'] = corr.round(3).to_dict()

            return analysis

        except Exception as e:
            return {'error': str(e)}


def get_model_description(model_type):
    """Get description of ML model"""
    descriptions = {
        'random_forest': {
            'name': 'Random Forest Regressor',
            'description': 'An ensemble learning method that constructs multiple decision trees and outputs the average prediction. Excellent for capturing non-linear relationships in Bitcoin price data.',
            'pros': ['Handles non-linear data well', 'Robust to outliers', 'Provides feature importance', 'Reduces overfitting'],
            'cons': ['Can be slow with large datasets', 'Less interpretable than linear models'],
            'best_for': 'General purpose prediction with high accuracy'
        },
        'linear_regression': {
            'name': 'Linear Regression',
            'description': 'A statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. Fast and interpretable.',
            'pros': ['Fast training and prediction', 'Highly interpretable', 'Works well with linear trends', 'Low computational cost'],
            'cons': ['Assumes linear relationships', 'Sensitive to outliers', 'May underfit complex patterns'],
            'best_for': 'Quick baseline predictions and trend analysis'
        },
        'svr': {
            'name': 'Support Vector Regression',
            'description': 'Uses support vector machines for regression tasks. Effective in high-dimensional spaces and uses a subset of training points (support vectors) for prediction.',
            'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile with different kernels'],
            'cons': ['Sensitive to feature scaling', 'Can be slow on large datasets', 'Requires careful parameter tuning'],
            'best_for': 'Complex patterns with limited training data'
        }
    }
    return descriptions.get(model_type, descriptions['random_forest'])
