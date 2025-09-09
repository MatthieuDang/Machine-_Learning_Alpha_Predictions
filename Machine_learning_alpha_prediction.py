# Refactored Machine Learning Alpha Predictor
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Download Data ---
tickers = ['AAPL', 'GOOG', 'PEP']
df = yf.download(tickers, start='2014-01-01',
                 end='2024-01-01', auto_adjust=True)
df.columns = [f'{col[1]} {col[0]}' for col in df.columns]

# --- Feature Engineering ---
close = df.filter(like='Close')
returns = close.pct_change()
volatility = close.rolling(window=20).std()
rsi = 100 - (100 / (1 + (returns.rolling(window=14).mean() /
             returns.rolling(window=14).std())))
ma_50 = close.rolling(window=50).mean()
ma_200 = close.rolling(window=200).mean()
past_5d_return = close.pct_change(5)
past_10d_return = close.pct_change(10)
lag_1d_return = returns.shift(1)

# --- Combine Features ---
features = pd.concat([
    returns,
    past_5d_return.add_suffix(' 5dR'),
    past_10d_return.add_suffix(' 10dR'),
    lag_1d_return.add_suffix(' Lag1d'),
    volatility.add_suffix(' Vol'),
    ma_50.add_suffix(' MA50'),
    ma_200.add_suffix(' MA200'),
    rsi.add_suffix(' RSI')
], axis=1).dropna()

# --- Labels: Predict 5-Day Forward Return Direction ---
future_return = close['AAPL Close'].pct_change(periods=5).shift(-5)
labels = (future_return > 0).astype(int)
features, labels = features.align(labels, join='inner', axis=0)

# --- Train-Test Split ---
X = features
y = labels
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Standardize Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Random Forest Model ---
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight=class_weight_dict)
model.fit(X_train_scaled, y_train)

# --- Predictions ---
y_pred = model.predict(X_test_scaled)

# --- Evaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Cross-validation Score ---
cross_val_scores = cross_val_score(
    model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(
    f"\nCross-Validation Accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")

# --- Feature Importance Plot ---
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh', title='Top 15 Feature Importances')
plt.tight_layout()
plt.show()
