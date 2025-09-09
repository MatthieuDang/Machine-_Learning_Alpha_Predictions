# Machine_Learning_Alpha_Predictions
This project explores short-term alpha prediction using machine learning on stock price data. It predicts whether a stock’s price will go up or down in the next 5 trading days, based on technical and statistical features.
Features
Data Source: Yahoo Finance (yfinance)
Assets: AAPL, GOOG, PEP (extendable to others)
Feature Engineering:
Daily, 5-day, and 10-day returns
Rolling volatility (20-day)
RSI (14-day momentum indicator)
Moving averages (MA50, MA200)
Lagged returns for autocorrelation effects
Model: Random Forest Classifier with class balancing
Evaluation: Accuracy, classification report, confusion matrix, feature importance
