import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

print("✅ All imports successful!")

# Download 1 week of test data
data = yf.download("AAPL", period="5d")
print(f"✅ Downloaded {len(data)} days of data")
print("\nYou're ready to start! 🚀")
