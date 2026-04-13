# test_setup.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("✅ All packages imported successfully!")

# Test data download
test_stock = yf.download("AAPL", start="2024-01-01", end="2024-01-10")
print(f"✅ Downloaded {len(test_stock)} days of AAPL data")
