import yfinance as yf
import numpy as np
import pandas as pd

def calibrate_market_params(ticker="SPY", period="5y"):
    print(f"📡 Fetching data for {ticker} over {period}...")
    
    # Explicitly set auto_adjust=True so 'Close' is always the Adjusted Close
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    
    if len(df) == 0:
        print("❌ Error: No data found. Check your internet or ticker symbol.")
        return

    print(f"✅ Loaded {len(df)} days of market data.")

    # Fix for MultiIndex columns (Common in new yfinance versions)
    # If columns look like ('Close', 'SPY'), flatten them to just 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Calculate Log Returns
    # Formula: ln(Price_t / Price_t-1)
    # NOW USING 'Close' because it is already adjusted
    try:
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    except KeyError:
        print("❌ Error: 'Close' column not found. Columns are:", df.columns)
        return

    # Drop the first NaN value
    df = df.dropna()

    # 3. Calculate Annualized Volatility (Sigma)
    # Standard Deviation of daily returns * Sqrt(Trading Days per Year)
    daily_std = df['Log_Ret'].std()
    
    # Ensure we get a scalar float, not a Series
    if isinstance(daily_std, pd.Series):
        daily_std = daily_std.item()

    annualized_sigma = daily_std * np.sqrt(252)
    
    # 4. Calculate Annualized Drift (Mu)
    daily_mean = df['Log_Ret'].mean()
    if isinstance(daily_mean, pd.Series):
        daily_mean = daily_mean.item()
        
    annualized_mu = (daily_mean * 252) + (0.5 * annualized_sigma**2)

    print("-" * 40)
    print(f"📊 CALIBRATION RESULTS ({ticker})")
    print("-" * 40)
    print(f"Realized Volatility (σ):  {annualized_sigma:.4f}  <-- USE THIS")
    print(f"Historical Drift (μ):     {annualized_mu:.4f}")
    print("-" * 40)
    
    return annualized_sigma

if __name__ == "__main__":
    calibrate_market_params()