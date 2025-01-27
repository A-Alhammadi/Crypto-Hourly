# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "cryptocurrencies",
    "user": "myuser",
    "password": "mypassword"
}

# Backtesting configuration
BACKTEST_CONFIG = {
    # Date range
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    
    # Trading pairs to test
    "symbols": ["BTC/USD", "ETH/USD", "XRP/USD", "AAVE/USD",
                 "ADA/USD", "ALGO/USD", "AVAX/USD", "BCH/USD",
                 "HBAR/USD", "LINK/USD", "LTC/USD", "UNI/USD","XLR/USD"],  # Add all currencies you want to test
    
    # Initial capital for each currency
    "initial_capital": 10000,
    
    # Position size (percentage of capital)
    "position_size": 0.95,  # 95% of capital
    
    # Technical indicators parameters
    "vwap": {
        "period": 24,  # 24 hours for daily VWAP
        "overbought": 1.02,  # 2% above VWAP
        "oversold": 0.98     # 2% below VWAP
    },
    "ema": {
        "short": 9,
        "medium": 21,
        "long": 50
    },
    
    "macd": {
        "fast": 12,
        "slow": 26,
        "signal": 9
    },
    
    "rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "overbought": 80,
        "oversold": 20
    },
    
    "volume_rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    
    # Trading fees
    "trading_fee": 0,  # 0.001 = 0.1%
    
    # Output configuration
    "results_dir": "backtest_results",  # Directory to save results
    "save_trades": True,  # Whether to save detailed trade information
    "save_plots": True   # Whether to save plots as PNG files
}