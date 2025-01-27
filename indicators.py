import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class TechnicalIndicators:
    @staticmethod
    def add_ema(df):
        config = BACKTEST_CONFIG['ema']
        
        df[f'ema_{config["short"]}'] = df['close_price'].ewm(span=config['short'], adjust=False).mean()
        df[f'ema_{config["medium"]}'] = df['close_price'].ewm(span=config['medium'], adjust=False).mean()
        df[f'ema_{config["long"]}'] = df['close_price'].ewm(span=config['long'], adjust=False).mean()
        
        return df

    @staticmethod
    def add_macd(df):
        config = BACKTEST_CONFIG['macd']
        
        # Calculate MACD line
        exp1 = df['close_price'].ewm(span=config['fast'], adjust=False).mean()
        exp2 = df['close_price'].ewm(span=config['slow'], adjust=False).mean()
        df['macd_line'] = exp1 - exp2
        
        # Calculate Signal line
        df['signal_line'] = df['macd_line'].ewm(span=config['signal'], adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_histogram'] = df['macd_line'] - df['signal_line']
        
        return df

    @staticmethod
    def add_rsi(df):
        config = BACKTEST_CONFIG['rsi']
        
        # Calculate price changes
        delta = df['close_price'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=config['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config['period']).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def add_stochastic(df):
        config = BACKTEST_CONFIG['stochastic']
        
        # Calculate %K
        lowest_low = df['low_price'].rolling(window=config['k_period']).min()
        highest_high = df['high_price'].rolling(window=config['k_period']).max()
        df['stoch_k'] = 100 * (df['close_price'] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=config['d_period']).mean()
        
        return df

    @staticmethod
    def add_volume_rsi(df):
        config = BACKTEST_CONFIG['volume_rsi']
        
        # Calculate volume changes
        delta = df['volume_crypto'].diff()
        
        # Separate increases and decreases
        gain = (delta.where(delta > 0, 0)).rolling(window=config['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config['period']).mean()
        
        # Calculate RS and Volume RSI
        rs = gain / loss
        df['volume_rsi'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def add_vwap(df):
        """
        Calculate VWAP (Volume Weighted Average Price)
        VWAP = Σ(Price * Volume) / Σ(Volume)
        """
        print("\nCalculating VWAP...")
        config = BACKTEST_CONFIG['vwap']
        
        # Start fresh each day
        df = df.copy()
        df['date'] = df.index.date
        
        # Group by date and calculate VWAP
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['pv'] = df['typical_price'] * df['volume_crypto']
        
        # Calculate running sum for each day
        df['cum_pv'] = df.groupby('date')['pv'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume_crypto'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cum_pv'] / df['cum_volume']
        
        # Clean up temporary columns
        df.drop(['typical_price', 'pv', 'cum_pv', 'cum_volume', 'date'], axis=1, inplace=True)
        
        print("VWAP calculated successfully")
        print("First few rows of close price and VWAP:")
        print(df[['close_price', 'vwap']].head())
        
        return df

    @staticmethod
    def add_market_characteristics(df):
        """Add various market characteristics for correlation analysis"""
        
        # Volatility (20-period standard deviation of returns)
        df['volatility'] = df['close_price'].pct_change().rolling(window=20).std()
        
        # Trend Strength (ADX - Average Directional Index)
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['close_price'].shift())
        low_close = abs(df['low_price'] - df['close_price'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()  # Average True Range
        
        # Volume characteristics
        df['volume_ma'] = df['volume_crypto'].rolling(window=20).mean()
        df['relative_volume'] = df['volume_crypto'] / df['volume_ma']
        
        # Price momentum
        df['momentum'] = df['close_price'].pct_change(periods=5)
        
        # Trend direction (using 20-period SMA slope)
        sma20 = df['close_price'].rolling(window=20).mean()
        df['trend_direction'] = sma20.diff(periods=5) / sma20
        
        return df

    @classmethod
    def add_all_indicators(cls, df):
        print("\nAdding all indicators...")
        print("Initial DataFrame columns:", df.columns.tolist())
        
        # Add market characteristics first
        df = cls.add_market_characteristics(df.copy())
        
        # Add each indicator
        df = cls.add_ema(df.copy())
        df = cls.add_macd(df.copy())
        df = cls.add_rsi(df.copy())
        df = cls.add_stochastic(df.copy())
        df = cls.add_volume_rsi(df.copy())
        df = cls.add_vwap(df.copy())
        
        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        
        print("Final DataFrame columns:", df.columns.tolist())
        return df