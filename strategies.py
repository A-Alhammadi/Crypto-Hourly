import pandas as pd
from config import BACKTEST_CONFIG

class TradingStrategies:
    @staticmethod
    def ema_strategy(df):
        signals = pd.Series(index=df.index, data=0)
        
        short = BACKTEST_CONFIG['ema']['short']
        medium = BACKTEST_CONFIG['ema']['medium']
        
        # Generate buy signal when short EMA crosses above medium EMA
        signals[df[f'ema_{short}'] > df[f'ema_{medium}']] = 1
        # Generate sell signal when short EMA crosses below medium EMA
        signals[df[f'ema_{short}'] < df[f'ema_{medium}']] = -1
        
        return signals

    @staticmethod
    def macd_strategy(df):
        signals = pd.Series(index=df.index, data=0)
        
        # Buy when MACD line crosses above signal line
        signals[df['macd_line'] > df['signal_line']] = 1
        # Sell when MACD line crosses below signal line
        signals[df['macd_line'] < df['signal_line']] = -1
        
        return signals

    @staticmethod
    def rsi_strategy(df):
        signals = pd.Series(index=df.index, data=0)
        config = BACKTEST_CONFIG['rsi']
        
        # Buy when RSI crosses above oversold level
        signals[df['rsi'] < config['oversold']] = 1
        # Sell when RSI crosses above overbought level
        signals[df['rsi'] > config['overbought']] = -1
        
        return signals

    @staticmethod
    def stochastic_strategy(df):
        signals = pd.Series(index=df.index, data=0)
        config = BACKTEST_CONFIG['stochastic']
        
        # Buy when both %K and %D are below oversold level
        signals[(df['stoch_k'] < config['oversold']) & 
               (df['stoch_d'] < config['oversold'])] = 1
        
        # Sell when both %K and %D are above overbought level
        signals[(df['stoch_k'] > config['overbought']) & 
               (df['stoch_d'] > config['overbought'])] = -1
        
        return signals

    @staticmethod
    def vwap_strategy(df):
        """
        VWAP trading strategy:
        - Buy when price drops below VWAP by oversold threshold
        - Sell when price rises above VWAP by overbought threshold
        """
        print("\nExecuting VWAP strategy...")
        print("Available columns:", df.columns.tolist())
        print("First few rows of price and VWAP:")
        print(df[['close_price', 'vwap']].head())
        
        signals = pd.Series(index=df.index, data=0)
        config = BACKTEST_CONFIG['vwap']
        
        try:
            # Buy when price is below VWAP by oversold threshold
            price_to_vwap = df['close_price'] / df['vwap']
            signals[price_to_vwap < config['oversold']] = 1
            
            # Sell when price is above VWAP by overbought threshold
            signals[price_to_vwap > config['overbought']] = -1
            
            print("VWAP signals generated successfully")
            print("Number of buy signals:", len(signals[signals == 1]))
            print("Number of sell signals:", len(signals[signals == -1]))
            
        except Exception as e:
            print(f"Error in VWAP strategy: {str(e)}")
            raise
        
        return signals

    @staticmethod
    def volume_rsi_strategy(df):
        signals = pd.Series(index=df.index, data=0)
        config = BACKTEST_CONFIG['volume_rsi']
        
        # Buy when Volume RSI is below oversold level
        signals[df['volume_rsi'] < config['oversold']] = 1
        # Sell when Volume RSI is above overbought level
        signals[df['volume_rsi'] > config['overbought']] = -1
        
        return signals

    @classmethod
    def get_all_strategies(cls):
        return {
            'EMA': cls.ema_strategy,
            'MACD': cls.macd_strategy,
            'RSI': cls.rsi_strategy,
            'Stochastic': cls.stochastic_strategy,
            'Volume RSI': cls.volume_rsi_strategy,
            'VWAP': cls.vwap_strategy
        }