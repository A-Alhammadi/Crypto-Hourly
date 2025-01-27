import pandas as pd
import numpy as np
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
    @staticmethod
    def adaptive_strategy(df):
        """
        Enhanced adaptive strategy with performance-based selection and risk management.
        """
        signals = pd.Series(index=df.index, data=0)
        
        # Get signals from all base strategies
        strategy_signals = {
            'ema': TradingStrategies.ema_strategy(df),
            'macd': TradingStrategies.macd_strategy(df),
            'rsi': TradingStrategies.rsi_strategy(df),
            'stoch': TradingStrategies.stochastic_strategy(df),
            'volume_rsi': TradingStrategies.volume_rsi_strategy(df),
            'vwap': TradingStrategies.vwap_strategy(df)
        }
        
        # Calculate market regime indicators
        # Volatility using ATR
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['close_price'].shift())
        low_close = abs(df['low_price'] - df['close_price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        volatility = atr / df['close_price']
        
        # Pre-calculate volatility regimes for the entire series
        vol_quantiles = volatility.quantile([0.33, 0.66])
        vol_regimes = pd.Series(index=volatility.index, data='medium')
        vol_regimes[volatility <= vol_quantiles[0.33]] = 'low'
        vol_regimes[volatility > vol_quantiles[0.66]] = 'high'
        
        # Enhanced trend strength indicators
        short_trend = df['ema_9'] / df['ema_21'] - 1
        medium_trend = df['ema_21'] / df['ema_50'] - 1
        trend_strength = (abs(short_trend) + abs(medium_trend)) / 2
        trend_direction = np.sign(short_trend + medium_trend)
        
        # Volume and momentum indicators
        relative_volume = df['volume_crypto'] / df['volume_crypto'].rolling(window=20).mean()
        momentum = df['close_price'].pct_change(5)
        
        # Calculate recent performance of each strategy
        lookback = 20  # Days to look back for performance evaluation
        strategy_performance = {}
        
        for name, strat_signals in strategy_signals.items():
            # Calculate returns if we followed each strategy
            returns = df['close_price'].pct_change() * strat_signals.shift()
            # Calculate rolling Sharpe ratio for each strategy
            rolling_sharpe = (returns.rolling(lookback).mean() / returns.rolling(lookback).std()) * np.sqrt(252)
            strategy_performance[name] = rolling_sharpe
        
        # Convert to DataFrame for easier manipulation
        performance_df = pd.DataFrame(strategy_performance)
        
        # Function to get optimal strategy based on market conditions and recent performance
        def get_optimal_strategy(i, vol_regime, trend_str, trend_dir, rel_vol, mom, perf_df):
            # Get top 2 performing strategies
            if i >= lookback:
                recent_performance = perf_df.iloc[i]
                top_strategies = recent_performance.nlargest(2).index.tolist()
            else:
                top_strategies = ['ema', 'macd']  # Default for initial period
            
            # Strategy selection logic based on market conditions and performance
            if vol_regime == 'high':
                if trend_dir > 0:
                    primary = 'macd' if 'macd' in top_strategies else top_strategies[0]
                    secondary = 'rsi'
                else:
                    primary = 'rsi' if 'rsi' in top_strategies else top_strategies[0]
                    secondary = 'vwap'
            elif vol_regime == 'low':
                if trend_dir > 0:
                    primary = 'ema' if 'ema' in top_strategies else top_strategies[0]
                    secondary = 'macd'
                else:
                    primary = 'vwap' if 'vwap' in top_strategies else top_strategies[0]
                    secondary = 'volume_rsi'
            else:  # medium volatility
                if trend_str > trend_strength.quantile(0.7):  # Strong trend
                    primary = top_strategies[0]
                    secondary = 'macd'
                else:  # Weak trend
                    primary = 'rsi' if 'rsi' in top_strategies else top_strategies[0]
                    secondary = 'vwap'
            
            return primary, secondary
        
        # Generate adaptive signals
        for i in range(len(df)):
            # Get optimal strategies for current conditions
            primary_strat, secondary_strat = get_optimal_strategy(
                i, vol_regimes.iloc[i], trend_strength.iloc[i], trend_direction.iloc[i],
                relative_volume.iloc[i], momentum.iloc[i], performance_df
            )
            
            # Weight between primary and secondary strategies based on confidence
            if i >= lookback:
                primary_weight = 0.7  # Higher weight to primary strategy
                secondary_weight = 0.3
                
                # Combine signals with weights
                combined_signal = (
                    strategy_signals[primary_strat].iloc[i] * primary_weight +
                    strategy_signals[secondary_strat].iloc[i] * secondary_weight
                )
                
                # Apply thresholds for final signal
                if combined_signal > 0.5:
                    signals.iloc[i] = 1
                elif combined_signal < -0.5:
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0
            else:
                # Use EMA strategy for initial period
                signals.iloc[i] = strategy_signals['ema'].iloc[i]
        
        # Apply risk management rules
        
        # 1. No trading during extreme volatility
        extreme_vol = volatility > volatility.quantile(0.95)
        signals[extreme_vol] = 0
        
        # 2. Trend confirmation
        # Only take long positions in uptrends and short positions in downtrends
        signals[(trend_direction < 0) & (signals == 1)] = 0
        signals[(trend_direction > 0) & (signals == -1)] = 0
        
        # 3. Volume confirmation
        # Don't trade on very low volume
        low_vol = relative_volume < relative_volume.quantile(0.2)
        signals[low_vol] = 0
        
        return signals
    @classmethod
    def get_all_strategies(cls):
        return {
            'EMA': cls.ema_strategy,
            'MACD': cls.macd_strategy,
            'RSI': cls.rsi_strategy,
            'Stochastic': cls.stochastic_strategy,
            'Volume RSI': cls.volume_rsi_strategy,
            'VWAP': cls.vwap_strategy,
            'Adaptive': cls.adaptive_strategy
        }