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
        Enhanced adaptive strategy optimized for performance while maintaining exact same logic.
        """
        signals = pd.Series(index=df.index, data=0.0)
        
        # Get signals from all base strategies
        strategy_signals = {
            'ema': TradingStrategies.ema_strategy(df),
            'macd': TradingStrategies.macd_strategy(df),
            'rsi': TradingStrategies.rsi_strategy(df),
            'stoch': TradingStrategies.stochastic_strategy(df),
            'volume_rsi': TradingStrategies.volume_rsi_strategy(df),
            'vwap': TradingStrategies.vwap_strategy(df)
        }
        
        # Pre-calculate market indicators vectorially
        price = df['close_price']
        returns = price.pct_change()
        returns_vol = returns.rolling(window=20).std().fillna(0)
        high_low_vol = ((df['high_price'] - df['low_price']) / df['close_price']).rolling(window=20).mean().fillna(0)
        
        # Trend indicators
        ema_short = df['ema_9']
        ema_med = df['ema_21']
        ema_long = df['ema_50']
        
        trend_strength = abs((ema_short - ema_long) / ema_long).fillna(0)
        trend_direction = np.sign(ema_short - ema_long)
        
        # Volume analysis - vectorized
        volume = df['volume_crypto']
        volume_ma = volume.rolling(window=20).mean()
        relative_volume = (volume / volume_ma).fillna(1)
        
        # Dynamic lookback calculation - vectorized
        base_lookback = 20
        vol_ratio = returns_vol / returns_vol.rolling(100).mean()
        vol_ratio = vol_ratio.fillna(1).replace([np.inf, -np.inf], 1)
        vol_adjusted_lookback = (base_lookback * (1 + vol_ratio)).clip(10, 30).astype(int)
        
        # Pre-calculate performance metrics for all strategies at once
        strategy_metrics = {}
        
        for name, strat_signals in strategy_signals.items():
            # Calculate strategy returns
            strat_returns = returns * strat_signals.shift()
            
            # Initialize metric arrays
            rolling_sharpe = np.full(len(df), np.nan)
            rolling_sortino = np.full(len(df), np.nan)
            win_rate = np.full(len(df), np.nan)
            
            # Vectorized calculation for rolling windows
            for i in range(base_lookback, len(df)):
                lookback = vol_adjusted_lookback.iloc[i]
                period_returns = strat_returns.iloc[i-lookback:i]
                
                # Sharpe Ratio with safety checks
                returns_std = period_returns.std()
                if returns_std != 0 and not np.isnan(returns_std):
                    rolling_sharpe[i] = (period_returns.mean() / returns_std) * np.sqrt(252)
                else:
                    rolling_sharpe[i] = 0
                
                # Sortino Ratio with safety checks
                downside_returns = period_returns[period_returns < 0]
                downside_std = downside_returns.std()
                if len(downside_returns) > 0 and downside_std != 0 and not np.isnan(downside_std):
                    rolling_sortino[i] = (period_returns.mean() / downside_std) * np.sqrt(252)
                else:
                    rolling_sortino[i] = 0
                
                # Win Rate
                win_rate[i] = (period_returns > 0).mean()
            
            strategy_metrics[name] = {
                'sharpe': pd.Series(rolling_sharpe, index=df.index).fillna(0),
                'sortino': pd.Series(rolling_sortino, index=df.index).fillna(0),
                'win_rate': pd.Series(win_rate, index=df.index).fillna(0)
            }
        
        # Pre-calculate percentile ranks for market conditions
        vol_rank = returns_vol.rank(pct=True)
        trend_rank = trend_strength.rank(pct=True)
        volume_rank = relative_volume.rank(pct=True)
        
        # Function to get strategy weights (kept the same for consistency)
        def get_strategy_weights(i, metrics, vol_pct, trend_pct, trend_dir, vol_pct_rank):
            if i < base_lookback:
                return {'ema': 1.0}
            
            weights = {}
            total_score = 0
            
            for strategy in strategy_metrics.keys():
                # Base score from performance metrics
                perf_score = (
                    metrics[strategy]['sharpe'].iloc[i] * 0.4 +
                    metrics[strategy]['sortino'].iloc[i] * 0.4 +
                    metrics[strategy]['win_rate'].iloc[i] * 0.2
                )
                
                # Adjust score based on market conditions
                if vol_pct > 0.8:  # High volatility
                    if strategy in ['rsi', 'vwap']:
                        perf_score *= 1.3
                elif vol_pct < 0.2:  # Low volatility
                    if strategy in ['ema', 'macd']:
                        perf_score *= 1.2
                
                if trend_pct > 0.7:  # Strong trend
                    if strategy in ['macd', 'ema']:
                        perf_score *= 1.25
                
                if vol_pct_rank > 0.75:  # High volume
                    if strategy in ['volume_rsi', 'vwap']:
                        perf_score *= 1.2
                
                weights[strategy] = max(0, perf_score)
                total_score += weights[strategy]
            
            if total_score > 0:
                weights = {k: v/total_score for k, v in weights.items()}
            else:
                weights = {k: 1/len(weights) for k in weights}
            
            return weights
        
        # Generate signals (kept the same for consistency)
        for i in range(len(df)):
            weights = get_strategy_weights(
                i, strategy_metrics,
                vol_rank.iloc[i],
                trend_rank.iloc[i],
                trend_direction.iloc[i],
                volume_rank.iloc[i]
            )
            
            combined_signal = 0
            for strategy, weight in weights.items():
                combined_signal += strategy_signals[strategy].iloc[i] * weight
            
            if combined_signal > 0.3:
                signals.iloc[i] = 1
            elif combined_signal < -0.3:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0
        
        # Apply risk management (kept exactly the same)
        vol_percentile = returns_vol.rank(pct=True)
        high_vol_mask = vol_percentile > 0.85
        signals[high_vol_mask] = signals[high_vol_mask] * 0.5
        
        trend_mask = (trend_direction * signals) < 0
        signals[trend_mask] = 0
        
        low_vol_mask = relative_volume < relative_volume.quantile(0.2)
        signals[low_vol_mask] = 0
        
        extreme_vol_mask = returns_vol > returns_vol.quantile(0.95)
        signals[extreme_vol_mask] = 0
        
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