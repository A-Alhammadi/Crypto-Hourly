#strategies.py

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
        Improved Adaptive Strategy for Hourly Crypto Data

        Key Changes:
        1. Evaluates each base strategy (EMA, MACD, RSI, Stoch, Volume RSI, VWAP) on
        both a short-term and medium-term rolling window.
        - Short-term performance: captures current alignment with immediate market moves.
        - Medium-term performance: avoids chasing noise and adds stability.

        2. Combines Sharpe, Sortino, and Win Rate into a performance score, but the weighting
        for short vs. medium term can be adjusted for best results.

        3. Eliminates large fixed multipliers for volatility/trend conditions; instead uses
        a more systematic approach to slightly reward or penalize a strategy's score
        if it historically performs well/poorly in the current environment. This should
        make it more robust for different coins/tokens.

        4. Risk management is more gradual:
        - High volatility or extremely low volume reduces position size
            rather than forcing signals fully to zero.
        - Trend mismatch still zeros out signals to avoid bucking a strong long-term trend.

        5. Uses sqrt(8760) in Sharpe/Sortino to reflect hourly bars (24 hours * 365 days = 8760).
        """

        import numpy as np
        import pandas as pd

        # Annualization factor for hourly data (24h * 365 = 8760)
        ANNUAL_FACTOR = np.sqrt(8760)

        # -------------------------------------------------------------------------
        # 1) Gather all individual strategy signals
        # -------------------------------------------------------------------------
        signals = pd.Series(index=df.index, data=0.0)

        # NOTE: We assume TradingStrategies is the same class that has .ema_strategy, etc.
        # If you're inside the same file, you don't need to import it at all.
        strategy_signals = {
            'ema': TradingStrategies.ema_strategy(df),
            'macd': TradingStrategies.macd_strategy(df),
            'rsi': TradingStrategies.rsi_strategy(df),
            'stoch': TradingStrategies.stochastic_strategy(df),
            'volume_rsi': TradingStrategies.volume_rsi_strategy(df),
            'vwap': TradingStrategies.vwap_strategy(df)
        }

        # -------------------------------------------------------------------------
        # 2) Calculate basic price returns for performance metrics
        #    We'll use the close_price column to derive percentage returns.
        # -------------------------------------------------------------------------
        price = df['close_price']
        returns = price.pct_change().fillna(0)

        # -------------------------------------------------------------------------
        # 3) Calculate rolling Sharpe, Sortino, Win Rate for each strategy
        #    on both a short-term window and a medium-term window.
        #    Example windows: short=15 bars, medium=60 bars.
        #    Adjust if you want more smoothing or different lookbacks.
        # -------------------------------------------------------------------------
        short_window = 15
        medium_window = 60

        strategy_metrics = {}
        for name, strat_signal_series in strategy_signals.items():
            # Strategy returns = price returns * previous bar's signal
            strat_returns = returns * strat_signal_series.shift(1).fillna(0)

            # Rolling Sharpe (short window)
            rolling_sharpe_short = strat_returns.rolling(short_window).apply(
                lambda x: (x.mean() / x.std()) * ANNUAL_FACTOR if x.std() != 0 else 0,
                raw=False
            ).fillna(0)

            # Rolling Sharpe (medium window)
            rolling_sharpe_med = strat_returns.rolling(medium_window).apply(
                lambda x: (x.mean() / x.std()) * ANNUAL_FACTOR if x.std() != 0 else 0,
                raw=False
            ).fillna(0)

            # Sortino ratio function (penalizes only negative volatility)
            def sortino(x):
                neg = x[x < 0]
                if len(neg) == 0 or neg.std() == 0:
                    return 0
                return (x.mean() / neg.std()) * ANNUAL_FACTOR

            # Rolling Sortino (short window)
            rolling_sortino_short = strat_returns.rolling(short_window).apply(
                sortino, raw=False
            ).fillna(0)

            # Rolling Sortino (medium window)
            rolling_sortino_med = strat_returns.rolling(medium_window).apply(
                sortino, raw=False
            ).fillna(0)

            # Rolling Win Rate (short vs medium)
            rolling_winrate_short = strat_returns.rolling(short_window).apply(
                lambda x: (x > 0).mean(), raw=False
            ).fillna(0)
            rolling_winrate_med = strat_returns.rolling(medium_window).apply(
                lambda x: (x > 0).mean(), raw=False
            ).fillna(0)

            # Store the rolling metrics
            strategy_metrics[name] = {
                'sharpe_short': rolling_sharpe_short,
                'sharpe_med': rolling_sharpe_med,
                'sortino_short': rolling_sortino_short,
                'sortino_med': rolling_sortino_med,
                'winrate_short': rolling_winrate_short,
                'winrate_med': rolling_winrate_med
            }

        # -------------------------------------------------------------------------
        # 4) Evaluate the market regime (volatility, volume, trend) for partial
        #    scoring adjustments. Subtle to avoid overfitting.
        # -------------------------------------------------------------------------
        # Volatility (20-bar std dev of returns)
        vol = returns.rolling(20).std().fillna(0)

        # Trend measure: difference between short & long EMA
        trend_direction = np.sign(df['ema_9'] - df['ema_50']).fillna(0)
        trend_strength = (df['ema_9'] - df['ema_50']).abs() / df['ema_50'].replace(0, np.nan)
        trend_strength = trend_strength.fillna(0)

        # Relative volume (vs 20-bar average)
        volume_ma = df['volume_crypto'].rolling(20).mean().fillna(method='bfill').replace(0, 1)
        relative_volume = df['volume_crypto'] / volume_ma

        # Convert them into percentile ranks for environment-based weighting
        vol_rank = vol.rank(pct=True).fillna(0.5)
        trend_rank = trend_strength.rank(pct=True).fillna(0.5)
        volume_rank = relative_volume.rank(pct=True).fillna(0.5)

        # -------------------------------------------------------------------------
        # 5) Dynamic weighting function for each bar
        # -------------------------------------------------------------------------
        def compute_weights(i):
            # If not enough history for the medium window, just do equal weighting
            if i < medium_window:
                w = {k: 1.0 for k in strategy_metrics.keys()}
                n = len(w)
                return {k: v / n for k, v in w.items()}

            # Current environment ranks
            v_rank = vol_rank.iloc[i]     # Volatility percentile
            t_rank = trend_rank.iloc[i]   # Trend-strength percentile
            volm_rank = volume_rank.iloc[i]  # Volume percentile

            scores = {}
            for strat_name, m in strategy_metrics.items():
                # Short-term performance
                sh_short = m['sharpe_short'].iloc[i]
                so_short = m['sortino_short'].iloc[i]
                wr_short = m['winrate_short'].iloc[i]

                # Medium-term performance
                sh_med = m['sharpe_med'].iloc[i]
                so_med = m['sortino_med'].iloc[i]
                wr_med = m['winrate_med'].iloc[i]

                # Weighted sum: more emphasis on short term, but keep medium for stability
                perf_score = (
                    0.4 * (sh_short + so_short + wr_short) +
                    0.6 * (sh_med + so_med + wr_med)
                )

                # Subtle environment adjustments
                if v_rank > 0.75 and strat_name in ['rsi', 'vwap']:
                    perf_score *= 1.10
                elif v_rank < 0.25 and strat_name in ['ema', 'macd']:
                    perf_score *= 1.08

                if t_rank > 0.7 and strat_name in ['ema', 'macd']:
                    perf_score *= 1.10

                if volm_rank > 0.75 and strat_name in ['volume_rsi', 'vwap']:
                    perf_score *= 1.10

                # No negative weights
                if perf_score < 0:
                    perf_score = 0
                scores[strat_name] = perf_score

            # Normalize so sum = 1
            total_score = sum(scores.values())
            if total_score == 0:
                w = {k: 1.0 / len(scores) for k in scores.keys()}
            else:
                w = {k: scores[k] / total_score for k in scores.keys()}

            return w

        # -------------------------------------------------------------------------
        # 6) Combine signals with dynamic weights
        # -------------------------------------------------------------------------
        combined_signal = pd.Series(index=df.index, data=0.0)
        for i in range(len(df)):
            w = compute_weights(i)
            s = 0.0
            for strat_name, weight in w.items():
                s += strategy_signals[strat_name].iloc[i] * weight
            # Threshold to finalize signal
            if s > 0.3:
                combined_signal.iloc[i] = 1
            elif s < -0.3:
                combined_signal.iloc[i] = -1
            else:
                combined_signal.iloc[i] = 0

        # -------------------------------------------------------------------------
        # 7) Risk management (gradual)
        #    - If volatility is extremely high (top 10%), reduce position 50%
        #    - If volume is extremely low (lowest 10%), reduce position 50%
        #    - If combined signal contradicts long-term trend, zero it out
        # -------------------------------------------------------------------------
        high_vol_threshold = vol.quantile(0.90)
        low_volm_threshold = volume_rank.quantile(0.10)

        for i in range(len(df)):
            sig = combined_signal.iloc[i]
            # (a) Trend mismatch => zero out
            if trend_direction.iloc[i] != 0 and np.sign(trend_direction.iloc[i]) != np.sign(sig):
                combined_signal.iloc[i] = 0
                continue

            # (b) High volatility => scale down 50%
            if vol.iloc[i] > high_vol_threshold and sig != 0:
                combined_signal.iloc[i] = sig * 0.5

            # (c) Extremely low volume => scale down 50%
            if volume_rank.iloc[i] < low_volm_threshold and combined_signal.iloc[i] != 0:
                combined_signal.iloc[i] = combined_signal.iloc[i] * 0.5

        return combined_signal
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