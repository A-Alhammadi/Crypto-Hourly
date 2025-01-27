import pandas as pd
import numpy as np
from database import DatabaseHandler
from indicators import TechnicalIndicators
from strategies import TradingStrategies
from backtester import Backtester
from config import BACKTEST_CONFIG
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from datetime import datetime

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_results_to_file(results_dict, symbol, output_dir, combined_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_dir = os.path.join(output_dir, symbol.replace('/', '_'))
    ensure_directory(symbol_dir)
    
    # Get the first strategy result
    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    
    print("\nAvailable columns in portfolio:", df.columns.tolist())
    
    # Calculate buy and hold metrics
    initial_price = float(df['close'].iloc[0])  # Changed from close_price to close
    final_price = float(df['close'].iloc[-1])   # Changed from close_price to close
    buy_hold_return = (final_price - initial_price) / initial_price
    buy_hold_annual_return = buy_hold_return * (365 / len(df))
    
    # Create buy and hold metrics
    buy_hold_metrics = {
        'Strategy': 'Buy and Hold',
        'Total Return': f"{buy_hold_return * 100:.2f}%",
        'Annual Return': f"{buy_hold_annual_return * 100:.2f}%",
        'Number of Trades': 1,
        'Trading Fees': f"${BACKTEST_CONFIG['initial_capital'] * BACKTEST_CONFIG['trading_fee']:.2f}"
    }
    
    # Save results
    summary_file = os.path.join(symbol_dir, f'summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("=== Backtest Configuration ===\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']}\n\n")
        
        f.write("=== Strategy Results ===\n")
        metrics_list = [result['metrics'] for result in results_dict.values()]
        metrics_list.append(buy_hold_metrics)
        f.write(tabulate(metrics_list, headers="keys", tablefmt="grid", numalign="right"))
        f.write("\n\n")
    
    # Store for combined results
    combined_results[symbol] = {
        'buy_hold': buy_hold_metrics,
        'strategies': metrics_list[:-1]
    }
    
    # Save detailed trades if enabled
    if BACKTEST_CONFIG['save_trades']:
        for strategy_name, result in results_dict.items():
            if not result['trades'].empty:
                trades_file = os.path.join(symbol_dir, f'trades_{strategy_name}_{timestamp}.csv')
                result['trades'].to_csv(trades_file)
    
    return summary_file

def analyze_strategy_correlations(results_dict, df, symbol, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_dir = os.path.join(output_dir, symbol.replace('/', '_'))
    analysis_file = os.path.join(symbol_dir, f'correlation_analysis_{timestamp}.txt')
    
    # Calculate daily returns for each strategy
    strategy_returns = {}
    for strategy_name, result in results_dict.items():
        strategy_returns[strategy_name] = result['portfolio']['total_value'].pct_change()
    
    returns_df = pd.DataFrame(strategy_returns)
    market_chars = df[['volatility', 'atr', 'relative_volume', 'momentum', 'trend_direction']]
    
    # Calculate correlations
    correlations = pd.DataFrame()
    
    # Helper function for regime creation
    def create_regime_labels(series, n_bins=4):
        try:
            return pd.qcut(series, n_bins, labels=['Low', 'Medium-Low', 'Medium-High', 'High'], duplicates='drop')
        except ValueError:
            try:
                return pd.qcut(series, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            except ValueError:
                return pd.cut(series, 2, labels=['Low', 'High'])
    
    # Helper function to calculate regime statistics
    def calculate_regime_stats(returns_df, mask):
        stats = {}
        for strategy in returns_df.columns:
            strategy_returns = returns_df[strategy][mask]
            annual_return = strategy_returns.mean() * 252
            std = strategy_returns.std()
            sharpe = (strategy_returns.mean() / std * np.sqrt(252)) if std != 0 else 0
            win_rate = (strategy_returns > 0).mean()
            
            stats[strategy] = {
                'Annual Return': annual_return,
                'Sharpe Ratio': sharpe,
                'Win Rate': win_rate
            }
        return pd.DataFrame(stats).T
    
    with open(analysis_file, 'w') as f:
        f.write(f"=== Strategy Performance Correlation Analysis for {symbol} ===\n\n")
        
        # 1. Overall correlations with market characteristics
        f.write("1. Overall Correlations with Market Characteristics:\n")
        for char in market_chars.columns:
            for strategy in returns_df.columns:
                corr = returns_df[strategy].corr(market_chars[char])
                correlations.loc[strategy, char] = corr
        f.write(tabulate(correlations.round(3), headers='keys', tablefmt='grid'))
        f.write("\n\n")
        
        # 2. Performance in different market conditions
        f.write("2. Strategy Performance in Different Market Conditions:\n\n")
        
        # a) Volatility regimes
        f.write("a) Performance in Different Volatility Regimes:\n")
        volatility_quartiles = create_regime_labels(df['volatility'])
        vol_performance = pd.DataFrame()
        
        for regime in volatility_quartiles.unique():
            mask = volatility_quartiles == regime
            regime_stats = calculate_regime_stats(returns_df, mask)
            vol_performance[regime] = regime_stats['Annual Return']
            
            f.write(f"\nVolatility Regime: {regime}\n")
            f.write(tabulate(regime_stats.round(3), headers='keys', tablefmt='grid'))
        
        # b) Trend regimes
        f.write("\n\nb) Performance in Different Trend Regimes:\n")
        trend_quartiles = create_regime_labels(abs(df['trend_direction']))
        trend_performance = pd.DataFrame()
        
        for regime in trend_quartiles.unique():
            mask = trend_quartiles == regime
            regime_stats = calculate_regime_stats(returns_df, mask)
            trend_performance[regime] = regime_stats['Annual Return']
            
            f.write(f"\nTrend Strength: {regime}\n")
            f.write(tabulate(regime_stats.round(3), headers='keys', tablefmt='grid'))
        
        # c) Volume regimes
        f.write("\n\nc) Performance in Different Volume Regimes:\n")
        volume_quartiles = create_regime_labels(df['relative_volume'])
        volume_performance = pd.DataFrame()
        
        for regime in volume_quartiles.unique():
            mask = volume_quartiles == regime
            regime_stats = calculate_regime_stats(returns_df, mask)
            volume_performance[regime] = regime_stats['Annual Return']
            
            f.write(f"\nRelative Volume: {regime}\n")
            f.write(tabulate(regime_stats.round(3), headers='keys', tablefmt='grid'))
        
        # 3. Key findings and recommendations
        f.write("\n\n3. Key Findings and Strategy Recommendations:\n")
        
        # Best strategy for each regime
        f.write("\na) Best Strategies by Market Regime:\n")
        
        # Helper function to get best strategy
        def get_best_strategy(performance_df, regime):
            if regime in performance_df.columns:
                strategy = performance_df[regime].idxmax()
                return strategy, performance_df.loc[strategy, regime]
            return None, None
        
        # Volatility
        f.write("\nBest strategy for each volatility regime:\n")
        for regime in volatility_quartiles.unique():
            strategy, perf = get_best_strategy(vol_performance, regime)
            if strategy:
                f.write(f"- {regime} volatility: {strategy} (Return: {perf:.2f}%)\n")
        
        # Trend
        f.write("\nBest strategy for each trend regime:\n")
        for regime in trend_quartiles.unique():
            strategy, perf = get_best_strategy(trend_performance, regime)
            if strategy:
                f.write(f"- {regime} trend: {strategy} (Return: {perf:.2f}%)\n")
        
        # Volume
        f.write("\nBest strategy for each volume regime:\n")
        for regime in volume_quartiles.unique():
            strategy, perf = get_best_strategy(volume_performance, regime)
            if strategy:
                f.write(f"- {regime} volume: {strategy} (Return: {perf:.2f}%)\n")
        
        # Overall recommendation
        f.write("\nb) Overall Strategy Recommendations:\n")
        f.write("\nBased on the analysis above:\n")
        high_vol_strategy, _ = get_best_strategy(vol_performance, 'High')
        strong_trend_strategy, _ = get_best_strategy(trend_performance, 'Strong')
        high_volume_strategy, _ = get_best_strategy(volume_performance, 'High')
        
        if high_vol_strategy:
            f.write(f"1. For high volatility periods: Use {high_vol_strategy}\n")
        if strong_trend_strategy:
            f.write(f"2. For strong trends: Use {strong_trend_strategy}\n")
        if high_volume_strategy:
            f.write(f"3. For high volume periods: Use {high_volume_strategy}\n")
        
    return analysis_file

def plot_results(results_dict, symbol, output_dir):
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Get first result for buy and hold calculation
    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    initial_price = float(df['close'].iloc[0])  # Changed from close_price to close
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = df['close'] * buy_hold_units  # Changed from close_price to close
    
    # Plot portfolio values
    ax1.plot(df.index, buy_hold_values, 
             label='Buy and Hold', linewidth=2, color='black', linestyle='--')
    
    for strategy_name, result in results_dict.items():
        portfolio = result['portfolio']
        ax1.plot(portfolio.index, portfolio['total_value'], 
                label=strategy_name, linewidth=1.5)
    
    ax1.set_title(f'Portfolio Value Over Time - {symbol}', fontsize=12, pad=20)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot drawdowns
    buy_hold_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    ax2.plot(df.index, buy_hold_dd, 
             label='Buy and Hold', linewidth=2, color='black', linestyle='--')
    
    for strategy_name, result in results_dict.items():
        portfolio = result['portfolio']
        drawdown = (portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()
        ax2.plot(portfolio.index, drawdown, label=strategy_name, linewidth=1.5)
    
    ax2.set_title('Strategy Drawdowns', fontsize=12, pad=20)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_dir = os.path.join(output_dir, symbol.replace('/', '_'))
    ensure_directory(symbol_dir)
    plot_file = os.path.join(symbol_dir, f'performance_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_combined_results(combined_results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = os.path.join(output_dir, f'combined_results_{timestamp}.txt')
    
    with open(combined_file, 'w') as f:
        f.write("=== Combined Backtest Results ===\n")
        f.write(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Initial Capital per Symbol: ${BACKTEST_CONFIG['initial_capital']}\n\n")
        
        for symbol, results in combined_results.items():
            f.write(f"\n=== {symbol} Results ===\n")
            all_results = results['strategies'] + [results['buy_hold']]
            f.write(tabulate(all_results, headers="keys", tablefmt="grid", numalign="right"))
            f.write("\n" + "="*80 + "\n")
    
    return combined_file

def main():
    try:
        print("Starting backtesting process...")
        
        # Initialize database connection
        db = DatabaseHandler()
        
        # Create results directory
        output_dir = BACKTEST_CONFIG['results_dir']
        ensure_directory(output_dir)
        print(f"\nResults will be saved to: {output_dir}")
        
        # Store results for all symbols
        combined_results = {}
        
        # Process each symbol
        for symbol in BACKTEST_CONFIG['symbols']:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}...")
            print(f"{'='*50}")
            
            # Get historical data
            print("\nFetching historical data...")
            data = db.get_historical_data(
                symbol,
                BACKTEST_CONFIG['start_date'],
                BACKTEST_CONFIG['end_date']
            )
            
            if len(data) == 0:
                print(f"No data found for {symbol}")
                continue
                
            print(f"Loaded {len(data)} records")
            print("Initial columns:", data.columns.tolist())
            
            # Add technical indicators
            print("\nCalculating technical indicators...")
            data = TechnicalIndicators.add_all_indicators(data)
            print("Added indicators. Final columns:", data.columns.tolist())
            
            # Run backtests
            results = {}
            print("\nRunning backtests for each strategy:")
            for strategy_name, strategy_func in TradingStrategies.get_all_strategies().items():
                print(f"\nBacktesting {strategy_name} strategy...")
                try:
                    backtester = Backtester(data, strategy_name, strategy_func)
                    results[strategy_name] = backtester.run()
                    print(f"✓ {strategy_name} strategy completed successfully")
                except Exception as e:
                    print(f"✗ Error in {strategy_name} strategy: {str(e)}")
                    raise
            
            # Save results
            print("\nSaving results...")
            summary_file = save_results_to_file(results, symbol, output_dir, combined_results)
            print(f"✓ Results saved to {summary_file}")
            
            # Generate correlation analysis
            print("\nGenerating correlation analysis...")
            analysis_file = analyze_strategy_correlations(results, data, symbol, output_dir)
            print(f"✓ Correlation analysis saved to {analysis_file}")
            
            # Create plots
            print("\nGenerating performance plots...")
            plot_results(results, symbol, output_dir)
            print("✓ Performance plots saved")
        
        # Save combined results
        if combined_results:
            print("\nSaving combined results...")
            combined_file = save_combined_results(combined_results, output_dir)
            print(f"✓ Combined results saved to {combined_file}")
        
        print("\nBacktesting process completed successfully!")
        
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        raise
    finally:
        print("\nClosing database connection...")
        db.close()

if __name__ == "__main__":
    main()