import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion
from pypfopt import risk_models, EfficientFrontier, objective_functions
from pypfopt.exceptions import OptimizationError
import statsmodels.api as sm
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# ---------------------- Load data ----------------------
rates_file_path = 'data/T-Bill Historical Rates.xlsx'
rates_df = pd.read_excel(rates_file_path)
rates_df['Date'] = pd.to_datetime(rates_df['Date'])
rates_df.set_index('Date', inplace=True)

posterior_days = 900

file_path = "data/BL Model Data 7-24-24 (1).xlsx"
data = pd.read_excel(file_path, header=1)
data['date'] = pd.to_datetime(data['date'])

views_file_path = 'data/BL Posterior Estimates v4.xlsx'
views_df = pd.read_excel(views_file_path)
views_df['Date'] = pd.to_datetime(views_df['Date'])

# ---------------------- Loop dates ----------------------
start_date = pd.to_datetime("2023-01-03")
end_date = pd.to_datetime("2024-06-28")
loop_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# ---------------------- State holders ----------------------
previous_weights = {ticker: 0 for ticker in data['Ticker'].unique()}

final_asset_weights = pd.DataFrame(columns=[
    'Date', 'Ticker', 'Expected Asset Posterior Return', 'Posterior Weight',
    'Market Weight', 'Asset Closing Price', 'Expected Posterior Return View',
    'Confidence Level', 'Actual Asset Daily Return'
])
final_portfolio_stats = []

# ---------------------- Backtest loop ----------------------
for current_date in loop_dates:
    if current_date not in data['date'].values:
        continue

    prior_start_date = current_date - pd.DateOffset(days=posterior_days)
    prior_end_date = current_date - pd.DateOffset(days=1)

    prior_data = data[(data['date'] >= prior_start_date) & (data['date'] <= prior_end_date)].copy()
    prior_data.set_index('date', inplace=True)

    adj_closing_prices = prior_data.pivot(columns='Ticker', values='Adj Closing Price')
    cov_matrix = risk_models.CovarianceShrinkage(adj_closing_prices).ledoit_wolf()

    xlb_df = prior_data[prior_data['Ticker'] == 'XLB']
    sp500_close_col = 'SP500 Closing Price'
    xlb_dates = xlb_df.index.unique()
    filtered_prices_df = prior_data.loc[prior_data.index.isin(xlb_dates)]
    market_prices = filtered_prices_df[sp500_close_col]

    if current_date in rates_df.index:
        risk_free_rate = (rates_df.loc[current_date, 'Rate']) / 100
    else:
        risk_free_rate = (rates_df.loc[:current_date, 'Rate'].iloc[-1]) / 100

    delta = market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=risk_free_rate)

    market_caps = data.loc[data['date'] == current_date, ['Ticker', 'Mcap (Billions)']].set_index('Ticker')['Mcap (Billions)']
    prior_returns = market_implied_prior_returns(market_caps, delta, cov_matrix, risk_free_rate=risk_free_rate)

    daily_returns = data.loc[(data['date'] == current_date), ['Ticker', 'Ticker Daily Return']].set_index('Ticker')['Ticker Daily Return']
    market_weights = market_caps / market_caps.sum()

    min_weights = {}
    if current_date != loop_dates[0] and not final_asset_weights.empty:
        for ticker in market_weights.index:
            min_weights[ticker] = 0.2 * previous_weights[ticker]
    else:
        for ticker in market_weights.index:
            min_weights[ticker] = previous_weights[ticker]

    current_views = views_df[views_df['Date'] == current_date]
    viewdict = current_views.set_index('Sector')['Expected Posterior Return'].to_dict()
    confidences = current_views.set_index('Sector')['Confidence Level'].to_list()

    bl = BlackLittermanModel(
        cov_matrix,
        pi=prior_returns,
        absolute_views=viewdict,
        omega='idzorek',
        view_confidences=np.array(confidences)
    )
    posterior_returns = bl.bl_returns()

    ef = EfficientFrontier(posterior_returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg)
    ef.add_constraint(lambda x: x >= pd.Series(min_weights))

    try:
        if max(posterior_returns) > risk_free_rate:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        else:
            ef.min_volatility()
    except OptimizationError:
        ef = EfficientFrontier(posterior_returns, cov_matrix)
        ef.add_objective(objective_functions.L2_reg)
        ef.add_constraint(lambda x: x >= pd.Series(min_weights))
        ef.min_volatility()

    bl_weights = ef.clean_weights()
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    previous_weights = market_weights.to_dict()
    daily_portfolio_return = daily_returns.dot(pd.Series(bl_weights))

    for ticker in bl_weights.keys():
        new_row = pd.DataFrame({
            'Date': [current_date],
            'Ticker': [ticker],
            'Expected Asset Posterior Return': [posterior_returns[ticker] if ticker in posterior_returns else None],
            'Posterior Weight': [bl_weights[ticker]],
            'Market Weight': [market_weights[ticker]],
            'Asset Closing Price': [data.loc[(data['date'] == current_date) & (data['Ticker'] == ticker), 'Adj Closing Price'].iloc[0]],
            'Expected Posterior Return View': [viewdict[ticker]],
            'Confidence Level': [confidences[list(viewdict.keys()).index(ticker)]],
            'Actual Asset Daily Return': [data.loc[(data['date'] == current_date) & (data['Ticker'] == ticker), 'Ticker Daily Return'].iloc[0]]
        })
        final_asset_weights = pd.concat([final_asset_weights, new_row], ignore_index=True)

    final_portfolio_stats.append({
        'Date': current_date,
        'Daily Risk-Free Rates': ((1 + risk_free_rate) ** (1 / 252) - 1),
        'Daily Portfolio Return': daily_portfolio_return,
        'Daily Market (S&P 500) Return': data.loc[(data['date'] == current_date) & (data['Ticker'] == ticker), 'SP500 Daily Return'].iloc[0],
        'Daily Excess Return': daily_portfolio_return - ((1 + risk_free_rate) ** (1 / 252) - 1),
        'Sharpe Ratio': sharpe_ratio,
        'Volatility': volatility
    })

# ---------------------- Post-processing ----------------------
final_asset_weights_df = pd.DataFrame(final_asset_weights)
final_portfolio_stats_df = pd.DataFrame(final_portfolio_stats)

cum_portfolio_return = (1 + final_portfolio_stats_df['Daily Portfolio Return']).prod() - 1
cum_market_return = (1 + final_portfolio_stats_df['Daily Market (S&P 500) Return']).prod() - 1

mean_excess = final_portfolio_stats_df['Daily Excess Return'].mean()
std_daily = final_portfolio_stats_df['Daily Portfolio Return'].std()
portfolio_sharpe_ratio = (mean_excess / std_daily) * np.sqrt(252)
portfolio_volatility = std_daily * np.sqrt(252)

# ---------------------- Alpha/Beta ----------------------
df = final_portfolio_stats_df.dropna(subset=['Daily Portfolio Return', 'Daily Market (S&P 500) Return', 'Daily Risk-Free Rates'])
portfolio_returns = df['Daily Portfolio Return']
market_returns = df['Daily Market (S&P 500) Return']
risk_free_rates = df['Daily Risk-Free Rates']

X = sm.add_constant(market_returns)
model = sm.OLS(portfolio_returns, X).fit()
beta = model.params['Daily Market (S&P 500) Return']
raw_alpha = model.params['const']

Rp = portfolio_returns.mean()
Rm = market_returns.mean()
Rf = risk_free_rates.mean()
jensen_alpha = Rp - Rf - beta * (Rm - Rf)

annualized_raw_alpha = ((1 + raw_alpha) ** 252) - 1
annualized_jensen_alpha = ((1 + jensen_alpha) ** 252) - 1

print(final_asset_weights_df)
print(final_portfolio_stats_df)

print("Cumulative Portfolio Return over test period:", cum_portfolio_return, f"({cum_portfolio_return:.2%})")
print("Cumulative Market Return over test period:", cum_market_return, f"({cum_market_return:.2%})")
print(f"Sharpe Ratio of Portfolio over test period: {portfolio_sharpe_ratio:.2f}")
print("Volatility of Portfolio over test period:", portfolio_volatility, f"({portfolio_volatility:.2%})")
print(f"Beta: {beta:.3f}")
print("Alpha (Daily):", raw_alpha)
print("Alpha (Annualized):", annualized_raw_alpha, f"({annualized_raw_alpha:.2%})")
print()

# ---------------------- Save to Excel ----------------------
final_asset_weights_df.to_excel("Final_Asset_Weights.xlsx", index=False)
final_portfolio_stats_df.to_excel("Final_Portfolio_Stats.xlsx", index=False)

summary_df = pd.DataFrame([{
    'Cumulative Portfolio Return': cum_portfolio_return,
    'Cumulative Market Return': cum_market_return,
    'Sharpe Ratio': portfolio_sharpe_ratio,
    'Volatility': portfolio_volatility,
    'Beta': beta,
    'Alpha (Daily)': raw_alpha,
    'Alpha (Annualized)': annualized_raw_alpha
}])

startcol = final_portfolio_stats_df.shape[1] + 2

with pd.ExcelWriter("Final_Portfolio_Stats.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    summary_df.to_excel(writer, startrow=0, startcol=startcol, index=False)

# ---------------------- Plots ----------------------
daily_weights = final_asset_weights_df.pivot(index='Date', columns='Ticker', values='Posterior Weight')
daily_market_weights = final_asset_weights_df.pivot(index='Date', columns='Ticker', values='Market Weight')

etf_order = daily_weights.columns.tolist()
mean_market_weights = daily_market_weights[etf_order].mean()

percent_fmt_2dp = FuncFormatter(lambda y, _: f'{y:.2%}')

# Mean BL weights vs market weights
mean_weights = daily_weights[etf_order].mean()
print("Mean Daily Black-Litterman Allocation Weights:")
for ticker, weight in mean_weights.items():
    print(f"{ticker}:", weight, f"({weight:.2%})")
print()

print("Mean Daily Market Weights:")
for ticker, weight in mean_market_weights.items():
    print(f"{ticker}:", weight, f"({weight:.2%})")
print()

combined_df = mean_weights.to_frame(name='Portfolio').join(mean_market_weights.to_frame(name='Market')) * 100
etfs = combined_df.index.tolist()
portfolio_weights = combined_df.iloc[:, 0].values
market_weights = combined_df.iloc[:, 1].values

x = np.arange(len(etfs)) * 1.2
width = 0.5

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width / 2, portfolio_weights, width, label='Portfolio', color='royalblue')
bars2 = ax.bar(x + width / 2, market_weights, width, label='Market', color='darkorange')

ax.set_ylabel('Allocation Mean (%)')
ax.set_title('Mean Daily Black-Litterman Portfolio Allocation vs. Daily Market Weights by ETF')
ax.set_xticks(x)
ax.set_xticklabels(etfs)
ax.set_xlabel('ETF')
ax.legend()
ax.yaxis.grid(True, linestyle='-', alpha=0.6)

max_val = max(np.max(portfolio_weights), np.max(market_weights))
y_max = np.ceil(max_val + 1)
ax.set_ylim(0, y_max)
ax.set_yticks(np.arange(0, y_max + 1, 2))
ax.set_yticklabels([f"{y:.2f}%" for y in ax.get_yticks()])

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Monthly average stacked bar
monthly_avg_weights = daily_weights.resample('ME').mean()
fig, ax = plt.subplots(figsize=(14, 6))
monthly_avg_weights.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
ax.yaxis.set_major_formatter(percent_fmt_2dp)
ax.yaxis.grid(True, linestyle='-', alpha=0.6)
ax.set_xticklabels([d.strftime('%b %Y') for d in monthly_avg_weights.index], rotation=45)
ax.set_title('Optimal Black-Litterman Portfolio Allocation % - Monthly Average')
ax.set_ylabel('Monthly Average Allocation (%)')
ax.set_xlabel('Month')
ax.legend(title='ETF', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Std of daily weights
weights_std = daily_weights[etf_order].std()
ceiling_std = math.ceil(weights_std.max() * 100) / 100

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(weights_std.index, weights_std.values, color='cornflowerblue')
ax.set_ylim(0, ceiling_std)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2), textcoords='offset points', ha='center', va='bottom')
ax.yaxis.set_major_formatter(percent_fmt_2dp)
ax.yaxis.grid(True, linestyle='-', alpha=0.6)
ax.set_title('Standard Deviation of Daily Black-Litterman Allocation Weights by ETF')
ax.set_ylabel('Allocation Standard Deviation (%)')
ax.set_xlabel('ETF')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cumulative returns
cumulative_returns = (1 + final_portfolio_stats_df[['Daily Portfolio Return', 'Daily Market (S&P 500) Return']]).cumprod()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(final_portfolio_stats_df['Date'], cumulative_returns['Daily Portfolio Return'] - 1, label='Portfolio Cumulative Return')
ax.plot(final_portfolio_stats_df['Date'], cumulative_returns['Daily Market (S&P 500) Return'] - 1, label='S&P 500 Cumulative Return', color='crimson')
ax.yaxis.set_major_formatter(percent_fmt_2dp)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
ax.set_title('Cumulative Returns Over Test Period: Portfolio vs. S&P 500')
ax.set_ylabel('Cumulative Growth of $1 Investment')
ax.set_xlabel('Date')
plt.xticks(rotation=45)
ax.legend()
ax.grid(True, linestyle='-', alpha=0.5)
plt.tight_layout()
plt.show()