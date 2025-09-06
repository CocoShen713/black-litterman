import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime as dt
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion
from pypfopt import risk_models, expected_returns
from pypfopt import EfficientFrontier, objective_functions, CLA
from pypfopt.exceptions import OptimizationError

# Load the historical risk-free rates from the CSV file
rates_file_path = 'data/T-Bill Historical Rates.xlsx'
rates_df = pd.read_excel(rates_file_path)

# Convert the 'Date' column to datetime if in a non-standard format
rates_df['Date'] = pd.to_datetime(rates_df['Date'])

# Set the 'Dates' column as the index for easy lookup
rates_df.set_index('Date', inplace=True)

viewdict = {}
confidences = []
posterior_days = 900

# Load the Excel file
file_path = "data/BL Model Data 7-24-24 (1).xlsx"
data = pd.read_excel(file_path, header=1)

# Ensure the Date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Proceed with extracting relevant columns as before
price_data = data[['date', 'Ticker', 'Adj Closing Price', 'Mcap (Billions)', 'Ticker Daily Return']]

# Extract the S&P 500 data
sp500_data = price_data[price_data['Ticker'] == 'SP500']

# Load the views file
views_file_path = 'BL Posterior Estimates (Original Posterior Views + Confidence Levels V2).xlsx'
views_df = pd.read_excel(views_file_path)

# Check the column names of views_df
#print(views_df)

# Convert the 'Date' column in the views file to datetime
views_df['Date'] = pd.to_datetime(views_df['Date'])

# Set up a loop for each date from Jan 3, 2023, to June 28, 2024
start_date = pd.to_datetime("2023-01-03")
end_date = pd.to_datetime("2024-06-28")
loop_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Initialize the dictionary to store previous day's market weights
previous_weights = {}

# Initialize previous weights with 0 for all tickers
for ticker in data['Ticker'].unique():
    previous_weights[ticker] = 0

# Initialize final DataFrame
final_asset_weights = pd.DataFrame(columns=['Date',
                                            'Ticker',
                                            'Expected Asset Posterior Return',
                                            'Posterior Weight',
                                            'Market Weight',
                                            'Asset Closing Price',
                                            'Expected Posterior Return View',
                                            'Confidence Level',
                                            'Actual Asset Daily Return'])

# Initialize lists to store the results
final_portfolio_stats = []

# Loop through each date in the range
for current_date in loop_dates:
    # Check if current_date is in the DataFrame
    if current_date not in data['date'].values:
        continue  # Skip this date if it's not in the data
    # Step 1: Define the prior return period (900 days before the current date)
    prior_start_date = current_date - pd.DateOffset(days=posterior_days)
    prior_end_date = current_date - pd.DateOffset(days=1)

    #print(prior_start_date, prior_end_date)

    # Step 2: Filter the data for the prior period
    prior_data = data[(data['date'] >= prior_start_date) & (data['date'] <= prior_end_date)]

    # Convert 'Date' column to datetime and set as index
    prior_data.set_index('date', inplace=True)

    # Pivot the data to get adjusted closing prices
    adj_closing_prices = prior_data.pivot(columns='Ticker', values='Adj Closing Price')
    #print(adj_closing_prices)

    # Calculate the covariance matrix of excess returns using covariance shrinkage
    cov_matrix = risk_models.CovarianceShrinkage(adj_closing_prices).ledoit_wolf()

    # Filter for XLB rows to get corresponding SP500 prices
    xlb_df = prior_data[prior_data['Ticker'] == 'XLB']

    # Ensure the SP500 closing prices are under the correct column name
    sp500_close_col = 'SP500 Closing Price'

    # Filter data to include only the dates in the XLB rows
    xlb_dates = xlb_df.index.unique()
    filtered_prices_df = prior_data.loc[prior_data.index.isin(xlb_dates)]

    # Extract SP500 closing prices corresponding to XLB dates
    market_prices = filtered_prices_df[sp500_close_col]
    #print(market_prices)

    # Check if current_date exists in rates_df before accessing
    if current_date in rates_df.index:
        risk_free_rate = (rates_df.loc[current_date, 'Rate']) / 100
    else:
        # Handle the case where current_date is missing
        # Use risk-free rate from the previous available date
        risk_free_rate = (rates_df.loc[:current_date, 'Rate'].iloc[-1]) / 100

    #print(risk_free_rate)

    # Calculate the risk aversion parameter
    delta = market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=risk_free_rate)

    # Calculate market equilibrium returns (prior) based on market cap weights
    market_caps = data.loc[data['date'] == current_date, ['Ticker', 'Mcap (Billions)']].set_index('Ticker')['Mcap (Billions)']
    #print(market_caps)
    prior_returns = market_implied_prior_returns(market_caps, delta, cov_matrix, risk_free_rate=risk_free_rate)
    #print(prior_returns)

    # Calculate daily returns for each ETF for the current date
    # mean_daily_returns = prior_data.groupby('Ticker')['Ticker Daily Return'].mean()
    daily_returns = data.loc[(data['date'] == current_date), ['Ticker', 'Ticker Daily Return']].set_index('Ticker')['Ticker Daily Return']
    #print(current_date, daily_returns)

    # Calculate market weights
    market_weights = market_caps / market_caps.sum()

    # Calculate market return based on market weights
    #market_return = daily_returns.dot(market_weights)

    min_weights = {}

    # Retrieve the previous day's market weights (for the constraint)
    if current_date != loop_dates[0] and not final_asset_weights.empty:  # Skip for the first date
        for ticker in market_weights.index:
            min_weights[ticker] = 0.2 * previous_weights[ticker]
    else:
        for ticker in market_weights.index:
            min_weights[ticker] = previous_weights[ticker]

    #print(previous_weights, min_weights)

    # Load the views for the current date from the views file
    current_views = views_df[views_df['Date'] == current_date]
    viewdict = current_views.set_index('Sector')['Expected Posterior Return'].to_dict()
    confidences = current_views.set_index('Sector')['Confidence Level'].to_list()
    '''for i in range(len(confidences)):
      if confidences[i] > 1 or confidences[i] < 0:
        confidences[i] = 0.9'''

    #print(confidences)

    #print(current_date, viewdict, confidences)

    # Initialize the Black-Litterman model
    bl = BlackLittermanModel(
        cov_matrix,  # Covariance Matrix
        pi=prior_returns,  # Prior expected returns
        absolute_views=viewdict,  # Views
        omega='idzorek',  # Confidence levels in %
        view_confidences=np.array(confidences)  # Confidences
    )

    # Calculate the posterior returns
    posterior_returns = bl.bl_returns()

    # New covariance matrix
    #bl_cov_matrix = bl.bl_cov()

    # Calculate the Efficient Frontier to get metrics
    ef = EfficientFrontier(posterior_returns, cov_matrix)
    ef.add_objective(objective_functions.L2_reg)
    ef.add_constraint(lambda x: x >= pd.Series(min_weights))  # Apply the min weight constraint

    try:
        if max(posterior_returns) > risk_free_rate:
            ef.max_sharpe(risk_free_rate=risk_free_rate)  # Optimize for Sharpe ratio if feasible
        else:
            ef.min_volatility()  # Optimize for minimum volatility if max_sharpe is not feasible
            #print(current_date, "Min Volatility")
    except OptimizationError:
        #print(current_date, "Min Volatility after OptError")
        #print(posterior_returns)
        ef = EfficientFrontier(posterior_returns, cov_matrix)
        ef.add_objective(objective_functions.L2_reg)
        ef.add_constraint(lambda x: x >= pd.Series(min_weights))  # Apply the min weight constraint
        ef.min_volatility()  # If max_sharpe fails, do min_volatility

    bl_weights = ef.clean_weights()

    #bl.bl_weights(delta)
    #weights = bl.clean_weights()

    #print(cleaned_weights)

    # Calculate the portfolio performance
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    #print(expected_return, posterior_returns.dot(pd.Series(bl_weights)))
    #expected_return = posterior_returns.dot(pd.Series(weights))

    previous_weights = market_weights.to_dict()

    # Calculate the daily portfolio return using the posterior weights from the Black-Litterman model
    daily_portfolio_return = daily_returns.dot(pd.Series(bl_weights))

    # Append results to final DataFrame
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

# Convert results to DataFrames
final_asset_weights_df = pd.DataFrame(final_asset_weights)
final_portfolio_stats_df = pd.DataFrame(final_portfolio_stats)

# Calculate cumulative portfolio and market returns
cum_portfolio_return = (1 + final_portfolio_stats_df['Daily Portfolio Return']).prod() - 1
cum_market_return = (1 + final_portfolio_stats_df['Daily Market (S&P 500) Return']).prod() - 1

# Compute annualized Sharpe ratio and volatility of portfolio
mean_excess = final_portfolio_stats_df['Daily Excess Return'].mean()
std_daily = final_portfolio_stats_df['Daily Portfolio Return'].std()

portfolio_sharpe_ratio = (mean_excess / std_daily) * np.sqrt(252)
portfolio_volatility = std_daily * np.sqrt(252)

import statsmodels.api as sm

# Extract the columns
df = final_portfolio_stats_df.dropna(subset=['Daily Portfolio Return',
                                              'Daily Market (S&P 500) Return',
                                              'Daily Risk-Free Rates'])

portfolio_returns = df['Daily Portfolio Return']
market_returns = df['Daily Market (S&P 500) Return']
risk_free_rates = df['Daily Risk-Free Rates']

# Calculating Beta and Raw Regression Alpha
X = sm.add_constant(market_returns)
model = sm.OLS(portfolio_returns, X).fit()

beta = model.params['Daily Market (S&P 500) Return']
raw_alpha = model.params['const']

# Calculate Jensen's Alpha
Rp = portfolio_returns.mean()
Rm = market_returns.mean()
Rf = risk_free_rates.mean()

jensen_alpha = Rp - Rf - beta * (Rm - Rf)

# Annualizing Alpha
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
#print("Jensen's Alpha (Daily):", jensen_alpha)
print("Alpha (Annualized):", annualized_raw_alpha, f"({annualized_raw_alpha:.2%})")
#print("Annualized Jensen's Alpha:", annualized_jensen_alpha)
print()

# Save results to Excel or display
final_asset_weights_df.to_excel("Final_Asset_Weights.xlsx", index=False)
final_portfolio_stats_df.to_excel("Final_Portfolio_Stats.xlsx", index=False)

# Create summary DataFrame (single row with custom column headers)
summary_df = pd.DataFrame([{
    'Cumulative Portfolio Return': cum_portfolio_return,
    'Cumulative Market Return': cum_market_return,
    'Sharpe Ratio': portfolio_sharpe_ratio,
    'Volatility': portfolio_volatility,
    'Beta': beta,
    'Alpha (Daily)': raw_alpha,
    'Alpha (Annualized)': annualized_raw_alpha
}])

# Determine start column to write beside main data
startcol = final_portfolio_stats_df.shape[1] + 2  # leave 1 column buffer

# Append the summary to the right
with pd.ExcelWriter("Final_Portfolio_Stats.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    summary_df.to_excel(writer, startrow=0, startcol=startcol, index=False)

# Pivot to get daily weights by ETF
daily_weights = final_asset_weights_df.pivot(index='Date', columns='Ticker', values='Posterior Weight')
daily_market_weights = final_asset_weights_df.pivot(index='Date', columns='Ticker', values='Market Weight')

# Preserve ETF order
etf_order = daily_weights.columns.tolist()
mean_market_weights = daily_market_weights[etf_order].mean()
market_weights_std = daily_market_weights[etf_order].std()

# Print standard deviations of daily market weights
'''print("Standard Deviations of Daily Market Weights:")
for ticker, std in market_weights_std.items():
    print(f"{ticker}:", std, f"({std:.2%})")

print()'''

import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Custom formatter for 2 decimal places in %
percent_fmt_2dp = FuncFormatter(lambda y, _: f'{y:.2%}')

# ---------- Mean of Daily Black-Litterman Allocation Weights vs. Daily Market Weights by ETF Plot ----------

# Preserve ETF order
mean_weights = daily_weights[etf_order].mean()

# Print mean daily optimal allocation weights
print("Mean Daily Black-Litterman Allocation Weights:")
for ticker, weight in mean_weights.items():
    print(f"{ticker}:", weight, f"({weight:.2%})")

print()

# Print mean daily market weights
print("Mean Daily Market Weights:")
for ticker, weight in mean_market_weights.items():
    print(f"{ticker}:", weight, f"({weight:.2%})")

print()

# Convert Series to DataFrames
mean_weights_df = mean_weights.to_frame(name='Portfolio')
mean_market_weights_df = mean_market_weights.to_frame(name='Market')

# Join them on index (ETF tickers)
combined_df = mean_weights_df.join(mean_market_weights_df)

# Convert to percentage
combined_df *= 100

# Extract values
etfs = combined_df.index.tolist()
portfolio_weights = combined_df.iloc[:, 0].values
market_weights = combined_df.iloc[:, 1].values

# Plotting parameters
x = np.arange(len(etfs)) * 1.2
width = 0.5

# Step 2: Create the bar chart
fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width / 2, portfolio_weights, width, label='Portfolio', color='royalblue')
bars2 = ax.bar(x + width / 2, market_weights, width, label='Market', color='darkorange')

# Grid and formatting
ax.set_ylabel('Allocation Mean (%)')
ax.set_title('Mean Daily Black-Litterman Portfolio Allocation vs. Daily Market Weights by ETF')
ax.set_xticks(x)
ax.set_xticklabels(etfs)
ax.set_xlabel('ETF')
ax.legend()
ax.yaxis.grid(True, linestyle='-', alpha=0.6)

# Y-axis tick formatting
max_val = max(np.max(portfolio_weights), np.max(market_weights))
y_max = np.ceil(max_val + 1)
ax.set_ylim(0, y_max)
ax.set_yticks(np.arange(0, y_max + 1, 2))
ax.set_yticklabels([f"{y:.2f}%" for y in ax.get_yticks()])

# Add labels below bar tops
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

print()

# ----------  Optimal Black-Litterman Portfolio Allocation % - Monthly Average Plot ----------

# Resample to monthly average
monthly_avg_weights = daily_weights.resample('ME').mean()

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
monthly_avg_weights.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')

# Format y-axis as %
ax.yaxis.set_major_formatter(percent_fmt_2dp)
ax.yaxis.grid(True, linestyle='-', alpha=0.6)

# Format x-axis labels to "Jan 2023"
ax.set_xticklabels([d.strftime('%b %Y') for d in monthly_avg_weights.index], rotation=45)

# Graph
ax.set_title('Optimal Black-Litterman Portfolio Allocation % - Monthly Average')
ax.set_ylabel('Monthly Average Allocation (%)')
ax.set_xlabel('Month')
ax.legend(title='ETF', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print()

# ---------- Standard Deviation of Daily Black-Litterman Allocation Weights by ETF Plot ----------

# Preserve ETF order
weights_std = daily_weights[etf_order].std()

max_val_std = weights_std.max()  # max bar height

# Round up to the next 1% (0.01)
ceiling_std = math.ceil(max_val_std * 100) / 100

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(weights_std.index, weights_std.values, color='cornflowerblue')

# Set y-axis upper limit to the ceiling
ax.set_ylim(0, ceiling_std)

# Show value labels above bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2), textcoords='offset points', ha='center', va='bottom')

# Format y-axis as %
ax.yaxis.set_major_formatter(percent_fmt_2dp)
ax.yaxis.grid(True, linestyle='-', alpha=0.6)

# Graph
ax.set_title('Standard Deviation of Daily Black-Litterman Allocation Weights by ETF')
ax.set_ylabel('Allocation Standard Deviation (%)')
ax.set_xlabel('ETF')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print()

# ---------- Cumulative Returns Over Test Period: Portfolio vs. S&P 500 Plot ----------

# Calculate cumulative returns
cumulative_returns = (1 + final_portfolio_stats_df[['Daily Portfolio Return', 'Daily Market (S&P 500) Return']]).cumprod()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(final_portfolio_stats_df['Date'], cumulative_returns['Daily Portfolio Return'] - 1, label='Portfolio Cumulative Return')
ax.plot(final_portfolio_stats_df['Date'], cumulative_returns['Daily Market (S&P 500) Return'] - 1, label='S&P 500 Cumulative Return', color='crimson')

# Format y-axis as %
ax.yaxis.set_major_formatter(percent_fmt_2dp)

# Use monthly ticks
locator = mdates.MonthLocator()
formatter = mdates.DateFormatter('%b %d, %Y')

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Graph
ax.set_title('Cumulative Returns Over Test Period: Portfolio vs. S&P 500')
ax.set_ylabel('Cumulative Growth of $1 Investment')
ax.set_xlabel('Date')
plt.xticks(rotation=45)
ax.legend()
ax.grid(True, linestyle='-', alpha=0.5)
plt.tight_layout()
plt.show()
