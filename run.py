import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

FILE_PATH = "di1v24.xlsx" 
DAYS = 172
TRIALS = 100000
EBITDA = 25451101.50

def load_and_clean_data(file_path): # Load and clean data
    df = pd.read_excel(file_path)
    df.replace('-', np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.rename(columns={'Fechamento\najust p/ prov\nEm moeda orig\nDI1V24':'Fechamento'}, inplace=True) # Rename column
    df.set_index('Data', inplace=True)
    return df

def calculate_log_returns(df): # Calculate log returns of DI1V24
    return np.log(1 + df.pct_change())

def plot_distribution(log_returns): 
    sns.distplot(log_returns.iloc[1:])
    plt.xlabel("Dayly log returns")
    plt.ylabel("Frequency")

def calculate_parameters(log_returns):
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    return drift, stdev

def simulate_monte_carlo(drift, stdev, days, trials):
    Z = norm.ppf(np.random.rand(days, trials))
    return np.exp(drift.values + stdev.values * Z)

def calculate_future_prices(daily_returns, df, days):
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = df.iloc[-1]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
    return pd.DataFrame(price_paths)

def calculate_cumulative_product(df):
    df2 = (df.pct_change() + 1)
    df2.replace({np.nan : EBITDA}, inplace=True)
    df3 = df2.cumprod()
    df3['Media'] = df3.mean(axis=1)
    return df3

def plot(df):
    plt.figure(figsize=(15,6))
    plt.title('Monte Carlo Simulation')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.plot(df)
    plt.show()

def main():
    df = load_and_clean_data(FILE_PATH)
    log_returns = calculate_log_returns(df)
    plot_distribution(log_returns)
    drift, stdev = calculate_parameters(log_returns)
    daily_returns = simulate_monte_carlo(drift, stdev, DAYS, TRIALS)
    price_paths = calculate_future_prices(daily_returns, df, DAYS)
    df4 = calculate_cumulative_product(price_paths)
    print(f'EBITDA {df4.iloc[-1,-1]}')
    plot(df4)

if __name__ == "__main__":
    main()

