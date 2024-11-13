import xarray as xr
import equinox as eqx
import xarray_jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

CACHE_PATH = "~/data/cache/crsp/D/"
EXCLUSION_FEATURE_LIST = ['date', 'permno', 'permco', 'hsiccd', 'hexcd', 'cusip', 'issuno', 'altprcdt']
INCLUSION_FEATURE_LIST = ['date', 'permno', 'bidlo', 'askhi', 'prc', 'vol', 'bid', 'ask', 'shrout', 'cfacpr', 'cfacshr', 'openprc', 'numtrd', 'retx', 'ret']  

LOOKBACK_DAYS = 252  # One year of trading days
QUANTILE = 0.2       # Top and bottom 20%


def load_crsp_data(cache_path: str):    
    # Read the Parquet file with specified columns
    df = pd.read_parquet(cache_path, columns=INCLUSION_FEATURE_LIST)
    
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert 'permno' to categorical to reduce memory usage
    df['permno'] = df['permno'].astype('category')
    
    # Drop duplicates and rows with missing 'date' or 'permno'
    df = df.drop_duplicates(subset=['date', 'permno'], keep='last')
    df = df.dropna(subset=['date', 'permno'])
    
    # Adjust prices on-load
    df['adj_prc'] = df['prc'] / df['cfacpr']
    df['adj_askhi'] = df['askhi'] / df['cfacpr']
    df['adj_bidlo'] = df['bidlo'] / df['cfacpr']
    
    # Keep only necessary columns
    df = df[['date', 'permno', 'adj_prc', 'adj_askhi', 'adj_bidlo', 'ret']]
    
    # Convert to Xarray Dataset
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_xarray.html
    ds = df.set_index(['date', 'permno']).to_xarray() 
    
    # Convert data variables to JAX arrays
    for var in ds.data_vars:
        ds[var].data = jnp.array(ds[var].data)
    
    return ds

@eqx.filter_jit()
def calculate_momentum(ds: xr.Dataset, lookback_days: int):
    # Sort the dataset by date
    ds = ds.sortby('date')

    # Compute cumulative returns over the lookback period
    # First, shift the returns to align correctly
    ds['shifted_ret'] = ds['ret'].shift(date=1)

    # Replace missing returns with zero for cumulative product
    ds['shifted_ret'] = ds['shifted_ret'].fillna(0)

    # Compute log returns for numerical stability
    ds['log_ret'] = xr.apply_ufunc(jnp.log1p, ds['shifted_ret'])

    # Compute rolling sum of log returns
    
    # CANNOT USE ROLLING (refer:   File "site-packages/xarray/core/rolling.py", line 414, in _construct
     #  window = obj.variable.rolling_window()
    ds['momentum'] = ds['log_ret'].rolling(date=lookback_days, min_periods=lookback_days).sum()

    # Exponentiate to get cumulative returns    
    ds['momentum'] = xr.apply_ufunc(jnp.expm1, ds['momentum'])

    return ds

@eqx.filter_jit
def rank_assets(ds: xr.Dataset, quantile: float):
    
    # Rank assets based on momentum
    def rank_by_momentum(momentum):
        ranks = momentum.rank(dim='permno', pct=True)
        return ranks
    
    ds['rank'] = ds['momentum'].groupby('date').map(rank_by_momentum)
    
    # Determine long and short positions
    ds['long'] = ds['rank'] >= (1 - quantile)
    ds['short'] = ds['rank'] <= quantile
    
    return ds

@eqx.filter_jit
def compute_strategy_returns(ds: xr.Dataset):
    # Calculate daily returns for long and short positions
    ds['long_ret'] = ds['ret'] * ds['long']
    ds['short_ret'] = ds['ret'] * ds['short']
    
    # Assume equal weighting
    long_counts = ds['long'].sum(dim='permno')
    short_counts = ds['short'].sum(dim='permno')
    
    # Avoid division by zero
    long_counts = long_counts.where(long_counts != 0, other=1)
    short_counts = short_counts.where(short_counts != 0, other=1)
    
    ds['long_ret_avg'] = ds['long_ret'].sum(dim='permno') / long_counts
    ds['short_ret_avg'] = ds['short_ret'].sum(dim='permno') / short_counts
    
    # Strategy return is long minus short
    ds['strategy_ret'] = ds['long_ret_avg'] - ds['short_ret_avg']
    
    # Cumulative returns
    ds['strategy_cum_ret'] = (1 + ds['strategy_ret']).cumprod(dim='date') - 1
    
    return ds


def main():
    # Load the CRSP data into an Xarray Dataset with JAX arrays
    ds = load_crsp_data(CACHE_PATH + 'data.parquet')
    print("Loaded data....")
    
    # Calculate momentum
    ds = calculate_momentum(ds, LOOKBACK_DAYS)
    print("Computed momentum....")
    
    # Rank assets and determine positions
    ds = rank_assets(ds, QUANTILE)
    
    # Compute strategy returns
    ds = compute_strategy_returns(ds)
    
    # Extract the strategy cumulative returns
    strategy_cum_ret = ds['strategy_cum_ret']
    
    strategy_cum_ret_df = strategy_cum_ret.to_dataframe().reset_index()
    
    print(strategy_cum_ret_df.head())
    
    # Adjust the figure size and DPI for better quality
    plt.figure(figsize=(12, 6), dpi=300)
    strategy_cum_ret_df.plot(x='date', y='strategy_cum_ret')
    plt.title('Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid()
    plt.savefig('strategy.png', dpi=300)
    plt.close()  

if __name__ == "__main__":
    main()