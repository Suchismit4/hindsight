# hindsight/main.py

from src import *
import xarray as xr
import xarray_jax as xj
import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx

def create_returns_dataframe(
    start_date: str,
    end_date: str,
    assets: list,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Creates a DataFrame of sample daily returns for a list of assets.
    
    Parameters:
        start_date: Start date for time series
        end_date: End date for time series
        assets: List of asset identifiers
        freq: Data frequency ('D' for daily)
    """
    # Create time index
    dates = pd.date_range(start_date, end_date, freq=freq)
    
    # Generate sample returns data
    data = {
        'time': np.repeat(dates, len(assets)),
        'asset': assets * len(dates),
        'returns': np.random.randn(len(dates) * len(assets))
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce some missing data (10% missing)
    mask = np.random.random(len(df)) > 0.9
    df.loc[mask, 'returns'] = np.nan
    
    return df

def create_characteristics_dataframe(
    start_date: str,
    end_date: str,
    assets: list,
    characteristics: list,
    freq: str = 'Q'
) -> pd.DataFrame:
    """
    Creates a DataFrame of sample characteristics data for a list of assets.
    
    Parameters:
        start_date: Start date for time series
        end_date: End date for time series
        assets: List of asset identifiers
        characteristics: List of characteristic names
        freq: Data frequency ('Q' for quarterly)
    """
    # Create time index
    dates = pd.date_range(start_date, end_date, freq=freq)
    
    # Generate sample characteristics data
    data_list = []
    for date in dates:
        for asset in assets:
            row = {
                'time': date,
                'asset': asset
            }
            for char in characteristics:
                row[char] = np.random.randn()
            data_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Introduce some missing data (10% missing)
    mask = np.random.random(len(df)) > 0.9
    for char in characteristics:
        df.loc[mask, char] = np.nan
    
    return df

def main():

    start_date = '2015-01-01'
    end_date = '2020-12-31'
    
    # Assets for returns and characteristics (intentionally different)
    assets_returns = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
    assets_characteristics = ['AAPL', 'MSFT', 'TSLA', 'NFLX']
    characteristics = ['PE_ratio', 'Market_Cap']
    
    print("Creating sample returns and characteristics datasets...")
    
    # Create returns DataFrame
    returns_df = create_returns_dataframe(
        start_date, 
        end_date, 
        assets_returns, 
        freq='D'
    )
        
    # Create characteristics DataFrame
    characteristics_df = create_characteristics_dataframe(
        start_date, 
        end_date, 
        assets_characteristics, 
        characteristics, 
        freq='Q'
    )
        
    # Use from_table to create DataArrays
    returns_data = DataArrayDateTimeAccessor.from_table(
        returns_df,
        time_column='time',
        asset_column='asset',
        feature_columns=['returns'],
        frequency='D'
    )
    
    
    # characteristics_data = DataArrayDateTimeAccessor.from_table(
    #     characteristics_df,
    #     time_column='time',
    #     asset_column='asset',
    #     feature_columns=characteristics,
    #     frequency='Q'
    # )
    
    print("\nReturns DataArray shape:", returns_data.shape)
    # print("Characteristics DataArray shape:", characteristics_data.shape)
        
    # # Test alignment between returns data and characteristics data
    # print("\nAligning returns data with characteristics data...")
    # # Use the dt accessor to align datasets (assuming align_with is properly implemented)
    # aligned_data = returns_data.dt.align_with(
    #     characteristics_data,
    #     method='outer',
    #     freq_method='ffill'
    # )
    
    # print("\nSuccessfully aligned datasets")
    # print("Merged Data:", aligned_data)
    
    # print("\nSample of merged data:")
    # # For demonstration purposes, we'll just print the first few entries
    # print(merged_data.isel(time=slice(0,5)).to_dataframe().head())

if __name__ == "__main__":
    main()
