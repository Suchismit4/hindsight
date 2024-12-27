import pandas as pd
import numpy as np
from src.data.core import DatasetDateTimeAccessor, FrequencyType

def generate_mock_crsp_data(
    num_identifiers=5, 
    start='1925-12-31', 
    end='2023-12-31'
) -> pd.DataFrame:
    """
    Generate a mock CRSP-like DataFrame with yearly data.
    
    Parameters:
        num_identifiers (int): How many distinct identifiers to create.
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: Emulated CRSP data with columns similar to what you listed.
    """

    # Generate a yearly date range ending in December each year.
    # freq='A' or freq='Y' both produce year-end dates (Dec 31).
    dates = pd.date_range(start, end, freq='A')  # e.g., 1925-12-31, 1926-12-31, ...
    
    # Create some identifiers (could be PERMNO, or anything).
    identifiers = np.arange(10006, 10006 + num_identifiers)
    
    # Cartesian product of all (date, identifier).
    df_index = pd.MultiIndex.from_product([dates, identifiers], names=['date', 'identifier'])
    
    # Initialize DataFrame from the MultiIndex, then reset index to make them columns.
    df = pd.DataFrame(index=df_index).reset_index()

    # For convenience
    n = len(df)

    # Fill in columns typically seen in CRSP-like data
    # These are random or semi-random just for demonstration.
    df['cusip'] = np.random.choice(['00080010','00299090','00338690','00369890','00462610','92835K10','G3323L10','12503M10','78513510','88160R10'], size=n)
    df['permco'] = np.random.randint(20000, 60000, size=n).astype(float)
    df['issuno'] = np.zeros(n)  # often 0.0 in many lines
    df['hexcd'] = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n) 
    df['hsiccd'] = np.random.randint(1000, 9999, size=n).astype(float)
    
    # Price/volume-like columns
    df['bidlo'] = np.random.uniform(1, 300, size=n)
    df['askhi'] = df['bidlo'] + np.random.uniform(0, 5, size=n)  # askhi slightly > bidlo
    df['prc']   = np.round(np.random.uniform(-300, 300, size=n), 3)  # CRSP can store negative as an indication
    df['vol']   = np.random.randint(0, 1_000_000, size=n).astype(float)
    df['ret']   = np.random.uniform(-0.1, 0.1, size=n)  # daily/annual returns, here random
    df['bid']   = df['bidlo'] + 0.3  # or random offset
    df['ask']   = df['askhi'] - 0.3
    
    # Corporate action columns
    df['shrout']   = np.random.randint(100, 5_000_000, size=n).astype(float)
    df['cfacpr']   = np.ones(n)      # e.g., split adjustment factor
    df['cfacshr']  = np.ones(n)      # e.g., share adjustment factor
    df['openprc']  = df['bidlo'] + np.random.uniform(0, 2, size=n)
    df['numtrd']   = np.random.randint(0, 5000, size=n).astype(float)
    df['retx']     = df['ret']  # ex-dividend return, for example

    # The final DataFrame
    return df

if __name__ == "__main__":
    df_mock = generate_mock_crsp_data(num_identifiers=5)
    
    # Show just the head
    print(df_mock.head(15))

    # Optionally save to parquet
    DatasetDateTimeAccessor.from_table(
            df_mock,
            time_column='date',
            asset_column='identifier',
            feature_columns=['issuno', 'hexcd', 'hsiccd',
       'bidlo', 'askhi', 'prc', 'vol', 'ret', 'bid', 'ask', 'shrout', 'cfacpr',
       'cfacshr', 'openprc', 'numtrd', 'retx'],
            frequency=FrequencyType.ANNUAL
        )