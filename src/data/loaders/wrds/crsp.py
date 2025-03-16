# data/loaders/wrds/crsp.py

import pandas as pd
import xarray as xr
from src.data.core.util import FrequencyType, Loader as ld
from .generic import GenericWRDSDataLoader
import pyreadstat
import numpy as np

class CRSPDataFetcher(GenericWRDSDataLoader):
    """
    Data loader for CRSP stock data.
    
    - Dynamically sets the LOCAL_SRC path based on the frequency string:
        * "D" -> dsf.sas7bdat (daily)
        * "M" -> msf.sas7bdat (monthly)
      If an unrecognized frequency is provided, defaults to daily.
    - Then calls the generic loader.
    - **Note:** The msenames table (company names) is no longer merged into the main
      dataframe. Instead, it is loaded separately and attached as a new variable in the dataset.
    """

    # Map user frequency strings to (filename, FrequencyType)
    CRSP_FREQUENCY_MAP = {
        'D': ('dsf.sas7bdat', FrequencyType.DAILY),
        'M': ('msf.sas7bdat', FrequencyType.MONTHLY),
    }

    def load_data(self, **config) -> xr.Dataset:
        """
        Adjust LOCAL_SRC depending on the requested frequency, call the generic loader,
        and then attach company names from the separate msenames table as a new DataArray.
        """
        user_freq_str = str(config.get('freq', 'D')).upper()
        
        # Find the correct file and frequency; default to daily if not found.
        filename, freq_enum = self.CRSP_FREQUENCY_MAP.get(
            user_freq_str, 
            ('dsf.sas7bdat', FrequencyType.DAILY)
        )
        
        # Construct the path for CRSP a_stock
        self.LOCAL_SRC = f"/wrds/crsp/sasdata/a_stock/{filename}"
        self.FREQUENCY = freq_enum
        
        # Load the main dataset via the generic loader.
        ds = super().load_data(**config)
        
        return ds

    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        CRSP-specific preprocessing:
         - Calls the generic _preprocess_df with date_col='date', identifier_col='permno'.
         - Converts 'permco' to int if present.
         - Merging of msenames is removed to avoid duplicating company names.
         - Merges msedist (distributions) and msedelist (delisting information) remain.
        """
                    
        df = super()._preprocess_df(
            df,
            date_col='date',
            identifier_col='permno',
            filters_config=config.get('filters_config', {})
        )

        # Convert 'permco' to int if present.
        if 'permco' in df.columns:
            df.loc[:, 'permco'] = df['permco'].astype(int)  # in-place
            
        return df
