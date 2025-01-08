# data/loaders/wrds/compustat.py

import pandas as pd
import xarray as xr
from src.data.core.util import FrequencyType
from typing import Dict, Any, List
from .generic import GenericWRDSDataLoader

class CompustatDataFetcher(GenericWRDSDataLoader):
    """
    Data loader for Compustat data.

    - Dynamically sets LOCAL_SRC based on frequency:
        * "Y" -> funda.sas7bdat (annual)
        * "Q" -> fundq.sas7bdat (quarterly)
      Defaults to annual if not recognized.
    """

    COMP_FREQUENCY_MAP = {
        'Y': ('funda.sas7bdat', FrequencyType.YEARLY),
    }

    def load_data(self, **config) -> xr.Dataset:
        """
        Determine which Compustat file to load (annual vs. quarterly) 
        based on user-supplied frequency, then call the generic loader.
        """
        user_freq_str = str(config.get('frequency', 'Y')).upper()
        
        # Pick from the map or default to annual
        filename, freq_enum = self.COMP_FREQUENCY_MAP.get(
            user_freq_str,
            ('funda.sas7bdat', FrequencyType.YEARLY)
        )

        # Construct the path
        self.LOCAL_SRC = f"/wrds/comp/sasdata/d_na/{filename}"
        self.FREQUENCY = freq_enum

        return super().load_data(**config)

    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        Compustat-specific preprocessing:
         - date_col='datadate' -> 'date'
         - identifier_col='gvkey' -> 'identifier'
        """
        df = super()._preprocess_df(
            df,
            date_col='datadate',
            identifier_col='gvkey',
            **config
        )
        
        # CompuStat has duplicate multiple entries on some timeframes.
        # We keep only the last one and forward dates to date end.
        # Ensure 'date' is a datetime object for proper comparison
        df['date'] = pd.to_datetime(df['date'])
            
        # Sort by 'date' and drop duplicates while keeping the last occurrence
        df = df.sort_values('date', ascending=False).drop_duplicates(subset='identifier', keep='last')
            
        # Set the date to the last day of the year
        df['date'] = df['date'].apply(lambda x: x.replace(month=12, day=31))
            
        return df