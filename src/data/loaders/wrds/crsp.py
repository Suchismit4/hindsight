# data/loaders/wrds/crsp.py

import pandas as pd
import xarray as xr
from src.data.core.util import FrequencyType
from .generic import GenericWRDSDataLoader

class CRSPDataFetcher(GenericWRDSDataLoader):
    """
    Data loader for CRSP stock data.
    
    - Dynamically sets the LOCAL_SRC path based on the frequency string:
        * "D" -> dsf.sas7bdat (daily)
        * "M" -> msf.sas7bdat (monthly)
      If an unrecognized frequency is provided, defaults to daily.
    - Then calls the generic loader.
    """

    # Map user frequency strings to (filename, FrequencyType)
    CRSP_FREQUENCY_MAP = {
        'D': ('dsf.sas7bdat', FrequencyType.DAILY),
        'M': ('msf.sas7bdat', FrequencyType.MONTHLY),
    }

    def load_data(self, **config) -> xr.Dataset:
        """
        Adjust LOCAL_SRC depending on the requested frequency, then call the generic loader.
        """
        user_freq_str = str(config.get('frequency', 'D')).upper()
        
        # Find the correct path + enum from the map; default to daily if not found
        filename, freq_enum = self.CRSP_FREQUENCY_MAP.get(
            user_freq_str, 
            ('dsf.sas7bdat', FrequencyType.DAILY)
        )
        
        # Construct the path for CRSP a_stock
        self.LOCAL_SRC = f"/wrds/crsp/sasdata/a_stock/{filename}"
        self.FREQUENCY = freq_enum

        return super().load_data(**config)

    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        CRSP-specific preprocessing:
         - Calls the generic _preprocess_df with date_col='date', identifier_col='permno'.
         - Converts 'permco' to int if present.
        """
        
        df = super()._preprocess_df(
            df,
            date_col='date',
            identifier_col='permno',
            **config
        )

        # convert 'permco' to int if present
        if 'permco' in df.columns:
            df['permco'] = df['permco'].astype(int)

        return df
