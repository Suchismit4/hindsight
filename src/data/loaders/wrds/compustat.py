# data/loaders/wrds/compustat.py

import pandas as pd
import xarray as xr
from src.data.core.util import FrequencyType
from typing import Dict, Any, List
from .generic import GenericWRDSDataLoader
import pyreadstat
from dateutil.relativedelta import *
from pandas.tseries.offsets import *

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
        - Adds CCM linkage information for later CRSP linking
        """
        
        df['gvkey'] = df['gvkey'].astype(str)
        df = super()._preprocess_df(
            df,
            date_col='datadate',
            identifier_col='gvkey',
            filters=config.get('filters', {}),
            **config
        )
        
        # Load CCM link table with basic filtering of invalid links
        ccm_path = "/wrds/crsp/sasdata/a_ccm/ccmxpf_linktable.sas7bdat"
        ccm, _ = pyreadstat.read_file_multiprocessing(
            pyreadstat.read_sas7bdat,
            ccm_path,
            num_processes=config.get('num_processes', 16)
        )
        
        # Format column names and filter by standard link type/primacy criteria
        ccm.columns = ccm.columns.str.lower()
        ccm = ccm[
            (ccm['linktype'].str.startswith('L')) & 
            ((ccm['linkprim'] == 'C') | (ccm['linkprim'] == 'P'))
        ]
        
        # Ensure date columns are properly converted to datetime format
        ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
        ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
        
        # Handle missing link end dates - ensure it's datetime type
        ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))
        
        # Ensure 'date' is a datetime object for proper handling
        df['date'] = pd.to_datetime(df['date'])
        
        # Create date fields matching Fama-French methodology
        df['yearend'] = df['date']    + pd.offsets.YearEnd(0)
        df['jdate']   = df['yearend'] + pd.offsets.MonthEnd(6)
        
        # Preserve the original CRSP permno name for clarity
        ccm = ccm.rename(columns={'lpermno': 'permno'})
        
        # Merge CCM data with Compustat
        df = pd.merge(df, ccm, left_on='identifier', right_on='gvkey', how='left')
                
        # Prune the valid links we can do bw comp and CCM.
        valid_links = (
            (df['jdate'] >= df['linkdt']) & 
            (df['jdate'] <= df['linkenddt'])
        )
        df = df[valid_links]
        
        # Set the date to the last day of the year
        df['date'] = df['date'].apply(lambda x: x.replace(month=12, day=31))
        
        # Drop redundant gvkey column since we already have 'identifier'
        if 'gvkey' in df.columns:
            df = df.drop('gvkey', axis=1)
            
        return df