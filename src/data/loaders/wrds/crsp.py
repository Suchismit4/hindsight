# data/loaders/wrds/crsp.py

import pandas as pd
import xarray as xr
from src.data.core.util import FrequencyType
from .generic import GenericWRDSDataLoader
import pyreadstat


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
            
        # 1. Merge msenames: company names
        try:
            names_path = "/wrds/crsp/sasdata/a_stock/msenames.sas7bdat"
            names_df, _ = pyreadstat.read_file_multiprocessing(
                pyreadstat.read_sas7bdat,
                names_path,
                num_processes=config.get('num_processes', 16)
            )
            names_df.columns = names_df.columns.str.lower()
            # Rename key columns as needed
            if 'permno' in names_df.columns and 'comnam' in names_df.columns:
                names_df.rename(columns={'permno': 'identifier', 'comnam': 'company_name'}, inplace=True)
            else:
                raise KeyError("Expected columns 'permno' and 'comnam' not found in msenames.")
            # Merge company names
            df = pd.merge(df, names_df[['identifier', 'company_name']], on='identifier', how='left')
        except Exception as e:
            print(f"Warning: Could not merge msenames data due to error: {e}")

        # 2. Merge msedist: distributions (e.g., dividends, repurchases)
        try:
            dist_path = "/wrds/crsp/sasdata/a_stock/msedist.sas7bdat"
            dist_df, _ = pyreadstat.read_file_multiprocessing(
                pyreadstat.read_sas7bdat,
                dist_path,
                num_processes=config.get('num_processes', 16)
            )
            dist_df.columns = dist_df.columns.str.lower()
            # Rename and select relevant columns; adjust as necessary.
            # Assuming msedist contains 'permno', 'date', and distribution details.
            if 'permno' in dist_df.columns and 'date' in dist_df.columns:
                dist_df.rename(columns={'permno': 'identifier'}, inplace=True)
                # Convert SAS date to datetime for merging, if needed.
                dist_df['date'] = self.convert_sas_date(dist_df['date'])
            else:
                raise KeyError("Expected columns 'permno' and 'date' not found in msedist.")
            # Merge distribution data on 'identifier' and 'date'
            # Using suffixes to avoid collisions if same column names exist.
            df = pd.merge(
                df,
                dist_df,
                on=['identifier', 'date'],
                how='left',
                suffixes=('', '_dist')
            )
        except Exception as e:
            print(f"Warning: Could not merge msedist data due to error: {e}")

        # 3. Merge msedelist: delisting information
        try:
            delist_path = "/wrds/crsp/sasdata/a_stock/msedelist.sas7bdat"
            delist_df, _ = pyreadstat.read_file_multiprocessing(
                pyreadstat.read_sas7bdat,
                delist_path,
                num_processes=config.get('num_processes', 16)
            )
            delist_df.columns = delist_df.columns.str.lower()
            # Rename and preprocess columns as necessary.
            if 'permno' in delist_df.columns and 'dlstdt' in delist_df.columns:
                delist_df.rename(columns={'permno': 'identifier', 'dlstdt': 'delist_date'}, inplace=True)
                # Convert SAS date to datetime for merging, if needed.
                delist_df['delist_date'] = self.convert_sas_date(delist_df['delist_date'])
            else:
                raise KeyError("Expected columns 'permno' and 'dlstdt' not found in msedelist.")
            # Merge delisting data.
            # This merge strategy depends on how you want to incorporate delisting info.
            # Here, we add delisting date information to each record if available.
            df = pd.merge(
                df,
                delist_df[['identifier', 'delist_date']],
                on='identifier',
                how='left',
                suffixes=('', '_delist')
            )
        except Exception as e:
            print(f"Warning: Could not merge msedelist data due to error: {e}")

        return df
