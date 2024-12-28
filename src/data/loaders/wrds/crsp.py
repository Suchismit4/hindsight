# data/loaders/wrds/crsp.py

import pandas as pd
from src.data.core.struct import FrequencyType
from .generic import GenericWRDSDataLoader

class CRSPDataFetcher(GenericWRDSDataLoader):
    """
    Data loader for CRSP data.

    Inherits the generic logic from GenericWRDSDataLoader, 
    while overriding the path, frequency, and any CRSP-specific transformations.
    """

    LOCAL_SRC: str = "/wrds/crsp/sasdata/a_stock/dsf.sas7bdat"
    FREQUENCY: FrequencyType = FrequencyType.DAILY

    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        CRSP-specific preprocessing:
          - SAS date conversion for 'date'
          - Convert 'permco'/'permno' -> integer
          - Rename 'permno' -> 'identifier'
          - Reorder columns, etc.
        """
        df.columns = df.columns.str.lower()
        df.reset_index(inplace=True, drop=True)

        # Convert SAS date integer to real datetime
        sas_epoch = pd.to_datetime('1960-01-01')
        if 'date' in df.columns:
            df['date'] = df['date'].astype(int)
            df['date'] = sas_epoch + pd.to_timedelta(df['date'], unit='D')

        # Convert permco/permno to int
        for col in ['permco', 'permno']:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Rename 'permno' -> 'identifier'
        if 'permno' in df.columns:
            df.rename(columns={'permno': 'identifier'}, inplace=True)

        # Order columns so 'Sindate' and 'identifier' come first
        required_cols = ['date', 'identifier']
        other_cols = [c for c in df.columns if c not in required_cols]
        df = df[required_cols + other_cols]

        return df
