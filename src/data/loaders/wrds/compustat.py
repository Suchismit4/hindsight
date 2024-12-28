# data/loaders/wrds/compustat.py

import pandas as pd
from src.data.core.struct import FrequencyType
from typing import Dict, Any, List
from .generic import GenericWRDSDataLoader

class CompustatDataFetcher(GenericWRDSDataLoader):
    """
    Data loader for Compustat data.

    Inherits from GenericWRDSDataLoader and implements 
    any dataset-specific transformations or filtering logic.
    """

    LOCAL_SRC: str = "/wrds/comp/sasdata/d_na/funda.sas7bdat"  # annual data
    FREQUENCY: FrequencyType = FrequencyType.YEARLY

    def _preprocess_df(self, df: pd.DataFrame, **config) -> pd.DataFrame:
        """
        Compustat-specific preprocessing:
          - SAS date conversion for 'datadate'
          - Rename 'datadate' -> 'date'
          - Rename 'gvkey' -> 'identifier'
          - Apply filters if provided
        """
        # Convert 'datadate' to datetime from SAS integer
        sas_epoch = pd.to_datetime('1960-01-01')
        if 'datadate' in df.columns:
            df['datadate'] = sas_epoch + pd.to_timedelta(df['datadate'], unit='D')
            df.rename(columns={'datadate': 'date'}, inplace=True)

        # Rename 'gvkey' -> 'identifier'
        if 'gvkey' in df.columns:
            df.rename(columns={'gvkey': 'identifier'}, inplace=True)

        # Apply filters if provided
        filters: Dict[str, Any] = config.get('filters', {})
        for col, val in filters.items():
            # Simple equality filters (extend as needed)
            df = df[df[col] == val]

        # Reorder columns
        required_cols = ['date', 'identifier']
        other_cols = [c for c in df.columns if c not in required_cols]
        df = df[required_cols + other_cols]

        # Sort for consistency
        df.sort_values(['date', 'identifier'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
