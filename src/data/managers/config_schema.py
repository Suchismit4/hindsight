"""
Enhanced Data Configuration Schema for Hindsight.

This module provides a cleaner, more semantic approach to data loading configuration
that separates concerns and makes the relationship between data sources and processing
more explicit and intuitive.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    provider: str
    dataset: str
    frequency: Optional[str] = None
    path: Optional[str] = None
    
    # Schema definition
    time_var: str = "date"
    identifier_var: str = "identifier"
    
    # Processing pipeline
    processors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Main data configuration container."""
    name: str
    start_date: str
    end_date: str
    cache_path: str = "data/cache/"
    
    # Data sources
    sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    
    # Global processing options
    global_processors: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Enhanced configuration loader that supports the new semantic schema.
    
    This loader provides a cleaner interface for defining data loading pipelines
    with clear separation between data sources and processing steps.
    """
    
    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> DataConfig:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Parsed DataConfig object
            
        Example YAML structure:
        ```yaml
        data:
          name: "equity-analysis"
          start_date: "2020-01-01"
          end_date: "2024-01-01"
          cache_path: "data/cache/"
          
          sources:
            equity_prices:
              provider: "wrds"
              dataset: "crsp"
              frequency: "daily"
              
              schema:
                time_var: "date"
                identifier_var: "permno"
                
              processors:
                filters:
                  share_classes: [10, 11]
                  exchanges: [1, 2, 3]
                  
                merges:
                  - source: "company_names"
                    type: "2d_table"
                    on: "permno"
                    columns: ["comnam", "exchcd"]
                    
                  - source: "distributions"
                    type: "4d_table" 
                    on: "permno"
                    time_column: "exdt"
                    columns: ["divamt", "distcd", "facpr"]
                    
                transforms:
                  - type: "set_coordinates"
                    coord_type: "permno"
                  - type: "fix_market_equity"
                  
            company_names:
              provider: "wrds"
              dataset: "crsp_names"
              
            distributions:
              provider: "wrds"
              dataset: "crsp_distributions"
        ```
        """
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
            
        data_config = raw_config.get('data', {})
        
        # Parse main configuration
        config = DataConfig(
            name=data_config.get('name', 'unnamed'),
            start_date=data_config.get('start_date'),
            end_date=data_config.get('end_date'),
            cache_path=data_config.get('cache_path', 'data/cache/'),
            global_processors=data_config.get('global_processors', {})
        )
        
        # Parse data sources
        sources_config = data_config.get('sources', {})
        for source_name, source_config in sources_config.items():
            # Extract schema if present
            schema = source_config.get('schema', {})
            
            config.sources[source_name] = DataSourceConfig(
                provider=source_config.get('provider'),
                dataset=source_config.get('dataset'),
                frequency=source_config.get('frequency'),
                path=source_config.get('path'),
                time_var=schema.get('time_var', 'date'),
                identifier_var=schema.get('identifier_var', 'identifier'),
                processors=source_config.get('processors', {})
            )
            
        return config
    
    @staticmethod
    def convert_to_legacy_format(config: DataConfig) -> Dict[str, Any]:
        """
        Convert the new configuration format to the legacy format for backward compatibility.
        
        This allows the new semantic configuration to work with the existing
        DataManager infrastructure while we transition. Maintains exact compatibility
        with the cache system by preserving the original request structure.
        
        Args:
            config: The new DataConfig object
            
        Returns:
            Configuration in the legacy format expected by DataManager
        """
        legacy_config = {
            "data_sources": []
        }
        
        # Only process the primary source (ignore auxiliary sources for now)
        primary_sources = [name for name, source_config in config.sources.items() 
                          if source_config.provider and source_config.dataset]
        
        for source_name in primary_sources:
            source_config = config.sources[source_name]
            
            # Map provider/dataset to legacy data_path format
            data_path = ConfigLoader._map_to_data_path(
                source_config.provider, 
                source_config.dataset
            )
            
            # Convert processors to legacy format
            legacy_processors = ConfigLoader._convert_processors(source_config.processors)
            
            # Build legacy source configuration that matches the original format
            # This is critical for cache compatibility
            legacy_source = {
                "data_path": data_path,
                "config": {
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "freq": source_config.frequency or "D",
                    "processors": legacy_processors  # Use Django-style processors
                }
            }
            
            legacy_config["data_sources"].append(legacy_source)
            
        return legacy_config
    
    @staticmethod
    def _map_to_data_path(provider: str, dataset: str) -> str:
        """Map provider/dataset combinations to legacy data_path format."""
        mapping = {
            ("wrds", "crsp"): "wrds/equity/crsp",
            ("wrds", "crsp_names"): "wrds/equity/crsp",  # Same loader, different processing
            ("wrds", "crsp_distributions"): "wrds/equity/crsp",  # Same loader, different processing
            ("wrds", "compustat"): "wrds/equity/compustat",
            ("openbb", "equity_prices"): "openbb/equity/price/historical",
            ("crypto", "binance_spot"): "crypto/spot/binance",
            ("crypto", "spot"): "crypto/spot/binance",  # Alternative naming
        }
        return mapping.get((provider, dataset), f"{provider}/{dataset}")
    
    @staticmethod
    def _convert_processors(processors: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new processor format to legacy format."""
        legacy_processors = {}
        
        # Handle filters
        filters_config = processors.get('filters', {})
        if filters_config:
            legacy_filters = {}
            
            # Convert semantic filter names to Django-style
            if 'share_classes' in filters_config:
                legacy_filters['shrcd__in'] = filters_config['share_classes']
            if 'exchanges' in filters_config:
                legacy_filters['exchcd__in'] = filters_config['exchanges']
                
            if legacy_filters:
                legacy_processors['filters'] = legacy_filters
        
        # Handle merges
        merges_config = processors.get('merges', [])
        # print(f"DEBUG: Merges config: {merges_config}")
        # quit()
        
        if merges_config:
            merge_2d_list = []
            merge_4d_list = []
            replace_values_list = []
            
            for merge in merges_config:
                merge_type = merge.get('type')
                source_path = merge.get('source_path')  # Use source_path instead of source
                
                if merge_type == '2d_table':
                    for column in merge.get('columns', []):
                        merge_2d_list.append({
                            "src": source_path,  # Use src for external file path
                            "identifier": merge['on'],
                            "column": column,
                            "axis": "asset"
                        })
                        
                elif merge_type == '4d_table':
                    merge_4d_list.append({
                        "src": source_path,  # Use src for external file path
                        "variables": merge.get('columns', []),
                        "identifier": merge['on'],
                        "time_column": merge.get('time_column', 'date')
                    })
                    
                elif merge_type == 'replace_values':
                    replace_values_list.append({
                        "src": source_path,  # Use src for external file path
                        "identifier": merge['on'],
                        "from_var": merge.get('from_var'),
                        "to_var": merge.get('to_var'),
                        "rename": [["dlstdt", "time"]] if merge.get('time_column') == 'dlstdt' else None
                    })
            
            if merge_2d_list:
                legacy_processors['merge_table'] = merge_2d_list
            if merge_4d_list:
                legacy_processors['merge_4d_table'] = merge_4d_list[0]  # For now, support one 4D merge
            if replace_values_list:
                legacy_processors['replace_values'] = replace_values_list
                
        # print(f"DEBUG: Legacy processors: {legacy_processors}")
        # quit()
        
        # Handle transforms
        transforms_config = processors.get('transforms', [])
        if transforms_config:  # Only process if not None or empty
            for transform in transforms_config:
                if transform.get('type') == 'set_coordinates':
                    coord_type = transform.get('coord_type', 'permno')
                    if coord_type == 'permno':
                        legacy_processors['set_permno_coord'] = True
                    elif coord_type == 'permco':
                        legacy_processors['set_permco_coord'] = True
                elif transform.get('type') == 'fix_market_equity':
                    legacy_processors['fix_market_equity'] = True
        
        return legacy_processors


class DataConfigBuilder:
    """
    Builder class for programmatically constructing data configurations.
    
    This provides a fluent interface for building configurations when YAML
    files are not desired or when configurations need to be built dynamically.
    """
    
    def __init__(self, name: str, start_date: str, end_date: str):
        self.config = DataConfig(
            name=name,
            start_date=start_date,
            end_date=end_date
        )
    
    def with_cache_path(self, cache_path: str) -> 'DataConfigBuilder':
        """Set the cache path."""
        self.config.cache_path = cache_path
        return self
    
    def add_source(self, name: str, provider: str, dataset: str, **kwargs) -> 'DataConfigBuilder':
        """Add a data source."""
        self.config.sources[name] = DataSourceConfig(
            provider=provider,
            dataset=dataset,
            **kwargs
        )
        return self
    
    def add_merge_2d(self, source_name: str, target_source: str, 
                     on: str, columns: List[str]) -> 'DataConfigBuilder':
        """Add a 2D table merge to a source."""
        if source_name not in self.config.sources:
            raise ValueError(f"Source '{source_name}' not found")
            
        merges = self.config.sources[source_name].processors.setdefault('merges', [])
        merges.append({
            'type': '2d_table',
            'source': target_source,
            'on': on,
            'columns': columns
        })
        return self
    
    def add_merge_4d(self, source_name: str, target_source: str,
                     on: str, columns: List[str], time_column: str = 'date') -> 'DataConfigBuilder':
        """Add a 4D table merge to a source."""
        if source_name not in self.config.sources:
            raise ValueError(f"Source '{source_name}' not found")
            
        merges = self.config.sources[source_name].processors.setdefault('merges', [])
        merges.append({
            'type': '4d_table',
            'source': target_source,
            'on': on,
            'columns': columns,
            'time_column': time_column
        })
        return self
    
    def add_filter(self, source_name: str, **filters) -> 'DataConfigBuilder':
        """Add filters to a source."""
        if source_name not in self.config.sources:
            raise ValueError(f"Source '{source_name}' not found")
            
        filter_config = self.config.sources[source_name].processors.setdefault('filters', {})
        filter_config.update(filters)
        return self
    
    def add_transform(self, source_name: str, transform_type: str, **params) -> 'DataConfigBuilder':
        """Add a transform to a source."""
        if source_name not in self.config.sources:
            raise ValueError(f"Source '{source_name}' not found")
            
        transforms = self.config.sources[source_name].processors.setdefault('transforms', [])
        transforms.append({
            'type': transform_type,
            **params
        })
        return self
    
    def build(self) -> DataConfig:
        """Build the final configuration."""
        return self.config 