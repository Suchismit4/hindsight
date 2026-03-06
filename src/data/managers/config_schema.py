"""
Legacy configuration schema (deprecated).

This module is maintained for backward compatibility with existing code.
New code should use the pipeline specification system in src.pipeline.spec instead.

The new system provides:
- Clearer separation of concerns (data vs features vs preprocessing)
- Content-addressable caching
- Explicit dependency tracking
- More extensible design
"""

import warnings
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml


# Emit deprecation warning when this module is imported
warnings.warn(
    "src.data.managers.config_schema is deprecated. "
    "Use src.pipeline.spec for new pipeline specifications.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class DataSourceConfig:
    """Legacy configuration for a single data source."""
    provider: str
    dataset: str
    frequency: Optional[str] = None
    path: Optional[str] = None
    time_var: str = "date"
    identifier_var: str = "identifier"
    filters: Dict[str, Any] = field(default_factory=dict)
    external_tables: List[Dict[str, Any]] = field(default_factory=list)
    processors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Legacy main data configuration container."""
    name: str
    start_date: str
    end_date: str
    cache_path: str = "data/cache/"
    sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    global_processors: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Legacy configuration loader (deprecated).
    
    This is maintained for backward compatibility only.
    New code should use src.pipeline.spec.SpecParser instead.
    """
    
    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> DataConfig:
        """
        Load legacy configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Parsed DataConfig object
        """
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
            
        data_config = raw_config.get('data', {})
        
        config = DataConfig(
            name=data_config.get('name', 'unnamed'),
            start_date=data_config.get('start_date'),
            end_date=data_config.get('end_date'),
            cache_path=data_config.get('cache_path', 'data/cache/'),
            global_processors=data_config.get('global_processors', {})
        )
        
        sources_config = data_config.get('sources', {})
        for source_name, source_config in sources_config.items():
            schema = source_config.get('schema', {})
            
            config.sources[source_name] = DataSourceConfig(
                provider=source_config.get('provider'),
                dataset=source_config.get('dataset'),
                frequency=source_config.get('frequency'),
                path=source_config.get('path'),
                time_var=schema.get('time_var', 'date'),
                identifier_var=schema.get('identifier_var', 'identifier'),
                filters=source_config.get('filters', {}),
                external_tables=source_config.get('external_tables', []),
                processors=source_config.get('processors', {})
            )
            
        return config
    
    @staticmethod
    def convert_to_legacy_format(config: DataConfig) -> Dict[str, Any]:
        """
        Convert configuration to legacy format for backward compatibility.
        
        Args:
            config: DataConfig object
            
        Returns:
            Dictionary in legacy format
        """
        data_sources = []
        
        for source_name, source_config in config.sources.items():
            # Map provider/dataset to data_path
            data_path = ConfigLoader._map_to_data_path(
                source_config.provider,
                source_config.dataset
            )
            
            request = {
                'data_path': data_path,
                'config': {
                    'start_date': config.start_date,
                    'end_date': config.end_date,
                    'frequency': source_config.frequency,
                }
            }
            
            # Add filters if present
            if source_config.filters:
                request['config']['filters'] = source_config.filters
            
            # Add external_tables if present
            if source_config.external_tables:
                request['config']['external_tables'] = source_config.external_tables
            
            # Add processors if present
            if source_config.processors:
                request['config']['processors'] = source_config.processors
            
            data_sources.append(request)
        
        return {
            'data_sources': data_sources
        }
    
    @staticmethod
    def _map_to_data_path(provider: str, dataset: str) -> str:
        """
        Map provider and dataset to data_path format.
        
        Args:
            provider: Provider name
            dataset: Dataset name
            
        Returns:
            Data path string
        """
        # Map common provider/dataset combinations
        provider_map = {
            'wrds': {
                'crsp': 'wrds/equity/crsp',
                'crsp_names': 'wrds/equity/crsp',
                'compustat': 'wrds/equity/compustat',
            },
            'crypto': {
                'binance_spot': 'crypto/spot/binance',
            }
        }
        
        if provider in provider_map and dataset in provider_map[provider]:
            return provider_map[provider][dataset]
        
        # Default format
        return f"{provider}/{dataset}"


class DataConfigBuilder:
    """
    Legacy configuration builder (deprecated).
    
    This is maintained for backward compatibility only.
    """
    
    def __init__(self, name: str, start_date: str, end_date: str):
        """Initialize builder with basic configuration."""
        self.config = DataConfig(
            name=name,
            start_date=start_date,
            end_date=end_date
        )
    
    def add_source(
        self,
        source_name: str,
        provider: str,
        dataset: str,
        frequency: Optional[str] = None
    ) -> 'DataConfigBuilder':
        """Add a data source."""
        self.config.sources[source_name] = DataSourceConfig(
            provider=provider,
            dataset=dataset,
            frequency=frequency
        )
        return self
    
    def build(self) -> DataConfig:
        """Build the final configuration."""
        return self.config

