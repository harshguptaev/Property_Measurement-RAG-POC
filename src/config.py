"""
Configuration management system with safety features and flexible access patterns.
"""
import os
import yaml
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path


class ConfigLoader:
    """
    Configuration loader with safety controls and runtime modification capabilities.
    """
    
    def __init__(self, config_file: str = "config.yaml", allow_new_keys: bool = True):
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to configuration file
            allow_new_keys: Whether to allow creation of new configuration keys
        """
        self.config_file = config_file
        self.allow_new_keys = allow_new_keys
        self._config = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file with fallback to defaults."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = {}
                logging.warning(f"Config file {self.config_file} not found, using defaults")
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            self._config = {}
            
        # Set default values
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            "model": {
                "text_generation": "anthropic.claude-3-sonnet-20240229-v1:0",
                "text_embedding": "amazon.titan-embed-text-v1",
                "temperature": 0.1,
                "max_tokens": 4096
            },
            "database": {
                "vector_store_type": "faiss",
                "collection_name": "property_documents",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "retrieval": {
                "k": 10,
                "score_threshold": 0.7
            },
            "aws": {
                "region": "us-east-1",
                "bedrock_runtime_endpoint": None
            },
            "processing": {
                "batch_size": 10,
                "max_workers": 4
            },
            "ui": {
                "port": 7860,
                "share": False
            }
        }
        
        # Merge defaults with existing config
        for section, values in defaults.items():
            if section not in self._config:
                self._config[section] = {}
            for key, value in values.items():
                if key not in self._config[section]:
                    self._config[section][key] = value
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if key is None:
            return self._config.get(section, default)
        
        section_config = self._config.get(section, {})
        if not isinstance(section_config, dict):
            return default
        return section_config.get(key, default)
    
    def set(self, section: str, key: str, value: Any, force: bool = False):
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
            force: Force creation of new keys even if allow_new_keys is False
        """
        if section not in self._config:
            if not self.allow_new_keys and not force:
                raise KeyError(f"Section '{section}' not allowed (allow_new_keys=False)")
            self._config[section] = {}
        
        if key not in self._config[section]:
            if not self.allow_new_keys and not force:
                raise KeyError(f"Key '{key}' in section '{section}' not allowed (allow_new_keys=False)")
        
        self._config[section][key] = value
    
    def set_allow_new_keys(self, allow: bool):
        """Set whether new keys can be created."""
        self.allow_new_keys = allow
    
    def get_bedrock_config(self) -> Dict[str, Any]:
        """Get Bedrock-specific configuration."""
        return {
            "region_name": self.get("aws", "region"),
            "endpoint_url": self.get("aws", "bedrock_runtime_endpoint"),
            "model_id": self.get("model", "text_generation"),
            "embedding_model_id": self.get("model", "text_embedding"),
            "temperature": self.get("model", "temperature"),
            "max_tokens": self.get("model", "max_tokens")
        }
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return {
            "store_type": self.get("database", "vector_store_type"),
            "collection_name": self.get("database", "collection_name"),
            "chunk_size": self.get("database", "chunk_size"),
            "chunk_overlap": self.get("database", "chunk_overlap")
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return {
            "k": self.get("retrieval", "k"),
            "score_threshold": self.get("retrieval", "score_threshold")
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Error saving config: {e}")


# Global configuration instance
config = ConfigLoader()
