"""
Configuration adapter for backward compatibility

This module provides compatibility layers for existing code that uses
the old configuration format, allowing gradual migration to the new system.
"""

import warnings
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from .manager import get_config_manager, get_config


class LegacyConfigAdapter:
    """
    Adapter to provide backward compatibility with old configuration usage patterns
    """
    
    def __init__(self):
        self._config_manager = None
        self._warned_methods = set()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Legacy method: Load configuration from file
        
        This method maintains compatibility with old code while encouraging migration
        """
        self._warn_deprecated("load_config", "Use ConfigurationManager.load_legacy_config() instead")
        
        if not hasattr(self, '_config_manager') or self._config_manager is None:
            self._config_manager = get_config_manager()
        
        return self._config_manager.load_legacy_config(config_path)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Legacy method: Get configuration value by key
        
        Supports both old flat key names and new dot notation
        """
        self._warn_deprecated("get_config_value", "Use get_config() and access attributes directly")
        
        config = get_config()
        
        # Handle common legacy key mappings
        legacy_mappings = {
            'chunk_size': 'text_splitter.chunk_size',
            'chunk_overlap': 'text_splitter.chunk_overlap',
            'model_name': 'generator.model_name',
            'embedding_model': 'embedder.model_name',
            'vector_store_path': 'vector_store.path',
            'collection_name': 'vector_store.collection_name',
            'temperature': 'generator.temperature',
            'max_tokens': 'generator.max_tokens',
            'top_k': 'retrieval.k_documents',
            'debug': 'system.debug',
            'max_workers': 'system.max_workers',
            'timeout': 'system.timeout'
        }
        
        # Try legacy mapping first
        if key in legacy_mappings:
            key = legacy_mappings[key]
        
        # Get value using dot notation
        return self._get_nested_value(config.dict(), key, default)
    
    def _get_nested_value(self, config_dict: Dict[str, Any], key: str, default: Any) -> Any:
        """Get nested value using dot notation"""
        keys = key.split('.')
        current = config_dict
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def _warn_deprecated(self, method_name: str, suggestion: str):
        """Warn about deprecated method usage"""
        if method_name not in self._warned_methods:
            warnings.warn(
                f"Method '{method_name}' is deprecated. {suggestion}",
                DeprecationWarning,
                stacklevel=3
            )
            self._warned_methods.add(method_name)


# Global adapter instance for backward compatibility
_legacy_adapter = LegacyConfigAdapter()


def load_config_yaml(config_path: str) -> Dict[str, Any]:
    """
    Legacy function: Load YAML configuration
    
    DEPRECATED: Use the new ConfigurationManager instead
    """
    warnings.warn(
        "load_config_yaml() is deprecated. Use ConfigurationManager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _legacy_adapter.load_config(config_path)


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Legacy function: Get configuration value
    
    DEPRECATED: Use get_config() and access attributes directly
    """
    return _legacy_adapter.get_config_value(key, default)


def create_config_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy function: Process configuration dictionary
    
    DEPRECATED: Use Pydantic models for validation
    """
    warnings.warn(
        "create_config_from_dict() is deprecated. Use RAGConfiguration model instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return config_dict


def migrate_old_config_usage():
    """
    Utility function to help identify and migrate old configuration usage
    
    This function can be called to scan for deprecated usage patterns
    """
    migration_guide = """
    Migration Guide for New Configuration System:
    
    OLD WAY:
        import yaml
        config = yaml.safe_load(open('config.yaml'))
        chunk_size = config.get('chunk_size', 500)
        
    NEW WAY:
        from src.config import get_config
        config = get_config()
        chunk_size = config.text_splitter.chunk_size
    
    OLD WAY:
        from src.rag.naive_rag import load_config
        config = load_config('config.yaml')
        
    NEW WAY:
        from src.config import get_config_manager
        manager = get_config_manager(environment='development')
        config = manager.get_config()
    
    Benefits of New System:
    - Type safety with Pydantic validation
    - Environment-specific configurations
    - Environment variable overrides
    - Secret management
    - Better error handling
    - Hierarchical configuration merging
    """
    
    print(migration_guide)


class ConfigurationMigrationHelper:
    """
    Helper class to assist with migrating from old to new configuration system
    """
    
    @staticmethod
    def convert_legacy_config(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy configuration format to new hierarchical format
        
        Args:
            legacy_config: Old flat configuration dictionary
            
        Returns:
            New hierarchical configuration dictionary
        """
        new_config = {
            'system': {},
            'text_splitter': {},
            'embedder': {},
            'vector_store': {},
            'generator': {},
            'retrieval': {},
            'context': {},
            'conversation': {},
            'security': {},
            'api': {},
            'logging': {},
            'monitoring': {},
            'cache': {},
            'evaluation': {},
            'features': {},
            'storage': {},
            'error_handling': {},
            'performance': {}
        }
        
        # Map old keys to new structure
        key_mappings = {
            # Text splitter
            'chunk_size': ('text_splitter', 'chunk_size'),
            'chunk_overlap': ('text_splitter', 'chunk_overlap'),
            'text_splitter': ('text_splitter', 'type'),
            
            # Embedder
            'embedding_model': ('embedder', 'model_name'),
            'embedder': ('embedder', 'type'),
            'cache_dir': ('embedder', 'cache_dir'),
            
            # Vector store
            'vector_store': ('vector_store', 'type'),
            'vector_store_path': ('vector_store', 'path'),
            'collection_name': ('vector_store', 'collection_name'),
            
            # Generator
            'generator': ('generator', 'type'),
            'model_name': ('generator', 'model_name'),
            'temperature': ('generator', 'temperature'),
            'max_tokens': ('generator', 'max_tokens'),
            'top_p': ('generator', 'top_p'),
            
            # Retrieval
            'top_k': ('retrieval', 'k_documents'),
            'reranker': ('retrieval', 'reranker_model'),
            
            # System
            'debug': ('system', 'debug'),
            'max_workers': ('system', 'max_workers'),
            'timeout': ('system', 'timeout'),
            
            # Context
            'max_context_tokens': ('context', 'max_context_length'),
            
            # Conversation
            'conversation_memory': ('conversation', 'memory_enabled'),
            'max_turns': ('conversation', 'max_turns'),
            
            # Cache
            'cache_enabled': ('system', 'cache_enabled'),
            'cache_ttl': ('system', 'cache_ttl'),
        }
        
        # Apply mappings
        for old_key, value in legacy_config.items():
            if old_key in key_mappings:
                section, new_key = key_mappings[old_key]
                new_config[section][new_key] = value
            else:
                # Try to guess the section based on key name
                if 'embed' in old_key.lower():
                    new_config['embedder'][old_key] = value
                elif 'vector' in old_key.lower() or 'chroma' in old_key.lower():
                    new_config['vector_store'][old_key] = value
                elif 'llm' in old_key.lower() or 'model' in old_key.lower():
                    new_config['generator'][old_key] = value
                elif 'retriev' in old_key.lower() or 'search' in old_key.lower():
                    new_config['retrieval'][old_key] = value
                else:
                    # Put in system section as fallback
                    new_config['system'][old_key] = value
        
        # Remove empty sections
        return {k: v for k, v in new_config.items() if v}
    
    @staticmethod
    def save_migrated_config(legacy_config: Dict[str, Any], 
                           output_path: str = "config/migrated.yaml"):
        """
        Save migrated configuration to new format
        
        Args:
            legacy_config: Legacy configuration dictionary
            output_path: Output file path for migrated config
        """
        new_config = ConfigurationMigrationHelper.convert_legacy_config(legacy_config)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Migrated configuration from legacy format\n")
            f.write("# Please review and adjust settings as needed\n\n")
            yaml.dump(new_config, f, default_flow_style=False, indent=2)
        
        print(f"Migrated configuration saved to: {output_path}")
    
    @staticmethod
    def validate_migration(legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that migration will work correctly
        
        Args:
            legacy_config: Legacy configuration to validate
            
        Returns:
            Validation report
        """
        report = {
            'migration_possible': True,
            'warnings': [],
            'unmapped_keys': [],
            'conflicts': []
        }
        
        known_keys = {
            'chunk_size', 'chunk_overlap', 'text_splitter',
            'embedding_model', 'embedder', 'cache_dir',
            'vector_store', 'vector_store_path', 'collection_name',
            'generator', 'model_name', 'temperature', 'max_tokens', 'top_p',
            'top_k', 'reranker', 'debug', 'max_workers', 'timeout',
            'max_context_tokens', 'conversation_memory', 'max_turns',
            'cache_enabled', 'cache_ttl'
        }
        
        for key in legacy_config.keys():
            if key not in known_keys:
                report['unmapped_keys'].append(key)
        
        if report['unmapped_keys']:
            report['warnings'].append(
                f"Some keys may not be properly migrated: {report['unmapped_keys']}"
            )
        
        return report