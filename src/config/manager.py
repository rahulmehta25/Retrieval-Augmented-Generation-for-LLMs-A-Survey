"""
Enhanced Configuration Manager with Pydantic validation and hierarchical loading
"""

import os
import logging
import warnings
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml
import json
from copy import deepcopy

from pydantic import ValidationError
from .models import RAGConfiguration, SecretConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class ConfigurationManager:
    """
    Enhanced configuration manager with hierarchical loading and validation
    
    Features:
    - Hierarchical configuration (base -> environment -> local overrides)
    - Pydantic validation
    - Environment variable overrides
    - Secret management
    - Backward compatibility
    - Configuration validation and diagnostics
    """
    
    # Default configuration paths
    DEFAULT_CONFIG_DIR = Path("config")
    DEFAULT_BASE_CONFIG = "base.yaml"
    
    # Environment variable prefix
    ENV_PREFIX = "RAG_"
    
    def __init__(self,
                 environment: str = "development",
                 config_dir: Optional[Union[str, Path]] = None,
                 base_config: Optional[str] = None,
                 enable_env_overrides: bool = True,
                 validate_on_load: bool = True,
                 strict_mode: bool = False,
                 load_secrets: bool = True):
        """
        Initialize configuration manager
        
        Args:
            environment: Environment name (development, testing, production)
            config_dir: Directory containing configuration files
            base_config: Base configuration file name
            enable_env_overrides: Allow environment variable overrides
            validate_on_load: Validate configuration using Pydantic models
            strict_mode: Fail on validation errors vs warnings
            load_secrets: Load secrets from environment variables
        """
        self.environment = environment
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.base_config = base_config or self.DEFAULT_BASE_CONFIG
        self.enable_env_overrides = enable_env_overrides
        self.validate_on_load = validate_on_load
        self.strict_mode = strict_mode
        self.load_secrets = load_secrets
        
        # Configuration storage
        self._raw_config: Dict[str, Any] = {}
        self._validated_config: Optional[RAGConfiguration] = None
        self._secrets: Optional[SecretConfig] = None
        self._config_sources: List[str] = []
        self._validation_warnings: List[str] = []
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load and merge configuration from multiple sources"""
        try:
            # 1. Load base configuration
            self._load_base_config()
            
            # 2. Load environment-specific configuration
            self._load_environment_config()
            
            # 3. Load local overrides if they exist
            self._load_local_overrides()
            
            # 4. Apply environment variable overrides
            if self.enable_env_overrides:
                self._apply_env_overrides()
            
            # 5. Load secrets
            if self.load_secrets:
                self._load_secrets()
            
            # 6. Validate configuration
            if self.validate_on_load:
                self._validate_configuration()
            
            logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            logger.info(f"Configuration sources: {', '.join(self._config_sources)}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_base_config(self):
        """Load base configuration file"""
        base_path = self.config_dir / self.base_config
        if not base_path.exists():
            raise ConfigurationError(f"Base configuration file not found: {base_path}")
        
        self._raw_config = self._load_yaml_file(base_path)
        self._config_sources.append(str(base_path))
        logger.debug(f"Loaded base configuration from {base_path}")
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        env_config_path = self.config_dir / f"{self.environment}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml_file(env_config_path)
            self._merge_configs(self._raw_config, env_config)
            self._config_sources.append(str(env_config_path))
            logger.debug(f"Loaded environment configuration from {env_config_path}")
        else:
            logger.warning(f"Environment configuration not found: {env_config_path}")
    
    def _load_local_overrides(self):
        """Load local configuration overrides"""
        local_config_path = self.config_dir / "local.yaml"
        if local_config_path.exists():
            local_config = self._load_yaml_file(local_config_path)
            self._merge_configs(self._raw_config, local_config)
            self._config_sources.append(str(local_config_path))
            logger.debug(f"Loaded local overrides from {local_config_path}")
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read {file_path}: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_overrides = {}
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(self.ENV_PREFIX):
                # Convert RAG_SECTION_KEY to section.key
                config_path = env_key[len(self.ENV_PREFIX):].lower().replace('_', '.')
                
                # Parse the value
                parsed_value = self._parse_env_value(env_value)
                
                # Set the value in the config
                self._set_nested_value(env_overrides, config_path, parsed_value)
        
        if env_overrides:
            self._merge_configs(self._raw_config, env_overrides)
            self._config_sources.append("environment_variables")
            logger.debug(f"Applied environment variable overrides: {list(env_overrides.keys())}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Handle special boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try to parse as JSON (for lists, dicts, etc.)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        current = config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Override non-dict values
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def _load_secrets(self):
        """Load secrets from environment variables"""
        try:
            self._secrets = SecretConfig()
            logger.debug("Loaded secrets from environment variables")
        except ValidationError as e:
            if self.strict_mode:
                raise ConfigurationError(f"Secret validation failed: {e}")
            else:
                logger.warning(f"Some secrets failed validation: {e}")
                self._secrets = SecretConfig()  # Load with defaults
    
    def _validate_configuration(self):
        """Validate configuration using Pydantic models"""
        try:
            self._validated_config = RAGConfiguration(**self._raw_config)
            logger.debug("Configuration validation successful")
        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e}"
            
            if self.strict_mode:
                raise ConfigurationError(error_msg)
            else:
                logger.warning(error_msg)
                self._validation_warnings.append(error_msg)
                # Try to create config with partial validation
                try:
                    self._validated_config = RAGConfiguration()
                    logger.warning("Using default configuration due to validation errors")
                except Exception:
                    raise ConfigurationError("Cannot create even default configuration")
    
    def get_config(self) -> RAGConfiguration:
        """Get validated configuration object"""
        if self._validated_config is None:
            raise ConfigurationError("Configuration not loaded or validation failed")
        return self._validated_config
    
    def get_secrets(self) -> SecretConfig:
        """Get secrets configuration"""
        if self._secrets is None:
            raise ConfigurationError("Secrets not loaded")
        return self._secrets
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return deepcopy(self._raw_config)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation path
        
        Args:
            path: Configuration path (e.g., 'system.debug')
            default: Default value if path not found
        """
        keys = path.split('.')
        current = self._raw_config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """
        Set configuration value by dot notation path
        
        Args:
            path: Configuration path (e.g., 'system.debug')
            value: Value to set
        """
        self._set_nested_value(self._raw_config, path, value)
        
        # Re-validate if validation is enabled
        if self.validate_on_load:
            try:
                self._validated_config = RAGConfiguration(**self._raw_config)
            except ValidationError as e:
                if self.strict_mode:
                    raise ConfigurationError(f"Configuration update validation failed: {e}")
                else:
                    logger.warning(f"Configuration update validation warning: {e}")
    
    def reload(self):
        """Reload configuration from files"""
        self._raw_config = {}
        self._validated_config = None
        self._secrets = None
        self._config_sources = []
        self._validation_warnings = []
        
        self._load_configuration()
    
    def save_config(self, output_path: Union[str, Path], include_secrets: bool = False):
        """
        Save current configuration to file
        
        Args:
            output_path: Output file path
            include_secrets: Include secret values (not recommended)
        """
        output_path = Path(output_path)
        
        config_to_save = deepcopy(self._raw_config)
        
        if include_secrets and self._secrets:
            # Add secrets to config (be careful with this!)
            secrets_dict = self._secrets.dict(exclude_none=True)
            if secrets_dict:
                config_to_save['secrets'] = secrets_dict
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            
            # Set restrictive permissions
            if os.name != 'nt':  # Unix-like systems
                os.chmod(output_path, 0o600)
            
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def validate_current_config(self) -> Dict[str, Any]:
        """
        Validate current configuration and return validation report
        
        Returns:
            Dict with validation results
        """
        report = {
            'valid': False,
            'errors': [],
            'warnings': self._validation_warnings.copy(),
            'environment': self.environment,
            'sources': self._config_sources.copy()
        }
        
        try:
            # Try to validate with current config
            test_config = RAGConfiguration(**self._raw_config)
            report['valid'] = True
            report['config_summary'] = {
                'system_name': test_config.system.name,
                'debug_mode': test_config.system.debug,
                'llm_model': test_config.generator.model_name,
                'embedding_model': test_config.embedder.model_name,
                'vector_store': test_config.vector_store.type
            }
        except ValidationError as e:
            for error in e.errors():
                field_path = '.'.join(str(x) for x in error['loc'])
                report['errors'].append({
                    'field': field_path,
                    'message': error['msg'],
                    'value': error.get('input')
                })
        
        return report
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment"""
        return {
            'environment': self.environment,
            'config_dir': str(self.config_dir),
            'sources': self._config_sources,
            'validation_enabled': self.validate_on_load,
            'strict_mode': self.strict_mode,
            'secrets_loaded': self._secrets is not None,
            'env_overrides_enabled': self.enable_env_overrides
        }
    
    def list_available_environments(self) -> List[str]:
        """List available environment configurations"""
        environments = []
        
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name != self.base_config and config_file.name != "local.yaml":
                env_name = config_file.stem
                environments.append(env_name)
        
        return sorted(environments)
    
    def create_environment_template(self, environment: str, base_on: Optional[str] = None):
        """
        Create a new environment configuration template
        
        Args:
            environment: New environment name
            base_on: Base the new config on this environment (default: development)
        """
        base_env = base_on or "development"
        base_path = self.config_dir / f"{base_env}.yaml"
        new_path = self.config_dir / f"{environment}.yaml"
        
        if new_path.exists():
            raise ConfigurationError(f"Environment configuration already exists: {new_path}")
        
        if not base_path.exists():
            raise ConfigurationError(f"Base environment not found: {base_path}")
        
        # Copy base environment and add header comment
        base_config = self._load_yaml_file(base_path)
        
        header_comment = f"""# {environment.title()} Environment Configuration
# Generated from {base_env}.yaml
# Customize this file for the {environment} environment

"""
        
        try:
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(header_comment)
                yaml.dump(base_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created new environment configuration: {new_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment template: {e}")
    
    # Backward compatibility methods
    def load_legacy_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load legacy configuration file and provide migration warnings
        
        Args:
            config_path: Path to legacy configuration file
            
        Returns:
            Legacy configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Legacy configuration file not found: {config_path}")
        
        # Add deprecation warning
        warnings.warn(
            f"Loading legacy configuration from {config_path}. "
            f"Consider migrating to the new hierarchical configuration system.",
            DeprecationWarning,
            stacklevel=2
        )
        
        return self._load_yaml_file(config_path)
    
    def migrate_legacy_config(self, legacy_path: Union[str, Path], 
                            target_environment: str = "development"):
        """
        Migrate legacy configuration to new format
        
        Args:
            legacy_path: Path to legacy configuration
            target_environment: Target environment for migration
        """
        legacy_config = self.load_legacy_config(legacy_path)
        
        # Save as new environment configuration
        target_path = self.config_dir / f"{target_environment}.yaml"
        
        header_comment = f"""# Migrated from legacy configuration: {legacy_path}
# Migration date: {os.path.getmtime(legacy_path)}
# Please review and adjust settings as needed

"""
        
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(header_comment)
                yaml.dump(legacy_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Migrated legacy configuration to {target_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to migrate configuration: {e}")


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(environment: Optional[str] = None, 
                      force_reload: bool = False) -> ConfigurationManager:
    """
    Get global configuration manager instance
    
    Args:
        environment: Environment name (if None, uses RAG_ENVIRONMENT env var or 'development')
        force_reload: Force reload of configuration
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    
    if _config_manager is None or force_reload:
        if environment is None:
            environment = os.environ.get('RAG_ENVIRONMENT', 'development')
        
        _config_manager = ConfigurationManager(environment=environment)
    
    return _config_manager


def get_config() -> RAGConfiguration:
    """Get validated configuration object"""
    return get_config_manager().get_config()


def get_secrets() -> SecretConfig:
    """Get secrets configuration"""
    return get_config_manager().get_secrets()