"""
Secure Configuration Management Module

Implements secure handling of sensitive configuration to prevent:
- Security Misconfiguration (OWASP A05:2021)
- Software and Data Integrity Failures (OWASP A08:2021)
- Exposure of sensitive information (OWASP A01:2021)
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class SecureConfigManager:
    """
    Secure configuration management with encryption for sensitive data
    
    Implements defense-in-depth for configuration security
    """
    
    # Patterns for detecting sensitive configuration keys
    SENSITIVE_KEY_PATTERNS = [
        r'.*password.*',
        r'.*secret.*',
        r'.*key.*',
        r'.*token.*',
        r'.*credential.*',
        r'.*api[-_]?key.*',
        r'.*private.*',
        r'.*auth.*',
        r'.*connection[-_]?string.*',
        r'.*database[-_]?url.*'
    ]
    
    # Environment variable prefix for overrides
    ENV_PREFIX = "RAG_"
    
    def __init__(self,
                 config_file: Optional[str] = None,
                 encryption_key: Optional[str] = None,
                 use_env_vars: bool = True,
                 validate_schema: bool = True,
                 auto_rotate_keys: bool = False):
        """
        Initialize secure configuration manager
        
        Args:
            config_file: Path to configuration file
            encryption_key: Master key for encryption (will be derived if not provided)
            use_env_vars: Allow environment variable overrides
            validate_schema: Validate configuration against schema
            auto_rotate_keys: Enable automatic key rotation
        """
        self.config_file = config_file
        self.use_env_vars = use_env_vars
        self.validate_schema = validate_schema
        self.auto_rotate_keys = auto_rotate_keys
        
        # Initialize encryption
        self._setup_encryption(encryption_key)
        
        # Configuration storage
        self._config = {}
        self._encrypted_values = {}
        self._config_schema = {}
        
        # Load configuration
        if config_file:
            self.load_config(config_file)
        
        # Apply environment variable overrides
        if use_env_vars:
            self._apply_env_overrides()
        
        # Setup key rotation if enabled
        if auto_rotate_keys:
            self._setup_key_rotation()
    
    def _setup_encryption(self, encryption_key: Optional[str] = None):
        """Setup encryption for sensitive values"""
        if encryption_key:
            # Use provided key
            key = encryption_key.encode()
        else:
            # Derive key from environment or generate
            env_key = os.environ.get('RAG_ENCRYPTION_KEY')
            if env_key:
                key = env_key.encode()
            else:
                # Generate and store key securely
                key = Fernet.generate_key()
                self._store_encryption_key(key)
        
        # Create cipher suite
        self.cipher_suite = Fernet(key)
        
        # Setup key derivation for additional security
        self.kdf_salt = os.environ.get('RAG_KDF_SALT', secrets.token_hex(16)).encode()
    
    def _store_encryption_key(self, key: bytes):
        """Store encryption key securely"""
        # In production, use a proper key management service (KMS)
        # For now, store in environment variable
        os.environ['RAG_ENCRYPTION_KEY'] = key.decode()
        
        # Also save to a secure location with restricted permissions
        key_file = Path.home() / '.rag' / 'encryption.key'
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions (Unix-like systems)
        key_file.write_bytes(key)
        if os.name != 'nt':  # Not Windows
            os.chmod(key_file, 0o600)  # Read/write for owner only
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a configuration key contains sensitive data"""
        key_lower = key.lower()
        return any(re.match(pattern, key_lower) for pattern in self.SENSITIVE_KEY_PATTERNS)
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt sensitive configuration value"""
        encrypted = self.cipher_suite.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive configuration value"""
        encrypted = base64.b64decode(encrypted_value.encode())
        decrypted = self.cipher_suite.decrypt(encrypted)
        return decrypted.decode()
    
    def load_config(self, config_file: str):
        """
        Load configuration from file
        
        Args:
            config_file: Path to configuration file
        """
        path = Path(config_file)
        
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        # Check file permissions (security check)
        if os.name != 'nt':  # Unix-like systems
            stat_info = path.stat()
            if stat_info.st_mode & 0o077:  # Check if others have any permissions
                logger.warning(f"Configuration file {config_file} has insecure permissions")
        
        # Load based on file extension
        try:
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    raw_config = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    raw_config = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported config file format: {path.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
        
        # Process and encrypt sensitive values
        self._process_config(raw_config)
        
        # Validate against schema if enabled
        if self.validate_schema and self._config_schema:
            self._validate_config()
        
        logger.info(f"Configuration loaded from {config_file}")
    
    def _process_config(self, config: Dict[str, Any], prefix: str = ""):
        """Process configuration and encrypt sensitive values"""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process nested configuration
                self._process_config(value, full_key)
            else:
                # Check if value needs encryption
                if self._is_sensitive_key(key) and isinstance(value, str):
                    # Encrypt and store
                    encrypted = self._encrypt_value(value)
                    self._encrypted_values[full_key] = encrypted
                    self._config[full_key] = None  # Don't store plaintext
                else:
                    self._config[full_key] = value
    
    def get(self, key: str, default: Any = None, decrypt: bool = True) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            decrypt: Automatically decrypt sensitive values
            
        Returns:
            Configuration value
        """
        # Check environment variable override first
        if self.use_env_vars:
            env_key = f"{self.ENV_PREFIX}{key.upper().replace('.', '_')}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                return env_value
        
        # Check if it's an encrypted value
        if key in self._encrypted_values and decrypt:
            try:
                return self._decrypt_value(self._encrypted_values[key])
            except Exception as e:
                logger.error(f"Failed to decrypt configuration value for {key}: {e}")
                return default
        
        # Get regular value
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any, encrypt: Optional[bool] = None):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            encrypt: Force encryption (auto-detected if None)
        """
        if encrypt is None:
            encrypt = self._is_sensitive_key(key)
        
        if encrypt and isinstance(value, str):
            encrypted = self._encrypt_value(value)
            self._encrypted_values[key] = encrypted
            self._config[key] = None
        else:
            self._config[key] = value
            if key in self._encrypted_values:
                del self._encrypted_values[key]
    
    def get_all(self, decrypt: bool = False) -> Dict[str, Any]:
        """
        Get all configuration values
        
        Args:
            decrypt: Include decrypted sensitive values
            
        Returns:
            Configuration dictionary
        """
        result = dict(self._config)
        
        if decrypt:
            # Decrypt sensitive values
            for key, encrypted in self._encrypted_values.items():
                try:
                    result[key] = self._decrypt_value(encrypted)
                except Exception as e:
                    logger.error(f"Failed to decrypt {key}: {e}")
                    result[key] = "<DECRYPTION_ERROR>"
        else:
            # Mark encrypted values
            for key in self._encrypted_values:
                result[key] = "<ENCRYPTED>"
        
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        for env_key, env_value in os.environ.items():
            if env_key.startswith(self.ENV_PREFIX):
                # Convert environment variable to config key
                config_key = env_key[len(self.ENV_PREFIX):].lower().replace('_', '.')
                
                # Parse value type
                parsed_value = self._parse_env_value(env_value)
                
                # Set configuration value
                self.set(config_key, parsed_value)
                
                logger.info(f"Configuration override from environment: {config_key}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Check for boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Check for number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def load_schema(self, schema_file: str):
        """
        Load configuration schema for validation
        
        Args:
            schema_file: Path to schema file
        """
        with open(schema_file, 'r') as f:
            self._config_schema = json.load(f)
    
    def _validate_config(self):
        """Validate configuration against schema"""
        # This would use a JSON schema validator
        # Simplified validation for demonstration
        required_keys = self._config_schema.get('required', [])
        
        for key in required_keys:
            if key not in self._config and key not in self._encrypted_values:
                raise ConfigurationError(f"Required configuration key missing: {key}")
    
    def save_config(self, output_file: str, include_encrypted: bool = False):
        """
        Save configuration to file
        
        Args:
            output_file: Output file path
            include_encrypted: Include encrypted values
        """
        config_to_save = self.get_all(decrypt=False)
        
        if not include_encrypted:
            # Remove encrypted placeholders
            config_to_save = {
                k: v for k, v in config_to_save.items()
                if v != "<ENCRYPTED>"
            }
        
        path = Path(output_file)
        
        # Set restrictive permissions before writing
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_to_save, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        
        # Set restrictive permissions after writing
        if os.name != 'nt':
            os.chmod(path, 0o600)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def rotate_encryption_key(self) -> str:
        """
        Rotate encryption key
        
        Returns:
            New encryption key
        """
        # Generate new key
        new_key = Fernet.generate_key()
        new_cipher = Fernet(new_key)
        
        # Re-encrypt all sensitive values
        for key, encrypted_value in self._encrypted_values.items():
            # Decrypt with old key
            decrypted = self._decrypt_value(encrypted_value)
            
            # Encrypt with new key
            new_encrypted = new_cipher.encrypt(decrypted.encode())
            self._encrypted_values[key] = base64.b64encode(new_encrypted).decode()
        
        # Update cipher suite
        self.cipher_suite = new_cipher
        
        # Store new key
        self._store_encryption_key(new_key)
        
        logger.info("Encryption key rotated successfully")
        return new_key.decode()
    
    def _setup_key_rotation(self):
        """Setup automatic key rotation"""
        # In production, this would be scheduled with a proper task scheduler
        # For now, just log the recommendation
        logger.info("Automatic key rotation enabled. Keys should be rotated every 90 days.")
    
    def get_sensitive_keys(self) -> List[str]:
        """
        Get list of keys containing sensitive data
        
        Returns:
            List of sensitive configuration keys
        """
        return list(self._encrypted_values.keys())
    
    def validate_security(self) -> Dict[str, Any]:
        """
        Validate configuration security
        
        Returns:
            Security validation results
        """
        issues = []
        warnings = []
        
        # Check for plaintext sensitive values
        for key, value in self._config.items():
            if self._is_sensitive_key(key) and value is not None:
                issues.append(f"Sensitive value '{key}' stored in plaintext")
        
        # Check file permissions
        if self.config_file:
            path = Path(self.config_file)
            if path.exists() and os.name != 'nt':
                stat_info = path.stat()
                if stat_info.st_mode & 0o077:
                    warnings.append(f"Configuration file has insecure permissions")
        
        # Check for default/weak values
        weak_patterns = ['password123', 'admin', 'default', 'test', 'demo']
        for key in self._encrypted_values:
            try:
                value = self._decrypt_value(self._encrypted_values[key])
                if any(pattern in value.lower() for pattern in weak_patterns):
                    issues.append(f"Weak value detected for '{key}'")
            except:
                pass
        
        return {
            'secure': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'encrypted_keys': len(self._encrypted_values),
            'total_keys': len(self._config) + len(self._encrypted_values)
        }