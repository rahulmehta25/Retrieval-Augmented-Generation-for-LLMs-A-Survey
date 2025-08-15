# RAG Configuration System

This directory contains the new hierarchical configuration system for the RAG project. The system provides environment-specific configurations, type safety, validation, and secure secret management.

## 📁 Configuration Files

### Core Configuration Files

- **`base.yaml`** - Shared defaults used across all environments
- **`development.yaml`** - Development environment overrides
- **`production.yaml`** - Production environment overrides  
- **`testing.yaml`** - Testing environment overrides
- **`local.yaml`** - Local overrides (optional, gitignored)

### Deprecated Files (Backward Compatibility)

- **`*_DEPRECATED.yaml`** - Old configuration files kept for reference
- These files are maintained for backward compatibility but should not be used

## 🚀 Quick Start

### 1. Basic Usage

```python
from src.config import get_config

# Get validated configuration for current environment
config = get_config()

# Access configuration values with type safety
chunk_size = config.text_splitter.chunk_size
model_name = config.generator.model_name
debug_mode = config.system.debug
```

### 2. Environment-Specific Configuration

```bash
# Set environment (default: development)
export RAG_ENVIRONMENT=production

# Or specify when creating config manager
from src.config import ConfigurationManager
manager = ConfigurationManager(environment='production')
config = manager.get_config()
```

### 3. Environment Variable Overrides

```bash
# Override any configuration value via environment variables
export RAG_SYSTEM_DEBUG=true
export RAG_GENERATOR_TEMPERATURE=0.9
export RAG_TEXT_SPLITTER_CHUNK_SIZE=256
```

### 4. Secret Management

```bash
# Set secrets via environment variables
export RAG_JWT_SECRET_KEY=your_secret_key
export OPENAI_API_KEY=your_openai_key
export RAG_DATABASE_URL=postgresql://user:pass@localhost/db

# Access secrets in code
from src.config import get_secrets
secrets = get_secrets()
api_key = secrets.openai_api_key
```

## 🏗️ Configuration Structure

The configuration system uses a hierarchical structure:

```
base.yaml (shared defaults)
    ↓
environment.yaml (environment overrides)
    ↓ 
local.yaml (local overrides, optional)
    ↓
Environment Variables (highest priority)
```

### Configuration Sections

- **`system`** - System-level settings (debug, workers, timeouts)
- **`text_splitter`** - Text chunking configuration
- **`embedder`** - Embedding model settings
- **`vector_store`** - Vector database configuration
- **`generator`** - LLM generation settings
- **`retrieval`** - Information retrieval configuration
- **`context`** - Context management settings
- **`conversation`** - Conversation memory settings
- **`security`** - Security and validation settings
- **`rate_limiting`** - API rate limiting configuration
- **`api`** - API server settings
- **`cors`** - Cross-origin resource sharing
- **`logging`** - Logging configuration
- **`monitoring`** - System monitoring settings
- **`cache`** - Caching configuration
- **`evaluation`** - Model evaluation settings
- **`features`** - Feature flags
- **`storage`** - Data storage settings
- **`error_handling`** - Error handling configuration
- **`performance`** - Performance tuning
- **`query_optimization`** - Query processing optimization
- **`knowledge_graph`** - Knowledge graph settings
- **`ab_testing`** - A/B testing configuration

## 🔧 Environment Variables

Environment variables follow the pattern: `RAG_SECTION_SUBSECTION_KEY`

Examples:
```bash
# System configuration
RAG_SYSTEM_DEBUG=true
RAG_SYSTEM_MAX_WORKERS=8

# Generator configuration  
RAG_GENERATOR_MODEL_NAME=llama2:7b
RAG_GENERATOR_TEMPERATURE=0.7

# Vector store configuration
RAG_VECTOR_STORE_TYPE=hybrid
RAG_VECTOR_STORE_COLLECTION_NAME=rag_production
```

See `.env.example` for a complete list of available environment variables.

## 🔒 Security Features

### Automatic Secret Detection
The system automatically detects and encrypts sensitive configuration values:
- API keys
- Passwords
- Secret keys
- Database URLs
- JWT secrets

### Environment Variable Validation
Secrets are validated using Pydantic models and loaded from environment variables only.

### File Permissions
Configuration files are automatically set with restrictive permissions (600) on Unix systems.

### Audit Logging
Configuration access can be logged for security auditing.

## 📊 Validation

The system uses Pydantic models for comprehensive validation:

```python
from src.config import get_config_manager

manager = get_config_manager()
validation_report = manager.validate_current_config()

if validation_report['valid']:
    print("✅ Configuration is valid!")
else:
    print("❌ Configuration errors:")
    for error in validation_report['errors']:
        print(f"  - {error['field']}: {error['message']}")
```

## 🔄 Migration from Old System

### Automatic Migration

```bash
# Migrate old configuration files
python -m src.config.migration_tool migrate config.yaml --env development
python -m src.config.migration_tool migrate config_production.yaml --env production

# Validate migrated configuration
python -m src.config.migration_tool validate --env development
```

### Manual Migration

1. **Identify current config files**:
   - `config.yaml` → `config/development.yaml`
   - `config_production.yaml` → `config/production.yaml`

2. **Update code**:
   ```python
   # OLD
   import yaml
   config = yaml.safe_load(open('config.yaml'))
   chunk_size = config.get('chunk_size', 500)
   
   # NEW
   from src.config import get_config
   config = get_config()
   chunk_size = config.text_splitter.chunk_size
   ```

3. **Move secrets to environment variables**:
   ```bash
   # Create .env file from .env.example
   cp .env.example .env
   # Edit .env with your secret values
   ```

### Backward Compatibility

The system maintains backward compatibility during transition:

```python
from src.config.adapter import get_config_value

# Old-style access (deprecated but works)
chunk_size = get_config_value('chunk_size', 500)
```

## 🎛️ Advanced Usage

### Creating Custom Environments

```bash
# Create new environment based on development
python -m src.config.migration_tool create staging --base development
```

### Dynamic Configuration Updates

```python
from src.config import get_config_manager

manager = get_config_manager()

# Update configuration value
manager.set('system.debug', True)

# Save current configuration
manager.save_config('config/current.yaml')

# Reload configuration
manager.reload()
```

### Configuration Introspection

```python
from src.config import get_config_manager

manager = get_config_manager()

# Get environment information
env_info = manager.get_environment_info()
print(f"Environment: {env_info['environment']}")
print(f"Sources: {env_info['sources']}")

# List available environments
environments = manager.list_available_environments()
print(f"Available: {environments}")
```

## 🧪 Testing Configuration

### Test-Specific Settings

The testing environment provides:
- Deterministic settings for reproducible tests
- Minimal resource usage
- Fast execution
- Disabled external dependencies

```python
# In tests
from src.config import ConfigurationManager

config_manager = ConfigurationManager(environment='testing')
config = config_manager.get_config()
```

### Mocking Configuration

```python
# Mock configuration for unit tests
from src.config.models import RAGConfiguration, SystemConfig

mock_config = RAGConfiguration(
    system=SystemConfig(debug=True, max_workers=1)
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Configuration not found**
   ```bash
   # Check available environments
   python -m src.config.migration_tool list
   ```

2. **Validation errors**
   ```bash
   # Validate current configuration
   python -m src.config.migration_tool validate --env development
   ```

3. **Missing environment variables**
   ```bash
   # Check .env file exists and has required values
   cp .env.example .env
   # Edit .env with your values
   ```

4. **Permission errors**
   ```bash
   # Fix file permissions
   chmod 600 config/*.yaml
   ```

### Debug Mode

```python
from src.config import ConfigurationManager

# Enable verbose logging and strict validation
manager = ConfigurationManager(
    environment='development',
    strict_mode=True,
    validate_on_load=True
)
```

## 📚 Examples

See `examples/configuration_examples.py` for comprehensive usage examples:

```bash
python examples/configuration_examples.py
```

## 🔗 Related Files

- **`.env.example`** - Template for environment variables
- **`src/config/`** - Configuration system implementation
- **`examples/configuration_examples.py`** - Usage examples
- **`src/config/migration_tool.py`** - Migration utilities

## 📋 Checklist for Migration

- [ ] Back up existing configuration files
- [ ] Run migration tool on old configs
- [ ] Review migrated configuration files
- [ ] Set up `.env` file with secrets
- [ ] Update code to use new configuration system
- [ ] Test in development environment
- [ ] Validate configuration with validation tool
- [ ] Deploy to staging/production
- [ ] Monitor for configuration-related issues
- [ ] Remove old configuration files when satisfied