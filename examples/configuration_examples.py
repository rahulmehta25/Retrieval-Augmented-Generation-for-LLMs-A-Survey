#!/usr/bin/env python3
"""
Configuration System Examples

This script demonstrates how to use the new hierarchical configuration system
with Pydantic validation, environment variable overrides, and secret management.
"""

import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config, get_config_manager, get_secrets
from src.config.manager import ConfigurationManager
from src.rag.rag_factory import RAGComponentFactory


def example_basic_usage():
    """Example 1: Basic configuration usage"""
    print("=== Example 1: Basic Configuration Usage ===")
    
    # Get validated configuration for current environment
    config = get_config()
    
    print(f"System name: {config.system.name}")
    print(f"Debug mode: {config.system.debug}")
    print(f"LLM model: {config.generator.model_name}")
    print(f"Embedding model: {config.embedder.model_name}")
    print(f"Vector store type: {config.vector_store.type}")
    print(f"Chunk size: {config.text_splitter.chunk_size}")
    print()


def example_environment_specific():
    """Example 2: Environment-specific configurations"""
    print("=== Example 2: Environment-Specific Configurations ===")
    
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        try:
            manager = ConfigurationManager(environment=env)
            config = manager.get_config()
            
            print(f"{env.upper()}:")
            print(f"  Debug: {config.system.debug}")
            print(f"  Workers: {config.system.max_workers}")
            print(f"  Model: {config.generator.model_name}")
            print(f"  Chunk size: {config.text_splitter.chunk_size}")
            
        except Exception as e:
            print(f"{env.upper()}: Configuration not available ({e})")
    
    print()


def example_environment_variables():
    """Example 3: Environment variable overrides"""
    print("=== Example 3: Environment Variable Overrides ===")
    
    # Set some environment variables to override configuration
    os.environ['RAG_SYSTEM_DEBUG'] = 'true'
    os.environ['RAG_GENERATOR_TEMPERATURE'] = '0.9'
    os.environ['RAG_TEXT_SPLITTER_CHUNK_SIZE'] = '256'
    
    # Create manager with environment overrides
    manager = ConfigurationManager(environment='development')
    config = manager.get_config()
    
    print("Configuration with environment variable overrides:")
    print(f"  Debug (RAG_SYSTEM_DEBUG=true): {config.system.debug}")
    print(f"  Temperature (RAG_GENERATOR_TEMPERATURE=0.9): {config.generator.temperature}")
    print(f"  Chunk size (RAG_TEXT_SPLITTER_CHUNK_SIZE=256): {config.text_splitter.chunk_size}")
    
    # Clean up environment variables
    del os.environ['RAG_SYSTEM_DEBUG']
    del os.environ['RAG_GENERATOR_TEMPERATURE']
    del os.environ['RAG_TEXT_SPLITTER_CHUNK_SIZE']
    
    print()


def example_secrets_management():
    """Example 4: Secrets management"""
    print("=== Example 4: Secrets Management ===")
    
    # Set some example secrets
    os.environ['RAG_JWT_SECRET_KEY'] = 'example_jwt_secret_key'
    os.environ['OPENAI_API_KEY'] = 'sk-example_openai_key'
    
    try:
        secrets = get_secrets()
        
        print("Available secrets (from environment variables):")
        print(f"  JWT Secret: {'✓' if secrets.jwt_secret_key else '✗'}")
        print(f"  OpenAI API Key: {'✓' if secrets.openai_api_key else '✗'}")
        print(f"  Anthropic API Key: {'✓' if secrets.anthropic_api_key else '✗'}")
        
        # Note: In production, secrets would be properly managed
        # and not printed to console
        
    except Exception as e:
        print(f"Secrets loading failed: {e}")
    
    # Clean up
    if 'RAG_JWT_SECRET_KEY' in os.environ:
        del os.environ['RAG_JWT_SECRET_KEY']
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    print()


def example_component_creation():
    """Example 5: Creating RAG components with new config system"""
    print("=== Example 5: Creating RAG Components ===")
    
    try:
        # Method 1: Use global configuration
        embedder = RAGComponentFactory.get_embedder()
        text_splitter = RAGComponentFactory.get_text_splitter()
        vector_store = RAGComponentFactory.get_vector_store()
        generator = RAGComponentFactory.get_generator()
        
        print("Created components using global configuration:")
        print(f"  Embedder: {type(embedder).__name__}")
        print(f"  Text Splitter: {type(text_splitter).__name__}")
        print(f"  Vector Store: {type(vector_store).__name__}")
        print(f"  Generator: {type(generator).__name__}")
        
        # Method 2: Create complete RAG system
        rag_system = RAGComponentFactory.create_rag_system()
        
        print(f"\nComplete RAG system created with {len(rag_system)} components")
        
        # Method 3: Environment-specific RAG system
        prod_rag = RAGComponentFactory.create_rag_system_from_environment('production')
        
        print("Production RAG system created")
        
    except Exception as e:
        print(f"Component creation failed: {e}")
    
    print()


def example_configuration_validation():
    """Example 6: Configuration validation"""
    print("=== Example 6: Configuration Validation ===")
    
    manager = get_config_manager()
    
    # Get validation report
    validation_report = manager.validate_current_config()
    
    print("Configuration Validation Report:")
    print(f"  Environment: {validation_report['environment']}")
    print(f"  Valid: {validation_report['valid']}")
    print(f"  Sources: {', '.join(validation_report['sources'])}")
    
    if validation_report['errors']:
        print("  Errors:")
        for error in validation_report['errors']:
            print(f"    - {error['field']}: {error['message']}")
    
    if validation_report['warnings']:
        print("  Warnings:")
        for warning in validation_report['warnings']:
            print(f"    - {warning}")
    
    print()


def example_configuration_introspection():
    """Example 7: Configuration introspection"""
    print("=== Example 7: Configuration Introspection ===")
    
    manager = get_config_manager()
    
    # Get environment info
    env_info = manager.get_environment_info()
    
    print("Environment Information:")
    print(f"  Environment: {env_info['environment']}")
    print(f"  Config directory: {env_info['config_dir']}")
    print(f"  Validation enabled: {env_info['validation_enabled']}")
    print(f"  Strict mode: {env_info['strict_mode']}")
    print(f"  Secrets loaded: {env_info['secrets_loaded']}")
    print(f"  Env overrides: {env_info['env_overrides_enabled']}")
    
    # List available environments
    environments = manager.list_available_environments()
    print(f"  Available environments: {', '.join(environments)}")
    
    print()


def example_backward_compatibility():
    """Example 8: Backward compatibility with old config format"""
    print("=== Example 8: Backward Compatibility ===")
    
    from src.config.adapter import get_config_value, ConfigurationMigrationHelper
    
    # Old-style configuration access (deprecated but supported)
    chunk_size = get_config_value('chunk_size', 500)
    model_name = get_config_value('model_name', 'default')
    debug = get_config_value('debug', False)
    
    print("Legacy configuration access (deprecated):")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Model name: {model_name}")
    print(f"  Debug: {debug}")
    
    # Migration helper example
    legacy_config = {
        'chunk_size': 512,
        'chunk_overlap': 128,
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_store': 'chromadb',
        'model_name': 'gemma:2b',
        'temperature': 0.7,
        'debug': True
    }
    
    migrated = ConfigurationMigrationHelper.convert_legacy_config(legacy_config)
    print(f"\nMigrated config sections: {list(migrated.keys())}")
    
    print()


def main():
    """Run all configuration examples"""
    print("RAG Configuration System Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_usage()
        example_environment_specific()
        example_environment_variables()
        example_secrets_management()
        example_component_creation()
        example_configuration_validation()
        example_configuration_introspection()
        example_backward_compatibility()
        
        print("All examples completed successfully! ✅")
        print()
        print("Next steps:")
        print("1. Review the configuration files in config/ directory")
        print("2. Set up your .env file based on .env.example")
        print("3. Use 'python -m src.config.migration_tool guide' for migration help")
        print("4. Run 'python -m src.config.migration_tool validate' to check your config")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()