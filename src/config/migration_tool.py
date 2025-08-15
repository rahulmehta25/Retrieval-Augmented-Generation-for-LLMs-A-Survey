#!/usr/bin/env python3
"""
Configuration Migration Tool

This script helps migrate from the old configuration system to the new
hierarchical configuration system with Pydantic validation.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .adapter import ConfigurationMigrationHelper
from .manager import ConfigurationManager


def migrate_legacy_config(input_file: str, output_env: str = "development"):
    """
    Migrate a legacy configuration file to the new system
    
    Args:
        input_file: Path to legacy configuration file
        output_env: Target environment (development, testing, production)
    """
    print(f"Migrating legacy configuration: {input_file}")
    print(f"Target environment: {output_env}")
    
    try:
        # Load legacy configuration
        helper = ConfigurationMigrationHelper()
        
        # Validate migration possibility
        import yaml
        with open(input_file, 'r') as f:
            legacy_config = yaml.safe_load(f)
        
        validation_report = helper.validate_migration(legacy_config)
        
        print("\nMigration Validation Report:")
        print(f"Migration possible: {validation_report['migration_possible']}")
        
        if validation_report['warnings']:
            print("\nWarnings:")
            for warning in validation_report['warnings']:
                print(f"  - {warning}")
        
        if validation_report['unmapped_keys']:
            print(f"\nUnmapped keys (will need manual review): {validation_report['unmapped_keys']}")
        
        # Perform migration
        output_path = f"config/{output_env}_migrated.yaml"
        helper.save_migrated_config(legacy_config, output_path)
        
        print(f"\nMigration completed! New configuration saved to: {output_path}")
        print("\nNext steps:")
        print("1. Review the migrated configuration file")
        print("2. Move any unmapped keys to appropriate sections")
        print("3. Test the new configuration")
        print("4. Update your code to use the new configuration system")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False
    
    return True


def validate_new_config(environment: str = "development"):
    """
    Validate the current configuration for an environment
    
    Args:
        environment: Environment to validate
    """
    print(f"Validating configuration for environment: {environment}")
    
    try:
        config_manager = ConfigurationManager(environment=environment)
        validation_report = config_manager.validate_current_config()
        
        print("\nConfiguration Validation Report:")
        print(f"Environment: {validation_report['environment']}")
        print(f"Valid: {validation_report['valid']}")
        print(f"Sources: {', '.join(validation_report['sources'])}")
        
        if validation_report['valid']:
            print("\n✅ Configuration is valid!")
            if 'config_summary' in validation_report:
                summary = validation_report['config_summary']
                print(f"System: {summary['system_name']}")
                print(f"Debug mode: {summary['debug_mode']}")
                print(f"LLM model: {summary['llm_model']}")
                print(f"Embedding model: {summary['embedding_model']}")
                print(f"Vector store: {summary['vector_store']}")
        else:
            print("\n❌ Configuration has errors:")
            for error in validation_report['errors']:
                print(f"  - {error['field']}: {error['message']}")
        
        if validation_report['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation_report['warnings']:
                print(f"  - {warning}")
    
    except Exception as e:
        print(f"Validation failed: {e}")
        return False
    
    return True


def list_environments():
    """List available environment configurations"""
    try:
        config_manager = ConfigurationManager()
        environments = config_manager.list_available_environments()
        
        print("Available environment configurations:")
        for env in environments:
            print(f"  - {env}")
        
        print(f"\nCurrent environment: {config_manager.environment}")
        
    except Exception as e:
        print(f"Failed to list environments: {e}")


def create_environment(env_name: str, base_on: str = "development"):
    """
    Create a new environment configuration
    
    Args:
        env_name: Name of new environment
        base_on: Base environment to copy from
    """
    try:
        config_manager = ConfigurationManager()
        config_manager.create_environment_template(env_name, base_on)
        print(f"Created new environment configuration: {env_name}")
        print(f"Based on: {base_on}")
        print(f"File location: config/{env_name}.yaml")
        
    except Exception as e:
        print(f"Failed to create environment: {e}")


def show_migration_guide():
    """Show detailed migration guide"""
    guide = """
    RAG Configuration System Migration Guide
    =====================================
    
    The new configuration system provides:
    
    ✅ Hierarchical configuration (base + environment overrides)
    ✅ Type safety with Pydantic validation
    ✅ Environment variable overrides
    ✅ Secret management
    ✅ Backward compatibility
    
    Migration Steps:
    
    1. Identify your current configuration files:
       - config.yaml (main config)
       - config_production.yaml (production config)
       - Any custom configuration files
    
    2. Run migration for each file:
       python -m src.config.migration_tool migrate config.yaml --env development
       python -m src.config.migration_tool migrate config_production.yaml --env production
    
    3. Review migrated files in config/ directory:
       - config/base.yaml (shared defaults)
       - config/development.yaml (development overrides)
       - config/production.yaml (production overrides)
       - config/testing.yaml (testing overrides)
    
    4. Update your code:
       
       OLD:
         import yaml
         config = yaml.safe_load(open('config.yaml'))
         chunk_size = config.get('chunk_size', 500)
       
       NEW:
         from src.config import get_config
         config = get_config()
         chunk_size = config.text_splitter.chunk_size
    
    5. Set environment variables:
       export RAG_ENVIRONMENT=production  # or development, testing
       
       For secrets, use .env file or environment variables:
       export RAG_JWT_SECRET_KEY=your_secret_key
       export OPENAI_API_KEY=your_openai_key
    
    6. Test the new configuration:
       python -m src.config.migration_tool validate --env development
    
    7. Update component creation:
       
       OLD:
         from src.rag.rag_factory import RAGComponentFactory
         embedder = RAGComponentFactory.get_embedder(config['embedder'])
       
       NEW:
         from src.rag.rag_factory import RAGComponentFactory
         embedder = RAGComponentFactory.get_embedder()  # Uses global config
         # OR
         embedder = RAGComponentFactory.get_embedder(config.embedder)
    
    Benefits:
    
    🔒 Security: Automatic encryption of sensitive values
    📝 Validation: Pydantic models catch configuration errors early  
    🌍 Environments: Easy switching between dev/test/prod configs
    🔧 Flexibility: Environment variable overrides for deployment
    📚 Documentation: Self-documenting configuration with type hints
    
    Need Help?
    
    - Use --help for command options
    - Check the examples in config/ directory
    - Review the .env.example file for environment variables
    - Use the validation command to check your configuration
    """
    print(guide)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Configuration Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate legacy configuration')
    migrate_parser.add_argument('input_file', help='Path to legacy configuration file')
    migrate_parser.add_argument('--env', default='development', 
                               help='Target environment (default: development)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--env', default='development',
                                help='Environment to validate (default: development)')
    
    # List command
    subparsers.add_parser('list', help='List available environments')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new environment')
    create_parser.add_argument('name', help='Name of new environment')
    create_parser.add_argument('--base', default='development',
                              help='Base environment to copy from (default: development)')
    
    # Guide command
    subparsers.add_parser('guide', help='Show migration guide')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'migrate':
        success = migrate_legacy_config(args.input_file, args.env)
        sys.exit(0 if success else 1)
    
    elif args.command == 'validate':
        success = validate_new_config(args.env)
        sys.exit(0 if success else 1)
    
    elif args.command == 'list':
        list_environments()
    
    elif args.command == 'create':
        create_environment(args.name, args.base)
    
    elif args.command == 'guide':
        show_migration_guide()


if __name__ == '__main__':
    main()