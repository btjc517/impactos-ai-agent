"""
Environment configuration and validation utilities.

Handles loading and validation of environment variables and API keys.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class EnvironmentConfig:
    """Manages environment configuration and validation."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment configuration.
        
        Args:
            env_file: Path to .env file (defaults to project root/.env)
        """
        # Load environment file
        if env_file is None:
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
        
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning(f"Environment file not found: {env_file}")
        
        # Load configuration
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            # API Keys
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            
            # System Settings
            'tokenizers_parallelism': os.getenv('TOKENIZERS_PARALLELISM', 'false'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            
            # Model Configuration
            'default_gpt_model': os.getenv('DEFAULT_GPT_MODEL', 'gpt-4-1106-preview'),
            'default_claude_model': os.getenv('DEFAULT_CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
            'max_context_tokens': int(os.getenv('MAX_CONTEXT_TOKENS', '4096')),
            
            # Data Processing
            'batch_size': int(os.getenv('BATCH_SIZE', '100')),
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '50')),
            'supported_formats': os.getenv('SUPPORTED_FORMATS', 'csv,xlsx,pdf,json,txt').split(','),
        }
    
    def _validate_config(self):
        """Validate configuration and log warnings for missing keys."""
        warnings = []
        
        if not self.config['openai_api_key']:
            warnings.append("OpenAI API key not found - GPT tools will be limited")
        
        if not self.config['anthropic_api_key']:
            warnings.append("Anthropic API key not found - Multi-agent system disabled")
        
        # Set tokenizers parallelism
        os.environ['TOKENIZERS_PARALLELISM'] = self.config['tokenizers_parallelism']
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        if not warnings:
            logger.info("All API keys configured successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.config['openai_api_key'])
    
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self.config['anthropic_api_key'])
    
    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.config['openai_api_key']
    
    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic API key."""
        return self.config['anthropic_api_key']
    
    def is_file_supported(self, file_path: str) -> bool:
        """Check if file format is supported."""
        extension = Path(file_path).suffix.lower().lstrip('.')
        return extension in self.config['supported_formats']
    
    def validate_file_size(self, file_path: str) -> bool:
        """Validate file size against limits."""
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return size_mb <= self.config['max_file_size_mb']
        except (OSError, FileNotFoundError):
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system configuration status."""
        return {
            'api_keys': {
                'openai': self.has_openai_key(),
                'anthropic': self.has_anthropic_key()
            },
            'models': {
                'gpt': self.config['default_gpt_model'],
                'claude': self.config['default_claude_model']
            },
            'limits': {
                'max_file_size_mb': self.config['max_file_size_mb'],
                'max_context_tokens': self.config['max_context_tokens'],
                'batch_size': self.config['batch_size']
            },
            'supported_formats': self.config['supported_formats']
        }


# Global configuration instance
_config = None

def get_config() -> EnvironmentConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = EnvironmentConfig()
    return _config


def setup_logging():
    """Setup logging based on environment configuration."""
    config = get_config()
    
    logging.basicConfig(
        level=getattr(logging, config.get('log_level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )