"""
Configuration for LLM API keys and settings.

API Key Setup:
1. Set environment variable: export OPENAI_API_KEY="your-key-here"
2. Or create a .env file in the project root with: OPENAI_API_KEY=your-key-here
3. Or set it in this file (NOT RECOMMENDED for production)
"""

import os
from typing import Optional

# Try to load from environment variable first
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

# If not in environment, try to load from .env file
if OPENAI_API_KEY is None or ANTHROPIC_API_KEY is None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    except ImportError:
        pass

# Fallback: You can set it directly here (NOT recommended for git commits)
# OPENAI_API_KEY = "your-key-here"
# ANTHROPIC_API_KEY = "your-key-here"

# LLM Configuration
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# API Settings
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
CACHE_RESPONSES = True
