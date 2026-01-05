"""Configuration management for the agentic system."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for the agentic system."""

    # API Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MODEL = os.getenv("MODEL", "claude-3-5-sonnet-20241022")

    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

    # Tool Configuration
    ENABLE_SHELL = os.getenv("ENABLE_SHELL", "false").lower() == "true"
    ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Please set it in your .env file or environment variables."
            )
