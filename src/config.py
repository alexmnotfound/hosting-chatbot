import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # Memory Configuration
    max_memory_messages: int = int(os.getenv("MAX_MEMORY_MESSAGES", "10"))
    memory_summary_threshold: int = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "5"))

    # Rate Limiting
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

    # Data Configuration
    properties_file: str = os.getenv("PROPERTIES_FILE", "data/properties.csv")

    class Config:
        env_file = ".env"

# Create global settings instance
settings = Settings()

# Validate required settings
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required") 