from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Define fields with aliases for environment variables
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    supabase_url: str = Field(alias="SUPABASE_URL")
    supabase_key: str = Field(alias="SUPABASE_KEY")
    google_api_key: str = Field(alias="GOOGLE_API_KEY")
    
    # Updated configuration for Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        populate_by_name=True,  # Allow population by field name and alias
        extra="ignore",  # Ignore extra fields
    )

# Initialize settings
settings = Settings()