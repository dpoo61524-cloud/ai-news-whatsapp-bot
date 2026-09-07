import os
import logging
from typing import List, Optional
from pydantic import Field

logger = logging.getLogger(__name__)

HAS_PYDANTIC_SETTINGS = False
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore

class Settings(BaseSettings):
    """
    Central application configuration loaded from environment variables or .env file.
    Includes type validation and default values for vector store, LLMs, DB, and Webhooks.
    """
    # Environment & Logging
    ENV: str = Field(default="development", description="Application environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # LLM & Embedding API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key for embeddings and LLM tasks")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API Key for Claude agents")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Model name for vector embeddings")
    
    # Vector Store & Persistence
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db", description="Path to ChromaDB persistent storage directory")
    SIMILARITY_THRESHOLD: float = Field(default=0.85, description="Cosine similarity threshold for deduplication (0.0 to 1.0)")
    
    # Relational Database
    DATABASE_URL: str = Field(default="sqlite:///./news_agents.db", description="SQLAlchemy database connection URL")
    
    # WhatsApp Webhook Integration
    WHATSAPP_VERIFY_TOKEN: str = Field(default="news_agents_verify_token", description="Webhook verification token for Meta WhatsApp API")
    WHATSAPP_API_TOKEN: str = Field(default="", description="Access token for sending WhatsApp messages")
    WHATSAPP_PHONE_NUMBER_ID: str = Field(default="", description="WhatsApp Business Phone Number ID")

    # Default News RSS Feeds across multiple categories
    DEFAULT_RSS_FEEDS: List[str] = Field(
        default_factory=lambda: [
            "https://feeds.bbci.co.uk/news/rss.xml",                  # World / Top News
            "https://feeds.bbci.co.uk/news/world/rss.xml",            # World News
            "https://feeds.bbci.co.uk/news/business/rss.xml",         # Business
            "https://feeds.bbci.co.uk/news/technology/rss.xml",       # Tech
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", # Science
            "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",  # Entertainment
            "https://cointelegraph.com/rss",                          # Crypto
            "https://coindesk.com/arc/outboundfeeds/rss",             # Crypto
            "https://decrypt.co/feed",                                # Crypto
            "https://feeds.content.dowjones.io/public/rss/mw_topstories", # Stock Market
            "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",# Stock Market & Economy
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",# Business & Stocks
            "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",   # India News
            "https://timesofindia.indiatimes.com/rssfeedstopstories.cms", # India News
            "https://indianexpress.com/section/india/feed/",          # India News
            "https://www.thehindu.com/news/national/feeder/default.rss", # India News
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", # India Stock Market
            "https://www.livemint.com/rss/markets",                    # India Stock Market
            "https://www.financialexpress.com/market/feed/",           # India Stock Market
            "https://www.moneycontrol.com/rss/MCtopnews.xml"           # India Finance & Stocks
        ],
        description="Default RSS feeds for periodic ingestion"
    )

    if HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

def get_settings() -> Settings:
    """
    Safely instantiates and returns the global Settings instance with error handling.
    """
    try:
        settings_instance = Settings()
        return settings_instance
    except Exception as e:
        logger.error(f"Failed to load application configuration: {str(e)}")
        return Settings()

settings = get_settings()
