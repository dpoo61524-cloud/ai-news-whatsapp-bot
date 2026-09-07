import hashlib
import logging
import uuid
from typing import List, Optional, Dict, Any

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    feedparser = None
    HAS_FEEDPARSER = False

try:
    from trafilatura import fetch_url, extract
    HAS_TRAFILATURA = True
except ImportError:
    fetch_url, extract = None, None
    HAS_TRAFILATURA = False

from app.config import settings
from app.models import ArticleCreate, IngestionResult, DuplicateCheckResult
from app.agents.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class NewsIngestor:
    """
    Handles news RSS feed ingestion, full text extraction using trafilatura,
    exact hash deduplication, and vector similarity deduplication.
    """
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.seen_hashes = set()

    @staticmethod
    def compute_content_hash(title: str, content: str) -> str:
        """
        Computes SHA-256 hash of normalized title and content for exact duplicate detection.
        """
        normalized_str = f"{title.strip().lower()}|{content.strip().lower()}"
        return hashlib.sha256(normalized_str.encode("utf-8")).hexdigest()

    def extract_full_text(self, url: str, fallback_text: str = "") -> str:
        """
        Fetches and extracts clean main text content from web URL using trafilatura.
        Falls back to RSS summary/description if fetching fails or trafilatura is absent.
        """
        if not url or not HAS_TRAFILATURA:
            return fallback_text.strip()

        try:
            downloaded = fetch_url(url)
            if downloaded:
                extracted = extract(downloaded, include_comments=False, include_tables=False)
                if extracted and len(extracted.strip()) > 50:
                    return extracted.strip()
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed for URL {url}: {str(e)}")

        return fallback_text.strip()

    def check_duplication(self, title: str, content: str, content_hash: str) -> DuplicateCheckResult:
        """
        Checks both exact content hash and semantic vector similarity.
        """
        # Step 1: Exact Hash Check
        if content_hash in self.seen_hashes:
            return DuplicateCheckResult(
                is_duplicate=True,
                similarity_score=1.0,
                reason="Exact content hash match in memory cache"
            )

        # Step 2: Semantic Vector Check
        is_vector_dup = self.vector_store.is_duplicate(
            text=f"{title}\n{content}",
            threshold=settings.SIMILARITY_THRESHOLD
        )

        if is_vector_dup:
            return DuplicateCheckResult(
                is_duplicate=True,
                similarity_score=settings.SIMILARITY_THRESHOLD,
                reason=f"Semantic vector similarity exceeded threshold {settings.SIMILARITY_THRESHOLD}"
            )

        return DuplicateCheckResult(
            is_duplicate=False,
            similarity_score=0.0,
            reason="Unique article"
        )

    @staticmethod
    def detect_category(feed_url: str, publisher: str, title: str = "", content: str = "") -> str:
        """Determines category tag based on feed URL, publisher title, or story content."""
        comb = f"{feed_url} {publisher} {title} {content}".lower()
        if any(w in comb for w in ["crypto", "bitcoin", "ethereum", "cointelegraph", "coindesk", "decrypt", "blockchain", "web3"]):
            return "Crypto"
        if any(w in comb for w in ["stock market", "stocks", "wall street", "dow jones", "nasdaq", "s&p 500", "nyse", "share prices", "equities", "dowjones", "economy.xml", "investing", "stock", "sensex", "nifty", "bse", "nse", "dalal street", "sebi", "rbi", "ipo", "livemint", "moneycontrol", "financialexpress"]):
            return "Stock Market"
        if any(w in comb for w in ["india", "indian", "delhi", "mumbai", "bengaluru", "hindu", "times of india", "indianexpress", "asia/india"]):
            return "India"
        if any(w in comb for w in ["business", "finance", "money", "revenue", "quarterly profit", "company"]):
            return "Business"
        if any(w in comb for w in ["tech", "technology", "software", "ai", "artificial intelligence", "gadget", "cyber"]):
            return "Technology"
        if any(w in comb for w in ["science", "environment", "climate", "space", "biology", "physics"]):
            return "Science"
        if any(w in comb for w in ["entertainment", "arts", "showbiz", "movie", "film", "music"]):
            return "Entertainment"
        if "sport" in comb:
            return "Sports"
        if "world" in comb:
            return "World"
        return "World"

    def ingest_feed(self, feed_url: str) -> IngestionResult:
        """
        Fast ingestion of articles from a single RSS feed URL.
        """
        logger.info(f"Starting fast ingestion for feed: {feed_url}")
        result = IngestionResult(total_fetched=0, inserted_count=0, duplicate_count=0, errors=[])

        if not HAS_FEEDPARSER:
            err_msg = "feedparser is not installed. Feed parsing skipped."
            logger.error(err_msg)
            result.errors.append(err_msg)
            return result

        try:
            feed = feedparser.parse(feed_url)
            if feed.get("bozo", 0) == 1 and not feed.get("entries"):
                err_msg = f"Failed to parse RSS feed or empty feed: {feed_url}"
                logger.error(err_msg)
                result.errors.append(err_msg)
                return result

            entries = feed.entries[:5]  # Limit to top 5 entries per feed
            result.total_fetched = len(entries)
            publisher = feed.feed.get("title", "News Source")

            from app.db.session import SessionLocal
            from app.models import DBArticle

            db = SessionLocal()
            try:
                db_records = db.query(DBArticle.url, DBArticle.content_hash).all()
                existing_urls = {r[0] for r in db_records if r[0]}
                existing_hashes = {r[1] for r in db_records if r[1]}
            finally:
                db.close()

            for entry in entries:
                try:
                    title = entry.get("title", "").strip()
                    url = entry.get("link", "").strip()
                    raw_summary = entry.get("summary", entry.get("description", "")).strip()

                    if not title or not url:
                        continue

                    # Use raw summary directly for fast instant response
                    full_content = raw_summary or title
                    if "<" in full_content and ">" in full_content:
                        import re
                        full_content = re.sub(r'<[^>]+>', '', full_content).strip()

                    category = self.detect_category(feed_url, publisher, title, full_content)

                    # Compute Hash
                    content_hash = self.compute_content_hash(title, full_content)

                    # Instant O(1) Memory & Hash Check
                    if url in existing_urls or content_hash in existing_hashes or content_hash in self.seen_hashes:
                        self.seen_hashes.add(content_hash)
                        result.duplicate_count += 1
                        continue

                    # Step 3: Semantic Vector Check (Only for truly new articles)
                    dup_result = self.check_duplication(title, full_content, content_hash)
                    if dup_result.is_duplicate:
                        result.duplicate_count += 1
                        continue

                    # Store new article into Vector Store
                    article_id = str(uuid.uuid4())
                    vector_added = self.vector_store.add_article(
                        article_id=article_id,
                        text=f"{title}\n{full_content}",
                        metadata={
                            "title": title,
                            "url": url,
                            "source": publisher,
                            "category": category,
                            "hash": content_hash
                        }
                    )

                    # Save to relational database
                    db = SessionLocal()
                    try:
                        db_article = DBArticle(
                            id=article_id,
                            title=title,
                            url=url,
                            content=full_content,
                            summary=full_content[:350],
                            category=category,
                            source=publisher,
                            content_hash=content_hash,
                            is_verified=True,
                            vector_id=article_id if vector_added else None
                        )
                        db.add(db_article)
                        db.commit()
                        self.seen_hashes.add(content_hash)
                        result.inserted_count += 1
                        logger.info(f"Successfully ingested [{category}] story into DB: '{title}'")
                    except Exception as db_e:
                        db.rollback()
                        logger.error(f"Failed DB save for '{title}': {str(db_e)}")
                    finally:
                        db.close()

                except Exception as entry_err:
                    err = f"Error processing entry '{entry.get('title', 'Unknown')}': {str(entry_err)}"
                    logger.error(err)
                    result.errors.append(err)

        except Exception as feed_err:
            err = f"Fatal error parsing feed {feed_url}: {str(feed_err)}"
            logger.error(err)
            result.errors.append(err)

        return result
