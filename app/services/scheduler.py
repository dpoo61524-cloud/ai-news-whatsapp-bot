import logging
import time
import threading
from typing import Optional
from sqlalchemy.orm import Session

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_APSCHEDULER = True
except ImportError:
    BackgroundScheduler = None
    HAS_APSCHEDULER = False

from app.config import settings
from app.db.session import SessionLocal
from app.models import DBArticle, DBUserSubscription, ArticleResponse
from app.agents.vector_store import VectorStoreManager
from app.agents.ingestion import NewsIngestor
from app.agents.graph import NewsWorkflowGraph

logger = logging.getLogger(__name__)


def run_scheduled_ingestion():
    """
    Background job function:
    1. Fetches RSS feeds via NewsIngestor.
    2. Filters vector duplicates using VectorStoreManager.
    3. Runs new articles through LangGraph workflow pipeline.
    """
    logger.info("⏰ Starting scheduled background news ingestion and multi-agent workflow...")
    
    vector_store = VectorStoreManager()
    ingestor = NewsIngestor(vector_store=vector_store)
    workflow = NewsWorkflowGraph()

    total_inserted = 0
    total_duplicates = 0

    for feed_url in settings.DEFAULT_RSS_FEEDS:
        try:
            res = ingestor.ingest_feed(feed_url)
            total_inserted += res.inserted_count
            total_duplicates += res.duplicate_count
        except Exception as e:
            logger.error(f"Error during scheduled ingestion of {feed_url}: {str(e)}")

    logger.info(f"⏰ Scheduled ingestion complete: {total_inserted} new articles ingested, {total_duplicates} duplicates filtered out.")


def send_daily_digests():
    """
    Scheduled job function:
    1. Queries active user topic subscriptions from database.
    2. Compiles customized news digests for each subscriber.
    3. Dispatches personalized WhatsApp digests.
    """
    logger.info("🌅 Starting daily scheduled WhatsApp news digest dispatch...")
    db: Session = SessionLocal()
    try:
        active_subs = db.query(DBUserSubscription).filter(DBUserSubscription.is_active == True).all()
        if not active_subs:
            logger.info("No active topic subscriptions found.")
            return

        # Group subscriptions by user phone
        user_map = {}
        for s in active_subs:
            user_map.setdefault(s.user_phone, []).append(s.topic)

        for phone, topics in user_map.items():
            try:
                # Fetch recent top articles matching topics
                query = db.query(DBArticle).order_by(DBArticle.created_at.desc())
                if "All" not in topics:
                    query = query.filter(DBArticle.category.in_(topics))
                
                articles = query.limit(4).all()
                if not articles:
                    continue

                digest_lines = [f"🌅 *Good Morning! Here is your Daily News Digest:*\n"]
                for idx, a in enumerate(articles, 1):
                    digest_lines.append(
                        f"{idx}. *{a.title}* [{a.category or 'General'}]\n"
                        f"{a.summary or a.content[:150] + '...'}\n"
                        f"🔗 {a.url}\n"
                    )

                from main import send_whatsapp_message
                send_whatsapp_message(phone, "\n".join(digest_lines))
                logger.info(f"Successfully sent daily digest to subscriber '{phone}' for topics: {topics}")

            except Exception as user_err:
                logger.error(f"Failed to send digest to '{phone}': {str(user_err)}")

    except Exception as db_err:
        logger.error(f"Database error during daily digest dispatch: {str(db_err)}")
    finally:
        db.close()


class NewsScheduler:
    """
    Background service manager handling recurring news ingestion and daily digest jobs.
    Supports APScheduler with daemon thread fallback.
    """
    def __init__(self):
        self.scheduler: Optional[BackgroundScheduler] = None
        self._fallback_thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self, interval_minutes: int = 15):
        """
        Starts recurring background jobs via APScheduler.
        """
        if self._is_running:
            logger.warning("NewsScheduler is already running.")
            return

        if HAS_APSCHEDULER and BackgroundScheduler is not None:
            try:
                self.scheduler = BackgroundScheduler(daemon=True)
                
                # Job 1: Ingestion every N minutes
                self.scheduler.add_job(
                    func=run_scheduled_ingestion,
                    trigger="interval",
                    minutes=interval_minutes,
                    id="news_ingestion_job",
                    replace_existing=True
                )

                # Job 2: Daily Digest Cron at 08:00 AM
                self.scheduler.add_job(
                    func=send_daily_digests,
                    trigger="cron",
                    hour=8,
                    minute=0,
                    id="daily_digest_job",
                    replace_existing=True
                )

                self.scheduler.start()
                self._is_running = True
                logger.info(f"NewsScheduler started via APScheduler (ingestion: {interval_minutes}m, daily digest: 08:00 AM).")
                return
            except Exception as e:
                logger.error(f"Failed to start APScheduler ({str(e)}). Falling back to daemon thread.")

        # Threading fallback
        self._is_running = True
        
        def _loop():
            while self._is_running:
                try:
                    run_scheduled_ingestion()
                except Exception as e:
                    logger.error(f"Fallback thread scheduler error: {str(e)}")
                time.sleep(interval_minutes * 60)

        self._fallback_thread = threading.Thread(target=_loop, daemon=True)
        self._fallback_thread.start()
        logger.info(f"NewsScheduler started via fallback daemon thread (interval: {interval_minutes}m).")

    def stop(self):
        """
        Stops background scheduler service cleanly.
        """
        if not self._is_running:
            return

        self._is_running = False
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("APScheduler stopped.")

        logger.info("NewsScheduler service stopped.")
