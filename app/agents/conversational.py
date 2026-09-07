import logging
import re
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.models import (
    DBArticle,
    DBUserSubscription,
    ArticleResponse,
    ConversationalResponse,
    SubscriptionResponse
)
from app.agents.vector_store import VectorStoreManager
from app.agents.ingestion import NewsIngestor

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """
    Conversational Agent routing user prompts to intent handlers including
    Latest News, Semantic Search, Category Filters, Topic Subscriptions,
    and Interactive WhatsApp Quick-Reply Buttons.
    """
    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.ingestor = NewsIngestor(vector_store=self.vector_store)

    def classify_intent(self, message: str) -> str:
        """
        Classifies incoming user message intent using exact word boundaries.
        """
        msg_lower = message.strip().lower()

        # Check interactive button payload IDs first
        if msg_lower.startswith("btn_"):
            if "latest" in msg_lower:
                return "LATEST_NEWS"
            elif "ai" in msg_lower:
                return "CATEGORY_SEARCH"
            elif "subscribe" in msg_lower:
                return "MY_SUBSCRIPTIONS"
            elif "help" in msg_lower:
                return "HELP"

        # Subscription Intent Matching
        if re.search(r"\b(unsubscribe|unsub)\b", msg_lower):
            return "UNSUBSCRIBE"
        elif re.search(r"\b(subscribe|sub)\b", msg_lower):
            return "SUBSCRIBE"
        elif re.search(r"\b(subscriptions|my subs|subbed)\b", msg_lower):
            return "MY_SUBSCRIPTIONS"

        # Standard Intent Matching
        if re.search(r"\b(help|commands|menu|hi|hello|start)\b", msg_lower):
            return "HELP"
        elif re.search(r"\b(refresh|ingest|fetch|update)\b", msg_lower):
            return "INGEST_REFRESH"
        elif re.search(r"\b(latest|recent|top|headlines|summary)\b", msg_lower):
            return "LATEST_NEWS"
        elif re.search(r"\b(ai|robotics|tech|technology|finance|business|science|politics|crypto|coin|bitcoin|ethereum|stock|market|stocks|markets|economy|investing|world|india|indian|delhi|mumbai|sensex|nifty|bse|nse|sebi|rbi|ipo)\b", msg_lower):
            return "CATEGORY_SEARCH"
        else:
            return "KEYWORD_SEARCH"

    @staticmethod
    def default_buttons() -> List[Dict[str, str]]:
        """Default quick-reply buttons."""
        return [
            {"id": "btn_latest", "title": "Latest News"},
            {"id": "btn_tech", "title": "Tech News"},
            {"id": "btn_help", "title": "Help Menu"}
        ]

    def handle_help(self) -> ConversationalResponse:
        """Returns structured help menu."""
        reply = (
            "*News Assistant Help*\n\n"
            "Here is what you can ask me:\n"
            "• *'Latest news'* - Get top recent executive summaries\n"
            "• *'Tech news'* / *'Business news'* - Get category-specific digests\n"
            "• *'Search [topic]'* - Search news articles\n"
            "• *'Subscribe Tech'* - Receive daily scheduled digests\n"
            "• *'My subscriptions'* - View active topic subscriptions\n"
            "• *'Refresh news'* - Trigger RSS feed ingestion\n"
            "• *'Help'* - Show this menu"
        )
        return ConversationalResponse(
            reply_text=reply,
            intent="HELP",
            buttons=[
                {"id": "btn_latest", "title": "Latest News"},
                {"id": "btn_tech", "title": "Tech News"},
                {"id": "btn_subscribe", "title": "Subscriptions"}
            ]
        )

    def handle_latest_news(self, limit: int = 5) -> ConversationalResponse:
        """Retrieves top verified articles from database."""
        db: Session = SessionLocal()
        try:
            db_articles = (
                db.query(DBArticle)
                .order_by(DBArticle.created_at.desc())
                .limit(limit)
                .all()
            )

            if not db_articles:
                return ConversationalResponse(
                    reply_text="No news articles found in the database yet. Send *'Refresh news'* to fetch stories!",
                    intent="LATEST_NEWS",
                    buttons=ConversationalAgent.default_buttons()
                )

            articles_dto = [ArticleResponse.model_validate(a) for a in db_articles]
            
            reply_lines = [f"*Top {len(articles_dto)} News Summaries:*\n"]
            for idx, a in enumerate(articles_dto, 1):
                cat_tag = f"[{a.category}]" if a.category else ""
                reply_lines.append(
                    f"{idx}. *{a.title}* {cat_tag}\n"
                    f"{a.summary or a.content[:150] + '...'}\n"
                    f"Source: {a.url}\n"
                )

            return ConversationalResponse(
                reply_text="\n".join(reply_lines),
                articles_referenced=articles_dto,
                intent="LATEST_NEWS",
                buttons=ConversationalAgent.default_buttons()
            )
        finally:
            db.close()

    def handle_category_search(self, category_keyword: str, limit: int = 5) -> ConversationalResponse:
        """Retrieves articles filtered by category domain."""
        db: Session = SessionLocal()
        try:
            category_mapping = {
                "stock market": "Stock Market",
                "indian stock": "Stock Market",
                "share market": "Stock Market",
                "stock": "Stock Market",
                "market": "Stock Market",
                "stocks": "Stock Market",
                "markets": "Stock Market",
                "sensex": "Stock Market",
                "nifty": "Stock Market",
                "bse": "Stock Market",
                "nse": "Stock Market",
                "sebi": "Stock Market",
                "rbi": "Stock Market",
                "ipo": "Stock Market",
                "economy": "Stock Market",
                "investing": "Stock Market",
                "crypto": "Crypto",
                "coin": "Crypto",
                "bitcoin": "Crypto",
                "ethereum": "Crypto",
                "ai": "AI & Robotics",
                "robotics": "AI & Robotics",
                "tech": "Software & Tech",
                "technology": "Software & Tech",
                "software": "Software & Tech",
                "finance": "Finance & Business",
                "business": "Finance & Business",
                "science": "Science & Health",
                "politics": "Politics & Policy",
                "world": "World",
                "india": "India",
                "indian": "India",
                "delhi": "India",
                "mumbai": "India",
            }
            
            matched_category = "General"
            # Sort mapping keys by length descending so longer specific terms (e.g. 'stock market') match before general terms ('india')
            for k in sorted(category_mapping.keys(), key=len, reverse=True):
                if k in category_keyword.lower():
                    matched_category = category_mapping[k]
                    break

            db_articles = (
                db.query(DBArticle)
                .filter(DBArticle.category.ilike(f"%{matched_category}%"))
                .order_by(DBArticle.created_at.desc())
                .limit(limit)
                .all()
            )

            if not db_articles:
                db_articles = db.query(DBArticle).order_by(DBArticle.created_at.desc()).limit(limit).all()

            articles_dto = [ArticleResponse.model_validate(a) for a in db_articles]
            
            reply_lines = [f"*Top {matched_category} Stories:*\n"]
            for idx, a in enumerate(articles_dto, 1):
                reply_lines.append(
                    f"{idx}. *{a.title}*\n"
                    f"{a.summary or a.content[:150] + '...'}\n"
                    f"Source: {a.url}\n"
                )

            return ConversationalResponse(
                reply_text="\n".join(reply_lines),
                articles_referenced=articles_dto,
                intent="CATEGORY_SEARCH",
                buttons=ConversationalAgent.default_buttons()
            )
        finally:
            db.close()

    def handle_keyword_search(self, query: str, limit: int = 4) -> ConversationalResponse:
        """Performs semantic vector search over stored articles."""
        results = self.vector_store.search_similar(query=query, top_k=limit)

        if not results:
            return ConversationalResponse(
                reply_text=f"No matching articles found for query: *'{query}'*.",
                intent="KEYWORD_SEARCH",
                buttons=ConversationalAgent.default_buttons()
            )

        reply_lines = [f"*Search Results for '{query}':*\n"]
        for idx, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            title = meta.get("title", "News Article")
            url = meta.get("url", "#")
            doc = r.get("document", "")[:200]
            similarity = r.get("similarity", 0.0)

            reply_lines.append(
                f"{idx}. *{title}* (Match: {similarity*100:.0f}%)\n"
                f"{doc}...\n"
                f"Source: {url}\n"
            )

        return ConversationalResponse(
            reply_text="\n".join(reply_lines),
            intent="KEYWORD_SEARCH",
            buttons=ConversationalAgent.default_buttons()
        )

    def handle_subscribe(self, message: str, sender_id: str) -> ConversationalResponse:
        """Subscribes user phone to a specific topic category."""
        if not sender_id or sender_id == "unknown":
            return ConversationalResponse(
                reply_text="Subscription requires a valid user ID.",
                intent="SUBSCRIBE"
            )

        # Detect requested topic
        msg_lower = message.lower()
        topic = "All"
        if "ai" in msg_lower:
            topic = "AI & Robotics"
        elif "tech" in msg_lower or "technology" in msg_lower:
            topic = "Software & Tech"
        elif "finance" in msg_lower or "business" in msg_lower:
            topic = "Finance & Business"
        elif "science" in msg_lower:
            topic = "Science & Health"

        db: Session = SessionLocal()
        try:
            existing = (
                db.query(DBUserSubscription)
                .filter(DBUserSubscription.user_phone == sender_id, DBUserSubscription.topic == topic)
                .first()
            )

            if existing:
                existing.is_active = True
                db.commit()
                reply = f"*Subscription Active:* You are already subscribed to *'{topic}'* daily digests."
            else:
                new_sub = DBUserSubscription(
                    user_phone=sender_id,
                    topic=topic,
                    digest_time="08:00",
                    is_active=True
                )
                db.add(new_sub)
                db.commit()
                reply = f"*Successfully Subscribed!* You will receive daily news digests for *'{topic}'* at 08:00 AM."

            return ConversationalResponse(
                reply_text=reply,
                intent="SUBSCRIBE",
                buttons=[
                    {"id": "btn_latest", "title": "Latest News"},
                    {"id": "btn_subscribe", "title": "My Subs"}
                ]
            )
        except Exception as e:
            db.rollback()
            return ConversationalResponse(
                reply_text=f"Failed to update subscription: {str(e)}",
                intent="SUBSCRIBE"
            )
        finally:
            db.close()

    def handle_unsubscribe(self, message: str, sender_id: str) -> ConversationalResponse:
        """Deactivates topic subscriptions for sender."""
        db: Session = SessionLocal()
        try:
            subs = (
                db.query(DBUserSubscription)
                .filter(DBUserSubscription.user_phone == sender_id, DBUserSubscription.is_active == True)
                .all()
            )

            if not subs:
                return ConversationalResponse(
                    reply_text="You do not have any active subscriptions.",
                    intent="UNSUBSCRIBE"
                )

            for s in subs:
                s.is_active = False
            db.commit()

            return ConversationalResponse(
                reply_text="*Unsubscribed:* All your daily news digests have been deactivated.",
                intent="UNSUBSCRIBE",
                buttons=ConversationalAgent.default_buttons()
            )
        finally:
            db.close()

    def handle_my_subscriptions(self, sender_id: str) -> ConversationalResponse:
        """Lists active subscriptions for sender."""
        db: Session = SessionLocal()
        try:
            subs = (
                db.query(DBUserSubscription)
                .filter(DBUserSubscription.user_phone == sender_id, DBUserSubscription.is_active == True)
                .all()
            )

            if not subs:
                reply = (
                    "*Active Subscriptions: None*\n\n"
                    "Send *'Subscribe Tech'* or *'Subscribe Business'* to receive daily digests at 08:00 AM."
                )
            else:
                lines = ["*Your Active News Subscriptions:*\n"]
                for s in subs:
                    lines.append(f"• *{s.topic}* (Daily at {s.digest_time})")
                reply = "\n".join(lines)

            return ConversationalResponse(
                reply_text=reply,
                intent="MY_SUBSCRIPTIONS",
                buttons=ConversationalAgent.default_buttons()
            )
        finally:
            db.close()

    def handle_ingest_refresh(self) -> ConversationalResponse:
        """Triggers manual feed ingestion across default feeds concurrently for sub-second performance."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total_inserted = 0
        total_duplicates = 0
        errors = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self.ingestor.ingest_feed, feed_url): feed_url for feed_url in settings.DEFAULT_RSS_FEEDS}
            for future in as_completed(future_to_url):
                try:
                    res = future.result()
                    total_inserted += res.inserted_count
                    total_duplicates += res.duplicate_count
                    errors.extend(res.errors)
                except Exception as exc:
                    errors.append(str(exc))

        reply = (
            f"Feed Ingestion Complete\n\n"
            f"• New Articles Inserted: {total_inserted}\n"
            f"• Duplicates Filtered: {total_duplicates}\n"
        )

        return ConversationalResponse(
            reply_text=reply,
            intent="INGEST_REFRESH",
            buttons=ConversationalAgent.default_buttons()
        )

    def process_message(self, message_body: str, sender_id: str = "") -> ConversationalResponse:
        """
        Main entry point for processing incoming WhatsApp/chat messages.
        """
        if not message_body or not message_body.strip():
            return self.handle_help()

        intent = self.classify_intent(message_body)
        logger.info(f"Processing chat message from '{sender_id}': '{message_body}' -> Intent: {intent}")

        if intent == "HELP":
            return self.handle_help()
        elif intent == "SUBSCRIBE":
            return self.handle_subscribe(message_body, sender_id)
        elif intent == "UNSUBSCRIBE":
            return self.handle_unsubscribe(message_body, sender_id)
        elif intent == "MY_SUBSCRIPTIONS":
            return self.handle_my_subscriptions(sender_id)
        elif intent == "LATEST_NEWS":
            return self.handle_latest_news()
        elif intent == "CATEGORY_SEARCH":
            return self.handle_category_search(message_body)
        elif intent == "INGEST_REFRESH":
            return self.handle_ingest_refresh()
        else:
            return self.handle_keyword_search(message_body)
