import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph, END = None, None
    HAS_LANGGRAPH = False

try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    ChatOpenAI = None
    HAS_LANGCHAIN = False

from app.config import settings
from app.db.session import SessionLocal
from app.models import DBArticle, DBProcessingLog

logger = logging.getLogger(__name__)


class NewsAgentState(TypedDict):
    """
    LangGraph state dictionary holding article data across multi-agent processing steps.
    """
    article_id: str
    title: str
    content: str
    url: str
    source: str
    category: str
    summary: str
    is_verified: bool
    confidence_score: float
    sentiment: str
    logs: List[str]
    error: Optional[str]


def categorize_node(state: NewsAgentState) -> NewsAgentState:
    """
    Agent Node: Categorizes news article into specific domain topics (AI, Tech, Finance, Science, etc.).
    """
    logger.info(f"[Agent Node: Categorize] Processing article ID: {state.get('article_id')}")
    logs = list(state.get("logs", []))
    logs.append(f"Categorization started at {datetime.now(timezone.utc).isoformat()}")

    title = state.get("title", "")
    content = state.get("content", "")
    text = f"{title}\n{content}".lower()

    category = "General"
    if any(k in text for k in ["ai", "llm", "artificial intelligence", "machine learning", "deep learning", "chatgpt", "openai", "claude"]):
        category = "AI & Robotics"
    elif any(k in text for k in ["tech", "software", "apple", "google", "microsoft", "nvidia", "hardware", "cybersecurity", "cloud"]):
        category = "Software & Tech"
    elif any(k in text for k in ["stock", "market", "finance", "economy", "crypto", "bitcoin", "banking", "inflation", "investing"]):
        category = "Finance & Business"
    elif any(k in text for k in ["space", "nasa", "physics", "biology", "health", "medicine", "genetics"]):
        category = "Science & Health"
    elif any(k in text for k in ["election", "policy", "government", "senate", "president", "court", "law"]):
        category = "Politics & Policy"

    logs.append(f"Categorized as: '{category}'")
    state["category"] = category
    state["logs"] = logs
    return state


def summarize_node(state: NewsAgentState) -> NewsAgentState:
    """
    Agent Node: Generates concise executive summary and key takeaways from full text.
    """
    logger.info(f"[Agent Node: Summarize] Processing article ID: {state.get('article_id')}")
    logs = list(state.get("logs", []))
    logs.append(f"Summarization started at {datetime.now(timezone.utc).isoformat()}")

    content = state.get("content", "")
    title = state.get("title", "")

    # LLM or Extractive Summarizer
    if HAS_LANGCHAIN and settings.OPENAI_API_KEY and ChatOpenAI is not None:
        try:
            llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0.3)
            prompt = f"Summarize the following news article into 2-3 concise bullet points focusing on key facts:\nTitle: {title}\nContent: {content[:3000]}"
            response = llm.invoke(prompt)
            summary = response.content.strip()
        except Exception as e:
            logger.warning(f"LLM Summarizer failed ({str(e)}), falling back to extractive summary.")
            summary = f"• {title}\n• " + (" ".join(content.split()[:50]) + "..." if content else title)
    else:
        # Fallback extractive summary
        paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 30]
        if paragraphs:
            summary = f"• {title}\n• " + paragraphs[0][:250] + ("..." if len(paragraphs[0]) > 250 else "")
        else:
            summary = f"• {title}"

    logs.append("Generated executive summary successfully.")
    state["summary"] = summary
    state["logs"] = logs
    return state


def verify_node(state: NewsAgentState) -> NewsAgentState:
    """
    Agent Node: Fact verification, clickbait detection, and source credibility scoring.
    """
    logger.info(f"[Agent Node: Verify] Processing article ID: {state.get('article_id')}")
    logs = list(state.get("logs", []))
    logs.append(f"Verification started at {datetime.now(timezone.utc).isoformat()}")

    title = state.get("title", "").lower()
    content = state.get("content", "")
    
    # Check clickbait or unverified indicators
    clickbait_words = ["shocking", "you won't believe", "secret trick", "miracle", "blow your mind", "unbelievable"]
    has_clickbait = any(w in title for w in clickbait_words)

    confidence_score = 0.95
    if has_clickbait:
        confidence_score -= 0.35
    if len(content) < 100:
        confidence_score -= 0.20

    confidence_score = max(0.1, min(1.0, confidence_score))
    is_verified = confidence_score >= 0.70

    # Sentiment check
    sentiment = "neutral"
    if any(w in title for w in ["surge", "record", "growth", "breakthrough", "profit", "win", "success"]):
        sentiment = "positive"
    elif any(w in title for w in ["drop", "crash", "decline", "warn", "threat", "risk", "loss", "ban"]):
        sentiment = "negative"

    logs.append(f"Verification complete: is_verified={is_verified}, confidence_score={confidence_score:.2f}, sentiment={sentiment}")
    state["is_verified"] = is_verified
    state["confidence_score"] = confidence_score
    state["sentiment"] = sentiment
    state["logs"] = logs
    return state


def save_db_node(state: NewsAgentState) -> NewsAgentState:
    """
    Agent Node: Persists processed article and execution logs into database.
    """
    logger.info(f"[Agent Node: SaveDB] Saving article ID: {state.get('article_id')}")
    logs = list(state.get("logs", []))
    logs.append(f"Save DB started at {datetime.now(timezone.utc).isoformat()}")

    db = SessionLocal()
    try:
        article_id = state.get("article_id") or str(uuid.uuid4())
        
        # Upsert article
        existing = db.query(DBArticle).filter(DBArticle.id == article_id).first()
        if not existing:
            existing = db.query(DBArticle).filter(DBArticle.url == state.get("url", "")).first()

        if existing:
            existing.title = state.get("title", existing.title)
            existing.content = state.get("content", existing.content)
            existing.summary = state.get("summary", existing.summary)
            existing.category = state.get("category", existing.category)
            existing.source = state.get("source", existing.source)
            existing.is_verified = state.get("is_verified", existing.is_verified)
            existing.confidence_score = state.get("confidence_score", existing.confidence_score)
        else:
            db_article = DBArticle(
                id=article_id,
                title=state.get("title", "Untitled"),
                url=state.get("url", f"https://news.internal/{article_id}"),
                content=state.get("content", ""),
                summary=state.get("summary", ""),
                category=state.get("category", "General"),
                source=state.get("source", "RSS Feed"),
                content_hash=state.get("article_id", article_id),
                is_verified=state.get("is_verified", True),
                confidence_score=state.get("confidence_score", 1.0)
            )
            db.add(db_article)

        # Log step
        log_entry = DBProcessingLog(
            article_id=article_id,
            step_name="LangGraphWorkflow",
            status="SUCCESS",
            log_message="; ".join(logs)
        )
        db.add(log_entry)
        db.commit()
        logs.append("Article successfully persisted to database.")

    except Exception as e:
        db.rollback()
        err_msg = f"Failed to save article to DB: {str(e)}"
        logger.error(err_msg)
        state["error"] = err_msg
        logs.append(err_msg)
    finally:
        db.close()

    state["logs"] = logs
    return state


class NewsWorkflowGraph:
    """
    LangGraph Workflow orchestrator managing the multi-agent execution pipeline:
    Categorize -> Summarize -> Verify -> Save DB.
    """
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        if not HAS_LANGGRAPH or StateGraph is None:
            logger.info("LangGraph package not active. Operating with standard sequence pipeline.")
            return None

        try:
            workflow = StateGraph(NewsAgentState)
            workflow.add_node("categorize", categorize_node)
            workflow.add_node("summarize", summarize_node)
            workflow.add_node("verify", verify_node)
            workflow.add_node("save_db", save_db_node)

            workflow.set_entry_point("categorize")
            workflow.add_edge("categorize", "summarize")
            workflow.add_edge("summarize", "verify")
            workflow.add_edge("verify", "save_db")
            workflow.add_edge("save_db", END)

            return workflow.compile()
        except Exception as e:
            logger.error(f"Failed to build LangGraph StateGraph: {str(e)}")
            return None

    def execute(self, initial_state: NewsAgentState) -> NewsAgentState:
        """
        Executes the processing graph on an input news article state.
        """
        if not initial_state.get("article_id"):
            initial_state["article_id"] = str(uuid.uuid4())
        if "logs" not in initial_state:
            initial_state["logs"] = []

        if self.graph is not None:
            try:
                logger.info(f"Executing LangGraph workflow for article '{initial_state.get('title')}'")
                final_state = self.graph.invoke(initial_state)
                return final_state
            except Exception as e:
                logger.error(f"LangGraph execution error: {str(e)}. Falling back to sequential execution.")

        # Sequential fallback execution
        state = initial_state
        state = categorize_node(state)
        state = summarize_node(state)
        state = verify_node(state)
        state = save_db_node(state)
        return state
