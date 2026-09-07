import logging
from typing import Dict, Any, Optional, List
import os
from fastapi import FastAPI, Request, Response, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests
import html

from app.config import settings
from app.db.session import init_db, get_db
from app.models import DBArticle, DBUserSubscription, ArticleResponse, SubscriptionResponse, SubscriptionCreate
from app.agents.conversational import ConversationalAgent
from app.agents.vector_store import VectorStoreManager

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("news_agents.main")

# Initialize FastAPI App
app = FastAPI(
    title="AI Multi-Agent News Platform & Web Dashboard",
    description="Autonomous news ingestion, vector deduplication, LangGraph multi-agent processing, web UI dashboard, and user topic subscriptions.",
    version="1.2.0"
)

# Global instances
vector_store = VectorStoreManager()
conversational_agent = ConversationalAgent(vector_store=vector_store)

# Mount static directory for frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", summary="Serve Web Dashboard")
def serve_index():
    """Serves the single page frontend web application."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI News Platform API is running. Access /docs for API documentation."}



@app.on_event("startup")
def on_startup():
    """App startup event initializing database schema."""
    logger.info("Starting AI News Agents Application...")
    init_db()


@app.get("/health", summary="Healthcheck endpoint")
def health_check():
    """Healthcheck endpoint returning system component status."""
    return {
        "status": "online",
        "env": settings.ENV,
        "vector_store_persist_dir": settings.CHROMA_PERSIST_DIR,
        "database_url": settings.DATABASE_URL
    }


# ---------------------------------------------------------------------------
# WhatsApp Webhook Integration Endpoints
# ---------------------------------------------------------------------------
@app.get("/webhook", summary="WhatsApp Webhook Verification (Meta API)")
def verify_whatsapp_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge")
):
    """
    Verification endpoint called by Meta WhatsApp Cloud API during webhook configuration.
    """
    logger.info(f"Webhook verification request: mode={hub_mode}, verify_token={hub_verify_token}")

    if hub_mode == "subscribe" and hub_verify_token == settings.WHATSAPP_VERIFY_TOKEN:
        logger.info("WhatsApp Webhook verification successful!")
        return PlainTextResponse(content=hub_challenge or "")
    
    logger.warning("WhatsApp Webhook verification failed due to invalid token match.")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhook", summary="WhatsApp Webhook Message Receiver")
async def receive_whatsapp_webhook(request: Request):
    """
    Receives incoming text or interactive button WhatsApp messages from Meta API or Twilio,
    routes message through ConversationalAgent, and sends response.
    Supports both Meta JSON payload and Twilio form-encoded webhook with TwiML.
    """
    try:
        content_type = request.headers.get("content-type", "")
        sender_id = "unknown"
        message_text = ""
        is_twilio = False

        if "application/x-www-form-urlencoded" in content_type:
            # Twilio Webhook Payload
            is_twilio = True
            form_data = await request.form()
            sender_id = form_data.get("From", "unknown").replace("whatsapp:", "")
            message_text = form_data.get("Body", "")
            logger.info(f"Received Twilio WhatsApp webhook: sender={sender_id}, text='{message_text}'")
        else:
            # Meta WhatsApp Cloud API Payload or generic JSON
            data = await request.json()
            logger.info(f"Received WhatsApp JSON webhook payload: {data}")

            # Extract Meta WhatsApp Cloud API payload format
            entries = data.get("entry", [])
            if entries:
                changes = entries[0].get("changes", [])
                if changes:
                    value = changes[0].get("value", {})
                    messages = value.get("messages", [])
                    if messages:
                        msg = messages[0]
                        sender_id = msg.get("from", "unknown")
                        msg_type = msg.get("type")

                        if msg_type == "text":
                            message_text = msg.get("text", {}).get("body", "")
                        elif msg_type == "interactive":
                            # Interactive Button Reply
                            interactive = msg.get("interactive", {})
                            if interactive.get("type") == "button_reply":
                                btn_reply = interactive.get("button_reply", {})
                                message_text = btn_reply.get("id") or btn_reply.get("title", "")

            # Fallback to simple JSON body
            if not message_text:
                sender_id = data.get("sender_id", data.get("from", "unknown"))
                message_text = data.get("message_body", data.get("text", data.get("body", "")))

        if not message_text:
            return {"status": "ignored", "reason": "No message body found"}

        # Route message through Conversational Agent
        conv_response = conversational_agent.process_message(
            message_body=message_text,
            sender_id=sender_id
        )

        # For Twilio, return TwiML XML for instant automatic reply
        if is_twilio:
            # Twilio WhatsApp caps single <Message> bodies at 1600 characters.
            # We wrap in CDATA and chunk long replies into multiple <Message> tags.
            reply_text = conv_response.reply_text or ""
            chunk_size = 1400
            chunks = [reply_text[i:i + chunk_size] for i in range(0, len(reply_text), chunk_size)] if reply_text else ["No content"]
            
            messages_xml = "".join([f"<Message><![CDATA[{c}]]></Message>" for c in chunks])
            twiml_xml = f'<?xml version="1.0" encoding="UTF-8"?><Response>{messages_xml}</Response>'
            return Response(content=twiml_xml, media_type="text/xml")

        # Send response via Meta WhatsApp Cloud API if configured
        if settings.WHATSAPP_API_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID and sender_id != "unknown":
            if conv_response.buttons:
                send_whatsapp_interactive_buttons(sender_id, conv_response.reply_text, conv_response.buttons)
            else:
                send_whatsapp_message(sender_id, conv_response.reply_text)

        return {
            "status": "success",
            "sender_id": sender_id,
            "intent": conv_response.intent,
            "reply_text": conv_response.reply_text,
            "buttons": conv_response.buttons
        }

    except Exception as e:
        logger.error(f"Error handling WhatsApp webhook payload: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


def send_whatsapp_message(to_phone: str, text_body: str):
    """Helper function sending outbound standard text message via Meta WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v18.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text_body}
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        logger.info(f"Outbound WhatsApp API response ({res.status_code}): {res.text}")
    except Exception as e:
        logger.error(f"Failed to post text message to WhatsApp Cloud API: {str(e)}")


def send_whatsapp_interactive_buttons(to_phone: str, body_text: str, buttons: List[Dict[str, str]]):
    """Helper function sending interactive quick-reply buttons via Meta WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v18.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json"
    }
    formatted_buttons = []
    for btn in buttons[:3]:  # WhatsApp API caps at 3 quick-reply buttons per message
        formatted_buttons.append({
            "type": "reply",
            "reply": {
                "id": btn.get("id", "btn_action"),
                "title": btn.get("title", "Action")[:20]  # Max 20 chars
            }
        })

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text[:1024]},
            "footer": {"text": "AI News Agent"},
            "action": {"buttons": formatted_buttons}
        }
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        logger.info(f"Outbound Interactive WhatsApp API response ({res.status_code}): {res.text}")
    except Exception as e:
        logger.error(f"Failed to post interactive buttons to WhatsApp Cloud API: {str(e)}")


# ---------------------------------------------------------------------------
# Interactive REST API Endpoints
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="Message string to send to conversational agent")
    sender_id: Optional[str] = Field("test_user", description="Identifier of sender")


@app.post("/api/chat", summary="Test Chat Endpoint")
def test_chat(payload: ChatRequest):
    """
    Interactive REST endpoint allowing direct test chat queries without WhatsApp client.
    """
    res = conversational_agent.process_message(
        message_body=payload.message,
        sender_id=payload.sender_id or "test_user"
    )
    return {
        "intent": res.intent,
        "reply_text": res.reply_text,
        "buttons": res.buttons,
        "articles_referenced": res.articles_referenced
    }


@app.post("/api/subscriptions", summary="Create User Topic Subscription")
def create_subscription(sub: SubscriptionCreate, db=Depends(get_db)):
    """Creates or updates a user topic subscription."""
    existing = (
        db.query(DBUserSubscription)
        .filter(DBUserSubscription.user_phone == sub.user_phone, DBUserSubscription.topic == sub.topic)
        .first()
    )
    if existing:
        existing.is_active = True
        db.commit()
        db.refresh(existing)
        return SubscriptionResponse.model_validate(existing)

    new_sub = DBUserSubscription(
        user_phone=sub.user_phone,
        topic=sub.topic,
        digest_time=sub.digest_time or "08:00",
        is_active=True
    )
    db.add(new_sub)
    db.commit()
    db.refresh(new_sub)
    return SubscriptionResponse.model_validate(new_sub)


@app.get("/api/subscriptions", summary="List Subscriptions")
def list_subscriptions(user_phone: Optional[str] = None, db=Depends(get_db)):
    """Retrieves list of active user subscriptions."""
    query = db.query(DBUserSubscription).filter(DBUserSubscription.is_active == True)
    if user_phone:
        clean_phone = user_phone.strip()
        if clean_phone.startswith(" "):
            clean_phone = "+" + clean_phone.lstrip()
        query = query.filter(
            (DBUserSubscription.user_phone == clean_phone) |
            (DBUserSubscription.user_phone == user_phone.strip())
        )
    subs = query.all()
    return [SubscriptionResponse.model_validate(s) for s in subs]


@app.post("/api/ingest", summary="Trigger Manual News Ingestion")
def trigger_ingest(feed_url: Optional[str] = None):
    """Triggers fast parallel feed ingestion for specified URL or default feed list."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if feed_url:
        res = conversational_agent.ingestor.ingest_feed(feed_url)
        return res.model_dump()
    
    total_inserted = 0
    total_duplicates = 0

    # Pick top 6 priority RSS feeds for sub-3-second Web UI response
    priority_feeds = settings.DEFAULT_RSS_FEEDS[:6]

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_url = {executor.submit(conversational_agent.ingestor.ingest_feed, url): url for url in priority_feeds}
        for f in as_completed(future_to_url, timeout=5.0):
            try:
                r = f.result()
                total_inserted += r.inserted_count
                total_duplicates += r.duplicate_count
            except Exception:
                pass

    return {
        "status": "success",
        "inserted_count": total_inserted,
        "duplicate_count": total_duplicates
    }


@app.get("/api/articles", summary="List Ingested News Articles")
def list_articles(limit: int = 25, db=Depends(get_db)):
    """Retrieves stored news articles from database."""
    articles = db.query(DBArticle).order_by(DBArticle.created_at.desc()).limit(limit).all()
    return [ArticleResponse.model_validate(a) for a in articles]


@app.get("/api/search", summary="Vector Semantic Search")
def search_articles(q: str = Query(..., description="Semantic search query"), limit: int = 5):
    """Executes semantic similarity search against ChromaDB vector store."""
    results = vector_store.search_similar(query=q, top_k=limit)
    return results

