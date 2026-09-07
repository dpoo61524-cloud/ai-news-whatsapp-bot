import sys
import os
import argparse
import json
import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.config import settings
from app.db.session import init_db, SessionLocal
from app.models import DBArticle
from app.agents.vector_store import VectorStoreManager
from app.agents.ingestion import NewsIngestor
from app.agents.graph import NewsWorkflowGraph
from app.agents.conversational import ConversationalAgent
from app.services.scheduler import NewsScheduler, run_scheduled_ingestion


def main():
    parser = argparse.ArgumentParser(
        description="AI Multi-Agent News Platform & WhatsApp Webhook CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Run RSS feed ingestion and agent workflow")
    ingest_parser.add_argument("--feed", type=str, default=None, help="Optional specific RSS feed URL")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Perform semantic vector search")
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # Command: latest
    latest_parser = subparsers.add_parser("latest", help="Show recent stored news articles")
    latest_parser.add_argument("--limit", type=int, default=5, help="Number of articles to show")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Simulate a conversational user message")
    chat_parser.add_argument("message", type=str, help="Chat message to send")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start background scheduler and FastAPI server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    # Initialize DB
    init_db()

    if args.command == "ingest":
        print("Starting news feed ingestion and LangGraph agent workflow...")
        vector_store = VectorStoreManager()
        ingestor = NewsIngestor(vector_store=vector_store)
        workflow = NewsWorkflowGraph()

        feeds = [args.feed] if args.feed else settings.DEFAULT_RSS_FEEDS
        total_new = 0

        for f in feeds:
            res = ingestor.ingest_feed(f)
            total_new += res.inserted_count
            print(f"Feed '{f}': {res.inserted_count} new inserted, {res.duplicate_count} duplicates skipped.")

        print(f"Ingestion complete: {total_new} new articles inserted into vector store & database.")

    elif args.command == "search":
        print(f"Executing semantic vector search for query: '{args.query}'...")
        vector_store = VectorStoreManager()
        results = vector_store.search_similar(query=args.query, top_k=args.limit)

        print(f"\nFound {len(results)} matching articles:")
        for idx, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            print(f"\n{idx}. {meta.get('title', 'Untitled')} (Match: {r.get('similarity', 0)*100:.1f}%)")
            print(f"   URL: {meta.get('url', '#')}")
            print(f"   Snippet: {r.get('document', '')[:150]}...")

    elif args.command == "latest":
        db = SessionLocal()
        try:
            articles = db.query(DBArticle).order_by(DBArticle.created_at.desc()).limit(args.limit).all()
            print(f"\nRetrieved {len(articles)} recent articles from database:")
            for idx, a in enumerate(articles, 1):
                print(f"\n{idx}. [{a.category or 'General'}] {a.title}")
                print(f"   Summary: {a.summary or a.content[:100] + '...'}")
                print(f"   Source: {a.url}")
        finally:
            db.close()

    elif args.command == "chat":
        agent = ConversationalAgent()
        res = agent.process_message(args.message)
        print(f"\n[Detected Intent: {res.intent}]")
        print("Bot Response:")
        print(res.reply_text)

    elif args.command == "serve":
        print("Starting background scheduler and FastAPI server...")
        scheduler = NewsScheduler()
        scheduler.start(interval_minutes=15)
        try:
            uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
        finally:
            scheduler.stop()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
