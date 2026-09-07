import sys
import os
import unittest
import uuid

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import init_db, SessionLocal
from app.models import DBArticle, DBProcessingLog
from app.agents.graph import (
    NewsAgentState,
    categorize_node,
    summarize_node,
    verify_node,
    save_db_node,
    NewsWorkflowGraph
)

class TestPhase2(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_individual_agent_nodes(self):
        """Test individual agent state transformation nodes."""
        state: NewsAgentState = {
            "article_id": str(uuid.uuid4()),
            "title": "Nvidia Releases New AI Chip Breakthrough",
            "content": "Nvidia announced a revolutionary GPU architecture for deep learning model training with record speed.",
            "url": "https://example.com/nvidia-ai-chip",
            "source": "Tech News",
            "category": "",
            "summary": "",
            "is_verified": False,
            "confidence_score": 0.0,
            "sentiment": "",
            "logs": [],
            "error": None
        }

        # 1. Test Categorize Node
        state = categorize_node(state)
        self.assertEqual(state["category"], "AI & Robotics")

        # 2. Test Summarize Node
        state = summarize_node(state)
        self.assertTrue(len(state["summary"]) > 0)
        self.assertIn("Nvidia", state["summary"])

        # 3. Test Verify Node
        state = verify_node(state)
        self.assertTrue(state["is_verified"])
        self.assertGreaterEqual(state["confidence_score"], 0.7)
        self.assertEqual(state["sentiment"], "positive")
        print("[OK] Individual agent nodes test passed.")

    def test_workflow_graph_execution_and_persistence(self):
        """Test end-to-end LangGraph workflow execution and database persistence."""
        article_id = str(uuid.uuid4())
        initial_state: NewsAgentState = {
            "article_id": article_id,
            "title": "Central Banks Announce Interest Rate Decision",
            "content": "Financial markets react as economic policy makers adjust interest rates to manage inflation risks.",
            "url": f"https://example.com/finance-{article_id}",
            "source": "Financial Digest",
            "category": "",
            "summary": "",
            "is_verified": False,
            "confidence_score": 0.0,
            "sentiment": "",
            "logs": [],
            "error": None
        }

        graph = NewsWorkflowGraph()
        final_state = graph.execute(initial_state)

        # Assert state outputs
        self.assertEqual(final_state["category"], "Finance & Business")
        self.assertTrue(len(final_state["summary"]) > 0)
        self.assertTrue(final_state["is_verified"])

        # Assert database record creation
        db = SessionLocal()
        try:
            db_article = db.query(DBArticle).filter(DBArticle.id == article_id).first()
            self.assertIsNotNone(db_article)
            self.assertEqual(db_article.title, "Central Banks Announce Interest Rate Decision")
            self.assertEqual(db_article.category, "Finance & Business")

            log_entry = db.query(DBProcessingLog).filter(DBProcessingLog.article_id == article_id).first()
            self.assertIsNotNone(log_entry)
            self.assertEqual(log_entry.status, "SUCCESS")
        finally:
            db.close()

        print("[OK] LangGraph workflow graph execution and DB persistence test passed.")

if __name__ == "__main__":
    unittest.main()
