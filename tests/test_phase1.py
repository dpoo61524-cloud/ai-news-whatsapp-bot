import sys
import os
import unittest
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings, Settings
from app.models import ArticleCreate, ArticleResponse, IngestionResult, DuplicateCheckResult
from app.agents.vector_store import VectorStoreManager
from app.agents.ingestion import NewsIngestor

class TestPhase1(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_chroma_db"
        self.vector_store = VectorStoreManager(persist_dir=self.test_dir)

    def test_config_loading(self):
        """Verify configuration settings load with default values."""
        self.assertIsNotNone(settings.CHROMA_PERSIST_DIR)
        self.assertIsNotNone(settings.SIMILARITY_THRESHOLD)
        self.assertGreaterEqual(settings.SIMILARITY_THRESHOLD, 0.0)
        self.assertLessEqual(settings.SIMILARITY_THRESHOLD, 1.0)
        print("[OK] Config settings test passed.")

    def test_models_validation(self):
        """Verify Pydantic models instantiate and validate fields correctly."""
        article = ArticleCreate(
            title="Test AI Discovery",
            url="https://example.com/test-ai",
            content="AI agent model achieves high accuracy on vector search tasks.",
            source="Tech Test",
            content_hash="abc123hash"
        )
        self.assertEqual(article.title, "Test AI Discovery")
        self.assertEqual(article.content_hash, "abc123hash")

        dup_result = DuplicateCheckResult(
            is_duplicate=True,
            similarity_score=0.92,
            reason="High semantic similarity"
        )
        self.assertTrue(dup_result.is_duplicate)
        print("[OK] Pydantic models test passed.")

    def test_vector_store_operations(self):
        """Verify ChromaDB vector store insertion, deduplication, and search."""
        doc_id = "test-doc-101"
        text = "Artificial intelligence multi-agent framework orchestrates complex workflows."
        metadata = {"title": "AI Multi-Agent Framework", "source": "AI Tech"}

        # 1. Add article
        added = self.vector_store.add_article(doc_id, text, metadata)
        self.assertTrue(added)

        # 2. Test semantic duplicate detection for identical/similar text
        is_dup = self.vector_store.is_duplicate(text, threshold=0.80)
        self.assertTrue(is_dup, "Identical text should be detected as duplicate.")

        # 3. Test non-duplicate check
        different_text = "Cooking recipe for homemade sourdough bread and pasta."
        is_not_dup = self.vector_store.is_duplicate(different_text, threshold=0.85)
        self.assertFalse(is_not_dup, "Unrelated text should not be marked as duplicate.")

        # 4. Search similar articles
        results = self.vector_store.search_similar("multi-agent workflows AI", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], doc_id)
        print("[OK] Vector store operations test passed.")

    def test_news_ingestor_hashing(self):
        """Verify exact content hashing in NewsIngestor."""
        h1 = NewsIngestor.compute_content_hash("Title 1", "Content 1")
        h2 = NewsIngestor.compute_content_hash("Title 1", "Content 1")
        h3 = NewsIngestor.compute_content_hash("Title 2", "Content 2")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        print("[OK] News Ingestor hash test passed.")

if __name__ == "__main__":
    unittest.main()
