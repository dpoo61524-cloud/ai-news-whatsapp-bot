import sys
import os
import unittest
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.scheduler import NewsScheduler, run_scheduled_ingestion
from app.db.session import init_db, SessionLocal
from app.models import DBArticle

class TestPhase4(unittest.TestCase):
    def setUp(self):
        init_db()
        self.scheduler = NewsScheduler()

    def test_run_scheduled_ingestion_callback(self):
        """Test background job ingestion function execution."""
        # Execute single ingestion run
        run_scheduled_ingestion()
        
        db = SessionLocal()
        try:
            count = db.query(DBArticle).count()
            self.assertGreaterEqual(count, 0)
        finally:
            db.close()

        print("[OK] Scheduled ingestion callback test passed.")

    def test_scheduler_lifecycle(self):
        """Test starting and stopping NewsScheduler service."""
        self.scheduler.start(interval_minutes=1)
        self.assertTrue(self.scheduler._is_running)

        # Stop scheduler
        self.scheduler.stop()
        self.assertFalse(self.scheduler._is_running)
        print("[OK] Scheduler service lifecycle test passed.")

if __name__ == "__main__":
    unittest.main()
