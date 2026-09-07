import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from app.db.session import init_db, SessionLocal
from app.models import DBUserSubscription
from app.agents.conversational import ConversationalAgent
from app.services.scheduler import send_daily_digests

class TestAddons(unittest.TestCase):
    def setUp(self):
        init_db()
        self.client = TestClient(app)
        self.agent = ConversationalAgent()

        # Clean test user subscriptions
        db = SessionLocal()
        try:
            db.query(DBUserSubscription).filter(
                DBUserSubscription.user_phone.in_(["+15559876543", "+1999888777"])
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()

    def test_topic_subscription_intent_and_persistence(self):
        """Test subscribing, listing, and unsubscribing via ConversationalAgent."""
        test_phone = "+15559876543"

        # 1. Subscribe to AI
        res_sub = self.agent.process_message("Subscribe AI", sender_id=test_phone)
        self.assertEqual(res_sub.intent, "SUBSCRIBE")
        self.assertIn("Subscribed", res_sub.reply_text)

        # 2. Query My Subscriptions
        res_my = self.agent.process_message("My subscriptions", sender_id=test_phone)
        self.assertEqual(res_my.intent, "MY_SUBSCRIPTIONS")
        self.assertIn("AI & Robotics", res_my.reply_text)

        # 3. Unsubscribe
        res_unsub = self.agent.process_message("Unsubscribe", sender_id=test_phone)
        self.assertEqual(res_unsub.intent, "UNSUBSCRIBE")
        self.assertIn("Unsubscribed", res_unsub.reply_text)
        print("[OK] Topic subscription intent and persistence test passed.")

    def test_interactive_buttons_in_response(self):
        """Test that responses attach Meta WhatsApp interactive quick-reply buttons."""
        res = self.agent.process_message("Help")
        self.assertIsNotNone(res.buttons)
        self.assertGreaterEqual(len(res.buttons), 2)
        self.assertEqual(res.buttons[0]["id"], "btn_latest")
        print("[OK] Interactive quick-reply buttons test passed.")

    def test_interactive_button_webhook_receiver(self):
        """Test receiving interactive button tap payloads from Meta WhatsApp Webhook."""
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123",
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "15551234567",
                            "type": "interactive",
                            "interactive": {
                                "type": "button_reply",
                                "button_reply": {
                                    "id": "btn_latest",
                                    "title": "📰 Latest News"
                                }
                            }
                        }]
                    }
                }]
            }]
        }
        response = self.client.post("/webhook", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["intent"], "LATEST_NEWS")
        print("[OK] Interactive button webhook receiver test passed.")

    def test_subscription_rest_apis(self):
        """Test REST endpoints POST /api/subscriptions and GET /api/subscriptions."""
        payload = {"user_phone": "+1999888777", "topic": "Finance & Business", "digest_time": "08:00"}
        res = self.client.post("/api/subscriptions", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["topic"], "Finance & Business")

        get_res = self.client.get("/api/subscriptions", params={"user_phone": "+1999888777"})
        self.assertEqual(get_res.status_code, 200)
        get_data = get_res.json()
        self.assertEqual(len(get_data), 1)
        self.assertEqual(get_data[0]["topic"], "Finance & Business")
        print("[OK] Subscription REST API test passed.")

    def test_send_daily_digests_callback(self):
        """Test execution of daily scheduled digest dispatcher."""
        send_daily_digests()
        print("[OK] Daily digest callback test passed.")

if __name__ == "__main__":
    unittest.main()
