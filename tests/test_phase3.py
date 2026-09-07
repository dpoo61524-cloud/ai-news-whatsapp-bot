import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from app.config import settings
from app.agents.conversational import ConversationalAgent

class TestPhase3(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.agent = ConversationalAgent()

    def test_whatsapp_webhook_verification_success(self):
        """Test successful Meta WhatsApp Webhook verification handshake."""
        url = f"/webhook?hub.mode=subscribe&hub.verify_token={settings.WHATSAPP_VERIFY_TOKEN}&hub.challenge=987654321"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "987654321")
        print("[OK] Webhook verification success test passed.")

    def test_whatsapp_webhook_verification_invalid_token(self):
        """Test failed Webhook verification handshake with invalid token."""
        url = "/webhook?hub.mode=subscribe&hub.verify_token=invalid_token_123&hub.challenge=987654321"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)
        print("[OK] Webhook invalid token rejection test passed.")

    def test_conversational_agent_intent_classification(self):
        """Test classification of user prompts into appropriate intents."""
        self.assertEqual(self.agent.classify_intent("Help me use the bot"), "HELP")
        self.assertEqual(self.agent.classify_intent("Show me the latest news headlines"), "LATEST_NEWS")
        self.assertEqual(self.agent.classify_intent("Give me AI and technology news"), "CATEGORY_SEARCH")
        self.assertEqual(self.agent.classify_intent("Refresh news feeds"), "INGEST_REFRESH")
        self.assertEqual(self.agent.classify_intent("Search for quantum computing advances"), "KEYWORD_SEARCH")
        print("[OK] Intent classification test passed.")

    def test_api_chat_endpoint(self):
        """Test interactive chat REST API endpoint /api/chat."""
        payload = {"message": "Help", "sender_id": "+1234567890"}
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["intent"], "HELP")
        self.assertIn("AI News Agent Assistance", data["reply_text"])
        print("[OK] Interactive chat REST API test passed.")

    def test_whatsapp_webhook_message_receiver(self):
        """Test incoming WhatsApp JSON payload processing via /webhook."""
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123",
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "15551234567",
                            "type": "text",
                            "text": {"body": "latest news"}
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
        print("[OK] WhatsApp webhook message receiver test passed.")

if __name__ == "__main__":
    unittest.main()
