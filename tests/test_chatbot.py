import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestChatbotConfiguration(unittest.TestCase):
    @patch('agents.advisory_agent.advisory_agent.GENAI_AVAILABLE', True)
    @patch('agents.advisory_agent.advisory_agent.genai')
    @patch('os.getenv', return_value="FAKE_KEY")
    def test_advisory_agent_client_reuse(self, mock_getenv, mock_genai):
        """Test that the chatbot can instantiate AdvisoryAgent and reuse the genai.Client."""
        from agents.advisory_agent.advisory_agent import AdvisoryAgent
        
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        agent = AdvisoryAgent()
        
        self.assertIsNotNone(agent.client)
        self.assertEqual(agent.client, mock_client)
        mock_genai.Client.assert_called_once_with(api_key="FAKE_KEY")

    def test_compact_knowledge_summary_extraction(self):
        """Test that the compact knowledge base dictionary is structured properly."""
        kb_context = {
            "disease_id": "TOM_EB_001",
            "overview": {"description": "A fungal disease affecting tomato leaves."},
            "symptoms": {"early": ["small brown spots"]},
            "treatment": {"organic": ["neem oil"]},
            "prevention": {"farm_hygiene": ["clean tools"]},
            "immediate_actions": {"today": ["remove affected leaves"]},
            "weather_influence": {"high_humidity": "promotes spore generation"}
        }
        
        overview = kb_context.get("overview", {})
        overview_desc = overview.get("description", "") if isinstance(overview, dict) else str(overview)
        
        kb_summary = {
            "overview": overview_desc,
            "symptoms": kb_context.get("symptoms", {}),
            "treatment": kb_context.get("treatment", {}),
            "prevention": kb_context.get("prevention", {}),
            "immediate_actions": kb_context.get("immediate_actions", {}),
            "weather_influence": kb_context.get("weather_influence", {})
        }
        
        self.assertEqual(kb_summary["overview"], "A fungal disease affecting tomato leaves.")
        self.assertIn("early", kb_summary["symptoms"])
        self.assertIn("organic", kb_summary["treatment"])
        self.assertIn("high_humidity", kb_summary["weather_influence"])

if __name__ == '__main__':
    unittest.main()
