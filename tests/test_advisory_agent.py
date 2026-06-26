import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.advisory_agent.advisory_agent import AdvisoryAgent

class TestAdvisoryAgent(unittest.TestCase):
    def setUp(self):
        # By default, setup agent without API key to test fallback
        pass

    @patch('os.getenv', return_value=None)
    def test_fallback_logic(self, mock_getenv):
        """Test the fallback logic when no API key is provided."""
        agent = AdvisoryAgent()
        result = agent.generate_advice("Tomato___Late_blight", "High")
        
        self.assertEqual(result["source"], "Fallback Knowledge Base")
        self.assertIn("treatment", result)
        self.assertIn("Consider destroying the heavily infected plant", result["treatment"][-1])

    def test_healthy_class(self):
        """Test that healthy class bypasses API entirely and returns positive advice."""
        agent = AdvisoryAgent()
        result = agent.generate_advice("Tomato___healthy", "None")
        
        self.assertEqual(result["source"], "Fallback Knowledge Base")
        self.assertEqual(result["treatment"], [])

    @patch('os.getenv', return_value="FAKE_KEY")
    @patch('agents.advisory_agent.advisory_agent.GENAI_AVAILABLE', True)
    @patch('agents.advisory_agent.advisory_agent.genai')
    def test_gemini_mocked_call(self, mock_genai, mock_getenv):
        """Test Gemini API call structure using mocks."""
        
        # Setup mock client
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "disease_description": "Mocked disease description.",
            "symptoms": ["Mocked symptom"],
            "treatment": ["Mocked treatment"],
            "prevention": ["Mocked prevention"],
            "fertilizer_recommendations": ["Mocked fert"],
            "weather_based_warnings": "Mocked warnings",
            "immediate_actions": "Mocked actions",
            "expert_consultation": "Mocked consultation",
            "source": "Gemini 2.5 Flash"
        })
        mock_client.models.generate_content.return_value = mock_response
        
        # Instantiate agent (it should pick up FAKE_KEY from mocked getenv)
        agent = AdvisoryAgent()
        
        # Execute
        result = agent.generate_advice("Tomato___Late_blight", "Medium")
        
        # Assertions
        self.assertEqual(result["source"], "Gemini 2.5 Flash")
        self.assertEqual(result["disease_description"], "Mocked disease description.")
        self.assertTrue(mock_client.models.generate_content.called)

if __name__ == "__main__":
    unittest.main()
