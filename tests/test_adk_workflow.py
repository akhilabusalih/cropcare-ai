import os
import sys
import unittest
from unittest.mock import patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.adk_workflow.adk_coordinator import ADKCoordinatorAgent

class TestADKCoordinatorAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ADKCoordinatorAgent()

    @patch('os.getenv', return_value="dummy_key")
    @patch('agents.adk_workflow.adk_coordinator.ADK_AVAILABLE', True)
    @patch('agents.adk_workflow.adk_coordinator.detect_disease_tool')
    @patch('agents.adk_workflow.adk_coordinator.analyze_severity_tool')
    @patch('google.adk.runners.Runner.run')
    @patch('agents.coordinator_agent.coordinator_agent.os.path.exists', return_value=True)
    def test_full_adk_success(self, mock_exists, mock_runner_run, mock_severity, mock_disease, mock_env):
        mock_disease.return_value = {"disease": "Tomato___Late_blight", "confidence": 95.0}
        mock_severity.return_value = {"severity": "High"}
        
        class MockEvent:
            def __init__(self):
                self.content = None
                self.output = '{"treatment": ["Test Treatment"]}'
                
        mock_runner_run.return_value = [MockEvent()]

        result = self.agent.process_image("dummy.jpg")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["workflow_engine"], "google_adk")
        self.assertEqual(result["agent_trace"], ["DiseaseAgent", "SeverityAgent", "AdvisoryAgent"])
        self.assertEqual(result["prediction"]["disease"], "Tomato___Late_blight")
        
    @patch('os.getenv', return_value=None)
    @patch('agents.coordinator_agent.coordinator_agent.CoordinatorAgent.process_image')
    def test_fallback_due_to_missing_key(self, mock_legacy, mock_env):
        mock_legacy.return_value = {
            "status": "success", 
            "warnings": [], 
            "workflow_engine": "legacy_coordinator",
            "agent_trace": ["DiseaseDetectionAgent", "SeverityAgent", "AdvisoryAgent"]
        }
        
        result = self.agent.process_image("dummy.jpg")
        
        self.assertTrue(mock_legacy.called)
        self.assertEqual(result["workflow_engine"], "legacy_coordinator")
        self.assertIn("ADK Workflow initialization failed (Missing GEMINI_API_KEY)", result["warnings"][0])

if __name__ == "__main__":
    unittest.main()
