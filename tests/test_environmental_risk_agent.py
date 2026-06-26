import unittest
from agents.environmental_risk_agent.environmental_risk_agent import EnvironmentalRiskAgent

class TestEnvironmentalRiskAgent(unittest.TestCase):
    def setUp(self):
        self.agent = EnvironmentalRiskAgent()

    def test_high_fungal_risk(self):
        weather_data = {"weather": {"temperature": 26, "humidity": 90, "precipitation": 5}}
        result = self.agent.assess_risk("Tomato___Early_blight", weather_data)
        
        self.assertEqual(result["fungal_risk"], "Very High")
        self.assertEqual(result["overall_risk"], "High")
        self.assertTrue(result["spread_probability"] > 80)
        self.assertTrue(any("fungal spore" in reason for reason in result["reasons"]))

    def test_high_bacterial_risk(self):
        weather_data = {"weather": {"temperature": 30, "humidity": 82, "precipitation": 0}}
        result = self.agent.assess_risk("Tomato___Bacterial_spot", weather_data)
        
        self.assertEqual(result["bacterial_risk"], "High")
        self.assertEqual(result["heat_stress_risk"], "Medium")

    def test_heat_stress(self):
        weather_data = {"weather": {"temperature": 38, "humidity": 40, "precipitation": 0}}
        result = self.agent.assess_risk("Tomato___healthy", weather_data)
        
        self.assertEqual(result["heat_stress_risk"], "High")
        self.assertEqual(result["fungal_risk"], "Low")

    def test_benign_conditions(self):
        weather_data = {"weather": {"temperature": 22, "humidity": 50, "precipitation": 0}}
        result = self.agent.assess_risk("Tomato___healthy", weather_data)
        
        self.assertEqual(result["overall_risk"], "Low")
        self.assertTrue(result["spread_probability"] < 50)

    def test_missing_weather_data(self):
        result = self.agent.assess_risk("Tomato___Early_blight", {"status": "unavailable", "error": "API down"})
        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["risk_level"])
        self.assertIsNone(result["spread_probability"])
        self.assertEqual(result["reason"], "Weather service unavailable")

if __name__ == "__main__":
    unittest.main()
