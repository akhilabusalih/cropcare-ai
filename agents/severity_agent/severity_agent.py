from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("severity_agent", "severity_agent.log")

class SeverityAgent:
    def __init__(self):
        """
        Initializes the Severity Agent.
        Currently uses deterministic rule-based logic.
        """
        self.assessment_method = "rule_based_v1"

    def analyze_severity(self, image_path: str, disease_class: str, confidence: float) -> Dict[str, Any]:
        """
        Analyzes the severity of the disease based on the prediction confidence.
        Future implementations can utilize `image_path` for CV-based severity estimation.
        """
        
        # 1. Healthy Class Handling
        if "healthy" in disease_class.lower():
            return {
                "severity": "None",
                "severity_score": 0,
                "details": "No disease detected.",
                "assessment_method": self.assessment_method
            }
            
        # 2. Rule-Based Logic
        if confidence >= 90.0:
            severity = "High"
            score = 3
            details = "High confidence indicates significant feature presence, mapping to high severity."
        elif confidence >= 70.0:
            severity = "Medium"
            score = 2
            details = "Moderate confidence indicates intermediate symptom presentation."
        else:
            severity = "Low"
            score = 1
            details = "Low confidence implies early-stage or ambiguous symptoms."
            
        result = {
            "severity": severity,
            "severity_score": score,
            "details": details,
            "assessment_method": self.assessment_method
        }
        
        logger.info(f"Severity analysis for {disease_class} (Conf: {confidence}%): {severity} (Score: {score})")
        return result
