from agents.disease_detection_agent.disease_detection_agent import DiseaseDetectionAgent
from agents.severity_agent.severity_agent import SeverityAgent
from agents.advisory_agent.advisory_agent import AdvisoryAgent
from agents.weather_agent.weather_agent import WeatherAgent
from agents.environmental_risk_agent.environmental_risk_agent import EnvironmentalRiskAgent
from src.services.knowledge_base.models import AdvisoryContext, DiseaseRecord

# Instantiate legacy agents to be reused across tools
_disease_agent = DiseaseDetectionAgent()
_severity_agent = SeverityAgent()
_advisory_agent = AdvisoryAgent()
_weather_agent = WeatherAgent()
_risk_agent = EnvironmentalRiskAgent()

def detect_disease_tool(image_path: str) -> dict:
    """
    Detects crop disease from an image.
    
    Args:
        image_path (str): The absolute path to the crop image file.
        
    Returns:
        dict: A dictionary containing 'disease', 'confidence', and 'confidence_level'.
    """
    try:
        return _disease_agent.predict(image_path)
    except Exception as e:
        return {"error": str(e)}

def analyze_severity_tool(image_path: str, disease_class: str, confidence: float) -> dict:
    """
    Analyzes the severity of a detected crop disease.
    
    Args:
        image_path (str): The absolute path to the crop image file.
        disease_class (str): The name of the detected disease (e.g. 'Tomato___Late_blight').
        confidence (float): The confidence percentage of the disease prediction.
        
    Returns:
        dict: A dictionary containing 'severity', 'severity_score', 'details', and 'assessment_method'.
    """
    try:
        return _severity_agent.analyze_severity(
            image_path=image_path,
            disease_class=disease_class,
            confidence=confidence
        )
    except Exception as e:
        return {"error": str(e)}

def generate_advice_tool(disease_data: dict, weather_data: dict, severity_data: dict, risk_data: dict, knowledge_context: dict) -> dict:
    """
    Generates agricultural advice based on disease, severity, weather, environmental risk, and knowledge context.
    
    Args:
        disease_data (dict): Prediction data.
        weather_data (dict): Current weather data.
        severity_data (dict): Severity assessment.
        risk_data (dict): Risk assessment.
        knowledge_context (dict): Disease knowledge base context.
        
    Returns:
        dict: Agricultural recommendations.
    """
    try:
        # Convert dictionary state back to Pydantic model for AdvisoryAgent
        context = AdvisoryContext(
            disease_data=disease_data,
            weather_data=weather_data,
            severity_data=severity_data,
            risk_data=risk_data,
            knowledge_context=DiseaseRecord(**knowledge_context) if knowledge_context else None
        )
        return _advisory_agent.generate_advice(context=context)
    except Exception as e:
        return {"error": str(e)}

def get_weather_tool(location_input: str = None, lat: float = None, lon: float = None) -> dict:
    """
    Retrieves weather data based on location string OR coordinates.
    """
    try:
        return _weather_agent.get_weather(location_input=location_input, lat=lat, lon=lon)
    except Exception as e:
        return {"error": str(e)}

def assess_risk_tool(disease_class: str, weather_data: dict) -> dict:
    """
    Calculates environmental disease risk based on weather data.
    """
    try:
        return _risk_agent.assess_risk(disease_class, weather_data)
    except Exception as e:
        return {"error": str(e)}
