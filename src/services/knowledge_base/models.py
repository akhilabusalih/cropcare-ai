from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelMapping(BaseModel):
    cnn_class: str
    aliases: List[str]

class DiseaseRecord(BaseModel):
    metadata: Dict[str, Any]
    disease_id: str
    disease_name: str
    common_name: str
    scientific_name: str
    crop: str
    disease_type: str
    pathogen_type: str
    model_mapping: ModelMapping
    overview: str
    symptoms: List[str]
    causes: List[str]
    infection_cycle: List[str]
    transmission: List[str]
    risk_factors: List[str]
    environmental_conditions: List[str]
    weather_thresholds: Dict[str, Any]
    weather_influence: str
    severity_levels: Dict[str, Any]
    severity_score: int
    immediate_actions: List[str]
    treatment: List[str]
    prevention: List[str]
    nutrient_management: List[str]
    disease_progression: str
    recovery_indicators: List[str]
    recovery: str
    economic_impact: str
    monitoring: List[str]
    faq: List[Dict[str, str]]
    educational_information: str
    ai_context: Dict[str, Any]
    prompt_templates: Dict[str, str]
    references: List[str]

class PredictionData(BaseModel):
    disease: str
    confidence: float
    status: str
    error: Optional[str] = None

class AdvisoryContext(BaseModel):
    """
    Unified payload passed to the Advisory Agent.
    """
    disease_data: Dict[str, Any] = Field(default_factory=dict)
    weather_data: Dict[str, Any] = Field(default_factory=dict)
    severity_data: Dict[str, Any] = Field(default_factory=dict)
    risk_data: Dict[str, Any] = Field(default_factory=dict)
    knowledge_context: Optional[DiseaseRecord] = None
