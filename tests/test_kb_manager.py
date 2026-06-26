import pytest
import os
from unittest.mock import patch, MagicMock

from src.services.knowledge_base.kb_manager import KnowledgeBaseManager
from src.services.knowledge_base.exceptions import DiseaseNotFound, InvalidKnowledgeBase
from src.services.knowledge_base.models import DiseaseRecord

@pytest.fixture
def mock_config():
    return {
        "supported_crops": ["Tomato", "Potato"]
    }

@pytest.fixture
def mock_tomato_data():
    return [
        {
            "metadata": {"review_status": "approved"},
            "disease_id": "TOM-001",
            "disease_name": "Tomato Late Blight",
            "common_name": "Late Blight",
            "scientific_name": "Phytophthora infestans",
            "crop": "Tomato",
            "disease_type": "Fungal",
            "pathogen_type": "Oomycete",
            "model_mapping": {
                "cnn_class": "Tomato___Late_blight",
                "aliases": ["late blight"]
            },
            "overview": "A devastating disease of tomatoes.",
            "symptoms": ["Dark lesions on leaves"],
            "causes": ["High humidity", "Cool temperatures"],
            "infection_cycle": ["Spores germinate in water"],
            "transmission": ["Wind", "Water splash"],
            "risk_factors": ["Poor air circulation"],
            "environmental_conditions": ["Cool and wet"],
            "weather_thresholds": {"humidity": "> 90%"},
            "weather_influence": "Thrives in cool, wet weather.",
            "severity_levels": {"High": "Complete defoliation"},
            "severity_score": 9,
            "immediate_actions": ["Apply fungicide"],
            "treatment": ["Copper-based fungicides"],
            "prevention": ["Crop rotation", "Resistant varieties"],
            "nutrient_management": ["Avoid excess nitrogen"],
            "disease_progression": "Rapid",
            "recovery_indicators": ["New healthy growth"],
            "recovery": "Possible if caught early.",
            "economic_impact": "Severe",
            "monitoring": ["Check lower leaves frequently"],
            "faq": [{"question": "Is it safe to eat?", "answer": "Yes, if unaffected."}],
            "educational_information": "First identified in 1840s.",
            "ai_context": {"summary": "Late blight"},
            "prompt_templates": {"default": "Advise on late blight"},
            "references": ["Ag Ext 101"]
        }
    ]

class TestKnowledgeBaseManager:
    @patch("src.services.knowledge_base.kb_loader.KnowledgeBaseLoader.load_config")
    @patch("src.services.knowledge_base.kb_loader.KnowledgeBaseLoader.load_crop_data")
    @patch("src.services.knowledge_base.kb_validator.KnowledgeBaseValidator.validate_crop_data")
    def test_initialization_and_cache_warmup(self, mock_validate, mock_load_data, mock_load_config, mock_config, mock_tomato_data):
        mock_load_config.return_value = mock_config
        mock_load_data.side_effect = lambda crop: mock_tomato_data if crop == "Tomato" else []
        
        manager = KnowledgeBaseManager(validate_on_startup=True)
        
        assert manager.config == mock_config
        # Should have built the disease index
        assert manager.cache.get_disease("Tomato___Late_blight") is not None
        assert manager.cache.get_disease("Tomato___Late_blight").disease_id == "TOM-001"
        assert mock_validate.called

    @patch("src.services.knowledge_base.kb_loader.KnowledgeBaseLoader.load_config")
    @patch("src.services.knowledge_base.kb_loader.KnowledgeBaseLoader.load_crop_data")
    def test_get_disease_not_found(self, mock_load_data, mock_load_config, mock_config):
        mock_load_config.return_value = mock_config
        mock_load_data.return_value = []
        
        manager = KnowledgeBaseManager(validate_on_startup=False)
        
        with pytest.raises(DiseaseNotFound):
            manager.get_disease("Unknown___Disease")

    @patch("src.services.knowledge_base.kb_loader.KnowledgeBaseLoader.load_config")
    @patch("src.services.knowledge_base.kb_loader.KnowledgeBaseLoader.load_crop_data")
    def test_get_crop(self, mock_load_data, mock_load_config, mock_config, mock_tomato_data):
        mock_load_config.return_value = mock_config
        mock_load_data.return_value = mock_tomato_data
        
        manager = KnowledgeBaseManager(validate_on_startup=False)
        crop_data = manager.get_crop("Tomato")
        
        assert len(crop_data) == 1
        assert isinstance(crop_data[0], DiseaseRecord)
        assert crop_data[0].disease_id == "TOM-001"
