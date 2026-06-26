import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List

from src.utils.logger import get_logger
from .exceptions import KnowledgeBaseNotFound, InvalidKnowledgeBase

logger = get_logger("knowledge_base", "knowledge_base.log")

class KnowledgeBaseLoader:
    """Handles all file I/O for the knowledge base."""

    def __init__(self, root_dir: str = None):
        if root_dir is None:
            self.root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'knowledge_base')))
        else:
            self.root_dir = Path(root_dir)
            
        self.config_path = self.root_dir / "config" / "knowledge_base_config.yaml"

    def load_config(self) -> Dict[str, Any]:
        """Loads the global knowledge base configuration."""
        if not self.config_path.exists():
            logger.error(f"Config file not found at {self.config_path}")
            raise KnowledgeBaseNotFound(f"Config file not found at {self.config_path}")
            
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise InvalidKnowledgeBase(f"Failed to load config: {e}")

    def load_crop_data(self, crop_name: str) -> List[Dict[str, Any]]:
        """Loads raw JSON data for a specific crop."""
        filename = f"{crop_name.lower().replace(' ', '_')}.json"
        file_path = self.root_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Crop file not found: {file_path}")
            return []
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise InvalidKnowledgeBase(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise InvalidKnowledgeBase(f"Failed to load {file_path}: {e}")
