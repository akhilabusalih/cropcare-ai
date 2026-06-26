import time
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger
from .exceptions import DiseaseNotFound, KnowledgeBaseError
from .models import DiseaseRecord
from .kb_loader import KnowledgeBaseLoader
from .kb_cache import KnowledgeBaseCache
from .kb_validator import KnowledgeBaseValidator

logger = get_logger("knowledge_base", "knowledge_base.log")

class KnowledgeBaseManager:
    """
    The single interface for the application to access the Disease Knowledge Base.
    Coordinates loading, caching, and validating disease information.
    """
    def __init__(self, validate_on_startup: bool = True):
        self.loader = KnowledgeBaseLoader()
        self.cache = KnowledgeBaseCache()
        self.validator = KnowledgeBaseValidator()
        self.config = {}
        
        self.initialize(validate_on_startup)

    def initialize(self, validate_on_startup: bool = True):
        """Loads configuration and optionally warms up the cache."""
        try:
            logger.info("Initializing KnowledgeBaseManager")
            t0 = time.time()
            
            self.config = self.loader.load_config()
            self._warm_cache(validate_on_startup)
            
            t1 = time.time()
            logger.info(f"KnowledgeBaseManager initialized in {(t1 - t0) * 1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Base Manager: {e}")
            # Do not fail application startup entirely, but log the error

    def _warm_cache(self, validate: bool):
        """Loads all supported crops into the cache to build the disease index."""
        supported_crops = self.config.get("supported_crops", [])
        for crop in supported_crops:
            self._load_and_cache_crop(crop, validate)

    def _load_and_cache_crop(self, crop_name: str, validate: bool = False) -> List[DiseaseRecord]:
        """Loads, parses, validates, and caches a specific crop."""
        raw_data = self.loader.load_crop_data(crop_name)
        if not raw_data:
            return []
            
        if validate:
            try:
                self.validator.validate_crop_data(crop_name, raw_data)
            except Exception as e:
                logger.error(f"Validation error for {crop_name}: {e}")
                
        # Parse into Pydantic models
        records = []
        for record_dict in raw_data:
            try:
                record = DiseaseRecord(**record_dict)
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to parse DiseaseRecord in {crop_name}: {e}")
                
        # Cache the parsed models (builds the O(1) index)
        self.cache.set_crop(crop_name, records)
        return records

    def get_disease(self, cnn_class: str) -> DiseaseRecord:
        """
        Retrieves a disease record by its cnn_class.
        Utilizes the O(1) cache index.
        """
        record = self.cache.get_disease(cnn_class)
        if record:
            return record
            
        # If not found in cache, it might be a lazy-loaded crop or invalid
        # To be safe, we can try to extract the crop name from the cnn_class
        crop_name = cnn_class.split('___')[0] if '___' in cnn_class else cnn_class
        # Load the specific crop
        self._load_and_cache_crop(crop_name)
        
        # Check again
        record = self.cache.get_disease(cnn_class)
        if not record:
            raise DiseaseNotFound(f"Disease with cnn_class '{cnn_class}' not found in the Knowledge Base.")
            
        return record

    def get_crop(self, crop_name: str) -> List[DiseaseRecord]:
        """Retrieves all disease records for a specific crop."""
        records = self.cache.get_crop(crop_name)
        if records is not None:
            return records
            
        return self._load_and_cache_crop(crop_name)

    def list_supported_crops(self) -> List[str]:
        """Returns a list of supported crops defined in the config."""
        return self.config.get("supported_crops", [])

    def reload(self):
        """Clears the cache and reloads the knowledge base."""
        logger.info("Reloading Knowledge Base")
        self.cache.clear()
        self.initialize(validate_on_startup=True)

    def clear_cache(self):
        """Clears the knowledge base cache."""
        logger.info("Clearing Knowledge Base cache")
        self.cache.clear()
