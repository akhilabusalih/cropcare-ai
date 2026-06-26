import threading
from typing import Dict, List, Optional
from .models import DiseaseRecord

class KnowledgeBaseCache:
    """Thread-safe cache for parsed DiseaseRecord objects."""
    
    def __init__(self):
        self._lock = threading.RLock()
        # Stores lists of records keyed by crop name
        self._crops_cache: Dict[str, List[DiseaseRecord]] = {}
        # Stores single records keyed by cnn_class for O(1) lookups
        self._disease_index: Dict[str, DiseaseRecord] = {}

    def get_crop(self, crop_name: str) -> Optional[List[DiseaseRecord]]:
        with self._lock:
            return self._crops_cache.get(crop_name)

    def set_crop(self, crop_name: str, records: List[DiseaseRecord]):
        with self._lock:
            self._crops_cache[crop_name] = records
            for record in records:
                cnn_class = record.model_mapping.cnn_class
                self._disease_index[cnn_class] = record

    def get_disease(self, cnn_class: str) -> Optional[DiseaseRecord]:
        with self._lock:
            return self._disease_index.get(cnn_class)

    def clear(self):
        with self._lock:
            self._crops_cache.clear()
            self._disease_index.clear()
