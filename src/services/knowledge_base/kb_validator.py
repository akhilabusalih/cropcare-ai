import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logger import get_logger
from .exceptions import ValidationError

logger = get_logger("knowledge_base", "knowledge_base.log")

# We dynamically import the existing validation script to reuse its logic
# without duplicating code.
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    from knowledge_base.tools.validate_knowledge_base import validate_record
except ImportError:
    validate_record = None
    logger.warning("Could not import existing validation script.")

class KnowledgeBaseValidator:
    """Provides runtime validation using existing validation logic."""
    
    @staticmethod
    def validate_crop_data(crop_name: str, records: List[Dict[str, Any]]):
        """Validates loaded crop data against the expected schema."""
        if not validate_record:
            logger.warning("Validation skipped because validator script is unavailable.")
            return
            
        errors = []
        for index, record in enumerate(records):
            issues = validate_record(record, f"{crop_name}.json", index)
            if issues:
                errors.extend([f"Record {index}: {issue}" for issue in issues])
                
        if errors:
            err_msg = f"Validation failed for {crop_name}.json: {errors}"
            logger.error(err_msg)
            raise ValidationError(err_msg)
