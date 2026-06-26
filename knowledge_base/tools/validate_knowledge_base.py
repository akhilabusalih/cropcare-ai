"""Validate the CropGuardian knowledge base dataset.

This script checks JSON validity, schema compliance, uniqueness constraints,
and basic scientific completeness signals. It never modifies any dataset file.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "disease_schema.json"
CONFIG_PATH = ROOT / "config" / "knowledge_base_config.yaml"
CROP_FILES = [
    "apple.json",
    "bell_pepper.json",
    "cherry.json",
    "corn.json",
    "grape.json",
    "orange.json",
    "peach.json",
    "potato.json",
    "squash.json",
    "strawberry.json",
    "tomato.json",
]
EXPECTED_METADATA_KEYS = {"review_status", "last_scientific_review", "content_quality", "difficulty_level"}
REQUIRED_FIELDS = {
    "metadata",
    "disease_id",
    "disease_name",
    "common_name",
    "scientific_name",
    "crop",
    "disease_type",
    "pathogen_type",
    "model_mapping",
    "overview",
    "symptoms",
    "causes",
    "infection_cycle",
    "transmission",
    "risk_factors",
    "environmental_conditions",
    "weather_thresholds",
    "weather_influence",
    "severity_levels",
    "severity_score",
    "immediate_actions",
    "treatment",
    "prevention",
    "nutrient_management",
    "disease_progression",
    "recovery_indicators",
    "recovery",
    "economic_impact",
    "monitoring",
    "faq",
    "educational_information",
    "ai_context",
    "prompt_templates",
    "references",
}


class ValidationError(Exception):
    pass


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_cnn_class(crop: str, label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return f"{crop}___{slug}"


def validate_record(record: dict, file_name: str, index: int) -> list[str]:
    issues: list[str] = []

    missing = sorted(REQUIRED_FIELDS.difference(record.keys()))
    if missing:
        issues.append(f"missing required fields: {', '.join(missing)}")

    unexpected = sorted(set(record.keys()).difference(REQUIRED_FIELDS))
    if unexpected:
        issues.append(f"unexpected fields present: {', '.join(unexpected)}")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("metadata must be an object")
    else:
        missing_meta = sorted(EXPECTED_METADATA_KEYS.difference(metadata.keys()))
        if missing_meta:
            issues.append(f"missing metadata fields: {', '.join(missing_meta)}")
        extra_meta = sorted(set(metadata.keys()).difference(EXPECTED_METADATA_KEYS))
        if extra_meta:
            issues.append(f"unexpected metadata fields: {', '.join(extra_meta)}")

    mapping = record.get("model_mapping")
    if not isinstance(mapping, dict):
        issues.append("model_mapping must be an object")
    else:
        cnn_class = mapping.get("cnn_class")
        aliases = mapping.get("aliases")
        if not isinstance(cnn_class, str) or not cnn_class.strip():
            issues.append("model_mapping.cnn_class must be a non-empty string")
        if not isinstance(aliases, list) or not aliases:
            issues.append("model_mapping.aliases must be a non-empty list")
        elif any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            issues.append("model_mapping.aliases contains an empty or non-string value")
        elif len(set(aliases)) != len(aliases):
            issues.append("model_mapping.aliases contains duplicates")

    for field in ["scientific_name", "pathogen_type", "treatment", "prevention", "weather_thresholds", "ai_context", "references"]:
        if field not in record:
            issues.append(f"missing scientific completeness field: {field}")

    references = record.get("references")
    if isinstance(references, list):
        if len(set(references)) != len(references):
            issues.append("duplicate references found")
        if any(not isinstance(item, str) or not item.strip() for item in references):
            issues.append("references contains an empty or non-string value")

    return issues


def main() -> int:
    errors: list[str] = []

    if not SCHEMA_PATH.exists():
        errors.append(f"missing schema file: {SCHEMA_PATH}")
    if not CONFIG_PATH.exists():
        errors.append(f"missing config file: {CONFIG_PATH}")

    try:
        load_json(SCHEMA_PATH)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"schema is not valid JSON: {exc}")

    disease_ids: list[str] = []
    cnn_classes: list[str] = []
    aliases: list[str] = []
    crop_file_counts: Counter[str] = Counter()

    for name in CROP_FILES:
        file_path = ROOT / name
        if not file_path.exists():
            errors.append(f"missing crop file: {name}")
            continue

        try:
            records = load_json(file_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid JSON in {name}: {exc}")
            continue

        if not isinstance(records, list):
            errors.append(f"{name} must contain a top-level array")
            continue

        crop_file_counts[name] = len(records)

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"{name}[{index}] is not an object")
                continue

            errors.extend(f"{name}[{index}] {issue}" for issue in validate_record(record, name, index))

            disease_id = record.get("disease_id")
            if isinstance(disease_id, str):
                disease_ids.append(disease_id)

            model_mapping = record.get("model_mapping")
            if isinstance(model_mapping, dict):
                cnn_class = model_mapping.get("cnn_class")
                if isinstance(cnn_class, str):
                    cnn_classes.append(cnn_class)
                mapping_aliases = model_mapping.get("aliases")
                if isinstance(mapping_aliases, list):
                    aliases.extend(mapping_aliases)

    duplicate_ids = [item for item, count in Counter(disease_ids).items() if count > 1]
    duplicate_classes = [item for item, count in Counter(cnn_classes).items() if count > 1]
    duplicate_aliases = [item for item, count in Counter(aliases).items() if count > 1]

    for item in duplicate_ids:
        errors.append(f"duplicate disease_id: {item}")
    for item in duplicate_classes:
        errors.append(f"duplicate cnn_class: {item}")
    for item in duplicate_aliases:
        errors.append(f"duplicate alias: {item}")

    expected_counts = {
        "apple.json": 4,
        "bell_pepper.json": 4,
        "cherry.json": 3,
        "corn.json": 4,
        "grape.json": 4,
        "orange.json": 4,
        "peach.json": 4,
        "potato.json": 4,
        "squash.json": 3,
        "strawberry.json": 3,
        "tomato.json": 9,
    }
    for name, expected in expected_counts.items():
        if crop_file_counts.get(name) not in (0, expected):
            errors.append(f"{name} has unexpected record count: {crop_file_counts[name]} (expected {expected})")

    if errors:
        for error in errors:
            print(error)
        return 1

    print("Knowledge base validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
