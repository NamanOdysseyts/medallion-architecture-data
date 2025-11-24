"""
Recipe ingestion pipeline v3.
Rewritten so that ingredient payloads are always flat dicts
({ingredient: number_or_string}) ready for Supabase jsonb columns.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from langdetect import detect
from supabase import Client, create_client

from ultimate_ingredient_parser_v6_1_final import UltimateIngredientParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LLM Client / Translation
# --------------------------------------------------------------------------- #
class LiteLLMClient:
    """Minimal chat-completion client used for translations."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        if not self.api_key:
            return ""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LiteLLM request failed: %s", exc)
            return ""


class LanguageProcessor:
    """Handles language detection and translation."""

    def __init__(self, llm_client: LiteLLMClient):
        self.llm = llm_client

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return detect(text) if text and text.strip() else "en"
        except Exception:
            return "en"

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text

        lang = self.detect_language(text)
        if lang == "en":
            return text

        logger.info("Translating from %s", lang)
        prompt = (
            "Translate the following recipe content to English. "
            "Return English text only.\n"
            f"{text}"
        )
        translated = self.llm.generate(prompt, max_tokens=600)
        return translated or text


def normalize_instructions(instructions) -> List[str]:
    """Return instructions as ['1. Step'] list."""
    if instructions is None:
        return []

    if isinstance(instructions, list):
        lines = [str(step).strip() for step in instructions if str(step).strip()]
    else:
        raw = str(instructions).replace("\r", "\n")
        lines = [seg.strip() for seg in re.split(r"[\n.]", raw) if seg.strip()]

    normalized = []
    for idx, step in enumerate(lines, start=1):
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", step).strip()
        if cleaned:
            normalized.append(f"{idx}. {cleaned}")
    return normalized


# --------------------------------------------------------------------------- #
# Config / Data access
# --------------------------------------------------------------------------- #
@dataclass
class RecipeIngestConfig:
    SUPABASE_URL: str = os.getenv("PROJECT_URL", "")
    SUPABASE_KEY: str = os.getenv("PROJECT_API_KEY", "")
    RECIPES_TABLE: str = "recipes"
    LLM_API_KEY: str = os.getenv("LLM_API_KEY") or os.getenv("LITELLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://litellm.confer.today")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-mini")
    TRANSLATE_TO_ENGLISH: bool = True
    MIN_INGREDIENTS: int = 1
    MIN_INSTRUCTION_CHARS: int = 20
    OPTIONAL_STRING_FIELDS: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "description": ["description", "summary"],
            "notes": ["notes", "note"],
            "image_url": ["image_url", "image"],
            "source_url": ["source_url", "url"],
            "source": ["source", "publisher"],
            "difficulty": ["difficulty"],
            "cuisine": ["cuisine", "cuisines"],
        }
    )
    NUMERIC_FIELDS: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "servings": ["servings", "yield"],
            "calories": ["calories"],
            "protein_g": ["protein", "protein_g"],
            "carbs_g": ["carbs", "carbohydrates", "carbs_g"],
            "fat_g": ["fat", "fat_g"],
            "fiber_g": ["fiber", "fiber_g"],
            "sugar_g": ["sugar", "sugars"],
            "sodium_mg": ["sodium", "sodium_mg"],
            "prep_time_minutes": ["prep_time", "prep_time_minutes"],
            "cook_time_minutes": ["cook_time", "cook_time_minutes"],
            "total_time_minutes": ["total_time", "total_time_minutes"],
        }
    )


class DataLoader:
    @staticmethod
    def load_data(path: str, source_type: str = "auto") -> Tuple[pd.DataFrame, str]:
        if source_type == "auto":
            if path.endswith(".csv"):
                source_type = "csv"
            elif path.endswith(".json"):
                source_type = "json"
            else:
                raise ValueError(f"Cannot infer file type for {path}")

        if source_type == "csv":
            df = pd.read_csv(path)
        elif source_type == "json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                df = pd.DataFrame(list(data.values()))
            else:
                df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        return df, os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- #
# Validation / Transformation
# --------------------------------------------------------------------------- #
class RecipeValidator:
    REQUIRED_FIELDS = ("title", "instructions", "ingredients")

    def __init__(self, config: RecipeIngestConfig):
        self.config = config

    def validate(self, row: Dict[str, Any], row_index: int) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for field in self.REQUIRED_FIELDS:
            value = self._get_field_value(row, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"Missing {field}")
                continue
            if field == "instructions" and isinstance(value, str):
                if len(value.strip()) < self.config.MIN_INSTRUCTION_CHARS:
                    errors.append("Instructions too short")
            if field == "ingredients":
                if isinstance(value, list) and len(value) < self.config.MIN_INGREDIENTS:
                    errors.append("Not enough ingredients")
        return (len(errors) == 0, errors)

    @staticmethod
    def _get_field_value(row: Dict[str, Any], field: str):
        if field in row:
            return row[field]
        for key, value in row.items():
            if key.lower() == field.lower():
                return value
        aliases = {
            "title": ["name", "recipe_title", "recipe_name"],
            "instructions": ["instructions", "steps", "directions", "method"],
            "ingredients": ["ingredients", "ingredient_list", "items", "ingredientLines"],
        }
        for alias in aliases.get(field, []):
            for key, value in row.items():
                if key.lower() == alias.lower():
                    return value
        return None


class RecipeTransformer:
    def __init__(
        self,
        config: RecipeIngestConfig,
        language_processor: LanguageProcessor,
        ingredient_parser: UltimateIngredientParser,
    ):
        self.config = config
        self.language_processor = language_processor
        self.parser = ingredient_parser

    def transform(self, row: Dict[str, Any], row_index: int, source_id: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        warnings: List[str] = []

        title = RecipeValidator._get_field_value(row, "title") or ""
        title = str(title).strip()

        instructions_raw = RecipeValidator._get_field_value(row, "instructions")
        instructions_text = self._stringify(instructions_raw)

        ingredients_raw = RecipeValidator._get_field_value(row, "ingredients")

        if self.config.TRANSLATE_TO_ENGLISH:
            if title:
                translated_title = self.language_processor.translate(title)
                if translated_title != title:
                    warnings.append("Translated title")
                title = translated_title
            if instructions_text:
                translated_instructions = self.language_processor.translate(instructions_text)
                if translated_instructions != instructions_text:
                    warnings.append("Translated instructions")
                instructions_text = translated_instructions
            ingredients_raw = self._translate_ingredients_if_needed(ingredients_raw, warnings)

        instructions = normalize_instructions(instructions_text)
        parsed_ingredients, parser_warnings = self.parser.parse_and_validate(ingredients_raw)
        warnings.extend(parser_warnings)

        if not parsed_ingredients:
            warnings.append("Skipping recipe - ingredients empty after parsing")
            return None, warnings

        transformed: Dict[str, Any] = {
            "title": title or "Untitled recipe",
            "instructions": instructions,
            "ingredients": parsed_ingredients,
            "_source_id": source_id,
            "_source_row_index": row_index,
            "_ingestion_timestamp": datetime.utcnow().isoformat(),
            "status": "active",
            "meal_type": row.get("meal_type") or "dinner",
            "market_country": row.get("market_country") or "US",
        }

        self._map_optional_strings(row, transformed)
        self._map_optional_numbers(row, transformed)
        self._ensure_total_time(transformed)

        return transformed, warnings

    def _translate_ingredients_if_needed(self, ingredients_raw, warnings: List[str]):
        if isinstance(ingredients_raw, str):
            translated = self.language_processor.translate(ingredients_raw)
            if translated != ingredients_raw:
                warnings.append("Translated ingredient block")
            return translated
        if isinstance(ingredients_raw, list):
            joined = "\n".join(str(item) for item in ingredients_raw)
            translated = self.language_processor.translate(joined)
            if translated != joined:
                warnings.append("Translated ingredient list")
            return translated.split("\n")
        return ingredients_raw

    def _map_optional_strings(self, row: Dict[str, Any], transformed: Dict[str, Any]):
        for output_field, aliases in self.config.OPTIONAL_STRING_FIELDS.items():
            value = self._first_non_empty(row, aliases)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                continue
            if self.config.TRANSLATE_TO_ENGLISH and output_field in {"description", "notes"}:
                translated = self.language_processor.translate(text_value)
                text_value = translated or text_value
            transformed[output_field] = text_value

    def _map_optional_numbers(self, row: Dict[str, Any], transformed: Dict[str, Any]):
        for output_field, aliases in self.config.NUMERIC_FIELDS.items():
            value = self._first_non_empty(row, aliases)
            if value is None:
                continue
            numeric = self._coerce_number(value)
            if numeric is not None:
                transformed[output_field] = numeric

    @staticmethod
    def _ensure_total_time(payload: Dict[str, Any]):
        prep = payload.get("prep_time_minutes")
        cook = payload.get("cook_time_minutes")
        total = payload.get("total_time_minutes")

        if total is None:
            if isinstance(prep, (int, float)) and isinstance(cook, (int, float)):
                payload["total_time_minutes"] = int(prep + cook)
            elif isinstance(prep, (int, float)):
                payload["total_time_minutes"] = int(prep)
            elif isinstance(cook, (int, float)):
                payload["total_time_minutes"] = int(cook)

        if "time_minutes" not in payload and payload.get("total_time_minutes"):
            payload["time_minutes"] = payload["total_time_minutes"]

    @staticmethod
    def _stringify(value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(step) for step in value)
        return str(value)

    @staticmethod
    def _first_non_empty(row: Dict[str, Any], keys: List[str]):
        for key in keys:
            if key in row and row[key] not in (None, ""):
                return row[key]
            for candidate, value in row.items():
                if candidate.lower() == key.lower() and value not in (None, ""):
                    return value
        return None

    @staticmethod
    def _coerce_number(value) -> Optional[int]:
        text = str(value)
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match.group(0))
        if abs(number - round(number)) < 1e-6:
            return int(round(number))
        return int(number)


# --------------------------------------------------------------------------- #
# Database insertion
# --------------------------------------------------------------------------- #
class RecipeInserter:
    def __init__(self, client: Client, config: RecipeIngestConfig):
        self.client = client
        self.config = config

    def ensure_table(self):
        try:
            self.client.table(self.config.RECIPES_TABLE).select("*").limit(1).execute()
        except Exception as exc:
            logger.warning("Could not verify table '%s': %s", self.config.RECIPES_TABLE, exc)

    def insert(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            payload = record.copy()
            self.client.table(self.config.RECIPES_TABLE).insert(payload).execute()
            return True, f"Inserted '{record['title']}'"
        except Exception as exc:
            return False, str(exc)


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #
class RecipeIngestionPipeline:
    def __init__(self, config: RecipeIngestConfig):
        self.config = config
        self.llm_client = LiteLLMClient(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            model=config.LLM_MODEL,
        )
        self.language_processor = LanguageProcessor(self.llm_client)
        self.parser = UltimateIngredientParser(llm_client=self.llm_client)
        self.validator = RecipeValidator(config)
        self.transformer = RecipeTransformer(config, self.language_processor, self.parser)
        self.client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.inserter = RecipeInserter(self.client, config)

    def run(self, path: str, source_type: str = "auto"):
        df, source_id = DataLoader.load_data(path, source_type)
        logger.info("Loaded %s rows from %s", len(df), source_id)

        self.inserter.ensure_table()

        valid_records: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            is_valid, errors = self.validator.validate(row_dict, idx)
            if not is_valid:
                logger.warning("Row %s skipped: %s", idx + 1, "; ".join(errors))
                continue
            transformed, warnings = self.transformer.transform(row_dict, idx, source_id)
            if not transformed:
                logger.warning("Row %s skipped during transform: %s", idx + 1, "; ".join(warnings))
                continue
            if warnings:
                logger.info("Row %s warnings: %s", idx + 1, "; ".join(warnings))
            valid_records.append(transformed)

        inserted = 0
        for record in valid_records:
            success, message = self.inserter.insert(record)
            if success:
                inserted += 1
                logger.info(message)
            else:
                logger.error("Insert failed: %s", message)

        logger.info("Finished. %s/%s recipes inserted.", inserted, len(valid_records))


def main():
    config = RecipeIngestConfig()
    pipeline = RecipeIngestionPipeline(config)
    pipeline.run("recipes.json", source_type="json")


if __name__ == "__main__":
    main()
