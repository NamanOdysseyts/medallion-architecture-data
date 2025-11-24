"""
Sample ingestion runner.
Loads a handful of recipes from the raw JSON drops and feeds them through
recipe_ingestion_v3.py so you can verify the Supabase payload locally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd

from recipe_ingestion_v3 import RecipeIngestConfig, RecipeIngestionPipeline


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "Raw_data" / "Raw_data"
SOURCE_FILES = [
    RAW_DIR / "recipes_raw_nosource_ar_first15.json",
    RAW_DIR / "recipes_raw_nosource_epi_first15.json",
    RAW_DIR / "recipes_raw_nosource_fn_first15.json",
]


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_first_n(path: Path, limit: int = 15) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for idx, (_, recipe) in enumerate(data.items()):
        if idx >= limit:
            break
        rows.append(recipe)
    return pd.DataFrame(rows)


def save_temp(df: pd.DataFrame, suffix: str) -> Path:
    temp_path = ROOT / f"temp_{suffix}.json"
    records = {f"recipe_{i}": rec for i, rec in enumerate(df.to_dict("records"))}
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return temp_path


def run_samples(files: List[Path]):
    load_env()
    config = RecipeIngestConfig(
        SUPABASE_URL=os.getenv("PROJECT_URL", ""),
        SUPABASE_KEY=os.getenv("PROJECT_API_KEY", ""),
        LLM_API_KEY=os.getenv("LLM_API_KEY") or os.getenv("LITELLM_API_KEY", ""),
        LLM_MODEL=os.getenv("LLM_MODEL", "gpt-5-mini"),
    )
    pipeline = RecipeIngestionPipeline(config)

    for path in files:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        print(f"\n=== Processing {path.name} ===")
        df = load_first_n(path, limit=15)
        temp_json = save_temp(df, path.stem)
        try:
            pipeline.run(str(temp_json), source_type="json")
        finally:
            if temp_json.exists():
                temp_json.unlink()


if __name__ == "__main__":
    run_samples(SOURCE_FILES)
