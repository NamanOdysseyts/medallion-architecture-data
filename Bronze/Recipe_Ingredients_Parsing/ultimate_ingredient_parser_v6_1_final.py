"""
Ultimate Ingredient Parser v6.2
--------------------------------
Transforms free-form ingredient strings into a flat dict of
{normalized_ingredient: grams_or_scalar_string}.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

Number = Union[float, str]


class UltimateIngredientParser:
    """Normalize ingredient text for Supabase jsonb ingestion."""

    MASS_UNITS = {
        "gram": 1.0,
        "grams": 1.0,
        "g": 1.0,
        "kilogram": 1000.0,
        "kilograms": 1000.0,
        "kg": 1000.0,
        "ounce": 28.3495,
        "ounces": 28.3495,
        "oz": 28.3495,
        "pound": 453.592,
        "pounds": 453.592,
        "lb": 453.592,
        "lbs": 453.592,
        "milligram": 0.001,
        "milligrams": 0.001,
        "mg": 0.001,
        "stick": 113.0,
        "sticks": 113.0,
    }

    GENERIC_VOLUME_TO_GRAMS = {
        "cup": 240.0,
        "cups": 240.0,
        "tablespoon": 15.0,
        "tablespoons": 15.0,
        "tbsp": 15.0,
        "teaspoon": 5.0,
        "teaspoons": 5.0,
        "tsp": 5.0,
        "fluid ounce": 29.57,
        "fluid ounces": 29.57,
        "fl oz": 29.57,
        "pint": 473.0,
        "pints": 473.0,
        "quart": 946.0,
        "quarts": 946.0,
        "liter": 1000.0,
        "liters": 1000.0,
        "milliliter": 1.0,
        "milliliters": 1.0,
        "ml": 1.0,
        "gallon": 3785.0,
        "gallons": 3785.0,
    }

    COUNT_UNITS = {
        "clove",
        "cloves",
        "piece",
        "pieces",
        "slice",
        "slices",
        "can",
        "cans",
        "jar",
        "jars",
        "package",
        "packages",
        "pkg",
        "egg",
        "eggs",
        "head",
        "heads",
        "fillet",
        "fillets",
        "sprig",
        "sprigs",
        "bunch",
        "bunches",
        "handful",
        "handfuls",
    }

    INGREDIENT_DENSITY = {
        "all-purpose flour": {"cup": 120.0},
        "granulated sugar": {"cup": 200.0},
        "brown sugar": {"cup": 210.0},
        "powdered sugar": {"cup": 120.0},
        "olive oil": {"tablespoon": 13.5, "teaspoon": 4.5},
        "butter": {"tablespoon": 14.2},
        "water": {"cup": 240.0},
        "milk": {"cup": 244.0},
        "diced tomato": {"cup": 180.0},
        "walnut": {"cup": 117.0},
    }

    INGREDIENT_ALIASES = {
        "white sugar": "granulated sugar",
        "caster sugar": "granulated sugar",
        "sea salt": "salt",
        "kosher salt": "salt",
        "all purpose flour": "all-purpose flour",
        "plain flour": "all-purpose flour",
        "garlic cloves": "garlic",
        "garlic clove": "garlic",
        "garlic": "garlic",
        "extra virgin olive oil": "olive oil",
        "virgin olive oil": "olive oil",
    }

    STOPWORDS = {
        "fresh",
        "freshly",
        "finely",
        "roughly",
        "chopped",
        "diced",
        "minced",
        "sliced",
        "ground",
        "optional",
        "about",
        "approximately",
        "plus",
        "divided",
        "packed",
        "unsalted",
        "salted",
        "to",
        "taste",
        "room",
        "temperature",
        "softened",
        "melted",
        "for",
        "serving",
        "garnish",
        "peeled",
        "seeded",
        "advertisement",
    }

    OPTIONAL_PHRASES = [
        "to taste",
        "as needed",
        "as desired",
        "for serving",
        "for garnish",
    ]

    SPECIAL_SECTION_MARKERS = (
        "special equipment",
        "equipment:",
        "decoration",
        "decorations",
        "assembly",
        "notes:",
        "note:",
    )

    FRACTIONS = {
        "¼": 0.25,
        "½": 0.5,
        "¾": 0.75,
        "⅓": 1 / 3,
        "⅔": 2 / 3,
        "⅛": 0.125,
        "⅜": 0.375,
        "⅝": 0.625,
        "⅞": 0.875,
    }

    WORD_NUMBERS = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "dozen": 12,
        "half": 0.5,
        "quarter": 0.25,
        "third": 1 / 3,
        "couple": 2,
        "few": 3,
        "several": 3,
    }

    PAREN_PATTERN = re.compile(
        r"\((?P<amount>[\d\/\.\s¼½¾⅓⅔⅛⅜⅝⅞]+)\s*(?P<unit>[a-zA-Z ]+)\)",
        re.IGNORECASE,
    )

    def __init__(self, llm_client=None):
        unit_keys = (
            set(self.MASS_UNITS.keys())
            | set(self.GENERIC_VOLUME_TO_GRAMS.keys())
            | self.COUNT_UNITS
        )
        self._unit_regex = re.compile(
            r"^(?:" + "|".join(sorted(unit_keys, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )
        self.llm_client = llm_client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse_and_validate(
        self, ingredients_input
    ) -> Tuple[Dict[str, Number], List[str]]:
        items = self._coerce_to_list(ingredients_input)
        parsed: Dict[str, Number] = {}
        warnings: List[str] = []
        unresolved: List[str] = []

        for item in items:
            result = self.parse_single_ingredient(item)
            if not result:
                unresolved.append(item)
                continue
            name, value = result
            if isinstance(value, str):
                unresolved.append(value)
            else:
                parsed[name] = self._merge_value(parsed.get(name), value)

        llm_results = self._llm_parse(unresolved)
        for name, value in llm_results.items():
            parsed[name] = self._merge_value(parsed.get(name), value)

        if not parsed:
            warnings.append("No ingredients parsed")

        return parsed, warnings

    def parse_single_ingredient(self, raw_text: str) -> Optional[Tuple[str, Number]]:
        if not raw_text:
            return None

        line = raw_text.strip().strip('"\'' )
        if not line or self._line_is_noise(line):
            return None

        lower = line.lower()
        for phrase in self.OPTIONAL_PHRASES:
            lower = lower.replace(phrase, "")
        line = re.sub(r"\s+", " ", lower).strip()
        if not line:
            line = raw_text.strip()

        parenthetical_mass, trimmed = self._extract_parenthetical_mass(line)
        quantity, unit, remainder = self._extract_quantity_and_unit(trimmed)
        name = self._clean_name(remainder or trimmed or line)
        if not name:
            return None

        value = self._calculate_value(quantity, unit, name, parenthetical_mass, raw_text.strip())
        return name, value

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _coerce_to_list(self, ingredients_input) -> List[str]:
        if ingredients_input is None:
            return []

        if isinstance(ingredients_input, list):
            return [str(item).strip() for item in ingredients_input if str(item).strip()]

        if isinstance(ingredients_input, dict):
            lines = []
            for name, qty in ingredients_input.items():
                if qty is None:
                    lines.append(str(name))
                else:
                    lines.append(f"{qty} {name}")
            return [line.strip() for line in lines if line.strip()]

        if isinstance(ingredients_input, str):
            text = self._strip_sections(ingredients_input.replace("\r", "\n"))
            text = text.replace("•", "\n")
            candidates = [seg.strip() for seg in re.split(r"[\n;]", text) if seg.strip()]
            if len(candidates) == 1:
                split_numbers = [
                    seg.strip(" ,")
                    for seg in re.split(r"(?=\d+\s)", text)
                    if seg.strip(" ,")
                ]
                if len(split_numbers) > 1:
                    candidates = split_numbers

            filtered: List[str] = []
            for candidate in candidates:
                c = candidate.strip(' "\'')
                if not c or self._line_is_noise(c):
                    continue
                filtered.append(c)
            return filtered

        return [str(ingredients_input).strip()]

    def _strip_sections(self, text: str) -> str:
        lowered = text.lower()
        cutoff = len(text)
        for marker in self.SPECIAL_SECTION_MARKERS:
            idx = lowered.find(marker)
            if idx != -1:
                cutoff = min(cutoff, idx)
        return text[:cutoff]

    def _line_is_noise(self, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return True
        if lowered == "advertisement":
            return True
        if lowered.startswith("special equipment"):
            return True
        if lowered.startswith("recipe "):
            return True
        return False

    def _extract_parenthetical_mass(self, text: str) -> Tuple[Optional[float], str]:
        match = self.PAREN_PATTERN.search(text)
        if not match:
            return None, text
        amount = self._fraction_to_float(match.group("amount").strip())
        unit = match.group("unit").strip().lower()
        cleaned = (text[: match.start()] + " " + text[match.end() :]).strip()
        if amount is None:
            return None, cleaned
        grams = self._convert_units(amount, unit)
        return (round(grams, 2) if grams is not None else None, cleaned)

    def _extract_quantity_and_unit(self, text: str) -> Tuple[Optional[float], Optional[str], str]:
        working = text.strip()
        if not working:
            return None, None, ""

        quantity = None
        # unicode fraction at start
        if working[0] in self.FRACTIONS:
            quantity = self.FRACTIONS[working[0]]
            working = working[1:].strip()
        else:
            match = re.match(
                r"^(\d+\s\d/\d|\d+/\d+|\d+(?:\.\d+)?|"
                + "|".join(sorted(self.WORD_NUMBERS.keys(), key=len, reverse=True))
                + r")\b",
                working,
                flags=re.IGNORECASE,
            )
            if match:
                quantity = self._fraction_to_float(match.group(0))
                working = working[match.end() :].strip()

        unit = None
        if working:
            unit_match = self._unit_regex.match(working)
            if unit_match:
                unit = unit_match.group(0).lower()
                working = working[unit_match.end() :].strip()

        working = re.sub(r"^\s*of\s+", "", working, flags=re.IGNORECASE)
        return quantity, unit, working.strip(",;:- ")

    def _fraction_to_float(self, token: str) -> Optional[float]:
        if token is None:
            return None
        token = token.strip().lower()
        if not token:
            return None
        if token in self.FRACTIONS:
            return self.FRACTIONS[token]
        if token in self.WORD_NUMBERS:
            return float(self.WORD_NUMBERS[token])
        if re.match(r"^\d+\s\d/\d$", token):
            whole, frac = token.split()
            numer, denom = frac.split("/")
            return float(int(whole) + int(numer) / int(denom))
        if re.match(r"^\d+/\d+$", token):
            numer, denom = token.split("/")
            return float(int(numer) / int(denom))
        try:
            return float(token)
        except ValueError:
            return None

    def _clean_name(self, text: str) -> str:
        cleaned = text.lower()
        cleaned = re.sub(r"\([^)]*\)", "", cleaned)
        cleaned = cleaned.replace("-", " ")
        cleaned = re.sub(r"[,.;:]", " ", cleaned)

        tokens: List[str] = []
        for token in cleaned.split():
            if token in self.STOPWORDS:
                continue
            if token.isdigit():
                continue
            tokens.append(token)

        normalized = " ".join(tokens).strip()
        return self.INGREDIENT_ALIASES.get(normalized, normalized)

    def _calculate_value(
        self,
        quantity: Optional[float],
        unit: Optional[str],
        normalized_name: str,
        fallback: Optional[float],
        raw: str,
    ) -> Number:
        if fallback is not None:
            return round(fallback, 2)
        if quantity is None:
            return raw
        if unit is None:
            return round(quantity, 2)

        unit_lower = unit.lower()
        density = self.INGREDIENT_DENSITY.get(normalized_name)
        if density and unit_lower in density:
            return round(quantity * density[unit_lower], 2)

        grams = self._convert_units(quantity, unit_lower)
        if grams is not None:
            return round(grams, 2)

        logger.debug("Unknown unit '%s', treating as count", unit_lower)
        return round(quantity, 2)

    def _convert_units(self, quantity: float, unit: str) -> Optional[float]:
        if unit in self.MASS_UNITS:
            return quantity * self.MASS_UNITS[unit]
        if unit in self.COUNT_UNITS:
            return quantity
        if unit in self.GENERIC_VOLUME_TO_GRAMS:
            return quantity * self.GENERIC_VOLUME_TO_GRAMS[unit]
        return None

    @staticmethod
    def _merge_value(existing: Optional[Number], new: Number) -> Number:
        if existing is None:
            return new
        if isinstance(existing, (int, float)) and isinstance(new, (int, float)):
            return round(float(existing) + float(new), 2)
        return new

    # ------------------------------------------------------------------ #
    # LLM fallback
    # ------------------------------------------------------------------ #
    def _llm_parse(self, unresolved: List[str]) -> Dict[str, Number]:
        if not self.llm_client or not unresolved:
            return {}

        prompt = f"""
You normalize ingredient lines into a JSON dictionary mapping normalized ingredient names
to numeric quantities (grams preferred). Follow these rules:
1. Keys must be lowercase singular ingredient names (e.g., "brown sugar").
2. Values must be either:
   - a number representing grams, or
   - an object {{"count": <number>, "unit": "<unit>"}} when grams are unclear.
3. Convert using typical kitchen equivalents:
   * 1 cup all-purpose flour = 120 g
   * 1 cup granulated sugar = 200 g
   * 1 cup brown sugar = 210 g
   * 1 tablespoon olive oil = 13.5 g
   * 1 tablespoon butter = 14.2 g
   * 1 cup water = 240 g
   * 1 cup milk = 244 g
4. For ranges, use the upper bound.
5. Never include commentary; output JSON only.

Ingredient lines:
{json.dumps(unresolved, ensure_ascii=False, indent=2)}
"""
        try:
            response = self.llm_client.generate(prompt, max_tokens=800)
            data = self._extract_json_dict(response)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LLM fallback failed: %s", exc)
            return {}

        normalized: Dict[str, Number] = {}
        for key, value in data.items():
            if key and isinstance(value, (int, float)):
                normalized[key.strip().lower()] = float(value)
            elif isinstance(value, dict):
                if "grams" in value and isinstance(value["grams"], (int, float)):
                    normalized[key.strip().lower()] = float(value["grams"])
                elif "count" in value:
                    normalized[key.strip().lower()] = {
                        "count": value.get("count"),
                        "unit": value.get("unit", "unit"),
                    }
            elif isinstance(value, str):
                normalized[key.strip().lower()] = value.strip()
        return normalized

    @staticmethod
    def _extract_json_dict(text: str) -> Dict[str, Union[int, float, Dict]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM response.")
        snippet = text[start : end + 1]
        data = json.loads(snippet)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object.")
        return data


if __name__ == "__main__":
    parser = UltimateIngredientParser()
    sample = """
    ADVERTISEMENT
    2 cups all-purpose flour (240g)
    1/2 teaspoon ground cinnamon
    1 cup water ADVERTISEMENT
    """
    result, warns = parser.parse_and_validate(sample)
    print(result)
    if warns:
        print("Warnings:", warns)

