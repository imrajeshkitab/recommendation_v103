from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

AGE_QUESTION_ID = "36d3b953-8f26-410f-8c64-729e778d8766"
KIDS_QUESTION_ID = "e81709b2-e9aa-42d6-b843-4e62cfc18f99"
MOTIVATION_QUESTION_ID = "d3349488-6ad2-465f-a8c0-653d4b9002f5"


def get_age_based_categories(age_group: str, has_kids: bool) -> str:
    """Return categories associated with a user's age group and parenting status."""
    age_categories = {
        "16-25 years": "Productivity Personal Development Motivation Communication Career & Skills Creativity",
        "26-35 years": "Productivity Leadership Personal Development Relationships",
        "36-50 years": "Leadership Planning Health Mindfulness & Meditation Relationships Productivity Purpose & Values",
        "50+ years": "Health & Longevity Purpose & Values Relationships Spirituality & Philosophy Mindfulness & Meditation Nature & Wellness Creativity & Learning",
    }

    categories = age_categories.get(age_group, "")

    if has_kids:
        if "Parenting" not in categories:
            categories = f"{categories} Parenting".strip()
        if age_group == "50+ years" and "Grandparenting" not in categories:
            categories = f"{categories} Grandparenting".strip()

    return categories.strip()


def get_motivation_categories(motivation_keys: List[str], motivation_mapping: Dict[str, Any]) -> str:
    """Return category names associated with selected motivation keys."""
    categories_set = set()

    for key in motivation_keys:
        key_str = str(key)
        if key_str in motivation_mapping:
            motivation_data = motivation_mapping[key_str]
            for cat in motivation_data.get("categories", []):
                categories_set.add(cat.get("name", ""))

    return " ".join(sorted(filter(None, categories_set)))


def build_search_query(
    questions: List[Dict[str, Any]],
    user_responses: List[Any],
    motivation_mapping: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Construct the search query string and return metadata about the user context."""
    response_texts: List[str] = []
    age_group = None
    has_kids = False
    motivation_keys: List[str] = []

    for idx, response in enumerate(user_responses):
        if idx >= len(questions):
            break

        question = questions[idx]
        question_id = question.get("id", "")
        question_text = question.get("question", "").lower()
        options_json = question.get("options", "{}")

        try:
            options_dict = json.loads(options_json) if options_json else {}
        except Exception:  # pylint: disable=broad-except
            options_dict = {}

        # Helper to resolve response key for the current question
        if isinstance(response, list) and response:
            response_key = str(response[0])
        else:
            response_key = str(response) if response is not None else ""

        # Identify age group
        if question_id == AGE_QUESTION_ID or "age group" in question_text:
            if response_key in options_dict:
                age_group = options_dict[response_key]

        # Identify parenting status
        if question_id == KIDS_QUESTION_ID or "kids" in question_text:
            if response_key in options_dict:
                answer_text = str(options_dict[response_key]).lower()
                has_kids = answer_text == "yes"

        # Collect motivation keys
        if question_id == MOTIVATION_QUESTION_ID or "motivates you" in question_text:
            if isinstance(response, list):
                motivation_keys = [str(key) for key in response]
            elif response_key:
                motivation_keys = [response_key]

        # Collect human readable answers for search query
        if isinstance(response, list):
            for key in response:
                key_str = str(key)
                if key_str in options_dict:
                    response_texts.append(options_dict[key_str])
        else:
            if response_key in options_dict:
                response_texts.append(options_dict[response_key])

    age_categories = get_age_based_categories(age_group, has_kids) if age_group else ""
    motivation_categories = (
        get_motivation_categories(motivation_keys, motivation_mapping)
        if motivation_keys and motivation_mapping
        else ""
    )

    query_parts: List[str] = []
    if response_texts:
        query_parts.extend(response_texts)
    if age_categories:
        query_parts.append(age_categories)
    if motivation_categories:
        query_parts.append(motivation_categories)

    combined_query = " ".join(part.strip() for part in query_parts if part).strip()

    context = {
        "age_group": age_group,
        "has_kids": has_kids,
        "motivation_keys": motivation_keys,
        "response_texts": response_texts,
        "age_categories": age_categories,
        "motivation_categories": motivation_categories,
    }

    return combined_query, context