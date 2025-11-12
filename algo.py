"""
Core recommendation logic utilities.

This module centralizes logic for:
- Building search queries from answered questions
- Creating embeddings and searching documents
- Calculating relevance scores with cosine similarity
- Extracting embeddings/similarity from search results
- Finding similar books based on title/author/category

Design goals:
- Pure functions (no Streamlit/UI dependencies)
- Parameterized clients (pass in Supabase and GenAI clients)
- Clear docstrings so the logic is easy to understand
"""

from typing import Any, Dict, List, Optional, Tuple

from supabase import Client as SupabaseClient
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

# Google Genai configuration
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GENAI_API_KEY:
    st.warning("Missing env vars SUPABASE_URL, SUPABASE_ANON_KEY/KEY, or GOOGLE_API_KEY.")


# Default behaviors (can be overridden by callers)
RELEVANCE_FALLBACK_SCORE = 0.2
RELEVANCE_HYBRID_ALPHA = 1.0  # reserved for future hybridization


def create_embedding(client: genai.Client, text: str, *, output_dim: int = 1536) -> Optional[List[float]]:
    """Create an embedding vector for the provided text using Gemini embeddings.

    Args:
        client: Initialized `genai.Client`.
        text: Text content to embed.
        output_dim: Dimensionality of the output vector.

    Returns:
        A list of floats representing the embedding, or None if creation fails.
    """
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=output_dim,
            ),
        )
        return list(result.embeddings[0].values)
    except Exception:
        return None


def _safe_norm(vec: List[float]) -> float:
    try:
        return sum((float(x) ** 2 for x in vec)) ** 0.5
    except Exception:
        return 0.0


def calculate_relevance_score(query_embedding: List[float], item_embedding: List[float]) -> float:
    """Cosine similarity between two vectors, normalized to [0,1].

    - If vectors are invalid or zero-norm, returns `RELEVANCE_FALLBACK_SCORE`.
    - Cosine in [-1, 1] mapped to [0,1] via (cos + 1) / 2.
    """
    if not isinstance(query_embedding, (list, tuple)) or not isinstance(item_embedding, (list, tuple)):
        return RELEVANCE_FALLBACK_SCORE
    if not query_embedding or not item_embedding:
        return RELEVANCE_FALLBACK_SCORE

    try:
        q = [float(x) for x in query_embedding]
        d = [float(x) for x in item_embedding]
        n = min(len(q), len(d))
        if n == 0:
            return RELEVANCE_FALLBACK_SCORE
        dot = sum(q[i] * d[i] for i in range(n))
        qn = _safe_norm(q[:n])
        dn = _safe_norm(d[:n])
        if qn == 0.0 or dn == 0.0:
            return RELEVANCE_FALLBACK_SCORE
        cos = dot / (qn * dn)
        cos = max(-1.0, min(1.0, cos))
        return (cos + 1.0) / 2.0
    except Exception:
        return RELEVANCE_FALLBACK_SCORE


def compute_relevance_score(user_vec: List[float], item_vec: List[float]) -> float:
    """Alias for `calculate_relevance_score` kept for convenience."""
    return calculate_relevance_score(user_vec, item_vec)


def extract_item_embedding_from_result(result: Dict[str, Any]) -> Optional[List[float]]:
    """Try to extract an item embedding from a search result dict.

    Checks common locations:
    - result['embedding'] or result['embeddings']
    - result['metadata']['embedding'] or ['embeddings']
    Accepts list/tuple directly or a string that can be parsed via `ast.literal_eval`.
    """
    if not isinstance(result, dict):
        return None

    try:
        if "embedding" in result:
            emb = result["embedding"]
            if isinstance(emb, (list, tuple)):
                return list(emb)
            if isinstance(emb, str):
                import ast
                parsed = ast.literal_eval(emb)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)

        if "embeddings" in result:
            emb = result["embeddings"]
            if isinstance(emb, (list, tuple)):
                return list(emb)
            if isinstance(emb, str):
                import ast
                parsed = ast.literal_eval(emb)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)

        md = result.get("metadata") or {}
        if isinstance(md, dict):
            for key in ("embedding", "embeddings"):
                emb = md.get(key)
                if isinstance(emb, (list, tuple)):
                    return list(emb)
                if isinstance(emb, str):
                    import ast
                    parsed = ast.literal_eval(emb)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
    except Exception:
        return None

    return None


def extract_similarity_from_result(result: Dict[str, Any]) -> Optional[float]:
    """Extract backend-provided similarity if present.

    If value is in [-1,1], map to [0,1]. Otherwise clamp to [0,1].
    """
    if not isinstance(result, dict):
        return None
    for key in ("similarity", "score"):
        if key in result:
            try:
                val = float(result[key])
                if -1.0 <= val <= 1.0:
                    val = (val + 1.0) / 2.0
                if val < 0.0:
                    val = 0.0
                if val > 1.0:
                    val = 1.0
                return val
            except Exception:
                return None
    return None


def fetch_embedding_from_supabase(supabase: SupabaseClient, summary_id: str) -> Optional[List[float]]:
    """Fetch an embedding vector from Supabase for a given summary entity id."""
    try:
        resp = (
            supabase.table("recommendations")
            .select("embeddings")
            .eq("entity_id", summary_id)
            .eq("entity_type", "summary")
            .execute()
        )
        if getattr(resp, "data", None):
            embeddings_str = resp.data[0].get("embeddings")
            if embeddings_str:
                if isinstance(embeddings_str, (list, tuple)):
                    return list(embeddings_str)
                if isinstance(embeddings_str, str):
                    import ast
                    parsed = ast.literal_eval(embeddings_str)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
    except Exception:
        return None
    return None


def search_documents(
    supabase: SupabaseClient,
    query_embedding: List[float],
    query_text: str,
    *,
    match_count: int = 10,
) -> List[Dict[str, Any]]:
    """Call the `hybrid_search` RPC and return raw results list."""
    try:
        resp = supabase.rpc(
            "hybrid_search",
            {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "match_count": match_count,
            },
        ).execute()
        data = resp.data if getattr(resp, "data", None) else []
        return data if data else []
    except Exception:
        return []


def score_and_sort_results(
    query_embedding: List[float],
    results: List[Dict[str, Any]],
    supabase: Optional[SupabaseClient] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Attach `score` to each result and return sorted list with metrics.

    - Prefers backend similarity if present.
    - Falls back to cosine similarity using item embeddings.
    - If embeddings are missing in results, optionally fetch from Supabase.
    Returns (scored_results, metrics_dict).
    """
    supabase_similarity_count = 0
    cosine_count = 0
    fallback_count = 0
    fetched_count = 0

    scored: List[Dict[str, Any]] = []
    for r in results or []:
        sim = extract_similarity_from_result(r)
        if sim is None:
            item_vec = extract_item_embedding_from_result(r)
            if not item_vec and supabase is not None:
                md = r.get("metadata", {})
                summary_id = md.get("id") or r.get("id")
                if summary_id:
                    item_vec = fetch_embedding_from_supabase(supabase, summary_id)
                    if item_vec:
                        fetched_count += 1
            if item_vec:
                sim = calculate_relevance_score(query_embedding, item_vec)
                cosine_count += 1
            else:
                sim = RELEVANCE_FALLBACK_SCORE
                fallback_count += 1
        else:
            supabase_similarity_count += 1

        r["score"] = float(sim)
        scored.append(r)

    scored = sorted(scored, key=lambda x: x.get("score", RELEVANCE_FALLBACK_SCORE), reverse=True)
    metrics = {
        "supabase": supabase_similarity_count,
        "cosine": cosine_count,
        "fetched": fetched_count,
        "fallback": fallback_count,
        "total": len(scored),
    }
    return scored, metrics


# --------- Question parsing and query building ---------

def _safe_int(value: Any, default: int = 999) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_responses(
    questions: List[Dict[str, Any]],
    responses: List[Any],
) -> Tuple[Optional[str], bool, List[str], List[str]]:
    """Parse raw responses to extract structured signals and readable texts.

    Returns: (age_group, has_kids, motivation_keys, response_texts)
    """
    age_group: Optional[str] = None
    has_kids = False
    motivation_keys: List[str] = []
    response_texts: List[str] = []

    AGE_QUESTION_ID = "36d3b953-8f26-410f-8c64-729e778d8766"
    KIDS_QUESTION_ID = "e81709b2-e9aa-42d6-b843-4e62cfc18f99"
    MOTIVATION_QUESTION_ID = "d3349488-6ad2-465f-a8c0-653d4b9002f5"

    for idx, response in enumerate(responses or []):
        if idx >= len(questions):
            break
        q = questions[idx]
        qid = q.get("id", "")
        qtext = (q.get("question", "") or "").lower()
        options_json = q.get("options", "{}")
        try:
            import json
            options = json.loads(options_json) if options_json else {}
        except Exception:
            options = {}

        def _resolve_option_text(sel: Any) -> Optional[str]:
            key = str(sel)
            return options.get(key)

        if qid == AGE_QUESTION_ID or "age group" in qtext:
            key = str(response[0]) if isinstance(response, list) and response else str(response)
            age_group = options.get(key)

        if qid == KIDS_QUESTION_ID or "kids" in qtext:
            key = str(response[0]) if isinstance(response, list) and response else str(response)
            text = (options.get(key) or "").lower()
            has_kids = text == "yes"

        if qid == MOTIVATION_QUESTION_ID or "motivates you" in qtext:
            if isinstance(response, list):
                motivation_keys = [str(x) for x in response]
            else:
                motivation_keys = [str(response)]

        if isinstance(response, list):
            for x in response:
                t = _resolve_option_text(x)
                if t:
                    response_texts.append(t)
        else:
            t = _resolve_option_text(response)
            if t:
                response_texts.append(t)

    return age_group, has_kids, motivation_keys, response_texts


def get_age_based_categories(age_group: Optional[str], has_kids: bool) -> str:
    """Map an age group and kids flag to category keywords.

    This is an example mapping; callers can replace or augment with domain data.
    """
    if not age_group:
        return ""
    g = (age_group or "").lower()
    categories: List[str] = []
    if "16" in g or "18" in g or "21" in g:
        categories.extend(["young adult", "career", "self-improvement"])
    elif "25" in g or "30" in g or "35" in g:
        categories.extend(["career growth", "productivity", "relationships"])
    elif "40" in g or "45" in g or "50" in g:
        categories.extend(["leadership", "family", "health"])
    else:
        categories.extend(["general", "popular science", "non-fiction"])

    if has_kids:
        categories.extend(["parenting", "family", "time management"])
    return " ".join(sorted(set(categories)))


def get_motivation_categories(motivation_keys: List[str], motivation_mapping: Dict[str, Any]) -> str:
    """Map selected motivations to category keywords using a provided mapping."""
    cats: List[str] = []
    for k in motivation_keys or []:
        if k in motivation_mapping:
            for c in motivation_mapping[k].get("categories", []):
                name = c.get("name")
                if name:
                    cats.append(name)
    return " ".join(sorted(set(cats)))


def build_search_query_from_responses(
    questions: List[Dict[str, Any]],
    responses: List[Any],
    motivation_mapping: Optional[Dict[str, Any]] = None,
) -> str:
    """Combine response texts and derived categories into a single query string."""
    age_group, has_kids, motivation_keys, response_texts = parse_responses(questions, responses)
    age_categories = get_age_based_categories(age_group, has_kids)
    motivation_categories = ""
    if motivation_keys and motivation_mapping:
        motivation_categories = get_motivation_categories(motivation_keys, motivation_mapping)

    parts: List[str] = []
    parts.extend(response_texts)
    if age_categories:
        parts.append(age_categories)
    if motivation_categories:
        parts.append(motivation_categories)
    return " ".join([p for p in parts if p]).strip()


def recommend_from_responses(
    genai_client: genai.Client,
    supabase: SupabaseClient,
    questions: List[Dict[str, Any]],
    responses: List[Any],
    motivation_mapping: Optional[Dict[str, Any]] = None,
    *,
    match_count: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """End-to-end: build query from responses, search, score, and sort results.

    Returns: (scored_results, metrics)
    """
    query = build_search_query_from_responses(questions, responses, motivation_mapping)
    if not query:
        return [], {"supabase": 0, "cosine": 0, "fetched": 0, "fallback": 0, "total": 0}

    embedding = create_embedding(genai_client, query)
    if embedding is None:
        return [], {"supabase": 0, "cosine": 0, "fetched": 0, "fallback": 0, "total": 0}

    raw = search_documents(supabase, embedding, query, match_count=match_count)
    scored, metrics = score_and_sort_results(embedding, raw, supabase)
    return scored, metrics


def find_similar_books(
    genai_client: genai.Client,
    supabase: SupabaseClient,
    title: str,
    author: str,
    category: str,
    *,
    match_count: int = 10,
    exclude_exact: bool = True,
    max_return: int = 9,
) -> List[Dict[str, Any]]:
    """Find books similar to the given title/author/category.

    - Creates an embedding from "title author category".
    - Searches via RPC and scores with backend similarity or cosine.
    - Optionally excludes the exact same book by title+author.
    - Returns up to `max_return` items.
    """
    query = f"{title} {author} {category}".strip()
    emb = create_embedding(genai_client, query)
    if emb is None:
        return []

    raw = search_documents(supabase, emb, query, match_count=match_count)
    scored, _ = score_and_sort_results(emb, raw, supabase)

    if exclude_exact:
        filtered: List[Dict[str, Any]] = []
        lt = (title or "").strip().lower()
        la = (author or "").strip().lower()
        for r in scored:
            md = r.get("metadata", {})
            rt = (md.get("title", "") or "").strip().lower()
            ra = (md.get("author", "") or "").strip().lower()
            if not (rt == lt and ra == la):
                filtered.append(r)
        scored = filtered

    return scored[:max_return]