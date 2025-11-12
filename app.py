import streamlit as st
from supabase import create_client, Client
from google import genai
from google.genai import types
import time
import html
import json
import base64
import csv
import logging
import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env using python-dotenv if available,
# otherwise fall back to a minimal loader.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    def _load_env_fallback():
        env_path = Path('.env')
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    _load_env_fallback()

# Read credentials from environment (populated by .env)
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
GENAI_API_KEY = os.getenv('GENAI_API_KEY', '')

from app_modules.query_builder import build_search_query
from app_modules.search_service import search_bytes, search_summaries

# Page configuration
st.set_page_config(
    page_title="Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for white and light purple theme
st.markdown("""
    <style>
    .main, .stApp {
        background-color: #F9FAFF;
    }
    .question-card {
        background: linear-gradient(135deg, #F8F4FF 0%, #F0ECFF 100%);
        padding: 2.2rem;
        border-radius: 20px;
        border: 1px solid rgba(124, 58, 237, 0.12);
        margin: 1.1rem 0;
        box-shadow: 0 22px 46px rgba(124, 58, 237, 0.12);
    }
    .card-inner {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.05rem 1.1rem 1.15rem;
        border: 1px solid rgba(124, 58, 237, 0.12);
        box-shadow: 0 14px 28px rgba(99, 102, 241, 0.12);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        width: 100%;
    }
    .card-inner:hover {
        transform: translateY(-4px);
        box-shadow: 0 26px 48px rgba(99, 102, 241, 0.18);
    }
    .result-card {
        background: linear-gradient(135deg, rgba(248, 244, 255, 0.9) 0%, rgba(226, 217, 255, 0.98) 100%);
        padding: 1.15rem 1.05rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.18);
        margin: 0.45rem auto 0;
        width: calc(100% - 0.3rem);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
        text-align: left;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .result-card:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 20px 36px rgba(124, 58, 237, 0.18);
        background: linear-gradient(135deg, rgba(248, 244, 255, 0.98) 0%, rgba(208, 196, 255, 1) 100%);
    }
    .detail-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F3F1FF 100%);
        padding: 1.6rem;
        border-radius: 22px;
        border: 1px solid rgba(124, 58, 237, 0.14);
        box-shadow: 0 28px 52px rgba(124, 58, 237, 0.15);
        margin: 0.6rem 0 1.1rem;
        width: 95%;
    }
    .description-box {
        background: #FFFFFF;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        padding: 1.2rem 1.5rem;
        box-shadow: 0 24px 40px rgba(15, 23, 42, 0.08);
        margin-top: 0.9rem;
        line-height: 1.55;
        color: #374151;
    }
    .book-title {
        color: #5B21B6 !important;
        margin-bottom: 0.35rem;
        font-size: 1rem !important;
        font-weight: 650;
        line-height: 1.22;
    }
    .book-author {
        color: #7C3AED !important;
        font-style: italic;
        margin-bottom: 0;
        font-size: 0.95rem !important;
        line-height: 1.18;
    }
    .meta-label {
        color: #4C1D95;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.01em;
    }
    .logo-container {
        float: left;
        margin-right: 1rem;
        margin-bottom: 0.5rem;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .progress-bar {
        background-color: rgba(124, 58, 237, 0.12);
        border-radius: 14px;
        padding: 0.6rem;
        margin: 1.1rem 0;
    }
    h1 {
        color: #6D28D9;
    }
    h2 {
        color: #5B21B6;
    }
    h3 {
        color: #4C1D95;
    }
    .stProgress .st-bo {
        background: linear-gradient(135deg, #8B5CF6, #6366F1);
        box-shadow: 0 6px 14px rgba(99, 102, 241, 0.25);
    }
    .stButton>button {
        background: linear-gradient(135deg, #8B5CF6, #6366F1);
        color: white;
        border-radius: 28px;
        border: none;
        padding: 0.55rem 1.8rem;
        font-weight: 600;
        box-shadow: 0 14px 28px rgba(99, 102, 241, 0.24);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #7C3AED, #4C1D95);
        box-shadow: 0 20px 36px rgba(76, 29, 149, 0.3);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Supabase configuration


# Relevance scoring configuration
# Toggle scoring/ranking, fallback score when missing data, and debug metrics rendering
RELEVANCE_SCORING_ENABLED = True
RELEVANCE_SHOW_DEBUG = False
RELEVANCE_FALLBACK_SCORE = 0.2  # Used when vectors/similarity are unavailable
RELEVANCE_HYBRID_ALPHA = 1.0    # Reserved for future hybridization

# Logger configuration (avoid duplicate handlers on Streamlit reruns)
logger = logging.getLogger("recommend_app")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Initialize session state
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "user_responses" not in st.session_state:
    st.session_state.user_responses = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "bytes_results" not in st.session_state:
    st.session_state.bytes_results = []
if "app_state" not in st.session_state:
    st.session_state.app_state = "questions"  # "questions", "results", or "similar_books"
if "selected_book" not in st.session_state:
    st.session_state.selected_book = None
if "similar_books_results" not in st.session_state:
    st.session_state.similar_books_results = []
if "selected_byte" not in st.session_state:
    st.session_state.selected_byte = None
if "similar_bytes_results" not in st.session_state:
    st.session_state.similar_bytes_results = []

def init_supabase_client():
    """Initialize and return Supabase client"""
    url = os.getenv('SUPABASE_URL', SUPABASE_URL)
    key = os.getenv('SUPABASE_KEY', SUPABASE_KEY)
    if not url or not key:
        st.error("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY in .env.")
        return None
    return create_client(url, key)

def init_genai_client():
    """Initialize and return Google Genai client"""
    api_key = os.getenv('GENAI_API_KEY', GENAI_API_KEY)
    if not api_key:
        st.error("Missing GENAI_API_KEY in .env.")
        return None
    return genai.Client(api_key=api_key)

@st.cache_data(ttl=3600)
def fetch_questions():
    """Load questions from CSV file"""
    try:
        csv_file = Path("seed_onboarding_questions_rows (2).csv")
        if not csv_file.exists():
            st.error("CSV file not found!")
            return []
        
        questions = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert sequence to int
                try:
                    sequence = int(row.get("sequence", 999))
                except (ValueError, TypeError):
                    sequence = 999
                
                question_data = {
                    "question": row.get("question", ""),
                    "type": row.get("type", "single_choice"),
                    "options": row.get("options", "{}"),
                    "images": row.get("images", "") or "",
                    "id": row.get("id", ""),
                    "sequence": sequence
                }
                questions.append(question_data)
        
        # Sort questions by sequence
        questions.sort(key=lambda x: x.get("sequence", 999))
        return questions
    except Exception as e:
        st.error(f"Error loading questions from CSV: {str(e)}")
        return []

def create_embedding(client: genai.Client, text: str):
    """Create embedding from text using Google Genai"""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=1536
            )
        )
        vec = list(result.embeddings[0].values)
        logger.info(
            "Created embedding: text_len=%d dim=%d",
            len(text or ""),
            len(vec or [])
        )
        return vec
    except Exception as e:
        logger.exception("Error creating embedding: %s", e)
        st.error(f"Error creating embedding: {str(e)}")
        return None

def _safe_norm(vec: list):
    try:
        return sum((float(x) ** 2 for x in vec)) ** 0.5
    except Exception:
        return 0.0

def calculate_relevance_score(query_embedding: list, item_embedding: list) -> float:
    """Calculate relevance score using cosine similarity between query and item embeddings.
    
    Args:
        query_embedding: The embedding vector of the search query
        item_embedding: The embedding vector of the item/document
        
    Returns:
        A relevance score between 0.0 and 1.0 (0% to 100% similarity)
        Returns RELEVANCE_FALLBACK_SCORE if calculation fails
    """
    if not isinstance(query_embedding, (list, tuple)) or not isinstance(item_embedding, (list, tuple)):
        logger.warning("Invalid embedding types for relevance calculation")
        return RELEVANCE_FALLBACK_SCORE
    
    if not query_embedding or not item_embedding:
        logger.warning("Empty embeddings for relevance calculation")
        return RELEVANCE_FALLBACK_SCORE
    
    try:
        # Convert to lists and ensure they're numeric
        query_vec = [float(x) for x in query_embedding]
        item_vec = [float(x) for x in item_embedding]
        
        # Guard against size mismatch by truncating to min length
        n = min(len(query_vec), len(item_vec))
        if n == 0:
            logger.warning("Zero-length embeddings after conversion")
            return RELEVANCE_FALLBACK_SCORE
        
        # Calculate dot product
        dot_product = sum(query_vec[i] * item_vec[i] for i in range(n))
        
        # Calculate norms
        query_norm = _safe_norm(query_vec[:n])
        item_norm = _safe_norm(item_vec[:n])
        
        if query_norm == 0.0 or item_norm == 0.0:
            logger.warning("Zero-norm vector in relevance calculation")
            return RELEVANCE_FALLBACK_SCORE
        
        # Calculate cosine similarity
        cosine_similarity = dot_product / (query_norm * item_norm)
        
        # Clip for numerical stability (cosine similarity should be in [-1, 1])
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        
        # Normalize to [0, 1] range: (cos + 1) / 2
        # This maps -1 -> 0, 0 -> 0.5, 1 -> 1.0
        normalized_score = (cosine_similarity + 1.0) / 2.0
        
        logger.debug(
            "Relevance score calculated: cosine=%.4f normalized=%.4f",
            cosine_similarity,
            normalized_score
        )
        
        return normalized_score
        
    except Exception as e:
        logger.exception("Error calculating relevance score: %s", e)
        return RELEVANCE_FALLBACK_SCORE

def compute_relevance_score(user_vec: list, item_vec: list, normalize: bool = True) -> float:
    """Compute cosine similarity between two vectors with numerical safety.

    Returns a score in [0,1] if normalize=True using (cos+1)/2.
    Falls back to RELEVANCE_FALLBACK_SCORE if vectors are invalid or zero-norm.
    
    Note: This function is kept for backward compatibility.
    Use calculate_relevance_score() for new code.
    """
    return calculate_relevance_score(user_vec, item_vec)

def _extract_item_embedding_from_result(result: dict):
    """Try to extract an item embedding from a result.

    Many backends do not include raw vectors in search responses. If absent, return None.
    Tries common locations like result['embedding'] or result['metadata']['embedding'].
    Also handles string-encoded embeddings that need to be parsed.
    """
    if not isinstance(result, dict):
        return None
    
    # Try direct embedding field
    if 'embedding' in result:
        emb = result['embedding']
        if isinstance(emb, (list, tuple)):
            return list(emb)
        elif isinstance(emb, str):
            # Try to parse string representation
            try:
                import ast
                parsed = ast.literal_eval(emb)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except:
                pass
    
    # Try embeddings field (plural)
    if 'embeddings' in result:
        emb = result['embeddings']
        if isinstance(emb, (list, tuple)):
            return list(emb)
        elif isinstance(emb, str):
            try:
                import ast
                parsed = ast.literal_eval(emb)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except:
                pass
    
    # Try metadata.embedding
    metadata = result.get('metadata') or {}
    if isinstance(metadata, dict):
        emb = metadata.get('embedding')
        if isinstance(emb, (list, tuple)):
            return list(emb)
        elif isinstance(emb, str):
            try:
                import ast
                parsed = ast.literal_eval(emb)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except:
                pass
        
        # Try metadata.embeddings
        emb = metadata.get('embeddings')
        if isinstance(emb, (list, tuple)):
            return list(emb)
        elif isinstance(emb, str):
            try:
                import ast
                parsed = ast.literal_eval(emb)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except:
                pass
    
    return None

def _extract_similarity_from_result(result: dict):
    """Extract a backend-provided similarity if present.

    Supabase matchers often include a 'similarity' or 'score' field; values are typically cosine or inner product.
    We assume higher is better. If present, map to [0,1] if value looks like [-1,1], else leave as-is and clamp.
    """
    if not isinstance(result, dict):
        return None
    for key in ('similarity', 'score'):
        if key in result:
            try:
                val = float(result[key])
                # Heuristic normalization: if in [-1,1], map to [0,1]
                if -1.0 <= val <= 1.0:
                    val = (val + 1.0) / 2.0
                # Clamp to [0,1]
                if val < 0.0:
                    val = 0.0
                if val > 1.0:
                    val = 1.0
                return val
            except Exception:
                return None
    return None

def fetch_embedding_from_supabase(supabase: Client, summary_id: str) -> list:
    """Fetch embedding from Supabase for a given summary ID"""
    try:
        response = (
            supabase.table("recommendations")
            .select("embeddings")
            .eq("entity_id", summary_id)
            .eq("entity_type", "summary")
            .execute()
        )
        
        if response.data and len(response.data) > 0:
            embeddings_str = response.data[0].get("embeddings")
            if embeddings_str:
                if isinstance(embeddings_str, (list, tuple)):
                    return list(embeddings_str)
                elif isinstance(embeddings_str, str):
                    import ast
                    parsed = ast.literal_eval(embeddings_str)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
        return None
    except Exception as e:
        logger.warning("Failed to fetch embedding from Supabase for id=%s: %s", summary_id, e)
        return None

def score_results(results: list, query_embedding: list, supabase: Client, context_label: str = "") -> list:
    """Attach relevance scores to results and sort them."""
    if not results or not RELEVANCE_SCORING_ENABLED:
        return results or []

    supabase_similarity_count = 0
    cosine_count = 0
    fallback_count = 0
    fetched_count = 0
    scored = []

    label_suffix = f" ({context_label})" if context_label else ""

    for result in results:
        sim = _extract_similarity_from_result(result)

        if sim is None:
            item_vec = _extract_item_embedding_from_result(result)

            if not item_vec:
                metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
                summary_id = metadata.get("id") or result.get("id") if isinstance(result, dict) else None
                if summary_id and supabase:
                    item_vec = fetch_embedding_from_supabase(supabase, summary_id)
                    if item_vec:
                        fetched_count += 1
                        logger.debug("Fetched embedding from Supabase for id=%s", summary_id)

            if item_vec:
                sim = calculate_relevance_score(query_embedding, item_vec)
                cosine_count += 1
                logger.info(
                    "Score via cosine%s: score=%.4f (%.0f%%) title='%s'",
                    label_suffix,
                    sim,
                    sim * 100,
                    (result.get("metadata") or {}).get("title", "") if isinstance(result, dict) else "",
                )
            else:
                sim = RELEVANCE_FALLBACK_SCORE
                fallback_count += 1
                logger.warning(
                    "Score via fallback%s: score=%.4f (%.0f%%) title='%s' - no embedding found",
                    label_suffix,
                    sim,
                    sim * 100,
                    (result.get("metadata") or {}).get("title", "") if isinstance(result, dict) else "",
                )
        else:
            supabase_similarity_count += 1
            logger.info(
                "Score via Supabase similarity%s: score=%.4f (%.0f%%) title='%s'",
                label_suffix,
                sim,
                sim * 100,
                (result.get("metadata") or {}).get("title", "") if isinstance(result, dict) else "",
            )

        if isinstance(result, dict):
            result["score"] = sim
        scored.append(result)

    scored = sorted(scored, key=lambda x: x.get("score", RELEVANCE_FALLBACK_SCORE) if isinstance(x, dict) else RELEVANCE_FALLBACK_SCORE, reverse=True)

    logger.info(
        "Scoring summary%s: supabase=%d cosine=%d fetched=%d fallback=%d total=%d",
        label_suffix,
        supabase_similarity_count,
        cosine_count,
        fetched_count,
        fallback_count,
        len(scored),
    )

    if RELEVANCE_SHOW_DEBUG:
        try:
            values = [
                item.get("score", RELEVANCE_FALLBACK_SCORE)
                for item in scored
                if isinstance(item, dict)
            ]
            if values:
                logger.info(
                    "Relevance stats%s: min=%.3f mean=%.3f max=%.3f fallbacks=%d/%d",
                    label_suffix,
                    min(values),
                    sum(values) / len(values),
                    max(values),
                    fallback_count,
                    len(values),
                )
        except Exception:  # pylint: disable=broad-except
            pass

    return scored


def render_recommendation_cards(
    results: list,
    key_prefix: str,
    button_label: str = "",
    button_callback=None,
    empty_message: str = "",
):
    """Render recommendation cards in a horizontal layout."""
    if not results:
        if empty_message:
            st.info(empty_message)
        return

    anchor_id = f"{key_prefix}_scroll_anchor"
    container_block = st.container()
    container_block.markdown(
        f"""
        <style>
        #{anchor_id} + div[data-testid="stHorizontalBlock"] {{
            display: flex;
            gap: 1.25rem;
            overflow-x: auto;
            padding: 0.35rem 0 0.9rem;
            align-items: stretch;
        }}
        #{anchor_id} + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
            flex: 0 0 360px;
            min-width: 360px;
            max-width: 360px;
        }}
        #{anchor_id} + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] button {{
            width: 100%;
        }}
        #{anchor_id} + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .card-inner {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.55rem;
            width: 100%;
        }}
        #{anchor_id} + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .card-inner img {{
            border-radius: 8px;
            width: 100%;
            height: auto;
        }}
        </style>
        <span id="{anchor_id}"></span>
        """,
        unsafe_allow_html=True,
    )

    columns = container_block.columns(len(results), gap="large")

    for idx, column in enumerate(columns):
        result = results[idx]
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        title = str(metadata.get("title", "Unknown Title") or "Unknown Title")
        author = str(metadata.get("author", "Unknown Author") or "Unknown Author")
        cover_page = metadata.get("cover_page", "")

        safe_title = html.escape(title)
        safe_author = html.escape(author)

        score_html = ""
        try:
            if isinstance(result, dict) and "score" in result:
                score = float(result["score"])
                score = max(0.0, min(1.0, score))
                score_html = (
                    f"<div style='color:#6B7280; font-size: 0.9rem; margin-top: 0.5rem;'>"
                    f"Relevance: {score * 100.0:.0f}%"
                    "</div>"
                )
        except Exception:  # pylint: disable=broad-except
            score_html = ""

        with column:
            st.markdown("<div class='card-inner'>", unsafe_allow_html=True)
            if cover_page:
                try:
                    st.image(cover_page, use_container_width=True)
                except Exception:  # pylint: disable=broad-except
                    st.markdown(
                        '<div style="background-color: #F8F4FF; padding: 1.2rem; border-radius: 8px; '
                        'text-align: center; color: #8B5CF6;">📖<br>Cover image not available</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="background-color: #F8F4FF; padding: 1.2rem; border-radius: 8px; '
                    'text-align: center; color: #8B5CF6;">📖<br>No cover image</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="result-card" style="width: 100%;">
                    <h2 class="book-title">{safe_title}</h2>
                    <p class="book-author"><b>By {safe_author}</b></p>
                    {score_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if button_label and button_callback:
                if st.button(
                    button_label,
                    key=f"{key_prefix}_button_{idx}",
                    help="Find similar recommendations",
                    width="stretch",
                ):
                    button_callback(result, idx)
            st.markdown("</div>", unsafe_allow_html=True)

def find_similar_books(title: str, author: str, category: str):
    """Find similar books based on title and author"""
    # Create search query from title and author
    search_query = f"{title} {author} {category}"
    
    # Show loading state
    status_placeholder = st.empty()
    status_placeholder.info("🔍 Finding similar books...")
    
    # Create embedding
    genai_client = init_genai_client()
    embedding = create_embedding(genai_client, search_query)
    
    if embedding is None:
        status_placeholder.empty()
        st.error("Failed to create embedding for similar books search.")
        return []
    
    # Search documents
    supabase = init_supabase_client()
    results_raw = search_summaries(supabase, embedding, search_query)
    results = score_results(results_raw, embedding, supabase, "similar_books")
    logger.info(
        "Similar books search: query='%s %s' results=%d",
        title,
        author,
        len(results or [])
    )

    # Compute/attach relevance scores and optionally sort
    if RELEVANCE_SCORING_ENABLED and results:
        fallback_count = 0
        cosine_count = 0
        supabase_similarity_count = 0
        fetched_count = 0
        scored = []
        
        for r in results:
            # Prefer backend similarity if available
            sim = _extract_similarity_from_result(r)
            
            if sim is None:
                # Try to extract embedding from result
                item_vec = _extract_item_embedding_from_result(r)
                
                # If not in result, try to fetch from Supabase
                if not item_vec:
                    metadata = r.get('metadata', {})
                    summary_id = metadata.get('id') or r.get('id')
                    if summary_id:
                        item_vec = fetch_embedding_from_supabase(supabase, summary_id)
                        if item_vec:
                            fetched_count += 1
                            logger.debug("Fetched embedding from Supabase for id=%s", summary_id)
                
                if item_vec:
                    sim = calculate_relevance_score(embedding, item_vec)
                    cosine_count += 1
                    logger.info(
                        "Score via cosine: score=%.4f (%.0f%%) title='%s'", 
                        sim,
                        sim * 100,
                        (r.get('metadata') or {}).get('title', '')
                    )
                else:
                    sim = RELEVANCE_FALLBACK_SCORE
                    fallback_count += 1
                    logger.warning(
                        "Score via fallback: score=%.4f (%.0f%%) title='%s' - no embedding found",
                        sim,
                        sim * 100,
                        (r.get('metadata') or {}).get('title', '')
                    )
            else:
                supabase_similarity_count += 1
                logger.info(
                    "Score via Supabase similarity: score=%.4f (%.0f%%) title='%s'",
                    sim,
                    sim * 100,
                    (r.get('metadata') or {}).get('title', '')
                )
            
            r['score'] = sim
            scored.append(r)
        
        # Sort by score desc; maintain stable order for ties
        scored = sorted(scored, key=lambda x: x.get('score', RELEVANCE_FALLBACK_SCORE), reverse=True)
        
        logger.info(
            "Scoring summary (similar_books): supabase=%d cosine=%d fetched=%d fallback=%d total=%d",
            supabase_similarity_count,
            cosine_count,
            fetched_count,
            fallback_count,
            len(scored)
        )
        
        if RELEVANCE_SHOW_DEBUG:
            try:
                values = [x.get('score', RELEVANCE_FALLBACK_SCORE) for x in scored]
                if values:
                    st.info(f"Relevance – min: {min(values)*100:.1f}%, mean: {sum(values)/len(values)*100:.1f}%, max: {max(values)*100:.1f}%, fallbacks: {fallback_count}/{len(values)}")
                    logger.info(
                        "Relevance stats: min=%.3f mean=%.3f max=%.3f fallbacks=%d/%d",
                        min(values),
                        (sum(values)/len(values)),
                        max(values),
                        fallback_count,
                        len(values)
                    )
            except Exception:
                pass
        results = scored
    
    status_placeholder.empty()
    
    # Filter out the exact same book from results
    filtered_results = []
    for result in results:
        metadata = result.get('metadata', {})
        result_title = metadata.get('title', '')
        result_author = metadata.get('author', '')
        # Exclude the exact same book
        if not (result_title.lower() == title.lower() and result_author.lower() == author.lower()):
            filtered_results.append(result)
    
    return filtered_results[:9]  # Return up to 9 similar books

def find_similar_bytes(title: str, author: str, category: str, byte_id: Optional[str]):
    """Find similar bytes based on title, author, and category"""
    parts = [title.strip() if title else "", author.strip() if author else "", category.strip() if category else ""]
    search_query = " ".join(filter(None, parts)) or title or author or category or ""

    status_placeholder = st.empty()
    status_placeholder.info("🔍 Finding similar bytes...")

    genai_client = init_genai_client()
    embedding = create_embedding(genai_client, search_query)

    if embedding is None:
        status_placeholder.empty()
        st.error("Failed to create embedding for similar bytes search.")
        return []

    supabase = init_supabase_client()
    results_raw = search_bytes(supabase, embedding, search_query)
    results = score_results(results_raw, embedding, supabase, "similar_bytes")

    logger.info(
        "Similar bytes search: query='%s' results=%d",
        search_query,
        len(results or []),
    )

    status_placeholder.empty()

    filtered_results = []
    normalized_byte_id = str(byte_id).lower() if byte_id else None
    lower_title = title.lower() if title else None
    lower_author = author.lower() if author else None

    for result in results:
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        result_title = (metadata.get("title") or "").lower()
        result_author = (metadata.get("author") or "").lower()
        result_id = metadata.get("id") or (result.get("id") if isinstance(result, dict) else None)
        normalized_result_id = str(result_id).lower() if result_id else None

        if normalized_byte_id and normalized_result_id and normalized_result_id == normalized_byte_id:
            continue

        same_title = bool(lower_title and result_title and result_title == lower_title)
        same_author = bool(lower_author and result_author and result_author == lower_author)

        if same_title and (not lower_author or same_author):
            continue

        filtered_results.append(result)

    return filtered_results[:9]

def reset_app():
    """Reset the app to initial state"""
    st.session_state.question_index = 0
    st.session_state.user_responses = []
    st.session_state.search_results = []
    st.session_state.bytes_results = []
    st.session_state.similar_books_results = []
    st.session_state.selected_book = None
    st.session_state.similar_bytes_results = []
    st.session_state.selected_byte = None
    st.session_state.app_state = "questions"

def main():
    # Add logo and title in header
    logo_path = Path("logo.png")
    
    # Create header with logo and title
    header_col1, header_col2 = st.columns([1, 12])
    
    with header_col1:
        if logo_path.exists():
            try:
                st.image("logo.png", width=60)
                # Add vertical spacing to align with title
                st.markdown("<br>", unsafe_allow_html=True)
            except Exception as e:
                st.write("")  # Empty if image fails to load
    
    with header_col2:
        st.markdown("""
            <style>
            .main-title {
                margin-top: 0.5rem;
            }
            </style>
            <div class="main-title">
                <h1 style="color: #8B5CF6; margin-bottom: 0;">📚 Personalized Book Recommendations</h1>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Initialize clients
    supabase = init_supabase_client()
    genai_client = init_genai_client()
    
    # Fetch questions if not already loaded
    if not st.session_state.questions:
        with st.spinner("Loading questions..."):
            st.session_state.questions = fetch_questions()
            if not st.session_state.questions:
                st.error("No questions found. Please check your database connection.")
                st.stop()
    
    # Main app flow
    if st.session_state.app_state == "questions":
        show_questions_flow()
    elif st.session_state.app_state == "similar_books":
        show_similar_books()
    elif st.session_state.app_state == "similar_bytes":
        show_similar_bytes()
    else:
        show_results()

def show_questions_flow():
    """Display questions one by one"""
    questions = st.session_state.questions
    current_idx = st.session_state.question_index
    
    if current_idx >= len(questions):
        # All questions answered, proceed to search
        process_responses()
        return
    
    # Get current question
    current_question = questions[current_idx]
    question_text = current_question.get('question', '')
    question_type = current_question.get('type', 'single_choice')
    options_json = current_question.get('options', '{}')
    images_json = current_question.get('images', '')
    
    # Parse options from JSON string
    try:
        options_dict = json.loads(options_json) if options_json else {}
    except:
        options_dict = {}
    
    # Parse images from JSON string if available
    images_dict = {}
    if images_json:
        try:
            images_dict = json.loads(images_json) if images_json else {}
        except:
            images_dict = {}
    
    # Display progress
    progress = (current_idx + 1) / len(questions)
    st.markdown(f'<div class="progress-bar"><b>Question {current_idx + 1} of {len(questions)}</b></div>', unsafe_allow_html=True)
    st.progress(progress)
    
    # Display question card
    safe_question = html.escape(question_text)
    st.markdown(f'''
        <div class="question-card">
            <h2>{safe_question}</h2>
        </div>
    ''', unsafe_allow_html=True)
    
    # Get existing response for this question if available
    existing_response = None
    if len(st.session_state.user_responses) > current_idx:
        existing_response = st.session_state.user_responses[current_idx]
    
    # Display options based on question type
    response = None
    
    if question_type == "single_choice":
        # Radio buttons for single choice
        option_keys = sorted(options_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        option_labels = [f"{options_dict[key]}" for key in option_keys]
        
        # Find default index if existing response
        default_index = 0
        if existing_response:
            try:
                # existing_response might be a string like "1" or a list with one element
                if isinstance(existing_response, list) and existing_response:
                    selected_key = str(existing_response[0])
                else:
                    selected_key = str(existing_response)
                if selected_key in option_keys:
                    default_index = option_keys.index(selected_key)
            except:
                pass
        
        selected_option = st.radio(
            "Select an option:",
            options=option_labels,
            index=default_index,
            key=f"radio_{current_idx}"
        )
        
        # Get the key for selected option
        selected_index = option_labels.index(selected_option)
        selected_key = option_keys[selected_index]
        response = selected_key
        
        # Display image if available
        if images_dict:
            selected_option_text = option_labels[selected_index]
            image_url = None
            
            # Try to find matching image by checking option text patterns
            for img_key, img_url in images_dict.items():
                # Check if image key matches the option key
                if img_key == str(selected_key):
                    image_url = img_url
                    break
                # Check if option text contains image key pattern (e.g., "16-25" matches "16_25")
                elif img_key.replace("_", "-") in selected_option_text or selected_option_text.replace("-", "_") in img_key:
                    image_url = img_url
                    break
            
            if image_url:
                # Display image in a smaller size (1/3 of container width)
                image_col1, image_col2, image_col3 = st.columns([1, 1, 1])
                with image_col2:
                    st.image(image_url, width=250, caption=selected_option)
    
    elif question_type == "multiple_choice":
        # Checkboxes for multiple choice
        option_keys = sorted(options_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        option_labels = {key: options_dict[key] for key in option_keys}
        
        # Get default selections if existing response
        default_selections = []
        if existing_response:
            if isinstance(existing_response, list):
                default_selections = [str(k) for k in existing_response]
            else:
                default_selections = [str(existing_response)]
        
        selected_options = []
        st.markdown("**Select all that apply:**")
        
        for key in option_keys:
            is_selected = str(key) in default_selections
            checked = st.checkbox(
                option_labels[key],
                value=is_selected,
                key=f"checkbox_{current_idx}_{key}"
            )
            if checked:
                selected_options.append(key)
        
        response = selected_options if selected_options else None
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    has_response = response is not None and (isinstance(response, list) and len(response) > 0 or not isinstance(response, list))
    
    with col1:
        if current_idx > 0:
            if st.button("← Previous"):
                # Save current response
                if has_response:
                    if len(st.session_state.user_responses) > current_idx:
                        st.session_state.user_responses[current_idx] = response
                    else:
                        st.session_state.user_responses.append(response)
                st.session_state.question_index -= 1
                st.rerun()
    
    with col2:
        pass  # Empty column for spacing
    
    with col3:
        if has_response:
            if current_idx < len(questions) - 1:
                if st.button("Next →"):
                    # Update or append response
                    if len(st.session_state.user_responses) > current_idx:
                        st.session_state.user_responses[current_idx] = response
                    else:
                        st.session_state.user_responses.append(response)
                    st.session_state.question_index += 1
                    st.rerun()
            else:
                if st.button("Submit ✨"):
                    # Update or append response
                    if len(st.session_state.user_responses) > current_idx:
                        st.session_state.user_responses[current_idx] = response
                    else:
                        st.session_state.user_responses.append(response)
                    process_responses()

@st.cache_data
def load_motivation_mapping():
    """Load motivation mapping from JSON file"""
    try:
        with open("motivation_mapping.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading motivation mapping: {str(e)}")
        return {}

def process_responses():
    """Combine responses and search for recommendations"""
    questions = st.session_state.questions
    motivation_mapping = load_motivation_mapping()
    combined_query, _ = build_search_query(questions, st.session_state.user_responses, motivation_mapping)

    if not combined_query:
        st.error("Please provide at least one answer to get recommendations.")
        st.session_state.app_state = "questions"
        st.session_state.question_index = 0
        st.rerun()
        return
    
    # Show loading state with progress steps
    status_placeholder = st.empty()
    status_placeholder.info("📝 Combining your responses...")
    time.sleep(0.5)
    
    # Create embedding
    status_placeholder.info("🧠 Creating search query...")
    genai_client = init_genai_client()
    embedding = create_embedding(genai_client, combined_query)
    
    if embedding is None:
        status_placeholder.empty()
        st.error("Failed to create embedding. Please try again.")
        if st.button("Try Again"):
            st.rerun()
        return
    
    # Search documents
    status_placeholder.info("🔍 Searching through our library...")
    supabase = init_supabase_client()
    summary_results_raw = search_summaries(supabase, embedding, combined_query)
    byte_results_raw = search_bytes(supabase, embedding, combined_query)
    summary_results = score_results(summary_results_raw, embedding, supabase, "questionnaire_summaries")
    byte_results = score_results(byte_results_raw, embedding, supabase, "questionnaire_bytes")
    
    status_placeholder.empty()
    
    if not summary_results and not byte_results:
        st.warning("No recommendations found. Try answering the questions differently.")
        if st.button("Start Over"):
            reset_app()
            st.rerun()
        return
    
    st.session_state.search_results = summary_results
    st.session_state.bytes_results = byte_results
    st.session_state.similar_books_results = []
    st.session_state.selected_book = None
    st.session_state.similar_bytes_results = []
    st.session_state.selected_byte = None
    st.session_state.app_state = "results"
    st.rerun()

def show_results():
    """Display search results in a scrollable format"""
    summary_results = st.session_state.search_results
    byte_results = st.session_state.bytes_results
    
    if not summary_results and not byte_results:
        st.info("No results to display.")
        if st.button("Start Over"):
            reset_app()
            st.rerun()
        return

    st.subheader("Summaries")

    def handle_similar_click(result, _idx):
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        title = metadata.get("title", "Unknown Title")
        author = metadata.get("author", "Unknown Author")
        cover_page = metadata.get("cover_page", "")
        category = metadata.get("category", "")

        st.session_state.selected_book = {
            "title": title,
            "author": author,
            "cover_page": cover_page,
        }
        similar_books = find_similar_books(title, author, category)
        st.session_state.similar_books_results = similar_books
        st.session_state.app_state = "similar_books"
        st.rerun()

    render_recommendation_cards(
        summary_results,
        key_prefix="summary",
        button_label="🔍 Find Similar Books",
        button_callback=handle_similar_click,
        empty_message="No summary recommendations available.",
    )

    st.markdown("---")

    st.subheader("Bytes")
    def handle_similar_bytes_click(result, _idx):
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        title = metadata.get("title", "Unknown Title")
        author = metadata.get("author", "Unknown Author")
        cover_page = metadata.get("cover_page", "")
        category = metadata.get("category", "")
        byte_id = metadata.get("id") or (result.get("id") if isinstance(result, dict) else None)

        st.session_state.selected_byte = {
            "title": title,
            "author": author,
            "cover_page": cover_page,
            "category": category,
            "id": byte_id,
            "description": metadata.get("description")
            or metadata.get("summary")
            or metadata.get("content")
            or metadata.get("byte_text"),
        }
        similar_bytes = find_similar_bytes(title, author, category, byte_id)
        st.session_state.similar_bytes_results = similar_bytes
        st.session_state.app_state = "similar_bytes"
        st.rerun()

    render_recommendation_cards(
        byte_results,
        key_prefix="bytes",
        button_label="🔍 Find Similar Bytes",
        button_callback=handle_similar_bytes_click,
        empty_message="No byte-sized recommendations available.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Start Over", width="stretch"):
            reset_app()
            st.rerun()

def show_similar_books():
    """Display similar books for a selected book"""
    selected_book = st.session_state.selected_book
    similar_books = st.session_state.similar_books_results
    
    if not selected_book:
        st.error("No book selected.")
        if st.button("← Back to Results"):
            st.session_state.app_state = "results"
            st.rerun()
        return
    
    # Display similar books
    st.header("🔍 Similar Books")
    st.markdown("---")

    def handle_nested_similar_click(result, _idx):
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        title = metadata.get("title", "Unknown Title")
        author = metadata.get("author", "Unknown Author")
        cover_page_value = metadata.get("cover_page", "")
        category = metadata.get("category", "")

        st.session_state.selected_book = {
            "title": title,
            "author": author,
            "cover_page": cover_page_value,
        }
        similar_books_new = find_similar_books(title, author, category)
        st.session_state.similar_books_results = similar_books_new
        st.rerun()

    render_recommendation_cards(
        similar_books,
        key_prefix="similar",
        button_label="🔍 Find Similar",
        button_callback=handle_nested_similar_click,
        empty_message="No similar books found.",
    )
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Results", width="stretch"):
            st.session_state.app_state = "results"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", width="stretch"):
            reset_app()
            st.rerun()
    with col3:
        pass

def show_similar_bytes():
    """Display similar bytes for a selected byte"""
    selected_byte = st.session_state.selected_byte
    similar_bytes = st.session_state.similar_bytes_results

    if not selected_byte:
        st.error("No byte selected.")
        if st.button("← Back to Results"):
            st.session_state.app_state = "results"
            st.rerun()
        return

    st.header("🔍 Similar Bytes")
    st.markdown("---")

    def handle_nested_similar_bytes_click(result, _idx):
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        title = metadata.get("title", "Unknown Title")
        author = metadata.get("author", "Unknown Author")
        cover_page_value = metadata.get("cover_page", "")
        category = metadata.get("category", "")
        byte_id_value = metadata.get("id") or (result.get("id") if isinstance(result, dict) else None)

        st.session_state.selected_byte = {
            "title": title,
            "author": author,
            "cover_page": cover_page_value,
            "category": category,
            "id": byte_id_value,
            "description": metadata.get("description")
            or metadata.get("summary")
            or metadata.get("content")
            or metadata.get("byte_text"),
        }
        similar_bytes_new = find_similar_bytes(title, author, category, byte_id_value)
        st.session_state.similar_bytes_results = similar_bytes_new
        st.rerun()

    render_recommendation_cards(
        similar_bytes,
        key_prefix="similar_bytes",
        button_label="🔍 Find Similar",
        button_callback=handle_nested_similar_bytes_click,
        empty_message="No similar bytes found.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Results", width="stretch"):
            st.session_state.app_state = "results"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", width="stretch"):
            reset_app()
            st.rerun()
    with col3:
        pass

if __name__ == "__main__":
    main()
