import streamlit as st
from supabase import create_client, Client
from google import genai
from google.genai import types
import time
import html
import json
import base64
import csv
from pathlib import Path

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
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    .question-card {
        background-color: #F8F4FF;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #E6D7FF;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(230, 215, 255, 0.2);
    }
    .result-card {
        background: linear-gradient(135deg, #F8F4FF 0%, #E6D7FF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #D4BFFF;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(139, 92, 246, 0.2);
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(139, 92, 246, 0.3);
        background: linear-gradient(135deg, #F3EBFF 0%, #D4BFFF 100%);
    }
    .book-title {
        color: #6D28D9 !important;
        margin-bottom: 0.5rem;
    }
    .book-author {
        color: #7C3AED !important;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .book-tagline {
        color: #5B21B6 !important;
        line-height: 1.6;
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
        background-color: #E6D7FF;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    h1 {
        color: #8B5CF6;
    }
    h2 {
        color: #7C3AED;
    }
    h3 {
        color: #6D28D9;
    }
    .stButton>button {
        background-color: #8B5CF6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #7C3AED;
    }
    </style>
""", unsafe_allow_html=True)

# Supabase configuration
SUPABASE_URL = "https://kijxqpprmvywetklzhbg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtpanhxcHBybXZ5d2V0a2x6aGJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4Njk4NjMsImV4cCI6MjA3NDQ0NTg2M30.d6eKlbz3s3KaqbMbxYceUBUFep3VehNZKEOe0ayPF2I"

# Google Genai configuration
GENAI_API_KEY = "AIzaSyBRpwUO7Y5aLqHUUBMWijdZx2SShFYEaUo"

# Relevance scoring configuration
# Toggle scoring/ranking, fallback score when missing data, and debug metrics rendering
RELEVANCE_SCORING_ENABLED = True
RELEVANCE_SHOW_DEBUG = False
RELEVANCE_FALLBACK_SCORE = 0.8  # Used when vectors/similarity are unavailable
RELEVANCE_HYBRID_ALPHA = 1.0    # Reserved for future hybridization

# Initialize session state
if "question_index" not in st.session_state:
    st.session_state.question_index = 0
if "user_responses" not in st.session_state:
    st.session_state.user_responses = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "app_state" not in st.session_state:
    st.session_state.app_state = "questions"  # "questions", "results", or "similar_books"
if "selected_book" not in st.session_state:
    st.session_state.selected_book = None
if "similar_books_results" not in st.session_state:
    st.session_state.similar_books_results = []

def init_supabase_client():
    """Initialize and return Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def init_genai_client():
    """Initialize and return Google Genai client"""
    return genai.Client(api_key=GENAI_API_KEY)

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
        return list(result.embeddings[0].values)
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None

def _safe_norm(vec: list):
    try:
        return sum((float(x) ** 2 for x in vec)) ** 0.5
    except Exception:
        return 0.0

def compute_relevance_score(user_vec: list, item_vec: list, normalize: bool = True) -> float:
    """Compute cosine similarity between two vectors with numerical safety.

    Returns a score in [0,1] if normalize=True using (cos+1)/2.
    Falls back to RELEVANCE_FALLBACK_SCORE if vectors are invalid or zero-norm.
    """
    if not isinstance(user_vec, (list, tuple)) or not isinstance(item_vec, (list, tuple)):
        return RELEVANCE_FALLBACK_SCORE
    if not user_vec or not item_vec:
        return RELEVANCE_FALLBACK_SCORE

    try:
        # Guard against size mismatch by truncating to min length
        n = min(len(user_vec), len(item_vec))
        if n == 0:
            return RELEVANCE_FALLBACK_SCORE
        dot = 0.0
        for i in range(n):
            dot += float(user_vec[i]) * float(item_vec[i])
        norm_u = _safe_norm(user_vec[:n])
        norm_v = _safe_norm(item_vec[:n])
        if norm_u == 0.0 or norm_v == 0.0:
            return RELEVANCE_FALLBACK_SCORE
        cos = dot / (norm_u * norm_v)
        # Clip for numerical stability
        if cos > 1.0:
            cos = 1.0
        elif cos < -1.0:
            cos = -1.0
        return (cos + 1.0) / 2.0 if normalize else cos
    except Exception:
        return RELEVANCE_FALLBACK_SCORE

def _extract_item_embedding_from_result(result: dict):
    """Try to extract an item embedding from a result.

    Many backends do not include raw vectors in search responses. If absent, return None.
    Tries common locations like result['embedding'] or result['metadata']['embedding'].
    """
    if not isinstance(result, dict):
        return None
    if 'embedding' in result and isinstance(result['embedding'], (list, tuple)):
        return result['embedding']
    metadata = result.get('metadata') or {}
    if isinstance(metadata, dict):
        emb = metadata.get('embedding')
        if isinstance(emb, (list, tuple)):
            return emb
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

def search_documents(supabase: Client, query_embedding: list):
    """Search for documents using vector similarity"""
    try:
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.6,
                "match_count": 10
            }
        ).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def find_similar_books(title: str, author: str):
    """Find similar books based on title and author"""
    # Create search query from title and author
    search_query = f"{title} {author}"
    
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
    results = search_documents(supabase, embedding)

    # Compute/attach relevance scores and optionally sort
    if RELEVANCE_SCORING_ENABLED and results:
        fallback_count = 0
        scored = []
        for r in results:
            # Prefer backend similarity if available
            sim = _extract_similarity_from_result(r)
            if sim is None:
                item_vec = _extract_item_embedding_from_result(r)
                sim = compute_relevance_score(embedding, item_vec, normalize=True) if item_vec else RELEVANCE_FALLBACK_SCORE
                if item_vec is None:
                    fallback_count += 1
            r['score'] = sim
            scored.append(r)
        # Sort by score desc; maintain stable order for ties
        scored = sorted(scored, key=lambda x: x.get('score', RELEVANCE_FALLBACK_SCORE), reverse=True)
        if RELEVANCE_SHOW_DEBUG:
            try:
                values = [x.get('score', RELEVANCE_FALLBACK_SCORE) for x in scored]
                if values:
                    st.info(f"Relevance – min: {min(values):.3f}, mean: {sum(values)/len(values):.3f}, max: {max(values):.3f}, fallbacks: {fallback_count}/{len(values)}")
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

def reset_app():
    """Reset the app to initial state"""
    st.session_state.question_index = 0
    st.session_state.user_responses = []
    st.session_state.search_results = []
    st.session_state.similar_books_results = []
    st.session_state.selected_book = None
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

def get_age_based_categories(age_group: str, has_kids: bool) -> str:
    """Get age-based categories based on life stage"""
    age_categories = {
        "16-25 years": "Productivity Personal Development Motivation Communication Career & Skills Creativity",
        "26-35 years": "Productivity Leadership Personal Development Relationships",
        "36-50 years": "Leadership Planning Health Mindfulness & Meditation Relationships Productivity Purpose & Values",
        "50+ years": "Health & Longevity Purpose & Values Relationships Spirituality & Philosophy Mindfulness & Meditation Nature & Wellness Creativity & Learning"
    }
    
    categories = age_categories.get(age_group, "")
    
    # Add Parenting if user has kids (for any age group)
    if has_kids:
        if "Parenting" not in categories:
            categories += " Parenting"
        # For 50+, also add Grandparenting
        if age_group == "50+ years" and "Grandparenting" not in categories:
            categories += " Grandparenting"
    
    return categories.strip()

def get_motivation_categories(motivation_keys: list, motivation_mapping: dict) -> str:
    """Get categories from motivation selections"""
    categories_set = set()
    
    for key in motivation_keys:
        key_str = str(key)
        if key_str in motivation_mapping:
            motivation_data = motivation_mapping[key_str]
            for cat in motivation_data.get("categories", []):
                categories_set.add(cat["name"])
    
    return " ".join(sorted(categories_set))

def process_responses():
    """Combine responses and search for recommendations"""
    # Convert responses from option keys to readable text
    questions = st.session_state.questions
    response_texts = []
    
    # Extract age group, has_kids, and motivation keys
    age_group = None
    has_kids = False
    motivation_keys = []
    
    # Question IDs for identification
    AGE_QUESTION_ID = "36d3b953-8f26-410f-8c64-729e778d8766"
    KIDS_QUESTION_ID = "e81709b2-e9aa-42d6-b843-4e62cfc18f99"
    MOTIVATION_QUESTION_ID = "d3349488-6ad2-465f-a8c0-653d4b9002f5"
    
    for idx, response in enumerate(st.session_state.user_responses):
        if idx < len(questions):
            question = questions[idx]
            question_id = question.get('id', '')
            question_text = question.get('question', '').lower()
            options_json = question.get('options', '{}')
            try:
                options_dict = json.loads(options_json) if options_json else {}
            except:
                options_dict = {}
            
            # Extract age group
            if question_id == AGE_QUESTION_ID or 'age group' in question_text:
                if isinstance(response, list) and response:
                    response_key = str(response[0])
                else:
                    response_key = str(response)
                if response_key in options_dict:
                    age_group = options_dict[response_key]
            
            # Extract has_kids
            if question_id == KIDS_QUESTION_ID or 'kids' in question_text:
                if isinstance(response, list) and response:
                    response_key = str(response[0])
                else:
                    response_key = str(response)
                if response_key in options_dict:
                    answer_text = options_dict[response_key]
                    has_kids = answer_text.lower() == "yes"
            
            # Extract motivation keys
            if question_id == MOTIVATION_QUESTION_ID or 'motivates you' in question_text:
                if isinstance(response, list):
                    motivation_keys = [str(key) for key in response]
                else:
                    motivation_keys = [str(response)]
            
            # Convert responses to readable text for search query
            if isinstance(response, list):
                # Multiple choice - get all selected option texts
                for key in response:
                    if str(key) in options_dict:
                        response_texts.append(options_dict[str(key)])
            else:
                # Single choice - get the selected option text
                if str(response) in options_dict:
                    response_texts.append(options_dict[str(response)])
    
    # Load motivation mapping
    motivation_mapping = load_motivation_mapping()
    
    # Get age-based categories
    age_categories = ""
    if age_group:
        age_categories = get_age_based_categories(age_group, has_kids)
    
    # Get motivation-based categories
    motivation_categories = ""
    if motivation_keys and motivation_mapping:
        motivation_categories = get_motivation_categories(motivation_keys, motivation_mapping)
    
    # Combine all components into search query
    query_parts = []
    if response_texts:
        query_parts.extend(response_texts)
    if age_categories:
        query_parts.append(age_categories)
    if motivation_categories:
        query_parts.append(motivation_categories)
    
    # Combine all response texts into a search query
    combined_query = " ".join(query_parts)
    
    if not combined_query.strip():
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
    results = search_documents(supabase, embedding)
    
    status_placeholder.empty()
    
    if not results:
        st.warning("No recommendations found. Try answering the questions differently.")
        if st.button("Start Over"):
            reset_app()
            st.rerun()
        return
    
    # Compute/attach relevance scores and optionally sort
    if RELEVANCE_SCORING_ENABLED and results:
        fallback_count = 0
        scored = []
        for r in results:
            # Prefer backend similarity if available
            sim = _extract_similarity_from_result(r)
            if sim is None:
                item_vec = _extract_item_embedding_from_result(r)
                sim = compute_relevance_score(embedding, item_vec, normalize=True) if item_vec else RELEVANCE_FALLBACK_SCORE
                if item_vec is None:
                    fallback_count += 1
            r['score'] = sim
            scored.append(r)
        # Sort by score desc; maintain stable order for ties
        scored = sorted(scored, key=lambda x: x.get('score', RELEVANCE_FALLBACK_SCORE), reverse=True)
        if RELEVANCE_SHOW_DEBUG:
            try:
                values = [x.get('score', RELEVANCE_FALLBACK_SCORE) for x in scored]
                if values:
                    st.info(f"Relevance – min: {min(values):.3f}, mean: {sum(values)/len(values):.3f}, max: {max(values):.3f}, fallbacks: {fallback_count}/{len(values)}")
            except Exception:
                pass
        results = scored

    st.session_state.search_results = results
    st.session_state.app_state = "results"
    st.rerun()

def show_results():
    """Display search results in a scrollable format"""
    results = st.session_state.search_results
    
    if not results:
        st.info("No results to display.")
        if st.button("Start Over"):
            reset_app()
            st.rerun()
        return
    
    # st.header("🎉 Your Personalized Recommendations")
    # st.markdown("---")
    
    # Create scrollable container
    results_container = st.container()
    
    with results_container:
        for idx, result in enumerate(results):
            metadata = result.get('metadata', {})
            
            # Extract book information
            title = metadata.get('title', 'Unknown Title')
            author = metadata.get('author', 'Unknown Author')
            tagline = metadata.get('tagline', '')
            cover_page = metadata.get('cover_page', '')
            
            # Create card layout with smaller image column
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if cover_page:
                    try:
                        # Display image at 1/3 size by using width constraint
                        st.image(
                            cover_page, 
                            width=200,
                            caption=f"{title} cover"
                        )
                    except Exception:
                        st.markdown(
                            '<div style="background-color: #F8F4FF; padding: 2rem; border-radius: 8px; text-align: center; color: #8B5CF6;">📖<br>Cover image not available</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div style="background-color: #F8F4FF; padding: 2rem; border-radius: 8px; text-align: center; color: #8B5CF6;">📖<br>No cover image</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Escape HTML to prevent XSS and formatting issues
                safe_title = html.escape(title)
                safe_author = html.escape(author)
                safe_tagline = html.escape(tagline)
                # Always show score if present
                score_html = ""
                try:
                    if 'score' in result:
                        pct = max(0.0, min(1.0, float(result['score']))) * 100.0
                        score_html = f"<div style='color:#6B7280; font-size: 0.9rem;'>Relevance: {pct:.0f}%</div>"
                except Exception:
                    score_html = ""

                st.markdown(f'''
                    <div class="result-card">
                        <h2 class="book-title">{safe_title}</h2>
                        <p class="book-author"><b>By {safe_author}</b></p>
                        <p class="book-tagline">{safe_tagline}</p>
                        {score_html}
                    </div>
                ''', unsafe_allow_html=True)
                
                # Add button to find similar books
                if st.button(f"🔍 Find Similar Books", key=f"similar_{idx}", use_container_width=True):
                    # Store selected book and find similar books
                    st.session_state.selected_book = {
                        'title': title,
                        'author': author,
                        'tagline': tagline,
                        'cover_page': cover_page
                    }
                    similar_books = find_similar_books(title, author)
                    st.session_state.similar_books_results = similar_books
                    st.session_state.app_state = "similar_books"
                    st.rerun()
            
            if idx < len(results) - 1:
                st.markdown("---")
    
    # Start Over button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Start Over", use_container_width=True):
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
    
    # Display selected book at the top
    st.header("📖 Selected Book")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if selected_book.get('cover_page'):
            try:
                st.image(
                    selected_book['cover_page'],
                    width=200,
                    caption=f"{selected_book['title']} cover"
                )
            except Exception:
                st.markdown(
                    '<div style="background-color: #F8F4FF; padding: 2rem; border-radius: 8px; text-align: center; color: #8B5CF6;">📖<br>Cover image not available</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="background-color: #F8F4FF; padding: 2rem; border-radius: 8px; text-align: center; color: #8B5CF6;">📖<br>No cover image</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        safe_title = html.escape(selected_book.get('title', 'Unknown Title'))
        safe_author = html.escape(selected_book.get('author', 'Unknown Author'))
        safe_tagline = html.escape(selected_book.get('tagline', ''))
        st.markdown(f'''
            <div class="result-card">
                <h2 class="book-title">{safe_title}</h2>
                <p class="book-author"><b>By {safe_author}</b></p>
                <p class="book-tagline">{safe_tagline}</p>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display similar books
    if not similar_books:
        st.info("No similar books found.")
    else:
        st.header("🔍 Similar Books")
        st.markdown("---")
        
        similar_container = st.container()
        
        with similar_container:
            for idx, result in enumerate(similar_books):
                metadata = result.get('metadata', {})
                
                # Extract book information
                title = metadata.get('title', 'Unknown Title')
                author = metadata.get('author', 'Unknown Author')
                tagline = metadata.get('tagline', '')
                cover_page = metadata.get('cover_page', '')
                
                # Create card layout
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if cover_page:
                        try:
                            st.image(
                                cover_page,
                                width=200,
                                caption=f"{title} cover"
                            )
                        except Exception:
                            st.markdown(
                                '<div style="background-color: #F8F4FF; padding: 2rem; border-radius: 8px; text-align: center; color: #8B5CF6;">📖<br>Cover image not available</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div style="background-color: #F8F4FF; padding: 2rem; border-radius: 8px; text-align: center; color: #8B5CF6;">📖<br>No cover image</div>',
                            unsafe_allow_html=True
                        )
                
                with col2:
                    safe_title = html.escape(title)
                    safe_author = html.escape(author)
                    safe_tagline = html.escape(tagline)
                    # Always show score if present
                    score_html = ""
                    try:
                        if 'score' in result:
                            pct = max(0.0, min(1.0, float(result['score']))) * 100.0
                            score_html = f"<div style='color:#6B7280; font-size: 0.9rem;'>Relevance: {pct:.0f}%</div>"
                    except Exception:
                        score_html = ""

                    st.markdown(f'''
                        <div class="result-card">
                            <h2 class="book-title">{safe_title}</h2>
                            <p class="book-author"><b>By {safe_author}</b></p>
                            <p class="book-tagline">{safe_tagline}</p>
                            {score_html}
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Add button to find similar books for this book too
                    if st.button(f"🔍 Find Similar", key=f"similar_similar_{idx}", use_container_width=True):
                        st.session_state.selected_book = {
                            'title': title,
                            'author': author,
                            'tagline': tagline,
                            'cover_page': cover_page
                        }
                        similar_books_new = find_similar_books(title, author)
                        st.session_state.similar_books_results = similar_books_new
                        st.rerun()
                
                if idx < len(similar_books) - 1:
                    st.markdown("---")
    
    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.app_state = "results"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over", use_container_width=True):
            reset_app()
            st.rerun()
    with col3:
        pass

if __name__ == "__main__":
    main()

