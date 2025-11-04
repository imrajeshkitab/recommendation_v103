import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google import genai
from google.genai import types


def load_env() -> None:
    # Load environment from .env if present
    load_dotenv()


def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is required")

    key = service_key or anon_key
    if not key:
        raise RuntimeError("SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY is required")

    return create_client(supabase_url, key)


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required")
    return genai.Client(api_key=api_key)


def safe_join_text(parts: List[Optional[str]]) -> str:
    cleaned: List[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            cleaned.append(s)
    return "\n\n".join(cleaned)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def output_path(base_dir: str, summary_id: Any) -> str:
    return os.path.join(base_dir, f"{summary_id}.json")


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class TransientError(Exception):
    pass


@retry(reraise=True,
       stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(TransientError))
def fetch_summaries(sb: Client, limit: Optional[int] = None, summary_id: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        query = sb.table("summaries").select("*")
        if summary_id:
            query = query.eq("id", summary_id)
        if limit is not None:
            query = query.limit(limit)
        resp = query.execute()
        return resp.data or []
    except Exception as e:
        # Treat network/timeouts as transient
        msg = str(e).lower()
        if any(k in msg for k in ["timeout", "temporar", "connection", "rate"]):
            raise TransientError(str(e))
        raise


@retry(reraise=True,
       stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(TransientError))
def fetch_chapters(sb: Client, summary_id: Any) -> List[Dict[str, Any]]:
    try:
        resp = (
            sb.table("summary_chapters")
              .select("*")
              .eq("summary_id", summary_id)
              .order("chapter")
              .execute()
        )
        return resp.data or []
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["timeout", "temporar", "connection", "rate"]):
            raise TransientError(str(e))
        raise


@retry(reraise=True,
       stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(TransientError))
def embed_text(client: genai.Client, text: str) -> List[float]:
    try:
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=1536,
            ),
        )
        if not res.embeddings:
            raise RuntimeError("No embeddings returned")
        # google-genai returns objects with .values
        return list(res.embeddings[0].values)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["timeout", "temporar", "connection", "rate", "quota", "unavailable"]):
            raise TransientError(str(e))
        raise


def build_full_content(chapters: List[Dict[str, Any]]) -> str:
    # Join chapter contents in order with blank line spacing
    pieces: List[str] = []
    for ch in chapters:
        c = ch.get("content")
        if c and str(c).strip():
            pieces.append(str(c).strip())
    return "\n\n".join(pieces)


def build_embedding_input(summary: Dict[str, Any], full_content: str) -> str:
    title = summary.get("title")
    category = summary.get("category")
    tagline = summary.get("tagline")
    author = summary.get("author")
    return safe_join_text([title, category, tagline, author, full_content])


def write_summary_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def process_summary(sb: Client, gclient: genai.Client, base_dir: str, summary: Dict[str, Any], force: bool = False) -> Optional[str]:
    sid = summary.get("id")
    if sid is None:
        return None

    ensure_output_dir(base_dir)
    out_path = output_path(base_dir, sid)
    if os.path.exists(out_path) and not force:
        return out_path

    chapters = fetch_chapters(sb, sid)
    full_content = build_full_content(chapters)

    text = build_embedding_input(summary, full_content)
    vector = embed_text(gclient, text)

    payload = {
        "id": sid,
        "title": summary.get("title"),
        "category": summary.get("category"),
        "author": summary.get("author"),
        "tagline": summary.get("tagline"),
        "content": full_content,
        "embedding": {
            "model": "gemini-embedding-001",
            "dim": 1536,
            "vector": vector,
        },
        "created_at": now_iso(),
    }

    write_summary_json(out_path, payload)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch summaries, generate Gemini embeddings, save JSON per summary.")
    parser.add_argument("--id", dest="one_id", help="Process only a single summary id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of summaries to process")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON files if present")
    parser.add_argument("--outdir", default="embeddings", help="Output directory, default ./embeddings")

    args = parser.parse_args(argv)

    load_env()
    sb = get_supabase_client()
    gclient = get_gemini_client()

    summaries = fetch_summaries(sb, limit=args.limit, summary_id=args.one_id)
    if not summaries:
        print("No summaries found matching criteria.")
        return 0

    processed = 0
    skipped = 0
    failures = 0

    for s in summaries:
        try:
            out = process_summary(sb, gclient, args.outdir, s, force=args.force)
            if out is None:
                skipped += 1
            else:
                print(f"Wrote {out}")
                processed += 1
        except Exception as e:
            failures += 1
            print(f"Failed for id={s.get('id')}: {e}", file=sys.stderr)
            continue

    print(f"Done. processed={processed} skipped={skipped} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
