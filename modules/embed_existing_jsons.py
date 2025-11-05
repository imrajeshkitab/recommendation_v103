import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google import genai
from google.genai import types


def load_env() -> None:
    # Load environment from .env if present
    load_dotenv()


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


class TransientError(Exception):
    pass


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


def build_embedding_input(payload: Dict[str, Any]) -> str:
    """Match the input composition used in embed_summaries.py.
    Concatenate title, category, tagline, author, and content.
    """
    title = payload.get("title")
    category = payload.get("category")
    tagline = payload.get("tagline")
    author = payload.get("author")
    content = payload.get("content")
    return safe_join_text([title, category, tagline, author, content])


def process_file(gclient: genai.Client, path: str, force: bool = False) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {path}: {e}", file=sys.stderr)
        return None

    # Skip if embedding already present unless forced
    if not force and isinstance(data, dict) and "embedding" in data:
        return path

    text = build_embedding_input(data)
    try:
        vector = embed_text(gclient, text)
    except Exception as e:
        print(f"Failed to embed {path}: {e}", file=sys.stderr)
        return None

    data["embedding"] = {
        "model": "gemini-embedding-001",
        "dim": 1536,
        "vector": vector,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write {path}: {e}", file=sys.stderr)
        return None

    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read JSON files from a directory (default: ./embeddings_copy), "
            "generate Gemini embeddings using the same logic as embed_summaries.py, "
            "and write the 'embedding' field back into each JSON file."
        )
    )
    parser.add_argument("--indir", default="embeddings_copy", help="Input directory containing JSON files")
    parser.add_argument("--one", dest="one_id", default=None, help="Process only a single file id (without .json)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing 'embedding' field if present")

    args = parser.parse_args(argv)

    load_env()
    gclient = get_gemini_client()

    if not os.path.isdir(args.indir):
        print(f"Input directory not found: {args.indir}", file=sys.stderr)
        return 1

    targets: List[str] = []
    if args.one_id:
        one_path = os.path.join(args.indir, f"{args.one_id}.json")
        if not os.path.exists(one_path):
            print(f"File not found: {one_path}", file=sys.stderr)
            return 1
        targets = [one_path]
    else:
        for name in os.listdir(args.indir):
            if name.endswith(".json"):
                targets.append(os.path.join(args.indir, name))

    if not targets:
        print("No JSON files found to process.")
        return 0

    processed = 0
    skipped = 0
    failures = 0

    for p in targets:
        out = process_file(gclient, p, force=args.force)
        if out is None:
            failures += 1
            continue
        if out and not args.force:
            # If embedding existed and we didn't force, treat as skipped
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if "embedding" in d:
                    skipped += 1
                    print(f"Skipped (embedding exists): {p}")
                    continue
            except Exception:
                # If we cannot verify, count as processed to avoid confusion
                pass
        processed += 1
        print(f"Wrote {p}")

    print(f"Done. processed={processed} skipped={skipped} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())