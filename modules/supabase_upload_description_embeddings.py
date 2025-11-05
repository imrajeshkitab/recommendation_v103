import json
from pathlib import Path

from supabase import create_client, Client

# Supabase connection (kept consistent with supabase_upload.py)
SUPABASE_URL = "https://kijxqpprmvywetklzhbg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtpanhxcHBybXZ5d2V0a2x6aGJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4Njk4NjMsImV4cCI6MjA3NDQ0NTg2M30.d6eKlbz3s3KaqbMbxYceUBUFep3VehNZKEOe0ayPF2I"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    files = sorted(Path("embeddings_content_desc").glob("*.json"))
    print(f"Found {len(files)} files in embeddings_content_desc")

    updated = 0
    skipped = 0
    failed = 0

    for fp in files:
        try:
            item = load_json(fp)
            entity_id = item.get("id")
            vector = (item.get("embedding") or {}).get("vector")

            if not entity_id or not vector:
                print(f"Skip {fp.name}: missing 'id' or 'embedding.vector'")
                skipped += 1
                continue

            # First try to update existing row by entity_id
            resp = (
                supabase.table("recommendations")
                .update({"description_embedding": vector})
                .eq("entity_id", entity_id)
                .execute()
            )

            data = getattr(resp, "data", None)
            if data and len(data) > 0:
                updated += 1
                print(f"Updated description_embedding for {entity_id}")
                continue

            # If no existing row, upsert on entity_id (creates or updates)
            payload = {
                "entity_id": entity_id,
                "entity_type": "summary",
                "description_embedding": vector,
            }
            resp2 = (
                supabase.table("recommendations")
                .upsert(payload, on_conflict="entity_id")
                .execute()
            )
            data2 = getattr(resp2, "data", None)
            if data2 is not None:
                updated += 1
                print(f"Upserted description_embedding for {entity_id}")
            else:
                failed += 1
                print(f"Failed to upsert for {entity_id}")

        except Exception as e:
            failed += 1
            print(f"Failed {fp.name}: {e}")

    print(f"Done. updated={updated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()