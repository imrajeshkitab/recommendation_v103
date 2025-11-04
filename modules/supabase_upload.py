import os
from supabase import create_client, Client
import json
from pathlib import Path

# Updated Supabase connection with hardcoded credentials
SUPABASE_URL = "https://kijxqpprmvywetklzhbg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtpanhxcHBybXZ5d2V0a2x6aGJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4Njk4NjMsImV4cCI6MjA3NDQ0NTg2M30.d6eKlbz3s3KaqbMbxYceUBUFep3VehNZKEOe0ayPF2I"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


all_data = [
    json.load(open(f, "r", encoding="utf-8"))
    for f in Path("embeddings").glob("*.json")
]

for item in all_data:
    temp = {
        "entity_id": item["id"],
        "entity_type": "summary",
        "embeddings": item["embedding"]['vector'],
        "metadata": {
            "title": item["title"],
            "category": item["category"],
            "tagline": item["tagline"],
            "author": item["author"],
            "cover_page": item["cover_page"],
        }
    }
    response = (
    supabase.table("recommendations")
    .insert(temp)
    .execute()
    )
    # break