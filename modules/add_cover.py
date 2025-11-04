import json
import os
from pathlib import Path
from supabase import create_client, Client

# Supabase connection
SUPABASE_URL = "https://ziifsmeagbyyjsnmxwns.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppaWZzbWVhZ2J5eWpzbm14d25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3ODQ0NjIsImV4cCI6MjA3MTM2MDQ2Mn0.86SwhbQRHmiK8wwL0tNqPHPcVyKmYcXRmCoD-U7qQLo"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def add_cover_and_page_to_embeddings(embeddings_folder: str = "embeddings"):
    """
    Read all JSON files from embeddings folder, fetch summary data from Supabase,
    and add cover and page fields to each JSON file.
    """
    folder = Path(embeddings_folder)
    
    if not folder.exists():
        print(f"Error: Folder '{embeddings_folder}' does not exist")
        return
    
    # Get all JSON files in the folder
    json_files = sorted(folder.glob("*.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        print(f"No JSON files found in '{embeddings_folder}'")
        return
    
    print(f"Processing {total_files} JSON files...")
    
    updated = 0
    skipped = 0
    errors = 0
    
    for json_file in json_files:
        try:
            # Read the JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            summary_id = data.get("id")
            
            if not summary_id:
                print(f"Skipping {json_file.name}: No 'id' field found")
                skipped += 1
                continue
            
            # Fetch summary from Supabase
            try:
                response = (
                    supabase.table("summaries")
                    .select("cover_page")
                    .eq("id", summary_id)
                    .execute()
                )
                
                if response.data and len(response.data) > 0:
                    summary = response.data[0]
                    
                    # Add cover and page fields
                    if "cover_page" in summary:
                        data["cover_page"] = summary["cover_page"]
                    
                    # Save updated JSON back to file
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    
                    print(f"✓ Updated {json_file.name} (ID: {summary_id})")
                    updated += 1
                else:
                    print(f"⚠ No summary found in Supabase for ID: {summary_id}")
                    skipped += 1
                    
            except Exception as e:
                print(f"✗ Error fetching from Supabase for {json_file.name}: {e}")
                errors += 1
                
        except json.JSONDecodeError as e:
            print(f"✗ Error reading JSON file {json_file.name}: {e}")
            errors += 1
        except Exception as e:
            print(f"✗ Unexpected error processing {json_file.name}: {e}")
            errors += 1
    
    print(f"\nDone!")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {total_files}")

if __name__ == "__main__":
    add_cover_and_page_to_embeddings()

