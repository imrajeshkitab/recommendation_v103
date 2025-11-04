import os
from supabase import create_client, Client

# Updated Supabase connection with hardcoded credentials
SUPABASE_URL = "https://ziifsmeagbyyjsnmxwns.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppaWZzbWVhZ2J5eWpzbm14d25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU3ODQ0NjIsImV4cCI6MjA3MTM2MDQ2Mn0.86SwhbQRHmiK8wwL0tNqPHPcVyKmYcXRmCoD-U7qQLo"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

response = (
        supabase.table("summaries")
        .select("*")
        .execute()
    )
summary_data = response.data
summary_data


for sum in summary_data:
    summary_chap = (
            supabase.table("summary_chapters")
            .select("*")
            .eq("summary_id", sum['id'])
            .order("chapter")
            .execute()
        )
    chapters = summary_chap.data
    for chap in chapters:
        print(chap['title'])
        print(chap['content'])
        print("--------------------------------")