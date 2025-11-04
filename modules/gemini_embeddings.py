from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyBRpwUO7Y5aLqHUUBMWijdZx2SShFYEaUo")

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents= [
            "What is the meaning of life?",
            "What is the purpose of existence?",
            "How do I bake a cake?"
        ],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY",output_dimensionality=1536))

for embedding in result.embeddings:
    print(embedding)