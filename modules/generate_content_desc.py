import os
import requests
import json
import time

# --- Configuration ---

# API Key - leave as an empty string. The environment will provide it.
API_KEY = "AIzaSyBRpwUO7Y5aLqHUUBMWijdZx2SShFYEaUo"

# API Endpoint for gemini-2.5-flash
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# Input and Output Directories
INPUT_DIR = "database/summary_text_copy"
OUTPUT_DIR = "database/semantic_desc"

# The powerful system prompt we designed
SYSTEM_PROMPT = """
You are a content profiling expert. Your task is to analyze a piece of content (like a book summary or article) and generate a single, descriptive paragraph. This paragraph will serve as the "semantic profile" for the content, describing its ideal audience and characteristics.

You **must** structure this paragraph by addressing **all 9** of the following questions, using the specific options provided.

---

### **Questionnaire Guide (Use these exact phrases)**

**Q1. What is your age group?**
* **Options:** `16-25 years`, `26-35 years`, `36-50 years`, `50+ years`
* *(Instruction: You can select one or more. If it's for everyone, state it's "suitable for all age groups".)*

**Q2. What's your gender?**
* **Options:** `Male`, `Female`, `Others`
* *(Instruction: Only mention if the content is *highly specific*. Otherwise, state it is "universally applicable and not gender-specific".)*

**Q3. What's your relationship status?**
* **Options:** `Single`, `In a relationship`, `Married`, `Divorced/Widowed`
* *(Instruction: Only mention if *highly specific* (e.g., "content for singles"). Otherwise, state it's "relevant for any relationship status".)*

**Q4. Do you have kids?**
* **Options:** `Yes`, `No`
* *(Instruction: Only mention if *highly specific* (e.g., "parenting content"). Otherwise, state it's "relevant for individuals with or without kids".)*

**Q5. What's on your mind these days? (Select all that apply)**
* **Options:** `I want to feel more calm and balanced`, `I want to be more confident`, `I want to improve my relationships`, `I want clarity about my life or career`, `I want to grow and be my best self`, `I just love learning new things`, `I'm just exploring`

**Q6. What motivates you the most right now? (Select all that apply)**
* **Options:** `Feeling peaceful inside`, `Achieving my goals with less stress`, `Discovering who I really am`, `Being a better friend/partner/parent`, `Finding direction and purpose`, `Making better life choices`

**Q7. How would you relate the following with yourself? (Select all that apply)**
* **Options:** `I focus on logic, analysis, and planning`, `I'm driven by values, emotion, and do what's right`, `I concentrate on details, facts, and believe in practicality`, `I love exploring new ideas and envision future possibilities`

**Q8. How much time do you want to spend daily on yourself? (Select all that apply)**
* **Options:** `5 min – Quick insight & reflection`, `10 min – A daily story + learning`, `15+ min – Deep dive into wisdom`
* *(Instruction: Estimate this based on the content's length and depth.)*

**Q9. Choose your learning style (Select the best fit)**
* **Options:** `Step by step, I like going slow and steady`, `Give me something exciting and powerful`, `I love deep thinking and reflection`, `Just show me what works`

---

### **Your Task**

Now, read the content provided by the user and generate the single descriptive paragraph. Start the paragraph by stating the most relevant age groups and demographic applicability, then flow through the answers for questions Q5-Q9.
"""

# --- End of Configuration ---


def generate_description(content_text):
    """
    Calls the Gemini API to generate a semantic description for the given text.
    Implements exponential backoff for retries.
    """
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{
            "parts": [{"text": content_text}]
        }],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        }
    }

    max_retries = 5
    delay = 1  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=45)

            # If successful, parse and return the text
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Robustly parse the response
                    description = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
                    if description:
                        return description
                    else:
                        print(f"   [Error] API response was successful but content was empty. Response: {result}")
                        return None
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"   [Error] Could not parse API response: {e}. Response text: {response.text}")
                    return None

            # Handle server-side errors (retry-able)
            elif response.status_code >= 500:
                print(f"   [Warning] Server error ({response.status_code}). Retrying in {delay}s...")
            
            # Handle client-side errors (not retry-able)
            elif response.status_code >= 400:
                print(f"   [Error] Client error ({response.status_code}). Aborting. Response: {response.text}")
                return None
            
            # Wait for the next retry
            time.sleep(delay)
            delay *= 2  # Exponential backoff

        except requests.exceptions.RequestException as e:
            print(f"   [Error] Request failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    print("   [Error] Max retries exceeded. Failed to process content.")
    return None


def main():
    """
    Main function to loop through files and process them.
    """
    print(f"Starting content processing...")
    print(f"Input folder:  {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        filenames = os.listdir(INPUT_DIR)
    except FileNotFoundError:
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        print("Please create the directory and add your .txt files.")
        return

    total_files = len([f for f in filenames if f.endswith('.txt')])
    processed_count = 0
    skipped_count = 0
    
    print(f"Found {total_files} .txt files to process.\n")

    for filename in filenames:
        if not filename.endswith('.txt'):
            continue
            
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # --- Checkpointing ---
        # Skip if the output file already exists
        if os.path.exists(output_path):
            print(f"[{processed_count+skipped_count+1}/{total_files}] Skipping '{filename}' (already processed).")
            skipped_count += 1
            continue

        print(f"[{processed_count+skipped_count+1}/{total_files}] Processing '{filename}'...")

        try:
            # Read the content from the input file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                print("   [Warning] File is empty. Skipping.")
                continue

            # Generate the description
            description = generate_description(content)

            # Write the new description to the output file
            if description:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(description)
                print(f"   [Success] Saved semantic description to {output_path}")
                processed_count += 1
            else:
                print(f"   [Failed] Could not generate description for {filename}")

        except Exception as e:
            print(f"   [FATAL ERROR] An unexpected error occurred with {filename}: {e}")

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Failed: {total_files - processed_count - skipped_count}")

if __name__ == "__main__":
    main()