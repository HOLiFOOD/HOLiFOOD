import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
import concurrent.futures
from datetime import datetime

# -----------------------------
# Model and Pipeline Setup
# -----------------------------
model_name = "huihui-ai/Llama-3.1-Nemotron-Nano-8B-v1-abliterated"
access_token = "replace"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=access_token,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True
)
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -----------------------------
# Helper Functions (unchanged)
# -----------------------------
def chunk_text_by_tokens(text, tokenizer, max_tokens=4096):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks

def summarize_with_llama(text, summarizer, tokenizer, max_tokens=4096, max_new_tokens=150):
    if not text.strip():
        return ""
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens)
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize the following text in 4 sentences maximum. "
            "ONLY output the summary, do not repeat the original text or include any additional commentary:\n\n"
            f"{chunk}\n\nSummary:"
        )
        print(f"[DEBUG] Summarizing chunk {i+1}/{len(chunks)}")
        output = summarizer(prompt, max_new_tokens=max_new_tokens, temperature=0.1)
        gen = output[0]["generated_text"]
        summary = gen.split("Summary:")[-1].strip() if "Summary:" in gen else gen.strip()
        summaries.append(summary)
    return " ".join(summaries)

def safe_summarize_with_timeout(text, summarizer, tokenizer, timeout=60, max_tokens=4096, max_new_tokens=150):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(summarize_with_llama, text, summarizer, tokenizer, max_tokens, max_new_tokens)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"[ERROR] Summarization timed out after {timeout} seconds.")
            return ""

# -----------------------------
# Processing Function: Only Save Summarized Chunk!
# -----------------------------
def process_json_file(input_json, output_json, start_index=0):
    try:
        with open(input_json, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON file {input_json}: {e}")
        return

    print(f"[DEBUG] Loaded {len(data)} records from {input_json}")
    subset = data[start_index:]
    total = len(subset)
    print(f"[DEBUG] Processing {total} records starting from index {start_index}")

    summarized_records = []  # NEW: only these will be saved!

    for offset, record in enumerate(tqdm(subset, total=total, desc="Summarizing records")):
        idx = start_index + offset
        url = record.get("URL", "")
        print(f"[DEBUG] Summarizing record {idx} from URL: {url}")
        content = record.get("Content", "")
        summary = safe_summarize_with_timeout(content, summarizer, tokenizer, timeout=60)
        record["Summary"] = summary
        summarized_records.append(record)  # NEW: Add to new list

        # --- Incremental save (only summarized chunk) ---
        try:
            with open(output_json, 'w', encoding='utf-8') as fout:
                json.dump(summarized_records, fout, ensure_ascii=False, indent=4)
            print(f"[DEBUG] Saved progress up to local chunk index {offset} → {output_json}")
        except Exception as e:
            print(f"[ERROR] Failed to save at index {idx}: {e}")

    print(f"[DEBUG] All done. Final output in {output_json}")
    return output_json

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    input_json = "food_safety_links_20250609_output.json"
    output_json = "data.json"
    start_index = 758
    process_json_file(input_json, output_json, start_index=start_index)
    print("[DEBUG] Summarization process completed successfully.")
