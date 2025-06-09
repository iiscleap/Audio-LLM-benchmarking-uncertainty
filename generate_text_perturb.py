import transformers
import torch
import pandas as pd
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate reworded question variants using LLM.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
parser.add_argument("--output_folder", type=str, required=True, help="Directory to save the output CSV.")
args = parser.parse_args()

# Load the LLM
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def extract_questions(text):
    return [q.strip() for q in text.split('\n') if q.strip()]

def generate_variants(question):
    """Generate 4 variations of a given question using the LLM."""
    prompt = f"Generate 4 rephrased versions of the following question while ensuring that the meaning is the EXACT same:{question}\nYour response should contain the 4 questions in 4 different lines ONLY."

    messages = [
        {"role": "system", "content": "You are an assistant that rephrases questions while keeping their meaning intact."},
        {"role": "user", "content": prompt},
    ]
    
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    
    generated_text = outputs[0]["generated_text"][-1]['content']
    print(generated_text)
    print("=" * 25)
    variants = extract_questions(generated_text)
    return variants

def process_csv(input_csv, output_folder):
    """Read a CSV, generate question variations, and save a new CSV with expanded rows."""
    df = pd.read_csv(input_csv)
    output_rows = []

    for _, row in df.iterrows():
        question = row["question"]
        audio_path = row["new_audio_path"]

        variants = generate_variants(question)

        for i, variant in enumerate(variants, 1):
            new_row = row.copy()
            new_row["idx"] = f"{row['idx']}_rephrased{i}"
            new_row["question"] = variant
            output_rows.append(new_row)
            print(variant)
        print("*" * 30)

    output_df = pd.DataFrame(output_rows)
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, f"reworded_{os.path.basename(input_csv)}")
    output_df.to_csv(output_csv_path, index=False)
    print(f"Processed CSV saved as: {output_csv_path}")

# Run processing
process_csv(args.input_csv, args.output_folder)
