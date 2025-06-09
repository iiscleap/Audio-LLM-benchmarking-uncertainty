import os
import csv
import torch
import argparse
from transformers import WhisperFeatureExtractor, pipeline
from config import Config
from models.salmonn import SALMONN
import transformers
from utils import prepare_one_sample

def get_prompt(method, question, optionA, optionB, optionC, optionD):
    if method == "vanilla":
        return f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"
    elif method == "cot":
        return f"You must provide step-by-step reasoning before answering the following question:\n{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
    elif method == "exp":
        return f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nState your answer and provide an explanation."
    elif method == "audio_desc":
        return "Please describe all the events and sounds occurring in the audio clip in detail. Identify and describe each sound source, such as objects, animals, weather, or environmental noises. Include information about the duration, count and sequence of events."
    else:
        raise ValueError(f"Unsupported method: {method}")

def extract_final_answer(llm_pipeline, model_output, row):
    question = row['question']
    optionA = row['optionA']
    optionB = row['optionB']
    optionC = row['optionC']
    optionD = row['optionD']
    idx = row["id"]

    prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
    text = f"""Given a model's output and a question with four options, choose the option that the output matches:

Model Output:
{model_output}

Question:
{prompt}

If the model output matches multiple options, respond with NA. If the model output does not match any of the options, respond with NA.
Your response should only contain the option (A,B,C or D) or NA, and nothing else."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]

    response = llm_pipeline(messages, max_new_tokens=10)
    final_answer = response[0]["generated_text"][-1]['content'].strip()
    return final_answer

def get_audio_desc_answer(llm_pipeline, desc, row):
    question = row['question']
    optionA = row['optionA']
    optionB = row['optionB']
    optionC = row['optionC']
    optionD = row['optionD']

    prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
    full_prompt = f"The description of the audio clip is given below:\n{desc}\nBased on the information above, answer the following:\n{prompt}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": full_prompt},
    ]

    outputs = llm_pipeline(messages, max_new_tokens=1)
    return outputs[0]["generated_text"][-1]["content"].strip()

def main(args):
    if args.method == "vanilla":
        cfg_path = "configs/decode_config_calibration.yaml"
    else:
        cfg_path = "configs/decode_config.yaml"
    
    output_folder = "salmonn_output"
    os.makedirs(output_folder, exist_ok=True)

    cfg = Config(argparse.Namespace(cfg_path=cfg_path, device="cuda:0", options=None))
    model = SALMONN.from_config(cfg.config.model)
    model.to("cuda:0")
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['id']
            wav_path = os.path.join(args.wav_folder, f"{idx}.wav")
            prompt_text = get_prompt(args.method, row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD'])

            samples = prepare_one_sample(wav_path, wav_processor)
            full_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]

            with torch.cuda.amp.autocast(dtype=torch.float16):
                text = model.generate(samples, cfg.config.generate, prompts=full_prompt)

            if args.method == "vanilla":
                output_file_path = os.path.join(output_folder, f"{args.task}_{args.method}_output.txt")
                with open(output_file_path, "a") as f:
                    f.write(f"{idx} {text}\n")
            
            else:
                os.makedirs(os.path.join(output_folder, f"{args.task}_{args.method}"), exist_ok=True)
                with open(os.path.join(output_folder, f"{args.task}_{args.method}", f"{idx}.txt"), "w") as f:
                    f.write(text)
            
    if args.method == "cot" or args.method == "exp":
        output_folder = "salmonn_output"
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                idx = row['id']
                with open(os.path.join(output_folder, f"{args.task}_{args.method}", f"{idx}.txt"), "r") as file:
                    output = file.read()
                
                final_ans = extract_final_answer(pipeline, output, row)
                output_file_path = os.path.join(output_folder, f"{args.task}_{args.method}_output.txt")
                with open(output_file_path, "a") as f:
                    f.write(f"{idx} {final_ans}\n")
    
    elif args.method == "audio_desc":
        output_folder = "salmonn_output"
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                idx = row['id']
                with open(os.path.join(output_folder, f"{args.task}_{args.method}", f"{idx}.txt"), "r") as file:
                    desc = file.read()
                
                final_ans = get_audio_desc_answer(pipeline, desc, row)
                output_file_path = os.path.join(output_folder, f"{args.task}_{args.method}_output.txt")
                with open(output_file_path, "a") as f:
                    f.write(f"{idx} {final_ans}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["vanilla", "cot", "exp", "audio_desc"], required=True)
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--wav_folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
