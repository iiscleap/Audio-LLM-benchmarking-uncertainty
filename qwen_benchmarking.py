import os
import csv
import torch
import argparse
import librosa
import transformers
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

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
Your response should only contain the option(A,B,C or D) or NA, and nothing else."""

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
    full_prompt = f"The description of the audio clip is given below:\n{desc}\nBased on the information above, answer the following:\n{prompt}Return only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": full_prompt},
    ]

    outputs = llm_pipeline(messages, max_new_tokens=1)
    return outputs[0]["generated_text"][-1]["content"].strip()

def main(args):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            audio_path = row['audio_path']
            idx = row['id']

            prompt = get_prompt(args.method, row['question'], row['optionA'], row['optionB'], row['optionC'], row['optionD'])

            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {"type": "text", "text": prompt}]}
            ]

            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            waveform, _ = librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)
                            audios.append(waveform)

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to("cuda")
            max_tokens = 1 if args.method == "vanilla" else 256

            generate_output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )

            generated_ids = generate_output[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            if args.method == "vanilla":
                os.makedirs("qwen_output", exist_ok=True)
                output_file_path = os.path.join("qwen_output", f"{args.task}_{args.method}_output.txt")
                with open(output_file_path, "a") as f:
                    f.write(f"{idx} {response}\n")
            
            else:
                os.makedirs(f"qwen_output/{args.task}_{args.method}", exist_ok=True)
                with open(os.path.join("qwen_output", f"{args.task}_{args.method}", f"{idx}.txt"), "w") as f:
                    f.write(response)

    if args.method == "cot" or args.method == "exp":
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
                with open(os.path.join("qwen_output", f"{args.task}_{args.method}", f"{idx}.txt"), "r") as file:
                    output = file.read()
                
                final_ans = extract_final_answer(pipeline, output, row)
                output_file_path = os.path.join("qwen_output", f"{args.task}_{args.method}_output.txt")
                with open(output_file_path, "a") as f:
                    f.write(f"{idx} {final_ans}\n")
    
    elif args.method == "audio_desc":
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
                with open(os.path.join("qwen_output", f"{args.task}_{args.method}", f"{idx}.txt"), "r") as file:
                    desc = file.read()
                
                final_ans = get_audio_desc_answer(pipeline, desc, row)
                output_file_path = os.path.join("qwen_output", f"{args.task}_{args.method}_output.txt")
                with open(output_file_path, "a") as f:
                    f.write(f"{idx} {final_ans}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["vanilla", "cot", "exp", "audio_desc"], required=True)
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
