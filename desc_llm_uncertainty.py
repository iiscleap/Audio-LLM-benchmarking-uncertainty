import os
import csv
import torch
import argparse
from transformers import WhisperFeatureExtractor, pipeline
from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def get_audio_desc_answer(desc, row):
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    gen_config = GenerationConfig(
    max_new_tokens=1,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    bos_token_id=128000,
    eos_token_id=[128001, 128008, 128009]
    )

    question = row['question']
    optionA = row['optionA']
    optionB = row['optionB']
    optionC = row['optionC']
    optionD = row['optionD']

    prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}"
    full_prompt = f"The description of the audio clip is given below:\n{desc}\nBased on the information above, answer the following:\n{prompt}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that must read the description and answer the question. Your response must contain only the option and nothing else"},
        {"role": "user", "content": full_prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True).to("cuda")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config,
        return_dict_in_generate=True,
        output_logits=True
    )

    generated_scores = output.logits
    generated_ids = output.sequences[:, input_ids.size(1):]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    score_tensor = generated_scores[0]
    probs = torch.softmax(score_tensor, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=1, dim=-1)

    top_ids = top_ids[0].tolist() 
    top_probs = top_probs[0].tolist()

    top_tokens = tokenizer.batch_decode([[tid] for tid in top_ids], skip_special_tokens=True)

    res = []
    for token, prob in zip(top_tokens, top_probs):
        token = token.strip()
        res.append(token)
        res.append(prob)
    
    return res[0],res[1]

def main(args):
    cfg_path = "configs/decode_config.yaml"
    output_folder = "confidence_estimation"
    os.makedirs(output_folder, exist_ok=True)

    cfg = Config(argparse.Namespace(cfg_path=cfg_path, device="cuda:0", options=None))
    model = SALMONN.from_config(cfg.config.model)
    model.to("cuda:0")
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['idx']
            audio_idx = result = "_".join(idx.split("_")[:3])
            wav_path = os.path.join(args.wav_folder, f"{audio_idx}.wav")

            prompt = "Please describe all the events and sounds occurring in the audio clip in detail. Identify and describe each sound source, such as objects, animals, weather, or environmental noises. Include information about the duration, count and sequence of events."

            samples = prepare_one_sample(wav_path, wav_processor)
            full_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]

            with torch.cuda.amp.autocast(dtype=torch.float16):
                text = model.generate(samples, cfg.config.generate, prompts=full_prompt)

            os.makedirs(os.path.join(output_folder, f"{args.task}_audio_desc"), exist_ok=True)
            with open(os.path.join(output_folder, f"{args.task}_audio_desc", f"{idx}.txt"), "w") as f:
                f.write(text)
    

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idx = row['idx']
            with open(os.path.join(output_folder, f"{args.task}_audio_desc", f"{idx}.txt"), "r") as file:
                desc = file.read()
            
            final_ans, prob = get_audio_desc_answer(desc, row)
            output_file_path = os.path.join(output_folder, f"desc_llm_{args.task}.txt")
            with open(output_file_path, "a") as f:
                f.write(f"{idx} {final_ans} {prob}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--wav_folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
