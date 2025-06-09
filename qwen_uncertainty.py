import os
import csv
import torch
import argparse
import librosa
import transformers
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


def main(args):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

    with open(args.csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            audio_path = row['new_audio_path']
            idx = row['idx']
            question = row["question"]
            optionA = row["optionA"]
            optionB = row["optionB"]
            optionC = row["optionC"]
            optionD = row["optionD"]
            
            prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

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

            generate_output = model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )

            generated_scores = generate_output.scores
            generated_ids = generate_output.sequences[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            score_tensor = generated_scores[0] 
            probs = torch.softmax(score_tensor, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=1, dim=-1)
            
            top_ids = top_ids[0].tolist() 
            top_probs = top_probs[0].tolist()

            top_tokens = processor.batch_decode([[tid] for tid in top_ids], skip_special_tokens=True)

            os.makedirs("confidence_estimation", exist_ok=True)
            output_file_path = os.path.join("confidence_estimation", f"qwen_{args.task}.txt")
            with open(output_file_path, "a") as f:
                for token, prob in zip(top_tokens, top_probs):
                    f.write(f"{idx} {token} {prob}\n")
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
