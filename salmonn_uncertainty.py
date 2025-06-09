import os
import csv
import torch
import argparse
from transformers import WhisperFeatureExtractor, pipeline
from config import Config
from models.salmonn import SALMONN
import transformers
from utils import prepare_one_sample

def main(args):
    cfg_path = "configs/decode_config_calibration.yaml"
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
            
            question = row["question"]
            optionA = row["optionA"]
            optionB = row["optionB"]
            optionC = row["optionC"]
            optionD = row["optionD"]
            prompt = f"{question}\nChoices:\nA. {optionA}\nB. {optionB}\nC. {optionC}\nD. {optionD}\nReturn only the option (A,B,C or D), and nothing else.\nMAKE SURE your output is A,B,C or D"

            samples = prepare_one_sample(wav_path, wav_processor)
            full_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]

            with torch.cuda.amp.autocast(dtype=torch.float16):
                text,generated_scores = model.generate_calibration(samples, cfg.config.generate, prompts=full_prompt)

            output_file_path = os.path.join(output_folder, f"salmonn_{args.task}.txt")
            with open(output_file_path, "a") as f:
                for token, prob in generated_scores.items():
                    f.write(f"{idx} {token} {prob}\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["count", "duration", "order"], required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--wav_folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
