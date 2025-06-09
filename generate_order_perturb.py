import argparse
import random
import os
from pydub import AudioSegment
import csv
from ast import literal_eval
import pandas as pd

random.seed(44)

def normalize_audio(audio, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)

def join_audio_with_crossfade(audio1, audio2, crossfade_duration=500, with_silence=True, silence_duration=1000):    
    if with_silence:
        silence_segment = AudioSegment.silent(duration=int(silence_duration)) 
        merged_audio = audio1.append(silence_segment, crossfade=crossfade_duration)
        merged_audio = merged_audio.append(audio2, crossfade=crossfade_duration)
    else:
        merged_audio = audio1.append(audio2, crossfade=crossfade_duration)
    return merged_audio

def add_silence(audio_files, output_folder, idx, orig_idx):
    silence_index = random.randint(1, 20)
    silence_clip = f"TREA_dataset/synthetic_silences/silent_{silence_index}.wav"
    silence_audio = AudioSegment.from_file(silence_clip, format="wav")
    silence_duration = len(silence_audio) / 1000 

    insert_position = random.randint(0, min(5, len(audio_files)))
    new_audio_files = audio_files[:insert_position] + [silence_clip] + audio_files[insert_position:]

    audio1 = AudioSegment.from_file(new_audio_files[0], format="wav")
    audio2 = AudioSegment.from_file(new_audio_files[1], format="wav")
    merged_res = join_audio_with_crossfade(audio1, audio2)
    for i in range(2, len(new_audio_files)):
        next_audio = AudioSegment.from_file(new_audio_files[i], format="wav")
        merged_res = join_audio_with_crossfade(merged_res, next_audio)

    new_audio_path = f"{output_folder}/{orig_idx}_silence_{idx}.wav"
    merged_res.export(new_audio_path, format="wav")

    return {
        "new_audio_path": new_audio_path,
        "silence_duration": silence_duration,
        "new_audio_files": new_audio_files
    }

def adjust_volume(audio_files, output_folder, idx, orig_idx):
    num_of_clips = random.randint(1, min(5, len(audio_files)))
    selected_files = random.sample(audio_files, num_of_clips)
    
    audio_segments = []
    volume_changes = {}

    for file in audio_files:
        audio = AudioSegment.from_file(file, format="wav")
        audio = normalize_audio(audio)
        if file in selected_files:
            db_change = random.choice([-10, -5, 5, 10, 15, 20])
            audio += db_change
            volume_changes[file] = db_change
        else:
            volume_changes[file] = 0
        audio_segments.append(audio)

    merged_res = audio_segments[0]
    for seg in audio_segments[1:]:
        merged_res = join_audio_with_crossfade(merged_res, seg)

    new_audio_path = f"{output_folder}/{orig_idx}_volume_{idx}.wav"
    merged_res.export(new_audio_path, format="wav")

    return {
        "new_audio_path": new_audio_path,
        "volume_changes": volume_changes
    }

def change_duration(audio_files, output_folder, idx, orig_idx):
    sampled_audio = random.choice(audio_files)
    repeat_count = random.randint(1, 6 - len(audio_files))
    index = audio_files.index(sampled_audio)
    new_audio_files = audio_files[:index + 1] + [sampled_audio] * repeat_count + audio_files[index + 1:]

    audio1 = AudioSegment.from_file(new_audio_files[0], format="wav")
    audio2 = AudioSegment.from_file(new_audio_files[1], format="wav")
    merged_res = join_audio_with_crossfade(audio1, audio2)
    for i in range(2, len(new_audio_files)):
        next_audio = AudioSegment.from_file(new_audio_files[i], format="wav")
        merged_res = join_audio_with_crossfade(merged_res, next_audio)

    new_audio_path = f"{output_folder}/{orig_idx}_duration_{idx}.wav"
    merged_res.export(new_audio_path, format="wav")

    return {
        "new_audio_path": new_audio_path,
        "repeated_path": sampled_audio,
        "repeated_num": repeat_count,
        "new_audio_files": new_audio_files
    }

def process_csv(input_csv, output_folder, num_samples, num_samples_per_type):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_csv)
    df["source_wavs"] = df["source_wavs"].apply(literal_eval)
    df["source_categories"] = df["source_categories"].apply(literal_eval)

    selected_indices = sorted(random.sample(range(min(200, len(df))), num_samples))
    per_row_cat_mapping = [
        {wav: cat for wav, cat in zip(wavs, cats)}
        for wavs, cats in zip(df["source_wavs"], df["source_categories"])
    ]

    perturbation_funcs = {
        'silence': add_silence,
        'duration': change_duration,
        'volume': adjust_volume
    }

    fieldnames_common = [
        'idx', 'orig_idx', 'new_audio_path', 'question', 'optionA', 'optionB', 'optionC',
        'optionD', 'answer', 'variation_type', 'cat_mapping'
    ]

    fieldnames_dict = {
        'silence': fieldnames_common + ['silence_duration', 'audio_files'],
        'duration': fieldnames_common + ['repeated_path', 'repeated_num', 'audio_files'],
        'volume': fieldnames_common + ['volume_changes', 'audio_files']
    }

    csv_writers = {}
    csv_files = {}
    for perturbation, fields in fieldnames_dict.items():
        out_path = os.path.join(output_folder, f"{perturbation}_perturbations.csv")
        f = open(out_path, 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        csv_writers[perturbation] = writer
        csv_files[perturbation] = f

    for idx in selected_indices:
        row = df.iloc[idx]
        audio_files = row["source_wavs"]
        cat_mapping = per_row_cat_mapping[idx]
        orig_id = row['audio_path'].split('/')[-1].split('.')[0]

        for i in range(num_samples_per_type):
            for perturbation_type, func in perturbation_funcs.items():
                metadata = func(audio_files, output_folder, i, orig_id)
                new_idx = metadata['new_audio_path'].split('/')[-1].split('.')[0]

                row_data = {
                    'idx': new_idx,
                    'orig_idx': orig_id,
                    'new_audio_path': metadata["new_audio_path"],
                    'question': row['question'],
                    'optionA': row['optionA'],
                    'optionB': row['optionB'],
                    'optionC': row['optionC'],
                    'optionD': row['optionD'],
                    'answer': row['correct'],
                    'variation_type': perturbation_type,
                    'cat_mapping': cat_mapping
                }

                if perturbation_type == 'silence':
                    row_data.update({
                        'silence_duration': metadata['silence_duration'],
                        'audio_files': metadata['new_audio_files']
                    })
                elif perturbation_type == 'duration':
                    row_data.update({
                        'repeated_path': metadata['repeated_path'],
                        'repeated_num': metadata['repeated_num'],
                        'audio_files': metadata['new_audio_files']
                    })
                elif perturbation_type == 'volume':
                    row_data.update({
                        'volume_changes': metadata['volume_changes'],
                        'audio_files': audio_files
                    })

                csv_writers[perturbation_type].writerow(row_data)

    for f in csv_files.values():
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate perturbed order audio samples from metadata.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to store output files")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of random rows to sample from first 200")
    parser.add_argument("--num_samples_per_type", type=int, default=5, help="Number of variations per perturbation type")
    
    args = parser.parse_args()
    process_csv(args.input_csv, args.output_folder, args.num_samples, args.num_samples_per_type)

    merged_csv_path = os.path.join(args.output_folder, "order_perturbations.csv")
    merged_df = pd.concat([
        pd.read_csv(os.path.join(args.output_folder, "silence_perturbations.csv")),
        pd.read_csv(os.path.join(args.output_folder, "duration_perturbations.csv")),
        pd.read_csv(os.path.join(args.output_folder, "volume_perturbations.csv"))
    ], ignore_index=True)

    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged CSV saved to: {merged_csv_path}")
