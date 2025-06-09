import random
import os
from pydub import AudioSegment
import csv
from collections import Counter
from ast import literal_eval
import pandas as pd
import argparse

random.seed(43)

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

def insertion_function(audio_list, path):
    num_insertions = random.randint(1, 6 - len(audio_list))
    insert_indices = random.sample(range(len(audio_list) + num_insertions), num_insertions)
    new_list = audio_list[:]
    for idx in sorted(insert_indices, reverse=True):
        new_list.insert(idx, path)
    return new_list, num_insertions

def deletion_function(audio_list, path):
    indices = [i for i, p in enumerate(audio_list) if p == path]
    remove_idx = random.choice(indices)
    new_list = audio_list[:]
    del new_list[remove_idx]
    return new_list, 1

def insert_delete_existing_audio(audio_files, output_folder, idx, orig_idx):
    path_counts = Counter(audio_files)
    sampled_path = random.choice(audio_files)

    if path_counts[sampled_path] == 1 or random.choice([True, False]):
        new_audio_files, num = insertion_function(audio_files, sampled_path)
        operation = "insertion"
    else:
        new_audio_files, num = deletion_function(audio_files, sampled_path)
        operation = "deletion"

    audio1 = AudioSegment.from_file(new_audio_files[0], format="wav")
    if len(new_audio_files) > 1:
        audio2 = AudioSegment.from_file(new_audio_files[1], format="wav")
        merged_res = join_audio_with_crossfade(audio1, audio2)
        for i in range(2, len(new_audio_files)):
            next_audio = AudioSegment.from_file(new_audio_files[i], format="wav")
            merged_res = join_audio_with_crossfade(merged_res, next_audio)
    else:
        merged_res = audio1

    new_audio_path = f"{output_folder}/{orig_idx}_add_delete_audio_{idx}.wav"
    merged_res.export(new_audio_path, format="wav")

    return {
        "new_audio_path": new_audio_path,
        "operation": operation,
        "operation_path": sampled_path,
        "operation_num": num,
        "new_audio_files": new_audio_files
    }

def process_csv(input_csv, output_folder, num_samples, num_samples_per_type):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_csv)
    df["source_wavs"] = df["source_wavs"].apply(literal_eval)
    df["source_categories"] = df["source_categories"].apply(literal_eval)

    per_row_cat_mapping = [
        {wav: cat for wav, cat in zip(wavs, cats)}
        for wavs, cats in zip(df["source_wavs"], df["source_categories"])
    ]

    available_indices = df.index.tolist()
    selected_idx = random.sample(available_indices, min(num_samples, len(available_indices)))

    perturbation_types = ['silence', 'add_delete_audio', 'volume']
    perturbation_funcs = {
        'silence': add_silence,
        'add_delete_audio': insert_delete_existing_audio,
        'volume': adjust_volume
    }

    fieldnames_common = [
        'idx', 'orig_idx', 'new_audio_path', 'question', 'optionA', 'optionB', 'optionC',
        'optionD', 'answer', 'variation_type', 'cat_mapping'
    ]
    fieldnames_dict = {
        'silence': fieldnames_common + ['silence_duration', 'audio_files'],
        'add_delete_audio': fieldnames_common + ['operation', 'operation_path', 'operation_num', 'audio_files'],
        'volume': fieldnames_common + ['volume_changes', 'audio_files']
    }

    csv_writers = {}
    csv_files = {}
    for perturbation in perturbation_types:
        csv_path = os.path.join(output_folder, f"{perturbation}_perturbations.csv")
        csv_files[perturbation] = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writers[perturbation] = csv.DictWriter(csv_files[perturbation], fieldnames=fieldnames_dict[perturbation])
        csv_writers[perturbation].writeheader()

    for idx in selected_idx:
        row = df.loc[idx]
        orig_idx = row['audio_path'].split('/')[-1].split('.')[0]
        cat_mapping = per_row_cat_mapping[int(orig_idx)]
        audio_files = row['source_wavs']

        for i in range(num_samples_per_type):
            for variation_type, func in perturbation_funcs.items():
                metadata = func(audio_files, output_folder, i, orig_idx)
                new_idx = metadata['new_audio_path'].split('/')[-1].split('.')[0]
                row_data = {
                    'idx': new_idx,
                    'orig_idx': orig_idx,
                    'new_audio_path': metadata["new_audio_path"],
                    'question': row['question'],
                    'optionA': row['optionA'],
                    'optionB': row['optionB'],
                    'optionC': row['optionC'],
                    'optionD': row['optionD'],
                    'answer': row['correct'],
                    'variation_type': variation_type,
                    'cat_mapping': cat_mapping
                }

                if variation_type == 'silence':
                    row_data.update({
                        'silence_duration': metadata['silence_duration'],
                        'audio_files': metadata['new_audio_files']
                    })
                elif variation_type == 'add_delete_audio':
                    row_data.update({
                        'operation': metadata['operation'],
                        'operation_path': metadata['operation_path'],
                        'operation_num': metadata['operation_num'],
                        'audio_files': metadata['new_audio_files']
                    })
                elif variation_type == 'volume':
                    row_data.update({
                        'volume_changes': metadata['volume_changes'],
                        'audio_files': row['source_wavs']
                    })

                csv_writers[variation_type].writerow(row_data)

    for f in csv_files.values():
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate perturbed count audio samples from metadata.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input metadata CSV file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save perturbed audio and CSVs.")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of random samples to process.")
    parser.add_argument("--num_samples_per_type", type=int, default=5, help="Number of variations per perturbation type")
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_folder, args.num_samples, args.num_samples_per_type)

    merged_csv_path = os.path.join(args.output_folder, "count_perturbations.csv")
    merged_df = pd.concat([
        pd.read_csv(os.path.join(args.output_folder, "silence_perturbations.csv")),
        pd.read_csv(os.path.join(args.output_folder, "add_delete_audio_perturbations.csv")),
        pd.read_csv(os.path.join(args.output_folder, "volume_perturbations.csv"))
    ], ignore_index=True)

    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged CSV saved to: {merged_csv_path}")
