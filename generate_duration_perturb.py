import random
import os
from pydub import AudioSegment
import csv
import argparse
import pandas as pd

random.seed(42)

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

def shuffle_order(audio_files, output_folder, idx, orig_idx):
    audio_files_copy = audio_files.copy()
    random.shuffle(audio_files_copy)

    audio1 = AudioSegment.from_file(audio_files_copy[0], format="wav")
    audio2 = AudioSegment.from_file(audio_files_copy[1], format="wav")
    merged_res = join_audio_with_crossfade(audio1, audio2)
    for i in range(2, len(audio_files_copy)):
        next_audio = AudioSegment.from_file(audio_files_copy[i], format="wav")
        merged_res = join_audio_with_crossfade(merged_res, next_audio)

    new_audio_path = f"{output_folder}/{orig_idx}_order_{idx}.wav"
    merged_res.export(new_audio_path, format="wav")

    return {
        "new_audio_path": new_audio_path,
        "new_audio_files": audio_files_copy
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
            audio = audio + db_change
            volume_changes[file] = db_change
        else:
            volume_changes[file] = 0
        audio_segments.append(audio)

    merged_res = join_audio_with_crossfade(audio_segments[0], audio_segments[1])
    for i in range(2, len(audio_segments)):
        merged_res = join_audio_with_crossfade(merged_res, audio_segments[i])

    new_audio_path = f"{output_folder}/{orig_idx}_volume_{idx}.wav"
    merged_res.export(new_audio_path, format="wav")

    return {
        "new_audio_path": new_audio_path,
        "volume_changes": volume_changes
    }

def process_csv(input_csv, output_folder, num_samples, num_samples_per_type):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    total_indices = [row['audio_path'].split('/')[-1].split('.')[0] for row in rows]
    total_indices = [int(idx) for idx in total_indices if idx.isdigit()]
    selected_idx = random.sample(total_indices, min(num_samples, len(total_indices)))

    perturbation_types = ['silence', 'order', 'volume']
    perturbation_funcs = {
        'silence': add_silence,
        'order': shuffle_order,
        'volume': adjust_volume
    }

    fieldnames_common = [
        'idx', 'orig_idx', 'new_audio_path', 'question', 'optionA', 'optionB', 'optionC',
        'optionD', 'answer', 'variation_type', 'cat_mapping'
    ]
    fieldnames_silence = fieldnames_common + ['silence_duration', 'audio_files']
    fieldnames_order = fieldnames_common + ['audio_files']
    fieldnames_volume = fieldnames_common + ['volume_changes', 'audio_files']

    csv_writers = {}
    csv_files = {}
    fieldnames_dict = {
        'silence': fieldnames_silence,
        'order': fieldnames_order,
        'volume': fieldnames_volume
    }

    for perturbation in perturbation_types:
        csv_path = os.path.join(output_folder, f"{perturbation}_perturbations.csv")
        csv_files[perturbation] = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writers[perturbation] = csv.DictWriter(csv_files[perturbation], fieldnames=fieldnames_dict[perturbation])
        csv_writers[perturbation].writeheader()

    for row in rows:
        orig_idx = row['audio_path'].split('/')[-1].split('.')[0]

        if int(orig_idx) in selected_idx:
            audio_files = eval(row['audiopath_list'])

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
                        'cat_mapping': row['cat_mapping']
                    }

                    if variation_type == 'silence':
                        row_data.update({
                            'silence_duration': metadata['silence_duration'],
                            'audio_files': metadata['new_audio_files']
                        })
                    elif variation_type == 'order':
                        row_data.update({
                            'audio_files': metadata['new_audio_files']
                        })
                    elif variation_type == 'volume':
                        row_data.update({
                            'volume_changes': metadata['volume_changes'],
                            'audio_files': row['audiopath_list']
                        })

                    csv_writers[variation_type].writerow(row_data)

    for file in csv_files.values():
        file.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate perturbed duration audio samples from metadata.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_folder", type=str, required=True, help="Directory to save perturbed audio and CSVs")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of random examples to perturb")
    parser.add_argument("--num_samples_per_type", type=int, default=5, help="Number of variations per perturbation type")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_csv(args.input_csv, args.output_folder, args.num_samples, args.num_samples_per_type)

    merged_csv_path = os.path.join(args.output_folder, "duration_perturbations.csv")
    merged_df = pd.concat([
        pd.read_csv(os.path.join(args.output_folder, "silence_perturbations.csv")),
        pd.read_csv(os.path.join(args.output_folder, "order_perturbations.csv")),
        pd.read_csv(os.path.join(args.output_folder, "volume_perturbations.csv"))
    ], ignore_index=True)

    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged CSV saved to: {merged_csv_path}")
