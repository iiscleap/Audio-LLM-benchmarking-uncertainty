# Audio-LLM-benchmarking-uncertainty

Official Implementation of the Interspeech, 2025 paper: **Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning**

**Paper link:** [https://arxiv.org/pdf/2505.13115](https://arxiv.org/pdf/2505.13115)

---

## 1. Generating perturbations

### 1.1 Clone the Repository

```bash
git clone https://github.com/iiscleap/Audio-LLM-benchmarking-uncertainty
cd Audio-LLM-benchmarking-uncertainty/
```

### 1.2 Download Dataset

Download the ESC-50 dataset from: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
Unzip the ESC-50 dataset.

### 1.3 Generate Audio Perturbations (Count Task)

```bash
python generate_count_perturb.py \
  --input_csv TREA_dataset/count/count_with_metadata.csv \
  --output_folder TREA_dataset/count_perturbed \
  --num_samples 15 \
  --num_samples_per_type 5
```

* `input_csv`: count task metadata CSV file
* `output_folder`: folder to save output CSV and audios
* `num_samples`: number of questions randomly sampled
* `num_samples_per_type`: number of perturbations per type

Generates 5 volume, 5 pitch, and 5 add/delete event perturbations per original sample (15 samples).
Total = 225 audio perturbations.

Repeat similarly for `duration` and `order` tasks.

### 1.4 Generate Text Perturbations

```bash
python generate_text_perturb.py \
 --input_csv TREA_dataset/count_perturbed/count_perturbations.csv \
 --output_folder TREA_dataset/count_perturbed
```

Output: `reworded_count_perturbations.csv` with 4 text perturbations per original sample.

---

## 2. Benchmarking

### 2.1 Qwen2-Audio

Download model and dependencies: [https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)

Run benchmarking:

```bash
python qwen_benchmarking.py \
  --method vanilla \
  --task count \
  --csv_path TREA_dataset/count/count.csv
```

* `method`: `vanilla`, `cot`, `exp`, or `audio_desc`
* `task`: `count`, `duration`, or `order`

Each line of the output txt file contains <idx> <output> <probability>

### 2.2 SALMONN

Setup SALMONN: [https://github.com/bytedance/SALMONN](https://github.com/bytedance/SALMONN)

Move `salmonn_benchmarking.py` to the SALMONN repo.
Resample audio files in `TREA_dataset/<task>/audios` to 16kHz and save as `TREA_dataset/<task>/<task>_resampled`

Run benchmarking:

```bash
python salmonn_benchmarking.py \
  --method <method_name> \
  --task <task_name> \
  --csv_path <path_to_csv_file> \
  --wav_folder <path_to_resampled_audio_folder>
```

### 2.3 Compute Accuracy

```bash
python benchmarking_acc.py --model qwen --task count --method vanilla
```

---

## 3. Confidence Estimation

### 3.1 Qwen2-Audio

```bash
python qwen_uncertainty.py \
  --task count \
  --csv_path /path/to/your_perturbed_input.csv
```

Output saved to: `confidence_estimation/qwen_count.txt`
Each line of the output txt file contains <idx> <output> <probability>

### 3.2 SALMONN

Replace `salmonn.py` in `SALMONN/models` with the provided version.
Move `salmonn_uncertainty.py` to SALMONN/

Run:

```bash
python salmonn_uncertainty.py \
  --task count \
  --csv_path /path/to/input.csv \
  --wav_folder /path/to/resampled_perturbed_audio_files/
```

### 3.3 SALMONN-based Audio Description + LLM Answer

Move `desc_llm_uncertainty.py` to SALMONN/

```bash
python desc_llm_uncertainty.py \
  --task count \
  --csv_path /path/to/input.csv \
  --wav_folder /path/to/resampled_perturbed_audio_files/
```

### 3.4 ECE and EUE Calculation

```bash
python conf_estimation_code.py --task count --model qwen
```
