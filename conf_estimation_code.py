import numpy as np
import csv
import argparse

def compute_ece(probabilities, labels, num_bins=10, squared=True, normalization=False, return_mce=False):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probabilities > bin_lower) & (probabilities <= bin_upper)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_accuracy = np.mean(labels[mask])
            bin_confidence = np.mean(probabilities[mask])
            bin_weight = bin_count / len(probabilities)
            mce = max(mce, np.abs(bin_confidence - bin_accuracy))
            if squared and not normalization:
                ece += ((bin_confidence - bin_accuracy) ** 2) * bin_weight
            elif not squared and not normalization:
                ece += np.abs(bin_confidence - bin_accuracy) * bin_weight
            elif not squared and normalization:
                ece += np.abs(bin_confidence - bin_accuracy) * bin_weight * 1 / bin_confidence
            else:
                ece += ((bin_confidence - bin_accuracy) ** 2) * bin_weight * 1 / (bin_confidence ** 2)

    return (ece, mce) if return_mce else ece

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row['idx']] = row['answer']
    return gt

# --- Config ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name used for predictions (e.g., qwen_output)")
parser.add_argument("--task", type=str, required=True, choices=["count", "duration", "order"], help="Task name")
args = parser.parse_args()

csv_path = f"TREA_dataset/{args.task}_perturbed/reworded_{args.task}_perturbations.csv"
perturbed_txt_path = f"confidence_estimation/{args.model}_{args.task}.txt"
unperturbed_txt_path = f"{args.model}_output/{args.task}_vanilla_output.txt"

# --- Load ground truth and original predictions ---
gt_dict = load_ground_truth(csv_path)

no_perturb_answer = {}
with open(unperturbed_txt_path, "r") as orig_txt:
    for line in orig_txt:
        idx, answer = line.strip().split(" ", maxsplit=1)
        no_perturb_answer[idx] = answer

# --- Evaluate perturbed predictions ---
probabilities = []
labels = []
num_toggle = 0
total = 0

with open(perturbed_txt_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        idx, pred_ans, prob = parts
        prob = float(prob)
        correct_ans = gt_dict.get(idx)

        # For ECE
        labels.append(int(pred_ans == correct_ans))
        probabilities.append(prob)

        # For Toggle
        orig_idx = idx.split("_")[0]
        if no_perturb_answer.get(orig_idx) != pred_ans:
            num_toggle += 1
        
        total += 1

# --- Final Metrics ---
probabilities = np.array(probabilities)
labels = np.array(labels)

ece = compute_ece(probabilities, labels, num_bins=10, squared=False, normalization=False)
percent_toggle = (num_toggle / total) * 100

# --- Output ---
print("="*20)
print(f"MODEL: {args.model}")
print(f"TASK: {args.task}")
print("="*20)
print(f"ECE: {ece:.4f}")
print(f"EUE: {percent_toggle:.2f}")
