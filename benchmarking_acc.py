import argparse
import csv
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name used for predictions (e.g., qwen_output)")
    parser.add_argument("--method", type=str, choices=["vanilla", "cot", "exp", "audio_desc"], required=True)
    parser.add_argument("--task", type=str, required=True, choices=["count", "duration", "order"], help="Task name")
    args = parser.parse_args()

    # Construct file paths
    pred_results = os.path.join(f"{args.model}_output", f"{args.task}_{args.method}_output.txt")
    actual_results = os.path.join("TREA_dataset", args.task, f"{args.task}.csv")

    # Dictionaries for predicted and actual results
    pred_dict = {}
    actual_dict = {}

    total = 200
    correct = 0

    # Read actual results
    with open(actual_results, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id = row["id"]
            answer = row["correct"].strip()
            actual_dict[id] = answer

    # Read predicted results
    with open(pred_results, "r", encoding='utf-8') as pred:
        possible_ans = ["A", "B", "C", "D"]
        for line in pred:
            line = line.strip("\n")
            id, pred_res = line.split(" ", 1)
            if pred_res[0] in possible_ans:
                if actual_dict[id] == pred_res[0]:
                    correct += 1

    print("="*20)
    print(f"MODEL: {args.model}")
    print(f"TASK: {args.task}")
    print(f"METHOD: {args.method}")
    print("="*20)
    print(f"ACCURACY: {(correct / total) * 100:.2f}%")

if __name__ == "__main__":
    main()
