import json
import os
import sys

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings or lists.
    Pure python implementation to avoid external dependencies.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    """
    Character Error Rate (CER) calculation.
    CER = Levenshtein Distance / Length of Reference
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)

def calculate_wer(reference, hypothesis):
    """
    Word Error Rate (WER) calculation.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
        
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)

def evaluate_samples(data):
    results = []
    total_raw_cer = 0
    total_ft_cer = 0
    total_raw_wer = 0
    total_ft_wer = 0
    
    count = len(data)
    
    if count == 0:
        print("No samples to evaluate.")
        return

    print(f"{'ID':<10} | {'Raw CER':<10} | {'FT CER':<10} | {'Raw WER':<10} | {'FT WER':<10} | {'Improvement (CER)'}")
    print("-" * 80)

    for i, item in enumerate(data):
        id_label = item.get("id", str(i))
        gt = item.get("ground_truth", "")
        raw = item.get("raw", "")
        ft = item.get("finetuned", "")
        
        raw_cer = calculate_cer(gt, raw)
        ft_cer = calculate_cer(gt, ft)
        raw_wer = calculate_wer(gt, raw)
        ft_wer = calculate_wer(gt, ft)
        
        improvement = raw_cer - ft_cer
        
        total_raw_cer += raw_cer
        total_ft_cer += ft_cer
        total_raw_wer += raw_wer
        total_ft_wer += ft_wer
        
        print(f"{id_label:<10} | {raw_cer:>10.2%} | {ft_cer:>10.2%} | {raw_wer:>10.2%} | {ft_wer:>10.2%} | {improvement:>+10.2%}")

    avg_raw_cer = total_raw_cer / count
    avg_ft_cer = total_ft_cer / count
    avg_raw_wer = total_raw_wer / count
    avg_ft_wer = total_ft_wer / count
    
    print("-" * 80)
    print(f"{'AVERAGE':<10} | {avg_raw_cer:>10.2%} | {avg_ft_cer:>10.2%} | {avg_raw_wer:>10.2%} | {avg_ft_wer:>10.2%} | {avg_raw_cer - avg_ft_cer:>+10.2%}")
    
    # Accuracy metric (1 - CER)
    print("\n--- Summary ---")
    print(f"Overall Raw Accuracy (1-CER): {1 - avg_raw_cer:.2%}")
    print(f"Overall Finetuned Accuracy (1-CER): {1 - avg_ft_cer:.2%}")
    print(f"Standard Error Reduction: {(avg_raw_cer - avg_ft_cer) / avg_raw_cer if avg_raw_cer > 0 else 0:.2%}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_evaluator.py <samples.json>")
        print("\nJSON expected format:")
        print('''[
  {
    "id": "1",
    "ground_truth": "Expected text",
    "raw": "Text from raw model",
    "finetuned": "Text from finetuned model"
  }
]''')
        return

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        evaluate_samples(data)
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
