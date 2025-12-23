import json
import os
import sys
import pandas as pd

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
    comparison_data = []
    total_raw_cer = 0
    total_ft_cer = 0
    total_raw_wer = 0
    total_ft_wer = 0
    count = len(data)

    if count == 0:
        print("No samples to evaluate.")
        return

    # Collect data for the table
    table_rows = []
    for i, sample in enumerate(data):
        # Support both 'ground_truth' (existing) and 'text' (from snippet)
        gt = sample.get("ground_truth", sample.get("text", ""))
        raw = sample.get("raw", "")
        ft = sample.get("finetuned", "")
        lang = sample.get("language", "unk")
        id_val = sample.get("id", str(i))

        raw_cer = calculate_cer(gt, raw)
        ft_cer = calculate_cer(gt, ft)
        raw_wer = calculate_wer(gt, raw)
        ft_wer = calculate_wer(gt, ft)

        comparison_data.append({
            "id": id_val,
            "language": lang,
            "ground_truth": gt,
            "raw": raw,
            "finetuned": ft
        })

        total_raw_cer += raw_cer
        total_ft_cer += ft_cer
        total_raw_wer += raw_wer
        total_ft_wer += ft_wer

        # Add to table if sampled
        if count <= 50 or i % (count // 20) == 0 or i == count - 1:
            improvement = raw_cer - ft_cer
            table_rows.append({
                "ID": id_val,
                "Language": lang,
                "Raw CER": f"{raw_cer:.2%}",
                "FT CER": f"{ft_cer:.2%}",
                "Raw WER": f"{raw_wer:.2%}",
                "FT WER": f"{ft_wer:.2%}",
                "Improvement": f"{improvement:+.2%}"
            })

    # Calculate averages
    avg_raw_cer = total_raw_cer / count
    avg_ft_cer = total_ft_cer / count
    avg_raw_wer = total_raw_wer / count
    avg_ft_wer = total_ft_wer / count

    # Add separator and average row to the table
    if table_rows:
        table_rows.append({
            "ID": "─" * 10,
            "Language": "─" * 8,
            "Raw CER": "─" * 8,
            "FT CER": "─" * 8,
            "Raw WER": "─" * 8,
            "FT WER": "─" * 8,
            "Improvement": "─" * 10
        })
        
        table_rows.append({
            "ID": "AVERAGE",
            "Language": "",
            "Raw CER": f"{avg_raw_cer:.2%}",
            "FT CER": f"{avg_ft_cer:.2%}",
            "Raw WER": f"{avg_raw_wer:.2%}",
            "FT WER": f"{avg_ft_wer:.2%}",
            "Improvement": f"{avg_raw_cer - avg_ft_cer:+.2%}"
        })

    # Create and display the DataFrame
    if table_rows:
        df = pd.DataFrame(table_rows)
        
        # Display with better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        # Use display if available (Jupyter), otherwise print
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df.to_string(index=False))

    # Print the summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Overall Raw CER: {avg_raw_cer:.2%}")
    print(f"Overall Finetuned CER: {avg_ft_cer:.2%}")
    print(f"Overall Raw WER: {avg_raw_wer:.2%}")
    print(f"Overall Finetuned WER: {avg_ft_wer:.2%}")
    print("-"*60)
    print(f"Overall Raw Accuracy (1-CER): {1 - avg_raw_cer:.2%}")
    print(f"Overall Finetuned Accuracy (1-CER): {1 - avg_ft_cer:.2%}")
    print(f"CER Improvement: {avg_raw_cer - avg_ft_cer:+.2%}")
    print(f"WER Improvement: {avg_raw_wer - avg_ft_wer:+.2%}")
    print(f"Relative CER Reduction: {(avg_raw_cer - avg_ft_cer) / avg_raw_cer if avg_raw_cer > 0 else 0:.1%}")

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
