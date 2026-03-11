import os
import csv
from datetime import datetime

def save_metrics(filename: str, seed: int, acc: float, prec: float, recall: float, f1: float, mcc: float, note: str = "", output_csv: str = "./results/results.csv"):
    """
    Saves model evaluation metrics to a CSV file. 
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_csv)
    
    # Open in append mode ('a') so we don't overwrite previous runs
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write headers if it's a brand new file
        if not file_exists:
            writer.writerow([
                "timestamp", 
                "filename", 
                "seed", 
                "accuracy", 
                "precision", 
                "recall", 
                "f1_score", 
                "mcc",
                "note"
            ])
        
        # Write the actual metrics, formatting floats to 4 decimal places
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            seed,
            f"{acc:.4f}",
            f"{prec:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{mcc:.4f}",
            note
        ])
        
    print(f"\n Results successfully appended to {output_csv}")