"""
Shared I/O and table formatting utilities for MIND-Crypt experiments.
"""

import csv
import json
import os
from typing import Any, Dict, List


def save_results(rows: List[Dict[str, Any]], json_path: str, csv_path: str) -> None:
    """Persist structured experiment records as both JSON and CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, 'w') as jf:
        json.dump(rows, jf, indent=2)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def print_table(title: str, headers: List[str], rows: List[List[Any]]) -> None:
    """Pretty-print an ASCII table with aligned columns."""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print(f"\n{title}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)) + " |")
    print(sep)
