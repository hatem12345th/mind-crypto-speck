"""
Shared I/O and table formatting utilities for MIND-Crypt experiments.
"""

import csv
import json
import math
import os
import statistics
from typing import Any, Dict, List, Sequence


def accuracy_significance(accuracy: float, n_test: int, z: float = 1.96) -> Dict[str, Any]:
    """
    Statistical context for a test accuracy measured on n_test samples.

    Returns the Wilson score 95% confidence interval for the accuracy and the
    two-sided p-value of a binomial test against the 0.5 chance baseline
    (normal approximation, accurate for the n_test >= 1,000 used here).
    """
    if n_test <= 0:
        raise ValueError(f"n_test must be positive, got {n_test}")
    acc = float(accuracy)
    denom = 1.0 + z * z / n_test
    centre = (acc + z * z / (2 * n_test)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n_test + z * z / (4 * n_test * n_test)) / denom
    z_stat = (acc - 0.5) / math.sqrt(0.25 / n_test)
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))
    return {
        'n_test': int(n_test),
        'ci95_low': round(centre - half, 4),
        'ci95_high': round(centre + half, 4),
        'p_vs_chance': float(f"{p_value:.3g}"),
    }


def annotate_significance(rows: List[Dict[str, Any]], n_test: int) -> List[Dict[str, Any]]:
    """Add n_test, Wilson 95% CI and p-value vs chance to every record (in place)."""
    for row in rows:
        row.update(accuracy_significance(row['accuracy'], n_test))
    return rows


METRIC_KEYS = (
    'accuracy', 'tpr', 'tnr', 'loss', 'roc_auc', 'ci95_low', 'ci95_high',
    'p_vs_chance', 'seconds', 'fit_seconds', 'cv_best_score', 'seed',
)


def summarize_over_seeds(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate records from repeated runs (one record per seed and setting).

    Records are grouped by every non-metric field (table, rounds, keys, sizes, ...)
    and the accuracy is summarised as mean, sample std, min and max across seeds.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple((k, v) for k, v in row.items() if k not in METRIC_KEYS)
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for key, members in groups.items():
        accs = [float(m['accuracy']) for m in members]
        rec: Dict[str, Any] = dict(key)
        rec['n_seeds'] = len(accs)
        rec['seeds'] = ' '.join(str(m.get('seed', '?')) for m in members)
        rec['acc_mean'] = round(statistics.fmean(accs), 4)
        rec['acc_std'] = round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0
        rec['acc_min'] = round(min(accs), 4)
        rec['acc_max'] = round(max(accs), 4)
        if all('roc_auc' in m for m in members):
            rec['auc_mean'] = round(statistics.fmean(float(m['roc_auc']) for m in members), 4)
        summary.append(rec)
    return summary


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
