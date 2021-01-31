from helper_metrics import check_match

""" ----------- SQUAD EVALUATION METRICS -----------
There is a single question type in the squad dataset

- F1 Score - Measure of avg overlap between prediction and ground-truth answer span
- Exact Match - Takes a value of 1 if predicted answer matches true answer exactly, otherwise 0

Exact match does not take punctuation and articles into account.
"""

