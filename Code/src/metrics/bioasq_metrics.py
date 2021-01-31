""" ----------- BIOASQ EVALUATION METRICS -----------
BioASQ metrics differ for different question types:
    - yes/no questions (produce a single yes or no answer per question)
        - accuracy -> correct answers / total questions
        - precision and recall
        - f1_score -> measured independently for "yes" (f1_y) and "no" (f1_n) answers
        - maF1 -> macro averaged f-measure, ((f1_y) + (f1_n)) / 2

    - factoid questions (produce a list of candidate answers per question)
        - strict accuracy -> question correct if first element in list matches expected answer
        - lenient accuracy -> question correct if any element in list matches expected answer
        - MRR (official eval metric)

    - list questions (produce a list of answers per question - up to 100)
        - for every answer list, compute precision, recall and f measure
        - compute mean avg precision, recall and f-measure (official eval metric)
"""


# --------- METRICS FOR YES/NO QUESTIONS ---------
def yes_no_evaluation(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        # not enough labels to match
        raise Exception(
            "There are {} predictions and {} ground truth values.".format(len(predictions), len(ground_truth)))

    true_yes = {"yes", "y", 1}
    true_no = {"no", "n", 0}

    tp, fp, tn, fn = 0, 0, 0, 0
    for idx, prediction in enumerate(predictions):  # for every prediction

        # get the corresponding ground truth label
        truth_value = ground_truth[idx]

        # ---------- Perform checks on the data first ----------
        if type(prediction) != type(truth_value):
            raise Exception(
                "Cannot compare prediction and ground truth label of type {} and {}".format(type(prediction),
                                                                                            type(truth_value)))

        if prediction not in true_yes or prediction not in true_no:
            raise Exception("Prediction of the form {} is not a valid yes or no response."
                            "Expected one of {} -> (yes) or {} -> (no)".format(prediction, true_yes, true_no))
        if truth_value not in true_no or truth_value not in true_no:
            raise Exception("Prediction of the form {} is not a valid yes or no response."
                            "Expected one of {} -> (yes) or {} -> (no)".format(truth_value, true_yes, true_no))

        # --- Evaluate answer ---
        if prediction in true_yes:  # predicted answer was yes
            if prediction == truth_value:   # ground truth answer was yes (true positive)
                tp += 1
            else:   # ground truth answer was no (false positive)
                fp += 1
        else:  # predicted answer was no
            if prediction == truth_value:   # ground truth answer was no (true negative)
                fp += 1
            else:   # ground truth answer was yes (false negative)
                fn += 1

    accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
    precision_y = round(tp / (tp + fp), 3)
    recall_y = round(tp / (tp + fn), 3)
    precision_n = round(tp / (tp + fp), 3)
    recall_n = round(tp / (tp + fn), 3)

    f1_y = round(2 * ((precision_y * recall_y) / (precision_y + recall_y)), 3)
    f1_n = round(2 * ((precision_n * recall_n) / (precision_n + recall_n)), 3)
    f1_ma = round((f1_y + f1_n) / 2, 3)

    metrics = {
        "accuracy": accuracy,
        "precision": precision_y,
        "recall": recall_y,
        "f1_y": f1_y,
        "f1_n": f1_n,
        "f1_ma": f1_ma,
    }
    return metrics

# --------- METRICS FOR LIST QUESTIONS ---------
