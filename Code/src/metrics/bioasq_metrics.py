from .helper_metrics import check_match

""" ----------- BIOASQ EVALUATION METRICS -----------
In the BioASQ challenge, task b phase b presents three different question types.
The questions are provided with a label marking their type.

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

    # print('predictions', predictions)
    true_yes = {"yes", "y", 1}
    true_no = {"no", "n", 0}

    tp, fp, tn, fn = 0, 0, 0, 0
    for idx, prediction in enumerate(predictions):  # for every prediction

        # get the corresponding ground truth label
        truth_value = ground_truth[idx].lower()

        # ---------- Perform checks on the data first ----------
        if type(prediction) != type(truth_value):
            raise Exception(
                "Cannot compare prediction and ground truth label of type {} and {}".format(type(prediction),
                                                                                            type(truth_value)))

        if prediction not in true_yes and prediction not in true_no:
            raise Exception("Prediction of the form {} is not a valid yes or no response. "
                            "Expected one of {} -> (yes) or {} -> (no)".format(prediction, true_yes, true_no))
        if truth_value not in true_yes and truth_value not in true_no:
            raise Exception("Prediction of the form {} is not a valid yes or no response. "
                            "Expected one of {} -> (yes) or {} -> (no)".format(truth_value, true_yes, true_no))

        # --- Evaluate answer ---
        if prediction in true_yes:  # predicted answer was yes
            if prediction == truth_value:   # ground truth answer was yes as well (true positive)
                tp += 1
            else:   # ground truth answer was no (false positive)
                fp += 1
        else:  # predicted answer was no
            if prediction == truth_value:   # ground truth answer was also no (true negative)
                tn += 1
            else:   # ground truth answer was yes (false negative)
                fn += 1

    accuracy = 0 if (tp + tn + fp + fn) == 0 else round((tp + tn) / (tp + tn + fp + fn), 3)
    precision_y = 0 if (tp + fp) == 0 else round(tp / (tp + fp), 3)
    recall_y = 0 if (tp + fn) == 0 else round(tp / (tp + fn), 3)
    precision_n = 0 if (tn + fn) == 0 else round(tn / (tn + fn), 3)
    recall_n = 0 if (tn + fp) == 0 else round(tn / (tn + fp), 3)

    try:  # try this with the caveat that precision_y + recall_y could be zero
        f1_y = round(2 * ((precision_y * recall_y) / (precision_y + recall_y)), 3)
    except ZeroDivisionError:
        f1_y = 0

    try:  # try this with the caveat that precision_n + recall_n could be zero
        f1_n = round(2 * ((precision_n * recall_n) / (precision_n + recall_n)), 3)
    except ZeroDivisionError:
        f1_n = 0

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


# --------- METRICS FOR FACTOID QUESTIONS ---------
def factoid_evaluation(predictions, ground_truths):
    if len(predictions) != len(ground_truths):
        # not enough labels to match
        raise Exception(
            "There are {} predictions and {} ground truth values.".format(len(predictions), len(ground_truths)))

    total_questions = len(predictions)
    correct_in_position_1 = 0
    correct_in_any_position = 0
    total_one_over_ri = 0

    for idx, prediction in enumerate(predictions):  # for every prediction
        # We expect each prediction to be a list of candidate answers,
        # however, the ground truth answer should be a single string.

        # get the corresponding ground truth label(s)
        truth_values = ground_truths[idx]  # get the list of correct answers

        match_position = None
        for list_pos, candidate_answer in enumerate(prediction):
            match = check_match(candidate_answer, truth_values)

            if match:
                correct_in_any_position += 1
                if list_pos == 0:  # if the first position is a match
                    correct_in_position_1 += 1
                match_position = list_pos + 1  # best score is meant to be in position 1
                break

        # 0 if no match was found, else 1 / r(i)
        one_over_ri = 0 if match_position is None else 1 / match_position
        total_one_over_ri += one_over_ri

    strict_accuracy = 0 if total_questions == 0 else round(correct_in_position_1 / total_questions, 3)
    leniant_accuracy = 0 if total_questions == 0 else round(correct_in_any_position / total_questions, 3)
    mrr = 0 if total_one_over_ri == 0 or total_questions == 0 else round((1 / total_questions) * total_one_over_ri, 3)

    metrics = {
        "strict_accuracy": strict_accuracy,
        "leniant_accuracy": leniant_accuracy,
        "mrr": mrr,
    }
    return metrics


# --------- METRICS FOR LIST QUESTIONS ---------
def list_evaluation(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        # not enough labels to match
        raise Exception(
            "There are {} predictions and {} ground truth values.".format(len(predictions), len(ground_truth)))

    """
        - list questions (produce a list of answers per question - up to 100)
            - for every answer list, compute precision, recall and f measure
            - compute mean avg precision, recall and f-measure (official eval metric)
    """

    total_questions = len(predictions)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for idx, prediction in enumerate(predictions):  # for every prediction
        print('Prediction (line 160):', prediction)

        # We expect each prediction to be a list of candidate answers,
        # The ground truth answer should also be a list of answers.
        truth_values = ground_truth[idx]  # get the corresponding ground truth answers

        # tns do not make sense in this case, as they are entities not in either list.
        tp, fn = 0, 0

        for truth_value in truth_values:  # for every truth value
            # search for a matching answer in candidate list
            match_found = False
            for candidate_answer in prediction:
                match = check_match(candidate_answer, truth_value)

                if match:  # successfully found a match
                    match_found = True
                    break

            # tp are entities that are in both lists
            tp += 1 if match_found else 0  # if a match is found between truth value and candidate answer
            # fn are entities in truth_values but not in prediction
            fn += 1 if not match_found else 0

        # fp are entities in prediction but not in truth_values
        fp = len(prediction) - tp

        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)  # save the rounding for avg calculation
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0 if (precision + recall) == 0 else 2 * ((precision * recall) / (precision + recall))

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    mean_average_precision = 0 if total_questions == 0 else round(total_precision / total_questions, 3)
    mean_average_recall = 0 if total_questions == 0 else round(total_recall / total_questions, 3)
    mean_average_f1 = 0 if total_questions == 0 else round(total_f1 / total_questions, 3)

    metrics = {
        "mean_average_precision": mean_average_precision,
        "mean_average_recall": mean_average_recall,
        "mean_average_f1": mean_average_f1,
    }
    return metrics