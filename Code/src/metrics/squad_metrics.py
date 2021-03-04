from .helper_metrics import check_match

""" ----------- SQUAD EVALUATION METRICS -----------
There is a single question type in the squad dataset

- F1 Score - Measure of avg overlap between prediction and ground-truth answer span
- Exact Match - Takes a value of 1 if predicted answer matches true answer exactly, otherwise 0

Exact match does not take punctuation and articles into account.
"""


# --------- HELPER METRICS ---------
def compute_f1_from_span(predicted_span, expected_span):
    """
    This metric measures the average overlap between the prediction and
    ground truth answer. We treat the prediction and ground truth as bags
    of tokens, and compute their F1. We take the maximum F1 over all of
    the ground truth answers for a given question, and then average
    over all of the questions.
    :return:
    """

    split_predicted = predicted_span.split(" ")
    split_expected = expected_span.split(" ")

    # tns do not make sense in this case, as they are entities not in either list.
    fn, tp = 0, 0
    for expected_token in split_expected:

        # search for a matching answer in candidate list
        match_found = False
        for predicted_token in split_predicted:
            match = check_match(predicted_token, expected_token)

            if match:  # successfully found a match
                match_found = True
                break

        # tp are entities that are in both lists
        tp += 1 if match_found else 0
        # fn are entities in split_expected but not in split_predicted
        fn += 1 if not match_found else 0

    # fp are entities in split_predicted but not in split_expected
    fp = len(split_predicted) - tp

    precision = tp / (tp + fp)  # save the rounding for avg calculation
    recall = tp / (tp + fn)
    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0

    return f1


def compute_em_from_span(predicted_span, expected_span):
    """
    Compute exact match from answer span. This assumes that only
    one answer exists.
    :return: boolean indicating if span is an exact match
    """
    return 1 if check_match(predicted_span, expected_span) else 0


def squad_evaluation(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        # not enough labels to match
        raise Exception(
            "There are {} predictions and {} ground truth values.".format(len(predictions), len(ground_truth)))

    total_questions = len(predictions)

    all_f1 = 0
    all_exact_match = 0

    for idx, prediction in enumerate(predictions):  # for every prediction
        # We expect each prediction to be a list of candidate answers,
        # however, the ground truth answer should be a single string.

        # get the corresponding ground truth label
        truth_values = ground_truth[idx]
        truth_value = truth_values[0]  # take the first, and most likely prediction for squad.

        max_exact_match = -float('inf')
        max_f1 = -float('inf')

        for candidate_answer_span in prediction:
            exact_match = compute_em_from_span(candidate_answer_span, truth_value)
            if exact_match > max_exact_match:
                max_exact_match = exact_match

            f1 = compute_f1_from_span(candidate_answer_span, truth_value)
            if f1 > max_f1:
                max_f1 = f1

        if max_exact_match > -float('inf') and max_f1 > -float('inf'):
            all_exact_match += max_exact_match
            all_f1 += max_f1

    avg_exact_match = round(all_exact_match / total_questions, 2)
    avg_f1 = round(all_f1 / total_questions, 2)

    metrics = {
        "exact_match": avg_exact_match,
        "f1": avg_f1,
    }
    return metrics