from tqdm import tqdm
from models import *
from data_processing import *
import string
from metrics.bioasq_metrics import yes_no_evaluation, factoid_evaluation, list_evaluation
from metrics.squad_metrics import squad_evaluation

word_nums = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]


def combine_tokens(token_list: list) -> str:
    build_string = []

    for token in token_list:
        if '#' not in token and token not in string.punctuation:  # the start of a word and not punctuation
            build_string.append(" " + token)
        else:
            raw_token = token.strip().lstrip('#')
            build_string.append(raw_token)

    return ''.join(build_string).strip()


def contains_k(text: str):
    # Given a piece of text, check if text contains k. e.g. give two examples of...
    # split on whitespace
    for word in text.split():
        if word in word_nums:
            return word_nums.index(word)
        if word.isdigit():
            return int(word)
    return None

"""
Refer to the following code
https://huggingface.co/transformers/task_summary.html#extractive-question-answering
"""
def evaluate_yesno(yes_no_model, test_dataloader, training=False, dataset="bioasq"):
    """
    Given a model and a test-set, evaluate the model's performance on the test set.
    The evaluation metrics we use depend on the dataset we use. Although, currently for
    yes/no questions, we only have one evaluation dataset (bioasq).
    For this question type, we do not expect to see any impossible (unanswerable) questions.
    However, we do expect to get questions without an answer, answer-start and answer-end.
    These are given when we use a non-golden bioasq test set, so we can't actually evaluate them.
    :param yes_no_model: pytorch model trained on yes no (binary classification) questions.
    :param test_dataloader: a dataloader iterable containing the yes/no test data.
    :param training: flag indicating whether we are running this from finetuning or evaluation.
    :param dataset: the dataset we are using determines the metrics we are using
    """

    results_by_question_id = {}  # initialise an empty dictionary for storing results by question id

    # when .eval() is set, all dropout layers are removed.
    yes_no_model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for eval_step, batch in enumerate(tqdm(test_dataloader, desc="Step")):
            inputs = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "token_type_ids": batch.token_type_ids,
            }

            outputs = yes_no_model(**inputs)  # model outputs are always tuples in transformers
            try:  # indexing outputs on CPU
                logits = outputs.logits
            except AttributeError:  # indexing outputs on CUDA
                logits = outputs[0]

            # print('logits', logits)
            # treat outputs as if they correspond to a yes/no question
            # dim=1 makes sure we produce an answer start for each x in batch
            class_probabilities = torch.softmax(logits, dim=1)
            # print('classes', class_probabilities)

            for question_idx, question_id in enumerate(batch.question_ids):
                # Note: expected answers could be None here if we don't have access to the answer.
                # We still need to include them in our dictionary, and filter them at the next step.
                expected_answer = batch.answer_text[question_idx]
                predicted_label = torch.argmax(class_probabilities[question_idx])
                # print('predicted label', predicted_label)
                if question_idx in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(predicted_label)
                else:
                    results_by_question_id[question_id] = {"predictions": [predicted_label],
                                                           "expected_answer": expected_answer}
    yes_no_model.train()  # back to train mode.

    # Iterate through predictions for each question. We need to combine these predictions to produce a final prediction.
    # create a list of predictions and a list of ground_truth for evaluation
    predictions_list, ground_truth_list = [], []
    for q_id in results_by_question_id:
        # results_by_question_id[q_id]["predictions"] is a list of scalar tensors e.g. [tensor(1), tensor(2)]
        pred_tensor = torch.Tensor(results_by_question_id[q_id]["predictions"])
        # print('pred tensor', pred_tensor)

        best_pred = torch.mode(pred_tensor, 0).values  # get the most common value in the prediction tensor
        # batch labels are varied
        predicted_answer = "yes" if best_pred == 1 else "no"  # convert 1s to yes and 0s to no
        results_by_question_id[q_id]["predictions"] = predicted_answer

        # We need to ensure that we don't try to evaluate the questions that don't have expected answers.
        if results_by_question_id[q_id]["expected_answer"] is not None:  # i.e. we have an answer
            predictions_list.append(predicted_answer)
            ground_truth_list.append(results_by_question_id[q_id]["expected_answer"])

    if dataset == "bioasq" or dataset == "boolq":  # Deploy the bioasq metrics on our results.
        # Evaluation metrics are empty if we didn't have any expected answers.
        if any(ground_truth_list):  # if any of the ground truth values are not None
            eval_metrics = {} if len(predictions_list) == 0 else yes_no_evaluation(predictions_list, ground_truth_list)
        else:
            eval_metrics = {}

        if training:
            return eval_metrics
        # return a dictionary of {question_id: prediction (i.e. "yes" or "no")}
        return results_by_question_id, eval_metrics
    raise Exception("Dataset name provided to evaluate_yesno must be 'bioasq' or 'boolq', "
                    "as no other datasets are handled at this time.")


def evaluate_factoid(factoid_model, test_dataloader, tokenizer, training=False, dataset="bioasq"):
    """
    Given an extractive question answering model and some test data, perform evaluation.
    :param factoid_model: Pytorch model trained on factoid (and list) questions
    :param test_dataloader: Dataloader iterable containing test data on which to evaluate
    :param tokenizer: The tokenizer used for converting text to tokens.
    :param k: Top k (best) predictions are chosen for factoid questions.
    :param training: Flag indicating whether we are calling this from fine-tuning or evaluation
    :param dataset: Name of the dataset we are using determines the metrics we are using.
    """
    results_by_question_id = {}
    special_tokens_ids = {tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

    factoid_model.eval()  # switch to evaluation mode
    with torch.no_grad():
        for eval_step, batch in enumerate(tqdm(test_dataloader, desc="Step")):
            # question_ids = batch.question_ids

            inputs = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "token_type_ids": batch.token_type_ids,
            }

            # model outputs are always tuples in transformers
            outputs = factoid_model(**inputs)
            try:  # indexing outputs on CPU
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
            except AttributeError:  # indexing outputs on CUDA
                start_logits, end_logits = outputs[0], outputs[1]

            # Each list in the tensor relates to a single question - and these are likelihood values (probabilities)
            # e.g. start_logits = tensor([[0.0943, 0.1020, 0.1816, ..., 0.0474, 0.1435, 0.0335], ...],)

            # Convert the batches of start and end logits into answer span position predictions
            # This will give us k predictions per feature in the batch
            answer_starts, start_indices = torch.topk(start_logits, k=5, dim=1)
            answer_ends, end_indices = torch.topk(end_logits, k=5, dim=1)

            start_end_positions = list(zip(start_indices, end_indices))
            start_end_probabilities = list(zip(answer_starts, answer_ends))

            # print('start_end_positions', start_end_positions)
            # print('start_end_probabilities', start_end_probabilities)

            # start_end_positions_and_probabilities.sort(key=lambda val: sum(val[0][0], val[0][1]))

            # iterate over our pairs of start and end indices - each loop represents a new question
            for index, (starts_tensor, ends_tensor) in enumerate(start_end_positions):
                probabilities_of_starts, probabilities_of_ends = start_end_probabilities[index]
                sub_start_end_positions = zip(starts_tensor, ends_tensor)  # zip the start and end positions
                sub_start_end_probabilities = list(zip(probabilities_of_starts, probabilities_of_ends))

                # get info about the current question
                input_ids = batch.input_ids[index]  # get the input ids for the particular question we're looking at
                expected_answer = batch.answer_text[index]  # Note: this will could be None for BioASQ test batches
                question_id = batch.question_ids[index]

                list_of_predictions = []  # gather all of the predictions for this question
                for sub_index, (s, e) in enumerate(sub_start_end_positions):  # convert the start and end positions to answers.
                    # get the probabilities associated with this prediction
                    probability_of_start, probability_of_end = sub_start_end_probabilities[sub_index]

                    if e <= s:  # if end position is less than or equal to start position, skip this pair
                        continue

                    clipped_ids = [t for t in input_ids[int(s):int(e)] if t not in special_tokens_ids]
                    clipped_tokens = tokenizer.convert_ids_to_tokens(clipped_ids, skip_special_tokens=True)
                    predicted_answer = tokenizer.combine_tokens(clipped_tokens)
                    # todo we need a way to do this that handles punctuation better

                    # put our prediction in the list, alongside the probabilities (pred, start_prob + end_prob)
                    # if neither start probability or end probability are negative
                    if probability_of_start > 0 and probability_of_end > 0:
                        list_of_predictions.append((predicted_answer, probability_of_start.item() + probability_of_end.item()))

                if question_id in results_by_question_id:
                    # todo we're modifiying this from a list of lists to extending the list
                    results_by_question_id[question_id]["predictions"].extend(list_of_predictions)

                    if type(expected_answer) == list:
                        # make sure we don't put the same expected answer in the list over and over again.
                        if expected_answer not in results_by_question_id[question_id]["expected_answers"]:
                            results_by_question_id[question_id]["expected_answers"].append(expected_answer)
                else:
                    results_by_question_id[question_id] = {"predictions": list_of_predictions,
                                                           "expected_answers": [expected_answer]}

    # group together the most likely predictions. (i.e. corresponding positions in prediction lists)
    predictions_list, ground_truth_list = [], []
    for q_id in results_by_question_id:  # Gather all predictions for a particular question
        # results_by_question_id[q_id]["predictions"] is a list of lists
        # we get a nested structure, where each sub-list is the pos pred for an example, sorted by most to least likely
        pred_lists = results_by_question_id[q_id]["predictions"]
        # get all of our predictions for this question, sort by the sum of start and end probabilities
        pred_lists.sort(key=lambda val: val[1], reverse=True)
        # print('prediction lists', pred_lists)

        # For each factoid question in BioASQ, each participating system will have to return a list of up to 5 entity names
        # (e.g., up to 5 names of drugs), numbers, or similar short expressions, ordered by decreasing confidence.
        k = 5 if dataset == "bioasq" else 1

        # pred_lists[: min(len(pred_lists), k)]  # take up to k of the best predictions
        best_predictions = []
        num_best_predictions = 0
        for pred, probability in pred_lists:
            if num_best_predictions >= k:
                break

            # don't put repeats in our list.
            if pred not in best_predictions:
                num_best_predictions += 1
                best_predictions.append(pred)

        # swap the huge list of all predictions for our short-list of best predictions
        results_by_question_id[q_id]["predictions"] = best_predictions

        # We need to ensure that we don't try to evaluate the questions that don't have expected answers.
        # If either of the below conditions are true, i.e. we have at least one valid
        if len(results_by_question_id[q_id]["expected_answers"]) > 1 or results_by_question_id[q_id]["expected_answers"][0] is not None:
            # predictions_list.append(predicted_answer)
            predictions_list.append(best_predictions)
            ground_truth_list.append(results_by_question_id[q_id]["expected_answers"])

    for i in range(len(predictions_list)):
        print('expected answers', ground_truth_list[i])
        print('predictions', predictions_list[i])

    if any(ground_truth_list):  # if any of the ground truth values are not None
        if dataset == "bioasq":
            evaluation_metrics = factoid_evaluation(predictions_list, ground_truth_list)
        elif dataset == "squad":
            # this should be able to handle lists of lists of ground truth values.
            evaluation_metrics = squad_evaluation(predictions_list, ground_truth_list)
        else:
            raise Exception('Only squad and bioasq are acceptable dataset names to be passed to evaluation functions.')
    else:
        evaluation_metrics = {}

    if training:
        return evaluation_metrics
    else:
        return results_by_question_id, evaluation_metrics


def evaluate_list(list_model, test_dataloader, tokenizer, training=False, dataset="bioasq"):
    """
    Given a pytorch model trained on factoid / list questions, we need to evaluate this model on a given dataset.
    The evaluation metrics we choose are dependent on our choice of dataset.
    :param list_model: Pytorch model capable of answering factoid questions
    :param test_dataloader: Dataloader containing evaluation data
    :param tokenizer: Tokenizer used to convert strings into tokens
    :param k: The number of predictions to harness from each example.
    :param dataset:
    :return:
    """
    results_by_question_id = {}
    special_tokens_ids = {tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

    list_model.eval()
    with torch.no_grad():
        for eval_step, batch in enumerate(tqdm(test_dataloader, desc="Evaluation Step")):
            inputs = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "token_type_ids": batch.token_type_ids,
            }

            # model outputs are always tuples in transformers
            outputs = list_model(**inputs)
            try:  # indexing outputs on CPU
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
            except AttributeError:  # indexing outputs on CUDA
                start_logits, end_logits = outputs[0], outputs[1]

            # Each list in the tensor relates to a single question - and these are likelihood values (probabilities)
            # e.g. start_logits = tensor([[0.0943, 0.1020, 0.1816, ..., 0.0474, 0.1435, 0.0335], ...],)

            # Convert the batches of start and end logits into answer span position predictions
            # This will give us k predictions per feature in the batch
            answer_starts, start_indices = torch.topk(start_logits, k=100, dim=1)
            answer_ends, end_indices = torch.topk(end_logits, k=100, dim=1)
            start_end_positions = [x for x in zip(start_indices, end_indices)]
            start_end_probabilities = [x for x in zip(answer_starts, answer_ends)]

            # iterate over our pairs of start and end indices
            for index, (start_tensor, end_tensor) in enumerate(start_end_positions):
                # e.g. start_tensor = tensor([110,  33,  38, 111,  35]), end_tensor = tensor([20,  0, 90, 36, 62])
                sub_start_end_positions = zip(start_tensor, end_tensor)  # zip the start and end positions
                input_ids = batch.input_ids[index]
                expected_answer = batch.answer_text[index]
                question_id = batch.question_ids[index]

                start_probabilities, end_probabilities = start_end_probabilities[index]
                sub_start_end_probabilities = list(zip(start_probabilities, end_probabilities))

                list_of_predictions = []  # gather all of the predictions for this question
                list_of_probability_pairs = []
                for pos_index, (s, e) in enumerate(sub_start_end_positions):  # convert the start and end positions to answers.
                    if e <= s:  # if end position is less than or equal to start position, skip this pair
                        continue

                    if dataset == "bioasq" and e - s > 100: # todo this is tokens not characters
                        continue  # if length is more than 100 and we are evaluating on bioasq, skip this pair

                    clipped_ids = [t for t in input_ids[int(s):int(e)] if t not in special_tokens_ids]
                    clipped_tokens = tokenizer.convert_ids_to_tokens(clipped_ids, skip_special_tokens=True)

                    # make sure we don't end up with special characters in our predicted
                    # todo we need a way to do this that handles punctuation better
                    predicted_answer = tokenizer.combine_tokens(clipped_tokens)

                    s_prob, e_prob = sub_start_end_probabilities[pos_index]
                    list_of_probability_pairs.append((s_prob, e_prob))
                    list_of_predictions.append(predicted_answer)

                # --- Perform probability thresholding on the list of predictions ---
                if question_id in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(list_of_predictions)
                    results_by_question_id[question_id]["prediction_probabilities"].append(list_of_probability_pairs)
                    # make sure we don't put the same expected answer in the list over and over again.
                    if expected_answer not in results_by_question_id[question_id]["expected_answers"]:
                        results_by_question_id[question_id]["expected_answers"].append(expected_answer)
                else:
                    results_by_question_id[question_id] = {"predictions": [list_of_predictions],
                                                           "prediction_probabilities": [list_of_probability_pairs],
                                                           "expected_answers": [expected_answer]}

    # group together the most likely predictions. (i.e. corresponding positions in prediction lists)
    predictions_list, ground_truth_list = [], []
    k = 100 if contains_k("") is None else contains_k("")  # todo this properly

    for q_id in results_by_question_id:  # Gather all predictions for a particular question
        # results_by_question_id[q_id]["predictions"] is a list of lists
        # we get a nested structure, where each sub-list is the pos pred for an example, sorted by most to least likely
        pred_lists = results_by_question_id[q_id]["predictions"]
        prob_lists = results_by_question_id[q_id]["prediction_probabilities"]

        # For each list question, each participating system will have to return a single list* of entity names, numbers,
        # or similar short expressions, jointly taken to constitute a single answer (e.g., the most common symptoms of
        # a disease). The returned list will have to contain no more than 100 entries of no more than
        # 100 characters each.
        best_predictions = []
        num_best_predictions = 0

        probability_threshold = 0.2

        zipped_predictions = list(zip(*pred_lists))
        zipped_probabilities = list(zip(*prob_lists))

        # iterate over this prediction list until we reach the end, or we have enough predictions.
        for idx, ordered_pred_list in enumerate(zipped_predictions):  # zip each of the prediction lists found in here
            ordered_prob_list = zipped_probabilities[idx]
            for pred_idx, pred in enumerate(ordered_pred_list):
                s_prob, e_prob = ordered_prob_list[pred_idx]

                # if our prediction does not meet the probability threshold
                # todo we need to be more certain that our answer contains correct answers than with factoid questions
                if s_prob < probability_threshold or e_prob < probability_threshold:
                    continue

                if num_best_predictions >= k:
                    break

                # Avoid putting repeats in our list.
                # list predictions are counted negatively if they aren't part of the golden list
                if predicted_answer not in best_predictions:
                    num_best_predictions += 1
                    best_predictions.append(pred)

            if num_best_predictions >= k:
                break

        # swap the huge list of all predictions for our short-list of best predictions
        results_by_question_id[q_id]["predictions"] = best_predictions

        # We need to ensure that we don't try to evaluate the questions that don't have expected answers.
        # If either of the below conditions are true, i.e. we have at least one valid
        if len(results_by_question_id[q_id]["expected_answers"]) > 1 or \
                results_by_question_id[q_id]["expected_answers"][0] is not None:
            predictions_list.append(predicted_answer)
            ground_truth_list.append(results_by_question_id[q_id]["expected_answers"])

    if any(ground_truth_list):  # if any of the ground truth values are not None
        if dataset == "bioasq":
            evaluation_metrics = list_evaluation(predictions_list, ground_truth_list)
        else:
            raise Exception('Only bioasq is an acceptable dataset name to be passed to list evaluation functions.')
    else:
        evaluation_metrics = {}

    if training:
        return evaluation_metrics
    else:
        return results_by_question_id, evaluation_metrics