from datasets import load_metric
import argparse
from tqdm import tqdm
from read_data import dataset_to_fc
from build_checkpoints import build_finetuned_from_checkpoint
from models import *
from utils import *
from data_processing import convert_examples_to_features, QADataset, collate_wrapper, datasets
from torch.utils.data import DataLoader
from metrics.bioasq_metrics import yes_no_evaluation, factoid_evaluation
from metrics.squad_metrics import squad_evaluation

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
    # yes_no_model.eval()  # switch to evaluation mode
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

            # treat outputs as if they correspond to a yes/no question
            # dim=1 makes sure we produce an answer start for each x in batch
            class_probabilities = torch.softmax(logits, dim=1)

            for question_idx, question_id in enumerate(batch.question_ids):
                # Note: expected answers could be None here if we don't have access to the answer.
                # We still need to include them in our dictionary, and filter them at the next step.
                expected_answer = batch.answer_text[question_idx]
                predicted_label = torch.argmax(class_probabilities[question_idx])
                if question_idx in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(predicted_label)
                else:
                    results_by_question_id[question_id] = {"predictions": [predicted_label],
                                                           "expected_answer": expected_answer}

    # Iterate through predictions for each question. We need to combine these predictions to produce a final prediction.
    # create a list of predictions and a list of ground_truth for evaluation
    predictions_list, ground_truth_list = [], []
    for q_id in results_by_question_id:
        # results_by_question_id[q_id]["predictions"] is a list of scalar tensors e.g. [tensor(1), tensor(2)]
        pred_tensor = torch.Tensor(results_by_question_id[q_id]["predictions"])
        best_pred = torch.mode(pred_tensor, 0).values  # get the most common value in the prediction tensor

        # todo best prediction always seems to be 1 / yes
        # batch labels are varied
        # print('best prediction', best_pred)
        predicted_answer = "yes" if best_pred == 1 else "no"  # convert 1s to yes and 0s to no
        results_by_question_id[q_id]["predictions"] = predicted_answer

        # We need to ensure that we don't try to evaluate the questions that don't have expected answers.
        if results_by_question_id[q_id]["expected_answer"] is not None:  # i.e. we have an answer
            predictions_list.append(predicted_answer)
            ground_truth_list.append(results_by_question_id[q_id]["expected_answer"])

    if dataset == "bioasq":  # Deploy the bioasq metrics on our results.
        # Evaluation metrics are empty if we didn't have any expected answers.
        eval_metrics = {} if len(predictions_list) == 0 else yes_no_evaluation(predictions_list, ground_truth_list)

        if training:
            return eval_metrics
        # return a dictionary of {question_id: prediction (i.e. "yes" or "no")}
        return results_by_question_id, eval_metrics
    raise Exception("Dataset name provided to evaluate_yesno must be 'bioasq', "
                    "as no other datasets are handled at this time.")


def evaluate_factoid(factoid_model, test_dataloader, tokenizer, k, training=False, dataset="bioasq"):
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

    # factoid_model.eval()  # switch to evaluation mode
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
            answer_starts, start_indices = torch.topk(start_logits, k=k, dim=1)
            answer_ends, end_indices = torch.topk(end_logits, k=k, dim=1)


            # todo perform thresholding if necessary

            # print('answer_starts', answer_starts)
            # print('start_indices', start_indices)
            # print('answer_ends', answer_ends)
            # print('end_indices', end_indices)
            # quit()

            start_end_positions = [x for x in zip(start_indices, end_indices)]
            # print('start_end_positions', start_end_positions)

            # iterate over our pairs of start and end indices
            for index, (start_tensor, end_tensor) in enumerate(start_end_positions):
                # e.g. start_tensor = tensor([110,  33,  38, 111,  35]), end_tensor = tensor([20,  0, 90, 36, 62])
                sub_start_end_positions = zip(start_tensor, end_tensor)  # zip the start and end positions
                input_ids = batch.input_ids[index]
                expected_answer = batch.answer_text[index]
                question_id = batch.question_ids[index]

                list_of_predictions = []  # gather all of the predictions for this question
                for (s, e) in sub_start_end_positions:  # convert the start and end positions to answers.
                    if e <= s:  # if end position is less than or equal to start position, skip this pair
                        continue
                    clipped_ids = [t for t in input_ids[int(s):int(e)] if t not in special_tokens_ids]
                    clipped_tokens = tokenizer.convert_ids_to_tokens(clipped_ids, skip_special_tokens=True)
                    # make sure we don't end up with special characters in our predicted
                    predicted_answer = tokenizer.convert_tokens_to_string(clipped_tokens)  # todo we need a way to do this that handles punctuation better
                    list_of_predictions.append(predicted_answer)

                if question_id in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(list_of_predictions)

                    # make sure we don't put the same expected answer in the list over and over again.
                    if expected_answer not in results_by_question_id[question_id]["expected_answers"]:
                        results_by_question_id[question_id]["expected_answers"].append(expected_answer)
                else:
                    results_by_question_id[question_id] = {"predictions": [list_of_predictions],
                                                           "expected_answers": [expected_answer]}

    # group together the most likely predictions. (i.e. corresponding positions in prediction lists)
    predictions_list, ground_truth_list = [], []
    for q_id in results_by_question_id:  # Gather all predictions for a particular question
        # results_by_question_id[q_id]["predictions"] is a list of lists
        # we get a nested structure, where each sub-list is the pos pred for an example, sorted by most to least likely
        pred_lists = results_by_question_id[q_id]["predictions"]

        # For each factoid question in BioASQ, each participating system will have to return a list* of up to 5 entity names
        # (e.g., up to 5 names of drugs), numbers, or similar short expressions, ordered by decreasing confidence.
        best_predictions = []
        num_best_predictions = 0

        # if dataset == "squad":
        #     k = 1

        # iterate over this prediction list until we reach the end, or we have enough predictions.
        for ordered_pred_list in zip(*pred_lists):  # zip each of the prediction lists found in here
            for pred in ordered_pred_list:
                if num_best_predictions >= k:
                    break

                num_best_predictions += 1
                best_predictions.append(pred)

            if num_best_predictions >= k:
                break

        # swap the huge list of all predictions for our short-list of best predictions
        results_by_question_id[q_id]["predictions"] = best_predictions

        # We need to ensure that we don't try to evaluate the questions that don't have expected answers.
        # If either of the below conditions are true, i.e. we have at least one valid
        if len(results_by_question_id[q_id]["expected_answers"]) > 1 or results_by_question_id[q_id]["expected_answers"][0] is not None:
            predictions_list.append(predicted_answer)
            ground_truth_list.append(results_by_question_id[q_id]["expected_answers"])

    if dataset == "bioasq":
        evaluation_metrics = factoid_evaluation(predictions_list, ground_truth_list)
    elif dataset == "squad":
        # this should be able to handle lists of lists of ground truth values.
        evaluation_metrics = squad_evaluation(predictions_list, ground_truth_list)
    else:
        raise Exception('Only squad and bioasq are acceptable dataset names to be passed to evaluation functions.')

    if training:
        return evaluation_metrics
    else:
        return results_by_question_id, evaluation_metrics


def evaluate_list(list_model, test_dataloader, tokenizer, k, dataset="bioasq"):
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

    # For each list question, each participating system will have to return a single list* of entity names, numbers,
    # or similar short expressions, jointly taken to constitute a single answer (e.g., the most common symptoms of
    # a disease). The returned list will have to contain no more than 100 entries of no more than 100 characters each.

    results_by_question_id = {}
    special_tokens_ids = {tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

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
            answer_starts, start_indices = torch.topk(start_logits, k=k, dim=1)
            answer_ends, end_indices = torch.topk(end_logits, k=k, dim=1)

            start_end_positions = [x for x in zip(start_indices, end_indices)]
            # print('start_end_positions', start_end_positions)

            # iterate over our pairs of start and end indices
            for index, (start_tensor, end_tensor) in enumerate(start_end_positions):
                # e.g. start_tensor = tensor([110,  33,  38, 111,  35]), end_tensor = tensor([20,  0, 90, 36, 62])
                sub_start_end_positions = zip(start_tensor, end_tensor)  # zip the start and end positions
                input_ids = batch.input_ids[index]
                expected_answer = batch.answer_text[index]
                question_id = batch.question_ids[index]

                list_of_predictions = []  # gather all of the predictions for this question
                for (s, e) in sub_start_end_positions:  # convert the start and end positions to answers.
                    if e <= s:  # if end position is less than or equal to start position, skip this pair
                        continue
                    clipped_ids = [t for t in input_ids[int(s):int(e)] if t not in special_tokens_ids]
                    clipped_tokens = tokenizer.convert_ids_to_tokens(clipped_ids, skip_special_tokens=True)
                    # make sure we don't end up with special characters in our predicted
                    predicted_answer = tokenizer.convert_tokens_to_string(
                        clipped_tokens)  # todo we need a way to do this that handles punctuation better
                    list_of_predictions.append(predicted_answer)

                if question_id in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(list_of_predictions)

                    # make sure we don't put the same expected answer in the list over and over again.
                    if expected_answer not in results_by_question_id[question_id]["expected_answers"]:
                        results_by_question_id[question_id]["expected_answers"].append(expected_answer)
                else:
                    results_by_question_id[question_id] = {"predictions": [list_of_predictions],
                                                           "expected_answers": [expected_answer]}




    if dataset == "bioasq":
        pass
    elif dataset == "squad":
        pass
    else:
        raise Exception('Only squad and bioasq are acceptable dataset names to be passed to evaluation functions.')

    return


if __name__ == "__main__":
    # Log the process ID
    print(f"Process ID: {os.getpid()}\n")

    # -- Parse command line arguments (checkpoint name and model size)
    parser = argparse.ArgumentParser(description='Overwrite default fine-tuning settings.')
    parser.add_argument(
        "--f-checkpoint",
        default="small_yesno_26_11229_1_374",  # we can no longer use recent here - got to get specific :(
        type=str,
        help="The name of the fine-tuning checkpoint to use e.g. small_factoid_15_10230_2_30487",
    )
    parser.add_argument(
        "--dataset",
        default="bioasq",
        choices=['squad', 'bioasq'],
        type=str,
        help="The name of the dataset to use in evaluated e.g. squad",
    )
    parser.add_argument(
        "--k",
        default="5",
        type=int,
        help="K-best predictions are selected for factoid and list questions (between 1 and 100)",
    )

    args = parser.parse_args()
    split_name = args.f_checkpoint.split("_")
    model_size, question_type = split_name[0], split_name[1]

    sys.stderr.write("--- ARGUMENTS ---")
    sys.stderr.write("\nEvaluating checkpoint: {}\nQuestion type: {}\nModel Size: {}\nK: {}"
                     .format(args.p_checkpoint, question_type, model_size, args.k))

    # ------- Check the validity of the arguments passed via command line -------
    try:  # check that the value of k is actually a number
        k = int(args.k)
        if k < 1 or k > 100:  # check that k is at least 1 and at most 100
            raise ValueError
    except ValueError:
        raise Exception("k must be an integer between 1 and 100. Got {}".format(args.k))

    # -- Override general config with model specific config, for models of different sizes
    config = get_model_config(model_size, pretrain=False)
    config["num_warmup_steps"] = 100  # dummy value to avoid an error when building fine-tuned checkpoint.

    # model_specific_pretrain_config = get_model_config(config['size'], pretrain=False)
    # model_specific_finetune_config = get_model_config(config['size'], pretrain=False)
    # config = {**model_specific_config, **config}

    # -- Find path to checkpoint directory - create the directory if it doesn't exist
    base_path = Path(__file__).parent
    checkpoint_name = ("", args.f_checkpoint)
    selected_dataset = args.dataset.lower()

    base_checkpoint_dir = (base_path / '../checkpoints').resolve()
    pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
    finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()
    dataset_dir = (base_checkpoint_dir / '../datasets').resolve()

    # Create the fine-tune directory if it doesn't exist already
    Path(finetune_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # -- Set device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(config["device"].upper()))

    # get basic model building blocks
    generator, discriminator, electra_tokenizer, discriminator_config = build_electra_model(model_size, get_config=True)

    # ---- Load the data and prepare it in squad format ----
    try:
        dataset_file_name = datasets[selected_dataset]["test"]
    except KeyError:
        raise KeyError("The dataset '{}' in {} does not contain a 'test' key.".format(selected_dataset, datasets))

    try:
        dataset_function = dataset_to_fc[selected_dataset]
    except KeyError:
        raise KeyError("The dataset '{}' is not contained in the dataset_to_fc map.".format(selected_dataset))

    all_datasets_dir = (base_checkpoint_dir / '../datasets').resolve()
    selected_dataset_dir = (all_datasets_dir / selected_dataset).resolve()
    dataset_file_path = (selected_dataset_dir / dataset_file_name).resolve()

    sys.stderr.write("\nReading raw dataset '{}' into SQuAD examples".format(dataset_file_name))
    read_raw_dataset, metrics = dataset_function(dataset_file_path, testing=True)  # returns a dictionary of question type to list

    print(read_raw_dataset)

    raw_dataset = read_raw_dataset[question_type]

    print("Converting raw text to features.")
    features = convert_test_samples_to_features(raw_dataset, electra_tokenizer, config["max_length"])

    print("Created {} features of length {}.".format(len(features), config["max_length"]))
    test_dataset = QADataset(features)  # todo change this

    # Random Sampler not used during evaluation - we need to maintain order.
    data_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_testing_wrapper)
    electra_for_qa, _, _, electra_tokenizer,\
    config = build_finetuned_from_checkpoint(model_size, config["device"], pretrain_checkpoint_dir,
                                             finetune_checkpoint_dir, checkpoint_name, question_type, config)

    # ---- Set torch backend and set seed ----
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])  # set seed for reproducibility

    # ------ START THE EVALUATION PROCESS ------
    if question_type == "factoid":
        results_by_question_id, metric_results = evaluate_factoid(electra_for_qa, data_loader, electra_tokenizer, k)
    elif question_type == "yesno":
        results_by_question_id, metric_results = evaluate_yesno(electra_for_qa, data_loader)