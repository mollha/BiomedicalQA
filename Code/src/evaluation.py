from datasets import load_metric
import argparse
from tqdm import tqdm
from read_data import dataset_to_fc
from build_checkpoints import build_finetuned_from_checkpoint
from models import *
from utils import *
from data_processing import convert_test_samples_to_features, QADataset, collate_testing_wrapper, datasets
from torch.utils.data import DataLoader
from metrics.bioasq_metrics import yes_no_evaluation, factoid_evaluation
from metrics.squad_metrics import squad_evaluation

"""
Refer to the following code
https://huggingface.co/transformers/task_summary.html#extractive-question-answering
"""


def evaluate_yesno(yes_no_model, test_dataloader, training=False):
    results_by_question_id = {}

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
            except Exception:  # indexing outputs on CUDA
                logits = outputs[0]

            # treat outputs as if they correspond to a yes/no question
            # dim=1 makes sure we produce an answer start for each x in batch
            class_probabilities = torch.softmax(logits, dim=1)

            for question_idx, question_id in enumerate(batch.question_ids):
                expected_answer = batch.answer_text[question_idx]
                predicted_label = torch.argmax(class_probabilities[question_idx])
                print(predicted_label)
                if question_idx in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(predicted_label)
                else:
                    results_by_question_id[question_id] = {"predictions": [predicted_label],
                                                           "expected_answer": expected_answer}

    # iterate through predictions for each question
    # we need to combine these predictions to produce a final "yes" or "no" prediction.
    predictions_list, ground_truth_list = [], []
    for q_id in results_by_question_id:
        # results_by_question_id[q_id]["predictions"] is a list of scalar tensors e.g. [tensor(1), tensor(2)]
        pred_tensor = torch.Tensor(results_by_question_id[q_id]["predictions"])
        best_pred = torch.mode(pred_tensor, 0).values  # get the most common value in the prediction tensor

        print('best prediction', best_pred)
        predicted_answer = "yes" if best_pred == 1 else "no"  # convert 1s to yes and 0s to no
        results_by_question_id[q_id]["predictions"] = predicted_answer
        predictions_list.append(predicted_answer)
        ground_truth_list.append(results_by_question_id[q_id]["expected_answer"])

    # create a list of predictions and a list of ground_truth for evaluation
    print('predictions list', predictions_list)
    print('ground truth list', ground_truth_list)
    evaluation_metrics = yes_no_evaluation(predictions_list, ground_truth_list)

    if training:
        return evaluation_metrics
    else:
        # return a dictionary of {question_id: prediction (i.e. "yes" or "no")}
        return results_by_question_id, evaluation_metrics


def evaluate_factoid(factoid_model, test_dataloader, tokenizer, k, training=False):
    # if training flag is set, we only care about the metrics
    # this fc is being called from finetuning

    results_by_question_id = {}
    special_tokens = {tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token}

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

            # print(outputs.start_logits)
            # answer_start = torch.argmax(outputs.start_logits,
            #                             dim=1)  # Get the most likely beginning of answer with the argmax of the score
            # todo outputs cant be indexed like this anymore (see yes no)
            answer_starts, start_indices = torch.topk(outputs.start_logits, k=k, dim=1)
            answer_ends, end_indices = torch.topk(outputs.end_logits, k=k, dim=1)

            # print('answer_start', answer_start)
            # print('indices', start_indices)

            start_end_positions = [x for x in zip(start_indices, end_indices)]

            for index, (start_tensor, end_tensor) in enumerate(start_end_positions):
                sub_start_end_positions = zip(start_tensor, end_tensor)
                input_ids = batch.input_ids[index]
                expected_answer = batch.answer_text[index]
                question_id = batch.question_ids[index]

                list_of_predictions = []

                # convert the start and end positions to answers.
                for (s, e) in sub_start_end_positions:
                    # print('s', s, 'e', e)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    clipped_tokens = [t for t in tokens[int(s):int(e)] if t not in special_tokens]
                    predicted_answer = tokenizer.convert_tokens_to_string(clipped_tokens)
                    list_of_predictions.append(predicted_answer)

                if question_id in results_by_question_id:
                    results_by_question_id[question_id]["predictions"].append(list_of_predictions)
                else:
                    results_by_question_id[question_id] = {"predictions": [list_of_predictions],
                                                           "expected_answer": expected_answer}

    # group together the most likely predictions. (i.e. corresponding positions in prediction lists)
    for ex in results_by_question_id:
        pred_lists = results_by_question_id[ex]["predictions"]

        all_predictions = []
        for pred_list in pred_lists:
            for p_idx in range(k):
                all_predictions.append([pred_list[p_idx]])

        results_by_question_id[ex]["predictions"] = all_predictions

        # print(all_predictions)
        break

    # todo add in a max length for this prediction list

    # todo evaluate factoid

    evaluation_metrics = {}

    if training:
        return evaluation_metrics
    else:
        return results_by_question_id, evaluation_metrics


def evaluate_list(list_model, test_dataloader, tokenizer, k):
    return



def evaluate(finetuned_model, test_dataloader, tokenizer):
    k = 5
    results = []

    predictions = []
    ground_truths = []


    # todo now we have predicted and expected answers, we need to turn these into metrics

    # metrics = squad_evaluation(predictions, ground_truths)
    # print(metrics)

    # return metrics

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