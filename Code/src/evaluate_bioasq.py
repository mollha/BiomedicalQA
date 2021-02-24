from evaluation import evaluate_yesno, evaluate_factoid, evaluate_list
from pathlib import Path
import sys
import json
from read_data import dataset_to_fc
from fine_tuning import datasets, build_finetuned_from_checkpoint
from models import *
from utils import *
from data_processing import *
from torch.utils.data import DataLoader


def write_predictions(path_to_read_file, path_to_write_file, predictions):
    print(predictions)
    """
    Write predictions to a file within a folder named after the dataset.

    The format of a bioasq dataset is as follow:
    -- questions
    ------ body
    ------ id
    ------ ideal_answer
    ------ exact_answer
    ------ documents
    ------ snippets
    ------ concepts
    ------ triples

    :param predictions: predictions is a dictionary of question id to either list (factoid and list) or string (yesno)
    :return: None
    """
    # we want to create file at path_to_new_file within which we can write our predictions
    with open(path_to_read_file, 'rb') as infile:
        data_dict = json.load(infile)

    # now we need to add our predictions into the data_dict
    questions = data_dict["questions"]

    # iterate through every question
    for question in questions:
        question_id = question["id"]

        if question_id in predictions:  # if we have a prediction for this question
            pred = predictions[question_id]["predictions"]
            question["exact_answer"] = pred

    with open(path_to_write_file, 'w') as outfile:
        json.dump(data_dict, outfile)


# in bioasq, we need to provide results by file
if __name__ == "__main__":

    # ---- Manually set configuration here ----
    yes_no_checkpoint = "small_yesno_13_68164_9_123"
    factoid_checkpoint = None # "small_factoid_26_11229_1_380"
    list_checkpoint = factoid_checkpoint # use the same checkpoint for factoid and list

    selected_dataset = "bioasq"
    evaluate_on_dataset = "raw_data/8B1_golden.json"

    # -------------------------------

    print("Using the following checkpoints for evaluation:\n\tyesno - {}\n\tfactoid - {}\n\tlist - {}"
          .format(yes_no_checkpoint, factoid_checkpoint, list_checkpoint))

    # ---- Set torch backend and set seed ----
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(0)  # set seed for reproducibility

    base_path = Path(__file__).parent
    base_checkpoint_dir = (base_path / '../checkpoints').resolve()
    pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
    finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()
    all_datasets_dir = (base_checkpoint_dir / '../datasets').resolve()
    predictions_dir = (base_checkpoint_dir / '../predictions').resolve()
    selected_dataset_dir = (all_datasets_dir / selected_dataset).resolve()
    dataset_file_path = (selected_dataset_dir / evaluate_on_dataset).resolve()

    Path(predictions_dir).mkdir(exist_ok=True, parents=True)  # create the predictions dir if it doesn't already exist.
    predictions_dataset_dir = (predictions_dir / selected_dataset).resolve()  # find directory specific for predictions
    Path(predictions_dataset_dir).mkdir(exist_ok=True, parents=True)  # create the predictions dataset dir

    # ---- build a dictionary for storing the metrics we collect
    all_question_types = ['yesno', 'factoid', 'list']
    # metric_dictionary = {q_type: {} for q_type in all_question_types}
    results_by_question_id_dictionary = {}
    # results_by_question_id_dictionary = {d_file: {} for d_file in dataset_files}

    print("\nEvaluating on the '{}' dataset using the file '{}'".format(selected_dataset, evaluate_on_dataset))
    _, _, electra_tokenizer, _ = build_electra_model("small", get_config=True)  # get basic model building blocks

    # ---- Load the data and prepare it in squad format ----
    dataset_function = dataset_to_fc[selected_dataset]  # Load the data and prepare it in squad format

    sys.stderr.write("\nReading raw dataset '{}' into examples".format(evaluate_on_dataset))
    # read_raw_dataset, metrics = dataset_function([dataset_file_path], testing=True)  # returns a dictionary of question type to list

    # read all question types
    raw_test_dataset = dataset_function([dataset_file_path], testing=True, question_types=all_question_types)

    question_order = ["yesno", "factoid", "list"]
    for checkpoint_idx, checkpoint_name in enumerate([yes_no_checkpoint, factoid_checkpoint, list_checkpoint]):
        if checkpoint_name is None:
            sys.stderr.write("\nSkipping {} question checkpoint as provided checkpoint name is None.".format(
                all_question_types[checkpoint_idx]))
            continue

        model_size = checkpoint_name.split("_")[0]
        question_type = question_order[checkpoint_idx]

        sys.stderr.write("\nEvaluating checkpoint {} - model size is {} and question type is {}\n"
                         .format(checkpoint_name, model_size, question_type))

        # -- Override general config with model specific config, for models of different sizes
        config = get_model_config(model_size, pretrain=False)
        config["num_warmup_steps"] = 100  # dummy value to avoid an error when building fine-tuned checkpoint.
        config["dataset"] = selected_dataset

        # -- Set device
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        # get only the data relevant to this specific question type
        raw_dataset = raw_test_dataset[question_type]

        print("Converting raw text to features.")
        features = convert_examples_to_features(raw_dataset, electra_tokenizer, config["max_length"])

        print("Created {} features of length {}.".format(len(features), config["max_length"]))
        test_dataset = QADataset(features)

        # Random Sampler not used during evaluation - we need to maintain order.
        data_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_wrapper)
        electra_for_qa, _, _, electra_tokenizer, config = build_finetuned_from_checkpoint(model_size, config["device"], pretrain_checkpoint_dir, finetune_checkpoint_dir, ("", checkpoint_name), question_type, config)

        # ------ START THE EVALUATION PROCESS ------
        if question_type == "factoid":
            results_by_question_id, metric_results = evaluate_factoid(electra_for_qa, data_loader, electra_tokenizer, k=5, training=True, dataset=selected_dataset)
        elif question_type == "yesno":
            results_by_question_id, metric_results = evaluate_yesno(electra_for_qa, data_loader)
        elif question_type == "list":
            results_by_question_id, metric_results = evaluate_list(electra_for_qa, data_loader, electra_tokenizer, k=5, training=True, dataset=selected_dataset)
        else:
            raise Exception("No other question types permitted except factoid, yesno and list.")

        results_by_question_id_dictionary = {**results_by_question_id_dictionary, **results_by_question_id}

        # metric_dictionary[question_type] = metric_results

    print(results_by_question_id_dictionary)

    print('----- METRICS -----')  # pretty-print the metrics
    print('\n--- Dataset: {}'.format(selected_dataset))
    # metrics_for_dset = metric_dictionary[selected_dataset]

    # for qt in metrics_for_dset.keys():
    #     metrics_for_qt_dset = metrics_for_dset[qt]
    #     print('Question Type: {}, Metrics: {}'.format(qt, metrics_for_qt_dset))

    # iterate through our predictions for each dataset and write them to text files
    selected_dataset_dir = (all_datasets_dir / selected_dataset).resolve()

    # get the name of the file name
    path_to_prediction_file = (predictions_dataset_dir / evaluate_on_dataset.split('/')[-1]).resolve()
    path_to_original_file = (selected_dataset_dir / evaluate_on_dataset).resolve()

    # for the bioasq challenge, we need to add our answers BACK into the dataset.
    # provide the name of the file we want to write predictions to
    write_predictions(path_to_original_file, path_to_prediction_file, results_by_question_id_dictionary)
