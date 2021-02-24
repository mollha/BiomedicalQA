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
    yes_no_checkpoint = "small_yesno_26_11229_1_374"
    factoid_checkpoint = "small_factoid_26_11229_1_380"
    list_checkpoint = factoid_checkpoint # use the same checkpoint for factoid and list

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

    # create the predictions dir if it doesn't already exist.
    Path(predictions_dir).mkdir(exist_ok=True, parents=True)

    selected_dataset = "bioasq"
    dataset_files = ["raw_data/8B1_golden.json", "raw_data/8B2_golden.json", "raw_data/8B3_golden.json",
                     "raw_data/8B4_golden.json", "raw_data/8B5_golden.json"]
    all_question_types = ['yesno', 'factoid', 'list']

    # find directory specific for predictions from our chosen dataset
    predictions_dataset_dir = (predictions_dir / selected_dataset).resolve()
    # create the predictions dir for this dataset if it doesn't already exist.
    Path(predictions_dataset_dir).mkdir(exist_ok=True, parents=True)

    # ---- build a dictionary for storing the metrics we collect
    metric_dictionary = {}
    for d_file in dataset_files:
        metric_dictionary[d_file] = {q_type: {} for q_type in all_question_types}

    results_by_question_id_dictionary = {d_file: {} for d_file in dataset_files}

    print("\nEvaluating on the '{}' dataset using the following files:".format(selected_dataset))
    for f in dataset_files:
        print("\f", f)

    # get basic model building blocks
    _, _, electra_tokenizer, _ = build_electra_model("small", get_config=True)

    # ---- Load the data and prepare it in squad format ----
    try:
        dataset_function = dataset_to_fc[selected_dataset]
    except KeyError:
        raise KeyError("The dataset '{}' is not contained in the dataset_to_fc map.".format(selected_dataset))

    for dataset_file_name in dataset_files:

        selected_dataset_dir = (all_datasets_dir / selected_dataset).resolve()
        dataset_file_path = (selected_dataset_dir / dataset_file_name).resolve()

        sys.stderr.write("\nReading raw dataset '{}' into SQuAD examples".format(dataset_file_name))
        read_raw_dataset, metrics = dataset_function(dataset_file_path,
                                                     testing=True)  # returns a dictionary of question type to list

        for checkpoint_idx, checkpoint_name in enumerate([yes_no_checkpoint, factoid_checkpoint, list_checkpoint]):
            if checkpoint_name is None:
                sys.stderr.write("\nSkipping {} question checkpoint as provided checkpoint name is None.".format(
                    all_question_types[checkpoint_idx]))
                continue

            split_name = checkpoint_name.split("_")
            model_size, question_type = split_name[0], split_name[1]

            sys.stderr.write("\nEvaluating checkpoint {} - model size is {} and question type is {}\n"
                             .format(checkpoint_name, model_size, question_type))

            # -- Override general config with model specific config, for models of different sizes
            config = get_model_config(model_size, pretrain=False)
            config["num_warmup_steps"] = 100  # dummy value to avoid an error when building fine-tuned checkpoint.

            # -- Set device
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

            # get only the data relevant to this specific question type
            raw_dataset = read_raw_dataset[question_type]

            print("Converting raw text to features.")
            features = convert_test_samples_to_features(raw_dataset, electra_tokenizer, config["max_length"])

            print("Created {} features of length {}.".format(len(features), config["max_length"]))
            test_dataset = QADataset(features)

            # Random Sampler not used during evaluation - we need to maintain order.
            data_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_testing_wrapper)
            electra_for_qa, _, _, electra_tokenizer, \
            config = build_finetuned_from_checkpoint(model_size, config["device"], pretrain_checkpoint_dir,
                                                     finetune_checkpoint_dir, ("", checkpoint_name), question_type, config)

            # ------ START THE EVALUATION PROCESS ------
            if question_type == "factoid":
                results_by_question_id, metric_results = evaluate_factoid(electra_for_qa, data_loader, electra_tokenizer, 1)
            elif question_type == "yesno":
                results_by_question_id, metric_results = evaluate_yesno(electra_for_qa, data_loader)
                results_by_question_id_dictionary[dataset_file_name] = {**results_by_question_id_dictionary[dataset_file_name],
                                                     **results_by_question_id}  # todo move this to the end once done (i.e. once we have results_by_question_id for all question types)
            elif question_type == "list":
                results_by_question_id, metric_results = evaluate_list(electra_for_qa, data_loader, electra_tokenizer, 1)  # todo this is not valid yet don't forget
            else:
                raise Exception("No other question types permitted except factoid, yesno and list.")

            metric_dictionary[dataset_file_name][question_type] = metric_results

    print(results_by_question_id_dictionary)

    print('----- METRICS -----')  # pretty-print the metrics
    for key in metric_dictionary.keys():
        print('\n--- Dataset: {}'.format(key))
        metrics_for_dset = metric_dictionary[key]

        for qt in metrics_for_dset.keys():
            metrics_for_qt_dset = metrics_for_dset[qt]
            print('Question Type: {}, Metrics: {}'.format(qt, metrics_for_qt_dset))

    # iterate through our predictions for each dataset and write them to text files
    for key, value in results_by_question_id_dictionary.items():
        selected_dataset_dir = (all_datasets_dir / selected_dataset).resolve()

        # get the name of the file name
        path_to_prediction_file = (predictions_dataset_dir / key.split('/')[-1]).resolve()
        path_to_original_file = (selected_dataset_dir / key).resolve()

        # for the bioasq challenge, we need to add our answers BACK into the dataset.
        # provide the name of the file we want to write predictions to
        write_predictions(path_to_original_file, path_to_prediction_file, value)
