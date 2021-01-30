import os
import sys
import torch
from datasets import load_metric
import argparse
from read_data import dataset_to_fc
from fine_tuning import datasets, build_finetuned_from_checkpoint
from models import *
from utils import *
from data_processing import convert_samples_to_features, SQuADDataset, collate_wrapper
from torch.utils.data import DataLoader, RandomSampler

accuracy = load_metric("accuracy")
f1 = load_metric("f1")


# ------------------ SPECIFY GENERAL MODEL CONFIG ------------------
config = {
    'seed': 0,
    "eval_batch_size": 12,
    "n_best_size": 20,  # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    "max_answer_length": 30,  # maximum length of a generated answer
    "version_2_with_negative": False,  # If true, the squad examples contain some that do not have an answer.
}



# ----------- DEFINE METRIC CONFIG --------
def precision_recall_f1():
    pass

def exact_match():
    pass

metrics = {}

def evaluate(finetuned_model, test_dataset):




    metrics = {

    }

    return metrics



if __name__ == "__main__":
    # Log the process ID
    print(f"Process ID: {os.getpid()}\n")

    # -- Parse command line arguments (checkpoint name and model size)
    parser = argparse.ArgumentParser(description='Overwrite default fine-tuning settings.')
    parser.add_argument(
        "--size",
        default="small",
        choices=['small', 'base', 'large'],
        type=str,
        help="The size of the electra model e.g. 'small', 'base' or 'large",
    )

    parser.add_argument(
        "--f-checkpoint",
        default="",  # if not provided, we assume fine-tuning from pre-trained
        type=str,
        help="The name of the fine-tuning checkpoint to use e.g. small_factoid_15_10230_2_30487",
    )
    parser.add_argument(
        "--question-type",
        default="factoid",
        choices=['factoid', 'yesno', 'list'],
        type=str,
        help="Type of fine-tuned model should be created - factoid, list or yesno?",
    )
    parser.add_argument(
        "--dataset",
        default="squad",
        choices=['squad'],
        type=str,
        help="The name of the dataset to use in training e.g. squad",
    )
    args = parser.parse_args()
    config['size'] = args.size
    config["question_type"] = args.question_type

    sys.stderr.write("Selected finetuning checkpoint {} and model size {}"
                     .format(args.f_checkpoint, args.size))

    if args.f_checkpoint != "" and args.f_checkpoint != "recent":
        if args.size not in args.f_checkpoint:
            raise Exception("If using a fine-tuned checkpoint, the model size of the checkpoint must match provided model size."
                            "e.g. --f-checkpoint small_factoid_15_10230_12_20420 --size small")
        if args.question_type not in args.f_checkpoint:
            raise Exception(
                "If using a fine-tuned checkpoint, the question type of the checkpoint must match question type."
                "e.g. --f-checkpoint small_factoid_15_10230_12_20420 --question-type factoid")

    # ---- Set torch backend and set seed ----
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])  # set seed for reproducibility

    # -- Override general config with model specific config, for models of different sizes
    model_specific_config = get_model_config(config['size'], pretrain=False)
    config = {**model_specific_config, **config}

    # -- Find path to checkpoint directory - create the directory if it doesn't exist
    base_path = Path(__file__).parent
    checkpoint_name = ("", args.f_checkpoint)
    selected_dataset = args.dataset.lower()

    base_checkpoint_dir = (base_path / '../checkpoints').resolve()
    pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
    finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()
    dataset_dir = (base_checkpoint_dir / '../datasets').resolve()

    # create the fine-tune directory if it doesn't exist already
    Path(finetune_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # -- Set device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(config["device"].upper()))

    # get basic model building blocks
    generator, discriminator, electra_tokenizer, discriminator_config = build_electra_model(config['size'],
                                                                                            get_config=True)

    # ---- Load the data and prepare it in squad format ----
    try:
        dataset_file_name = datasets[selected_dataset]["test"]
    except KeyError:
        raise KeyError("The dataset '{}' in {} does not contain a 'test' key.".format(selected_dataset, datasets))

    # todo should we put which datasets were used to fine-tune checkpoint
    try:
        dataset_function = dataset_to_fc[selected_dataset]
    except KeyError:
        raise KeyError("The dataset '{}' is not contained in the dataset_to_fc map.".format(selected_dataset))

    all_datasets_dir = (base_checkpoint_dir / '../datasets').resolve()
    selected_dataset_dir = (all_datasets_dir / selected_dataset).resolve()
    dataset_file_path = (selected_dataset_dir / dataset_file_name).resolve()

    sys.stderr.write("\nReading raw dataset '{}' into SQuAD examples".format(dataset_file_name))
    read_raw_dataset = dataset_function(dataset_file_path)

    print("Converting raw text to features.".format(dataset_file_name))
    features = convert_samples_to_features(read_raw_dataset, electra_tokenizer, config["max_length"])

    print("Created {} features of length {}.".format(len(features), config["max_length"]))
    test_dataset = SQuADDataset(features)  # todo change this

    # Random Sampler not used during evaluation - we need to maintain order.
    data_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_wrapper)
    electra_for_qa, _, _, electra_tokenizer,\
    config = build_finetuned_from_checkpoint(config["size"], config["device"], pretrain_checkpoint_dir,
                                             finetune_checkpoint_dir, checkpoint_name, config["question_type"], config)

    # ------ START THE EVALUATION PROCESS ------
    evaluate(electra_for_qa, data_loader)
    # todo what happens if we start from a checkpoint here (vs passing a checkpoint from fine-tuning)
