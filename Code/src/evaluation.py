from datasets import load_metric
import argparse
from tqdm import tqdm
from read_data import dataset_to_fc
from fine_tuning import datasets, build_finetuned_from_checkpoint
from models import *
from utils import *
from data_processing import convert_samples_to_features, SQuADDataset, collate_wrapper
from torch.utils.data import DataLoader



""" ----------- SQUAD EVALUATION METRICS -----------

"""

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


def perform_bioasq_evaluation():
    pass


"""
Refer to the following code
https://huggingface.co/transformers/task_summary.html#extractive-question-answering
"""

def evaluate(finetuned_model, test_dataloader, tokenizer):
    results = []
    step_iterator = tqdm(test_dataloader, desc="Step")

    for eval_step, batch in enumerate(step_iterator):
        question_ids = batch.question_ids
        is_impossible = batch.is_impossible

        inputs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "token_type_ids": batch.token_type_ids,
        }

        # model outputs are always tuples in transformers
        outputs = finetuned_model(**inputs)

        # dim=1 makes sure we produce an answer start for each x in batch
        answer_start = torch.argmax(outputs.start_logits, dim=1)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(outputs.end_logits, dim=1) + 1  # Get the most likely end of answer with the argmax of the score

        # pair the start and end positions
        start_end_positions = zip(answer_start, answer_end)
        special_tokens = {tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token}

        batch_results = []

        # convert the start and end positions to answers.
        for index, (s, e) in enumerate(start_end_positions):
            input_ids = batch.input_ids[index]
            expected_answer = batch.answer_text[index]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            clipped_tokens = [t for t in tokens[int(s):int(e)] if t not in special_tokens]
            predicted_answer = tokenizer.convert_tokens_to_string(clipped_tokens)

            batch_results.append((predicted_answer, expected_answer))
            print('Predicted Answer: {}, Expected Answer: {}'.format(predicted_answer, expected_answer))

        results.append(batch_results)

    # todo now we have predicted and expected answers, we need to turn these into metrics

    return results



if __name__ == "__main__":
    # Log the process ID
    print(f"Process ID: {os.getpid()}\n")

    # -- Parse command line arguments (checkpoint name and model size)
    parser = argparse.ArgumentParser(description='Overwrite default fine-tuning settings.')
    parser.add_argument(
        "--f-checkpoint",
        default="small_factoid_26_11229_1_73",  # we can no longer use recent here - got to get specific :(
        type=str,
        help="The name of the fine-tuning checkpoint to use e.g. small_factoid_15_10230_2_30487",
    )
    parser.add_argument(
        "--dataset",
        default="squad",
        choices=['squad'],
        type=str,
        help="The name of the dataset to use in evaluated e.g. squad",
    )

    args = parser.parse_args()

    split_name = args.f_checkpoint.split("_")
    model_size, question_type = split_name[0], split_name[1]

    sys.stderr.write("Selected finetuning checkpoint {} and model size {}"
                     .format(args.f_checkpoint, model_size))

    # ---- Set torch backend and set seed ----
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])  # set seed for reproducibility

    # -- Override general config with model specific config, for models of different sizes
    model_specific_config = get_model_config(model_size, pretrain=False)
    config = {**model_specific_config, **config}
    config["num_warmup_steps"] = 100  # dummy value to avoid an error when building fine-tuned checkpoint.

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
    generator, discriminator, electra_tokenizer, discriminator_config = build_electra_model(model_size, get_config=True)

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
    config = build_finetuned_from_checkpoint(model_size, config["device"], pretrain_checkpoint_dir,
                                             finetune_checkpoint_dir, checkpoint_name, question_type, config)

    # ------ START THE EVALUATION PROCESS ------
    evaluate(electra_for_qa, data_loader, electra_tokenizer)
    # todo what happens if we start from a checkpoint here (vs passing a checkpoint from fine-tuning)
