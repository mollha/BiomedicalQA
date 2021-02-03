from datasets import load_metric
import argparse
from tqdm import tqdm
from read_data import dataset_to_fc
from fine_tuning import datasets, build_finetuned_from_checkpoint
from models import *
from utils import *
from data_processing import convert_test_samples_to_features, QADataset, collate_testing_wrapper
from torch.utils.data import DataLoader

from metrics.bioasq_metrics import yes_no_evaluation, factoid_evaluation
from metrics.squad_metrics import squad_evaluation

"""
Refer to the following code
https://huggingface.co/transformers/task_summary.html#extractive-question-answering
"""


def evaluate_yesno(yes_no_model, test_dataloader):
    results_by_question_id = {}

    for eval_step, batch in enumerate(tqdm(test_dataloader, desc="Step")):
        question_ids = batch.question_ids

        inputs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "token_type_ids": batch.token_type_ids,
        }

        # model outputs are always tuples in transformers
        outputs = yes_no_model(**inputs)

        # treat outputs as if they correspond to a yes/no question
        # dim=1 makes sure we produce an answer start for each x in batch
        predicted_labels = torch.softmax(outputs.logits, dim=1).tolist()[0]

        for question_idx, question_id in enumerate(question_ids):
            predicted_answer = ["yes" if predicted_labels[question_idx] == 1 else "no"]
            if question_idx in results_by_question_id:
                results_by_question_id[question_id].append(predicted_answer)
            else:
                results_by_question_id[question_id] = [predicted_answer]

    return results_by_question_id


def evaluate_factoid(factoid_model, test_dataloader, tokenizer, k):
    results_by_question_id = {}

    for eval_step, batch in enumerate(tqdm(test_dataloader, desc="Step")):
        # question_ids = batch.question_ids

        inputs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "token_type_ids": batch.token_type_ids,
        }

        # model outputs are always tuples in transformers
        outputs = factoid_model(**inputs)

        answer_starts = torch.topk(outputs.start_logits, k=k, dim=1)
        answer_ends = torch.topk(outputs.end_logits, k=k, dim=1)

        print('answer_starts', answer_starts)
        quit()
        # answer_start = torch.argmax(outputs.start_logits, dim=1)  # Get the most likely beginning of answer with the argmax of the score
        # answer_end = torch.argmax(outputs.end_logits, dim=1) + 1  # Get the most likely end of answer with the argmax of the score

        for i in range(k):  # iterate in order of most likely to least in top_k
            answer_end, answer_start = answer_ends[i], answer_starts[i]

            # pair the start and end positions
            start_end_positions = zip(answer_start, answer_end)
            special_tokens = {tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token}

            # convert the start and end positions to answers.
            for index, (s, e) in enumerate(start_end_positions):
                input_ids = batch.input_ids[index]
                expected_answer = batch.answer_text[index]
                question_id = batch.question_ids[index]

                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                clipped_tokens = [t for t in tokens[int(s):int(e)] if t not in special_tokens]
                predicted_answer = tokenizer.convert_tokens_to_string(clipped_tokens)

            if question_id in results_by_question_id:
                results_by_question_id[question_id].append(predicted_answer)
            else:
                results_by_question_id[question_id] = [predicted_answer]



    return results_by_question_id


def evaluate(finetuned_model, test_dataloader, tokenizer):
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
        default="small_factoid_26_11229_1_380",  # we can no longer use recent here - got to get specific :(
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

    args = parser.parse_args()
    split_name = args.f_checkpoint.split("_")
    model_size, question_type = split_name[0], split_name[1]

    sys.stderr.write("Selected finetuning checkpoint {} and model size {}"
                     .format(args.f_checkpoint, model_size))

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

    # config = {
    #     "eval_batch_size": 12,
    #     "n_best_size": 20,
    #     # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    #     "max_answer_length": 30,  # maximum length of a generated answer
    # }

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

    # todo should we put which datasets were used to fine-tune checkpoint
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

    question_type = "factoid"
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
    # todo for now we're only allowing factoid evals

    #evaluate(electra_for_qa, data_loader, electra_tokenizer)

    evaluate_factoid(electra_for_qa, data_loader, electra_tokenizer, 5)
    # todo what happens if we start from a checkpoint here (vs passing a checkpoint from fine-tuning)
