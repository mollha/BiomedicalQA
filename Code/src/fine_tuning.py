import torch
import argparse
import os
from read_data import dataset_to_fc
import sys
from pathlib import Path
from tqdm import trange
from tqdm import tqdm
from torch import nn
from models import *
import numpy as np
from utils import *
from glob import glob
from data_processing import convert_samples_to_features, SQuADDataset
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from transformers import AdamW, get_linear_schedule_with_warmup
from pre_training import build_pretrained_from_checkpoint
from torch.utils.data import DataLoader, RandomSampler
# from run_factoid import train, evaluate

from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForQuestionAnswering,
    ElectraForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)

# Ensure that lowercase model is used for model_type
# ------------- DEFINE TRAINING AND EVALUATION SETTINGS -------------
# hyperparameters are mostly the same for large models as base models - except for a few

config = {
    "learning_rate": 8e-6,  # The initial learning rate for Adam.
    "decay": 0.0,  # Weight decay if we apply some.
    "epsilon": 1e-8,  # Epsilon for Adam optimizer.
    "max_grad_norm": 1.0,  # Max gradient norm.
    "update_steps": 500,
    'seed': 0,
    'loss': [],
    'num_workers': 3 if torch.cuda.is_available() else 0,
    "max_epochs": 2,  # can override the val in config
    "current_epoch": 0,  # track the current epoch in config for saving checkpoints
    "steps_trained": 0,  # track the steps trained in config for saving checkpoints
    "global_step": -1,  # total steps over all epochs
}

# Adam's beta 1 and 2 are set to 0.9, and 0.999. They are default values for Adam optimizer.
finetune_qa_config = {
    "lr": 2e-4,
    "batch_size": 32,
    "max_steps": 400 * 1000,
    "max_length": 512,
    "generator_size_divisor": 4,
    'adam_bias_correction': False
}

# ----------------------- SPECIFY DATASET PATHS -----------------------
datasets = {
    # "bioasq": {"train_file": "../qa_datasets/QA/BioASQ/BioASQ-train-factoid-7b.json",
    #            "golden_file": "../qa_datasets/QA/BioASQ/7B_golden.json",
    #            "official_eval_dir": "./scripts/bioasq_eval"},
    "squad": {
        "train": "train-v2.0.json",
        "test": "dev-v2.0.json",
    }
}

def load_and_cache_examples(tokenizer, model_path, train_file, evaluate=False, output_examples=False):
    overwrite_cached_features = True
    max_seq_length = 384  # The maximum total input sequence length after WordPiece tokenization. Sequences " "longer than this will be truncated, and sequences shorter than this will be padded."
    predict_file = "gdrive/My Drive/BioBERT/qa_datasets/QA/BioASQ/BioASQ-test-factoid-7b.json"  # "..qa_datasets/QA/BioASQ/BioASQ-test-factoid-7b.json" # # The input evaluation file. If a data dir is specified, will look for the file there" If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    version_2_with_negative = False
    doc_stride = 128  # When splitting up a long document into chunks, how much stride to take between chunks
    max_query_length = 64  # The maximum number of tokens for the question. Questions longer than this will " "be truncated to this length.
    data_dir = None  # The input data dir. Should contain the .json files for the task. If no data dir or train/predict files are specified, will run with tensorflow_datasets."

    if doc_stride >= max_seq_length - max_query_length:
        print("WARNING - Doc stride may be larger than question length in some "
              "samples. This could result in errors when building features from the examples. Please reduce the doc "
              "stride or increase the maximum length to ensure the features are correctly built.")

    # Load data features from cache or dataset file
    input_dir = data_dir if data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_path.split("/"))).pop(),
            str(max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not overwrite_cached_features:
        print("Loading features from cached file {}".format(cached_features_file))
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        print("Creating features from dataset file at {}".format(input_dir))

        if not data_dir and ((evaluate and not predict_file) or (not evaluate and not train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if version_2_with_negative:
                print("tensorflow_datasets does not handle version 2 of squad.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(data_dir, filename=predict_file)
            else:
                examples = processor.get_train_examples(data_dir, filename=train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
        )

        print("Saving features into cached file {}".format(cached_features_file))
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def loss_function():
    return ""

# ---------- DEFINE MAIN FINE-TUNING LOOP ----------
def fine_tune(train_dataloader, qa_model, scheduler, optimizer, settings, checkpoint_dir):
    qa_model.to(settings["device"])

    # ------------------ PREPARE TO START THE TRAINING LOOP ------------------
    print("\n---------- BEGIN FINE-TUNING ----------")
    sys.stderr.write("\nDevice = {}\nModel Size = {}\nTotal Epochs = {}\nStart training from Epoch = {}\nStart training from Step = {}\nBatch size = {}\nCheckpoint Steps = {}\nMax Sample Length = {}\n\n"
                     .format(settings["device"].upper(), settings["size"], settings["max_epochs"], settings["current_epoch"],
                             settings["steps_trained"], settings["batch_size"], settings["update_steps"],
                             settings["max_length"]))

    total_training_loss, logging_loss = 0.0, 0.0
    qa_model.zero_grad()
    # Added here for reproducibility
    set_seed(settings["seed"])

    # evaluate during training always.

    # Resume training from the epoch we left off at earlier.
    train_iterator = trange(settings["current_epoch"], int(settings["max_epochs"]), desc="Epoch")

    # resume training
    steps_trained = settings["steps_trained"]

    for epoch_number in train_iterator:
        epoch_iterator = tqdm(data_loader, desc="Iteration")

        # update the current epoch
        settings["current_epoch"] = epoch_number  # update the number of epochs

        for training_step, batch in enumerate(epoch_iterator):

            print("Batch: ", batch)

            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue  # skip this step

            batch = batch.to(settings["device"])  # project batch to correct device
            qa_model.train()  # train model one step

            print('Batch: ', batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = qa_model(**inputs)

            # model outputs are always tuples in transformers
            loss = outputs.loss
            loss.backward()

            total_training_loss += loss.item()

            nn.utils.clip_grad_norm_(qa_model.parameters(), 1.)

            # perform steps
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            qa_model.zero_grad()

            settings["steps_trained"] = training_step
            settings["global_step"] += 1

            # Log metrics
            if settings["global_step"] > 0 and settings["update_steps"] > 0 and settings["global_step"] % settings[
                "update_steps"] == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                # Evaluate all checkpoints starting with same prefix as model_name ending and ending with step number

                print("{} steps trained in current epoch, {} steps trained overall."
                      .format(settings["steps_trained"], settings["global_step"]))

                # Save model checkpoint
                # putting loss here is probably wrong.
                save_checkpoint(qa_model, optimizer, scheduler, loss_function, settings, checkpoint_dir)

    # todo update loss function statistics
    # ------------- SAVE FINE-TUNED MODEL -------------
    save_checkpoint(qa_model, optimizer, scheduler, loss_function, settings, checkpoint_dir)


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
        "--checkpoint",
        default="recent",
        type=str,
        help="The name of the pre-training checkpoint to use e.g. small_15_10230",
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

    print("Selected checkpoint {} and model size {}".format(args.checkpoint, args.size))
    if args.checkpoint != "recent" and args.size not in args.checkpoint:
        raise Exception("If not using the most recent checkpoint, the checkpoint type must match model size."
                        "e.g. --checkpoint small_15_10230 --size small")

    # -- Set torch backend and set seed
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])  # set seed for reproducibility

    # -- Override general config with model specific config, for models of different sizes
    model_specific_config = get_model_config(config['size'], pretrain=False)
    config = {**model_specific_config, **config}

    # -- Find path to checkpoint directory - create the directory if it doesn't exist
    base_path = Path(__file__).parent
    checkpoint_name = args.checkpoint
    selected_dataset = args.dataset.lower()

    base_checkpoint_dir = (base_path / '../checkpoints').resolve()
    pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
    finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()
    dataset_dir = (base_checkpoint_dir / '../datasets').resolve()

    # create the finetune directory if it doesn't exist already
    Path(finetune_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # -- Set device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(config["device"].upper()))

    # get pre-trained model from which to begin fine-tuning
    electra_model, _, _, electra_tokenizer, _, model_config, disc_config = \
        build_pretrained_from_checkpoint(config['size'], config['device'], pretrain_checkpoint_dir, checkpoint_name)

    discriminator = electra_model.discriminator
    generator = electra_model.generator

    electra_for_qa = ElectraForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None,
                                                                 state_dict=discriminator.state_dict(),
                                                                 config=disc_config)

    # -- Load the data and prepare it in squad format
    try:
        dataset_file_name = datasets[selected_dataset]["train"]
    except KeyError:
        raise KeyError("The dataset '{}' in {} does not contain a 'train' key.".format(selected_dataset, datasets))

    try:
        dataset_function = dataset_to_fc[selected_dataset]
    except KeyError:
        raise KeyError("The dataset '{}' is not contained in the dataset_to_fc map.".format(selected_dataset))

    dataset_file_path = (
                base_checkpoint_dir / '../datasets/{}/{}'.format(selected_dataset, dataset_file_name)).resolve()

    print("\nReading raw dataset '{}' into SQuAD examples".format(dataset_file_name))
    read_raw_dataset = dataset_function(dataset_file_path)

    print("Converting raw text to features.".format(dataset_file_name))
    features = convert_samples_to_features(read_raw_dataset, electra_tokenizer, config["max_length"])

    print("Created {} features of length {}.".format(len(features), config["max_length"]))

    train_dataset = SQuADDataset(features)  # not valid - todo change this

    # Random Sampler used during training.
    data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=1) # batch_size=config["batch_size"])

    layerwise_learning_rates = get_layer_lrs(electra_for_qa.named_parameters(),
                                             config["lr"], config["layerwise_lr_decay"], disc_config.num_hidden_layers)

    # Prepare optimizer and schedule (linear warm up and decay)
    no_decay = ["bias", "LayerNorm", "layer_norm"]

    # todo check if lr should be dependent on non-zero wd
    layerwise_params = []
    for n, p in electra_for_qa.named_parameters():
        wd = config["decay"] if not any(nd in n for nd in no_decay) else 0
        lr = layerwise_learning_rates[n]
        layerwise_params.append({"params": [p], "weight_decay": wd, "lr": lr})

    # Create the optimizer and scheduler
    optimizer = AdamW(layerwise_params, eps=config["epsilon"], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(len(data_loader) // config["max_epochs"]) * config["warmup_fraction"],
                                                num_training_steps=-1)
    # todo check whether num_training_steps should be -1


    # ------ START THE FINE-TUNING LOOP ------
    fine_tune(data_loader, electra_for_qa, scheduler, optimizer, config, finetune_checkpoint_dir)

    # # output folder for model checkpoints and predictions
    # save_dir = "./output"
    #
    # # DECIDE WHETHER TO TRAIN, EVALUATE, OR BOTH.
    # train_model, evaluate_model = True, True
    #
    # model_info = {"model_path": "google/electra-base-discriminator", "uncased": False}
    # dataset_info = datasets["bioasq"]