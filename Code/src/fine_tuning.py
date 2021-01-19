import torch
import random
import argparse
import os
import sys
from pathlib import Path
from tqdm import trange
from models import get_model_config, get_layer_lrs
import numpy as np
from utils import *
from glob import glob
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from transformers import AdamW, get_linear_schedule_with_warmup
from pre_training import build_pretrained_from_checkpoint
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
    "max_epochs": 2, # can override the val in config
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
    "bioasq": {"train_file": "../qa_datasets/QA/BioASQ/BioASQ-train-factoid-7b.json",
               "golden_file": "../qa_datasets/QA/BioASQ/7B_golden.json",
               "official_eval_dir": "./scripts/bioasq_eval"},
}


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def load_pretrained_model_tokenizer(model_path, uncased_model, device):
    # Load pre-trained model and tokenizer
    # pre-trained config name or path if not the same as model_name
    config = AutoConfig.from_pretrained(model_path)
    # model_path same as tokenizer name
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=uncased_model)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, from_tf=bool(".ckpt" in model_path),
                                                          config=config)

    model.to(device)
    return model, tokenizer


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
                print("tensorflow_datasets does not handle version 2 of SQuAD.")

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


# ---------- DEFINE MAIN FINE-TUNING LOOP ----------
# def fine_tune(dataset, model, scheduler, optimizer, settings, checkpoint_name="recent"):
#     pass

# def pre_train(dataset, model, scheduler, tokenizer, optimizer, loss_function, settings, checkpoint_dir):

def fine_tune(test_dataset, qa_model, tokenizer, settings, checkpoint_dir):

    qa_model.to(settings["device"])

    # ------------------ PREPARE TO START THE TRAINING LOOP ------------------
    print("\n---------- BEGIN FINE-TUNING ----------")
    sys.stderr.write(
        "\nDevice = {}\nModel Size = {}\nTotal Epochs = {}\nStart training from Epoch = {}\nStart training from Step = {}\nBatch size = {}\nCheckpoint Steps = {}\nMax Sample Length = {}\n\n"
        .format(settings["device"].upper(), settings["size"], settings["max_epochs"], settings["current_epoch"],
                settings["steps_trained"], settings["batch_size"], settings["update_steps"],
                settings["max_length"]))

    qa_model.zero_grad()

    # evaluate during training always.

    # Resume training from the epoch we left off at earlier.
    train_iterator = trange(settings["current_epoch"], int(settings["max_epochs"]), desc="Epoch")

    for epoch_number in train_iterator:
        iterable_dataset = iter(dataset)


        for training_step in range(settings["max_steps"]):

            # Log metrics
            if settings["global_step"] > 0 and settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                # Evaluate all

                save_pretrained()




if __name__ == "__main__":
    # Log the process ID
    print(f"Process ID: {os.getpid()}")

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

    print(config)

    # -- Find path to checkpoint directory - create the directory if it doesn't exist
    base_path = Path(__file__).parent
    checkpoint_name = args.checkpoint

    base_checkpoint_dir = (base_path / '../checkpoints').resolve()
    pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
    finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()

    # create the finetune directory if it doesn't exist already
    Path(finetune_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # -- Set device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(config["device"].upper()))

    # get pre-trained model from which to begin finetuning
    electra_model, _, _, electra_tokenizer, _, model_config, disc_config =\
        build_pretrained_from_checkpoint(config['size'], config['device'], pretrain_checkpoint_dir, checkpoint_name)

    discriminator = electra_model.discriminator
    generator = electra_model.generator

    electra_for_qa = ElectraForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None,
                                                                 state_dict=discriminator.state_dict(),
                                                                 config=disc_config)

    layerwise_learning_rates = get_layer_lrs(config["lr"], config["layerwise_lr_decay"], disc_config.num_hidden_layers)

    # Prepare optimizer and schedule (linear warm up and decay)
    # no_decay = ["bias", "LayerNorm.weight"]

    # opt_params = [
    #     {
    #         "params": [p for n, p in electra_for_qa.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": settings["decay"],
    #     },
    #     {"params": [p for n, p in electra_for_qa.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]

    "lr": 3e-4,
    "layerwise_lr_decay": 0.8,
    "max_epochs": 2,  # this is the number of epochs typical for squad
    "warmup": 0.1,
    "batch_size": 32,
    "attention_dropout": 0.1,
    "dropout": 0.1,
    "max_length": 128,
    "decay": 0.0,  # Weight decay if we apply some.
    "epsilon": 1e-8,  # Epsilon for Adam optimizer.

    optimizer = AdamW(layerwise_learning_rates, eps=config["epsilon"], weight_decay=config["decay"], lr=config["lr"],
                      correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000,
                                                num_training_steps=model_settings["max_steps"])

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
    #                                             num_training_steps=len(data_loader) // settings["epochs"])
    #


    # Load the dataset and prepare it in squad format


    qa_dataset_squad_format = {}

    # ------ START THE FINE-TUNING LOOP ------
    fine_tune(qa_dataset_squad_format, electra_for_qa, electra_tokenizer, finetune_checkpoint_dir)

    quit()

    # output folder for model checkpoints and predictions
    save_dir = "./output"

    # DECIDE WHETHER TO TRAIN, EVALUATE, OR BOTH.
    train_model, evaluate_model = True, True


    model_info = {"model_path": "google/electra-base-discriminator", "uncased": False}
    dataset_info = datasets["bioasq"]


    # Training
    if train_model:
        training_set = load_and_cache_examples(tokenizer, f'google/electra-{config["size"]}-discriminator',
                                               dataset_info["train_file"],
                                               evaluate=False, output_examples=False)

        train(training_set, model, tokenizer, model_info, device, save_dir, config, dataset_info)

    # --------------- LOAD FINE-TUNED MODEL AND VOCAB ---------------
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if evaluate_model:
        if train_model:
            print("Loading checkpoints saved during training for evaluation")
            checkpoints = [save_dir]

            if config["evaluate_all_checkpoints"]:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob(save_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        else:
            print("Loading checkpoint {} for evaluation".format(model_info["model_path"]))
            checkpoints = [model_info["model_path"]]

        print("Evaluate the following checkpoints: {}".format(checkpoints))

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(device)

            dataset, examples, features = load_and_cache_examples(tokenizer, save_dir, dataset_info["train_file"],
                                                                  evaluate=True, output_examples=True)

            # Evaluate
            evaluate(model, tokenizer, save_dir, device, dataset, examples, features,
                     dataset_info, eval_settings, prefix=global_step)
