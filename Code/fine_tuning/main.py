from torch import save, load
import torch
import random
import os
import sys
from pathlib import Path
import numpy as np
from glob import glob
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from ..pre_training.models import build_electra_model
from run_factoid import train, evaluate

from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForQuestionAnswering,
    ElectraConfig,
    ElectraForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)

# Ensure that lowercase model is used for model_type
# ------------- DEFINE TRAINING AND EVALUATION SETTINGS -------------
config = {
    "batch_size": 8,
    "epochs": 1,
    "learning_rate": 8e-6,  # The initial learning rate for Adam.
    "decay": 0.0,  # Weight decay if we apply some.
    "epsilon": 1e-8,  # Epsilon for Adam optimizer.
    "max_grad_norm": 1.0,  # Max gradient norm.
    "evaluate_all_checkpoints": False,
    "update_steps": 500,
    "size": "small"
}

eval_settings = {
    "eval_batch_size": 12,
    "n_best_size": 20,  # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    "max_answer_length": 30,  # maximum length of a generated answer
    "version_2_with_negative": False,  # If true, the SQuAD examples contain some that do not have an answer.
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


def load_pretrained_checkpoint(path_to_checkpoint, device):
    sys.stderr.write("Loading model checkpoint from {}\n".format(path_to_checkpoint))
    settings = torch.load(os.path.join(path_to_checkpoint, "train_settings.bin"))
    model_size = settings["size"]  # get the model size from the checkpoint

    generator, discriminator, electra_tokenizer, disc_config = build_electra_model(model_size, get_config=True)

    path_to_discriminator = os.path.join(path_to_checkpoint, "discriminator.pt")
    if os.path.isfile(path_to_discriminator):
        discriminator.load_state_dict(torch.load(path_to_discriminator, map_location=torch.device(device)))

    sys.stderr.write(
        "Electra model and tokenizer were saved on {} at {}.\n".format(settings["saved_on"], settings["saved_at"]))
    sys.stderr.write("Model was pre-trained for {} epoch(s) and {} step(s).\n"
                     .format(settings["current_epoch"], settings["steps_trained"]))

    disc_state_dict = torch.load(path_to_discriminator, map_location=torch.device(device))
    electra_for_qa = ElectraForQuestionAnswering.from_pretrained(state_dict=disc_state_dict, config=disc_config)
    return electra_for_qa, electra_tokenizer


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
        features_and_dataset = load(cached_features_file)
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
        save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


if __name__ == "__main__":
    base_path = Path(__file__).parent

    # output folder for model checkpoints and predictions
    save_dir = "./output"

    pre_trained_checkpoint_dir = (base_path / '../pre_training/checkpoints/pretrain').resolve()

    # DECIDE WHETHER TO TRAIN, EVALUATE, OR BOTH.
    train_model, evaluate_model = True, True
    model_info = {"model_path": "google/electra-base-discriminator", "uncased": False}
    dataset_info = datasets["bioasq"]

    # device = device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(str(device).upper()))

    set_seed(0)  # fix seed for reproducibility
    model, tokenizer = load_pretrained_model_tokenizer(f'google/electra-{config["size"]}-discriminator',
                                                       uncased_model=False, device=device)

    # Training
    if train_model:
        training_set = load_and_cache_examples(tokenizer, f'google/electra-{config["size"]}-discriminator',
                                               dataset_info["train_file"],
                                               evaluate=False, output_examples=False)

        train(training_set, model, tokenizer, model_info, device, save_dir, config, dataset_info)

    # --------------- LOAD FINE-TUNED MODEL AND VOCAB ---------------
    model = AutoModelForQuestionAnswering.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir, do_lower_case=model_info["uncased"])
    model.to(device)

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
