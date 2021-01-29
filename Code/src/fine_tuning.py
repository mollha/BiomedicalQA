import torch
import argparse
import os
from read_data import dataset_to_fc

from functools import partial
import sys
from pathlib import Path
from tqdm import trange
from tqdm import tqdm

from torch import nn
from models import *
import numpy as np
from utils import *
from glob import glob
import tokenizers
from data_processing import convert_samples_to_features, SQuADDataset, collate_wrapper
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor, SquadExample
from transformers import AdamW, get_linear_schedule_with_warmup
from pre_training import build_pretrained_from_checkpoint
from torch.utils.data import DataLoader, RandomSampler
# from run_factoid import train, evaluate

# print(tokenizers.__version__) # returns 0.9.4 (latest compatible version)

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
    'seed': 0,
    'loss': [],
    'num_workers': 3 if torch.cuda.is_available() else 0,
    "max_epochs": 2,  # can override the val in config
    "current_epoch": 0,  # track the current epoch in config for saving checkpoints
    "steps_trained": 0,  # track the steps trained in config for saving checkpoints
    "global_step": -1,  # total steps over all epochs
    "update_steps": 50,
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



def build_finetuned_from_checkpoint(model_size, device, checkpoint_directory, checkpoint_name, config={}):

    # create the checkpoint directory if it doesn't exist
    Path(checkpoint_directory).mkdir(exist_ok=True, parents=True)

    # -- Override general config with model specific config, for models of different sizes
    model_settings = get_model_config(model_size)
    generator, discriminator, electra_tokenizer = build_electra_model(model_size)
    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)

    # Prepare optimizer and schedule (linear warm up and decay)
    # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01
    optimizer = AdamW(electra_model.parameters(), eps=1e-6, weight_decay=0.01, lr=model_settings["lr"],
                      correct_bias=model_settings["adam_bias_correction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000,
                                                num_training_steps=model_settings["max_steps"])

    #   -------- DETERMINE WHETHER TRAINING FROM A CHECKPOINT OR FROM SCRATCH --------
    valid_checkpoint, path_to_checkpoint = False, None

    if checkpoint_name.lower() == "recent":

        subfolders = [x for x in Path(checkpoint_directory).iterdir() \
                      if x.is_dir() and model_size in str(x)[str(x).rfind('/') + 1:]]

        if len(subfolders) > 0:
            path_to_checkpoint = get_recent_checkpoint_name(checkpoint_directory, subfolders)
            print("\nTraining from the most advanced checkpoint - {}\n".format(path_to_checkpoint))
            valid_checkpoint = True
    elif checkpoint_name:
        path_to_checkpoint = os.path.join(checkpoint_directory, checkpoint_name)
        if os.path.exists(path_to_checkpoint):
            print(
                "Checkpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
            # if the directory with the checkpoint name exists, we can retrieve the correct config from here
            valid_checkpoint = True
        else:
            print(
                "WARNING: Checkpoint {} does not exist at path {}.\n".format(checkpoint_name, path_to_checkpoint))

    if valid_checkpoint:
        electra_model, optimizer, scheduler, loss_function,\
        new_config = load_checkpoint(path_to_checkpoint, electra_model, optimizer, scheduler, device)

        config = update_settings(config, new_config, exceptions=["update_steps", "device"])

    else:
        print("\nTraining from scratch - no checkpoint provided.\n")

    return electra_model, optimizer, scheduler, electra_tokenizer, loss_function, config

def fine_tune(train_dataloader, qa_model, scheduler, optimizer, settings, checkpoint_dir):
    qa_model.to(settings["device"])

    # ------------------ PREPARE TO START THE TRAINING LOOP ------------------
    sys.stderr.write("\n---------- BEGIN FINE-TUNING ----------")
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
        step_iterator = tqdm(data_loader, desc="Step")

        # update the current epoch
        settings["current_epoch"] = epoch_number  # update the number of epochs

        for training_step, batch in enumerate(step_iterator):
            question_ids = batch.question_ids
            is_impossible = batch.is_impossible

            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue  # skip this step

            # batch = batch.to(settings["device"])  # project batch to correct device
            qa_model.train()  # train model one step

            inputs = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "token_type_ids": batch.token_type_ids,
                "start_positions": batch.answer_start,
                "end_positions": batch.answer_end,
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
                save_checkpoint(qa_model, optimizer, scheduler, settings, checkpoint_dir,
                                pre_training=False)

    # todo update loss function statistics
    # ------------- SAVE FINE-TUNED MODEL -------------
    save_checkpoint(qa_model, optimizer, scheduler, settings, checkpoint_dir, pre_training=False)


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
        help="The name of the pre-training, or fine-tuning, checkpoint to use e.g. small_15_10230",
    )
    parser.add_argument(
        "--finetuned",
        default=False,
        type=bool,
        help="Whether training from fine-tuned checkpoint or from pretrained checkpoint",
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
    from_finetuned = args.finetuned

    sys.stderr.write("Selected checkpoint {} and model size {}".format(args.checkpoint, args.size))
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

    # create the fine-tune directory if it doesn't exist already
    Path(finetune_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # -- Set device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}\n".format(config["device"].upper()))

    # get basic model building blocks
    generator, discriminator, electra_tokenizer, discriminator_config = build_electra_model(config['size'], get_config=True)

    # -- Load the data and prepare it in squad format
    try:
        dataset_file_name = datasets[selected_dataset]["train"]
    except KeyError:
        raise KeyError("The dataset '{}' in {} does not contain a 'train' key.".format(selected_dataset, datasets))

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
    train_dataset = SQuADDataset(features)  # not valid - todo change this

    # Random Sampler used during training.
    data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config["batch_size"],
                             collate_fn=collate_wrapper)

    # ------ LOAD MODEL FROM PRE-TRAINED CHECKPOINT OR FROM FINE-TUNED CHECKPOINT ------
    # get pre-trained model from which to begin fine-tuning
    layerwise_learning_rates = get_layer_lrs(discriminator.named_parameters(), config["lr"],
                                             config["layerwise_lr_decay"], discriminator_config.num_hidden_layers)

    no_decay = ["bias", "LayerNorm", "layer_norm"]  # Prepare optimizer and schedule (linear warm up and decay)

    layerwise_params = []  # todo check if lr should be dependent on non-zero wd
    for n, p in discriminator.named_parameters():
        wd = config["decay"] if not any(nd in n for nd in no_decay) else 0
        lr = layerwise_learning_rates[n]
        layerwise_params.append({"params": [p], "weight_decay": wd, "lr": lr})

    # Create the optimizer and scheduler
    optimizer = AdamW(layerwise_params, eps=config["epsilon"], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(len(data_loader) // config["max_epochs"]) *
                                                                 config["warmup_fraction"],
                                                num_training_steps=-1)  # todo check whether num_training_steps should be -1

    if from_finetuned:  # load the fine-tuned checkpoint
        discriminator, optimizer, scheduler, settings = load_checkpoint(path_to_checkpoint, discriminator, optimizer,
                                                                        scheduler, config["device"], pre_training=False)
    else:
        pretrained_model, _, _, electra_tokenizer, _, model_config = build_pretrained_from_checkpoint(config['size'],
                                                                                                   config['device'],
                                                                                                   pretrain_checkpoint_dir,
                                                                                                   checkpoint_name)

        discriminator = pretrained_model.discriminator


    electra_for_qa = ElectraForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None,
                                                                 state_dict=discriminator.state_dict(),
                                                                 config=discriminator_config)





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