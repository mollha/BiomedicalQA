from data_processing import ELECTRADataProcessor, MaskedLM, IterableCSVDataset
from loss_functions import ELECTRALoss
from models import ELECTRAModel, get_model_config, save_checkpoint, load_checkpoint, build_electra_model
from hugdatafast import *
import numpy as np
import os
import torch
from torch import nn
from tqdm import trange
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
import sys

# ------------------ SPECIFY GENERAL MODEL CONFIG ------------------
config = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'seed': 0,
    'adam_bias_correction': False,
    'generator_loss': [],
    'discriminator_loss': [],
    'size': 'large',  # electra small too small for QA
    'num_workers': 3 if torch.cuda.is_available() else 0,
    "max_epochs": 9999,
    "current_epoch": 0,  # track the current epoch in config for saving checkpoints
    "steps_trained": 0,  # track the steps trained in config for saving checkpoints
    "global_step": -1,  # total steps over all epochs
    "update_steps": 20000,
}


# ----------------- HELPER FUNCTIONS --------------------
def set_seed(seed_value: int) -> None:
    """
    Fix a seed for reproducability.
    :param seed_value: seed to set
    :return: None
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def update_settings(settings: dict, update: dict) -> dict:
    """
    Override config in settings dict with config in update dict. This allows
    model specific config to be merged with general training settings to create
    a single dictionary containing configuration.
    :param settings: dictionary containing general model settings
    :param update: dictionary containing update settings.
    :return: merged config dictionary
    """
    for key, value in update.items():
        settings[key] = value

    return settings


def get_recent_checkpoint(directory, subfolders: list):
    """
    Find the name of the most advanced model checkpoint saved in the checkpoints directory.
    This is the model checkpoint that has been trained the most, so it is the best candidate to
    start from if no specific checkpoint name was provided to the pre-training loop.
    :param directory: directory containing model checkpoints.
    :param subfolders: list of checkpoint directories
    :return:
    """
    directory = str(directory)

    def parse_name(subdir: str):
        config_str = str(subdir)[str(subdir).find(directory) + len(directory):]
        first_undsc, second_undsc = config_str.find('_'), config_str.rfind('_')
        return int(config_str[first_undsc + 1: second_undsc]), int(config_str[second_undsc + 1:])

    max_file, max_epoch, max_step_in_epoch = None, None, None
    for subdirectory in subfolders:
        epoch, step = parse_name(subdirectory)

        if max_epoch is None or epoch > max_epoch:
            max_epoch = epoch
            max_step_in_epoch = step
            max_file = subdirectory
        elif epoch == max_epoch:
            if step > max_step_in_epoch:
                max_step_in_epoch = step
                max_file = subdirectory
    return max_file


# ---------- DEFINE MAIN PRE-TRAINING LOOP ----------
def pre_train(dataset, model, scheduler, optimizer, settings, checkpoint_name="recent"):
    """ Train the model """
    # Specify which directory model checkpoints should be saved to.
    # Make the checkpoint directory if it does not exist already.
    checkpoint_dir = (base_path / 'checkpoints/pretrain').resolve()
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    loss_function = ELECTRALoss()

    model.to(settings["device"])

    #   -------- DETERMINE WHETHER TRAINING FROM A CHECKPOINT OR FROM SCRATCH --------
    valid_checkpoint, path_to_checkpoint = False, None

    if checkpoint_name.lower() == "recent":

        subfolders = [x for x in Path(checkpoint_dir).iterdir() \
                      if x.is_dir() and settings["size"] in str(x)[str(x).rfind('/') + 1:]]

        print("Subfolders", subfolders)

        if len(subfolders) > 0:
            path_to_checkpoint = get_recent_checkpoint(checkpoint_dir, subfolders)
            print("Pre-training from the most advanced checkpoint - {}\n".format(path_to_checkpoint))
            valid_checkpoint = True
    elif checkpoint_name:
        path_to_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(path_to_checkpoint):
            print(
                "Checkpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
            # if the directory with the checkpoint name exists, we can retrieve the correct config from here
            valid_checkpoint = True
        else:
            print(
                "WARNING: Checkpoint {} does not exist at path {}.\n".format(checkpoint_name, path_to_checkpoint))

    if valid_checkpoint:
        model, optimizer, scheduler, loss_function, new_settings = load_checkpoint(path_to_checkpoint, model, optimizer,
                                                                                   scheduler, settings["device"])
        settings = update_settings(settings, new_settings)
    else:
        print("Pre-training from scratch - no checkpoint provided.\n")

    # ------------------ PREPARE TO START THE TRAINING LOOP ------------------
    print("\n---------- BEGIN TRAINING ----------")
    sys.stderr.write("\nDevice = {}\nModel Size = {}\nTotal Epochs = {}\nStart training from Epoch = {}\nStart training from Step = {}\nBatch size = {}\nCheckpoint Steps = {}\nMax Sample Length = {}\n\n"
                     .format(settings["device"].upper(), settings["size"], settings["max_epochs"], settings["current_epoch"],
                             settings["steps_trained"], settings["batch_size"], settings["update_steps"],
                             settings["max_length"]))
    total_training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # Added here for reproducibility
    set_seed(settings["seed"])

    # Resume training from the epoch we left off at earlier.
    train_iterator = trange(settings["current_epoch"], int(settings["max_epochs"]), desc="Epoch")

    # # 2. Masked language model objective
    mlm = MaskedLM(mask_tok_id=electra_tokenizer.mask_token_id,
                   special_tok_ids=electra_tokenizer.all_special_ids,
                   vocab_size=electra_tokenizer.vocab_size,
                   mlm_probability=config["mask_prob"],
                   replace_prob=0.0,
                   original_prob=0.15)

    # resume training
    steps_trained = settings["steps_trained"]

    for epoch_number in train_iterator:
        iterable_dataset = iter(dataset)
        iterable_dataset.resume_from_step(steps_trained)

        # update the current epoch
        settings["current_epoch"] = epoch_number  # update the number of epochs
        for training_step in range(settings["max_steps"]):
            batch = next(iterable_dataset)

            if batch is None:
                print("Reached the end of the dataset")
                break

            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue

            batch = batch.to(settings["device"])  # project batch to correct device
            inputs, targets = mlm.mask_batch(batch)  # mask the batch before passing it to the model

            model.train()  # train model one step
            outputs = model(*inputs)  # inputs = (masked_inputs, is_mlm_applied, labels)

            loss = loss_function(outputs, *targets)  # targets = (labels,)
            loss.backward()
            total_training_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # perform steps
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            settings["steps_trained"] = training_step
            settings["global_step"] += 1

            # Log metrics
            if settings["global_step"] > 0 and settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                # Evaluate all checkpoints starting with same prefix as model_name ending and ending with step number

                print("{} steps trained in current epoch, {} steps trained overall."
                                 .format(settings["steps_trained"], settings["global_step"]))

                # Save model checkpoint
                save_checkpoint(model, optimizer, scheduler, loss_function, settings, checkpoint_dir)

        loss_function.update_statistics()  # update the loss function statistics before saving loss fc with checkpoint
        save_checkpoint(model, optimizer, scheduler, loss_function, settings, checkpoint_dir)


# ---------- PREPARE OBJECTS AND SETTINGS FOR MAIN PRE-TRAINING LOOP ----------
if __name__ == "__main__":
    # Log Process ID
    print(f"Process ID: {os.getpid()}\n")

    # Override general config with model specific config, for models of different sizes
    model_specific_config = get_model_config(config['size'])
    config = {**model_specific_config, **config}

    # Set torch backend and set seed
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])

    base_path = Path(__file__).parent
    generator, discriminator, electra_tokenizer = build_electra_model(config['size'])
    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)

    # Prepare optimizer and schedule (linear warm up and decay)
    # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01
    optimizer = AdamW(electra_model.parameters(), eps=1e-6, weight_decay=0.01, lr=config["lr"],
                      correct_bias=config["adam_bias_correction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000,
                                                num_training_steps=config["max_steps"])

    # ------ PREPARE DATA FOR NETWORK CONSUMPTION ------
    data_pre_processor = ELECTRADataProcessor(tokenizer=electra_tokenizer, max_length=config["max_length"])
    csv_data_dir = (base_path / '../datasets/PubMed/processed_data').resolve()
    print('\nLoading data from {} and initialising Pytorch Dataset.\n'.format(csv_data_dir))
    dataset = IterableCSVDataset(csv_data_dir, config["batch_size"], config["device"], transform=data_pre_processor)

    # ------ START THE PRE-TRAINING LOOP ------
    pre_train(dataset, electra_model, scheduler, optimizer, config)
