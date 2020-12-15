from data_processing import ELECTRADataProcessor, MaskedLM, IterableCSVDataset
from loss_functions import ELECTRALoss
from models import ELECTRAModel, get_model_config, save_checkpoint, load_checkpoint
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
import numpy as np
import os
import torch
from torch import nn
from tqdm import trange
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
import sys


# define config here
config = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'seed': 0,
    'adam_bias_correction': False,
    'electra_mask_style': True,
    'generator_loss': [],
    'discriminator_loss': [],
    'size': 'small',
    'num_workers': 3 if torch.cuda.is_available() else 0,
    "training_epochs": 9999,    # todo change this for proper training 9999,
    "batch_size": 128,
    "current_epoch": 0,
    "steps_trained": 0,
    "global_step": -1,   # total steps over all epochs
    "update_steps": 20000,  # Save checkpoint and log every X updates steps. - based on rate of NCC (1000 steps every 12 mins)
    "analyse_all_checkpoints": True
}


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def update_settings(settings, update):
    for key, value in update.items():
        settings[key] = value

    return settings


def get_recent_checkpoint(directory, subfolders):
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


def pre_train(dataset, model, scheduler, optimizer, settings, checkpoint_name="recent"):
    """ Train the model """
    # Specify which directory model checkpoints should be saved to.
    # Make the checkpoint directory if it does not exist already.
    checkpoint_dir = (base_path / 'checkpoints/pretrain').resolve()
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    loss_function = ELECTRALoss()

    model.to(settings["device"])

    valid_checkpoint = False
    path_to_checkpoint = None

    if checkpoint_name.lower() == "recent":
        subfolders = [x for x in Path(checkpoint_dir).iterdir() if x.is_dir()]
        if len(subfolders) > 0:
            path_to_checkpoint = get_recent_checkpoint(checkpoint_dir, subfolders)
            sys.stderr.write("\nPre-training from the most advanced checkpoint - {}".format(path_to_checkpoint))
            valid_checkpoint = True
    elif checkpoint_name:
        path_to_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(path_to_checkpoint):
            sys.stderr.write("Checkpoint '{}' exists - Loading config values from memory.".format(path_to_checkpoint))
            # if the directory with the checkpoint name exists, we can retrieve the correct config from here
            valid_checkpoint = True
        else:
            sys.stderr.write("WARNING: Checkpoint {} does not exist at path {}.".format(checkpoint_name, path_to_checkpoint))

    if valid_checkpoint:
        model, optimizer, scheduler, loss_function, new_settings = load_checkpoint(path_to_checkpoint, model, optimizer,
                                                                                   scheduler, settings["device"])
        settings = update_settings(settings, new_settings)
    else:
        sys.stderr.write("Pre-training from scratch - no checkpoint provided.")

    sys.stderr.write("Save model checkpoints every {} steps.".format(settings["update_steps"]))
    sys.stderr.write("\n---------- BEGIN TRAINING ----------")
    sys.stderr.write("Current Epoch = {}\nTotal Epochs = {}\nBatch size = {}\n"
          .format(settings["current_epoch"], settings["training_epochs"], settings["batch_size"]))

    total_training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # Added here for reproducibility
    set_seed(settings["seed"])

    # Resume training from the epoch we left off at earlier.
    train_iterator = trange(settings["current_epoch"], int(settings["training_epochs"]), desc="Epoch")

    # # 2. Masked language model objective
    mlm = MaskedLM(mask_tok_id=electra_tokenizer.mask_token_id,
                   special_tok_ids=electra_tokenizer.all_special_ids,
                   vocab_size=electra_tokenizer.vocab_size,
                   mlm_probability=config["mask_prob"],
                   replace_prob=0.0 if config["electra_mask_style"] else 0.1,
                   orginal_prob=0.15 if config["electra_mask_style"] else 0.1)

    # resume training
    steps_trained = settings["steps_trained"]

    for epoch_number in train_iterator:
        iterable_dataset = iter(dataset)
        iterable_dataset.resume_from_step(steps_trained)

        settings["current_epoch"] = epoch_number  # update the number of epochs
        for training_step in range(settings["max_steps"]):
            batch = next(iterable_dataset)

            if batch is None:
                sys.stderr.write("Reached the end of the dataset")
                break

            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue

            batch = batch.to(settings["device"])
            inputs, targets = mlm.mask_batch(batch)

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

                sys.stderr.write("{} steps trained in current epoch, {} steps trained overall."
                      .format(settings["steps_trained"], settings["global_step"]))

                # Save model checkpoint
                save_checkpoint(model, optimizer, scheduler, loss_function, settings, checkpoint_dir)

        save_checkpoint(model, optimizer, scheduler, loss_function, settings, checkpoint_dir)
        loss_function.update_statistics()

    # ------------- SAVE FINE-TUNED MODEL ONCE MORE AT THE END OF TRAINING -------------
    save_checkpoint(model, optimizer, scheduler, settings, checkpoint_dir)


if __name__ == "__main__":
    base_path = Path(__file__).parent
    # Merge general config with model specific config
    # Setting of different sizes
    model_specific_config = get_model_config(config['size'])
    config = {**config, **model_specific_config}

    # ------ DEFINE GENERATOR AND DISCRIMINATOR CONFIG ------
    discriminator_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-discriminator')
    generator_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-generator')

    # note that public electra-small model is actually small++ and don't scale down generator size
    generator_config.hidden_size = int(discriminator_config.hidden_size / config["generator_size_divisor"])
    generator_config.num_attention_heads = discriminator_config.num_attention_heads // config["generator_size_divisor"]
    generator_config.intermediate_size = discriminator_config.intermediate_size // config["generator_size_divisor"]

    electra_tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-{config["size"]}-generator')

    # Path to data
    Path((base_path / '../datasets').resolve(), exist_ok=True)

    # Print info
    sys.stderr.write(f"process id: {os.getpid()}")
    pre_processor = ELECTRADataProcessor(tokenizer=electra_tokenizer, max_length=config["max_length"],
                                         device=config["device"])

    sys.stderr.write('Load in the dataset.')
    csv_data_dir = (base_path / '../datasets/PubMed/processed_data').resolve()
    dataset = IterableCSVDataset(csv_data_dir, config["batch_size"], config["device"], transform=pre_processor)

    # # 5. Train - Seed & PyTorch benchmark
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])

    # Generator and Discriminator
    generator = ElectraForMaskedLM(generator_config).from_pretrained(f'google/electra-{config["size"]}-generator')
    discriminator = ElectraForPreTraining(discriminator_config).from_pretrained(f'google/electra-{config["size"]}-discriminator')
    discriminator.electra.embeddings = generator.electra.embeddings
    generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)

    # Prepare optimizer and schedule (linear warm up and decay)
    # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01
    optimizer = AdamW(electra_model.parameters(), eps=1e-6, weight_decay=0.01, lr=config["lr"],
                      correct_bias=config["adam_bias_correction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=config["max_steps"])
    pre_train(dataset, electra_model, scheduler, optimizer, config)
