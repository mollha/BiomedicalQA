from data_processing import newELECTRADataProcessor, ELECTRADataProcessor, MaskedLM, CSVDataset
from loss_functions import ELECTRALoss
from models import ELECTRAModel, get_model_config, save_checkpoint, load_checkpoint
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
import pickle

import os
from time import time
import torch
from torch import save, ones, int64, nn, no_grad
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult
from torch.utils.tensorboard import SummaryWriter

import datetime

now = datetime.datetime.now()
today = datetime.date.today()

# define config here
config = {
    'device': "cuda:0" if torch.cuda.is_available() else "cpu:0",
    'seed': 0,
    'adam_bias_correction': False,
    'electra_mask_style': True,
    'size': 'small',
    'num_workers': 3 if torch.cuda.is_available() else 0,
    "training_epochs": 9999,
    "batch_size": 30,
    "epochs_trained": 0,
    "steps_trained": 0,
    "global_step": 0,   # total steps over all epochs
    "update_steps": 10,  # Save checkpoint and log every X updates steps.
    "analyse_all_checkpoints": True
}


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def analyse_checkpoint(tb_writer, model, settings, total_training_loss):
    tb_writer.add_scalar("checkpoint_{}", settings["epochs_trained"])
    tb_writer.add_scalar("Avg. training loss = {}", total_training_loss / settings["global_step"])


def update_settings(settings, update):
    for key, value in update.items():
        settings[key] = value

    return settings


def pre_train(data_loader, model, tokenizer, scheduler, optimizer, settings, checkpoint_name=""):
    """ Train the model """
    # Specify which directory model checkpoints should be saved to.
    # Make the checkpoint directory if it does not exist already.
    checkpoint_dir = "./checkpoints/pretrain"
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    model.to(settings["device"])

    path_to_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    if checkpoint_name and os.path.exists(path_to_checkpoint):
        print("\nCheckpoint '{}' exists - loading config values from memory.".format(checkpoint_name))
        # if the directory with the checkpoint name exists, we can retrieve the correct config from here
        model, optimizer, scheduler, new_settings = load_checkpoint(path_to_checkpoint, model, optimizer, scheduler)
        settings = update_settings(settings, new_settings)

    elif checkpoint_name:
        print("WARNING: Checkpoint {} does not exist at path {}.".format(checkpoint_name, path_to_checkpoint))

    print("Save model checkpoints every {} steps.".format(settings["update_steps"]))
    print("\n---------- BEGIN TRAINING ----------")
    print("Dataset Size = {}\nNumber of Epochs = {}\nBatch size = {}\n"
          .format(config["sample_size"], settings["training_epochs"], settings["batch_size"]))

    total_training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # Added here for reproducibility
    # TODO SET SEED HERE AGAIN

    tb_writer = SummaryWriter()  # Create a SummaryWriter()

    # Resume training from the epoch we left off at earlier.
    train_iterator = trange(settings["epochs_trained"], int(settings["training_epochs"]), desc="Epoch")

    loss_function = ELECTRALoss()

    # # 2. Masked language model objective
    mlm = MaskedLM(mask_tok_id=electra_tokenizer.mask_token_id,
                   special_tok_ids=electra_tokenizer.all_special_ids,
                   vocab_size=electra_tokenizer.vocab_size,
                   mlm_probability=config["mask_prob"],
                   replace_prob=0.0 if config["electra_mask_style"] else 0.1,
                   orginal_prob=0.15 if config["electra_mask_style"] else 0.1)

    steps_trained = settings["steps_trained"]

    for epoch_number in train_iterator:
        print('Number of Epochs', len(train_iterator))
        epoch_iterator = tqdm(data_loader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue

            batch = (batch,)
            inputs, targets = mlm.mask_batch(batch)

            # train model one step
            model.train()

            # inputs = (masked_inputs, sent_lengths, is_mlm_applied, labels),  targets = (labels,)
            outputs = model(*inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = loss_function(outputs, *targets)

            loss.backward()
            total_training_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            # Log metrics
            if settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"

                if settings["analyse_all_checkpoints"]:
                    # pass some objects over to be analysed
                    # I am not sure how this will be implemented yet, but will allow us to write some data to
                    # tensorboard about the state of pre-trained checkpoints.
                    analyse_checkpoint(tb_writer, model, settings, total_training_loss)

                tb_writer.add_scalar("lr", scheduler.get_lr()[0], settings["global_step"])
                tb_writer.add_scalar("loss", (total_training_loss - logging_loss) / settings["update_steps"], settings["global_step"])
                logging_loss = total_training_loss

            settings["steps_trained"] = step
            settings["global_step"] += 1

            print("Steps trained", settings["steps_trained"])
            print("Global steps", settings["global_step"])

            # Save model checkpoint
            if settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                save_checkpoint(model, optimizer, scheduler, settings, checkpoint_dir)

        # update the number of epochs that have passed.
        settings["epochs_trained"] = epoch_number
    tb_writer.close()

    # ------------- SAVE FINE-TUNED MODEL ONCE MORE AT THE END OF TRAINING -------------
    save_checkpoint(model, optimizer, scheduler, settings, checkpoint_dir)


if __name__ == "__main__":
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
    Path('../datasets', exist_ok=True)
    Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)
    edl_cache_dir = Path("../datasets/electra_dataloader")
    edl_cache_dir.mkdir(exist_ok=True)

    # Print info
    print(f"process id: {os.getpid()}")

    newELECTRADataProcessor = newELECTRADataProcessor(tokenizer=electra_tokenizer, max_length=config["max_length"],
                               device=config["device"])

    print('Load in the dataset.')
    # todo see if its possible to implement caching? e.g. cache_dir='../datasets' in huggingface datasets
    dataset = CSVDataset('./datasets/fibro_abstracts.csv', transform=newELECTRADataProcessor)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"])

    # # 5. Train - Seed & PyTorch benchmark
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])

    # Generator and Discriminator
    generator = ElectraForMaskedLM(generator_config)
    discriminator = ElectraForPreTraining(discriminator_config)
    discriminator.electra.embeddings = generator.electra.embeddings
    generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)
    # dls.to(torch.device(config["device"]))

    config["sample_size"] = len(dataset)

    # Prepare optimizer and schedule (linear warm up and decay)
    # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01
    optimizer = AdamW(electra_model.parameters(), eps=1e-6, weight_decay=0.01, lr=config["lr"],
                      correct_bias=config["adam_bias_correction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=config["steps"])

    pre_train(data_loader, electra_model, electra_tokenizer, scheduler, optimizer, config, checkpoint_name="01_12_20.16_20_03")
