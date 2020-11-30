from data_processing import ELECTRADataProcessor, MaskedLM
from loss_functions import ELECTRALoss
from models import ELECTRAModel, get_model_config
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


# Save checkpoint and log every X updates steps.
def save_model(model, tokenizer, optimizer, scheduler, settings, total_training_loss, checkpoint_dir):

    # NOW THAT WE HAVE LOADED IN STATES FROM PREVIOUS CHECKPOINT - WE NEED TO CREATE A NEW CHECKPOINT NAME
    current_day = today.strftime("%d_%m_%y")
    current_time = now.strftime("%H_%M_%S")
    checkpoint_name = current_day + '.' + current_time

    # ------------- SAVE FINE-TUNED TOKENIZER AND MODEL -------------
    save_dir = os.path.join(checkpoint_dir, checkpoint_name)

    # Take care of distributed/parallel training
    # saving_model = model.module if hasattr(model, "module") else model
    # saving_model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)

    # save training settings with trained model
    save(settings, os.path.join(save_dir, "train_settings.bin"))
    save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    print("Saving model checkpoint, optimizer and scheduler states to {}".format(save_dir))
    # print("Avg. training loss = {}".format(total_training_loss / settings["global_step"]))


def update_settings(settings, update):
    for key, value in update.items():
        settings[key] = value

    return settings


def pre_train(data_loader, model, tokenizer, scheduler, optimizer, settings, checkpoint_name=""):
    """ Train the model """
    # Specify which directory model checkpoints should be saved to.
    # Make the checkpoint directory if it does not exist already.
    checkpoint_dir = "./checkpoints"
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    model.to(settings["device"])

    path_to_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    if checkpoint_name and os.path.exists(path_to_checkpoint):
        # if the directory with the checkpoint name exists, we can retrieve the correct config from here
        # Load in optimizer, tokenizer and scheduler states

        path_to_optimizer = os.path.join(path_to_checkpoint, "optimizer.pt")
        if os.path.isfile(path_to_optimizer):
            optimizer.load_state_dict(torch.load(path_to_optimizer))

        # path_to_tokenizer = os.path.join(path_to_checkpoint, "tokenizer.pt")
        # if os.path.isfile(path_to_tokenizer):
        #     tokenizer.load_state_dict(torch.load(path_to_tokenizer))

        path_to_scheduler = os.path.join(path_to_checkpoint, "scheduler.pt")
        if os.path.isfile(path_to_scheduler):
            scheduler.load_state_dict(torch.load(path_to_scheduler))

        path_to_model = os.path.join(path_to_checkpoint, "model.pt")
        if os.path.isfile(path_to_model):
            model.load_state_dict(torch.load(path_to_model))

        new_settings = torch.load(os.path.join(path_to_checkpoint, "train_settings.bin"))
        settings = update_settings(settings, new_settings)

        if settings["epochs_trained"] > 0:
            print("Continuing training from checkpoint, will skip to saved global_step")
            print("Continuing training from epoch {}.".format(settings["epochs_trained"]))

        if settings["steps_trained"] > 0:
            print("Skip the first {} steps in the first epoch", settings["steps_trained"])

    print("\n---------- BEGIN TRAINING ----------")
    print("Dataset Size = {}\nNumber of Epochs = {}\nBatch size = {}\n"
          .format(config["sample_size"], settings["training_epochs"], settings["batch_size"]))
    # todo check that replacing len(dataset) with len(data_loader) is a fair / valid exchange

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


    for epoch_number in train_iterator:
        print('Max epochs', len(train_iterator))
        epoch_iterator = tqdm(data_loader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if settings["steps_trained"] > 0:
                settings["steps_trained"] -= 1
                continue

            # MASK INPUTS HERE

            # train model one step
            model.train()
            # print(batch)
            #
            # print("Length of batch", len(batch))
            # print("Type of batch", type(batch))
            #
            # print("Length of first element in batch", len(batch[0]))
            # print("Type of first element in batch", type(batch[0]))
            #
            # print("Length of first element in batch", len(batch[1]))
            # print("Type of first element in batch", type(batch[1]))

            batch = tuple(t.to(settings["device"]) for t in batch)
            print(batch)
            # batch is probably not a tuple anymore, since removing sentA_lengths


            # print(len(batch[0]))
            # print("batch size", len(batch))


            # inputs = {'input_ids': batch[0], 'sentA_length': batch[0]}
            # inputs = (masked_inputs, sent_lengths, is_mlm_applied, labels),  targets = (labels,)
            inputs, targets = mlm.mask_batch(batch)

            outputs = model(*inputs)

            # model outputs are always tuple in transformers (see doc)
            # loss = outputs[0]

            loss = loss_function(outputs, *targets)

            loss.backward()
            total_training_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            # # Log metrics
            # if settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
            #     # Only evaluate when single GPU otherwise metrics may not average well
            #     # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
            #     if settings["evaluate_all_checkpoints"]:
            #         results = evaluate(model, tokenizer, model_info["model_type"], save_dir, device, settings["evaluate_all_checkpoints"], dataset_info)
            #         for key, value in results.items():
            #             tb_writer.add_scalar("eval_{}".format(key), value, settings["global_step"])
            #     tb_writer.add_scalar("lr", scheduler.get_lr()[0], settings["global_step"])
            #     tb_writer.add_scalar("loss", (total_training_loss - logging_loss) / settings["update_steps"], settings["global_step"])
            #     logging_loss = total_training_loss

            # Save model checkpoint
            if settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                save_model(model, tokenizer, optimizer, scheduler, settings, total_training_loss, checkpoint_dir)

    tb_writer.close()

    # ------------- SAVE FINE-TUNED MODEL -------------
    save_model(model, tokenizer, optimizer, scheduler, settings, total_training_loss, checkpoint_dir)


if __name__ == "__main__":
    # define config here
    config = {
        'device': "cuda:0" if torch.cuda.is_available() else "cpu:0",
        # "cuda:0" if torch.cuda.is_available() else "cpu:0",
        'seed': 0,
        'adam_bias_correction': False,
        'schedule': 'original_linear',
        'electra_mask_style': True,
        'size': 'small',
        'num_workers': 3 if torch.cuda.is_available() else 0,  # this might be wrong - it initially was just 3
        "training_epochs": 9999,
        "batch_size": 30,
        "epochs_trained": 0,
        "steps_trained": 0,
        "update_steps": 500
    }

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

    # creating this partial function is the first place that electra_tokenizer is used.
    ELECTRAProcessor = partial(ELECTRADataProcessor, tokenizer=electra_tokenizer, max_length=config["max_length"])


    print('Load in the dataset.')
    dataset = datasets.load_dataset('csv', cache_dir='../datasets', data_files='./datasets/fibro_abstracts.csv')[
        'train']

    print('Create or load cached ELECTRA-compatible data.')
    # apply_cleaning is true by default e.g. ELECTRAProcessor(dataset, apply_cleaning=False) if no cleaning
    electra_dataset = ELECTRAProcessor(dataset).map(
        cache_file_name=f'./electra_custom_dataset_{config["max_length"]}.arrow', num_proc=1)

    print(electra_dataset[0])
    print(electra_dataset[1])
    print(electra_dataset[2])

    # hf_dsets = HF_Datasets({"train": electra_dataset}, cols={'input_ids': TensorText, 'sentA_length': noop},
    #                        hf_toker=electra_tokenizer, n_inp=2)

    hf_dsets = HF_Datasets({"train": electra_dataset}, cols={'input_ids': TensorText},
                           hf_toker=electra_tokenizer, n_inp=1)

    print(electra_dataset[0], dataset[0])
    print(electra_dataset[1], dataset[1])
    # Random Sampler used during training.

    data_loader = DataLoader(electra_dataset, shuffle=True, batch_size=config["batch_size"])


    dl = hf_dsets.dataloaders(bs=config["batch_size"], num_workers=config["num_workers"], pin_memory=False,
                               shuffle_train=True,
                               srtkey_fc=False,
                               cache_dir='../datasets/electra_dataloader', cache_name='dl_{split}.json')[0]

    print(dl.one_batch())
    # print(next(iter(data_loader)))

    # # 5. Train
    # Seed & PyTorch benchmark
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


    def set_seed(seed_value):
        # dls[0].rng = random.Random(seed_value)  # for fastai dataloader
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)


    set_seed(config["seed"])

    # Generator and Discriminator
    generator = ElectraForMaskedLM(generator_config)
    discriminator = ElectraForPreTraining(discriminator_config)
    discriminator.electra.embeddings = generator.electra.embeddings
    generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

    # ELECTRA training loop
    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)

    # Learner
    # dls.to(torch.device(config["device"]))

    # Prepare optimizer and schedule (linear warm up and decay)
    # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01

    # dl = dls[0]

    config["sample_size"] = len(dataset)
    print('DATASET SIZE: ', len(dataset))
    print('DATASET SIZE AFTER ELECTRAFYING: ', len(electra_dataset))

    print("steps", config["steps"])


    optimizer = AdamW(electra_model.parameters(), eps=1e-6, weight_decay=0.01, lr=config["lr"],
                      correct_bias=config["adam_bias_correction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=config["steps"])

    pre_train(dl, electra_model, electra_tokenizer, scheduler, optimizer, config, checkpoint_name="")