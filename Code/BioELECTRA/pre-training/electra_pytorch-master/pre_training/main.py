from callback_functions import MaskedLMCallback, GradientClipping, RunSteps
from data_processing import ELECTRADataProcessor
from loss_functions import ELECTRALoss
from models import ELECTRAModel, get_model_config
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
from _utils.utils import *
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

now = datetime.now()
today = datetime.today()


# Random Sampler used during training.
data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=settings["batch_size"])

def create_training_objects(model):
    """
    Given whatever parameters necessary - return a new optimizer, tokenizer and scheduler to be used in
    the training loop
    :param model:
    :return: tuple
    """

    # Prepare optimizer and schedule (linear warm up and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": settings["decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=settings["epsilon"], lr=settings["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(data_loader) // settings["epochs"])


    return optimizer, tokenizer, scheduler


# Save checkpoint and log every X updates steps.
def save_model(model, tokenizer, optimizer, scheduler, settings, global_step, tr_loss, save_dir):
    # set global_step to global_step of last saved checkpoint from model path

    # ------------- SAVE FINE-TUNED TOKENIZER AND MODEL -------------
    save_dir = os.path.join(save_dir, "checkpoint-{}".format(global_step))

    # Take care of distributed/parallel training
    saving_model = model.module if hasattr(model, "module") else model
    saving_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # save training settings with trained model
    save(settings, os.path.join(save_dir, "train_settings.bin"))
    save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    print("Saving model checkpoint, optimizer and scheduler states to {}".format(save_dir))
    print("global_step = {}, average loss = {}".format(global_step, tr_loss))



def update_settings(settings, update):
    for key, value in update.items():
        settings[key] = value

    return settings



def pre_train(data_loader, model, tokenizer, scheduler, optimizer, model_info, device, settings, dataset_info, checkpoint_name=None):
    """ Train the model """
    # pass in model, data_loader, optimizer, scheduler and tokenizer
    # pass in model_name
    # pass in checkpoint_name
    settings = {
        "trainings_epochs": 10,
        "batch_size": 30,
        "epochs_trained": 0,
        "steps_trained": 0,
        "global_step": 1
    }


    # Specify which directory model checkpoints should be saved to.
    # Make the checkpoint directory if it does not exist already.
    checkpoint_dir = "./checkpoints"
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

    path_to_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    if checkpoint_name and os.path.exists(path_to_checkpoint):
        # if the directory with the checkpoint name exists, we can retrieve the correct config from here
        # Load in optimizer, tokenizer and scheduler states

        path_to_optimizer = os.path.join(path_to_checkpoint, "optimizer.pt")
        if os.path.isfile(path_to_optimizer):
            optimizer.load_state_dict(torch.load(path_to_optimizer))

        path_to_tokenizer = os.path.join(path_to_checkpoint, "tokenizer.pt")
        if os.path.isfile(path_to_tokenizer):
            tokenizer.load_state_dict(torch.load(path_to_tokenizer))

        path_to_scheduler = os.path.join(path_to_checkpoint, "scheduler.pt")
        if os.path.isfile(path_to_scheduler):
            scheduler.load_state_dict(torch.load(path_to_scheduler))

        new_settings = torch.load(os.path.join(path_to_checkpoint, "train_settings.bin"))
        settings = update_settings(settings, new_settings)

        if settings["epochs_trained"] > 0:
            print("Continuing training from checkpoint, will skip to saved global_step")
            print("Continuing training from epoch {} and global step {}".format(settings["epochs_trained"], settings["global_step"]))

        if settings["steps_trained"] > 0:
            print("Skip the first {} steps in the first epoch", settings["steps_trained"])

        # Check if continuing training from a checkpoint
        if os.path.exists(model_info["model_path"]):
            pass


    # NOW THAT WE HAVE LOADED IN STATES FROM PREVIOUS CHECKPOINT - WE NEED TO CREATE A NEW CHECKPOINT NAME
    current_day = today.strftime("%d_%m_%y")
    current_time = now.strftime("%H_%M_%S")
    checkpoint_name = current_day + '.' + current_time

    version_2_with_negative = False

    print("\n---------- BEGIN TRAINING ----------")
    print("Dataset Size = {}\nNumber of Epochs = {}\nBatch size = {}\n"
          .format(len(data_loader), settings["training_epochs"], settings["batch_size"]))
    # todo check that replacing len(dataset) with len(data_loader) is a fair / valid exchange

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(settings["epochs_trained"], int(settings["epochs"]), desc="Epoch")

    # Added here for reproducibility
    # TODO SET SEED HERE AGAIN

    tb_writer = SummaryWriter()  # Create a SummaryWriter()
    for _ in train_iterator:
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if settings["steps_trained"] > 0:
                settings["steps_trained"] -= 1
                continue

            # train model one step
            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if model_info["model_type"] in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if model_info["model_type"] in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (ones(batch[0].shape, dtype=int64) * 0).to(device)}
                    )

            outputs = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % 1 == 0:
                nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                settings["global_step"] += 1

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                # Log metrics
                if settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
                    if settings["evaluate_all_checkpoints"]:
                        results = evaluate(model, tokenizer, model_info["model_type"], save_dir, device, settings["evaluate_all_checkpoints"], dataset_info)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, settings["global_step"])
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], settings["global_step"])
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / settings["update_steps"], settings["global_step"])
                    logging_loss = tr_loss

                # Save model checkpoint
                if settings["update_steps"] > 0 and settings["global_step"] % settings["update_steps"] == 0:
                    save_model(model, tokenizer, optimizer, scheduler, settings, settings["global_step"], tr_loss / settings["global_step"], save_dir)

    tb_writer.close()

    # ------------- SAVE FINE-TUNED MODEL -------------
    save_model(model, tokenizer, optimizer, scheduler, settings, settings["global_step"], tr_loss / settings["global_step"], save_dir)



if __name__ == "__main__":

    # define config here
    config = {
        'device': "cuda:0" if torch.cuda.is_available() else "cpu:0",
        'seed': 0,
        'adam_bias_correction': False,
        'schedule': 'original_linear',
        'electra_mask_style': True,
        'size': 'small',
        'num_workers': 3 if torch.cuda.is_available() else 0,           # this might be wrong - it initially was just 3
    }

    # Check and Default
    name_of_run = 'Electra_Seed_{}'.format(config["seed"])

    # merge general config with model specific config
    # Setting of different sizes
    model_specific_config = get_model_config(config['size'])
    config = {**config, **model_specific_config}

    discriminator_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-discriminator')
    generator_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-generator')

    # note that public electra-small model is actually small++ and don't scale down generator size
    generator_config.hidden_size = int(discriminator_config.hidden_size/config["generator_size_divisor"])
    generator_config.num_attention_heads = discriminator_config.num_attention_heads//config["generator_size_divisor"]
    generator_config.intermediate_size = discriminator_config.intermediate_size//config["generator_size_divisor"]
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
    dataset = datasets.load_dataset('csv', cache_dir='../datasets', data_files='./datasets/fibro_abstracts.csv')['train']

    print('Create or load cached ELECTRA-compatible data.')
    # apply_cleaning is true by default e.g. ELECTRAProcessor(dataset, apply_cleaning=False) if no cleaning
    e_dataset = ELECTRAProcessor(dataset).map(cache_file_name=f'electra_customdataset_{config["max_length"]}.arrow', num_proc=1)

    hf_dsets = HF_Datasets({'train': e_dataset}, cols={'input_ids': TensorText, 'sentA_length': noop},
                           hf_toker=electra_tokenizer, n_inp=2)

    # data loader
    dls = hf_dsets.dataloaders(bs=config["bs"], num_workers=config["num_workers"], pin_memory=False,
                               shuffle_train=True,
                               srtkey_fc=False,
                               cache_dir='../datasets/electra_dataloader', cache_name='dl_{split}.json')


    # # 2. Masked language model objective
    # 2.1 MLM objective callback
    mlm_cb = MaskedLMCallback(mask_tok_id=electra_tokenizer.mask_token_id,
                              special_tok_ids=electra_tokenizer.all_special_ids,
                              vocab_size=electra_tokenizer.vocab_size,
                              mlm_probability=config["mask_prob"],
                              replace_prob=0.0 if config["electra_mask_style"] else 0.1,
                              orginal_prob=0.15 if config["electra_mask_style"] else 0.1)

    # mlm_cb.show_batch(dls[0], idx_show_ignored=electra_tokenizer.convert_tokens_to_ids(['#'])[0])

    # # 5. Train
    # Seed & PyTorch benchmark
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


    def set_seed(seed_value):
        dls[0].rng = random.Random(seed_value)  # for fastai dataloader
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


    # Optimizer
    if config["adam_bias_correction"]:
        opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
    else:
        opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)


    # Learner
    dls.to(torch.device(config["device"]))

    # Learner is the basic fast ai class for handling the training loop
    # dls: data loaders
    # model: the model to train
    # loss_func: the loss function to use
    # opt_func: used to create an optimiser when Learner.fit is called
    # lr: is the default learning rate
    # :
    learn = Learner(dls, electra_model,
                    loss_func=ELECTRALoss(),
                    opt_func=opt_func,
                    path='./checkpoints',
                    model_dir='pretrain',
                    cbs=[mlm_cb, RunSteps(config["steps"], [0.0625, 0.125, 0.25, 0.5, 1.0], name_of_run+"_{percent}")],
                    )

    # Mixed precison and Gradient clip
    learn.to_native_fp16(init_scale=2.**11)

    # add callback
    learn.add_cb(GradientClipping(1.))

    # Print time and run name
    print(f"{name_of_run} , starts at {datetime.now()}")

    # Learning rate schedule
    lr_schedule = ParamScheduler({'lr': partial(linear_warmup_and_decay,
                                                lr_max=config["lr"],
                                                warmup_steps=10000,
                                                total_steps=config["steps"],)})


    # Run
    learn.fit(9999, cbs=[lr_schedule])