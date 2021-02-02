import argparse
from read_data import dataset_to_fc
from tqdm import trange
from tqdm import tqdm
from models import *
from utils import *
from data_processing import convert_samples_to_features, SQuADDataset, collate_wrapper
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification
)
from pre_training import build_pretrained_from_checkpoint
from torch.utils.data import DataLoader, RandomSampler

# Ensure that lowercase model is used for model_type
# ------------- DEFINE TRAINING AND EVALUATION SETTINGS -------------
# hyperparameters are mostly the same for large models as base models - except for a few

config = {
    'seed': 0,
    'losses': [],
    'avg_loss': [0, 0],
    'num_workers': 3 if torch.cuda.is_available() else 0,
    # "max_epochs": 2,  # can override the val in config
    "current_epoch": 0,  # track the current epoch in config for saving checkpoints
    "steps_trained": 0,  # track the steps trained in config for saving checkpoints
    "global_step": -1,  # total steps over all epochs
    "update_steps": 5,
    "pretrained_settings": {
        "epochs": 0,
        "steps": 0,
    },
    "evaluate_during_training": True
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


def build_finetuned_from_checkpoint(model_size, device, pretrained_checkpoint_dir, finetuned_checkpoint_dir,
                                    checkpoint_name, question_type, config={}):

    pretrained_checkpoint_name, finetuned_checkpoint_name = checkpoint_name

    # create the checkpoint directory if it doesn't exist
    Path(pretrained_checkpoint_dir).mkdir(exist_ok=True, parents=True)
    Path(finetuned_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    model_settings = get_model_config(model_size, pretrain=False)  # override general config with model specific config
    generator, discriminator, electra_tokenizer,\
    discriminator_config = build_electra_model(model_size, get_config=True)  # get basic model building blocks

    # ------ LOAD MODEL FROM PRE-TRAINED CHECKPOINT OR FROM FINE-TUNED CHECKPOINT ------
    # get pre-trained model from which to begin fine-tuning
    layerwise_learning_rates = get_layer_lrs(discriminator.named_parameters(), model_settings["lr"],
                                             model_settings["layerwise_lr_decay"],
                                             discriminator_config.num_hidden_layers)

    no_decay = ["bias", "LayerNorm", "layer_norm"]  # Prepare optimizer and schedule (linear warm up and decay)

    layerwise_params = []  # todo check if lr should be dependent on non-zero wd
    for n, p in discriminator.named_parameters():
        wd = model_settings["decay"] if not any(nd in n for nd in no_decay) else 0
        lr = layerwise_learning_rates[n]
        layerwise_params.append({"params": [p], "weight_decay": wd, "lr": lr})

    # Create the optimizer and scheduler
    optimizer = AdamW(layerwise_params, eps=model_settings["epsilon"], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=(config["num_warmup_steps"]) * model_settings["warmup_fraction"],
                num_training_steps=-1)  # todo check whether num_training_steps should be -1

    #   -------- DETERMINE WHETHER TRAINING FROM A FINE-TUNED CHECKPOINT OR FROM PRETRAINED CHECKPOINT --------
    valid_finetune_checkpoint, path_to_checkpoint, building_from_pretrained = False, None, True

    if finetuned_checkpoint_name != "":  # if we have a valid name for fine-tuned checkpoint
        if finetuned_checkpoint_name.lower() == "recent":  # if fine-tuned checkpoint_name is recent
            # get the fine-tuned checkpoint with highest fine-tuned epochs and steps
            # (does not care about the most pretrained)

            # pick the most recent fine-tuned checkpoint
            subfolders = [x for x in Path(finetuned_checkpoint_dir).iterdir() \
                          if x.is_dir() and model_size in str(x)[str(x).rfind('/') + 1:]
                          and question_type in str(x)[str(x).rfind('/') + 1:]]

            if len(subfolders) > 0:
                path_to_checkpoint = get_recent_checkpoint_name(finetuned_checkpoint_dir, subfolders)
                print("\nTraining from the most advanced checkpoint - {}\n".format(path_to_checkpoint))
                valid_finetune_checkpoint = True
                building_from_pretrained = False

        else:  # fine-tuned checkpoint name should contain full checkpoint info (pretrained and fine-tuned)
            path_to_checkpoint = os.path.join(finetuned_checkpoint_dir, finetuned_checkpoint_name)
            if os.path.exists(path_to_checkpoint):
                print(
                    "Checkpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
                # if the directory with the checkpoint name exists, we can retrieve the correct config from here
                valid_finetune_checkpoint = True
                building_from_pretrained = False
            else:
                print(
                    "WARNING: Checkpoint {} does not exist at path {}.\n".format(checkpoint_name, path_to_checkpoint))

        if valid_finetune_checkpoint:
            # check if the question_type is yesno or factoid
            if question_type == "factoid" or question_type == "list":
                qa_model = ElectraForQuestionAnswering(config=discriminator_config)
            elif question_type == "yesno":
                qa_model = ElectraForSequenceClassification(config=discriminator_config, return_dict=True)
            else:
                raise Exception("Question type must be factoid, list or yesno.")

            electra_for_qa, optimizer, scheduler, new_config = load_checkpoint(path_to_checkpoint, qa_model,
                                                                              optimizer, scheduler, device,
                                                                              pre_training=False)

            config = update_settings(config, new_config, exceptions=["update_steps", "device", "evaluate_during_training"])
            building_from_pretrained = False

        else:
            print("\nFine-tuning from the most advanced pre-trained checkpoint - invalid checkpoint '{}' provided.\n"
                  .format(finetuned_checkpoint_name))

    if building_from_pretrained:
        # no fine-tuned checkpoint provided so we just fine-tune the most advanced pre-trained checkpoint (using existing logic)
        pretrained_model, _, _, electra_tokenizer, _, p_model_config = build_pretrained_from_checkpoint(model_size, device,
                                                                                                      pretrained_checkpoint_dir,
                                                                                                      pretrained_checkpoint_name)

        config["pretrained_settings"] = {"epochs": p_model_config["current_epoch"],
                                         "steps": p_model_config["steps_trained"]}
        discriminator = pretrained_model.discriminator

        if question_type == "factoid" or question_type == "list":
            electra_for_qa = ElectraForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None,
                                                                     state_dict=discriminator.state_dict(),
                                                                     config=discriminator_config)
        elif question_type == "yesno":
            electra_for_qa = ElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=None,
                                                                     state_dict=discriminator.state_dict(),
                                                                     config=discriminator_config)
        else:
            raise Exception("Question type must be factoid, list or yesno.")

    return electra_for_qa, optimizer, scheduler, electra_tokenizer, config


def fine_tune(train_dataloader, qa_model, scheduler, optimizer, settings, checkpoint_dir):
    qa_model.to(settings["device"])

    # ------------------ PREPARE TO START THE TRAINING LOOP ------------------
    sys.stderr.write("\n---------- BEGIN FINE-TUNING ----------")
    sys.stderr.write("\nDevice = {}\nModel Size = {}\nTotal Epochs = {}\nStart training from Epoch = {}\nStart training from Step = {}\nBatch size = {}\nCheckpoint Steps = {}\nMax Sample Length = {}\n\n"
                     .format(settings["device"].upper(), settings["size"], settings["max_epochs"], settings["current_epoch"],
                             settings["steps_trained"], settings["batch_size"], settings["update_steps"],
                             settings["max_length"]))

    qa_model.zero_grad()
    # Added here for reproducibility
    set_seed(settings["seed"])

    # Resume training from the epoch we left off at earlier.
    train_iterator = trange(settings["current_epoch"], int(settings["max_epochs"]), desc="Epoch")

    # resume training
    steps_trained = settings["steps_trained"]

    for epoch_number in train_iterator:
        step_iterator = tqdm(train_dataloader, desc="Step")

        # update the current epoch
        settings["current_epoch"] = epoch_number  # update the number of epochs

        for training_step, batch in enumerate(step_iterator):
            question_ids = batch.question_ids
            is_impossible = batch.is_impossible

            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue  # skip this step

            qa_model.train()  # train model one step

            inputs = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "token_type_ids": batch.token_type_ids,
                "start_positions": batch.answer_start,
                "end_positions": batch.answer_end,
            }

            if settings["question_type"] == "factoid" or settings["question_type"] == "list":
                inputs = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                    "token_type_ids": batch.token_type_ids,
                    "start_positions": batch.answer_start,
                    "end_positions": batch.answer_end,
                }
            elif settings["question_type"] == "yesno":
                inputs = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                    "token_type_ids": batch.token_type_ids,
                }
            else:
                raise Exception("Question type must be factoid, list or yesno.")

            outputs = qa_model(**inputs)
            print(outputs)

            # model outputs are always tuples in transformers
            loss = outputs.loss
            loss.backward()

            settings["avg_loss"][0] += float(loss.item())
            settings["avg_loss"][1] += 1

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
    settings["losses"].append(settings["avg_loss"][0] / settings["avg_loss"][1])   # bank stats
    settings["avg_loss"] = [0, 0]  # reset stats
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
        "--p-checkpoint",
        default="recent",
        type=str,
        help="The name of the pre-training checkpoint to use e.g. small_15_10230.",
    )
    parser.add_argument(
        "--f-checkpoint",
        default="recent",  # if not provided, we assume fine-tuning from pre-trained
        type=str,
        help="The name of the fine-tuning checkpoint to use e.g. small_factoid_15_10230_2_30487",
    )
    parser.add_argument(
        "--question-type",
        default="yesno",
        choices=['factoid', 'yesno', 'list'],
        type=str,
        help="Type of fine-tuned model should be created - factoid, list or yesno?",
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
    config["question_type"] = args.question_type

    sys.stderr.write("Selected finetuning checkpoint {}, pretraining checkpoint {} and model size {}"
                     .format(args.f_checkpoint, args.p_checkpoint, args.size))

    if args.f_checkpoint != "" and args.f_checkpoint != "recent":
        if args.size not in args.f_checkpoint:
            raise Exception("If using a fine-tuned checkpoint, the model size of the checkpoint must match provided model size."
                            "e.g. --f-checkpoint small_factoid_15_10230_12_20420 --size small")
        if args.question_type not in args.f_checkpoint:
            raise Exception(
                "If using a fine-tuned checkpoint, the question type of the checkpoint must match question type."
                "e.g. --f-checkpoint small_factoid_15_10230_12_20420 --question-type factoid")
    elif args.p_checkpoint != "recent" and args.size not in args.p_checkpoint:
        raise Exception("If not using the most recent checkpoint, the model size of the checkpoint must match model size."
                        "e.g. --p-checkpoint small_15_10230 --size small")

    # -- Set torch backend and set seed
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])  # set seed for reproducibility

    # -- Override general config with model specific config, for models of different sizes
    model_specific_config = get_model_config(config['size'], pretrain=False)
    config = {**model_specific_config, **config}

    # -- Find path to checkpoint directory - create the directory if it doesn't exist
    base_path = Path(__file__).parent
    f_checkpoint_name = args.f_checkpoint
    p_checkpoint_name = args.p_checkpoint
    checkpoint_name = (p_checkpoint_name, f_checkpoint_name)

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
    train_dataset = SQuADDataset(features)  # todo change this

    # Random Sampler used during training.
    data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config["batch_size"],
                             collate_fn=collate_wrapper)

    config["num_warmup_steps"] = len(data_loader) // config["max_epochs"]

    electra_for_qa, optimizer, scheduler, electra_tokenizer,\
    config = build_finetuned_from_checkpoint(config["size"], config["device"], pretrain_checkpoint_dir,
                                             finetune_checkpoint_dir, checkpoint_name, config["question_type"], config)

    # ------ START THE FINE-TUNING LOOP ------
    fine_tune(data_loader, electra_for_qa, scheduler, optimizer, config, finetune_checkpoint_dir)
