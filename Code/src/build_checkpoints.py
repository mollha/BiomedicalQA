from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    ElectraForQuestionAnswering,
)
from models import *
from loss_functions import *
from pathlib import Path
from utils import get_recent_checkpoint_name, update_settings


def build_pretrained_from_checkpoint(model_size, device, checkpoint_directory, checkpoint_name, config={}):
    """ Note: If we don't pass config and the checkpoint name is valid - config will be populated with
    model-specific config only. This is useful when build_from_checkpoint is called from fine-tuning,
    but we don't need the pre-training configuration - only model configuration."""
    # create the checkpoint directory if it doesn't exist
    Path(checkpoint_directory).mkdir(exist_ok=True, parents=True)

    # -- Override general config with model specific config, for models of different sizes
    model_settings = get_model_config(model_size)
    generator, discriminator, electra_tokenizer = build_electra_model(model_size)
    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)

    # Prepare optimizer and schedule (linear warm up and decay) # eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01
    optimizer = AdamW(electra_model.parameters(), eps=1e-6, weight_decay=0.01, lr=model_settings["lr"],
                      correct_bias=model_settings["adam_bias_correction"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000,
                                                num_training_steps=model_settings["max_steps"])

    #   -------- DETERMINE WHETHER TRAINING FROM A CHECKPOINT OR FROM SCRATCH --------
    loss_function = ELECTRALoss()
    valid_checkpoint, path_to_checkpoint = False, None

    if checkpoint_name.lower() == "recent":
        subfolders = [x for x in Path(checkpoint_directory).iterdir() if x.is_dir() and model_size in str(x)[str(x).rfind('/') + 1:]]

        if len(subfolders) > 0:
            path_to_checkpoint = get_recent_checkpoint_name(checkpoint_directory, subfolders)
            print("\nTraining from the most advanced checkpoint - {}\n".format(path_to_checkpoint))
            valid_checkpoint = True
    elif checkpoint_name:
        path_to_checkpoint = os.path.join(checkpoint_directory, checkpoint_name)
        if os.path.exists(path_to_checkpoint):
            print("\nCheckpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
            valid_checkpoint = True  # if dir with checkpoint name exists, we can retrieve correct config from here
        else:
            print("WARNING: Checkpoint {} does not exist at path {}.\n".format(checkpoint_name, path_to_checkpoint))

    if valid_checkpoint:
        electra_model, optimizer, scheduler, populated_loss_function, new_config = load_checkpoint(path_to_checkpoint, electra_model, optimizer, scheduler, device)
        config = update_settings(config, new_config, exceptions=["update_steps", "device"])
        loss_function.__dict__.update(populated_loss_function.__dict__)
        print('Successfully copied loss function statistics:', loss_function.mid_epoch_stats)
    else:
        print("\nTraining from scratch - no checkpoint provided.\n")
    return electra_model, optimizer, scheduler, electra_tokenizer, loss_function, config


# This function is only applicable to fine-tuning as we want to use a different optimizer and scheduler.
def get_optimizer_and_scheduler(model, model_config, model_settings, num_warmup_steps):
    """ Get an optimizer and scheduler, adjusted for the particular parameters in our model.
    model_settings are settings relating to the model, config are extra parameters """
    # ------ LOAD MODEL FROM PRE-TRAINED CHECKPOINT OR FROM FINE-TUNED CHECKPOINT ------
    layerwise_learning_rates = get_layer_lrs(model.named_parameters(), model_settings["lr"],
                                             model_settings["layerwise_lr_decay"],
                                             model_config.num_hidden_layers)
    no_decay = ["bias", "LayerNorm", "layer_norm"]  # do not have decay on these layers.
    # Prepare optimizer and scheduler (linear warm up and decay)
    layerwise_params = []
    for n, p in model.named_parameters():
        wd = model_settings["decay"] if not any(nd in n for nd in no_decay) else 0
        lr = layerwise_learning_rates[n]
        layerwise_params.append({"params": [p], "weight_decay": wd, "lr": lr})

    # Create the optimizer and scheduler
    optimizer = AdamW(layerwise_params, eps=model_settings["epsilon"], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(num_warmup_steps) * model_settings["warmup_fraction"], num_training_steps=num_warmup_steps)
    return optimizer, scheduler


def build_finetuned_from_checkpoint(model_size, device, pretrained_checkpoint_dir, finetuned_checkpoint_dir, checkpoint_name, question_type, config={}):
    # -- Create the checkpoint directories if they don't exist --
    pretrained_checkpoint_name, finetuned_checkpoint_name = checkpoint_name
    Path(pretrained_checkpoint_dir).mkdir(exist_ok=True, parents=True)
    Path(finetuned_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # -- Override general config with model specific config --
    model_settings = {**get_model_config(model_size, pretrain=False),
                      **get_data_specific_config(config['size'], config["dataset"])}

    generator, discriminator, electra_tokenizer, \
    discriminator_config = build_electra_model(model_size, get_config=True)  # get basic model building blocks

    #   -------- DETERMINE WHETHER TRAINING FROM A FINE-TUNED CHECKPOINT OR FROM PRETRAINED CHECKPOINT --------
    valid_finetune_checkpoint, path_to_checkpoint, building_from_pretrained = False, None, True

    if finetuned_checkpoint_name != "":  # if we have a valid name for fine-tuned checkpoint
        if finetuned_checkpoint_name.lower() == "recent":  # if fine-tuned checkpoint_name is recent
            # get the finetuned checkpoint with highest finetuned epochs and steps (doesn't care about most pre-trained)
            all_subfolders = []  # get the subfolders containing checkpoints
            for x in Path(finetuned_checkpoint_dir).iterdir():
                if x.is_dir() and model_size in str(x)[str(x).rfind('/') + 1:]:
                    for qt in question_type:
                        if qt in str(x)[str(x).rfind('/') + 1:]:
                            all_subfolders.append(x)

            if len(all_subfolders) > 0:  # pick the most recent fine-tuned checkpoint
                path_to_checkpoint = get_recent_checkpoint_name(finetuned_checkpoint_dir, all_subfolders)
                print("\nTraining from the most advanced checkpoint - {}\n".format(path_to_checkpoint))
                valid_finetune_checkpoint = True
                building_from_pretrained = False

        else:  # fine-tuned checkpoint name should contain full checkpoint info (pretrained and fine-tuned)
            path_to_checkpoint = os.path.join(finetuned_checkpoint_dir, finetuned_checkpoint_name)
            if os.path.exists(path_to_checkpoint):
                print("\nCheckpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
                # if the directory with the checkpoint name exists, we can retrieve the correct config from here
                valid_finetune_checkpoint, building_from_pretrained = True, False
            else:
                print("\nWARNING: Checkpoint {} does not exist at path {}.\n".format(checkpoint_name, path_to_checkpoint))

        # ---- If we're training from a valid finetuned checkpoint ----
        if valid_finetune_checkpoint:
            if "factoid" in question_type or "list" in question_type:  # check if the question_type is list or factoid
                qa_model = ElectraForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None, state_dict=discriminator.state_dict(), config=discriminator_config)  # create extractive QA model

            elif "yesno" in question_type:  # check if the question_type is yes/no
                qa_model = CostSensitiveSequenceClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=discriminator.state_dict(), config=discriminator_config)  # create binary model
            else:
                raise Exception("Question type list must be contain factoid, list or yesno.")

            # get the template within which to load optimizer and scheduler
            optimizer, scheduler = get_optimizer_and_scheduler(qa_model, discriminator_config, model_settings, config["num_warmup_steps"])
            electra_for_qa, new_optimizer, new_scheduler, new_config = load_checkpoint(path_to_checkpoint, qa_model, optimizer, scheduler, device, pre_training=False)

            # config["dataset"] contains the new dataset we're finetuning on
            # if new_config["dataset"] (i.e. the old dataset) does not match, then we need a new scheduler and optimizer
            # instead of the one we just loaded from the checkpoint.
            if config["dataset"] == new_config["dataset"]:  # we're continuing training on the same dataset
                optimizer, scheduler = new_optimizer, new_scheduler  # overwrite optimizer and scheduler with loaded

            config = update_settings(config, new_config, exceptions=["update_steps", "device", "evaluate_during_training"])
            building_from_pretrained = False
        else:
            print("\nFine-tuning from the most advanced pre-trained checkpoint - invalid checkpoint '{}' provided.\n"
                  .format(finetuned_checkpoint_name))

    # ---- No ft checkpoint provided, so fine-tune from the most advanced pt checkpoint ----
    if building_from_pretrained:
        pretrained_model, _, _, electra_tokenizer, _, p_model_config =\
            build_pretrained_from_checkpoint(model_size, device, pretrained_checkpoint_dir, pretrained_checkpoint_name)
        config["pretrained_settings"] = {"epochs": p_model_config["current_epoch"], "steps": p_model_config["steps_trained"]}
        config = update_settings(config, model_settings)

        discriminator = pretrained_model.discriminator

        if "factoid" in question_type or "list" in question_type:  # check if the question_type is list or factoid
            electra_for_qa = ElectraForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=None, state_dict=discriminator.state_dict(), config=discriminator_config)
        elif "yesno" in question_type:  # check if the question_type is yes/no
            electra_for_qa = CostSensitiveSequenceClassification.from_pretrained(pretrained_model_name_or_path=None, state_dict=discriminator.state_dict(), config=discriminator_config)
        else:
            raise Exception("Question type list must be contain factoid, list or yesno.")

        optimizer, scheduler = get_optimizer_and_scheduler(electra_for_qa, discriminator_config, model_settings, config["num_warmup_steps"])
    return electra_for_qa, optimizer, scheduler, electra_tokenizer, config
