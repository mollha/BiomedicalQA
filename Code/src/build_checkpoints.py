from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification
)
from models import *
from loss_functions import *
from pathlib import Path
from utils import get_recent_checkpoint_name, update_settings


def build_pretrained_from_checkpoint(model_size, device, checkpoint_directory, checkpoint_name, config={}):
    """

    Note: If we don't pass config and the checkpoint name is valid - config will be populated with
    model-specific config only. This is useful when build_from_checkpoint is called from fine-tuning,
    but we don't need the pre-training configuration - only model configuration.


    :param model_size:
    :param device:
    :param checkpoint_name:
    :param config:
    :return:
    """
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
    loss_function = ELECTRALoss()
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
                "\nCheckpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
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


def build_finetuned_from_checkpoint(model_size, device, pretrained_checkpoint_dir, finetuned_checkpoint_dir,
                                    checkpoint_name, question_type, config={}):
    pretrained_checkpoint_name, finetuned_checkpoint_name = checkpoint_name

    # create the checkpoint directory if it doesn't exist
    Path(pretrained_checkpoint_dir).mkdir(exist_ok=True, parents=True)
    Path(finetuned_checkpoint_dir).mkdir(exist_ok=True, parents=True)

    model_settings = get_model_config(model_size, pretrain=False)  # override general config with model specific config
    generator, discriminator, electra_tokenizer, \
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
                                                num_warmup_steps=(config["num_warmup_steps"]) * model_settings[
                                                    "warmup_fraction"],
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
                    "\nCheckpoint '{}' exists - Loading config values from memory.\n".format(path_to_checkpoint))
                # if the directory with the checkpoint name exists, we can retrieve the correct config from here
                valid_finetune_checkpoint = True
                building_from_pretrained = False
            else:
                print(
                    "\nWARNING: Checkpoint {} does not exist at path {}.\n".format(checkpoint_name, path_to_checkpoint))

        if valid_finetune_checkpoint:
            # check if the question_type is yesno or factoid
            if question_type == "factoid" or question_type == "list":
                qa_model = ElectraForQuestionAnswering(config=discriminator_config)
            elif question_type == "yesno":
                qa_model = ElectraForSequenceClassification(config=discriminator_config)
            else:
                raise Exception("Question type must be factoid, list or yesno.")

            electra_for_qa, optimizer, scheduler, new_config = load_checkpoint(path_to_checkpoint, qa_model,
                                                                               optimizer, scheduler, device,
                                                                               pre_training=False)

            config = update_settings(config, new_config,
                                     exceptions=["update_steps", "device", "evaluate_during_training"])
            building_from_pretrained = False

        else:
            print("\nFine-tuning from the most advanced pre-trained checkpoint - invalid checkpoint '{}' provided.\n"
                  .format(finetuned_checkpoint_name))

    if building_from_pretrained:
        # no fine-tuned checkpoint provided so we just fine-tune the most advanced pre-trained checkpoint (using existing logic)
        pretrained_model, _, _, electra_tokenizer, _, p_model_config = build_pretrained_from_checkpoint(model_size,
                                                                                                        device,
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