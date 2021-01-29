import random
import torch
import numpy as np

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


def update_settings(settings: dict, update: dict, exceptions=[]) -> dict:
    """
    Override config in settings dict with config in update dict. This allows
    model specific config to be merged with general training settings to create
    a single dictionary containing configuration.
    :param settings: dictionary containing general model settings
    :param update: dictionary containing update settings.
    :param exceptions: list of keys to avoid updating. e.g. we want to keep our original config here.
    :return: merged config dictionary
    """
    for key, value in update.items():
        if key in exceptions:
            continue
        settings[key] = value

    return settings


def get_recent_checkpoint_name(directory, subfolders: list):
    """
    Find the name of the most advanced model checkpoint saved in the checkpoints directory.
    This is the model checkpoint that has been trained the most, so it is the best candidate to
    start from if no specific checkpoint name was provided to the pre-training loop.

    Pre-trained checkpoints have the form {size}_{p_epoch}_{p_step}
    Fine-tuned checkpoints have the form {size}_{p_epoch}_{p_step}_{t_epoch}_{t_step}

    :param directory: directory containing model checkpoints.
    :param subfolders: list of checkpoint directories
    :return:
    """
    directory = str(directory)

    def parse_name(subdir: str):
        config_str = str(subdir)[str(subdir).find(directory) + len(directory):]
        elements = config_str.split("_")

        if len(elements) == 3:  # treat this as a pre-trained checkpoint
            p_epoch, p_step = int(elements[1]), int(elements[2])
            return p_epoch, p_step
        elif len(elements) == 5:  # treat this as a fine-tuned checkpoint
            p_epoch, p_step, t_epoch, t_step = int(elements[1]), int(elements[2]), int(elements[3]), int(elements[4])
            return p_epoch, p_step, t_epoch, t_step
        else:
            raise Exception("Checkpoint name is {} when it should be of the form [size]_[p_epoch]_[p_step] for "
                            "pre_trained or [size]_[p_epoch]_[p_step]_[t_epoch]_[t_step]".format(config_str))

    max_file, max_epoch, max_step_in_epoch = None, None, None
    for subdirectory in subfolders:
        parsed_name = parse_name(subdirectory)

        if len(parsed_name) == 2:
            p_epoch, p_step, t_epoch, t_step = parsed_name
        elif len(parsed_name) == 4:
            p_epoch, p_step = parsed_name
        else:
            raise Exception("parse_name should only be returning 2 or 4 elements - returned {}.".format(parsed_name))

        if max_epoch is None or p_epoch > max_epoch:
            max_epoch = p_epoch
            max_step_in_epoch = p_step
            max_file = subdirectory
        elif p_epoch == max_epoch:
            if p_step > max_step_in_epoch:
                max_step_in_epoch = p_step
                max_file = subdirectory
    return max_file