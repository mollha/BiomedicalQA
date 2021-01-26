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