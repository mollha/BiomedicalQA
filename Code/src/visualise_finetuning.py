import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle

import torch

base_path = Path(__file__).parent
checkpoint_dir = (base_path / '../checkpoints/finetune').resolve()
graphs_path = (base_path / 'visualisations/fine_tuning_graphs').resolve()
Path(graphs_path).mkdir(exist_ok=True, parents=True)


def draw_graph(graph_title, data, data_label, epochs, checkpoint_name, y_label=None, more_data=None, more_data_label=None):
    plt.plot(epochs, data, label=data_label)
    if more_data is not None:
        plt.plot(epochs, more_data, label=more_data_label)

    plt.title(graph_title)
    plt.xlabel("Epochs")
    y_label = y_label if y_label is not None else data_label
    graph_save_name = checkpoint_name + "_" + y_label.lower().replace(" ", "_") + "_epochs.png"

    plt.ylabel(y_label)
    plt.legend()
    plt.savefig((graphs_path / graph_save_name).resolve())
    plt.show()


def pretty_print_settings(settings):
    print("--- SETTINGS ---")
    for key, value in settings.items():
        print("{}: {}".format(key, value))
    print()


def load_stats_from_checkpoint(path_to_checkpoint, checkpoint_name):

    path_to_settings = os.path.join(path_to_checkpoint, "train_settings.bin")
    if os.path.isfile(path_to_settings):
        settings = torch.load(path_to_settings)
        pretty_print_settings(settings)
    else:
        raise Exception("No training statistics to display.")

    num_epochs = range(1, settings["current_epoch"] + 1)  # might need to add 1
    losses = settings["losses"]

    if len(losses) == 0:
        raise Exception("Losses list does not contain any data.")
    print(losses)

    print("Creating graph of model loss")
    draw_graph(graph_title="Loss during fine-tuning",
               data=losses,
               data_label="Generator loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss")

    print("Graph creation complete.\n")




if __name__ == "__main__":
    chckpt_name = "small_yesno_13_68164_9_123"    # e.g. small_10_50

    if len(chckpt_name) == 0:
        raise ValueError("Checkpoint name must be the name of a valid checkpoint e.g. small_10_50")

    checkpoint_path = (checkpoint_dir / chckpt_name).resolve()
    load_stats_from_checkpoint(checkpoint_path, chckpt_name)
