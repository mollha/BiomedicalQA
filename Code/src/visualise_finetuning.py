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


def visualise_yes_no(checkpoint_name, metrics):
    # Which metrics should we plot for yes no questions?
    # Plot all of these metrics on the same graph.

    # metrics = {
    #         "accuracy": accuracy,
    #         "precision": precision_y,
    #         "recall": recall_y,
    #         "f1_y": f1_y,
    #         "f1_n": f1_n,
    #         "f1_ma": f1_ma,
    #     }

    num_epochs = range(1, len(metrics) + 1)  # might need to add 1
    fig_size = (5, 5)

    accuracy = [metric_dict["accuracy"] for metric_dict in metrics]
    precision = [metric_dict["precision_y"] for metric_dict in metrics]
    recall = [metric_dict["recall_y"] for metric_dict in metrics]
    # f1_y = [metric_dict["f1_y"] for metric_dict in metrics]
    # f1_n = [metric_dict["f1_n"] for metric_dict in metrics]
    f1_ma = [metric_dict["f1_ma"] for metric_dict in metrics]

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Accuracy",
               data=accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Precision and Recall",
               data=precision,
               data_label="Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Metric Value",
               more_data=recall,
               more_data_label="Recall")

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="F1 Score",
               data=f1_ma,
               data_label="Macro Avg F1 Score",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)


def load_stats_from_checkpoint(path_to_checkpoint, checkpoint_name):

    path_to_settings = os.path.join(path_to_checkpoint, "train_settings.bin")
    if os.path.isfile(path_to_settings):
        settings = torch.load(path_to_settings)
        print("--- SETTINGS ---")
        for key, value in settings.items():
            print("{}: {}".format(key, value))
        print()
    else:
        raise Exception("No training statistics to display.")

    # Get saved metrics from settings - we should have a metric dict for each epoch
    metrics = settings["finetune_statistics"]["metrics"]
    print(metrics)

    num_epochs = range(0, settings["current_epoch"] + 1)  # might need to add 1
    losses = settings["finetune_statistics"]["losses"]

    if len(losses) == 0:
        raise Exception("Losses list does not contain any data.")
    print(losses)

    print("Creating graph of model loss")
    draw_graph(graph_title="Loss during fine-tuning",
               data=losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss")


    visualise_yes_no(checkpoint_name, metrics)

    print("Graph creation complete.\n")




if __name__ == "__main__":
    chckpt_name = "small_yesno_14_79670_9_103"    # e.g. small_10_50

    if len(chckpt_name) == 0:
        raise ValueError("Checkpoint name must be the name of a valid checkpoint e.g. small_10_50")

    checkpoint_path = (checkpoint_dir / chckpt_name).resolve()
    load_stats_from_checkpoint(checkpoint_path, chckpt_name)
