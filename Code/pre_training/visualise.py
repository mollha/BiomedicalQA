import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path
import pickle

base_path = Path(__file__).parent
checkpoint_dir = (base_path / 'checkpoints/pretrain').resolve()
graphs_path = (base_path / 'pre_training_graphs').resolve()
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


def load_stats_from_checkpoint(path_to_checkpoint, checkpoint_name):
    path_to_loss_fc = os.path.join(path_to_checkpoint, "loss_function.pkl")
    if os.path.isfile(path_to_loss_fc):
        with open(path_to_loss_fc, 'rb') as input_file:
            loss_function = pickle.load(input_file)

    loss_function.get_statistics()

    combined_losses, generator_losses, discriminator_losses, discriminator_accuracy, \
    discriminator_precision, discriminator_recall = loss_function.get_statistics()
    num_epochs = range(1, len(generator_losses) + 1)

    print("Combined Losses:", combined_losses)
    print("Generator Losses:", generator_losses)
    print("Discriminator Losses:", discriminator_losses)
    print("Discriminator Accuracy:", discriminator_accuracy)
    print("Discriminator Precision:", discriminator_precision)
    print("Discriminator Recall:", discriminator_recall)


    print("Creating graph of Generator and Discriminator Loss")
    draw_graph(graph_title="Generator and Discriminator Loss during pre-training",
               data=generator_losses,
               data_label="Generator loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss",
               more_data=discriminator_losses,
               more_data_label="Discriminator loss")

    print("Creating graph of Combined Loss")
    draw_graph(graph_title="Loss during pre-training",
               data=combined_losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Creating graph of Discriminator Accuracy")
    draw_graph(graph_title="Discriminator Accuracy during pre-training",
               data=discriminator_accuracy,
               data_label="Discriminator Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Creating graph of Discriminator Precision")
    draw_graph(graph_title="Discriminator Precision during pre-training",
               data=discriminator_precision,
               data_label="Discriminator Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Creating graph of Discriminator Recall")
    draw_graph(graph_title="Discriminator Recall during pre-training",
               data=discriminator_recall,
               data_label="Discriminator Recall",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Graph creation complete.\n")


if __name__ == "__main__":
    chckpt_name = "small_9_6565"    # e.g. small_10_50

    if len(chckpt_name) == 0:
        raise ValueError("Checkpoint name must be the name of a valid checkpoint e.g. small_10_50")

    checkpoint_path = (checkpoint_dir / chckpt_name).resolve()
    load_stats_from_checkpoint(checkpoint_path, chckpt_name)
