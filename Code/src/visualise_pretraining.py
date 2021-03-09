import matplotlib.pyplot as plt
from matplotlib import rc
import os
import torch
from pathlib import Path
import pickle

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


base_path = Path(__file__).parent
checkpoint_dir = (base_path / '../checkpoints/pretrain').resolve()
graphs_path = (base_path / 'visualisations/pre_training_graphs').resolve()
Path(graphs_path).mkdir(exist_ok=True, parents=True)



def draw_graph(graph_title, data, data_label, epochs, checkpoint_name, y_label=None, more_data=None,
               more_data_label=None, save_figure=True, color="b"):
    plt.plot(epochs, data, label=data_label, color=color)
    if more_data is not None:
        plt.plot(epochs, more_data, label=more_data_label)

    plt.title(r"\textbf{" + graph_title + "}")
    plt.xlabel("Epochs")
    y_label = y_label if y_label is not None else data_label
    graph_save_name = checkpoint_name + "_" + y_label.lower().replace(" ", "_") + "_epochs.png"

    plt.ylabel(y_label)
    plt.legend()

    if save_figure:
        plt.savefig((graphs_path / graph_save_name).resolve())  # this is saving them weird due to subplots
        plt.show()
    return plt


def load_stats_from_checkpoint(path_to_checkpoint):
    path_to_loss_fc = os.path.join(path_to_checkpoint, "loss_function.pkl")
    print(path_to_loss_fc)
    if os.path.isfile(path_to_loss_fc):
        with open(path_to_loss_fc, 'rb') as input_file:
            loss_function = pickle.load(input_file)

    loss_function.get_statistics()

    combined_losses, generator_losses, discriminator_losses, discriminator_accuracy, \
    discriminator_precision, discriminator_recall = loss_function.get_statistics()

    # accuracy_constant = 62.5299709349       # fix accuracy (roughly)
    # discriminator_accuracy = [a * accuracy_constant for a in discriminator_accuracy]

    print("Combined Losses:", combined_losses)
    print("Generator Losses:", generator_losses)
    print("Discriminator Losses:", discriminator_losses)
    print("Discriminator Accuracy:", discriminator_accuracy)
    print("Discriminator Precision:", discriminator_precision)
    print("Discriminator Recall:", discriminator_recall)

    return combined_losses, generator_losses, discriminator_losses, discriminator_accuracy, discriminator_precision, \
           discriminator_recall


def create_subplots(statistics, checkpoint_name):
    combined_losses, generator_losses, discriminator_losses, discriminator_accuracy, discriminator_precision, \
    discriminator_recall = statistics

    num_epochs = range(1, len(generator_losses) + 1)
    fig = plt.figure(0, figsize=(6, 6))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axis_font_size = 14
    # fig.suptitle("Statistics during Pre-training", fontsize=12, fontweight="bold", y=0.96)

    # Note: we don't include this in the subplots for now. It can be a standalone graph.
    # print("Creating graph of Generator and Discriminator Loss")
    # draw_graph(graph_title="Generator and Discriminator Loss during pre-training",
    #                                      data=generator_losses,
    #                                      data_label="Generator loss",
    #                                      epochs=num_epochs,
    #                                      checkpoint_name=checkpoint_name,
    #                                      y_label="Loss",
    #                                      more_data=discriminator_losses,
    #                                      more_data_label="Discriminator loss")

    plt.subplot(2, 2, 1)
    print("\nCreating graph of Combined Loss")
    draw_graph(graph_title="Combined Loss",
               data=combined_losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="b",
               save_figure=False)
    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)


    print("Creating graph of Discriminator Accuracy")
    plt.subplot(2, 2, 2)
    draw_graph(graph_title="Discriminator Accuracy",
               data=discriminator_accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="g",
               save_figure=False)
    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)



    plt.subplot(2, 2, 3)
    print("Creating graph of Discriminator Precision")
    draw_graph(graph_title="Discriminator Precision",
               data=discriminator_precision,
               data_label="Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="r",
               save_figure=False)
    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)


    plt.subplot(2, 2, 4)
    print("Creating graph of Discriminator Recall")
    draw_graph(graph_title="Discriminator Recall",
               data=discriminator_recall,
               data_label="Recall",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="c",
               save_figure=False)
    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplots_adjust(hspace=0.4, wspace=0.45)
    plt.savefig((graphs_path / "pretraining_subplot.png").resolve())  # this is saving them weird due to subplots
    plt.show()
    print("\nGraph creation complete.\n")


def create_separate_plots(statistics, checkpoint_name):
    combined_losses, generator_losses, discriminator_losses, discriminator_accuracy, discriminator_precision, \
    discriminator_recall = statistics

    fig_size = (5, 5)
    num_epochs = range(1, len(generator_losses) + 1)

    print("Creating graph of Generator and Discriminator Loss")
    plt.figure(1, figsize=fig_size)
    draw_graph(graph_title="Generator and Discriminator Loss during pre-training",
               data=generator_losses,
               data_label="Generator loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss",
               more_data=discriminator_losses,
               more_data_label="Discriminator loss")

    print("\nCreating graph of Combined Loss")
    plt.figure(2, figsize=fig_size)
    draw_graph(graph_title="Loss during Pre-training",
               data=combined_losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Creating graph of Discriminator Accuracy")
    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Discriminator Accuracy during Pre-training",
               data=discriminator_accuracy,
               data_label="Discriminator Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Creating graph of Discriminator Precision")
    plt.figure(4, figsize=fig_size)
    draw_graph(graph_title="Discriminator Precision during Pre-training",
               data=discriminator_precision,
               data_label="Discriminator Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    print("Creating graph of Discriminator Recall")
    plt.figure(5, figsize=fig_size)
    draw_graph(graph_title="Discriminator Recall during pre-training",
               data=discriminator_recall,
               data_label="Discriminator Recall",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)
    print("\nGraph creation complete.\n")


if __name__ == "__main__":
    chckpt_name = "small_17_12583"  # e.g. small_10_50

    if len(chckpt_name) == 0:
        raise ValueError("Checkpoint name must be the name of a valid checkpoint e.g. small_10_50")

    checkpoint_path = (checkpoint_dir / chckpt_name).resolve()
    stats = load_stats_from_checkpoint(checkpoint_path)

    create_subplots(stats, chckpt_name)
    create_separate_plots(stats, chckpt_name)
