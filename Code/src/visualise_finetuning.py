import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle

import torch
from matplotlib import rc


axis_font_size=13

base_path = Path(__file__).parent
checkpoint_dir = (base_path / '../checkpoints/finetune').resolve()
graphs_path = (base_path / 'visualisations/fine_tuning_graphs').resolve()
Path(graphs_path).mkdir(exist_ok=True, parents=True)

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


def draw_graph(graph_title, data, data_label, epochs, checkpoint_name, y_label=None, more_data=None,
               more_data_label=None, save_figure=True, color="b"):
    plt.plot(epochs, data, label=data_label, color=color)
    if more_data is not None:
        plt.plot(epochs, more_data, label=more_data_label)

    plt.title(r"\textbf{" + graph_title + "}", fontsize=axis_font_size)
    plt.xlabel("Epochs", fontsize=axis_font_size)
    y_label = y_label if y_label is not None else data_label
    graph_save_name = checkpoint_name + "_" + y_label.lower().replace(" ", "_") + "_epochs.png"

    plt.ylabel(y_label, fontsize=axis_font_size)

    if more_data is not None:
        plt.legend()

    if save_figure:
        plt.savefig((graphs_path / graph_save_name).resolve())  # this is saving them weird due to subplots
        plt.show()
    return plt


def create_yesno_subplots(checkpoint_name, metrics, losses):
    num_epochs = range(1, len(metrics) + 1)  # might need to add 1

    accuracy = [metric_dict["yesno"][0]["accuracy"] for metric_dict in metrics]
    precision = [metric_dict["yesno"][0]["precision"] for metric_dict in metrics]
    recall = [metric_dict["yesno"][0]["recall"] for metric_dict in metrics]
    # f1_y = [metric_dict["yesno"][0]["f1_y"] for metric_dict in metrics]
    # f1_n = [metric_dict["yesno"][0]["f1_n"] for metric_dict in metrics]
    f1_ma = [metric_dict["yesno"][0]["f1_ma"] for metric_dict in metrics]

    plt.subplot(2, 2, 1)
    draw_graph(graph_title="Accuracy",
               data=accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="r",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplot(2, 2, 2)
    draw_graph(graph_title="Precision and Recall",
               data=precision,
               data_label="Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Metric Value",
               more_data=recall,
               more_data_label="Recall",
               color="g",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplot(2, 2, 3)
    draw_graph(graph_title="F1 Score",
               data=f1_ma,
               data_label="Macro Avg F1 Score",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="c",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)


    plt.subplot(2, 2, 4)
    draw_graph(graph_title="Loss",
               data=losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss",
               color="C0",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplots_adjust(hspace=0.6, wspace=0.45)
    plt.savefig(str(graphs_path) + "/" + checkpoint_name + "_yesno_finetuning_subplot.png")  # this is saving them weird due to subplots
    plt.show()
    print("\nGraph creation complete.\n")

def create_factoid_subplots(checkpoint_name, metrics, losses):
    num_epochs = range(1, len(metrics) + 1)  # might need to add 1

    strict_accuracy = [metric_dict["factoid"][0]["strict_accuracy"] for metric_dict in metrics]
    lenient_accuracy = [metric_dict["factoid"][0]["leniant_accuracy"] for metric_dict in metrics]
    mrr = [metric_dict["factoid"][0]["mrr"] for metric_dict in metrics]

    plt.subplot(2, 2, 1)
    draw_graph(graph_title="Strict Accuracy",
               data=strict_accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="r",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplot(2, 2, 2)
    draw_graph(graph_title="Lenient Accuracy",
               data=lenient_accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="g",
               save_figure=False)


    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplot(2, 2, 3)
    draw_graph(graph_title="MRR Score",
               data=mrr,
               data_label="Mean Reciprocal Rank (MRR)",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="c",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)


    plt.subplot(2, 2, 4)
    draw_graph(graph_title="Loss",
               data=losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss",
               color="C0",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplots_adjust(hspace=0.6, wspace=0.45)
    plt.savefig(str(graphs_path) + "/" + checkpoint_name + "_factoid_finetuning_subplot.png")  # this is saving them weird due to subplots
    plt.show()
    print("\nGraph creation complete.\n")


def create_list_subplots(checkpoint_name, metrics, losses):
    num_epochs = range(1, len(metrics) + 1)  # might need to add 1

    mean_average_precision = [metric_dict["list"][0]["mean_average_precision"] for metric_dict in metrics]
    mean_average_recall = [metric_dict["list"][0]["mean_average_recall"] for metric_dict in metrics]
    mean_average_f1 = [metric_dict["list"][0]["mean_average_f1"] for metric_dict in metrics]

    plt.subplot(2, 2, 1)
    draw_graph(graph_title="Mean Avg Precision",
               data=mean_average_precision,
               data_label="Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="r",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplot(2, 2, 2)
    draw_graph(graph_title="Mean Avg Recall",
               data=mean_average_recall,
               data_label="Recall",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="g",
               save_figure=False)


    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplot(2, 2, 3)
    draw_graph(graph_title="Mean Avg F1",
               data=mean_average_f1,
               data_label="F1 Score",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               color="c",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)


    plt.subplot(2, 2, 4)
    draw_graph(graph_title="Loss",
               data=losses,
               data_label="Loss",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name,
               y_label="Loss",
               color="C0",
               save_figure=False)

    for item in ([plt.gca().xaxis.label, plt.gca().yaxis.label]): item.set_fontsize(axis_font_size)

    plt.subplots_adjust(hspace=0.6, wspace=0.45)
    plt.savefig(str(graphs_path) + "/" + checkpoint_name + "_list_finetuning_subplot.png")  # this is saving them weird due to subplots
    plt.show()
    print("\nGraph creation complete.\n")


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

    accuracy = [metric_dict["yesno"][0]["accuracy"] for metric_dict in metrics]
    precision = [metric_dict["yesno"][0]["precision"] for metric_dict in metrics]
    recall = [metric_dict["yesno"][0]["recall"] for metric_dict in metrics]
    # f1_y = [metric_dict["yesno"][0]["f1_y"] for metric_dict in metrics]
    # f1_n = [metric_dict["yesno"][0]["f1_n"] for metric_dict in metrics]
    f1_ma = [metric_dict["yesno"][0]["f1_ma"] for metric_dict in metrics]

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


def visualise_factoid(checkpoint_name, metrics):
    num_epochs = range(1, len(metrics) + 1)  # might need to add 1
    fig_size = (5, 5)

    strict_accuracy = [metric_dict["factoid"][0]["strict_accuracy"] for metric_dict in metrics]
    lenient_accuracy = [metric_dict["factoid"][0]["leniant_accuracy"] for metric_dict in metrics]
    mrr = [metric_dict["factoid"][0]["mrr"] for metric_dict in metrics]

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Strict Accuracy",
               data=strict_accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)


    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Lenient Accuracy",
               data=lenient_accuracy,
               data_label="Accuracy",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="MRR Score",
               data=mrr,
               data_label="Mean Reciprocal Rank (MRR)",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)


def visualise_list(checkpoint_name, metrics):
    # metrics = {
    #         "mean_average_precision": mean_average_precision,
    #         "mean_average_recall": mean_average_recall,
    #         "mean_average_f1": mean_average_f1,
    #     }

    num_epochs = range(1, len(metrics) + 1)  # might need to add 1
    fig_size = (5, 5)

    mean_average_precision = [metric_dict["list"][0]["mean_average_precision"] for metric_dict in metrics]
    mean_average_recall = [metric_dict["list"][0]["mean_average_recall"] for metric_dict in metrics]
    mean_average_f1 = [metric_dict["list"][0]["mean_average_f1"] for metric_dict in metrics]

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Mean Avg Precision",
               data=mean_average_precision,
               data_label="Precision",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)


    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Mean Avg Recall",
               data=mean_average_recall,
               data_label="Recall",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)

    plt.figure(3, figsize=fig_size)
    draw_graph(graph_title="Mean Avg F1",
               data=mean_average_f1,
               data_label="F1 Score",
               epochs=num_epochs,
               checkpoint_name=checkpoint_name)



def load_stats_from_checkpoint(path_to_checkpoint, checkpoint_name, question_type):

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

    losses = settings["finetune_statistics"]["losses"]

    if len(losses) == 0:
        raise Exception("Losses list does not contain any data.")
    print(losses)

    if question_type == "yesno":
        visualise_yes_no(checkpoint_name, metrics)
        create_yesno_subplots(checkpoint_name, metrics, losses)
    elif question_type == "factoid":
        visualise_factoid(checkpoint_name, metrics)
        create_factoid_subplots(checkpoint_name, metrics, losses)
    elif question_type == "list":
        visualise_list(checkpoint_name, metrics)
        create_list_subplots(checkpoint_name, metrics, losses)

    print("Graph creation complete.\n")


if __name__ == "__main__":
    chckpt_name = "small_factoid,list_18_64089_29_249"    # e.g. small_10_50

    if len(chckpt_name) == 0:
        raise ValueError("Checkpoint name must be the name of a valid checkpoint e.g. small_10_50")

    checkpoint_path = (checkpoint_dir / chckpt_name).resolve()
    load_stats_from_checkpoint(checkpoint_path, chckpt_name, "list")
