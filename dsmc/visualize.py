import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.manifold import TSNE
import settings
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
import matplotlib


def plot_loss(df, technique, loss_type, save_fig=True):
    matplotlib.rcParams.update({"font.size": 5})
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
    # add more ticks
    if "Reconstruction" in loss_type:
        colors = ["blue", "orange"]
        ax.set_yticks(
            np.arange(
                0, np.array([df["train_loss"].max(), df["val_loss"].max()]).max(), 0.2
            )
        )
        df.plot(x="epochs", ax=ax, linewidth=0.3, color=colors)
        legend = ax.legend(
            ["Train loss", "Validation loss"], fontsize=5, loc="upper right"
        )
    else:
        colors = ["green", "red"]
        ax.set_yticks(
            np.arange(
                0,
                np.array(
                    [df["train_time_loss"].max(), df["val_time_loss"].max()]
                ).max(),
                0.1,
            )
        )
        df.plot(x="epochs", ax=ax, linewidth=0.3, color=colors)
        legend = ax.legend(
            ["Train time loss", "Validation time loss"], fontsize=5, loc="upper right"
        )
    legend.get_frame().set_linewidth(0)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(loss_type)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # change the name of the legend
    if save_fig:
        # save figure
        fig.savefig(
            f"results/{technique}/figs/{loss_type}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


def visualize_time_gradients(time_grad_df, technique, save_fig=True):
    matplotlib.rcParams.update({"font.size": 6})
    # short time gradients by 'time' column
    time_grad_df = time_grad_df.sort_values(by=["time"])
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
    for k in range(len(time_grad_df.columns) - 1):
        x = time_grad_df["time"].values
        y = time_grad_df.iloc[:, k].values
        window_size = 1
        moving_avg = pd.DataFrame(y).rolling(window_size).mean().values
        lower_bound = y - np.std(y)
        upper_bound = y + np.std(y)
        ax.plot(x, moving_avg, linewidth=0.3, label=f"z{k}")
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.2)
        # create a subplot inside the plot for the first z only
        matplotlib.rcParams.update({"font.size": 4})
        if (k == 1 and technique == "mimic") or (k == 2 and technique == "cmaps"):
            if k == 1 and technique == "mimic":
                fc = "orange"
                # Create the second subplot (zoomed-in plot) within the first subplot
                left, bottom, width, height = [
                    0.3,
                    0.25,
                    0.4,
                    0.2,
                ]  # Define the position and size
            else:
                fc = "green"
                left, bottom, width, height = [
                    0.3,
                    0.4,
                    0.4,
                    0.2,
                ]  # Define the position and size

            axins = fig.add_axes([left, bottom, width, height])

            # Plot the zoomed-in line (Line 1) on the second subplot
            axins.plot(x, moving_avg, linewidth=0.3, color=fc, label=f"z{k}")
            axins.fill_between(x, lower_bound, upper_bound, color=fc, alpha=0.2)

            # Set labels and legend for the second subplot
            legend = axins.legend(fontsize=3, loc="best")
            legend.get_frame().set_linewidth(0)
            axins.set_xlabel("Sequential samples")
            axins.set_ylabel("Time feature gradients")

    matplotlib.rcParams.update({"font.size": 6})
    # put in custom position the legend (to lower right but a little bit upper)
    legend = ax.legend(fontsize=5, loc="upper right", bbox_to_anchor=(1.05, 0.5))
    legend.get_frame().set_linewidth(0)
    ax.set_xlabel("Sequential samples")
    ax.set_ylabel("Time feature gradients")
    # set y axis to start from 0 until the point that it wants
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_fig:
        # save figure
        fig.savefig(
            f"results/{technique}/figs/time_gradients.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


def visualize_cluster_embeddings(
    features, clusters, n_clusters, epoch, technique, save_fig=True
):
    matplotlib.rcParams.update({"font.size": 5})
    projections = TSNE(
        n_components=2,
        random_state=settings.seed_number,
        perplexity=15.0,
        n_iter=500,
        init="pca",
    ).fit_transform(features)
    projections = MinMaxScaler().fit_transform(projections)
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
    df = pd.DataFrame()
    df["label"] = clusters + 1
    df["Component 1"] = projections[:, 0]
    df["Component 2"] = projections[:, 1]
    df.to_csv(f"results/{technique}/cluster_embds/embds_epoch_{epoch}.csv", index=False)
    sns.scatterplot(
        data=df,
        x="Component 1",
        y="Component 2",
        hue=df.label.tolist(),
        ax=ax,
        s=8,
        palette=sns.color_palette("husl", n_clusters),
    )
    legend = ax.legend(
        fontsize=5, markerscale=0.2, loc="lower left", bbox_to_anchor=(0.93, 0.01)
    )
    legend.get_frame().set_linewidth(0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if save_fig:
        # save figure
        fig.savefig(
            f"results/{technique}/figs/cluster_embeddings_epoch_{epoch}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


def visualize_z_space(monotonic_results, times, epoch, technique, save_fig=True):
    matplotlib.rcParams.update({"font.size": 5})
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
    monotonic_df = pd.DataFrame(monotonic_results)
    monotonic_df["time"] = times
    monotonic_df = monotonic_df.sort_values(by=["time"])
    for k in range(len(monotonic_results[0])):
        x = monotonic_df["time"].values
        y = monotonic_df.iloc[:, k].values
        window_size = 30
        if technique == "mimic":
            a = 0.2
        else:
            a = 0.2
        moving_avg = pd.DataFrame(y).rolling(window_size).mean().values
        # moving_avg = np.convolve(y, np.ones(window_size) / window_size, mode='same')
        lower_bound = y - np.std(y)
        upper_bound = y + np.std(y)
        ax.plot(x, moving_avg, linewidth=0.3, label=f"z{k}")
        ax.fill_between(x, lower_bound, upper_bound, alpha=a)
    ax.set_xlabel("Sequential samples")
    ax.set_ylabel("Z space")
    if technique == "mimic":
        legend = ax.legend(fontsize=5, loc="lower right", bbox_to_anchor=(1.15, 0.01))
    else:
        legend = ax.legend(fontsize=5, loc="upper left")
    legend.get_frame().set_linewidth(0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_fig:
        # save figure
        fig.savefig(
            f"results/{technique}/figs/z_space_epoch_{epoch}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


def visualize_cluster_results(df_list, n_clusters, technique, save_fig):
    matplotlib.rcParams.update({"font.size": 5})
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
    labels = []
    for i, df in enumerate(df_list):
        df.plot(x="time", ax=ax, linewidth=0.3)
        if technique == "mimic":
            label = f"Patient {i+1}"
            ax.set_xlabel("Time steps (h)")
        elif technique == "cmaps":
            label = f"Engine {i+1}"
            ax.set_xlabel("Time steps (cycles)")
        else:
            label = f"Specimen {i+1}"
            ax.set_xlabel("Time steps (seconds)")
        labels.append(label)

    legend = ax.legend(
        labels, fontsize=4, loc="lower right", bbox_to_anchor=(1.15, 0.03)
    )
    legend.get_frame().set_linewidth(0)
    ax.set_ylabel("Clusters")
    ax.set_yticks(range(1, n_clusters + 1))
    # remove box around plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_fig:
        fig.savefig(
            f"./results/{technique}/figs/clustering_results_{technique}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
