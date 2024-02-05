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


def reliability_mortality_rate(technique, save_fig=True):
    matplotlib.rcParams.update({"font.size": 6})
    rul_pdfs_dict = json.load(
        open(f"results/{technique}/prognostics/pdf_ruls_{technique}.json", "r")
    )
    # cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    lengths = []
    for i in range(5, 9):
        lengths.append(len(rul_pdfs_dict[f"traj_{i}"]))
    for i in range(4):
        for j in range(4):
            if j == 0:
                timestep = 0
            else:
                timestep = int(0.25 * j * lengths[i])
            pdf = rul_pdfs_dict[f"traj_{i+5}"][f"timestep_{timestep}"]
            pdf = pdf[: lengths[i]]
            pdf_sum = np.cumsum(pdf)
            cdf = pdf_sum / pdf_sum[-1]
            cdf = 1 - cdf
            ax[i, j].plot(cdf, linewidth=1.0)
            if technique == "mimic":
                ax[i, j].set_xlabel("Time (h)")
                ax[i, j].set_ylabel("Survivability rate")
            else:
                ax[i, j].set_xlabel("Time (cycles)")
                ax[i, j].set_ylabel("Reliability")
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            if technique == "mimic":
                ax[i, j].set_title(
                    f"Patient {i+6} at {int(0.25*j*lengths[i])} h",
                    fontsize=6,
                    weight="bold",
                )
            else:
                ax[i, j].set_title(
                    f"Engine {i+6} at {int(0.25*j*lengths[i])} cycles",
                    fontsize=6,
                    weight="bold",
                )
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.3)
    if save_fig:
        # save figure
        fig.savefig(
            f"results/{technique}/figs/cdf_plot.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
    matplotlib.rcParams.update({"font.size": 5})


def kaplan_meier(technique, enable_pvalues=True, save_fig=True):
    pd.options.mode.chained_assignment = None
    matplotlib.rcParams.update({"font.size": 5})
    df_test = pd.read_csv(f"events/test_{technique}_events.csv")
    cm = 1 / 2.54  # centimeters in inches
    fig = plt.figure(figsize=(6.5 * cm, 6.5 * cm))
    df = pd.DataFrame()
    df_test["group"] = 0
    df_test["events"] = df_test["events"].astype(bool)
    true_events = df_test["time"]
    pred_events_dict = json.load(
        open(f"results/{technique}/prognostics/mean_rul_per_step_{technique}.json", "r")
    )
    for plot, t in enumerate(true_events):
        eol_all = []
        eol_all.append(0)
        for i in range(
            plot
        ):  # this event has happened, so we need to replace the predicted with the true value
            eol_all.append(true_events[i + 1])
        for i in range(plot, 10):
            if t >= len(pred_events_dict[f"traj_{i}"]):
                eol_all.append(pred_events_dict[f"traj_{i}"][-1] + t)
            else:
                eol_all.append(pred_events_dict[f"traj_{i}"][t] + t)
        df_pred = pd.DataFrame()
        df_pred["time"] = eol_all
        df_pred["events"] = np.ones((len(true_events),))
        df_pred["events"][0] = 0
        df_pred["group"] = plot
        df = pd.concat([df, df_pred])
    # convert events from integers to booleans
    df["events"] = df["events"].astype(int).astype(bool)
    # keep the values that come from group 10
    df_true = df[df["group"] == 10]
    group_indicator_true = df_true["group"].values
    # reverse the column positions
    df_true = df_true[["group", "events", "time"]]
    df_true = df_true[["events", "time"]]

    for group in df["group"].unique():
        mask = df["group"] == group
        if group < 10:
            df_pred = pd.DataFrame(columns=["events", "time"])
            df_pred = pd.concat([df_pred, df[["events", "time"]][mask]])
            group_indicator_pred = df["group"][mask].values
            df_pred = pd.concat([df_pred, df_true])
            df_data = df_pred.to_numpy()
            struct_arr = [(e1, e2) for e1, e2 in df_data]
            y = np.array(struct_arr, dtype=[("events", "?"), ("time", "<f8")])
            group_indicator = np.concatenate(
                (group_indicator_true, group_indicator_pred), axis=0
            )
            _, pvalue = compare_survival(y, group_indicator)
        time, survival_prob, ci = kaplan_meier_estimator(
            df["events"][mask], df["time"][mask], conf_level=0.95, conf_type="log-log"
        )
        if technique == "mimic":
            label = "Ground truth" if group == 10 else f"{true_events[group]} h"
        else:
            label = "Ground truth" if group == 10 else f"{true_events[group]} c"
        if group < 10:
            if enable_pvalues:
                label += f", CI={np.mean(ci[0]):.2f}-{np.mean(ci[1]):.2f}, p-value={pvalue:.2f}"
            else:
                label += f", CI={np.mean(ci[0]):.2f}-{np.mean(ci[1]):.2f}"
        plt.step(time, survival_prob, where="post", label=label, linewidth=0.3)
    # set color of the lines, there should be 11 colors
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "black",
    ]
    for i, line in enumerate(plt.gca().lines):
        line.set_color(colors[i])

    plt.ylim(0, 1.1)
    if technique == "mimic":
        legend = plt.legend(loc="best", fontsize=3.5)
        plt.xlabel("Time (h)")
    else:
        legend = plt.legend(loc="best", fontsize=3.0)
        plt.xlabel("Time (cycles)")
    plt.ylabel("Survival probability")
    legend.set_zorder(0)
    legend.get_frame().set_linewidth(0)
    # set the spines invisible for figure
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save_fig:
        # save figure
        fig.savefig(
            f"results/{technique}/figs/km_plot.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )


def auc_curve(
    y_true,
    y_pred_model,
    y_pred_sofa,
    y_pred_saps,
    y_pred_apache,
    lengths,
    technique="mimic",
    save_fig=True,
):
    matplotlib.rcParams.update({"font.size": 6})
    cm = 1 / 2.54  # centimeters in inches
    for n_fig in range(4):
        fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_model[:, n_fig])
        ax.plot(
            fpr,
            tpr,
            label=f"DSMC Model: AUC = {roc_auc_score(y_true, y_pred_model[:, n_fig]):.2f}",
            linewidth=0.3,
        )
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_sofa[:, n_fig])
        ax.plot(
            fpr,
            tpr,
            label=f"SOFA: AUC = {roc_auc_score(y_true, y_pred_sofa[:, n_fig]):.2f}",
            linewidth=0.3,
        )
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_saps[:, n_fig])
        ax.plot(
            fpr,
            tpr,
            label=f"SAPS III: AUC = {roc_auc_score(y_true, y_pred_saps[:, n_fig]):.2f}",
            linewidth=0.3,
        )
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_apache[:, n_fig])
        ax.plot(
            fpr,
            tpr,
            label=f"APACHE II: AUC = {roc_auc_score(y_true, y_pred_apache[:, n_fig]):.2f}",
            linewidth=0.3,
        )
        # show threshold
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=0.3,
            color="black",
            label=f"Threshold = 0.50",
        )
        ax.legend(loc="lower right")
        ax.set_title(f"{lengths[n_fig]} hours", weight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if save_fig:
            # save figure
            fig.savefig(
                f"results/{technique}/figs/roc_{lengths[n_fig]}.pdf",
                format="pdf",
                dpi=300,
                bbox_inches="tight",
            )
    n_fig = 2
    print(
        f"After {lengths[n_fig]} hours, the AUC score of DSMC Model is {roc_auc_score(y_true, y_pred_model[:, n_fig]):.2f},"
        f"the AUC score of SOFA is {roc_auc_score(y_true, y_pred_sofa[:, n_fig]):.2f},"
        f"the AUC score of SAPS III is {roc_auc_score(y_true, y_pred_saps[:, n_fig]):.2f},"
        f" and the AUC score of APACHE II is {roc_auc_score(y_true, y_pred_apache[:, n_fig]):.2f}"
    )
    print(
        f"\nAdditional scores for those models are: \n"
        f"Precision of DSMC Model is {precision_score(y_true, y_pred_model[:, n_fig] > 0.5):.2f},"
        f"Precision of SOFA is {precision_score(y_true, y_pred_sofa[:, n_fig] > 0.5):.2f},"
        f"Precision of SAPS III is {precision_score(y_true, y_pred_saps[:, n_fig] > 0.5):.2f},"
        f"Precision of APACHE II is {precision_score(y_true, y_pred_apache[:, n_fig] > 0.5):.2f}"
    )
    print(
        f"\nRecall of DSMC Model is {recall_score(y_true, y_pred_model[:, n_fig] > 0.5):.2f},"
        f"Recall of SOFA is {recall_score(y_true, y_pred_sofa[:, n_fig] > 0.5):.2f},"
        f"Recall of SAPS III is {recall_score(y_true, y_pred_saps[:, n_fig] > 0.5):.2f},"
        f" and Recall of APACHE II is {recall_score(y_true, y_pred_apache[:, n_fig] > 0.5):.2f}"
    )
    print(
        f"\nF1 score of DSMC Model is {f1_score(y_true, y_pred_model[:, n_fig] > 0.5):.2f},"
        f"F1 score of SOFA is {f1_score(y_true, y_pred_sofa[:, n_fig] > 0.5):.2f},"
        f"F1 score of SAPS III is {f1_score(y_true, y_pred_saps[:, n_fig] > 0.5):.2f},"
        f" and F1 score of APACHE II is {f1_score(y_true, y_pred_apache[:, n_fig] > 0.5):.2f}"
    )


def plot_rul(
    path, technique, rmse, true_ruls, pred_ruls, lower_ruls, upper_ruls, save_fig=True
):
    color = "blue"
    method = "HSMM"
    matplotlib.rcParams.update({"font.size": 6})
    cm = 1 / 2.54  # centimeters in inches
    files = glob.glob(path + f"clustering_results_{technique}_Test_*")
    files = sorted(files, key=lambda x: int(x.split("_sp")[-1].split(".")[0]))
    for graph in range(1, len(pred_ruls) + 1):
        times = pd.read_csv(files[graph - 1], header=0).values[:, 1]

        fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
        ax.plot(
            times,
            pred_ruls[f"traj_{graph-1}"],
            linewidth=0.3,
            color=color,
            label=f"{method} model prediction (rmse: {rmse[graph-1]:.2f})",
        )
        if lower_ruls is not None:
            ax.plot(
                times,
                lower_ruls[f"traj_{graph-1}"],
                linewidth=0.1,
                ls="-.",
                color=color,
            )
            ax.plot(
                times,
                upper_ruls[f"traj_{graph-1}"],
                linewidth=0.1,
                ls="-.",
                color=color,
                label=f"{method} model 95% Confidence intervals",
            )

        ax.plot(
            times,
            true_ruls[f"traj_{graph-1}"],
            linewidth=0.3,
            ls="--",
            color="black",
            label="Ground truth",
        )
        ax.legend(fontsize=4)

        fig.suptitle(f"Test specimen {graph}", weight="bold")
        ax.set_ylim(-30, 500)
        ax.set_xlabel("Cycles")
        ax.set_ylabel("RUL (cycles)")
        # set the spines invisible for figure
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if save_fig:
            fig.savefig(
                f"results/{technique}/figs/ruls_specimen_{graph}.pdf",
                format="pdf",
                dpi=300,
                bbox_inches="tight",
            )


def compare_plot_rul(
    rmse_gbdt,
    ruls_gbdt,
    true_ruls,
    lower_ruls_gbdt,
    upper_ruls_gbdt,
    rmse_svr,
    ruls_svr,
    lower_ruls_svr,
    upper_ruls_svr,
    rmse_hsmm,
    ruls_hsmm,
    lower_ruls_hsmm,
    upper_ruls_hsmm,
    save_fig=True,
):
    colors = ["blue", "red", "green"]
    methods = ["GBDT", "SVR", "HSMM"]
    matplotlib.rcParams.update({"font.size": 6})
    cm = 1 / 2.54  # centimeters in inches
    for graph in [4, 9]:
        fig, ax = plt.subplots(1, figsize=(6.5 * cm, 6.5 * cm))
        for i, (rmse, ruls, lower_ruls, upper_ruls) in enumerate(
            zip(
                [rmse_gbdt, rmse_svr, rmse_hsmm],
                [ruls_gbdt, ruls_svr, ruls_hsmm],
                [lower_ruls_gbdt, lower_ruls_svr, lower_ruls_hsmm],
                [upper_ruls_gbdt, upper_ruls_svr, upper_ruls_hsmm],
            )
        ):
            ax.plot(
                ruls[f"traj_{graph}"],
                linewidth=0.3,
                color=colors[i],
                label=f"{methods[i]} model prediction (rmse: {rmse[graph]:.2f})",
            )
            if lower_ruls is not None:
                ax.plot(
                    lower_ruls[f"traj_{graph}"],
                    linewidth=0.1,
                    ls="-.",
                    color=colors[i],
                )
                ax.plot(
                    upper_ruls[f"traj_{graph}"],
                    linewidth=0.1,
                    ls="-.",
                    color=colors[i],
                    label=f"{methods[i]} model 95% Confidence intervals",
                )

        ax.plot(
            true_ruls[f"traj_{graph}"],
            linewidth=0.3,
            ls="--",
            color="black",
            label="Ground truth",
        )
        ax.legend(fontsize=3)
        fig.suptitle(f"Test engine {graph + 1}", weight="bold")
        ax.set_ylim(-30, 500)
        ax.set_xlabel("Cycles")
        ax.set_ylabel("RUL (cycles)")
        # set the spines invisible for figure
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if save_fig:
            fig.savefig(
                f"results/cmaps/figs/compare_ruls_engine_{graph+1}.pdf",
                format="pdf",
                dpi=300,
                bbox_inches="tight",
            )


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
