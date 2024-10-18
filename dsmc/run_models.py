from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch as th
from torch import nn
import utils, visualize
import settings
import pandas as pd
import numpy as np


def train_ae(
    model,
    technique,
    train_loader,
    val_loader,
    ae_lr,
    alpha,
    device,
    epochs=100,
    bayes_opt=False,
    save=True,
):
    """
    Train the first part DSMC model, i.e. the Autoencoder (AE), given a dataset, a model instance and various configuration parameters.
    :param model: instance of Dataloader to use for static data
    :param technique: technique to use for training
    :param train_loader: instance of Dataloader to use for training
    :param val_loader: instance of Dataloader to use for validation
    :param ae_lr: learning rate for the AE model
    :param alpha: weight of the reconstruction loss
    :param device: device to use for training, defaults to 'gpu' if available
    :param epochs: number of training epochs
    :param bayes_opt: whether to use bayesian optimization for hyperparameter tuning
    :param save: whether to save the figures
    :return: AE model instance
    """

    if technique == "mimic":
        use_demo = True
    else:
        use_demo = False
    loss_function = th.nn.MSELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=ae_lr)
    train_loss_log = []
    train_time_loss_log = []
    val_loss_log = []
    val_time_loss_log = []
    for epoch in tqdm(range(epochs)):
        reconstructed_loss = 0.0
        time_loss = 0.0
        train_loss = 0.0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if not technique == "fmoc":
                if use_demo:
                    time_series, time, demo = x
                    demo = demo.to(device).float().requires_grad_(True)
                else:
                    time_series, time = x
                    demo = None
                y = y.to(device).float()
                time_series = time_series.to(device).float()
            else:
                time_sequences, images = x
                time_series, time = time_sequences
                time_series = time_series.to(device).float()
                images = images.to(device).float()
                y_acoustic, y_images = y
                y_acoustic = y_acoustic.to(device).float()
                y_images = y_images.to(device).float()
                time_series = (time_series, images)
                y = (y_acoustic, y_images)
                demo = None

            time = time.to(device).float().requires_grad_(True)
            optimizer.zero_grad()
            (
                reconstructed,
                reconstructed_t,
                encoder_out,
                encoded_hidden_dim,
                decoded,
            ) = model(time_series, time, demo)
            loss0 = loss_function(encoded_hidden_dim, decoded)
            loss1 = loss_function(reconstructed_t.reshape(-1, 1), time)
            if technique == "fmoc":
                loss2 = loss_function(reconstructed[0], y[0]) + loss_function(
                    reconstructed[1], y[1]
                )
            else:
                loss2 = loss_function(reconstructed, y)
            loss = alpha * loss0 + alpha * loss1 + loss2
            loss.backward()
            optimizer.step()
            reconstructed_loss += loss2.item()
            time_loss += loss1.item()
            train_loss += loss.item()

        train_loss = train_loss / (i + 1)
        time_loss = time_loss / (i + 1)
        reconstructed_loss = reconstructed_loss / (i + 1)

        if not bayes_opt:
            model.eval()
            with th.no_grad():
                val_reconstructed_loss = 0.0
                val_time_loss = 0.0
                val_loss = 0.0
                for i, (x, y) in enumerate(val_loader):
                    if not technique == "fmoc":
                        if use_demo:
                            time_series, time, demo = x
                            demo = demo.to(device).float().requires_grad_(True)
                        else:
                            time_series, time = x
                            demo = None
                        y = y.to(device).float()
                        time_series = time_series.to(device).float()
                    else:
                        time_sequences, images = x
                        time_series, time = time_sequences
                        time_series = time_series.to(device).float()
                        images = images.to(device).float()
                        y_acoustic, y_images = y
                        y_acoustic = y_acoustic.to(device).float()
                        y_images = y_images.to(device).float()
                        time_series = (time_series, images)
                        y = (y_acoustic, y_images)
                        demo = None

                    time = time.to(device).float().requires_grad_(True)
                    (
                        reconstructed,
                        reconstructed_t,
                        encoder_out,
                        encoded_hidden_dim,
                        decoded,
                    ) = model(time_series, time, demo)
                    loss0 = loss_function(encoded_hidden_dim, decoded)
                    loss1 = loss_function(reconstructed_t.reshape(-1, 1), time)
                    if technique == "fmoc":
                        loss2 = 0.5 * (
                            loss_function(reconstructed[0], y[0])
                            + loss_function(reconstructed[1], y[1])
                        )
                    else:
                        loss2 = loss_function(reconstructed, y)
                    loss = alpha * loss0 + alpha * loss1 + loss2
                    val_reconstructed_loss += loss2.item()
                    val_time_loss += loss1.item()
                    val_loss += loss.item()
                val_loss = val_loss / (i + 1)
                val_time_loss = val_time_loss / (i + 1)
                val_reconstructed_loss = val_reconstructed_loss / (i + 1)
                """
                print(f'Epoch: {epoch + 1}, \t Train Loss: {train_loss:.4f} \t Val Loss: {val_loss:.4f} \t '
                      f'Reconstr.Loss: {reconstructed_loss:.4f} \t Val Reconstr.Loss: {val_reconstructed_loss:.4f}, \t '
                      f'Time Loss: {time_loss}, \t Val Time Loss: {val_time_loss}')
                """
                train_loss_log.append(train_loss)
                train_time_loss_log.append(time_loss)
                val_loss_log.append(val_loss)
                val_time_loss_log.append(val_time_loss)

    print("\nFinished the training of the AE model, saving variables for plotting...")

    train_loss_log = np.array(train_loss_log)
    train_time_loss_log = np.array(train_time_loss_log)
    val_loss_log = np.array(val_loss_log)
    val_time_loss_log = np.array(val_time_loss_log)

    if not bayes_opt:
        df_learn_loss = pd.DataFrame(
            {
                "epochs": np.arange(len(train_loss_log)),
                "train_loss": train_loss_log,
                "val_loss": val_loss_log,
            }
        )
        df_time_loss = pd.DataFrame(
            {
                "epochs": np.arange(len(train_loss_log)),
                "train_time_loss": train_time_loss_log,
                "val_time_loss": val_time_loss_log,
            }
        )
        visualize.plot_loss(
            df_learn_loss, technique, "Reconstruction loss", save_fig=save
        )
        visualize.plot_loss(df_time_loss, technique, "Time loss", save_fig=save)
        # save as csv
        df_learn_loss.to_csv(f"results/{technique}/loss/learning_loss.csv")
        df_time_loss.to_csv(f"results/{technique}/loss/time_loss.csv")
    return model


def evaluate_ae(model, scaler_t_f, technique, static_loader, device, save=True):
    print("\nEvaluating the AE model...")
    if technique == "mimic":
        use_demo = True
    else:
        use_demo = False
    model.train()  # it's not going to be trained, but we need to set it to train mode to get the gradients
    th.set_grad_enabled(True)
    model.disable_bn(True)
    monotonic_results = []
    time_grad_arr = np.zeros((len(static_loader), model.n_features))
    times = []
    for i, (x, y) in enumerate(static_loader):
        if not technique == "fmoc":
            if use_demo:
                time_series, time, demo = x
                demo = demo.to(device).float().requires_grad_(True)
            else:
                time_series, time = x
                demo = None
            time_series = time_series.to(device).float()
        else:
            time_sequences, images = x
            time_series, time = time_sequences
            time_series = time_series.to(device).float().requires_grad_(True)
            images = images.to(device).float().requires_grad_(True)
            time_series = (time_series, images)
            demo = None

        time = time.to(device).float().requires_grad_(True)
        time_grads = []
        time.register_hook(lambda grad: time_grads.append(grad[0].cpu().numpy()))
        y_pred, _, encoder_out, _, _ = model(time_series, time, demo)
        utils.time_gradient(encoder_out)
        time_grad_arr[i] = np.array(time_grads).squeeze()
        monotonic_results.append(encoder_out.detach().cpu().numpy())
        time = time.detach().cpu().numpy()
        time = scaler_t_f.inverse_transform(time.reshape(-1, 1)).squeeze()
        times.append(time)
    times = np.array(times).squeeze()
    time_grads_df = pd.DataFrame(
        {f"time_grads{k}": time_grad_arr[:, k] for k in range(time_grad_arr.shape[1])}
    )
    time_grads_df["time"] = times
    # save the time gradients
    time_grads_df.to_csv(f"results/{technique}/time_grads/time_grads.csv", index=False)
    visualize.visualize_time_gradients(time_grads_df, technique, save_fig=save)
    monotonic_results = np.array(monotonic_results).squeeze()
    z_df = pd.DataFrame(
        {f"z{k}": monotonic_results[:, k] for k in range(monotonic_results.shape[1])}
    )
    z_df["time"] = times
    z_df.to_csv(f"results/{technique}/z_space/z_space_epoch_1.csv", index=False)
    visualize.visualize_z_space(monotonic_results, times, 1, technique, save_fig=save)


def train_cluster(
    static_loader: th.utils.data.DataLoader,
    train_loader: th.utils.data.DataLoader,
    cluster_model: th.nn.Module,
    scaler_t_f: MinMaxScaler,
    technique: str,
    alpha: float,
    beta: float,
    epochs: int,
    dc_lr: float,
    device: th.device,
    bayes_opt: bool = False,
    save: bool = True,
) -> th.nn.Module:
    """
    Train the second part of the DSMC model, i.e. the deep clustering (DC) model given a dataset, a model instance and various configuration parameters.
    :param static_loader: instance of Dataloader to use for static data
    :param train_loader: instance of Dataloader to use for training
    :param val_loader: instance of Dataloader to use for validation
    :param cluster_model: instance of DC model to train
    :param scaler_t_f: scaler instance to use for time and feature scaling
    :param technique: technique to use for training
    :param alpha: weight of the reconstruction loss
    :param beta: weight of the time loss
    :param epochs: number of training epochs
    :param dc_lr: learning rate for the DC model
    :param device: device to use for training, defaults to 'gpu' if available
    :param bayes_opt: whether to use bayesian optimization for hyperparameter tuning
    :param save: whether to save the figures
    :return: cluster_model instance
    """

    if technique == "mimic":
        use_demo = True
    else:
        use_demo = False

    print(
        "\nApply 1 forward pass of the entire dataset for centroid initialization (with k-means)"
    )
    """
    # print last layer weights
    print("before kmeans")
    for name, param in cluster_model.named_parameters():
        print(name, param.data)
    """
    cluster_model.eval()
    kmeans = KMeans(
        n_clusters=cluster_model.cluster_number,
        n_init=20,
        random_state=settings.seed_number,
    )
    features = []
    # form initial cluster centres
    for i, (x, y) in enumerate(tqdm(static_loader)):
        if not technique == "fmoc":
            if use_demo:
                time_series, time, demo = x
                demo = demo.to(device).float().requires_grad_(True)
            else:
                time_series, time = x
                demo = None
            time_series = time_series.to(device).float()
        else:
            time_sequences, images = x
            time_series, time = time_sequences
            time_series = time_series.to(device).float()
            images = images.to(device).float()
            time_series = (time_series, images)
            demo = None

        time = time.to(device).float().requires_grad_(True)
        features.append(
            cluster_model.encoder(time_series, time, demo)[2].detach().cpu()
        )
    predicted = kmeans.fit_predict(th.cat(features).numpy())
    if not bayes_opt:
        if not technique == "fmoc":
            visualize.visualize_cluster_embeddings(
                th.cat(features).numpy(),
                predicted,
                int(cluster_model.cluster_number),
                1,
                technique,
                save_fig=save,
            )
    kmeans.cluster_centers_ = np.sort(kmeans.cluster_centers_, axis=0)
    cluster_centers = th.tensor(
        kmeans.cluster_centers_, dtype=th.float, requires_grad=True
    )
    if device.type == "cuda":
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    else:
        cluster_centers = cluster_centers.cpu()

    with th.no_grad():
        # initialise the cluster centers
        cluster_model.assignment.cluster_centers.data.copy_(cluster_centers)

    # optimizer = th.optim.SGD(cluster_model.parameters(), lr=dc_lr, momentum=0.9)
    optimizer = th.optim.Adam(cluster_model.parameters(), lr=dc_lr)
    loss_function = nn.KLDivLoss(size_average=False)
    loss_mse = nn.MSELoss()
    cluster_model.train()
    cluster_model.encoder.disable_bn(True)

    print("\nTraining the second stage of the DSMC model")
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0

        for i, (x, y) in enumerate(train_loader):
            if not technique == "fmoc":
                if use_demo:
                    time_series, time, demo = x
                    demo = demo.to(device).float().requires_grad_(True)
                else:
                    time_series, time = x
                    demo = None
                y = y.to(device).float()
                time_series = time_series.to(device).float()
            else:
                time_sequences, images = x
                time_series, time = time_sequences
                time_series = time_series.to(device).float()
                images = images.to(device).float()
                y_acoustic, y_images = y
                y_acoustic = y_acoustic.to(device).float()
                y_images = y_images.to(device).float()
                time_series = (time_series, images)
                y = (y_acoustic, y_images)
                demo = None

            time = time.to(device).float().requires_grad_(True)
            (
                output,
                reconstructed,
                reconstructed_t,
                encoded,
                encoded_hidden_dim,
                decoded,
            ) = cluster_model(time_series, time, demo)
            target = utils.target_distribution(output).detach()
            loss1 = loss_function(output.log(), target) / output.shape[0]
            if technique == "fmoc":
                loss2 = (
                    0.5
                    * (
                        loss_mse(reconstructed[0], y[0])
                        + loss_mse(reconstructed[1], y[1])
                    )
                    + alpha * loss_mse(reconstructed_t.reshape(-1, 1), time)
                    + alpha * loss_mse(encoded_hidden_dim, decoded)
                )
            else:
                loss2 = (
                    loss_mse(reconstructed, y)
                    + alpha * loss_mse(reconstructed_t.reshape(-1, 1), time)
                    + alpha * loss_mse(encoded_hidden_dim, decoded)
                )
            loss = loss1 + beta * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            loss_value = float(loss.item())
            epoch_loss += loss_value
            loss_value1 = float(loss1.item())
            loss_value2 = float(loss2.item())
            epoch_loss1 += loss_value1
            epoch_loss2 += loss_value2

    predicted, times, embeds = predict_cluster(
        static_loader, cluster_model, scaler_t_f, technique, device
    )
    predicted = predicted.numpy()
    times = times.numpy().squeeze()
    embeds = embeds.numpy()
    if not bayes_opt:
        z_df = pd.DataFrame({f"z{k}": embeds[:, k] for k in range(embeds.shape[1])})
        z_df["time"] = times
        z_df.to_csv(
            f"results/{technique}/z_space/z_space_epoch_{epochs}.csv", index=False
        )
        visualize.visualize_z_space(embeds, times, epochs, technique, save_fig=save)
        if not technique == "fmoc":
            visualize.visualize_cluster_embeddings(
                embeds,
                predicted,
                int(cluster_model.cluster_number),
                epoch + 1,
                technique,
                save_fig=save,
            )
    cluster_model.train()
    cluster_model.encoder.disable_bn(True)
    print("\nFinished the training of DSMC model")
    return cluster_model


def predict_cluster(static_loader, model, scaler_t_f, technique, device):
    if technique == "mimic":
        use_demo = True
    else:
        use_demo = False
    features = []
    model.eval()
    times = []
    embeds = []
    with th.no_grad():
        for i, (x, y) in enumerate(static_loader):
            if not technique == "fmoc":
                if use_demo:
                    time_series, time, demo = x
                    demo = demo.to(device).float().requires_grad_(True)
                else:
                    time_series, time = x
                    demo = None
                time_series = time_series.to(device).float()
            else:
                time_sequences, images = x
                time_series, time = time_sequences
                time_series = time_series.to(device).float()
                images = images.to(device).float()
                time_series = (time_series, images)
                demo = None

            time = time.to(device).float().requires_grad_(True)
            output, _, _, encoded, _, _ = model(time_series, time, demo)
            time = time.detach().cpu().numpy()
            time = scaler_t_f.inverse_transform(time.reshape(-1, 1))
            times.append(th.from_numpy(time))
            features.append(
                output.detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU
            embeds.append(encoded.detach().cpu())
    return th.cat(features).max(1)[1], th.cat(times), th.cat(embeds)


def evaluate_dc_model(
    cluster_model,
    train_dict,
    test_dict,
    time_train_dict,
    time_test_dict,
    train_demo_dict,
    test_demo_dict,
    scaler_x,
    scaler_t_f,
    scaler_demo,
    n_clusters,
    window_length,
    step_size,
    device,
    technique,
    save,
):
    print("\nApplying DSMC model to test data")

    for process in ["Train", "Test"]:
        if process == "Train":
            dict_letter = "a"
            data_dict = train_dict
            time_dict = time_train_dict
            demo_dict = train_demo_dict
        else:
            dict_letter = "te_a"
            data_dict = test_dict
            time_dict = time_test_dict
            demo_dict = test_demo_dict

        df_list = []
        for i in range(
            len(data_dict[0]) if isinstance(data_dict, tuple) else len(data_dict)
        ):
            if isinstance(data_dict, tuple):
                trajectory_acoustic = np.array(
                    list(data_dict[0][dict_letter + str(i + 1)].values())
                )
                trajectory_acoustic = np.transpose(trajectory_acoustic, (1, 2, 0))
                # Normalize data
                shape_tr = trajectory_acoustic.shape
                trajectory_acoustic = trajectory_acoustic.reshape(
                    -1, trajectory_acoustic.shape[-1]
                )
                trajectory_acoustic = scaler_x.transform(trajectory_acoustic)
                trajectory_acoustic = trajectory_acoustic.reshape(shape_tr)
                trajectory_dic = np.array(list(data_dict[1][dict_letter + str(i + 1)]))
                # add one dimension to position 1
                trajectory_dic = np.expand_dims(trajectory_dic, axis=1)
                trajectory = (trajectory_acoustic, trajectory_dic)

                assert (
                    trajectory_acoustic.shape[0] == trajectory_dic.shape[0]
                ), "In {} phase, the number of windows of acoustic emission and DIC images should be the same, but they are {} and {} respectively".format(
                    process, trajectory_acoustic.shape[0], trajectory_dic.shape[0]
                )
            else:
                trajectory = np.array(
                    list(data_dict[dict_letter + str(i + 1)].values())
                )
                trajectory = np.transpose(trajectory, (1, 2, 0))
                # Normalize data
                shape_tr = trajectory.shape
                trajectory = trajectory.reshape(-1, trajectory.shape[-1])
                trajectory = scaler_x.transform(trajectory)
                trajectory = trajectory.reshape(shape_tr)
            time = np.array(time_dict[dict_letter + str(i + 1)])
            time = scaler_t_f.transform(time.reshape(-1, 1))
            if demo_dict is not None:
                demo = np.array(list(demo_dict[dict_letter + str(i + 1)].values()))
                demo = np.transpose(demo, (1, 0))
                demo = scaler_demo.transform(demo)
            else:
                demo = None
            features = []
            times = []
            embds = []
            for j in range(
                trajectory[0].shape[0]
                if isinstance(trajectory, tuple)
                else trajectory.shape[0]
            ):
                if isinstance(trajectory, tuple):
                    t_series = (
                        th.from_numpy(trajectory[0][j]).float().unsqueeze(0).to(device),
                        th.from_numpy(trajectory[1][j]).float().unsqueeze(0).to(device),
                    )
                else:
                    t_series = (
                        th.from_numpy(trajectory[j]).float().unsqueeze(0).to(device)
                    )
                t = th.from_numpy(time[j]).float().unsqueeze(0).to(device)
                d = (
                    th.from_numpy(demo[j]).float().unsqueeze(0).to(device)
                    if demo is not None
                    else None
                )
                output, _, _, encoded, _, _ = cluster_model(t_series, t, d)
                times.append(t.detach().cpu())
                features.append(
                    output.detach().cpu()
                )  # move to CPU to prevent out of memory on the GPU
                embds.append(encoded.detach().cpu())
            labels, times = th.cat(features).max(1)[1], th.cat(times)
            labels = labels.numpy()
            times = times.numpy().squeeze()
            times = scaler_t_f.inverse_transform(times.reshape(-1, 1)).squeeze()

            # convert auxiliary time to real time
            if technique == "fmoc":
                auxiliary_times = np.arange(times.shape[0]) + 1
                auxiliary_times[0] = (window_length - 1) * 50 * auxiliary_times[
                    0
                ] + 6000
                for k in range(1, len(auxiliary_times)):
                    auxiliary_times[k] = auxiliary_times[k - 1] + 50 * step_size
            else:
                auxiliary_times = np.arange(times.shape[0])
            df = pd.DataFrame({"clusters": labels, "time": auxiliary_times})
            # sort dataframe according to column time
            df = df.sort_values(by=["time"])

            # the value of the last row of df should be the last cluster as we reach the EOL
            df.iloc[-1, 0] = n_clusters
            if technique == "fmoc":
                df.iloc[-1, 1] = auxiliary_times[-1]

            path = f"./results/{technique}/clusters/clustering_results_{technique}_{process}_sp{i + 1}.csv"
            df.to_csv(path, index=False)
            df_list.append(df)
        if process == "Test":
            visualize.visualize_cluster_results(
                df_list,
                int(cluster_model.cluster_number),
                technique,
                save_fig=save,
            )