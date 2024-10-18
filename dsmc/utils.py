import os
import random
import json
import glob
import pandas as pd
import argparse
import run_models
import models
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from tslearn.preprocessing import TimeSeriesResampler
import joblib
import torch as th
from torch.utils.data import Dataset, DataLoader
from bayesian_opt import BayesianOptimization
from sepsis_score_systems import *
import settings


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class SynchronizeImageAE:
    def __init__(self, dic_specimens, ae_specimens, ignore_steps_images=2):
        self.dic_specimens = dic_specimens
        self.ae_specimens = ae_specimens
        self.ignore_steps_images = ignore_steps_images
        self.global_avg_index = None

    def synchronize_data(self, dic, ae):
        def find_nearest_timestamp_index(ae_timestamps, timestamp):
            closest_index = np.argmin(np.abs(ae_timestamps - timestamp))
            # closest_value = ae_timestamps[closest_index]
            return closest_index

        # store every self.ignore_steps_images, store every 50 * ignore_steps_images time steps
        time_window = 50 * self.ignore_steps_images

        d_index = []
        indx_val = []
        for file in dic:
            # Get the timestamp from the file name
            timestamp = int(file.split("_")[1].split(".")[0])
            assert timestamp >= time_window, "Time window is too large"

            # Matching AE data indices with DIC window indices
            start_timestamp = (
                timestamp - time_window
            )  # Get the timestamp for DIC window start
            end_timestamp = timestamp  # Get the timestamp for DIC window end

            start_index = find_nearest_timestamp_index(
                ae["hit_time"].values, start_timestamp
            )
            end_index = find_nearest_timestamp_index(
                ae["hit_time"].values, end_timestamp
            )
            d_index.append(end_index - start_index)
            indx_val.append((start_index, end_index))
        d_avg_index = int(sum(d_index) / len(d_index))

        return d_avg_index, indx_val

        # Now, ae_indices_dic_based contains corresponding indices in AE data for each DIC window

    def update_files(self, dic_specimens, ae_specimens):
        self.dic_specimens = dic_specimens
        self.ae_specimens = ae_specimens

    def create_sync_indexes(self, process="train"):
        d_avg_index = []
        indexes_per_specimen = []
        for dic_specimen, ae_specimen in zip(self.dic_specimens, self.ae_specimens):
            dic_files = glob.glob(f"sensors/DIC/{dic_specimen}_*")
            dic_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            # dic_files = dic_files[::self.ignore_steps_images]
            d_avg_indx, indexes_per_image = self.synchronize_data(
                dic_files, ae_specimen
            )
            indexes_per_specimen.append(indexes_per_image)
            d_avg_index.append(d_avg_indx)
        if (
            process == "train"
        ):  # the indices of the train set should be used for the test set as well
            global_avg_index = int(sum(d_avg_index) / len(d_avg_index))
            self.global_avg_index = global_avg_index
        return self.global_avg_index, indexes_per_specimen


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_folders(folders, results_subfolder):
    """
    Create folders and subfolders
    :param folders: List of folder names
    :param results_subfolder: List of subfolder names, for the results folder
    :return: None
    """

    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        else:
            print(f"Folder already exists: {path}")

    # Create folders
    for folder_name in folders:
        folder_path = os.path.join(os.getcwd(), folder_name)
        create_folder(folder_path)

    # Create subfolders
    for folder_name in folders:
        if (
            "scalers" in folder_name
            or "models" in folder_name
            or "results" in folder_name
        ):
            subfolder_names = ["cmaps", "mimic", "fmoc"]
            for subfolder_name in subfolder_names:
                subfolder_path = os.path.join(os.getcwd(), folder_name, subfolder_name)
                create_folder(subfolder_path)
            if "results" in folder_name:
                subsubfolder_names = results_subfolder
                for subfolder_name in subfolder_names:
                    for subsubfolder_name in subsubfolder_names:
                        subsubfolder_path = os.path.join(
                            os.getcwd(), folder_name, subfolder_name, subsubfolder_name
                        )
                        create_folder(subsubfolder_path)
        elif "MIMIC" in folder_name:
            if not os.path.exists(os.path.join(os.getcwd(), folder_name, "data")):
                create_folder(os.path.join(os.getcwd(), folder_name, "data"))
                files = [
                    "ADMISSIONS.csv",
                    "PATIENTS.csv",
                    "D_ITEMS.csv",
                    "LABEVENTS.csv",
                    "D_LABITEMS.csv",
                    "CHARTEVENTS.csv",
                ]
                for file in files:
                    os.rename(
                        os.path.join(os.getcwd(), file),
                        os.path.join(os.getcwd(), folder_name, "data", file),
                    )
        elif "CMAPS" in folder_name:
            file = "train_FD001.txt"
            if not os.path.exists(os.path.join(os.getcwd(), folder_name, file)):
                os.rename(
                    os.path.join(os.getcwd(), file),
                    os.path.join(os.getcwd(), folder_name, file),
                )
        elif "sensors" in folder_name:
            if not os.path.exists(os.path.join(os.getcwd(), "sensors", "ACOUSTIC")):
                os.rename(
                    os.path.join(os.getcwd(), "ACOUSTIC"),
                    os.path.join(os.getcwd(), "sensors", "ACOUSTIC"),
                )
            if not os.path.exists(os.path.join(os.getcwd(), "sensors", "DIC")):
                os.rename(
                    os.path.join(os.getcwd(), "DIC"),
                    os.path.join(os.getcwd(), "sensors", "DIC"),
                )


def seed(seed_number):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed_number)
    th.use_deterministic_algorithms(True)
    np.random.seed(seed_number)
    np.random.RandomState(seed_number)
    random.seed(seed_number)
    th.manual_seed(seed_number)
    th.cuda.manual_seed(seed_number)
    th.cuda.manual_seed_all(seed_number)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.use_deterministic_algorithms(True)
    th.set_num_threads(1)


def seed_worker(worker_id):
    """
    Need to seed again for each worker
    :param worker_id: int, number of parallel workers
    :return: None
    """
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def average_pooling(image_array, target_size):
    # Verify that the input array has the correct dimensions
    if len(image_array.shape) != 2:
        print("Error: Input array should have 2 dimensions.")
        return None

    # Calculate the pool size based on the original and target sizes
    pool_size = (
        image_array.shape[0] // target_size[0],
        image_array.shape[1] // target_size[1],
    )

    # Reshape the array into non-overlapping blocks
    reshaped_array = image_array[
        : target_size[0] * pool_size[0], : target_size[1] * pool_size[1]
    ].reshape(target_size[0], pool_size[0], target_size[1], pool_size[1])

    # Calculate the average value for each block
    downsampled_array = reshaped_array.mean(axis=(1, 3))

    return downsampled_array


def custom_train_test_split(data, targets, time, demo, test_ratio):
    """
    Custom train test split that fits to the dataset format, particularly, custom train validation split
    :param data: np.array of inputs
    :param targets: np.array of targets, i.e. reconstruction of input
    :param time: np.array
    :param demo: np.array
    :param test_ratio: float
    :return: train_data, test_data, train_targets, test_targets, train_time, test_time, train_demo [optional], test_demo [optional]
    """

    # check if data is a tuple
    if isinstance(data, tuple):
        n = data[0].shape[0]
    else:
        n = data.shape[0]
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    if demo is None:
        if isinstance(data, tuple):
            train_data, train_time, train_targets = (
                tuple(dat[train_indices] for dat in data),
                time[train_indices],
                tuple(targ[train_indices] for targ in targets),
            )
            test_data, test_time, test_targets = (
                tuple(dat[test_indices] for dat in data),
                time[test_indices],
                tuple(targ[test_indices] for targ in targets),
            )
        else:
            train_data, train_time, train_targets = (
                data[train_indices],
                time[train_indices],
                targets[train_indices],
            )
            test_data, test_time, test_targets = (
                data[test_indices],
                time[test_indices],
                targets[test_indices],
            )
        return train_data, test_data, train_targets, test_targets, train_time, test_time
    else:
        train_data, train_time, train_targets, train_demo = (
            data[train_indices],
            time[train_indices],
            targets[train_indices],
            demo[train_indices],
        )
        test_data, test_time, test_targets, test_demo = (
            data[test_indices],
            time[test_indices],
            targets[test_indices],
            demo[test_indices],
        )
        return (
            train_data,
            test_data,
            train_targets,
            test_targets,
            train_time,
            test_time,
            train_demo,
            test_demo,
        )


def split_image_dict(x_dict, train=True):
    """
    Split a nested dictionary with keys 'a1', 'a2', ... and subkeys 'images', 'times' into 2 dictionaries with keys 'a1', 'a2', ...
    :param x_dict: nested dict
    :return: 2 simple dicts
    """
    # print('train = ', train, 'L_dict = ', len(x_dict))
    if train:
        dict_name = "a"
    else:
        dict_name = "te_a"
    image_dict = {dict_name + str(i + 1): [] for i in range(len(x_dict))}
    time_dict = {dict_name + str(i + 1): [] for i in range(len(x_dict))}
    for i in range(len(x_dict)):
        image_dict[dict_name + str(i + 1)] = x_dict[dict_name + str(i + 1)]["images"]
        time_dict[dict_name + str(i + 1)] = x_dict[dict_name + str(i + 1)]["time"]
    return image_dict, time_dict


def split_series_dict(x_dict, train=True):
    """
    Split a nested dictionary with keys 'a1', 'a2', ... and subkeys 'Age', 'Gender', ... 'times' into 2 dictionaries with keys 'a1', 'a2', ...
    :param x_dict: nested dict
    :return: 2 simple dicts
    """

    df_columns = [
        "Heart Rate",
        "Arterial BP [Systolic]",
        "Arterial BP [Diastolic]",
        "Respiratory Rate",
        "Skin [Temperature]",
        "SpO2",
        "GCS Total",
    ]
    df_lab_columns = [
        "Anion gap",
        "Bicarbonate",
        "Bilirubin",
        "Creatinine",
        "Chloride",
        "Glucose",
        "Hematocrit",
        "Hemoglobin",
        "Lactate",
        "Platelet",
        "Potassium",
        "PT",
        "Sodium",
        "BUN",
        "WBC",
    ]
    if train:
        dict_name = "a"
    else:
        dict_name = "te_a"

    series_dict = {
        dict_name + str(i + 1): {col: [] for col in df_columns}
        for i in range(len(x_dict))
    }
    demo_dict = {
        dict_name + str(i + 1): {"Age": [], "Gender": []} for i in range(len(x_dict))
    }
    lab_dict = {
        dict_name + str(i + 1): {col: [] for col in df_lab_columns}
        for i in range(len(x_dict))
    }
    for i in range(len(x_dict)):
        for col in x_dict[dict_name + str(i + 1)].keys():
            if col == "Age" or col == "Gender":
                demo_dict[dict_name + str(i + 1)][col].append(
                    x_dict[dict_name + str(i + 1)][col]
                )
            elif col in df_lab_columns:
                lab_dict[dict_name + str(i + 1)][col].append(
                    x_dict[dict_name + str(i + 1)][col]
                )
            else:
                series_dict[dict_name + str(i + 1)][col] = x_dict[
                    dict_name + str(i + 1)
                ][col]
    # merge demo and lab dictionaries
    event_dict = {
        dict_name
        + str(i + 1): {
            **demo_dict[dict_name + str(i + 1)],
            **lab_dict[dict_name + str(i + 1)],
        }
        for i in range(len(x_dict))
    }

    return series_dict, event_dict



def overlapping_windows_comparative_study(
    x_dict,
    x_demo_dict,
    window_length,
    step_size,
    train=True,
    t_avg=0
):

    if train:
        dict_name = "a"
        for i in range(len(x_dict)):
            # keep only the first values
            keep_lengths = ((np.floor(0.1 * len(x_dict[dict_name + str(i + 1)])) - window_length) / step_size) + 1
            for col in x_dict[dict_name + str(i + 1)].keys():
                x_dict[dict_name + str(i + 1)][col] = x_dict[dict_name + str(i + 1)][col][:int(keep_lengths)]
    else:
        dict_name = "te_a"

    dict_keys = list(x_dict[dict_name + "1"].keys())

    time_dict = {dict_name + str(i + 1): [] for i in range(len(x_dict))}

    x = np.empty((0, window_length, len(dict_keys)))
    x_demo = []
    time_feature = []
    tot_len = 0
    for i in range(len(x_dict)):
        L = np.lib.stride_tricks.sliding_window_view(
            x_dict[dict_name + str(i + 1)][dict_keys[0]], window_length, axis=0
        )[::step_size].shape[0]
        tot_len += L
        if x_demo_dict is not None:
            len_dict = len(x_demo_dict[dict_name + str(i + 1)])
            val_list = []
            for col in x_demo_dict[dict_name + str(i + 1)].keys():
                val = (
                    np.repeat(
                        np.array(x_demo_dict[dict_name + str(i + 1)][col][0]), L
                    )
                    .flatten()
                    .reshape(-1, 1)
                )
                val_list.append(val)
                x_demo_dict[dict_name + str(i + 1)][col] = np.concatenate(
                    val, axis=0
                ).flatten()
            val_arr = (
                np.concatenate(val_list, axis=0).flatten().reshape(L, len_dict)
            )
            x_demo.extend(val_arr)

        if x_demo != []:
            x_demo = np.concatenate((x_demo), axis=0).reshape(-1, len_dict)
        if train:
            t_avg = tot_len / (i + 1)
    for i in range(len(x_dict)):
        L = np.lib.stride_tricks.sliding_window_view(
            x_dict[dict_name + str(i + 1)][dict_keys[0]], window_length, axis=0
        )[::step_size].shape[0]
        x1 = np.empty((L, window_length, 0))
        for col in dict_keys:
            x_dict[dict_name + str(i + 1)][
                col
            ] = np.lib.stride_tricks.sliding_window_view(
                x_dict[dict_name + str(i + 1)][col], window_length, axis=0
            )[
                ::step_size
            ]
            x1 = np.append(
                x1, np.expand_dims(x_dict[dict_name + str(i + 1)][col], 2), axis=2
            )

        time_feature.extend(
            list(
                np.linspace(
                    0, t_avg, x_dict[dict_name + str(i + 1)][dict_keys[0]].shape[0]
                )
            )
        )
        time_dict[dict_name + str(i + 1)] = list(
            np.linspace(
                0, t_avg, x_dict[dict_name + str(i + 1)][dict_keys[0]].shape[0]
            )
        )

        x = np.append(x, x1, axis=0)
    return (
        x,
        x_demo,
        np.array(time_feature).squeeze(),
        x_dict,
        time_dict,
        x_demo_dict,
        t_avg
    )














def overlapping_windows(
    technique,
    x_dict,
    x_demo_dict,
    window_length,
    step_size,
    train=True,
    t_avg=0,
    h=None,
    w=None,
):
    """
    Create overlapping windows of length window_length with step size step_size
    :param x_dict: dictionary of input series
    :param x_demo_dict: dictionary of demographic and lab data
    :param window_length: length of each window (int)
    :param step_size: int
    :param train: bool, whether to use train or test data
    :param t_avg: float, average length (time) of the training data
    :param h: int, height of the images if they exist
    :param w: int, width of the images if they exist
    :return: x, x_demo, time_feature, x_dict, time_dict, x_demo_dict, t_avg
    """

    if train:
        dict_name = "a"
    else:
        dict_name = "te_a"

    dict_keys = list(x_dict[dict_name + "1"].keys())

    if not technique == "dic":
        time_dict = {dict_name + str(i + 1): [] for i in range(len(x_dict))}

        x = np.empty((0, window_length, len(dict_keys)))
        x_demo = []
        time_feature = []
        tot_len = 0
        for i in range(len(x_dict)):
            L = np.lib.stride_tricks.sliding_window_view(
                x_dict[dict_name + str(i + 1)][dict_keys[0]], window_length, axis=0
            )[::step_size].shape[0]
            tot_len += L
            if x_demo_dict is not None:
                len_dict = len(x_demo_dict[dict_name + str(i + 1)])
                val_list = []
                for col in x_demo_dict[dict_name + str(i + 1)].keys():
                    val = (
                        np.repeat(
                            np.array(x_demo_dict[dict_name + str(i + 1)][col][0]), L
                        )
                        .flatten()
                        .reshape(-1, 1)
                    )
                    val_list.append(val)
                    x_demo_dict[dict_name + str(i + 1)][col] = np.concatenate(
                        val, axis=0
                    ).flatten()
                val_arr = (
                    np.concatenate(val_list, axis=0).flatten().reshape(L, len_dict)
                )
                x_demo.extend(val_arr)

        if x_demo != []:
            x_demo = np.concatenate((x_demo), axis=0).reshape(-1, len_dict)
        if train:
            t_avg = tot_len / (i + 1)
        for i in range(len(x_dict)):
            L = np.lib.stride_tricks.sliding_window_view(
                x_dict[dict_name + str(i + 1)][dict_keys[0]], window_length, axis=0
            )[::step_size].shape[0]
            x1 = np.empty((L, window_length, 0))
            for col in dict_keys:
                x_dict[dict_name + str(i + 1)][
                    col
                ] = np.lib.stride_tricks.sliding_window_view(
                    x_dict[dict_name + str(i + 1)][col], window_length, axis=0
                )[
                    ::step_size
                ]
                x1 = np.append(
                    x1, np.expand_dims(x_dict[dict_name + str(i + 1)][col], 2), axis=2
                )

            time_feature.extend(
                list(
                    np.linspace(
                        0, t_avg, x_dict[dict_name + str(i + 1)][dict_keys[0]].shape[0]
                    )
                )
            )
            time_dict[dict_name + str(i + 1)] = list(
                np.linspace(
                    0, t_avg, x_dict[dict_name + str(i + 1)][dict_keys[0]].shape[0]
                )
            )
            x = np.append(x, x1, axis=0)
    else:
        x_demo = []
        x, time_feature = np.empty((0, window_length, h, w)), []
        # split the nested dict train_dict with subkeys 'images' , 'times' into two new dicts
        x_dict, time_dict = split_image_dict(x_dict, train=train)
        for i in range(len(x_dict)):
            x_dict[dict_name + str(i + 1)] = (
                np.array(x_dict[dict_name + str(i + 1)])
                .flatten()
                .reshape((-1, window_length, h, w))
            )

            time_dict[dict_name + str(i + 1)] = np.array(
                time_dict[dict_name + str(i + 1)]
            ).squeeze()
            x = np.concatenate(
                (
                    x,
                    np.array(x_dict[dict_name + str(i + 1)])
                    .flatten()
                    .reshape((-1, window_length, h, w)),
                ),
                axis=0,
            )

        for i in range(len(x_dict)):
            time_feature.extend(
                list(np.linspace(0, t_avg, x_dict[dict_name + str(i + 1)].shape[0]))
            )
            time_dict[dict_name + str(i + 1)] = list(
                np.linspace(0, t_avg, x_dict[dict_name + str(i + 1)].shape[0])
            )

        time_feature = np.array(time_feature)
        x = np.expand_dims(x, axis=1)
    return (
        x,
        x_demo,
        np.array(time_feature).squeeze(),
        x_dict,
        time_dict,
        x_demo_dict,
        t_avg,
    )


def interpolate_data(x, max_length):
    """
    Interpolate data from 0 to max_length
    :param x: input series
    :param max_length: float, maximum (or average) length of the series
    :return: x
    """

    x = TimeSeriesResampler(sz=max_length).fit_transform(
        np.expand_dims(x, axis=0), random_state=settings.seed_number
    )
    x = x.squeeze()
    return x


class BuildDataset(Dataset):
    def __init__(self, x, y, technique=None):
        self.x = x
        self.y = y
        if technique == "fmoc":
            self.x1, self.x2 = self.x
            self.y1, self.y2 = self.y
            self.x = [(self.x1[i], self.x2[i]) for i in range(len(self.x1))]
            self.y = [(self.y1[i], self.y2[i]) for i in range(len(self.y1))]

    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


class BuildDataloaders:
    """
    Create dataloaders for training, validation and testing and apply min-max normalization
    """

    def __init__(self, batch, technique):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch = batch
        self.technique = technique

    def create_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.train_dataset = BuildDataset(X_train, y_train, self.technique)
        self.val_dataset = BuildDataset(X_val, y_val, self.technique)
        self.test_dataset = BuildDataset(X_test, y_test, self.technique)

    def preprocess(
        self, x, y, x_test, y_test, x_demo_train, x_demo_test, t_f, t_f_test
    ):
        """
        Preprocess the data and apply min-max normalization
        :param x: np.array of inputs if technique is cmaps or mimic, tuple of np.arrays if technique is fmoc
        :param y: np.array of inputs if technique is cmaps or mimic, tuple of np.arrays if technique is fmoc
        :param x_test: np.array of inputs if technique is cmaps or mimic, tuple of np.arrays if technique is fmoc
        :param y_test: np.array of inputs if technique is cmaps or mimic, tuple of np.arrays if technique is fmoc
        :param x_demo_train: np.array of demographic and lab data, else None
        :param x_demo_test:  np.array of demographic and lab data, else None
        :param t_f: np.array of time feature for training
        :param t_f_test: np.array of time feature for testing

        x_train is not returned as np.array but as a tuple of np.arrays containing time series, time feature,
        and (optionally) demographic data. If technique is fmoc then x_train, x_val, x_test are a tuple containing
        a tuple with acoustic and time feature within, and the DIC data.
        """

        if self.technique == "fmoc":
            test_ratio = 0.1
        else:
            test_ratio = 0.001

        scaler_demo = MinMaxScaler(feature_range=(0, 1))

        if x_demo_train != []:
            (
                x_train,
                x_val,
                y_train,
                y_val,
                t_f_train,
                t_f_val,
                x_demo_train,
                x_demo_val,
            ) = custom_train_test_split(x, y, t_f, x_demo_train, test_ratio=test_ratio)
            assert len(x_train) == len(
                x_demo_train
            ), f"x_train {len(x_train)} and x_demo_train {len(x_demo_train)} should have the same length"
            # normalize the x_demo with min max normalization in range (0,1)
            x_demo_train = scaler_demo.fit_transform(x_demo_train)
            x_demo_val = scaler_demo.transform(x_demo_val)
            x_demo_test = scaler_demo.transform(x_demo_test)
        else:
            (
                x_train,
                x_val,
                y_train,
                y_val,
                t_f_train,
                t_f_val,
            ) = custom_train_test_split(x, y, t_f, None, test_ratio=test_ratio)

        # normalize the t_f_train and t_f_test with min max normalization in range (0,1)
        scaler_t_f = MinMaxScaler(feature_range=(0, 1))
        t_f_train = scaler_t_f.fit_transform(t_f_train.reshape(-1, 1))
        t_f_val = scaler_t_f.transform(t_f_val.reshape(-1, 1))
        t_f_test = scaler_t_f.transform(t_f_test.reshape(-1, 1))

        if self.technique == "fmoc":
            x_train_tuple, x_val_tuple, x_test_tuple = x_train, x_val, x_test
            y_train_tuple, y_val_tuple, y_test_tuple = y_train, y_val, y_test
            assert all(
                isinstance(var, tuple)
                for var in [
                    x_train_tuple,
                    x_val_tuple,
                    x_test_tuple,
                    y_train_tuple,
                    y_val_tuple,
                    y_test_tuple,
                ]
            ), "All variables should be tuples (x_train_tuple, x_val_tuple, x_test_tuple, y_train_tuple, y_val_tuple, y_test_tuple)"
            x_train, _ = x_train_tuple
            x_val, _ = x_val_tuple
            x_test, _ = x_test_tuple
            y_train, _ = y_train_tuple
            y_val, _ = y_val_tuple
            y_test, _ = y_test_tuple

        # Normalize the time series data
        shape_trx, shape_try = x_train.shape, y_train.shape
        shape_valx, shape_valy = x_val.shape, y_val.shape
        shape_tex, shape_tey = x_test.shape, y_test.shape
        '''
        scaler_x, scaler_y = MinMaxScaler(
            feature_range=(-1, 1), copy=False
        ), MinMaxScaler(feature_range=(0, 1), copy=False)
        '''
        scaler_x, scaler_y = StandardScaler(copy=False), StandardScaler(copy=False)
        x_train, x_val, x_test = (
            x_train.reshape(-1, x_train.shape[-1]),
            x_val.reshape(-1, x_val.shape[-1]),
            x_test.reshape(-1, x_test.shape[-1]),
        )
        y_train, y_val, y_test = (
            y_train.reshape(-1, y_train.shape[-1]),
            y_val.reshape(-1, y_val.shape[-1]),
            y_test.reshape(-1, y_test.shape[-1]),
        )
        scaler_x.fit_transform(x_train), scaler_y.fit_transform(y_train)
        scaler_x.transform(x_val), scaler_y.transform(y_val)
        scaler_x.transform(x_test), scaler_y.transform(y_test)
        x_train, x_val, x_test = (
            x_train.reshape(shape_trx),
            x_val.reshape(shape_valx),
            x_test.reshape(shape_tex),
        )
        y_train, y_val, y_test = (
            x_train.reshape(shape_try),
            x_val.reshape(shape_valy),
            x_test.reshape(shape_tey),
        )

        new_x = []
        for i in range(x_train.shape[0]):
            if x_demo_train != []:
                new_x.append((x_train[i], t_f_train[i], x_demo_train[i]))
            else:
                new_x.append((x_train[i], t_f_train[i]))
        x_train = new_x
        new_x = []
        for i in range(x_val.shape[0]):
            if x_demo_train != []:
                new_x.append((x_val[i], t_f_val[i], x_demo_val[i]))
            else:
                new_x.append((x_val[i], t_f_val[i]))
        x_val = new_x
        new_x = []
        for i in range(x_test.shape[0]):
            if x_demo_train != []:
                new_x.append((x_test[i], t_f_test[i], x_demo_test[i]))
            else:
                new_x.append((x_test[i], t_f_test[i]))
        x_test = new_x

        if self.technique == "fmoc":
            x_train = (x_train, x_train_tuple[1])
            x_val = (x_val, x_val_tuple[1])
            x_test = (x_test, x_test_tuple[1])
            y_train = (y_train, y_train_tuple[1])
            y_val = (y_val, y_val_tuple[1])
            y_test = (y_test, y_test_tuple[1])

        # save scalers
        if x_demo_train != []:
            joblib.dump(scaler_demo, f"scalers/{self.technique}/scaler_demo.save")
        joblib.dump(scaler_t_f, f"scalers/{self.technique}/scaler_t_f.save")
        joblib.dump(scaler_x, f"scalers/{self.technique}/scaler_x.save")
        joblib.dump(scaler_y, f"scalers/{self.technique}/scaler_y.save")
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
        )
        return scaler_x, scaler_t_f, scaler_demo

    def create_dataloaders(self):
        self.create_datasets(
            self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test
        )
        g = th.Generator()
        g.manual_seed(settings.seed_number)

        # create dataloaders
        static_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch,
            sampler=th.utils.data.SubsetRandomSampler(
                range(len(self.train_dataset)), generator=g
            ),
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        return static_loader, train_loader, val_loader, test_loader


def target_distribution(x: th.Tensor) -> th.Tensor:
    """
    Compute the target distribution p_ij, given x (q_ij), as discussed in Equation 2 of the Methods section
    This is used for the KL-divergence loss function (see Equation 3 in paper).
    :param x: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (x**2) / th.sum(x, 0)
    return (weight.t() / th.sum(weight, 1)).t()


def time_gradient(out):
    """
    Helper function to be used for calculating the gradients of output with respect to the time feature
    :param out: Encoder's output
    :return: None
    """
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j].backward(retain_graph=True)


class BayesianOptDSMC:
    """
    Class for Bayesian Optimization for hyperparameter tuning of the DSMC model as described in paper
    """

    def __init__(
        self,
        data_loaders_obj,
        t_avg,
        n_inputs,
        n_seq,
        train_dict,
        time_train_dict,
        demo_dict,
        scaler_x,
        scaler_t_f,
        scaler_demo,
        n_clusters,
        batch,
        technique,
        device,
        n_inputs_acoustic=None,
        n_inputs_dic=None,
        n_seq_acoustic=None,
        n_seq_dic=None,
        H=None,
        W=None,
        lstm_method=False,
    ):
        self.data_loaders_obj = data_loaders_obj
        self.t_avg = t_avg
        self.n_inputs = n_inputs
        self.n_inputs_acoustic = n_inputs_acoustic
        self.n_inputs_dic = n_inputs_dic
        self.n_seq = n_seq
        self.n_seq_acoustic = n_seq_acoustic
        self.n_seq_dic = n_seq_dic
        self.train_dict = train_dict
        self.time_train_dict = time_train_dict
        self.demo_dict = demo_dict
        self.scaler_x = scaler_x
        self.scaler_t_f = scaler_t_f
        self.scaler_demo = scaler_demo
        self.n_clusters = n_clusters
        self.batch = batch
        self.technique = technique
        self.device = device
        self.H = H
        self.W = W
        self.lstm_method = lstm_method

        self.pbounds = {
            "lr_ae": (1e-4, 5e-3),
            "alpha": (0.7, 2.0),
            "epochs_ae": (50, 200),
            "encoder_dim": (3, 10),
            "hidden_dim": (32, 128),
            "dr_rate": (0.1, 0.3),
            "lr_dc": (5e-5, 1e-3),
            "beta": (1e-2, 5.0),
            "epochs_dc": (10, 30),
        }
        self.best_model = []
        self.optimized_models = self.run_bayes()

    def bayes_train(
        self,
        lr_ae,
        alpha,
        epochs_ae,
        encoder_dim,
        hidden_dim,
        dr_rate,
        lr_dc,
        beta,
        epochs_dc,
    ):
        """
        Train the BO algorithm with the custom objevtive function as discussed in Methods section  of the paper
        """

        EPOCHS_AE = int(epochs_ae)
        ENCODER_DIM = int(encoder_dim)
        HIDDEN_DIM = int(hidden_dim)
        EPOCHS_DC = int(epochs_dc)
        # round to the nth Decimal
        AE_LR = round(lr_ae, 4)
        ALPHA = round(alpha, 3)
        DR_RATE = round(dr_rate, 1)
        DC_LR = round(lr_dc, 4)
        BETA = round(beta, 3)
        (
            static_loader,
            train_loader,
            val_loader,
            _,
        ) = self.data_loaders_obj.create_dataloaders()
        if "cmaps" in self.technique:
            self.model_ae = models.AE(
                self.n_inputs, self.n_seq, ENCODER_DIM, HIDDEN_DIM, DR_RATE
            ).to(self.device)
        elif "mimic" in self.technique:
            self.model_ae = models.AE(
                self.n_inputs,
                self.n_seq,
                ENCODER_DIM,
                HIDDEN_DIM,
                DR_RATE,
                use_demo=True,
            ).to(self.device)
        else:
            self.model_ae = models.AE_ACOUSTIC_DIC(
                self.n_inputs_acoustic,
                self.n_inputs_dic,
                self.n_seq_acoustic,
                self.n_seq_dic,
                self.H,
                self.W,
                ENCODER_DIM,
                HIDDEN_DIM,
                DR_RATE
            ).to(self.device)

        # Train the autoencoder
        self.model_ae = run_models.train_ae(
            self.model_ae,
            self.technique,
            train_loader,
            val_loader,
            AE_LR,
            ALPHA,
            self.device,
            EPOCHS_AE,
            bayes_opt=True,
        )

        self.model_dc = models.DC(
            self.n_clusters, ENCODER_DIM, self.model_ae, self.device
        )
        # Train the cluster model
        self.model_dc = run_models.train_cluster(
            static_loader,
            train_loader,
            self.model_dc,
            self.scaler_t_f,
            self.technique,
            ALPHA,
            BETA,
            EPOCHS_DC,
            DC_LR,
            self.device,
            bayes_opt=True,
        )
        self.model_dc.eval()
        dict_letter = "a"
        data_dict = self.train_dict
        time_dict = self.time_train_dict
        demo_dict = self.demo_dict

        if isinstance(data_dict, tuple):
            negative_steps = np.zeros((len(data_dict[0])))
            step_wrong = np.zeros((len(data_dict[0])))
            last_cluster_len = np.zeros((len(data_dict[0])))
        else:
            negative_steps = np.zeros((len(data_dict)))
            step_wrong = np.zeros((len(data_dict)))
            last_cluster_len = np.zeros((len(data_dict)))

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
                trajectory_acoustic = self.scaler_x.transform(trajectory_acoustic)
                trajectory_acoustic = trajectory_acoustic.reshape(shape_tr)
                trajectory_dic = np.array(list(data_dict[1][dict_letter + str(i + 1)]))
                # add one dimension to position 1
                trajectory_dic = np.expand_dims(trajectory_dic, axis=1)
                trajectory = (trajectory_acoustic, trajectory_dic)
                assert (
                    trajectory_acoustic.shape[0] == trajectory_dic.shape[0]
                ), "The number of windows of acoustic emission and DIC images should be the same"
            else:
                trajectory = np.array(
                    list(data_dict[dict_letter + str(i + 1)].values())
                )
                trajectory = np.transpose(trajectory, (1, 2, 0))
                # Normalize data
                shape_tr = trajectory.shape
                trajectory = trajectory.reshape(-1, trajectory.shape[-1])
                trajectory = self.scaler_x.transform(trajectory)
                trajectory = trajectory.reshape(shape_tr)
            time = np.array(time_dict[dict_letter + str(i + 1)])
            time = self.scaler_t_f.transform(time.reshape(-1, 1))
            if demo_dict is not None:
                demo = np.array(list(demo_dict[dict_letter + str(i + 1)].values()))
                demo = np.transpose(demo, (1, 0))
                demo = self.scaler_demo.transform(demo)
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
                        th.from_numpy(trajectory[0][j])
                        .float()
                        .unsqueeze(0)
                        .to(self.device),
                        th.from_numpy(trajectory[1][j])
                        .float()
                        .unsqueeze(0)
                        .to(self.device),
                    )
                else:
                    t_series = (
                        th.from_numpy(trajectory[j])
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                t = th.from_numpy(time[j]).float().unsqueeze(0).to(self.device)
                d = (
                    th.from_numpy(demo[j]).float().unsqueeze(0).to(self.device)
                    if demo is not None
                    else None
                )
                output, _, _, encoded, _, _ = self.model_dc(t_series, t, d)
                times.append(t.detach().cpu())
                features.append(
                    output.detach().cpu()
                )  # move to CPU to prevent out of memory on the GPU
                embds.append(encoded.detach().cpu())
            labels, times, features = (
                th.cat(features).max(1)[1],
                th.cat(times),
                th.cat(features),
            )
            labels = labels.numpy()
            labels = np.insert(labels, 0, 0)
            labels = np.append(labels, self.n_clusters)
            df = pd.DataFrame({f"sp{i + 1}": labels})

            # count the maximum times in a row that df equals to the last cluster
            max_value = df[f"sp{i + 1}"].max()
            last_cluster_len[i] = max(
                (df[f"sp{i + 1}"] == max_value)
                .astype(int)
                .groupby((df[f"sp{i + 1}"] != max_value).astype(int).cumsum())
                .cumsum()
            )
            if last_cluster_len[i] > self.t_avg:
                last_cluster_len[i] -= self.t_avg
            else:
                last_cluster_len[i] = 0

            # keep only the first occurence of each value if this value occurs sequentially before a new value
            df = df[df[f"sp{i + 1}"].shift(1) != df[f"sp{i + 1}"]]
            # add those unique labels to the corresponding column of tot_cluster_score
            cluster_steps = np.diff(df[f"sp{i + 1}"].values)
            negative_steps[i] = np.sum(
                cluster_steps < 0
            )  # encourage some negative steps
            absolute_cluster_steps = abs(cluster_steps)
            step_wrong[i] = np.sum(
                absolute_cluster_steps[absolute_cluster_steps > 1]
            )  # less large jumps

        self.best_model.append((self.model_ae, self.model_dc))

        negative_steps = np.mean(negative_steps)
        step_wrong = np.mean(step_wrong)
        last_cluster_len = np.mean(last_cluster_len)
        return 0.6 * negative_steps - (step_wrong + last_cluster_len)

    def run_bayes(self):
        """
        Run bayesian optimization to find the best hyperparameters
        """

        print(
            "\nBayesian Optimization for hyperparameter tuning of the DSMC model."
            "This will run for 100 iterations, it will take a long, long time.\n"
        )

        optim = BayesianOptimization(
            f=self.bayes_train,
            pbounds=self.pbounds,
            random_state=settings.seed_number,
            verbose=2,
        )
        optim.maximize(init_points=5, n_iter=95)
        print(optim.max)
        print(optim.res)
        best_results_index = np.argmax([res["target"] for res in optim.res])
        lr_ae = np.around(optim.res[best_results_index]["params"]["lr_ae"], 4)
        alpha = np.around(optim.res[best_results_index]["params"]["alpha"], 3)
        epochs_ae = int(optim.res[best_results_index]["params"]["epochs_ae"])
        encoder_dim = int(optim.res[best_results_index]["params"]["encoder_dim"])
        hidden_dim = int(optim.res[best_results_index]["params"]["hidden_dim"])
        dr_rate = np.around(optim.res[best_results_index]["params"]["dr_rate"], 1)
        lr_dc = np.around(optim.res[best_results_index]["params"]["lr_dc"], 4)
        beta = np.around(optim.res[best_results_index]["params"]["beta"], 3)
        epochs_dc = int(optim.res[best_results_index]["params"]["epochs_dc"])
        print(
            f"\nbest hyperparameters: lr_ae: {lr_ae}, alpha: {alpha}, epochs_ae: {epochs_ae}, encoder_dim: {encoder_dim}, hidden_dim: {hidden_dim}, dr_rate: {dr_rate}, lr_dc: {lr_dc}, beta: {beta}, epochs_dc: {epochs_dc}"
        )
        return (
            lr_ae,
            alpha,
            epochs_ae,
            encoder_dim,
            hidden_dim,
            dr_rate,
            lr_dc,
            beta,
            epochs_dc,
        )



