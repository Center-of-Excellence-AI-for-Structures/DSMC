import glob
import os
import pandas as pd
import json
import numpy as np
from PIL import Image
import vallenae as vae
import matplotlib.pyplot as plt
from utils import (
    NumpyArrayEncoder,
    SynchronizeImageAE,
    interpolate_data,
    average_pooling,
)


def txt_to_csv(filename, file_save, sorted_file_save):
    df = pd.read_csv(filename, delimiter=" ")
    tot_len = 0
    for i in range(100):
        # j goes until the number of i that exist in the first column of df
        L = df[df.columns[0]].value_counts(sort=False).iloc[i]
        df_traj = df[tot_len : tot_len + L]
        df_traj = df_traj.drop(df_traj.columns[[0, 1, 26, 27]], axis=1)
        df_traj.to_csv(file_save + "_" + str(i) + ".csv", index=False)
        tot_len += L
    # sort the csv files based on the length of the dataframe
    lengths = []
    csv_files = glob.glob(file_save + "*.csv")
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        lengths.append(len(df))
    csv_files = [x for _, x in sorted(zip(lengths, csv_files))]
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None)
        df.to_csv(sorted_file_save + "_" + str(i) + ".csv", index=False)


def load_mimic_data(path, n_test_files=10):
    df_columns = [
        "Age",
        "Gender",
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
    # load csv files in the order of the number index with glob
    csv_files = glob.glob(os.path.join(path + "vital_signs/", "*.csv"))
    csv_files = sorted(csv_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    train_dict = {
        f"a{i + 1}": {col: [] for col in df_columns + df_lab_columns}
        for i in range(len(csv_files) - n_test_files)
    }
    test_dict = {
        f"te_a{i + 1}": {col: [] for col in df_columns + df_lab_columns}
        for i in range(n_test_files)
    }
    # from the len(csv_files) keep 10 files for testing that are chosen with a corresponding step size across the
    csv_files_test = csv_files[:: int(len(csv_files) / n_test_files)][1:]
    lab_arr = pd.read_csv(path + "lab.csv", header=0)
    csv_files_train = [f for f in csv_files if f not in csv_files_test]
    demographic_arr = pd.read_csv(path + "demographic.csv", header=0)
    demographic_arr = demographic_arr.drop(["Mortality"], axis=1)

    demographic_arr_test = demographic_arr[:: int(len(demographic_arr) / n_test_files)][
        1:
    ]
    demographic_arr_train = demographic_arr[
        ~demographic_arr.index.isin(demographic_arr_test.index)
    ]

    lab_arr_test = lab_arr[:: int(len(lab_arr) / n_test_files)][1:]
    lab_arr_train = lab_arr[~lab_arr.index.isin(lab_arr_test.index)]

    for process in ["train", "test"]:
        max_lengths = []
        if process == "train":
            csv_files = csv_files_train
            demographic_arr = demographic_arr_train.copy()
            lab_arr = lab_arr_train.copy()
        else:
            csv_files = csv_files_test
            demographic_arr = demographic_arr_test.copy()
            lab_arr = lab_arr_test.copy()
        for i, f in enumerate(csv_files):
            # read the csv file
            arr = pd.read_csv(f, header=0)
            max_lengths.append(len(arr))
            for col in df_columns + df_lab_columns:
                if process == "train":
                    if col == "Age" or col == "Gender":
                        train_dict[f"a{i + 1}"][col].append(
                            demographic_arr[col].values[i]
                        )
                    elif col in lab_arr.columns:
                        train_dict[f"a{i + 1}"][col].append(lab_arr[col].values[i])
                    else:
                        train_dict[f"a{i + 1}"][col].extend(arr[col].values)
                else:
                    if col == "Age" or col == "Gender":
                        test_dict[f"te_a{i + 1}"][col].append(
                            demographic_arr[col].values[i]
                        )
                    elif col in lab_arr.columns:
                        test_dict[f"te_a{i + 1}"][col].append(lab_arr[col].values[i])
                    else:
                        test_dict[f"te_a{i + 1}"][col].extend(arr[col].values)
        events = np.ones(len(max_lengths) + 1)
        events[0] = 0
        time = np.array(max_lengths)
        time = np.insert(time, 0, 0)
        # create a dataframe with the number of events at each time step
        df_events = pd.DataFrame(events, columns=["events"])
        df_events["time"] = time
        # save the dataframe into a csv file
        df_events.to_csv(f"events/{process}_mimic_events.csv", index=False)
    # save the dictionaries into a json file
    with open("train_mimic.json", "w") as fp:
        json.dump(train_dict, fp, cls=NumpyArrayEncoder)
    with open("test_mimic.json", "w") as fp:
        json.dump(test_dict, fp, cls=NumpyArrayEncoder)


def load_cmaps_data(txt_file, n_test_files=10):
    # create a folder inside CMAPS named original and another one named sorted
    path1 = "CMAPS/original"
    path2 = "CMAPS/sorted"
    os.makedirs(path2, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    txt_to_csv(txt_file, path1 + "/sp", path2 + "/sp")
    path = path2
    csv_files = glob.glob(os.path.join(path + "/", "*.csv"))
    train_dict = {
        f"a{i + 1}": {col: [] for col in [str(name) for name in range(24)]}
        for i in range(len(csv_files) - n_test_files)
    }
    test_dict = {
        f"te_a{i + 1}": {col: [] for col in [str(name) for name in range(24)]}
        for i in range(n_test_files)
    }
    csv_files = sorted(csv_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    csv_files_test = csv_files[:: int(len(csv_files) / n_test_files)]
    csv_files_train = [f for f in csv_files if f not in csv_files_test]
    processes = ["train", "test"]

    for process in processes:
        max_lengths = []
        if process == "train":
            csv_files = csv_files_train
        else:
            csv_files = csv_files_test
        for i, f in enumerate(csv_files):
            # read the csv file but only specific columns
            #arr = pd.read_csv(f, header=None, usecols=[4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23])
            arr = pd.read_csv(f, header=0)

            max_lengths.append(len(arr))
            for j, col in enumerate([str(name) for name in range(24)]):
                if process == "train":
                    train_dict[f"a{i + 1}"][col].extend(arr.values[:, j])
                else:
                    test_dict[f"te_a{i + 1}"][col].extend(arr.values[:, j])
        events = np.ones(len(max_lengths) + 1)
        events[0] = 0
        time = np.array(max_lengths)
        time = np.insert(time, 0, 0)
        # create a dataframe with the number of events at each time step
        df_events = pd.DataFrame(events, columns=["events"])
        df_events["time"] = time
        # save the dataframe into a csv file
        df_events.to_csv(f"events/{process}_cmaps_events.csv", index=False)

    # save the dictionaries into a json file
    with open("train_cmaps_fd001.json", "w") as fp:
        json.dump(train_dict, fp)
    with open("test_cmaps_fd001.json", "w") as fp:
        json.dump(test_dict, fp)


def prepare_image_data(
    i,
    specimens,
    path,
    window_length,
    step_size,
    normalize_value_dataset,
    target_size,
):
    files = [
        filename
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename)) and specimens[i] in filename
    ]
    # sort files by number in format specimens[i]_0_number_...
    files = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    # convert tiff to numpy
    image_data = [np.array(Image.open(path + "/" + filename)) for filename in files]
    # downsample images
    image_data = [average_pooling(image, target_size) for image in image_data]
    # normalize images to 0, 1 substract the minimum value and divide by the maximum value
    image_data = [image / normalize_value_dataset for image in image_data]
    # convert each numpy array to list
    image_data = [image.tolist() for image in image_data]
    # create sequence of images with window size and step size
    image_data = [
        image_data[i : i + window_length]
        for i in range(0, len(image_data) - window_length + 1, step_size)
    ]
    return image_data


def load_dic_data(
    specimens_sorted, path, target_size, window_length=5, step_size=2, n_test_files=2
):
    """
    load and convert image data to tensor
    """

    print("\nCreating training/testing json files for image data...")

    # Data related parameters
    normalize_value_dataset = 255

    # Load Image data with PIL with .tiff extension
    train_dict = {
        f"a{i + 1}": {"images": [], "time": []}
        for i in range(len(specimens_sorted) - n_test_files)
    }  # ignore the 1st specimen (A003)
    test_dict = {
        f"te_a{i + 1}": {"images": [], "time": []} for i in range(n_test_files)
    }

    max_len = 0
    L_list = []
    for i in range(len(specimens_sorted)):
        image_data = prepare_image_data(
            i,
            specimens_sorted,
            path,
            window_length,
            step_size,
            normalize_value_dataset,
            target_size,
        )
        L = len(image_data)
        L_list.append(L)
        if max_len < L:
            max_len = L

    # sort the specimens list according to the length of the trajectories
    # specimens = [x for _, x in sorted(zip(L_list, specimens))]
    specimens = specimens_sorted
    # from the specimens keep 2 files for testing: 1 from the average length and 1 from the right outliers
    test_specimens = [specimens[3], specimens[-2]]
    train_specimens = [f for f in specimens if f not in test_specimens]

    for process in ["train", "test"]:
        if process == "train":
            specimens = train_specimens
        else:
            specimens = test_specimens
        for i in range(len(specimens)):
            image_data = prepare_image_data(
                i,
                specimens,
                path,
                window_length,
                step_size,
                normalize_value_dataset,
                target_size,
            )
            times = list(np.linspace(0, max_len, len(image_data)))

            if process == "train":
                train_dict[f"a{i + 1}"]["images"].extend(image_data)
                train_dict[f"a{i + 1}"]["time"].extend(times)
            else:  # these specimens are used for testing (not the smallest, not the biggest ones)
                test_dict[f"te_a{i + 1}"]["images"].extend(image_data)
                test_dict[f"te_a{i + 1}"]["time"].extend(times)

    # save the dictionaries into a json file
    with open("train_dic.json", "w") as fp:
        json.dump(train_dict, fp)
    with open("test_dic.json", "w") as fp:
        json.dump(test_dict, fp)


def load_acoustic_data(path, rows_to_skip=1, images_to_skip=2, n_test_files=2):
    df_columns = [
        "hit_time",
        "channel",
        "param_id",
        "amplitude",
        "duration",
        "energy",
        "rms",
        "threshold",
        "rise_time",
        "signal_strength",
        "counts",
        "marker_time",
        "set_type",
        "data",
        "number",
        "param_time",
        "param_id",
        "pctd",
        "pcta",
        "pa0",
        "pa1",
    ]

    df_to_drop = [
        "rms",
        "signal_strength",
        "threshold",
        "set_type",
        "data",
        "number",
        "param_id",
        "pctd",
        "pcta",
        "pa0",
        "pa1",
        "marker_time",
        "param_time",
    ]

    # hold the remaining column names
    df_to_keep_with_channel = [col for col in df_columns if col not in df_to_drop]
    df_to_keep = [col for col in df_to_keep_with_channel if col != "channel"]
    use_cols = df_to_keep_with_channel

    # load csv files
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    # sort files according to their size
    csv_files = sorted(csv_files, key=lambda x: os.path.getsize(x))

    # create a dictionary to hold the data with each column to be an empty list
    train_dict = {
        f"a{i + 1}": {col: [] for col in df_to_keep}
        for i in range(len(csv_files) - n_test_files)
    }
    test_dict = {
        f"te_a{i + 1}": {col: [] for col in df_to_keep} for i in range(n_test_files)
    }

    # from the specimens keep 2 files for testing: 1 from the average length and 1 from the right outliers
    csv_files_test = [csv_files[3], csv_files[-2]]
    csv_files_train = [f for f in csv_files if f not in csv_files_test]

    print("\nCreating training/testing json files for acoustic emission data...")

    for process in ["train", "test"]:
        if process == "train":
            csv_files_use = csv_files_train
        else:
            csv_files_use = csv_files_test
        ae_specimens = []
        dic_specimens = []
        for i, f in enumerate(csv_files_use):
            specimen = f.split(".")[0][-6:]
            # read the csv file and skip every 4 rows
            arr = pd.read_csv(
                f, header=0, usecols=use_cols, skiprows=lambda x: x % rows_to_skip != 0
            )

            # keep rows that correspond to column with channel = 2
            arr = arr[arr["channel"] == 1]
            arr = arr.drop(columns=["channel"])

            # drop nans
            arr = arr.dropna()

            # convert μV to db
            arr["amplitude"] = 20 * np.log10(arr["amplitude"].values / 1e-6)

            ae_specimens.append(arr)
            dic_specimens.append(specimen)

        # Synchronize acoustic emission with DIC
        if process == "train":
            sync = SynchronizeImageAE(
                dic_specimens, ae_specimens, ignore_steps_images=images_to_skip
            )
            (
                ae_window_length_per_image,
                indexes_per_specimen,
            ) = sync.create_sync_indexes()
        else:
            sync.update_files(dic_specimens, ae_specimens)
            (
                _,
                indexes_per_specimen,
            ) = sync.create_sync_indexes(process="test")

        for i, arr in enumerate(ae_specimens):
            assert len(ae_specimens) == len(indexes_per_specimen), (
                "ae_specimens and indexes_per_specimen should have the same length,"
                "but they have lengths {} and {}".format(
                    len(ae_specimens), len(indexes_per_specimen)
                )
            )
            # for each representative image, keep only the rows that correspond to the indexes for the AE
            full_arr = pd.DataFrame()
            for j in range(len(indexes_per_specimen[i])):
                indexes_to_use = list(
                    range(indexes_per_specimen[i][j][0], indexes_per_specimen[i][j][1])
                )
                arr_for_inter = arr.iloc[indexes_to_use]
                arr_val = arr_for_inter.values
                arr_val = interpolate_data(arr_val, ae_window_length_per_image)
                arr_for_inter = arr_val.copy()
                del arr_val
                full_arr = full_arr.append(
                    pd.DataFrame(arr_for_inter, columns=arr.columns)
                )
            arr = full_arr.copy()
            del full_arr
            for col in df_to_keep:
                if process == "train":
                    train_dict[f"a{i + 1}"][col].extend(arr[col].values)
                else:
                    test_dict[f"te_a{i + 1}"][col].extend(arr[col].values)

    # save the dictionaries into a json file
    with open("train_acoustic.json", "w") as fp:
        json.dump(train_dict, fp)
    with open("test_acoustic.json", "w") as fp:
        json.dump(test_dict, fp)

    csv_files = [f.split("\\")[-1].split(".")[0] for f in csv_files]

    return ae_window_length_per_image, csv_files


class Pridb:
    """A class for everything related to pridb/csv files"""

    def __init__(self, folder_name, save_folder):
        self.folder_name = folder_name
        self.save_folder = save_folder
        self.hits = None
        self.abs_start_time = None

        # Make a list of all pridb files
        files = glob.glob(self.folder_name + "/*.pridb")
        self.pridb_files = files
        self.pridb_read_hits()

    def pridb_read_hits(self):
        """Function to retrieve the hits from the pridb file"""
        for i, file in enumerate(self.pridb_files):
            print("\nReading file {} out of {}".format(i + 1, len(self.pridb_files)))
            pridb = vae.io.PriDatabase(file)
            hits = pridb.read_hits()
            hits = hits.rename(columns={"time": "hit_time"})
            markers = pridb.read_markers()
            markers = markers.rename(columns={"time": "marker_time"})
            param = pridb.read_parametric()
            param = param.rename(columns={"time": "param_time"})
            hits, markers, param = (
                pd.DataFrame(hits),
                pd.DataFrame(markers),
                pd.DataFrame(param),
            )
            df = pd.concat([hits, markers, param], axis=1)
            df.to_csv(file.split(".")[0] + ".csv", index=False)
            # delete the pridb file
            pridb.close()
            os.remove(file)
