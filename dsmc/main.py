import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
import fnmatch
import argparse
import settings
import json
import hyperparameters
import matplotlib.pyplot as plt
import torch as th
import utils
from utils import str2bool
import run_models, models, read_files
from mimic_data import create_clinical_data


def run_process():
    """
    Run the entire process
    """

    if args.mimic:
        TECHNIQUE = "mimic"
    else:
        TECHNIQUE = "cmaps"
    if args.fmoc:
        TECHNIQUE = "fmoc"

    H, W = None, None

    SAVE = args.save
    BAYESIAN_OPT = args.bayesian_opt
    ENABLE_VISUALS = args.enable_visuals
    pretrained = args.pretrained
    pretrained_ae = pretrained
    pretrained_dc = pretrained

    if TECHNIQUE == "cmaps":
        technique_folder = "CMAPS"
    elif TECHNIQUE == "mimic":
        technique_folder = "MIMIC"
    else:
        technique_folder = "sensors"

    folders = [
        technique_folder,
        "events",
        "models",
        "results",
        "hyperparameters",
        "scalers",
    ]
    results_subfolders = [
        "cluster_embds",
        "clusters",
        "figs",
        "loss",
        "prognostics",
        "time_grads",
        "z_space",
    ]

    utils.create_folders(folders, results_subfolders)

    # control randomness for reproducibility
    settings.init()
    utils.seed(settings.seed_number)

    # load from json

    if TECHNIQUE == "cmaps":
        train_file = "train_cmaps_fd001.json"
        test_file = "test_cmaps_fd001.json"
        # save cmaps data to json if not already
        if not all(
            [
                os.path.exists(os.path.join(os.getcwd(), train_file)),
                os.path.exists(os.path.join(os.getcwd(), test_file)),
            ]
        ):
            txt_file = "CMAPS/train_FD001.txt"
            read_files.load_cmaps_data(txt_file)
        hyperparameters.save_hypers()
        # Load the hyperparameters from the JSON file
        with open("hyperparameters/hyper_cmaps.json", "r") as f:
            h = json.load(f)
    elif TECHNIQUE == "mimic":
        train_file = "train_mimic.json"
        test_file = "test_mimic.json"
        # save mimic data to json if not already
        if not all(
            [
                os.path.exists(os.path.join(os.getcwd(), train_file)),
                os.path.exists(os.path.join(os.getcwd(), test_file)),
            ]
        ):
            file_admission = "MIMIC/data/ADMISSIONS.csv"
            file_patient = "MIMIC/data/PATIENTS.csv"
            file_chart_items = "MIMIC/data/D_ITEMS.csv"
            file_chart_events = "MIMIC/data/CHARTEVENTS.csv"
            folder_save = "MIMIC/"
            print(
                "\n Creating the MIMIC-III dataset, this may take approx. 4-5 minutes..."
            )
            create_clinical_data(
                file_admission,
                file_patient,
                file_chart_items,
                file_chart_events,
                folder_save,
            )
            read_files.load_mimic_data(folder_save)
        hyperparameters.save_hypers()
        # Load the hyperparameters from the JSON file
        with open("hyperparameters/hyper_mimic.json", "r") as f:
            h = json.load(f)
    else:
        hyperparameters.save_hypers()
        # Load the hyperparameters from the JSON file
        with open("hyperparameters/hyper_fmoc.json", "r") as f:
            h = json.load(f)

        assert h["step_size_dic"] <= 5, "# max images to ignore should be 5"
        assert (
            h["window_length_dic"] > h["step_size_dic"]
        ), "Window length must be larger than the step size"

        train_file_acoustic = "train_acoustic.json"
        test_file_acoustic = "test_acoustic.json"
        # save acoustic data to json if not already
        file = "sensors/ACOUSTIC"
        # check if .pridb files exist in the folder even though some csv files exist
        files_in_folder = os.listdir(file)
        if fnmatch.filter(
            files_in_folder, "*.pridb"
        ):  # check if any .pridb file exists, otherwise we already converted them to csv files
            print("\nReading and converting .pridb files to .csv files...")
            read_files.Pridb(file, file)

        file_window_length = "window_length.txt"
        if not os.path.exists(os.path.join(os.getcwd(), file_window_length)) or not all(
            [
                os.path.exists(os.path.join(os.getcwd(), train_file_acoustic)),
                os.path.exists(os.path.join(os.getcwd(), test_file_acoustic)),
            ]
        ):
            WINDOW_LENGTH_ACOUSTIC, specimens_sorted = read_files.load_acoustic_data(
                file, rows_to_skip=10, images_to_skip=h["step_size_dic"]
            )
            with open(file_window_length, "w") as f:
                f.write(str(WINDOW_LENGTH_ACOUSTIC))
        else:
            WINDOW_LENGTH_ACOUSTIC = int(open(file_window_length, "r").read())
        train_file_dic = "train_dic.json"
        test_file_dic = "test_dic.json"
        # save dic data to json if not already
        if not all(
            [
                os.path.exists(os.path.join(os.getcwd(), train_file_dic)),
                os.path.exists(os.path.join(os.getcwd(), test_file_dic)),
            ]
        ):
            read_files.load_dic_data(
                specimens_sorted,
                "sensors/DIC/",
                (h["image_height"], h["image_width"]),
                step_size=h["step_size_dic"],
                window_length=h["window_length_dic"],
            )

    BATCH = h["batch_size"]
    EPOCHS_AE = h["epochs_ae"]
    EPOCHS_DC = h["epochs_dc"]
    ENCODER_DIM = h["encoder_dim"]
    HIDDEN_DIM = h["hidden_dim"]
    DR_RATE = h["dr_rate"]
    AE_LR = h["lr_ae"]
    DC_LR = h["lr_dc"]
    n_clusters = h["clusters"]
    ALPHA = h["alpha"]
    BETA = h["beta"]

    print("\nLoading json files...")

    if not TECHNIQUE == "fmoc":
        WINDOW_LENGTH = h["window_length"]
        STEP_SIZE = h["step_size"]
        assert (
            WINDOW_LENGTH % STEP_SIZE == 0
        ), "Window length must be a multiple of step size"

        with open(train_file, "r") as fp:
            train_dict = json.load(fp)

        # Test dictionary with keys: 'te_a1', 'te_a1', ...
        with open(test_file, "r") as fp:
            test_dict = json.load(fp)
    else:
        WINDOW_LENGTH_DIC = h["window_length_dic"]
        STEP_SIZE_DIC = h["step_size_dic"]
        STEP_SIZE_ACOUSTIC = WINDOW_LENGTH_ACOUSTIC * STEP_SIZE_DIC
        WINDOW_LENGTH_ACOUSTIC *= WINDOW_LENGTH_DIC

        assert (
            WINDOW_LENGTH_ACOUSTIC % STEP_SIZE_ACOUSTIC == 0
        ), "Window length must be a multiple of step size, but is {} and {} respectively".format(
            WINDOW_LENGTH_ACOUSTIC, STEP_SIZE_ACOUSTIC
        )
        assert (
            WINDOW_LENGTH_DIC % STEP_SIZE_DIC == 0
        ), "Window length must be a multiple of step size"

        with open(train_file_acoustic, "r") as fp:
            train_dict_acoustic = json.load(fp)

        with open(train_file_dic, "r") as fp:
            train_dict_dic = json.load(fp)

        with open(test_file_acoustic, "r") as fp:
            test_dict_acoustic = json.load(fp)

        with open(test_file_dic, "r") as fp:
            test_dict_dic = json.load(fp)

        H = h["image_height"]
        W = h["image_width"]

    # Build and pre-train the AE
    if "mimic" in TECHNIQUE:
        train_dict, train_demo_dict = utils.split_series_dict(train_dict)
        test_dict, test_demo_dict = utils.split_series_dict(test_dict, train=False)
    else:
        train_demo_dict, test_demo_dict = None, None

    print("\nApplying overlapping windows...")

    if not TECHNIQUE == "fmoc":
        (
            X_train,
            X_demo_train,
            time_feature_train,
            train_dict,
            time_train_dict,
            train_demo_dict,
            t_avg,
        ) = utils.overlapping_windows(
            TECHNIQUE,
            train_dict,
            train_demo_dict,
            window_length=WINDOW_LENGTH,
            step_size=STEP_SIZE,
            train=True,
        )
        (
            X_test,
            X_demo_test,
            time_feature_test,
            test_dict,
            time_test_dict,
            test_demo_dict,
            _,
        ) = utils.overlapping_windows(
            TECHNIQUE,
            test_dict,
            test_demo_dict,
            window_length=WINDOW_LENGTH,
            step_size=STEP_SIZE,
            train=False,
            t_avg=t_avg,
        )
    else:
        (
            X_train_acoustic,
            X_demo_train,
            time_feature_train,
            train_dict_acoustic,
            time_train_dict,
            train_demo_dict,
            t_avg,
        ) = utils.overlapping_windows(
            "acoustic",
            train_dict_acoustic,
            train_demo_dict,
            window_length=WINDOW_LENGTH_ACOUSTIC,
            step_size=STEP_SIZE_ACOUSTIC,
            train=True,
            h=H,
            w=W,
        )
        (
            X_test_acoustic,
            X_demo_test,
            time_feature_test,
            test_dict_acoustic,
            time_test_dict,
            test_demo_dict,
            _,
        ) = utils.overlapping_windows(
            "acoustic",
            test_dict_acoustic,
            test_demo_dict,
            window_length=WINDOW_LENGTH_ACOUSTIC,
            step_size=STEP_SIZE_ACOUSTIC,
            train=False,
            t_avg=t_avg,
        )
        (
            X_train_dic,
            X_demo_train,
            time_feature_train,
            train_dict_dic,
            time_train_dict,
            train_demo_dict,
            _,
        ) = utils.overlapping_windows(
            "dic",
            train_dict_dic,
            train_demo_dict,
            window_length=WINDOW_LENGTH_DIC,
            step_size=STEP_SIZE_DIC,
            train=True,
            h=H,
            w=W,
            t_avg=t_avg,
        )
        (
            X_test_dic,
            X_demo_test,
            time_feature_test,
            test_dict_dic,
            time_test_dict,
            test_demo_dict,
            _,
        ) = utils.overlapping_windows(
            "dic",
            test_dict_dic,
            test_demo_dict,
            window_length=WINDOW_LENGTH_DIC,
            step_size=STEP_SIZE_DIC,
            train=False,
            t_avg=t_avg,
            h=H,
            w=W,
        )
        assert X_train_acoustic.shape[0] == X_train_dic.shape[0], (
            "Acoustic and DIC datasets must have the same number of samples (windows),"
            " but they have {} and {} respectively".format(
                X_train_acoustic.shape[0], X_train_dic.shape[0]
            )
        )

    # dataloaders (x, y are the same!)
    data_loaders = utils.BuildDataloaders(BATCH, TECHNIQUE)
    if TECHNIQUE == "fmoc":
        WINDOW_LENGTH = WINDOW_LENGTH_DIC
        STEP_SIZE = STEP_SIZE_DIC
        X_train = (X_train_acoustic, X_train_dic)
        X_test = (X_test_acoustic, X_test_dic)
    scaler_x, scaler_t_f, scaler_demo = data_loaders.preprocess(
        X_train,
        X_train,
        X_test,
        X_test,
        X_demo_train,
        X_demo_test,
        time_feature_train,
        time_feature_test,
    )
    (
        static_loader,
        train_loader,
        val_loader,
        test_loader,
    ) = data_loaders.create_dataloaders()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    print("\nCurrent dataset: ", TECHNIQUE)

    if TECHNIQUE == "fmoc":
        print("\ntrain acoustic shape: ", X_train_acoustic.shape)
        print("train dic shape: ", X_train_dic.shape)
        print("test acoustic shape: ", X_test_acoustic.shape)
        print("test dic shape: ", X_test_dic.shape)
    else:
        print("\ntrain shape: ", X_train.shape)
        print("test shape: ", X_test.shape)

    print("\nCurrent device:", device)

    if TECHNIQUE == "fmoc":
        train_dict = (train_dict_acoustic, train_dict_dic)
        test_dict = (test_dict_acoustic, test_dict_dic)
        n_inputs_acoustic = X_train[0].shape[2]
        n_inputs_dic = 1
        n_seq_acoustic = X_train[0].shape[1]
        n_seq_dic = X_train[1].shape[2]
        n_inputs = None
        n_seq = None
    else:
        n_inputs = X_train.shape[2]
        n_seq = X_train.shape[1]
        n_inputs_acoustic = None
        n_inputs_dic = None
        n_seq_acoustic = None
        n_seq_dic = None

    if BAYESIAN_OPT:
        (
            AE_LR,
            ALPHA,
            EPOCHS_AE,
            ENCODER_DIM,
            HIDDEN_DIM,
            DR_RATE,
            DC_LR,
            BETA,
            EPOCHS_DC,
        ) = utils.BayesianOptDSMC(
            data_loaders,
            t_avg,
            n_inputs,
            n_seq,
            train_dict,
            time_train_dict,
            train_demo_dict,
            scaler_x,
            scaler_t_f,
            scaler_demo,
            n_clusters,
            BATCH,
            TECHNIQUE,
            device,
            n_inputs_acoustic,
            n_inputs_dic,
            n_seq_acoustic,
            n_seq_dic,
            H=H,
            W=W,
        ).optimized_models
        # in case you accidentally put pretrained = True with bayesian optimization
        pretrained_ae = False
        pretrained_dc = False

    if "cmaps" in TECHNIQUE:
        model_ae = models.AE(
            X_train.shape[2],
            X_train.shape[1],
            ENCODER_DIM,
            HIDDEN_DIM,
            dr_rate=DR_RATE,
        ).to(device)
    elif "mimic" in TECHNIQUE:
        model_ae = models.AE(
            X_train.shape[2],
            X_train.shape[1],
            ENCODER_DIM,
            HIDDEN_DIM,
            dr_rate=DR_RATE,
            use_demo=True,
        ).to(device)
    else:
        model_ae = models.AE_ACOUSTIC_DIC(
            X_train[0].shape[2],
            1,
            X_train[0].shape[1],
            X_train[1].shape[2],
            X_train[1].shape[3],
            X_train[1].shape[4],
            ENCODER_DIM,
            HIDDEN_DIM,
            dr_rate=DR_RATE,
        ).to(device)

    if pretrained_ae:
        model_ae.load_state_dict(
            th.load(
                f"models/{TECHNIQUE}/ae_model_" + TECHNIQUE + ".pt", map_location=device
            )
        )
    else:
        print("\nTraining the AE model")
        model_ae = run_models.train_ae(
            model_ae,
            TECHNIQUE,
            train_loader,
            val_loader,
            AE_LR,
            ALPHA,
            device,
            epochs=EPOCHS_AE,
            save=SAVE,
        )
        if SAVE:
            th.save(
                model_ae.state_dict(),
                f"./models/{TECHNIQUE}/ae_model_" + TECHNIQUE + ".pt",
            )

    if not TECHNIQUE == "fmoc":
        if SAVE:
            run_models.evaluate_ae(
                model_ae, scaler_t_f, TECHNIQUE, static_loader, device, save=SAVE
            )

    del X_train, X_test

    # Build the cluster model
    cluster_model = models.DC(n_clusters, ENCODER_DIM, model_ae, device)

    if pretrained_dc:
        cluster_model = th.load(
            f"models/{TECHNIQUE}/dc_model_" + TECHNIQUE + ".pt", map_location=device
        )
    else:
        # Train the cluster model
        cluster_model = run_models.train_cluster(
            static_loader,
            train_loader,
            cluster_model,
            scaler_t_f,
            TECHNIQUE,
            ALPHA,
            BETA,
            EPOCHS_DC,
            DC_LR,
            device,
            save=SAVE,
        )
        if SAVE:
            th.save(
                cluster_model, f"./models/{TECHNIQUE}/dc_model_" + TECHNIQUE + ".pt"
            )

    # Cluster the real data according to the trained model
    cluster_model.eval()
    run_models.evaluate_dc_model(
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
        WINDOW_LENGTH,
        STEP_SIZE,
        device,
        TECHNIQUE,
        SAVE,
    )


    if ENABLE_VISUALS:
        plt.show()


if __name__ == "__main__":
    print(
        "This is the code for the paper 'A robust generalized deep monotonic feature extraction model for label-free "
        "prediction of degenerative phenomena', where we develop the Deep Soft Monotonic Clustering (DSMC) model."
        "\n\nThe DSMC model is developed in two stages: i) a deep autoencoder (AE) is trained to extract the features, "
        "and ii) a deep clustering (DC) model is trained to cluster the extracted features. These models are unique as"
        " they consider partially the time domain to their calculation\n"
    )

    print(f"PyTorch version: {th.__version__}")
    print(f"CUDA version: {th.version.cuda}")
    print("Code running...\n")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fmoc",
        default=False,
        type=str2bool,
        help="Use F-MOC dataset. default=False",
    )
    parser.add_argument(
        "--mimic",
        default=False,
        type=str2bool,
        help="Use MIMIC-III dataset. If False and fmoc=False, it uses the C-MAPSS dataset, default=False",
    )
    parser.add_argument(
        "--bayesian_opt",
        default=False,
        help="Enable Bayesian Optimization for hyperparameters tuning, default=False",
    )
    parser.add_argument(
        "--save",
        default=True,
        type=str2bool,
        help="Enable saving of results, default=True",
    )
    parser.add_argument(
        "--enable_visuals",
        default=True,
        type=str2bool,
        help="Enable viewing the figures, default=True",
    )
    parser.add_argument(
        "--pretrained",
        default=False,
        type=str2bool,
        help="Whether we already have a pretrained DSMC model, default=False",
    )

    args = parser.parse_args()

    run_process()
