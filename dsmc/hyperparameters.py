import json


# Define the hyperparameters as a Python dictionary
def save_hypers():
    hyper_cmaps = {
        "window_length": 10,
        "step_size": 1,
        "encoder_dim": 4,
        "hidden_dim": 123,
        "dr_rate": 0.3,
        "clusters": 10,
        "lr_ae": 0.0005,
        "lr_dc": 0.0003,
        "batch_size": 32,
        "epochs_ae": 105,
        "epochs_dc": 23,
        "alpha": 0.772,
        "beta": 2.756,
    }

    hyper_mimic = {
        "window_length": 10,
        "step_size": 1,
        "encoder_dim": 8,
        "hidden_dim": 116,
        "dr_rate": 0.2,
        "clusters": 10,
        "lr_ae": 0.0012,
        "lr_dc": 0.0006,
        "batch_size": 128,
        "epochs_ae": 179,
        "epochs_dc": 26,
        "alpha": 1.795,
        "beta": 1.02,
    }

    hyper_fmoc = {
        "window_length_dic": 6,
        "step_size_dic": 3,  # max images to ignore should be 5 and always less than window_length
        "encoder_dim": 9,
        "hidden_dim": 48,
        "dr_rate": 0.14,
        "clusters": 30,
        "lr_ae": 5e-3,
        "lr_dc": 9e-4,
        "batch_size": 128,
        "epochs_ae": 112,
        "epochs_dc": 17,
        "alpha": 2.0,
        "beta": 0.964,
        "image_height": 128,
        "image_width": 64,
    }

    # Save the hyperparameters to a JSON file
    with open("hyperparameters/hyper_cmaps.json", "w") as f:
        json.dump(hyper_cmaps, f)

    with open("hyperparameters/hyper_mimic.json", "w") as f:
        json.dump(hyper_mimic, f)

    with open("hyperparameters/hyper_fmoc.json", "w") as f:
        json.dump(hyper_fmoc, f)

    print("\nFinished saving hyperparameters.")
