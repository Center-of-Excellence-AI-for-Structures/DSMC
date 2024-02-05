import os
import json
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import sys

sys.path.append("..")
import utils
import settings
from run_prognostics.hsmm.hsmm_base import GaussianHSMM


class PrognosticModels:
    def __init__(self, path, technique, n_clusters, len_obs_state=10):
        self.technique = technique
        self.n_clusters = n_clusters
        self.len_obs_state = len_obs_state
        self.f_value = n_clusters + 5
        self.max_len = None
        self.lengths = None
        self.rmse_gb = None
        self.ruls_gb = None
        self.lower_ruls_gb = None
        self.upper_ruls_gb = None
        self.rmse_svr = None
        self.ruls_svr = None
        self.rmse_hsmm = None
        self.ruls_hsmm = None
        self.lower_ruls_hsmm = None
        self.upper_ruls_hsmm = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.true_ruls = None

        def create_data(files):
            x = []
            lengths = []
            for i, file in enumerate(files):
                cur_df = pd.read_csv(file, header=0).iloc[:, 0]
                x.append(cur_df.values.reshape(-1, 1))
                lengths.append(len(cur_df))

            x = np.concatenate(x, axis=0)

            # create the labels of RUL according to length
            y = []
            for i, length in enumerate(lengths):
                y.extend([length - j for j in range(length)])
            y = np.array(y)
            return x, y, lengths

        def create_data_hsmm(files, max_len):
            obs_concat = np.zeros((len(files), max_len))

            for i, file in enumerate(files):
                obs = pd.read_csv(file, usecols=[0])
                obs1 = obs.to_numpy(copy=True).reshape((1, len(obs)))

                for j in range(len(obs)):
                    if obs1[0, j] != self.n_clusters:
                        obs_concat[i, j] = (
                            obs1[0, j] + 1
                        )  # HSMM requires the first observation to not be 0
                        index = j

                # Creation of the last observed state (failure state) with final value=15 and duration=10
                # - assists in the model fit
                for k in range(self.len_obs_state):
                    obs_concat[i, index + 1 + k] = self.f_value

            if self.technique == "mimic" and "Train" in files[0]:
                obs_concat = obs_concat[
                    5:, :
                ]  # First 5 train trajectories are excluded due to short length

            self.max_len = max_len
            return obs_concat

        current_directory = os.path.dirname(__file__)
        path = os.path.join(current_directory, os.pardir, path)

        if not self.technique == "mimic":
            # data for svm and GBDT
            files = glob.glob(path + f"clustering_results_{self.technique}_Train_*")
            self.X_train, self.y_train, train_lengths = create_data(files)

            # shuffle the data
            idx = np.arange(len(self.y_train))
            np.random.shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]

            files = glob.glob(path + f"clustering_results_{self.technique}_Test_*")
            files = sorted(files, key=lambda x: int(x.split("_sp")[-1].split(".")[0]))
            self.X_test, self.y_test, self.lengths = create_data(files)

            # true_ruls creation
            y_true = self.y_test.copy()
            self.true_ruls = {f"traj_{i}": [] for i in range(len(self.lengths))}
            for i, length in enumerate(self.lengths):
                self.true_ruls[f"traj_{i}"].extend(y_true[:length])
                y_true = y_true[length:]
        else:
            train_lengths = [2100, 2100]
            self.lengths = [2100, 2100]

        # data for hsmm
        files = glob.glob(path + f"clustering_results_" + self.technique + "_Train_*")
        files = sorted(files, key=lambda x: int(x.split("_sp")[-1].split(".")[0]))

        train = create_data_hsmm(
            files, max_len=max(max(train_lengths), max(self.lengths)) + 100
        )

        files = glob.glob(path + f"clustering_results_" + self.technique + "_Test_*")
        files = sorted(files, key=lambda x: int(x.split("_sp")[-1].split(".")[0]))
        test = create_data_hsmm(
            files, max_len=max(max(train_lengths), max(self.lengths)) + 100
        )

        self.train_hsmm = train
        self.test_hsmm = test

    def store_rul(self, y_pred, y_lower, y_upper):
        ruls = {f"traj_{i}": [] for i in range(len(self.lengths))}
        lower_ruls = {f"traj_{i}": [] for i in range(len(self.lengths))}
        upper_ruls = {f"traj_{i}": [] for i in range(len(self.lengths))}
        for i, length in enumerate(self.lengths):
            ruls[f"traj_{i}"].extend(y_pred[:length])

            if y_lower is not None:
                lower_ruls[f"traj_{i}"].extend(y_lower[:length])
                upper_ruls[f"traj_{i}"].extend(y_upper[:length])
                y_lower = y_lower[length:]
                y_upper = y_upper[length:]
            y_pred = y_pred[length:]

            # save ruls
            current_directory = os.path.dirname(__file__)
            # Go one directory back using os.path.normpath
            current_directory = os.path.normpath(
                os.path.join(current_directory, os.pardir)
            )
            path = os.path.join(
                current_directory, "results", self.technique, "prognostics"
            )
            path_rul = path + f"/rul_per_traj_{self.technique}.json"
            path_lower_rul = path + f"/lower_rul_per_traj_{self.technique}.json"
            path_upper_rul = path + f"/upper_rul_per_traj_{self.technique}.json"
            path_true_rul = path + f"/true_rul_per_traj_{self.technique}.json"

            with open(path_rul, "w") as fp:
                json.dump(ruls, fp, cls=utils.NumpyArrayEncoder)

            with open(path_lower_rul, "w") as fp:
                json.dump(lower_ruls, fp, cls=utils.NumpyArrayEncoder)

            with open(path_upper_rul, "w") as fp:
                json.dump(upper_ruls, fp, cls=utils.NumpyArrayEncoder)

            with open(path_true_rul, "w") as fp:
                json.dump(self.true_ruls, fp, cls=utils.NumpyArrayEncoder)

        return ruls, lower_ruls, upper_ruls

    def gradient_boosting(self):
        """
        Perform Gradient Boosting Regression for RUL prediction with uncertainty estimation.

        Parameters:
        - X_train: Training data features (time series data)
        - y_train: Training data labels (RUL)
        - X_test: Test data features (time series data)
        - y_test: Test data labels (RUL)
        - n_estimators: Number of boosting stages
        - max_depth: Maximum depth of the individual trees
        - learning_rate: Shrinkage parameter for the ensemble

        Returns:
        - Predicted RUL
        - Lower bound of the prediction interval
        - Upper bound of the prediction interval
        """

        all_models = {}
        for alpha in [0.05, 0.5, 0.95]:
            # Fit the Gradient Boosting Regression model
            gb_reg = GradientBoostingRegressor(
                loss="quantile", alpha=alpha, random_state=settings.seed_number
            )
            all_models["q %1.2f" % alpha] = gb_reg.fit(self.X_train, self.y_train)

        y_lower = all_models["q 0.05"].predict(self.X_test)  # 95% confidence interval
        y_upper = all_models["q 0.95"].predict(self.X_test)  # 95% confidence interval
        y_pred = all_models["q 0.50"].predict(self.X_test)  # mean

        ruls, lower_ruls, upper_ruls = self.store_rul(y_pred, y_lower, y_upper)
        return ruls, lower_ruls, upper_ruls

    def svm(self):
        svr_reg = SVR(kernel="poly")
        svr_reg.fit(self.X_train, self.y_train)
        y_pred = svr_reg.predict(self.X_test)
        ruls, _, _ = self.store_rul(y_pred, None, None)
        return ruls, None, None

    def hsmm(self, n_states=4):
        hsmm_model = GaussianHSMM(
            n_states=n_states,
            n_durations=int((self.max_len) / (n_states - 1)),
            n_iter=100,
            tol=5e-1,
            left_to_right=True,
            obs_state_len=self.len_obs_state,
            f_value=self.f_value,
            random_state=settings.seed_number,
        )

        hsmm_model.fit(self.train_hsmm)
        ruls, lower_ruls, upper_ruls = hsmm_model.prognostics(
            self.test_hsmm, max_timesteps=self.max_len, technique=self.technique
        )
        return ruls, lower_ruls, upper_ruls

    def estimate_rul_per_model(self):
        n_states = 7 if not self.technique == "fmoc" else 4
        if not self.technique == "mimic":
            (
                self.ruls_gb,
                self.lower_ruls_gb,
                self.upper_ruls_gb,
            ) = self.gradient_boosting()
            self.ruls_svr, _, _ = self.svm()

            self.ruls_hsmm, self.lower_ruls_hsmm, self.upper_ruls_hsmm = self.hsmm(
                n_states=n_states
            )

            self.rmse_gb = utils.calculate_rmse_rul_cmaps(self.ruls_gb, self.true_ruls)
            self.rmse_svr = utils.calculate_rmse_rul_cmaps(
                self.ruls_svr, self.true_ruls
            )
            self.rmse_hsmm = utils.calculate_rmse_rul_cmaps(
                self.ruls_hsmm, self.true_ruls
            )
        else:
            self.ruls_hsmm, self.lower_ruls_hsmm, self.upper_ruls_hsmm = self.hsmm(
                n_states=n_states
            )

    def return_results(self):
        return (
            self.rmse_gb,
            self.ruls_gb,
            self.true_ruls,
            self.lower_ruls_gb,
            self.upper_ruls_gb,
            self.rmse_svr,
            self.ruls_svr,
            None,
            None,
            self.rmse_hsmm,
            self.ruls_hsmm,
            self.lower_ruls_hsmm,
            self.upper_ruls_hsmm,
        )
