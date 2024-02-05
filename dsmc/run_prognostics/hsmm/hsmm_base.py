import os
import numpy as np
import json
import pickle
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from itertools import zip_longest

from run_prognostics.hsmm.mle import _forward, _backward, _u_only
import run_prognostics.hsmm.smoothed as core
from run_prognostics.hsmm.hsmm_utils import *
from dsmc.utils import NumpyArrayEncoder


np.seterr(invalid="ignore")


class HSMM:
    def __init__(
        self,
        n_states=2,
        n_durations=5,
        n_iter=20,
        tol=1e-2,
        left_to_right=False,
        obs_state_len=None,
        f_value=None,
        random_state=None,
        name="",
    ):
        if not n_states >= 2:
            raise ValueError("number of states (n_states) must be at least 2")
        if not n_durations >= 1:
            raise ValueError("number of durations (n_durations) must be at least 1")

        if obs_state_len is not None and f_value is not None:
            self.last_observed = True
        elif obs_state_len is not None and f_value is None:
            raise ValueError("provide the observed state's final value")
        elif obs_state_len is None and f_value is not None:
            raise ValueError("provide the observed state's length")
        else:
            self.last_observed = False

        self.n_states = n_states
        self.n_durations = n_durations
        self.n_iter = n_iter
        self.tol = tol
        self.left_to_right = left_to_right
        self.obs_state_len = obs_state_len
        self.f_value = f_value
        self.random_state = random_state
        self.name = name
        self._print_name = ""

    # _init: initializes model parameters if there are none yet
    # if left_to_right is True then the first state has start probability=1 and the tmat has transition probability=1
    # only for i+1 state. Last state is observed
    def _init(self, X=None):
        if self.name != "" and self._print_name == "":
            self._print_name = f" ({self.name})"
        if not hasattr(self, "pi") and not self.left_to_right:
            self.pi = np.full(self.n_states, 1.0 / self.n_states)
        elif not hasattr(self, "pi") and self.left_to_right:
            self.pi = np.zeros(self.n_states)
            self.pi[0] = 1
        if not hasattr(self, "tmat") and not self.left_to_right:
            self.tmat = np.full(
                (self.n_states, self.n_states), 1.0 / (self.n_states - 1)
            )
            for i in range(self.n_states):
                self.tmat[i, i] = 0.0  # no self-transitions in EDHSMM
        elif not hasattr(self, "tmat") and self.left_to_right:
            self.tmat = np.zeros((self.n_states, self.n_states))

            for i in range(len(self.tmat)):
                for j in range(len(self.tmat[i]) - 1):
                    if i == j and j < len(self.tmat[i]) - 2:
                        self.tmat[i, j + 1] = 1

                    elif i == j and j == len(self.tmat[i]) - 2:
                        self.tmat[i, j + 1] = 1
            # self.tmat[-1, -1] = 1

        self._dur_init()  # duration

    # _check: check if properties of model parameters are satisfied
    def _check(self):
        # starting probabilities
        self.pi = np.asarray(self.pi)
        if self.pi.shape != (self.n_states,):
            raise ValueError(
                "start probabilities (self.pi) must have shape ({},)".format(
                    self.n_states
                )
            )
        if not np.allclose(self.pi.sum(), 1.0):
            raise ValueError("start probabilities (self.pi) must add up to 1.0")
        # transition probabilities
        self.tmat = np.asarray(self.tmat)
        if self.tmat.shape != (self.n_states, self.n_states):
            raise ValueError(
                "transition matrix (self.tmat) must have shape ({0}, {0})".format(
                    self.n_states
                )
            )
        if not np.allclose(self.tmat.sum(axis=1), 1.0) and not self.left_to_right:
            raise ValueError("transition matrix (self.tmat) must add up to 1.0")
        if not self.left_to_right:
            for i in range(self.n_states):
                if self.tmat[i, i] != 0.0:  # check for diagonals
                    raise ValueError(
                        "transition matrix (self.tmat) must have all diagonals equal to 0.0"
                    )
        # duration probabilities
        self._dur_check()

    # _dur_init: initializes duration parameters if there are none yet
    def _dur_init(self):
        """
        arguments: (self)
        return: None
        > initialize the duration parameters
        """
        pass  # implemented in subclass

    # _dur_check: checks if properties of duration parameters are satisfied
    def _dur_check(self):
        """
        arguments: (self)
        return: None
        > check the duration parameters
        """
        pass  # implemented in subclass

    # _dur_probmat: compute the probability per state of each duration
    def _dur_probmat(self):
        """
        arguments: (self)
        return: duration probability matrix
        """
        pass  # implemented in subclass

    # _dur_mstep: perform m-step for duration parameters
    def _dur_mstep(self):
        """
        arguments: (self, new_dur)
        return: None
        > compute the duration parameters
        """
        pass  # implemented in subclass

    # _emission_logl: compute the log-likelihood per state of each observation
    def _emission_logl(self):
        """
        arguments: (self, X)
        return: logframe
        """
        pass  # implemented in subclass

    # _emission_pre_mstep: prepare m-step for emission parameters
    def _emission_pre_mstep(self):
        """
        arguments: (self, gamma, emission_var)
        return: None
        > process gamma and save output to emission_var
        """
        pass  # implemented in subclass

    # _emission_mstep: perform m-step for emission parameters
    def _emission_mstep(self):
        """
        arguments: (self, X, emission_var)
        return: None
        > compute the emission parameters
        """
        pass  # implemented in subclass

    # _state_sample: generate observation for given state
    def _state_sample(self):
        """
        arguments: (self, state, random_state=None)
        return: np.ndarray of length equal to dimension of observation
        > generate sample from state
        """
        pass  # implemented in subclass

    # sample: generate random observation series
    def sample(self, n_samples=5, left_censor=0, right_censor=1, random_state=None):
        self._init(None)  # see "note for programmers" in init() in GaussianHSMM
        # self._check()
        # setup random state
        if random_state is None:
            random_state = self.random_state
        rnd_checked = np.random.default_rng(random_state)
        # adapted from hmmlearn 0.2.3 (see _BaseHMM.score function)
        pi_cdf = np.cumsum(self.pi)
        tmat_cdf = np.cumsum(self.tmat, axis=1)
        dur_cdf = np.cumsum(self._dur_probmat(), axis=1)
        # for first state
        currstate = (
            pi_cdf > rnd_checked.random()
        ).argmax()  # argmax() returns only the first occurrence
        currdur = (dur_cdf[currstate] > rnd_checked.random()).argmax() + 1
        if left_censor != 0:
            currdur -= rnd_checked.integers(
                low=0, high=currdur
            )  # if with left censor, remove some of X
        if right_censor == 0 and currdur > n_samples:
            print(
                f"SAMPLE{self._print_name}: n_samples is too small to contain the first state duration."
            )
            return None
        state_sequence = [currstate] * currdur
        X = [
            self._state_sample(currstate, rnd_checked) for i in range(currdur)
        ]  # generate observation
        ctr_sample = currdur
        # for next state transitions
        while ctr_sample < n_samples:
            currstate = (tmat_cdf[currstate] > rnd_checked.random()).argmax()
            currdur = (dur_cdf[currstate] > rnd_checked.random()).argmax() + 1
            # test if now in the end of generating samples
            if ctr_sample + currdur > n_samples:
                if right_censor != 0:
                    currdur = (
                        n_samples - ctr_sample
                    )  # if with right censor, cap the samples to n_samples
                else:
                    break  # else, do not include exceeding state duration
            state_sequence += [currstate] * currdur
            X += [
                self._state_sample(currstate, rnd_checked) for i in range(currdur)
            ]  # generate observation
            ctr_sample += currdur
        return ctr_sample, np.atleast_2d(X), np.array(state_sequence, dtype=int)

    # _core_u_only: Python implementation
    def _core_u_only(self, logframe):
        n_samples = logframe.shape[0]
        u = np.empty((n_samples, self.n_states, self.n_durations))
        _u_only(n_samples, self.n_states, self.n_durations, logframe, u)
        return u

    # _core_forward: Python implementation
    def _core_forward(self, u, logdur, left_censor, right_censor):
        n_samples = u.shape[0]
        if right_censor != 0:
            eta_samples = n_samples + self.n_durations - 1
        else:
            eta_samples = n_samples
        eta = np.empty((eta_samples + 1, self.n_states, self.n_durations))  # +1
        xi = np.empty((n_samples + 1, self.n_states, self.n_states))  # +1
        alpha = _forward(
            n_samples,
            self.n_states,
            self.n_durations,
            log_mask_zero(self.pi),
            log_mask_zero(self.tmat),
            logdur,
            left_censor,
            right_censor,
            eta,
            u,
            xi,
        )
        return eta, xi, alpha

    # _core_backward: Python implementation
    def _core_backward(self, u, logdur, right_censor):
        n_samples = u.shape[0]
        beta = np.empty((n_samples, self.n_states))
        betastar = np.empty((n_samples, self.n_states))
        _backward(
            n_samples,
            self.n_states,
            self.n_durations,
            log_mask_zero(self.pi),
            log_mask_zero(self.tmat),
            logdur,
            right_censor,
            beta,
            u,
            betastar,
        )
        return beta, betastar

    # _core_smoothed: The SLOWEST fnc if implemented in python
    # still in Cython
    def _core_smoothed(self, beta, betastar, right_censor, eta, xi):
        n_samples = beta.shape[0]
        gamma = np.empty((n_samples, self.n_states))
        core._smoothed(
            n_samples,
            self.n_states,
            self.n_durations,
            beta,
            betastar,
            right_censor,
            eta,
            xi,
            gamma,
        )
        return gamma

    # _core_viterbi: container for core._viterbi (for multiple observation sequences)
    def _core_viterbi(self, u, logdur, left_censor, right_censor):
        n_samples = u.shape[0]
        state_sequence, state_logl = core._viterbi(
            n_samples,
            self.n_states,
            self.n_durations,
            log_mask_zero(self.pi),
            log_mask_zero(self.tmat),
            logdur,
            left_censor,
            right_censor,
            u,
        )
        return state_sequence, state_logl

    # score: log-likelihood computation from observation series
    def score(self, X, left_censor=1, right_censor=0):
        self._init(X)
        # self._check()
        logdur = log_mask_zero(self._dur_probmat())  # build logdur
        # main computations
        score = 0

        logframe = self._emission_logl(X)  # build logframe
        u = self._core_u_only(logframe)
        if left_censor != 0:
            eta, xi = self._core_forward(u, logdur, left_censor, right_censor)
            beta, betastar = self._core_backward(u, logdur, right_censor)
            gamma = self._core_smoothed(beta, betastar, right_censor, eta, xi)
            score += logsumexp(gamma[0, :])
        else:  # if without left censor, computation can be simplified
            _, betastar = self._core_backward(u, logdur, right_censor)
            gammazero = log_mask_zero(self.pi) + betastar[0]
            score += logsumexp(gammazero)
        return score

    # predict: hidden state & duration estimation from observation series
    def predict(self, X, left_censor=1, right_censor=0):
        self._init(X)
        # self._check()
        logdur = log_mask_zero(self._dur_probmat())  # build logdur
        # main computations
        state_logl = 0  # math note: this is different from score() output
        state_sequence = np.empty(X.shape[0], dtype=int)  # total n_samples = X.shape[0]
        logframe = self._emission_logl(X)  # build logframe
        u = self._core_u_only(logframe)
        iter_state_sequence, iter_state_logl = self._core_viterbi(
            u, logdur, left_censor, right_censor
        )
        state_logl += iter_state_logl
        state_sequence = iter_state_sequence
        return state_sequence, state_logl

    # fit: parameter estimation from observation series
    def fit(self, X, left_censor=1, right_censor=0, save_iters=False):
        score_per_iter = []
        score_per_sample = []

        last_history = X[-1, :].reshape((X.shape[1], 1))
        last_history = last_history[~np.all(last_history == 0, axis=1)]

        self._init(last_history)  # initialization with the last(longer) history
        self._check()

        # main computations
        for itera in range(self.n_iter):
            score = 0

            pi_num = mean_numerator = cov_numerator = denominator = np.full(
                self.n_states, -np.inf
            )
            tmat_num = dur_num = gamma_num = -np.inf

            for i in tqdm(range(len(X)), desc=f"iteration {itera+1}/{self.n_iter}"):
                history = X[i, :].reshape((X.shape[1], 1))
                history = history[~np.all(history == 0, axis=1)]
                emission_var = np.empty(
                    (history.shape[0], self.n_states)
                )  # cumulative concatenation of gammas
                logdur = log_mask_zero(self._dur_probmat())  # build logdur
                j = len(history)

                logframe = self._emission_logl(history)  # build logframe
                logframe[
                    logframe > 0
                ] = 0  # necessary condition for histories with discrete observations; as the model
                # converges and calculates close-to-zero covariances, the probabilities of
                # observing the means get close to 1. So to avoid positive logframe values
                # we set them to 0 (exp(0)=1)

                u = self._core_u_only(logframe)
                eta, xi, alpha = self._core_forward(
                    u, logdur, left_censor, right_censor
                )
                beta, betastar = self._core_backward(u, logdur, right_censor)
                gamma = self._core_smoothed(beta, betastar, right_censor, eta, xi)
                sample_score = logsumexp(gamma[0, :])
                score_per_sample.append(
                    sample_score
                )  # this saves the scores of every history for every iter
                score += sample_score  # this is the total likelihood for all histories for current iter

                # preparation for reestimation / M-step
                if eta.shape[0] != j + 1:
                    eta = eta[: j + 1]
                if gamma.shape[0] != j + 1:
                    gamma = gamma[: j + 1]

                # normalization of each history's xi, eta and gamma with its likelihood
                norm_xi = np.subtract(xi, sample_score)
                norm_eta = np.subtract(eta, sample_score)
                norm_gamma = np.subtract(gamma, sample_score)

                ##############emission matrix estimation##############
                log_history = log_mask_zero(history)
                log_history[np.isnan(log_history)] = -np.inf
                mean_num = (
                    gamma + log_history
                )  # numerator for mean re-estimation of current history
                mean_num = np.subtract(mean_num, sample_score)

                dist = history - self.mean[:, None]
                dist = np.square(dist.reshape((dist.shape[0], dist.shape[1])).T)
                log_dist = log_mask_zero(dist)
                log_dist[np.isnan(log_dist)] = -np.inf
                cov_num = (
                    gamma + log_dist
                )  # numerator for covars re-estimation of current history
                cov_num = np.subtract(cov_num, sample_score)

                # add the mean numerator, covars numerator and denominator of prev history at the end of the current
                # ones
                mean_num_multiple_histories = np.vstack((mean_num, mean_numerator))
                cov_num_multiple_histories = np.vstack((cov_num, cov_numerator))
                denominator_multiple_histories = np.vstack((norm_gamma, denominator))

                # sum over time and histories
                mean_numerator = logsumexp(mean_num_multiple_histories, axis=0)
                cov_numerator = logsumexp(cov_num_multiple_histories, axis=0)
                denominator = logsumexp(denominator_multiple_histories, axis=0)
                ########################################################

                # append the previous sum of xi and eta to the last position of the new xi and eta
                norm_xi[j] = tmat_num
                norm_eta[j] = dur_num

                # Calculation of he total xi, eta and gamma variables for all the histories
                pi_num = logsumexp([pi_num, norm_gamma[0]], axis=0)
                tmat_num = logsumexp(norm_xi, axis=0)
                dur_num = logsumexp(norm_eta, axis=0)

            ############################################################################################################
            # check for loop break

            if itera > 0 and abs(score - old_score) < self.tol:
                print(
                    f"\nFIT{self._print_name}: converged at loop {itera + 1} with score: {score}."
                )
                break
            elif itera > 0 and (np.isnan(score) or np.isinf(score)):
                print("\nThere is no possible solution. Try different parameters.")
                break
            else:
                score_per_iter.append(score)
                old_score = score

            # save the previous version of the model prior to updating
            if save_iters:
                with open("model_iter" + str(itera + 1) + ".txt", "wb") as f:
                    pickle.dump(self, f)

            # emission parameters re-estimation
            weight = mean_numerator - denominator
            weight1 = cov_numerator - denominator

            mean = np.exp(weight)
            covmat = np.exp(weight1)

            for k in range(len(covmat)):
                if covmat[k,] == 0 or np.isnan(covmat[k,]):
                    covmat[k,] = 1e-30

            # reestimation of the rest of the model parameters and model update
            self.pi = np.exp(pi_num - logsumexp(pi_num))
            self.tmat = np.exp(tmat_num - logsumexp(tmat_num, axis=1)[None].T)
            self.dur = np.exp(dur_num - logsumexp(dur_num, axis=1)[None].T)
            self.mean = mean.reshape((mean.shape[0], 1))
            self.covmat = covmat.reshape((covmat.shape[0], 1, 1))

            self.tmat[-1, :] = np.zeros(self.n_states)
            print(
                f"\nFIT{self._print_name}: reestimation complete for loop {itera + 1} with score: {score}."
            )

        score_per_sample = np.array(score_per_sample).reshape((-1, X.shape[0])).T
        score_per_iter = np.array(score_per_iter).reshape(len(score_per_iter), 1)

        if self.last_observed:
            self.dur[-1, self.obs_state_len] = 0
            self.dur[-1, self.obs_state_len - 1] = 1

        return self, score_per_iter, score_per_sample

    def RUL(self, viterbi_states, max_samples, equation=1):
        """
        :param viterbi_states: Single history
        :param max_samples: maximum length of RUL (default: 3000)
        :param equation: 1=best (with reduction with sojourn time to both terms)
        :return:

        Works for a single state history.
        """

        RUL = np.zeros((len(viterbi_states - self.obs_state_len + 1), max_samples))
        mean_RUL, LB_RUL, UB_RUL = (
            np.zeros(len(viterbi_states - self.obs_state_len + 1)) for _ in range(3)
        )
        dur = self.dur
        prev_state, stime = 0, 0
        n_states = self.n_states

        for i, state in enumerate(viterbi_states):
            first, second = (np.zeros_like(dur[0, :]) for _ in range(2))
            first[1] = second[1] = 1
            cdf_curr_state = np.cumsum(dur[state, :])
            if state == prev_state:
                stime += 1
            else:
                prev_state = state
                stime = 1

            if stime < len(cdf_curr_state):
                d_value = cdf_curr_state[stime]
            else:
                d_value = cdf_curr_state[-1]

            available_states = np.arange(state, n_states - 1)

            for j in available_states:
                if j != available_states[-1]:
                    first = np.convolve(first, dur[j, :])
                    second = np.convolve(second, dur[j + 1, :])

                else:
                    first = np.convolve(first, dur[j, :])

            if equation == 1:
                first_red = np.zeros_like(first)
                first_red = first[stime:]

                # make sure that after subtracting the soujourn time from the pmf of the first term, that it still sums to 1
                if first_red.sum() != 1:
                    first_red[0] = first_red[0] + (1 - first_red.sum())

            else:
                first_red = first

            first_red = first_red * (1 - d_value)
            second = second * d_value

            result = [sum(n) for n in zip_longest(first_red, second, fillvalue=0)]

            if available_states.size > 0 or not self.last_observed:
                RUL[i, :] = [
                    sum(n) for n in zip_longest(RUL[i, :], result, fillvalue=0)
                ]
                cdf_curr_RUL = np.cumsum(RUL[i, :])

                # UB RUL
                X, y = [], []
                for l, value in enumerate(cdf_curr_RUL):
                    if value > 0.05:
                        X = [cdf_curr_RUL[l - 1], value]
                        y = [l - 1, l]
                        break
                X = np.asarray(X).reshape(-1, 1)
                y = np.asarray(y).reshape(-1, 1)
                UB_RUL[i] = (
                    LinearRegression()
                    .fit(X, y)
                    .predict(np.asarray(0.05).reshape(-1, 1))
                )

                # LB RUL
                X, y = [], []
                for l, value in enumerate(cdf_curr_RUL):
                    if value > 0.95:
                        X = [cdf_curr_RUL[l - 1], value]
                        y = [l - 1, l]
                        break
                X = np.asarray(X).reshape(-1, 1)
                y = np.asarray(y).reshape(-1, 1)
                LB_RUL[i] = (
                    LinearRegression()
                    .fit(X, y)
                    .predict(np.asarray(0.95).reshape(-1, 1))
                )

                # mean RUL
                value = np.arange(0, RUL.shape[1])
                mean_RUL[i] = sum(RUL[i, :] * value)

            elif not available_states.size > 0 and self.last_observed:
                RUL[i, :], mean_RUL[i], UB_RUL[i], LB_RUL[i] = 0, 0, 0, 0
                mean_RUL = np.hstack(
                    (np.delete(mean_RUL, mean_RUL == 0), np.array((0)))
                )
                UB_RUL = np.hstack((np.delete(UB_RUL, UB_RUL == 0), np.array((0))))
                LB_RUL = np.hstack((np.delete(LB_RUL, LB_RUL == 0), np.array((0))))
                break

        return RUL, mean_RUL, UB_RUL, LB_RUL

    def prognostics(
        self, data, max_timesteps, technique, max_samples=4000, plot_rul=False
    ):
        """

        :param HSMM: Trained HSMM model
        :param data: degradation histories
        :param max_timesteps: maximum timesteps of degradation histories
        :param max_samples: maximum length of RUL (default: 3000)
        :param obs_state_len: legnth of observable state if last_observed=True
        :param COMPARE: True for test dataset of CMAPSS
        :param technique: 'cmapss', 'mimic', or 'fmoc' for file name
        :param plot_rul: Display RUL plot for each sample
        :param last_observed: True if the last state is observable
        :return: None, json files are saved for pdf_rul and mean_rul
        """

        data_list = []
        for i in range(data.shape[0]):
            history = data[i, 0:max_timesteps].reshape((max_timesteps, 1))
            data_list.append(history[~np.all(history == 0, axis=1)])

        viterbi_states_all = get_viterbi(
            self, data
        )  # this has the full length of the observed state

        viterbi_list = []
        for i in range(viterbi_states_all.shape[0]):
            # this has a single timestep for the observed state - Ready for RUL
            viterbi_single_state = get_single_history_states(
                viterbi_states_all, i, obs_state_len=self.obs_state_len
            )
            viterbi_list.append(viterbi_single_state)

        pdf_ruls_all = {
            f"traj_{j}": {
                f"timestep_{i}": np.zeros((max_samples,))
                for i in range(len(viterbi_list[j]))
            }
            for j in range(len(viterbi_list))
        }

        mean_rul_per_step = {
            f"traj_{i}": np.zeros(
                (
                    len(
                        viterbi_list[i],
                    )
                )
            )
            for i in range(len(viterbi_list))
        }
        upper_rul_per_step = {
            f"traj_{i}": np.zeros(
                (
                    len(
                        viterbi_list[i],
                    )
                )
            )
            for i in range(len(viterbi_list))
        }
        lower_rul_per_step = {
            f"traj_{i}": np.zeros(
                (
                    len(
                        viterbi_list[i],
                    )
                )
            )
            for i in range(len(viterbi_list))
        }

        for i in range(viterbi_states_all.shape[0]):
            viterbi_single_state = get_single_history_states(
                viterbi_states_all, i, obs_state_len=self.obs_state_len
            )
            RUL_pred, mean_RUL, UB_RUL, LB_RUL = self.RUL(
                viterbi_single_state, max_samples=max_samples, equation=1
            )

            for j in range(RUL_pred.shape[0]):
                pdf_ruls_all[f"traj_{i}"][f"timestep_{j}"] = RUL_pred[j, :].copy()
                mean_rul_per_step[f"traj_{i}"] = mean_RUL.copy()
                upper_rul_per_step[f"traj_{i}"] = UB_RUL.copy()
                lower_rul_per_step[f"traj_{i}"] = LB_RUL.copy()

        current_directory = os.path.dirname(__file__)
        # Go two directories back using os.path.normpath
        current_directory = os.path.normpath(
            os.path.join(current_directory, os.pardir, os.pardir)
        )
        path = os.path.join(current_directory, "results", technique, "prognostics")
        path_mean_rul = path + f"/mean_rul_per_step_{technique}.json"
        path_pdf_rul = path + f"/pdf_ruls_{technique}.json"

        with open(path_mean_rul, "w") as fp:
            json.dump(mean_rul_per_step, fp, cls=NumpyArrayEncoder)

        with open(path_pdf_rul, "w") as fp:
            json.dump(pdf_ruls_all, fp, cls=NumpyArrayEncoder)

        print(f"Prognostics complete. Results saved to folder: {path}")

        return mean_rul_per_step, lower_rul_per_step, upper_rul_per_step


# Sample Subclass: Explicit Duration HSMM with Gaussian Emissions
class GaussianHSMM(HSMM):
    def __init__(
        self,
        n_states=2,
        n_durations=5,
        n_iter=20,
        tol=1e-2,
        left_to_right=False,
        obs_state_len=None,
        f_value=None,
        random_state=None,
        name="",
        kmeans_init="k-means++",
        kmeans_n_init="auto",
    ):
        super().__init__(
            n_states,
            n_durations,
            n_iter,
            tol,
            left_to_right,
            obs_state_len,
            f_value,
            random_state,
            name,
        )
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init

    def _init(self, X=None):
        super()._init()
        # note for programmers: for every attribute that needs X in score()/predict()/fit(),
        # there must be a condition "if X is None" because sample() doesn't need an X, but
        # default attribute values must be initiated for sample() to proceed.
        if (
            not hasattr(self, "mean") and not self.left_to_right
        ):  # also set self.n_dim here
            if X is None:  # default for sample()
                self.n_dim = 1
                self.mean = np.arange(0.0, self.n_states)[
                    :, None
                ]  # = [[0.], [1.], [2.], ...]
            else:
                self.n_dim = X.shape[1]
                kmeans = cluster.KMeans(
                    n_clusters=self.n_states,
                    random_state=self.random_state,
                    init=self.kmeans_init,
                    n_init=self.kmeans_n_init,
                )
                kmeans.fit(X)
                self.mean = kmeans.cluster_centers_

        elif (
            not hasattr(self, "mean") and self.left_to_right
        ):  # also set self.n_dim here
            if X is None:  # default for sample()
                self.n_dim = 1
                self.mean = np.arange(0.0, self.n_states)[
                    :, None
                ]  # = [[0.], [1.], [2.], ...]
            else:
                self.n_dim = X.shape[1]
                kmeans = cluster.KMeans(
                    n_clusters=self.n_states - 1,
                    random_state=self.random_state,
                    init=self.kmeans_init,
                    n_init=self.kmeans_n_init,
                )
                kmeans.fit(X[: -self.obs_state_len])
                clusters_sorted = np.sort(kmeans.cluster_centers_, axis=0)
                self.mean = np.vstack((clusters_sorted, [self.f_value]))
        else:
            self.n_dim = self.mean.shape[1]  # also default for sample()
        if not hasattr(self, "covmat"):
            if X is None:  # default for sample()
                self.covmat = np.repeat(
                    np.identity(self.n_dim)[None], self.n_states, axis=0
                )
            else:
                self.covmat = np.repeat(
                    np.identity(self.n_dim)[None], self.n_states, axis=0
                )

    def _check(self):
        super()._check()
        # means
        self.mean = np.asarray(self.mean)
        if self.mean.shape != (self.n_states, self.n_dim):
            raise ValueError(
                "means (self.mean) must have shape ({}, {})".format(
                    self.n_states, self.n_dim
                )
            )
        # covariance matrices
        self.covmat = np.asarray(self.covmat)
        if self.covmat.shape != (self.n_states, self.n_dim, self.n_dim):
            raise ValueError(
                "covariance matrices (self.covmat) must have shape ({0}, {1}, {1})".format(
                    self.n_states, self.n_dim
                )
            )

    def _dur_init(self):
        # non-parametric duration
        if not hasattr(self, "dur") and not self.left_to_right:
            self.dur = np.full(
                (self.n_states, self.n_durations), 1.0 / self.n_durations
            )

        elif not hasattr(self, "dur") and self.left_to_right:
            self.dur = np.zeros((self.n_states, self.n_durations))
            self.dur[:-1, 1:].fill(1.0 / (self.n_durations - 1))
            self.dur[-1, self.obs_state_len] = 1

    def _dur_check(self):
        self.dur = np.asarray(self.dur)
        if self.dur.shape != (self.n_states, self.n_durations):
            raise ValueError(
                "duration probabilities (self.dur) must have shape ({}, {})".format(
                    self.n_states, self.n_durations
                )
            )
        if not np.allclose(self.dur.sum(axis=1), 1.0):
            raise ValueError("duration probabilities (self.dur) must add up to 1.0")

    def _dur_probmat(self):
        # non-parametric duration
        return self.dur

    def _dur_mstep(self, new_dur):
        # non-parametric duration
        self.dur = new_dur

    def _emission_logl(self, X):
        # abort EM loop if any covariance matrix is not symmetric, positive-definite.
        # adapted from hmmlearn 0.2.3 (see _utils._validate_covars function)
        for n, cv in enumerate(self.covmat):
            if not np.allclose(cv, cv.T) or np.any(np.linalg.eigvalsh(cv) <= 0):
                raise ValueError(
                    "component {} of covariance matrix is not symmetric, positive-definite.".format(
                        n
                    )
                )                
        n_samples = X.shape[0]
        logframe = np.empty((n_samples, self.n_states))
        for i in range(self.n_states):
            # math note: since Gaussian distribution is continuous, probability density
            # is what's computed here. thus log-likelihood can be positive!
            multigauss = multivariate_normal(self.mean[i], self.covmat[i])
            for j in range(n_samples):
                logframe[j, i] = log_mask_zero(multigauss.pdf(X[j]))
        return logframe

    def _emission_mstep(self, X, emission_var, inplace=True):
        denominator = logsumexp(emission_var, axis=0)
        # denominator = emission_var
        weight_normalized = np.exp(emission_var - denominator)[None].T
        # compute means (from definition; weighted)
        mean = (weight_normalized * X).sum(1)
        # compute covariance matrices (from definition; weighted)
        dist = X - self.mean[:, None]
        covmat = ((dist * weight_normalized)[:, :, :, None] * dist[:, :, None]).sum(1)
        if not inplace:
            return mean, covmat
        else:
            self.mean = mean
            self.covmat = covmat

    def _state_sample(self, state, random_state=None):
        rnd_checked = np.random.default_rng(random_state)
        return rnd_checked.multivariate_normal(self.mean[state], self.covmat[state])
