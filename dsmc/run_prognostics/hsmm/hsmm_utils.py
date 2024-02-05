import numpy as np
import pickle
import json


# masks error when applying log(0)
def log_mask_zero(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


def get_single_history(data, index):
    history = data[index, :].reshape((data.shape[1], 1))
    history = history[~np.all(history == 0, axis=1)]

    return history


def get_single_history_states(states, index, obs_state_len):
    history_states = states[index, :].reshape((states.shape[1], 1))

    for j in range(len(history_states)):
        if history_states[j, 0] > history_states[j + 1, 0]:
            history_states = history_states[0 : j + 2 - obs_state_len, 0]
            break
    return history_states


def get_viterbi(HSMM, data):
    results = np.zeros_like(data)

    for i in range(data.shape[0]):
        history = get_single_history(data, i)
        newstate_t = HSMM.predict(history)
        pred_states = np.asarray(newstate_t[0])
        results[i, : len(pred_states)] = pred_states.copy()

    return np.asarray(results, dtype=int)
