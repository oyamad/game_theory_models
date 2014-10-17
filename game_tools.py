"""
Filename: game_tools.py

Author: Daisuke Oyama

Tools for Game Theory.

"""
from __future__ import division

import numpy as np


class Player_NormalFormGame_2P(object):
    """
    Class representing a player in a two-player normal form game.

    """
    def __init__(self, payoff_matrix, init_action=None):
        self.payoff_matrix = np.asarray(payoff_matrix)

        if len(self.payoff_matrix.shape) != 2:
            raise ValueError('payoff matrix must be 2-dimensional')

        self.n, self.m = self.payoff_matrix.shape

        self.current_action = init_action

    def best_response(self, mixed_strategy,
                      tie_breaking=True, payoff_perturbations=None):
        pass

    def random_choice(self):
        pass


class NormalFormGame_2P(object):
    """
    Class representing a two-player normal form game.

    """
    player_indices = [0, 1]

    def __init__(self, payoffs):
        payoffs_ndarray = np.asarray(payoffs)
        if payoffs_ndarray.ndim == 3:  # bimatrix game
            self.bimatrix = payoffs_ndarray
            self.matrices = [payoffs_ndarray[:, :, 0],
                             payoffs_ndarray[:, :, 1].T]
        elif payoffs_ndarray.ndim == 2:  # symmetric game
            try:
                self.bimatrix = np.dstack((payoffs_ndarray, payoffs_ndarray.T))
            except:
                raise ValueError('a symmetric game must be a square array')
            self.matrices = [payoffs_ndarray, payoffs_ndarray]
        else:
            raise ValueError(
                'the game must be represented by a bimatrix or a square matrix'
            )

        self.players = [
            Player_NormalFormGame_2P(self.matrices[i])
            for i in self.player_indices
        ]


def br_corr(mixed_action, payoff_matrix):
    """
    Best response correspondence in pure actions.

    """
    payoff_vec = np.dot(payoff_matrix, mixed_action)
    return np.where(payoff_vec == payoff_vec.max())[0]


def random_choice(actions):
    if len(actions) == 1:
        return actions[0]
    else:
        return np.random.choice(actions)


def pure2mixed(num_actions, action):
    """
    Convert a pure action to the corresponding mixed action.

    """
    mixed_action = np.zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
