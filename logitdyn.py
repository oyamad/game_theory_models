"""
Filename: logitdyn.py

Authors: Daisuke Oyama

Logit dynamics.

"""
from __future__ import division

import numpy as np


class LogitDynamics(object):
    """
    Parameters
    ----------
    g : NormalFormGame

    beta : scalar(float)

    """
    def __init__(self, g, beta=1.0):
        self.g = g
        self.N = self.g.N
        self.players = self.g.players
        self.nums_actions = self.g.nums_actions

        self.beta = beta

        self.current_actions = np.zeros(self.N, dtype=int)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self._set_choice_probs()

    def _set_choice_probs(self):
        for player in self.players:
            payoff_array_rotated = \
                player.payoff_array.transpose(list(range(1, self.N)) + [0])
            # Shift payoffs so that max = 0 for each opponent action profile
            payoff_array_rotated -= \
                payoff_array_rotated.max(axis=-1)[..., np.newaxis]
            # cdfs left unnormalized
            player.logit_choice_cdfs = \
                np.exp(payoff_array_rotated*self.beta).cumsum(axis=-1)
            # player.logit_choice_cdfs /= player.logit_choice_cdfs[..., [-1]]

    def set_init_actions(self, init_actions=None):
        if init_actions is None:
            init_actions = np.empty(self.N, dtype=int)
            for i in range(self.N):
                init_actions[i] = np.random.randint(self.nums_actions[i])

        self.current_actions[:] = init_actions

    def play(self, player_ind):
        i = player_ind

        # Tuple of the actions of opponent players i+1, ..., N, 0, ..., i-1
        opponent_actions = \
            tuple(self.current_actions[i+1:]) + tuple(self.current_actions[:i])

        cdf = self.players[i].logit_choice_cdfs[opponent_actions]
        random_value = np.random.random()
        next_action = cdf.searchsorted(random_value*cdf[-1], side='right')
        self.current_actions[i] = next_action

    def simulate(self, ts_length, init_actions=None):
        """
        Return array of ts_length arrays of N actions

        """
        actions_sequence = np.empty((ts_length, self.N), dtype=int)
        actions_sequence_iter = \
            self.simulate_iter(ts_length, init_actions=init_actions)

        for t, actions in enumerate(actions_sequence_iter):
            actions_sequence[t] = actions

        return actions_sequence

    def simulate_iter(self, ts_length, init_actions=None):
        """
        Iterator version of `simulate`

        """
        self.set_init_actions(init_actions=init_actions)
        player_ind_sequence = np.random.randint(self.N, size=ts_length)

        for t in range(ts_length):
            yield self.current_actions
            self.play(player_ind=player_ind_sequence[t])

    def replicate(self, T, num_reps, init_actions=None):
        out = np.empty((num_reps, self.N), dtype=int)

        for j in range(num_reps):
            actions_sequence_iter = \
                self.simulate_iter(T+1, init_actions=init_actions)
            for actions in actions_sequence_iter:
                x = actions
            out[j] = x

        return out
