"""
Filename: brd.py

Authors: Daisuke Oyama

Best response dynamics type models.

"""
from __future__ import division

import numpy as np
from game_tools import Player


class BRD(object):
    def __init__(self, payoff_matrix, N):
        A = np.asarray(payoff_matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('payoff matrix must be square')
        self.num_actions = A.shape[0]  # Number of actions
        self.N = N  # Number of players

        self.player = Player(A)  # "Representative player"

        # Current action distribution
        self.current_action_dist = np.zeros(self.num_actions, dtype=int)
        self.current_action_dist[0] = self.N  # Initialization

    def set_init_action_dist(self, init_action_dist=None):
        if init_action_dist is None:
            actions = np.random.randint(self.num_actions, size=self.N)
            init_action_dist = np.bincount(actions, minlength=self.num_actions)
        self.current_action_dist = init_action_dist

    def play(self, current_action):
        next_action_dists = \
            self.best_response_transition(
                self.current_action_dist, current_action
            )[current_action]

        if len(next_action_dists) > 1:
            np.random.shuffle(next_action_dists)
        self.current_action_dist = next_action_dists[0]

    def best_response_transition(self, current_action_dist, action=None):
        if action is None:
            actions = np.nonzero(current_action_dist)[0]
        else:
            actions = [action]
        out = dict()
        for action in actions:
            current_action_dist[action] -= 1
            brs = self.player.best_response(current_action_dist,
                                            tie_breaking=False)
            num_brs = len(brs)
            out[action] = np.empty((num_brs, self.num_actions), dtype=int)
            out[action][:] = current_action_dist
            out[action][np.arange(num_brs), brs] += 1
            current_action_dist[action] += 1
        return out

    def simulate(self, ts_length, init_action_dist=None):
        action_dist_sequence = \
            np.empty((ts_length, self.num_actions), dtype=int)
        action_dist_sequence_iter = \
            self.simulate_iter(ts_length, init_action_dist=init_action_dist)

        for t, action_dist in enumerate(action_dist_sequence_iter):
            action_dist_sequence[t] = action_dist

        return action_dist_sequence

    def simulate_iter(self, ts_length, init_action_dist=None):
        self.set_init_action_dist(init_action_dist=init_action_dist)

        for t in range(ts_length):
            yield self.current_action_dist
            player_ind = np.random.randint(self.N)  # Player to revise
            action = np.searchsorted(
                self.current_action_dist.cumsum(), player_ind, side='right'
            )  # Action the revising player is playing
            self.play(current_action=action)

    def replicate(self, T, init_action_dist=None):
        pass


class KMR(BRD):
    def __init__(self, payoff_matrix, N):
        BRD.__init__(self, payoff_matrix, N)

        # Mutation probability
        self.epsilon = 0.

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def play(self, current_action):
        if np.random.random() < self.epsilon:  # Mutation
            next_action = self.player.random_choice()
            self.current_action_dist[current_action] -= 1
            self.current_action_dist[next_action] += 1
        else:  # Best response
            next_action_dists = \
                self.best_response_transition(
                    self.current_action_dist, current_action
                )[current_action]

            if len(next_action_dists) > 1:
                np.random.shuffle(next_action_dists)
            self.current_action_dist = next_action_dists[0]
