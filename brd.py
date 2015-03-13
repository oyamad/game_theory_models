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
        self.tie_breaking = 'smallest'

        # Current action distribution
        self.current_action_dist = np.zeros(self.num_actions, dtype=int)
        self.current_action_dist[0] = self.N  # Initialization

    def set_init_action_dist(self, init_action_dist=None):
        """
        Set the attribute `current_action_dist` to `init_action_dist`.

        Parameters
        ----------
        init_action_dist : array_like(float, ndim=1),
                           optional(default=None)
            Array containing the initial action distribution. If not
            supplied, randomly chosen uniformly over the set of possible
            action distributions.

        """
        if init_action_dist is None:  # Randomly choose an action distribution
            cutoffs = np.empty(self.num_actions, dtype=int)
            cutoffs[-1] = self.N + self.num_actions - 1
            cutoffs[:-1] = np.random.choice(self.N+self.num_actions-1,
                                            self.num_actions-1, replace=False)
            cutoffs[:-1].sort()
            cutoffs[1:] -= cutoffs[:-1] + 1
            init_action_dist = cutoffs
        self.current_action_dist[:] = init_action_dist

    def play(self, current_action):
        self.current_action_dist[current_action] -= 1
        opponent_action_dist = self.current_action_dist
        next_action = self.player.best_response(opponent_action_dist,
                                                tie_breaking=self.tie_breaking)
        self.current_action_dist[next_action] += 1

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

        # Sequence of randomly chosen players to revise
        player_ind_sequence = np.random.randint(self.N, size=ts_length)

        for t in range(ts_length):
            yield self.current_action_dist
            action = np.searchsorted(
                self.current_action_dist.cumsum(), player_ind_sequence[t],
                side='right'
            )  # Action the revising player is playing
            self.play(current_action=action)

    def replicate(self, T, num_reps, init_action_dist=None):
        out = np.empty((num_reps, self.num_actions), dtype=int)

        for j in range(num_reps):
            action_dist_sequence_iter = \
                self.simulate_iter(T+1, init_action_dist=init_action_dist)
            for action_dist in action_dist_sequence_iter:
                x = action_dist
            out[j] = x

        return out


class KMR(BRD):
    def __init__(self, payoff_matrix, N, epsilon=0.1):
        BRD.__init__(self, payoff_matrix, N)

        # Mutation probability
        self.epsilon = epsilon

    def play(self, current_action):
        if np.random.random() < self.epsilon:  # Mutation
            self.current_action_dist[current_action] -= 1
            next_action = self.player.random_choice()
            self.current_action_dist[next_action] += 1
        else:  # Best response
            BRD.play(self, current_action)


class SamplingBRD(BRD):
    def __init__(self, payoff_matrix, N, k=2):
        BRD.__init__(self, payoff_matrix, N)

        # Sample size
        self.k = k

    def play(self, current_action):
        self.current_action_dist[current_action] -= 1
        opponent_action_dist = self.current_action_dist
        actions = np.random.choice(self.num_actions, size=self.k, replace=True,
                                   p=opponent_action_dist/(self.N-1))
        sample_action_dist = np.bincount(actions, minlength=self.num_actions)
        next_action = self.player.best_response(sample_action_dist,
                                                tie_breaking=self.tie_breaking)
        self.current_action_dist[next_action] += 1
