"""
Filename: localint.py

Authors: Daisuke Oyama, Atsushi Yamagishi

Local interaction model.

"""
from __future__ import division

import numpy as np
from scipy import sparse
from game_tools import Player


class LocalInteraction(object):
    """
    Local Interaction Model

    """
    def __init__(self, payoff_matrix, adj_matrix):
        self.adj_matrix = sparse.csr_matrix(adj_matrix)
        M, N = self.adj_matrix.shape
        if N != M:
            raise ValueError('adjacency matrix must be square')
        self.N = N  # Number of players

        A = np.asarray(payoff_matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('payoff matrix must be square')
        self.num_actions = A.shape[0]  # Number of actions

        self.players = [Player(A) for i in range(self.N)]

        init_actions = np.zeros(self.N, dtype=int)
        self.current_actions_mixed = sparse.csr_matrix(
            (np.ones(self.N, dtype=int), init_actions, np.arange(self.N+1)),
            shape=(self.N, self.num_actions)
        )
        self.current_actions = self.current_actions_mixed.indices.view()

    def set_init_actions(self, init_actions=None):
        if init_actions is None:
            init_actions = np.random.randint(self.num_actions, size=self.N)

        self.current_actions[:] = init_actions

    def play(self, revision='simultaneous'):
        """
        The method used to proceed the game by one period.

        """
        if revision == 'simultaneous':
            opponent_act_dists = \
                self.adj_matrix.dot(self.current_actions_mixed).toarray()
            best_responses = np.empty(self.N, dtype=int)
            for i, player in enumerate(self.players):
                best_responses[i] = player.best_response(opponent_act_dists[i, :])

            self.current_actions[:] = best_responses

        if revision == 'sequential':
            i = np.random.choice(range(self.N))  # The index of chosen player
            mover = self.players[i]
            opponent_act_dists = \
                self.adj_matrix.dot(self.current_actions_mixed).toarray()
            best_response = mover.best_response(opponent_act_dists[i, :])
            self.current_actions[i] = best_response

    def simulate(self, ts_length, init_actions=None, revision='simultaneous'):
        """
        Return array of ts_length arrays of N actions

        """
        self.set_init_actions(init_actions=init_actions)

        actions_sequence = np.empty([ts_length, self.N])
        for t in range(ts_length):
            actions_sequence[t] = self.current_actions
            self.play(revision=revision)

        return actions_sequence

    def simulate_iter(self, T, init_actions=None, revision='simultaneous'):
        """
        Iterator version of `simulate`

        """
        self.set_init_actions(init_actions=init_actions)

        for t in range(T):
            yield self.current_actions
            self.play(revision=revision)
