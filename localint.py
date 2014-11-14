"""
Filename: localint.py

Author: Daisuke Oyama

Local interaction model.

"""
from __future__ import division

import numpy as np
from scipy import sparse
from game_tools import Player


class LocalInteraction(object):
    """
    Class representing "Local Interaction Model."
    This can handle N action games on any network.
    
    Parameters
    ----------
    payoff_matrix: array_like(float, ndim=2)
        The payoff matrix of the game played in each interaction.

    adj_matrix : array_like(int, ndim=2)
        The adjacency matrix of the unweighted network to be simulated.


    Attributes
    ----------
    Players : list(player(payoff_matrix))
        The list consisting of all players with the given payoff matrix.
        Players are represented by "Player" instances from "game_tools."

    adj_matrix : scipy.sparse.csr.csr_matrix(float, ndim=2) <- int?
        The adjancency matrix of the network in sparse matrix form.

    N : int
        The Number of players.

    num_actions : int
        The number of actions

    current_actions_mixed : scipy.sparse.csr.csr_matrix(int, ndim=2)
        (N)*(num_actions) matrix. The ij element is determined by the rule:  
        "If Player i is taking action j, it is 1. Otherwise, 0." 

    current_actions : ndarray(int, ndim=1)
        This array represents which action each player is taking.

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
            n = np.random.choice(range(self.N)) # The index of chosen player
            mover = self.players[n]
            opponent_act_dists = \
                self.adj_matrix.dot(self.current_actions_mixed).toarray()
            best_response = mover.best_response(opponent_act_dists[n, :])
            self.current_actions[n] = best_response

    def simulate(self, T, init_actions=None, revision='simultaneous'):
        """
        Return array of T arrays of N actions

        """
        self.set_init_actions(init_actions=init_actions)

        history_of_action_profiles = np.empty([T, self.N])
        for i in range(T):
            history_of_action_profiles[i] = self.current_actions
            self.play(revision=revision)

        return history_of_action_profiles

    def simulate_gen(self, T, init_actions=None, revision='simultaneous'):
        """
        Generator version of `simulate`

        """
        self.set_init_actions(init_actions=init_actions)

        for t in range(T):
            yield self.current_actions
            self.play(revision=revision)
