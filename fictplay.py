"""
Filename: fictplay.py

Authors: Daisuke Oyama

Fictitious play model.

"""
from __future__ import division

import numpy as np
from game_tools import NormalFormGame, pure2mixed


class FictitiousPlay(object):
    """
    Fictitious play with two players

    Parameters
    ----------
    data : array_like(float) or NormalFormGame


    Attributes
    ----------
    g : NormalFormGame

    players : list(Player)  # tuple

    nums_actions : tuple(int)

    current_beliefs : tuple(ndarray(float, ndim=1))

    """
    def __init__(self, data):
        if isinstance(data, NormalFormGame):
            if data.N != 2:
                raise ValueError('input game must be a two-player game')
            self.g = data
        else:  # data must be array_like
            payoffs = np.asarray(data)
            if not (payoffs.ndim in [2, 3]):
                raise ValueError(
                    'input data must be a square matrix or a bimatrix'
                )
            self.g = NormalFormGame(payoffs)

        self.N = self.g.N  # Must be 2
        self.players = self.g.players
        self.belief_sizes = tuple(
            self.g.nums_actions[1-i] for i in range(self.N)
        )

        # Create instance variable `current_belief` for self.players
        for player, belief_size in zip(self.players, self.belief_sizes):
            player.current_belief = np.empty(belief_size)

        self.current_actions = np.zeros(self.N, dtype=int)
        self.set_init_actions()

        self.step_size = lambda t: 1 / (t+1)

        self.tie_breaking = 'smallest'

    def __repr__(self):
        msg = "Fictitious play for "
        g_repr = self.g.__repr__()
        msg += g_repr
        return msg

    def __str__(self):
        return self.__repr__()

    def set_init_actions(self, init_actions=None):
        if init_actions is None:
            init_actions = np.zeros(self.N, dtype=int)
            for i, n in enumerate(self.g.nums_actions):
                init_actions[i] = np.random.randint(n)
        self.current_actions[:] = init_actions
        for i, player in enumerate(self.players):
            player.current_belief[:] = \
                pure2mixed(self.belief_sizes[i], init_actions[1-i])

    @property
    def current_beliefs(self):
        return tuple(player.current_belief for player in self.players)

    def play(self):
        for i, player in enumerate(self.players):
            self.current_actions[i] = \
                player.best_response(player.current_belief,
                                     tie_breaking=self.tie_breaking)

    def update_beliefs(self, step_size):
        for i, player in enumerate(self.players):
            # x[i] = x[i] + step_size * (a[1-i] - x[i])
            #      = (1-step_size) * x[i] + step_size * a[1-i]
            # where x[i] = player's current_belief,
            #       a[1-i] = opponent's current_action.
            player.current_belief *= 1 - step_size
            player.current_belief[self.current_actions[1-i]] += step_size

    def simulate(self, ts_length, init_actions=None):
        beliefs_array = np.empty((ts_length, sum(self.g.nums_actions)))
        beliefs_iter = self.simulate_iter(ts_length, init_actions)

        for t, beliefs in enumerate(beliefs_iter):
            (beliefs_array[t, :self.belief_sizes[0]],
             beliefs_array[t, self.belief_sizes[0]:]) = beliefs

        return (beliefs_array[:, :self.belief_sizes[0]],
                beliefs_array[:, self.belief_sizes[0]:])

    def simulate_iter(self, ts_length, init_actions=None):
        self.set_init_actions(init_actions)

        for t in range(ts_length):
            yield self.current_beliefs
            self.play()
            self.update_beliefs(self.step_size(t+1))

    def replicate(self, T, num_reps, init_actions=None):
        """
        Returns
        -------
        out : tuple(ndarray(float, ndim=2))

        """
        out = np.empty((num_reps, sum(self.g.nums_actions)))

        for j in range(num_reps):
            beliefs_iter = self.simulate_iter(T+1, init_actions)
            for beliefs in beliefs_iter:
                x = beliefs
            out[j, :self.belief_sizes[0]], out[j, self.belief_sizes[0]:] = x

        return out[:, :self.belief_sizes[0]], out[:, self.belief_sizes[0]:]


class StochasticFictitiousPlay(FictitiousPlay):
    """
    Stochastic fictitious play with two players.

    """
    def __init__(self, data, distribution='extreme', sigma=1.0, epsilon=None):
        FictitiousPlay.__init__(self, data)
        self.n_0, self.n_1 = self.g.nums_actions
        self.n = self.n_0 + self.n_1

        if distribution == 'extreme':  # extreme-value, or gumbel, distribution
            loc = -np.euler_gamma * np.sqrt(6) / np.pi
            scale = np.sqrt(6) / np.pi
            self.payoff_perturbation_dist = \
                lambda size: np.random.gumbel(loc=loc, scale=scale, size=size)
        elif distribution == 'normal':  # normal distribution
            self.payoff_perturbation_dist = np.random.standard_normal
        else:
            raise ValueError("distribution must be 'extreme' or 'normal'")

        self.sigma = sigma

        if epsilon is not None:
            self.step_size = lambda t: epsilon

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_epsilon(self, epsilon):
        self.step_size = lambda t: epsilon

    def play(self):
        """

        """
        n_0, n_1 = self.g.nums_actions
        n = n_0 + n_1
        random_values = self.payoff_perturbation_dist(size=n)
        payoff_perturbations = (random_values[:n_0], random_values[n_0:])

        for i, player in enumerate(self.players):
            self.current_actions[i] = player.best_response(
                player.current_belief,
                tie_breaking=self.tie_breaking,
                payoff_perturbation=payoff_perturbations[i]
            )
