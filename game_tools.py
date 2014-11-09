r"""
Filename: game_tools.py

Authors: Tomohiro Kusano, Daisuke Oyama

Tools for Game Theory.

Definitions and Basic Concepts
------------------------------

Creating a NormalFormGame
-------------------------

Creating a Player
-----------------

>>> coordination_game_matrix = [[4, 0], [3, 2]]
>>> player = Player(coordination_game_matrix)
>>> player.payoff_array
array([[4, 0],
       [3, 2]])

>>> player = Player((2, 2))
>>> player.payoff_array[0, 0] = 4
>>> player.payoff_array[0, 1] = 0
>>> player.payoff_array[1, 0] = 3
>>> player.payoff_array[1, 1] = 2
>>> player.payoff_array
array([[ 4.,  0.],
       [ 3.,  2.]])

"""
from __future__ import division

import numpy as np


class Player(object):
    """
    Class representing a player in an N-player normal form game.

    Parameters
    ----------
    data : array_like(float)
        If dimension >= 2, it is treated as an array representing the
        player's payoffs. If dimension == 1, it is treated as an array
        containing the numbers of available actions for the player and
        his opponents.

    Attributes
    ----------
    payoff_array : ndarray(float, ndim=N)
        Array representing the players payoffs.

    num_actions : int
        The number of actions available to the player.

    action_sizes : tuple(int)

    Examples
    --------
    >>> P = game_tools.Player([[4,0,6], [3,2,5]])
    >>> P
    Player_2P:
        payoff_matrix: array([[4,0,6],[3,2,5]])
        num_actions: 2L

    """
    def __init__(self, data):
        data = np.asarray(data)

        if data.ndim <= 1:  # data represents action sizes
            # payoff_array filled with zeros
            self.payoff_array = np.zeros(data)
        else:  # data represents a payoff array
            self.payoff_array = data

        self.action_sizes = self.payoff_array.shape
        self.num_actions = self.action_sizes[0]

    def payoff_vector(self, opponents_actions):
        """
        Return an array of payoff values, one for each own action, given
        a profile of the opponents' actions.

        TO BE IMPLEMENTED

        """
        pass

    def is_best_response(self, own_action, opponents_actions):
        """
        Return True if `own_action` is a best response to
        `opponents_actions`.

        """
        payoff_vector = self.payoff_vector(opponents_actions)
        payoff_max = payoff_vector.max()

        if isinstance(own_action, int):
            return np.isclose(payoff_vector[own_action], payoff_max)
        else:
            return np.isclose(np.dot(own_action, payoff_vector), payoff_max)

    def best_response(self, opponents_actions,
                      tie_breaking=True, payoff_perturbations=None):
        """
        Return the best response actions to `opponents_actions`.

        Parameters
        ----------

        Returns
        -------

        """
        br_actions = br_correspondence(opponents_actions, self.payoff_array)

        if tie_breaking:
            return random_choice(br_actions)
        else:
            return br_actions

    def random_choice(self):
        """
        Return a pure action chosen at random from the player's actions.

        Parameters
        ----------

        Returns
        -------

        """
        return random_choice(range(self.num_actions))


class NormalFormGame(object):
    """
    Class representing a two-player normal form game.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """
    def __init__(self, data):
        # data represents a list of Players
        if hasattr(data, '__getitem__') and isinstance(data[0], Player):
            N = len(data)

            # Check that action_sizes are consistent
            action_sizes_0 = data[0].action_sizes
            for i in range(1, N):
                if not (
                    data[i].action_sizes ==
                    tuple(action_sizes_0[j] for j in np.arange(i, i+N) % N)
                ):
                    raise ValueError(
                        'shapes of payoff arrays must be consistent'
                    )

            self.players = list(data)

        # data represents action sizes or a payoff array
        else:
            data = np.asarray(data)

            if data.ndim == 0:  # data represents action size
                # Degenerate game consisting of one player
                N = 1
                self.players = [Player(data)]

            elif data.ndim == 1:  # data represents action sizes
                N = data.size
                # payoff_arrays filled with zeros
                # Payoff values set via __setitem__
                self.players = [
                    Player(data[np.arange(i, i+N) % N]) for i in range(N)
                ]

            else:  # data represents a payoff array
                N = data.ndim - 1
                self.players = [
                    Player(
                        data.take(i, axis=-1).transpose(np.arange(i, i+N) % N)
                    ) for i in range(N)
                ]

        self.N = N  # Number of players

    def __getitem__(self, action_profile):
        if len(action_profile) != self.N:
            raise IndexError

        index = np.asarray(action_profile)
        N = self.N
        payoff_profile = [
            player.payoff_array[tuple(index[np.arange(i, i+N) % N])]
            for i, player in enumerate(self.players)
        ]

        return payoff_profile

    def __setitem__(self, action_profile, payoff_profile):
        """
        TO BE IMPLEMENTED
        """
        pass

    def is_nash(self, action_profile):
        """
        Return True if `action_profile` is a Nash equilibrium.

        """
        action_profile_permed = list(action_profile)

        for i, player in enumerate(self.players):
            own_action = action_profile_permed.pop(0)
            opponents_actions = action_profile_permed

            if not player.is_best_response(own_action, opponents_actions):
                return False

            action_profile_permed.append(own_action)

        return True


def br_correspondence(opponents_actions, payoffs):
    """
    Best response correspondence in pure actions. It returns a list of
    the pure best responses to a profile of N-1 opponents' actions.

    TODO: Revise Docstring.

    Parameters
    ----------
    opponents_actions : array_like(float, ndim=1 or 2) or
                        array_like(int, ndim=1) or scalar(int)
        A profile of N-1 opponents' actions. If N=2, then it must be a
        1-dimensional array_like of floats (in which case it is treated
        as the opponent's mixed action) or a scalar of integer (in which
        case it is treated as the opponent's pure action). If N>2, then
        it must be a 2-dimensional array_like of floats (profile of
        mixed actions) or a 1-dimensional array_like of integers
        (profile of pure actions).

    payoffs : array_like(float, ndim=N)
        An N-dimensional array_like representing the player's payoffs.

    Returns
    -------
    ndarray(int)
        An array of the pure action best responses to opponents_actions.

    """
    payoffs_ndarray = np.asarray(payoffs)
    N = payoffs_ndarray.ndim
    if N == 2:
        if isinstance(opponents_actions, int):
            payoff_vec = payoffs_ndarray[:, opponents_actions]
        else:
            payoff_vec = payoffs_ndarray.dot(opponents_actions)
    else:
        payoff_vec = payoffs_ndarray
        for i in reversed(range(N-1)):
            if isinstance(opponents_actions[i], int):
                payoff_vec = payoff_vec.take(opponents_actions[i], axis=-1)
            else:
                payoff_vec = payoff_vec.dot(opponents_actions[i])

    return np.where(np.isclose(payoff_vec, payoff_vec.max()))[0]


def random_choice(actions):
    """
    Choose an action randomly from given actions.

    Parameters
    ----------
    actions : list(int)
        A list of pure actions represented by nonnegative integers.

    Returns
    -------
    int
        A pure action chosen at random from given actions.

    """
    if len(actions) == 1:
        return actions[0]
    else:
        return np.random.choice(actions)


def pure2mixed(num_actions, action):
    """
    Convert a pure action to the corresponding mixed action.

    Parameters
    ----------
    num_actions : int
        The number of pure actions.

    action : int
        The pure action you want to convert to the corresponding
        mixed action.

    Returns
    -------
    ndarray(int) <- float?
        The corresponding mixed action for the given pure action.

    """
    mixed_action = np.zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
