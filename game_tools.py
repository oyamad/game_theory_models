r"""
Filename: game_tools.py

Authors: Tomohiro Kusano, Daisuke Oyama

Tools for Game Theory.

Definitions and Basic Concepts
------------------------------

Creating a NormalFormGame
-------------------------

>>> matching_pennies_bimatrix = [[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]]
>>> g = NormalFormGame(matching_pennies_bimatrix)
>>> g.players[0].payoff_array
array([[ 1, -1],
       [-1,  1]])
>>> g.players[1].payoff_array
array([[-1,  1],
       [ 1, -1]])

>>> g = NormalFormGame((2, 3, 4))
>>> g.players[0].payoff_array
array([[[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]],

       [[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]]])
>>> g[0, 0, 0]
[0.0, 0.0, 0.0]

Creating a Player
-----------------

>>> coordination_game_matrix = [[4, 0], [3, 2]]
>>> player = Player(coordination_game_matrix)
>>> player.payoff_array
array([[4, 0],
       [3, 2]])

"""
from __future__ import division

import numpy as np


class Player(object):
    """
    Class representing a player in an N-player normal form game.

    Parameters
    ----------
    payoff_array : array_like(float)
        Array representing the player's payoff function, where
        payoff_array[a_0, a_1, ..., a_{N-1}] is the payoff to the player
        when the player plays action a_0 while his N-1 opponents play
        action a_1, ..., a_{N-1}, respectively.

    Attributes
    ----------
    payoff_array : ndarray(float, ndim=N)
        Array representing the players payoffs.

    num_actions : int
        The number of actions available to the player.

    num_opponents : int
        The number of opponent players.

    action_sizes : tuple(int)

    Examples
    --------
    >>> P = game_tools.Player([[4,0,6], [3,2,5]])
    >>> P
    Player_2P:
        payoff_matrix: array([[4,0,6],[3,2,5]])
        num_actions: 2L

    """
    def __init__(self, payoff_array):
        self.payoff_array = np.asarray(payoff_array)

        if self.payoff_array.ndim == 0:
            raise ValueError('payoff_array must be an array_like')

        self.num_opponents = self.payoff_array.ndim - 1
        self.action_sizes = self.payoff_array.shape
        self.num_actions = self.action_sizes[0]

    def payoff_vector(self, opponents_actions):
        """
        Return an array of payoff values, one for each own action, given
        a profile of the opponents' actions.

        """
        def reduce_last_player(payoff_array, action):
            """
            Given `payoff_array` with ndim=M, return the payoff array
            with ndim=M-1 fixing the last player's action to be `action`
            """
            if isinstance(action, int):  # pure action
                return payoff_array.take(action, axis=-1)
            else:  # mixed action
                return payoff_array.dot(action)

        if self.num_opponents == 1:
            payoff_vector = \
                reduce_last_player(self.payoff_array, opponents_actions)
        elif self.num_opponents >= 2:
            payoff_vector = self.payoff_array
            for i in reversed(range(self.num_opponents)):
                payoff_vector = \
                    reduce_last_player(payoff_vector, opponents_actions[i])
        else:  # Degenerate case with self.num_opponents == 0
            payoff_vector = self.payoff_array

        return payoff_vector

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
        Return the best response action(s) to `opponents_actions`.

        TODO: Revise Docstring.

        Parameters
        ----------
        opponents_actions : array_like(float, ndim=1 or 2) or
                            array_like(int, ndim=1) or scalar(int)
            A profile of N-1 opponents' actions. If N=2, then it must be
            a 1-dimensional array_like of floats (in which case it is
            treated as the opponent's mixed action) or a scalar of
            integer (in which case it is treated as the opponent's pure
            action). If N>2, then it must be a 2-dimensional array_like
            of floats (profile of mixed actions) or a 1-dimensional
            array_like of integers (profile of pure actions).

        tie_breaking : bool

        Returns
        -------
        int or ndarray(int, ndim=1)
            If tie_breaking=True, returns an integer representing a best
            response pure action (when there are more than one best
            responses, one action is randomly chosen);
            if tie_breaking=False, returns an array of all the best
            response pure actions.

        """
        payoff_vector = self.payoff_vector(opponents_actions)
        best_responses = \
            np.where(np.isclose(payoff_vector, payoff_vector.max()))[0]

        if tie_breaking:
            return random_choice(best_responses)
        else:
            return best_responses

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
                action_sizes = data[i].action_sizes
                if not (
                    len(action_sizes) == N and
                    action_sizes ==
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
                self.players = [Player(np.zeros(data))]

            elif data.ndim == 1:  # data represents action sizes
                N = data.size
                # N instances of Player created
                # with payoff_arrays filled with zeros
                # Payoff values set via __setitem__
                self.players = [
                    Player(np.zeros(data[np.arange(i, i+N) % N]))
                    for i in range(N)
                ]

            elif data.ndim == 2 and data.shape[1] >= 2:
                # data represents a payoff array for symmetric two-player game
                # Number of actions must be >= 2
                if data.shape[0] != data.shape[1]:
                    raise ValueError(
                        'symmetric two-player game must be represented ' +
                        'by a square matrix'
                    )
                N = 2
                self.players = [Player(data) for i in range(N)]

            else:  # data represents a payoff array
                # data must be of shape (n_0, ..., n_{N-1}, N),
                # where n_i is the number of actions available to player i,
                # and the last axis contains the payoff profile
                N = data.ndim - 1
                if data.shape[-1] != N:
                    raise ValueError(
                        'size of innermost array must be equal to ' +
                        'the number of players'
                    )
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
        if self.N == 2:
            for i, player in enumerate(self.players):
                own_action, opponent_action = \
                    action_profile[i], action_profile[1-i]
                if not player.is_best_response(own_action, opponent_action):
                    return False

        elif self.N >= 3:
            action_profile = np.asarray(action_profile)
            N = self.N

            for i, player in enumerate(self.players):
                own_action = action_profile[i]
                opponents_actions = action_profile[np.arange(i+1, i+N) % N]

                if not player.is_best_response(own_action, opponents_actions):
                    return False

        else:  # Degenerate case with self.N == 1
            if not self.players[0].is_best_response(action_profile[0], None):
                return False

        return True


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
