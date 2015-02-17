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

>>> g = NormalFormGame((2, 2))
>>> g.players[0].payoff_array
array([[ 0.,  0.],
       [ 0.,  0.]])
>>> g[0, 0]
[0.0, 0.0]
>>> g[0, 0] = (0, 10)
>>> g[0, 1] = (0, 10)
>>> g[1, 0] = (3, 5)
>>> g[1, 1] = (-2, 0)
>>> g.players[0].payoff_array
array([[ 0.,  0.],
       [ 3., -2.]])
>>> g.players[1].payoff_array
array([[ 10.,   5.],
       [ 10.,   0.]])

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
        Tuple representing the number of actions of each player.

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

        self.tol = 1e-8

    def payoff_vector(self, opponents_actions):
        """
        Return an array of payoff values, one for each own action, given
        a profile of the opponents' actions.

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

        Returns
        -------
        payoff_vector : ndarray(float, ndim=1)
            If there are at least one opponents (self.num_opponents >= 1), 
            returns an array representing the player's payoffs given the 
            profile of the opponents' actions;
            if there are no opponents (self.num_opponents == 0),
            returns the original own payoffs.

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

        Parameters
        ----------
        own_action : int or array_like(float, ndim=1)
            If int, represents a pure action;
            if array_like, represents a mixed action.

        opponents_actions : array_like(float, ndim=1 or 2) or
                            array_like(int, ndim=1) or scalar(int)

        Returns
        -------
        bool

        """
        payoff_vector = self.payoff_vector(opponents_actions)
        payoff_max = payoff_vector.max()

        if isinstance(own_action, int):
            return payoff_vector[own_action] >= payoff_max - self.tol
        else:
            return np.dot(own_action, payoff_vector) >= payoff_max - self.tol

    def best_response(self, opponents_actions,
                      tie_breaking='first', payoff_perturbations=None):
        """
        Return the best response action(s) to `opponents_actions`.

        TODO: Revise Docstring.

        Parameters
        ----------
        opponents_actions : array_like(float, ndim=1 or 2) or
                            array_like(int, ndim=1) or scalar(int)

        tie_breaking : bool

        Returns
        -------
        int or ndarray(int, ndim=1)
            If tie_breaking='first' or 'random', returns an integer 
            representing a best response pure action (when there are more
            than one best responses, 'first' chooses the smallest action 
            and 'random' chooses one action randomly);
            if tie_breaking=False, returns an array of all the best
            response pure actions.

        """
        payoff_vector = self.payoff_vector(opponents_actions)

        if tie_breaking == 'first':
            best_responses = np.argmax(payoff_vector)
            return best_responses
        else:
            best_responses = \
                np.where(payoff_vector >= payoff_vector.max() - self.tol)[0]
            if tie_breaking == 'random':
                return random_choice(best_responses)
            elif tie_breaking is False:
                return best_responses
            else:
                msg = "tie_breaking must be one of 'first', 'random' or False"
                raise ValueError(msg)

    def random_choice(self, actions=None):
        """
        Return a pure action chosen at random from a given player's actions.

        Parameters
        ----------
        actions : list(int)

        Returns
        -------
        int
            If `actions` is given, returns an integer representing a pure
            action chosen randomly from the actions;
            if not, the integer is chosen randomly from the player's all 
            actions.

        """
        if actions:
            return random_choice(actions)
        else:
            return random_choice(range(self.num_actions))


class NormalFormGame(object):
    """
    Class representing a two-player normal form game.

    Parameters
    ----------
    data : tuple(?) or int or
           array_like(float, ndim=1 or 2 or N+1)

    Attributes
    ----------
    players : tuple(?)
        Tuple representing the players of the game. 

    N : int
        The number of players.

    nums_actions : tuple(int)
        Tuple representing the number of actions, one for each players.

    Examples
    --------
    >>> g = game_tools.NormalFormGame([[4,0], [3,2]])
    >>> g
    Coordination Game:
        players: 
        N: 2
        nums_actions: (2L, 2L)

    """
    def __init__(self, data):
        # data represents a tuple of Players
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

            self.players = tuple(data)

        # data represents action sizes or a payoff array
        else:
            data = np.asarray(data)

            if data.ndim == 0:  # data represents action size
                # Degenerate game consisting of one player
                N = 1
                self.players = tuple([Player(np.zeros(data))])

            elif data.ndim == 1:  # data represents action sizes
                N = data.size
                # N instances of Player created
                # with payoff_arrays filled with zeros
                # Payoff values set via __setitem__
                self.players = tuple([
                    Player(np.zeros(data[np.arange(i, i+N) % N]))
                    for i in range(N)
                ])

            elif data.ndim == 2 and data.shape[1] >= 2:
                # data represents a payoff array for symmetric two-player game
                # Number of actions must be >= 2
                if data.shape[0] != data.shape[1]:
                    raise ValueError(
                        'symmetric two-player game must be represented ' +
                        'by a square matrix'
                    )
                N = 2
                self.players = tuple([Player(data) for i in range(N)])

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
                self.players = tuple([
                    Player(
                        data.take(i, axis=-1).transpose(np.arange(i, i+N) % N)
                    ) for i in range(N)
                ])

        self.N = N  # Number of players
        self.nums_actions = tuple([player.num_actions for player in self.players])

    def __repr__(self):
        msg = "N = {0}".format(self.N)
        if self.N == 2:
            P0 = self.players[0].payoff_array
            P1 = self.players[1].payoff_array
            bimatrix = np.dstack((P0, P1.T))
            return msg + "\nPayoff Matrix:\n{0}".format(bimatrix)
        else:
            return msg

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, action_profile):
        try:
            if len(action_profile) != self.N:
                raise IndexError('index must be of length N')
        except TypeError:
            raise TypeError('index must be a tuple')

        index = np.asarray(action_profile)
        N = self.N
        payoff_profile = [
            player.payoff_array[tuple(index[np.arange(i, i+N) % N])]
            for i, player in enumerate(self.players)
        ]

        return payoff_profile

    def __setitem__(self, action_profile, payoff_profile):
        try:
            if len(action_profile) != self.N:
                raise IndexError('index must be of length N')
        except TypeError:
            raise TypeError('index must be a tuple')

        try:
            if len(payoff_profile) != self.N:
                raise ValueError('value must be an array_like of length N')
        except TypeError:
            raise TypeError('value must be a tuple')

        index = np.asarray(action_profile)
        N = self.N
        for i, player in enumerate(self.players):
            player.payoff_array[tuple(index[np.arange(i, i+N) % N])] = \
                payoff_profile[i]

    def is_nash(self, action_profile):
        """
        Return True if `action_profile` is a Nash equilibrium.

        Parameters
        ----------
        action_profile : list(int) or array_like(ndim=2)

        Returns
        -------
        bool

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
