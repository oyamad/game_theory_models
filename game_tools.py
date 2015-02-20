r"""
Filename: game_tools.py

Authors: Tomohiro Kusano, Daisuke Oyama

Tools for Game Theory.

Definitions and Basic Concepts
------------------------------

An :math:`N`-player *normal form game* :math:`g = (I, (A_i)_{i \in I},
(u_i)_{i \in I})` consists of

- the set of *players* :math:`I = \{0, \ldots, N-1\}`,
- the set of *actions* :math:`A_i = \{0, \ldots, n_i-1\}` for each
  player :math:`i \in I`, and
- the *payoff function* :math:`u_i \colon A_i \times A_{i+1} \times
  \cdots \times A_{i+N-1} \to \mathbb{R}` for each player :math:`i \in
  I`,

where :math:`i+j` is understood modulo :math:`N`. Note that we adopt the
convention that the 0-th argument of the payoff function :math:`u_i` is
player :math:`i`'s own action and the :math:`j`-th argument is player
(:math:`i+j`)'s action (modulo :math:`N`). A mixed action for player
:math:`i` is a probability distribution on :math:`A_i` (while an element
of :math:`A_i` is referred to as a pure action). A pure action
:math:`a_i \in A_i` is identified with the mixed action that assigns
probability one to :math:`a_i`. Denote the set of mixed actions of
player :math:`i` by :math:`X_i`. We also denote :math:`A_{-i} = A_{i+1}
\times \cdots \times A_{i+N-1}` and :math:`X_{-i} = X_{i+1} \times
\cdots \times X_{i+N-1}`.

The (pure-action) *best response correspondence* :math:`b_i \colon
X_{-i} \to A_i` for each player :math:`i` is defined by

.. math::

    b_i(x_{-i}) = \{a_i \in A_i \mid
        u_i(a_i, x_{-i}) \geq u_i(a_i', x_{-i})
        \ \forall\,a_i' \in A_i\},

where :math:`u_i(a_i, x_{-i}) = \sum_{a_{-i} \in A_{-i}} u_i(a_i,
a_{-i}) \prod_{j=1}^{N-1} x_{i+j}(a_j)` is the expected payoff to action
:math:`a_i` against mixed actions :math:`x_{-i}`. A profile of mixed
actions :math:`x^* \in X_0 \times \cdots \times X_{N-1}` is a *Nash
equilibrium* if for all :math:`i \in I` and :math:`a_i \in A_i`,

.. math::

    x_i^*(a_i) > 0 \Rightarrow a_i \in b_i(x_{-i}^*),

or equivalently, :math:`x_i^* \cdot v_i(x_{-i}^*) \geq x_i \cdot
v_i(x_{-i}^*)` for all :math:`x_i \in X_i`, where :math:`v_i(x_{-i})` is
the vector of player :math:`i`'s payoffs when the opponent players play
mixed actions :math:`x_{-i}`.

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
        See Parameters.

    num_actions : scalar(int)
        The number of actions available to the player.

    num_opponents : scalar(int)
        The number of opponent players.

    action_sizes : tuple(int)
        Tuple of length N representing the numbers of actions of the
        players.

    """
    def __init__(self, payoff_array):
        self.payoff_array = np.asarray(payoff_array)

        if self.payoff_array.ndim == 0:
            raise ValueError('payoff_array must be an array_like')

        self.num_opponents = self.payoff_array.ndim - 1
        self.action_sizes = self.payoff_array.shape
        self.num_actions = self.action_sizes[0]

        self.tol = 1e-8

    def __repr__(self):
        N = self.num_opponents + 1
        msg = "Player in a {0}-player normal form game".format(N)

        if N == 2:
            matrix_str = \
                " with payoff matrix:\n{0}".format(self.payoff_array.tolist())
            msg += matrix_str

        return msg

    def __str__(self):
        return self.__repr__()

    def payoff_vector(self, opponents_actions):
        """
        Return an array of payoff values, one for each own action, given
        a profile of the opponents' actions.

        Parameters
        ----------
        opponents_actions : see `best_response`.

        Returns
        -------
        payoff_vector : ndarray(float, ndim=1)
            An array representing the player's payoff vector given the
            profile of the opponents' actions.

        """
        def reduce_last_player(payoff_array, action):
            """
            Given `payoff_array` with ndim=M, return the payoff array
            with ndim=M-1 fixing the last player's action to be `action`.

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
        own_action : scalar(int) or array_like(float, ndim=1)
            An integer representing a pure action, or an array of floats
            representing a mixed action.

        opponents_actions : see `best_response`

        Returns
        -------
        bool
            True if `own_action` is a best response to
            `opponents_actions`; False otherwise.

        """
        payoff_vector = self.payoff_vector(opponents_actions)
        payoff_max = payoff_vector.max()

        if isinstance(own_action, int):
            return payoff_vector[own_action] >= payoff_max - self.tol
        else:
            return np.dot(own_action, payoff_vector) >= payoff_max - self.tol

    def best_response(self, opponents_actions, tie_breaking='smallest',
                      payoff_perturbations=None):
        """
        Return the best response action(s) to `opponents_actions`.

        Parameters
        ----------
        opponents_actions : array_like(int or array_like(float)) or
                            array_like(int, ndim=1) or scalar(int)
            A profile of N-1 opponents' actions. If N=2, then it must be
            a 1-dimensional array of floats (in which case it is treated
            as the opponent's mixed action) or a scalar of integer (in
            which case it is treated as the opponent's pure action). If
            N>2, then it must be an array of N-1 objects, where each
            object must be an integer (pure action) or an array of
            floats (mixed action).

        tie_breaking : {'smallest', 'random', False}
            Controls how to break a tie (see Returns for details).

        Returns
        -------
        scalar(int) or ndarray(int, ndim=1)
            If tie_breaking=False, returns an array containing all the
            best response pure actions. If tie_breaking='smallest',
            returns the best response action with the smallest index; if
            tie_breaking='random', returns an action randomly chosen
            from the best response actions.

        """
        payoff_vector = self.payoff_vector(opponents_actions)

        if tie_breaking == 'smallest':
            best_response = np.argmax(payoff_vector)
            return best_response
        else:
            best_responses = \
                np.where(payoff_vector >= payoff_vector.max() - self.tol)[0]
            if tie_breaking == 'random':
                return random_choice(best_responses)
            elif tie_breaking is False:
                return best_responses
            else:
                msg = "tie_breaking must be one of 'smallest', 'random' or False"
                raise ValueError(msg)

    def random_choice(self, actions=None):
        """
        Return a pure action chosen randomly from `actions`.

        Parameters
        ----------
        actions : array_like(int)
            An array of integers representing pure actions.

        Returns
        -------
        scalar(int)
            If `actions` is given, returns an integer representing a
            pure action chosen randomly from `actions`; if not, an
            action is chosen randomly from the player's all actions.

        """
        if actions:
            return random_choice(actions)
        else:
            return random_choice(range(self.num_actions))


class NormalFormGame(object):
    """
    Class representing an N-player normal form game.

    Parameters
    ----------
    data : array_like(Player) or array_like(int, ndim=1) or
           array_like(float, ndim=2 or N+1)
        Data to initialize a NormalFormGame. `data` may be an array of
        Players, in which case the shapes of the Players' payoff arrays
        must be consistent. If `data` is an array of N integers, then
        these integers are treated as the numbers of actions of the N
        players and a NormalFormGame is created consisting of payoffs
        all 0 with `data[i]` actions for each player `i`. `data` may
        also be an (N+1)-dimensional array representing payoff profiles.
        If `data` is a square matrix (2-dimensional array), then the
        game will be a symmetric two-player game where the payoff matrix
        of each player is given by the input matrix.

    Attributes
    ----------
    players : tuple(Player)
        Tuple of the Player instances of the game.

    N : scalar(int)
        The number of players.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    """
    def __init__(self, data):
        # data represents an array_like of Players
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
                self.players = (Player(np.zeros(data)),)

            elif data.ndim == 1:  # data represents action sizes
                N = data.size
                # N instances of Player created
                # with payoff_arrays filled with zeros
                # Payoff values set via __setitem__
                self.players = tuple(
                    Player(np.zeros(data[np.arange(i, i+N) % N]))
                    for i in range(N)
                )

            elif data.ndim == 2 and data.shape[1] >= 2:
                # data represents a payoff array for symmetric two-player game
                # Number of actions must be >= 2
                if data.shape[0] != data.shape[1]:
                    raise ValueError(
                        'symmetric two-player game must be represented ' +
                        'by a square matrix'
                    )
                N = 2
                self.players = tuple(Player(data) for i in range(N))

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
                self.players = tuple(
                    Player(
                        data.take(i, axis=-1).transpose(np.arange(i, i+N) % N)
                    ) for i in range(N)
                )

        self.N = N  # Number of players
        self.nums_actions = tuple(
            player.num_actions for player in self.players
        )

    def __repr__(self):
        msg = "{0}-player NormalFormGame".format(self.N)

        if self.N == 2:
            P0 = self.players[0].payoff_array
            P1 = self.players[1].payoff_array
            bimatrix = np.dstack((P0, P1.T))
            bimatrix_str = \
                " with payoff bimatrix:\n{0}".format(bimatrix.tolist())
            msg += bimatrix_str

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
        action_profile : array_like(int or array_like(float))
            An array of N objects, where each object must be an integer 
            (pure action) or an array of floats (mixed action).

        Returns
        -------
        bool
            True if `action_profile` is a Nash equilibrium; False
            otherwise.

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
    Choose an action randomly from `actions`.

    Parameters
    ----------
    actions : array_like(int)
        An array of pure actions represented by nonnegative integers.

    Returns
    -------
    scalar(int)
        A pure action randomly chosen from `actions`.

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
    num_actions : scalar(int)
        The number of the pure actions (= the length of a mixed action).

    action : scalar(int)
        The pure action to convert to the corresponding mixed action.

    Returns
    -------
    ndarray(float, ndim=1)
        The mixed action representation of the given pure action.

    """
    mixed_action = np.zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
