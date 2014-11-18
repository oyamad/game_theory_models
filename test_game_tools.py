"""
Filename: test_game_tools.py
Author: Daisuke Oyama

Tests for game_tools.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal
import nose
from nose.tools import eq_, ok_, raises

from game_tools import Player, NormalFormGame


# Player #

class TestPlayer_1opponent:
    """Test the methods of Player with one opponent player"""

    def setUp(self):
        """Setup a Player instance"""
        coordination_game_matrix = [[4, 0], [3, 2]]
        self.player = Player(coordination_game_matrix)

    def test_best_response_against_pure(self):
        eq_(self.player.best_response(1), 1)

    def test_best_response_against_mixed(self):
        eq_(self.player.best_response([1/2, 1/2]), 1)

    def test_best_response_list_when_tie(self):
        """best_response with tie_breaking=False"""
        assert_array_equal(
            sorted(self.player.best_response([2/3, 1/3], tie_breaking=False)),
            sorted([0, 1])
        )

    def test_best_response_with_tie_breaking(self):
        """best_response with tie_breaking=True (default)"""
        ok_(self.player.best_response([2/3, 1/3]) in [0, 1])

    def test_is_best_response_against_pure(self):
        ok_(self.player.is_best_response(0, 0))

    def test_is_best_response_against_mixed(self):
        ok_(self.player.is_best_response([1/2, 1/2], [2/3, 1/3]))


class TestPlayer_2opponents:
    """Test the methods of Player with two opponent players"""

    def setUp(self):
        """Setup a Player instance"""
        payoffs_2opponents = [[[3, 6],
                               [4, 2]],
                              [[1, 0],
                               [5, 7]]]
        self.player = Player(payoffs_2opponents)

    def test_payoff_vector_against_pure(self):
        assert_array_equal(self.player.payoff_vector((0, 1)), [6, 0])

    def test_is_best_response_against_pure(self):
        ok_(not self.player.is_best_response(0, (1, 0)))

    def test_best_response_against_pure(self):
        eq_(self.player.best_response((1, 1)), 1)

    def test_best_response_list_when_tie(self):
        """best_response against a mixed action profile with tie_breaking=False
        """
        assert_array_equal(
            sorted(self.player.best_response(([3/7, 4/7], [1/2, 1/2]),
                                             tie_breaking=False)),
            sorted([0, 1])
        )


# NormalFormGame #

class TestNormalFormGame_Sym2p:
    """Test the methods of NormalFormGame with symmetric two players"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        coordination_game_matrix = [[4, 0], [3, 2]]
        self.g = NormalFormGame(coordination_game_matrix)

    def test_getitem(self):
        assert_array_equal(self.g[0, 1], [0, 3])

    def test_is_nash_pure(self):
        ok_(self.g.is_nash((0, 0)))

    def test_is_nash_mixed(self):
        ok_(self.g.is_nash(([2/3, 1/3], [2/3, 1/3])))


class TestNormalFormGame_Asym2p:
    """Test the methods of NormalFormGame with asymmetric two players"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        matching_pennies_bimatrix = [[( 1, -1), (-1,  1)],
                                     [(-1,  1), ( 1, -1)]]
        self.g = NormalFormGame(matching_pennies_bimatrix)

    def test_getitem(self):
        assert_array_equal(self.g[1, 0], [-1, 1])

    def test_is_nash_against_pure(self):
        ok_(not self.g.is_nash((0, 0)))

    def test_is_nash_against_mixed(self):
        ok_(self.g.is_nash(([1/2, 1/2], [1/2, 1/2])))


class TestNormalFormGame_3p:
    """Test the methods of NormalFormGame with three players"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        payoffs_2opponents = [[[3, 6],
                               [4, 2]],
                              [[1, 0],
                               [5, 7]]]
        player = Player(payoffs_2opponents)
        self.g = NormalFormGame([player for i in range(3)])

    def test_getitem(self):
        assert_array_equal(self.g[0, 0, 1], [6, 4, 1])

    def test_is_nash_pure(self):
        ok_(self.g.is_nash((0, 0, 0)))
        ok_(not self.g.is_nash((0, 0, 1)))

    def test_is_nash_mixed(self):
        p = (1 + np.sqrt(65)) / 16
        ok_(self.g.is_nash(([1 - p, p], [1 - p, p], [1 - p, p])))


def test_normalformgame_input_action_sizes():
    g = NormalFormGame((2, 3, 4))

    eq_(g.N, 3)  # Number of players

    assert_array_equal(
        g.players[0].payoff_array,
        np.zeros((2, 3, 4))
    )
    assert_array_equal(
        g.players[1].payoff_array,
        np.zeros((3, 4, 2))
    )
    assert_array_equal(
        g.players[2].payoff_array,
        np.zeros((4, 2, 3))
    )


# Degenerate cases with one player #

class TestPlayer_0opponents:
    """Test for degenerate Player with no opponent player"""

    def setUp(self):
        """Setup a Player instance"""
        payoffs = [0, 1]
        self.player = Player(payoffs)

    def test_payoff_vector(self):
        """Degenerate player: payoff_vector"""
        assert_array_equal(self.player.payoff_vector(None), [0, 1])

    def test_is_best_response(self):
        """Degenerate player: is_best_response"""
        ok_(self.player.is_best_response(1, None))

    def test_best_response(self):
        """Degenerate player: best_response"""
        eq_(self.player.best_response(None), 1)


class TestNormalFormGame_1p:
    """Test for degenerate NormalFormGame with a single player"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        data = [[0], [1], [1]]
        self.g = NormalFormGame(data)

    def test_construction(self):
        """Degenerate game: construction"""
        ok_(self.g.N == 1)
        assert_array_equal(self.g.players[0].payoff_array, [0, 1, 1])

    def test_is_nash_pure(self):
        """Degenerate game: is_nash with pure action"""
        ok_(self.g.is_nash((1,)))
        ok_(not self.g.is_nash((0,)))

    def test_is_nash_mixed(self):
        """Degenerate game: is_nash with mixed action"""
        ok_(self.g.is_nash(([0, 1/2, 1/2],)))


def test_normalformgame_input_action_sizes_1p():
    g = NormalFormGame(2)

    eq_(g.N, 1)  # Number of players

    assert_array_equal(
        g.players[0].payoff_array,
        np.zeros(2)
    )


# Invalid inputs #

@raises(ValueError)
def test_normalformgame_invalid_input_players_shape_inconsistent():
    p0 = Player(np.zeros((2, 3)))
    p1 = Player(np.zeros((2, 3)))
    g = NormalFormGame([p0, p1])


@raises(ValueError)
def test_normalformgame_invalid_input_players_num_inconsistent():
    p0 = Player(np.zeros((2, 2, 2)))
    p1 = Player(np.zeros((2, 2, 2)))
    g = NormalFormGame([p0, p1])


@raises(ValueError)
def test_normalformgame_invalid_input_nosquare_matrix():
    g = NormalFormGame(np.zeros((2, 3)))


@raises(ValueError)
def test_normalformgame_invalid_input_payoff_profiles():
    g = NormalFormGame(np.zeros((2, 2, 1)))


if __name__ == '__main__':
    import sys

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
