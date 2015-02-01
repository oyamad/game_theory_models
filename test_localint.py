"""
Filename: test_localint.py
Author: Tomohiro Kusano

Tests for localint.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_, raises

from localint import LocalInteraction


class TestLocalInteraction:
    '''Test the methods of LocalInteraction'''

    def setUp(self):
        '''Setup a LocalInteraction instance'''
        # Circle network with 5 players
        adj_matrix = [[0, 1, 0, 0, 1],
                      [1, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 1],
                      [1, 0, 0, 1, 0]]
        # 2x2 coordination game with action 1 risk-dominant
        payoff_matrix = [[4, 0],
                         [2, 3]]
        self.li = LocalInteraction(payoff_matrix, adj_matrix)

    def test_set_init_actions_with_given_init_actions(self):
        self.li.set_init_actions([0, 1, 1, 0, 0])
        assert_array_equal(self.li.current_actions, [0, 1, 1, 0, 0])

    def test_set_init_actions_when_init_actions_None(self):
        self.li.set_init_actions()  # Actions randomly assigned
        ok_(all(
            action in list(range(2)) for action in self.li.current_actions)
            )

    def test_play_when_player_ind_None(self):
        self.li.set_init_actions([1, 0, 0, 0, 0])
        self.li.play()  # All players revise
        assert_array_equal(self.li.current_actions, [0, 1, 0, 0, 1])

    def test_play_when_player_ind_int(self):
        self.li.set_init_actions([1, 0, 0, 0, 0])
        self.li.play(player_ind=1)  # Player 1 revises
        assert_array_equal(self.li.current_actions, [1, 1, 0, 0, 0])

    def test_play_when_player_ind_list(self):
        self.li.set_init_actions([1, 0, 0, 0, 0])
        self.li.play(player_ind=[0, 1, 2])  # Players 0, 1, and 2 revises
        assert_array_equal(self.li.current_actions, [0, 1, 0, 0, 0])

    def test_simulate_with_simultaneous_revision(self):
        assert_array_equal(
            self.li.simulate(ts_length=3, init_actions=[1, 0, 0, 0, 1]),
            [[1, 0, 0, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1]]
            )

    def test_simulate_with_sequential_revison(self):
        np.random.seed(60)
        assert_array_equal(
            self.li.simulate(ts_length=4, init_actions=[1, 0, 0, 0, 1],
                             revision='sequential'),
            [[1, 0, 0, 0, 1],
             [1, 1, 0, 0, 1],
             [1, 1, 1, 0, 1],
             [1, 1, 1, 1, 1]]
            )


# Invalid inputs #

@raises(ValueError)
def test_localint_invalid_input_nonsquare_adj_matrix():
    li = LocalInteraction(payoff_matrix=np.zeros((2, 2)),
                          adj_matrix=np.zeros((2, 3)))


@raises(ValueError)
def test_localint_invalid_input_nonsquare_payoff_matrix():
    li = LocalInteraction(payoff_matrix=np.zeros((2, 3)),
                          adj_matrix=np.zeros((2, 2)))


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
