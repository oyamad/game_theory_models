"""
Filename: test_fictplay.py
Author: Daisuke Oyama

Tests for fictplay.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp
from nose.tools import eq_, ok_, raises

from fictplay import FictitiousPlay
from game_tools import NormalFormGame


class TestFictitiousPlay_square_matrix:
    '''Test the methods of FictitiousPlay with square matrix'''

    def setUp(self):
        '''Setup a FictitiousPlay instance'''
        # symmetric 2x2 coordination game
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.fp = FictitiousPlay(payoff_matrix)

    def test_set_init_actions_with_given_init_actions(self):
        init_actions = (0, 1)

        self.fp.set_init_actions(init_actions)
        assert_array_equal(self.fp.current_actions, init_actions)

        for i, current_belief in enumerate(self.fp.current_beliefs):
            ok_(current_belief[init_actions[1-i]] == 1 and
                current_belief.sum() == 1)

    def test_set_init_actions_when_init_action_dist_None(self):
        self.fp.set_init_actions()  # Action dist randomly chosen
        init_actions = self.fp.current_actions

        for i, current_belief in enumerate(self.fp.current_beliefs):
            ok_(current_belief[init_actions[1-i]] == 1 and
                current_belief.sum() == 1)

    def test_play(self):
        init_actions = (0, 1)
        best_responses = (1, 0)
        self.fp.set_init_actions(init_actions)
        self.fp.play()
        assert_array_equal(self.fp.current_actions, best_responses)

    def test_simulate_rest_point(self):
        beliefs_sequence = \
            self.fp.simulate(ts_length=3, init_actions=(0, 0))
        assert_array_equal(
            beliefs_sequence[0],
            [[1, 0],
             [1, 0],
             [1, 0]]
            )

    def test_simulate(self):
        beliefs_sequence = \
            self.fp.simulate(ts_length=3, init_actions=(0, 1))
        # played actions: (0, 1), (1, 0), (0, 1)
        assert_array_almost_equal_nulp(
            beliefs_sequence[0],
            [[0, 1],
             [1/2, 1/2],
             [1/3, 2/3]]
            )


class TestFictitiousPlay_bimatrix:
    '''Test the methods of FictitiousPlay with bimatrix'''

    def setUp(self):
        '''Setup a FictitiousPlay instance'''
        payoff_bimatrix = np.zeros((2, 3, 2))  # 2 x 3 game
        g = NormalFormGame(payoff_bimatrix)
        self.fp = FictitiousPlay(g)

    def test_set_init_actions_with_given_init_actions(self):
        init_actions = (0, 2)

        self.fp.set_init_actions(init_actions)
        assert_array_equal(self.fp.current_actions, init_actions)

        for i, current_belief in enumerate(self.fp.current_beliefs):
            ok_(current_belief[init_actions[1-i]] == 1 and
                current_belief.sum() == 1)

    def test_set_init_actions_when_init_action_dist_None(self):
        self.fp.set_init_actions()  # Action dist randomly chosen
        init_actions = self.fp.current_actions

        for i, current_belief in enumerate(self.fp.current_beliefs):
            ok_(current_belief[init_actions[1-i]] == 1 and
                current_belief.sum() == 1)


# Invalid inputs #

@raises(ValueError)
def test_fp_invalid_input():
    fp = FictitiousPlay(np.zeros((2, 3, 4, 3)))  # three-player game


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
