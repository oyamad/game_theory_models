"""
Filename: test_brd.py
Author: Daisuke Oyama

Tests for brd.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_, raises

from brd import BRD


class TestBRD:
    '''Test the methods of BRD'''

    def setUp(self):
        '''Setup a BRD instance'''
        # 2x2 coordination game with action 1 risk-dominant
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4  # 4 players
        self.brd = BRD(payoff_matrix, self.N)

    def test_set_init_action_dist_with_given_init_action_dist(self):
        self.brd.set_init_action_dist([1, 3])
        assert_array_equal(self.brd.current_action_dist, [1, 3])

    def test_set_init_action_dist_when_init_action_dist_None(self):
        self.brd.set_init_action_dist()  # Action dist randomly chosen
        ok_(all(self.brd.current_action_dist >= 0))
        ok_(self.brd.current_action_dist.sum() == self.N)

    def test_play(self):
        self.brd.set_init_action_dist([2, 2])
        self.brd.play(current_action=1)  # Player playing 1 revises
        ok_(np.array_equal(self.brd.current_action_dist, [3, 1]) or
            np.array_equal(self.brd.current_action_dist, [2, 2]))

    def test_simulate_rest_point(self):
        assert_array_equal(
            self.brd.simulate(ts_length=3, init_action_dist=[4, 0]),
            [[4, 0],
             [4, 0],
             [4, 0]]
            )

    def test_simulate(self):
        np.random.seed(22)
        assert_array_equal(
            self.brd.simulate(ts_length=3, init_action_dist=[2, 2]),
            [[2, 2],
             [1, 3],
             [0, 4]]
            )


# Invalid inputs #

@raises(ValueError)
def test_brd_invalid_input_nonsquare_payoff_matrix():
    brd = BRD(payoff_matrix=np.zeros((2, 3)), N=5)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
