"""
Filename: test_logitdyn.py
Author: Tomohiro Kusano

Tests for logitdyn.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal, \
    assert_array_almost_equal_nulp
from nose.tools import eq_, ok_, raises

from logitdyn import LogitDynamics
from game_tools import NormalFormGame


class TestLogitDynamics:
    '''Test the methods of LogitDynamics'''

    def setUp(self):
        '''Setup a LogitDynamics instance'''
        # symmetric 2x2 coordination game
        payoff_matrix = [[4, 0],
                         [3, 2]]
        g = NormalFormGame(payoff_matrix)
        self.ld = LogitDynamics(g)

    def test_set_init_actions_with_given_init_actions(self):
        init_actions = (0, 1)
        self.ld.set_init_actions(init_actions)
        assert_array_equal(self.ld.current_actions, init_actions)

    def test_set_init_actions_when_init_action_dist_None(self):
        self.ld.set_init_actions()  # Action dist randomly chosen
        init_actions = self.ld.current_actions

        ok_(all(
            action in list(range(2)) for action in self.ld.current_actions)
            )

    def test_set_choice_probs_with_asymmetric_payoff_matrix(self):
        asym_payoffs = [[(4, 4), (1, 1), (0, 3)], [(3, 0), (1, 1), (2, 2)]]
        g_asym = NormalFormGame(asym_payoffs)
        self.ld = LogitDynamics(g_asym)
        assert_array_almost_equal_nulp(
            self.ld.players[0].logit_choice_cdfs,
            [[1., 1.36787944],
             [1., 2.],
             [0.13533528, 1.13533528]],
            nulp=1000000000
        )

    def test_simulate_seed(self):
        np.random.seed(0)
        seq = self.ld.simulate(ts_length=10, init_actions=(0, 0))
        assert_array_equal(
            seq,
            [[0, 0],
             [0, 0],
             [0, 0],
             [0, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 1]]
        )

    def test_simulate_LLN(self):
        n = 1000
        seq = self.ld.replicate(T=100, num_reps=n)
        count = 0
        for i in range(n):
            if all(seq[i, :] == [1, 1]):
                count += 1
        ratio = count / n
        assert_almost_equal(ratio, 0.61029569, decimal=1)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
