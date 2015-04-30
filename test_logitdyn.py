"""
Filename: test_logitdyn.py
Author: Tomohiro Kusano

Tests for logitdyn.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp
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
        beta = 4.0
        g = NormalFormGame(payoff_matrix)
        self.ld = LogitDynamics(g, beta=beta)

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

    def test_simulate_seed(self):
        np.random.seed(291)
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

    def test_simulate_lln(self):
        n = 100
        T = 1000
        seq = self.ld.replicate(T=T, num_reps=n)
        count = 0
        for i in range(n):
            if all(seq[i, :] == [1, 1]):
                count += 1
        frequency = count / n

        # 0.981367209 = prob that the stationary distribution assigns to [1, 1]
        ok_(np.abs(frequency-0.981367209) < 0.05)


def test_set_choice_probs_with_asymmetric_payoff_matrix():
    bimatrix = np.array([[(4, 4), (1, 1), (0, 3)],
                             [(3, 0), (1, 1), (2, 2)]])
    beta = 1.0
    g = NormalFormGame(bimatrix)
    ld = LogitDynamics(g, beta=beta)

    # (Normalized) CDFs of logit choice
    cdfs = np.ones((bimatrix.shape[1], bimatrix.shape[0]))
    cdfs[:, 0] = 1 / (1 + np.exp(beta*(bimatrix[1, :, 0]-bimatrix[0, :, 0])))

    # self.ld.players[0].logit_choice_cdfs: unnormalized
    cdfs_computed = ld.players[0].logit_choice_cdfs
    cdfs_computed = cdfs_computed / cdfs_computed[..., [-1]]  # Normalized

    assert_array_almost_equal_nulp(cdfs_computed, cdfs)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
