"""
Filename: test_logitdyn.py
Author: Tomohiro Kusano

Tests for logitdyn.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
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

    def test_play_seed(self):
        init_actions = (0, 1)
        self.ld.set_init_actions(init_actions)
        np.random.seed(0)
        self.ld.play(0)
        actions1 = np.array(self.ld.current_actions)
        self.ld.set_init_actions(init_actions)
        np.random.seed(7)
        self.ld.play(0)
        actions2 = np.array(self.ld.current_actions)
        action_profiles = np.vstack((actions1, actions2))
        assert_array_equal(
            action_profiles,
            [[1, 1],
             [0, 1]]
            )

    def test_play_LLN(self):
        init_actions = (0, 1)
        count = 0
        for i in range(100000):
            self.ld.set_init_actions(init_actions)
            self.ld.play(0)
            if self.ld.current_actions[0] == 1:
                count += 1
        assert_almost_equal(count/100000, 0.881, decimal=2)

    def test_simulate_seed(self):
        np.random.seed(7)
        seq = self.ld.simulate(ts_length=10)
        assert_array_equal(
            seq,
            [[1, 0],
             [1, 1],
             [0, 1],
             [0, 0],
             [0, 0],
             [0, 0],
             [0, 1],
             [1, 1],
             [1, 0],
             [0, 0]]
        )

    def test_simulate_LLN(self):
        counts = np.zeros(1000)
        for i in range(1000):
            seq = self.ld.simulate(ts_length=100)
            count = 0
            for j in range(100):
                if all(seq[j, :] == [1, 1]):
                    count += 1
            counts[i] = count
        m = counts.mean() / 100
        assert_almost_equal(m, 0.6, decimal=1)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
