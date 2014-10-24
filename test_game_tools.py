"""
Filename: test_game_tools.py
Author: Daisuke Oyama

Tests for game_tools.py

"""
from __future__ import division

import sys
from numpy.testing import assert_array_equal
import nose
from nose.tools import eq_, ok_

from game_tools import Player_2P


class TestPlayer_2P:
    """Test the methods of Player_2P"""

    def setUp(self):
        """Setup a Player_2P instance"""
        coordination_game_matrix = [[4, 0], [3, 2]]
        self.player = Player_2P(coordination_game_matrix)

    def test_best_response_against_pure(self):
        """Best response against a pure action"""
        eq_(self.player.best_response(1), 1)

    def test_best_response_against_mixed(self):
        """Best response against a mixed action"""
        eq_(self.player.best_response([1/2, 1/2]), 1)

    def test_best_response_list_when_tie(self):
        """Best response with tie_breaking=False"""
        assert_array_equal(
            sorted(self.player.best_response([2/3, 1/3], tie_breaking=False)),
            sorted([0, 1])
        )

    def test_best_response_with_tie_breaking(self):
        """Best response with tie_breaking=True (default)"""
        ok_(self.player.best_response([2/3, 1/3]) in [0, 1])


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
