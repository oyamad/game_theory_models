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

from game_tools import Player, br_correspondence


class TestPlayer:
    """Test the methods of Player"""

    def setUp(self):
        """Setup a Player instance"""
        coordination_game_matrix = [[4, 0], [3, 2]]
        self.player = Player(coordination_game_matrix)

    def test_best_response_against_pure(self):
        """best_response against a pure action"""
        eq_(self.player.best_response(1), 1)

    def test_best_response_against_mixed(self):
        """best_response against a mixed action"""
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


class TestBRCorrespondence_2opponents:
    """Test br_correspondence with two opponents"""

    def setUp(self):
        """Set up a payoff function with two opponents"""
        self.payoffs_2opponents = [[[3, 6],
                                    [4, 2]],
                                   [[1, 0],
                                    [5, 7]]]

    def test_br_correspondence_against_pure(self):
        """br_correspondence against a profile of pure actions"""
        eq_(br_correspondence([0, 0], self.payoffs_2opponents), 0)

    def test_br_correspondence_against_mixed(self):
        """br_correspondence against a profile of mixed actions"""
        assert_array_equal(
            sorted(br_correspondence([[3/7, 4/7], [1/2, 1/2]],
                                     self.payoffs_2opponents)),
            sorted([0, 1])
        )


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
