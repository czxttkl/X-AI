import constant
from card import HeroClass
import logging
import multiprocessing
from match_multiprocess import Match
from player.random_player import RandomPlayer
import time
import numpy


class Environment:

    def output(self, match_num=100):
        start_health = 30
        deck = constant.mage_fix_deck
        logger = logging.getLogger('hearthstone')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.WARNING)
        player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', first_player=True,
                               start_health=start_health, fix_deck=deck)
        player2 = RandomPlayer(cls=HeroClass.MAGE, name='player2', first_player=False,
                               start_health=start_health, fix_deck=deck)
        # test
        # logger.setLevel(logging.INFO)
        player1.reset(test=True)
        player2.reset(test=True)
        match = Match(player1, player2)

        start_time = time.time()
        win_results = []
        p = multiprocessing.Pool()
        for win_player in p.imap_unordered(match.play_one_match, range(match_num)):
            win_results.append(win_player.name)
        duration = time.time() - start_time
        player1_win_rate = numpy.mean(numpy.array(win_results) == "player1")

        # print("win result:", win_results)
        print("player1 win result:", player1_win_rate)
        print("duration:", duration)
        return player1_win_rate