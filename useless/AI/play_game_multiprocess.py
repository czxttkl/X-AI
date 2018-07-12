import constant
from card import HeroClass
import logging
import multiprocessing
from match_multiprocess import Match
from player.random_player import RandomPlayer
import time
import numpy


def test_rd_vs_rd_all_fireblast_deck(arg):
    """ test random vs. random """
    match, idx = arg
    return match.play_one_match(idx).name

if __name__ == "__main__":
    match_num = 6000

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
    for res in p.imap_unordered(test_rd_vs_rd_all_fireblast_deck, [(match, i) for i in range(match_num)]):
        win_results.append(res)
    duration = time.time() - start_time

    # print("win result:", win_results)
    print("player1 win result:", numpy.mean(numpy.array(win_results) == "player1"))
    print("duration:", duration)