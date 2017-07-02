from match import Match
import constant
from card import HeroClass
from player import RandomPlayer, QLearningPlayer
import logging
import numpy


def test_rd_vs_ql_exact_all_fireblast_deck():
    """
    If start_health set to 7, 9 or 15, test player1 win rate should be 0.

    1. if start_health set to 15, Q-learning should learn to
    use three heropowers in turn 1 - 3 and then use all fireblasts.
    or: not use Coin in the first turn, use hero power in second turn, and start to use
    Coin + fireblast in the third turn, and only firefblast afterwards
    2. if start_health set to 7, Q-learning should learn to not use Coin in the
     2nd turn, use heropower in the 4th turn, and use Coin then Fireblast in
     the 6th turn
    3. if start_health set to 9, win rate should also be 0.
    4. if start_health set to 8, test player1 win rate should be around 0.1 - 0.3. No matter how Q-learning
    learns, player1 can play two heropowers in the first three turns by chance, and then
    use fireblast in the fourth turn
    """
    start_health = 8
    gamma = 1.0  # discounting factor
    epsilon = 0.2  # epsilon-greedy
    alpha = 1.0  # learning rate
    logger = logging.getLogger('hearthstone')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.WARNING)
    player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', start_health=start_health,
                           first_player=True, fix_deck=constant.all_fireblast_deck)
    player2 = QLearningPlayer(cls=HeroClass.MAGE, name='player2', start_health=start_health,
                              first_player=False, fix_deck=constant.all_fireblast_deck,
                              method='exact', gamma=gamma, epsilon=epsilon, alpha=alpha, test=False,
                              annotation='all_fireblast_deck_strthl{0}'.format(start_health),
                              )
    # train
    match = Match(player1, player2)
    match.play_n_match(n=0)
    # test
    logger.setLevel(logging.INFO)
    player1.reset(test=True)
    player2.reset(test=True)
    match = Match(player1, player2)
    match.play_n_match(n=10)


def test_rd_vs_ql_exact_mage_fix_deck():
    """
    the test for real game with mage_fix_deck. exact method will failed
    because """
    start_health = 30
    gamma = 1.0  # discounting factor
    epsilon = 0.2  # epsilon-greedy
    alpha = 1.0  # learning rate
    logger = logging.getLogger('hearthstone')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.WARNING)
    player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', first_player=True,
                           start_health=start_health, fix_deck=constant.mage_fix_deck, )
    player2 = QLearningPlayer(cls=HeroClass.MAGE, name='player2', first_player=False,
                              start_health=start_health, fix_deck=constant.mage_fix_deck,
                              method='exact', annotation='mage_fix_deck_strthl{0}'.format(start_health),
                              gamma=gamma, epsilon=epsilon, alpha=alpha, test=False)
    # train
    match = Match(player1, player2)
    match.play_n_match(n=99999999999)
    # test
    logger.setLevel(logging.INFO)
    player1.reset(test=True)
    player2.reset(test=True)
    match = Match(player1, player2)
    match.play_n_match(n=100)


def test_rd_vs_ql_la_all_fireblast_deck():
    """
    test q learning linear approximation with deck=all_fireblast deck.
    However, always observe weight update explosion.
    """
    start_health = 8
    gamma = 0.95   # discounting factor
    epsilon = 0.2  # epsilon-greedy
    alpha = 0.1    # learning rate
    logger = logging.getLogger('hearthstone')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', first_player=True,
                           start_health=start_health, fix_deck=constant.all_fireblast_deck, )
    player2 = QLearningPlayer(cls=HeroClass.MAGE, name='player2', first_player=False,
                              start_health=start_health, fix_deck=constant.all_fireblast_deck,
                              method='linear', annotation='_all_fireblast_deck_strthl{0}'.format(start_health),
                              degree=1, gamma=gamma, epsilon=epsilon, alpha=alpha, test=False)
    # train
    match = Match(player1, player2)
    match.play_n_match(n=10)
    # test
    logger.setLevel(logging.INFO)
    player1.reset(test=True)
    player2.reset(test=True)
    match = Match(player1, player2)
    match.play_n_match(n=2)


def test_rd_vs_ql_dqn_all_fireblast_deck():
    """ test q learningdqn with Deep Q-Network"""
    start_health = 15
    gamma = 1.0     # discounting factor
    epsilon = 0.3   # epsilon-greedy
    alpha = 0.01    # learning rate
    hidden_dim = 50   # hidden unit dimension for 2 hidden layer NN
    logger = logging.getLogger('hearthstone')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.WARNING)
    player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', first_player=True,
                           start_health=start_health, fix_deck=constant.all_fireblast_deck, )
    player2 = QLearningPlayer(cls=HeroClass.MAGE, name='player2', first_player=False,
                              start_health=start_health, fix_deck=constant.all_fireblast_deck,
                              method='dqn', annotation='all_fireblast_deck_strthl{0}'.format(start_health),
                              hidden_dim=hidden_dim, gamma=gamma, epsilon=epsilon, alpha=alpha, test=False)
    # train
    match = Match(player1, player2)
    match.play_n_match(n=1000)
    # test
    logger.setLevel(logging.INFO)
    player1.reset(test=True)
    player2.reset(test=True)
    match = Match(player1, player2)
    match.play_n_match(n=100)


if __name__ == "__main__":
    numpy.set_printoptions(linewidth=1000, precision=5, threshold=1000)

    # test_rd_vs_ql_exact_all_fireblast_deck()
    # test_rd_vs_ql_all_fireblast_deck()
    # test_rd_vs_ql_la_all_fireblast_deck()
    test_rd_vs_ql_dqn_all_fireblast_deck()


