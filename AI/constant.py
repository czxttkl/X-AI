import logging
from card import HeroClass
from player import RandomPlayer, QLearningTabularPlayer

# q learning
qltabqvalues = 'data/qltabqvalues'
qltab_save_freq = 50000  # num of matches to save q-learning values once
gamma = 0.95       # discounting factor
epsilon = 0.2     # epsilon-greedy
alpha = 1.0       # learning rate

# logger
logger = logging.getLogger('hearthstone')
logger.addHandler(logging.StreamHandler())
# logger.addHandler(logging.FileHandler('out.txt', mode='w'))
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.INFO)


# game
play_n_match = 9999999999
player1_win_rate_num_games = 1000         # calculate player 1's win rate based on how many recent games
start_health = 30

mage_fix_deck = [
                 'Mana Wyrm', 'Mirror Image',
                 'Bloodfen Raptor', 'Bloodfen Raptor', 'Bluegill Warriors', 'River Crocolisk', 'River Crocolisk',
                 'Magma Rager', 'Magma Rager', 'Wolfrider', 'Wolfrider',
                 'Chillwind Yeti', 'Chillwind Yeti', 'Fireball', 'Fireball', 'Silvermoon Guardian',
                 'Oasis Snapjaw', 'Oasis Snapjaw', 'Polymorph', 'Polymorph', 'Stormwind Knight', 'Stormwind Knight'
                 ]

# a test deck with all fireball cards. qlearning player should easily learn to defeat opponents.
test_fix_deck = [
                 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                 ]

# random player plays first
player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', first_player=True, fix_deck=mage_fix_deck)
player2 = QLearningTabularPlayer(cls=HeroClass.MAGE, name='player2', first_player=False, fix_deck=mage_fix_deck,
                                 gamma=gamma, epsilon=epsilon, alpha=alpha, test=False)
# player2 = RandomPlayer(cls=HeroClass.MAGE, name='player2', first_player=False, fix_deck=test_fix_deck)
