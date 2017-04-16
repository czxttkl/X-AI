# constant for test

import logging

# q learning
ql_exact_data_path = 'data/qltab_exact'
ql_linear_data_path = 'data/qltab_linear'
ql_exact_save_freq = 5000  # num of matches to save q-learning tabular values once
ql_linear_save_freq = 50000  # num of matches to save q-learning linear weights once

# logger
# logger = logging.getLogger('hearthstone')
# logger.addHandler(logging.StreamHandler())
# logger.addHandler(logging.FileHandler('out.txt', mode='w'))
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.INFO)

# game
player1_win_rate_num_games = 1000         # calculate player 1's win rate based on how many recent games

mage_fix_deck = [
                 'Mana Wyrm', 'Mirror Image',
                 'Bloodfen Raptor', 'Bloodfen Raptor', 'Bluegill Warriors', 'River Crocolisk', 'River Crocolisk',
                 'Magma Rager', 'Magma Rager', 'Wolfrider', 'Wolfrider',
                 'Chillwind Yeti', 'Chillwind Yeti', 'Fireball', 'Fireball', 'Silvermoon Guardian',
                 'Oasis Snapjaw', 'Oasis Snapjaw', 'Polymorph', 'Polymorph', 'Stormwind Knight', 'Stormwind Knight'
                 ]

# a test deck with all fireball cards. qlearning player should easily learn to defeat opponents.
all_fireblast_deck = [
                      'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                      'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                      'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                      'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball', 'Fireball',
                     ]

# random player plays first

# player2 = RandomPlayer(cls=HeroClass.MAGE, name='player2', first_player=False, fix_deck=test_fix_deck)
