# constant for test

import logging
from card import Card

# q learning
ql_exact_data_path = 'data/ql_tab'
ql_linear_data_path = 'data/ql_linear'
ql_dqn_data_path = 'data/ql_dqn'
ql_exact_save_freq = 5000  # num of matches to save q-learning tabular values once
ql_linear_save_freq = 100  # num of matches to save q-learning linear weights once
ql_dqn_save_freq = 100     # num of matches to save q-learning DQN weights once

ql_dqn_memory_start_train_size = 50
ql_dqn_pos_batch_size = 20
ql_dqn_neg_batch_size = 20
# self_h, oppo_h, self_m, self_hp_used, self_used_cards, oppo_used_cards, self_intable, oppo_intable
# and double for non-end-turn and end-turn action
ql_dqn_k = (1 + 1 + 1 + 1 + Card.all_diff_cards_size * 2 + 35 * 2) * 2
ql_dqn_mem_pos_size = 5000
ql_dqn_mem_neg_size = 5000
ql_dqn_train_loss_hist_size = 500

# logger
# logger = logging.getLogger('hearthstone')
# logger.addHandler(logging.StreamHandler())
# logger.addHandler(logging.FileHandler('out.txt', mode='w'))
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.INFO)

# game
player1_win_rate_num_games = 100    # calculate player 1's win rate based on how many recent games
test_win_rate_num_games = 1000      # test win rate of two players after this number of training games

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
