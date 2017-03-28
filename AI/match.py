import random
import copy
from actions import *
from card import *
from player import *
from collections import deque
import numpy


class GameWorld:
    def __init__(self, player1, player2, turn):
        self.data = {player1.name: {'intable': player1.intable,
                                    'inhands': player1.inhands,
                                    'health': player1.health,
                                    'mana': player1.this_turn_mana,
                                    'heropower': player1.heropower,
                                    'rem_deck': player1.deck.deck_remain_size},

                     player2.name: {'intable': player2.intable,
                                    'inhands': player2.inhands,
                                    'health': player2.health,
                                    'mana': player2.this_turn_mana,
                                    'heropower': player2.heropower,
                                    'rem_deck': player2.deck.deck_remain_size}}
        self.player1_name = player1.name
        self.player2_name = player2.name
        self.turn = turn
        self.data = copy.deepcopy(self.data)  # make sure game world is a copy of player states
                                              # so altering game world will not really affect player states

    def __repr__(self):
        str = 'Turn %d\n' % self.turn
        str += '%r. health: %d, mana: %d\n' % \
              (self.player1_name, self[self.player1_name]['health'], self[self.player1_name]['mana'])
        str += 'intable: %r\n' % self[self.player1_name]['intable']
        str += 'inhands: %r\n' % self[self.player1_name]['inhands']
        str += "-----------------------------------------------\n"
        str += '%r. health: %d, mana: %d\n' % \
              (self.player2_name, self[self.player2_name]['health'], self[self.player2_name]['mana'])
        str += 'intable: %r\n' % self[self.player2_name]['intable']
        str += 'inhands: %r\n' % self[self.player2_name]['inhands']
        return str

    def update_after_action(self, last_action):
        """ update this copy of game world after the last_action is executed.
        This update will not change the real player's states. """
        for data in self.data.values():
            for card in data['intable']:
                if card.health <= 0:
                    data['intable'].remove(card)

        src_player = last_action.src_player
        for pawn in self[src_player]['intable']:
            if pawn.last_played_card_effect:
                if isinstance(last_action, SpellPlay) and pawn.last_played_card_effect == "cast_spell_attack+1":
                    pawn.attack += 1

    def __getitem__(self, player_name):
        return self.data[player_name]

    def copy(self):
        return copy.deepcopy(self)

    def update_player(self, player1, player2):
        """ update player1 and player2 according to this game world
        This represents the real updates, the updates really affect player states """
        player1.intable = self[player1.name]['intable']
        player1.inhands = self[player1.name]['inhands']
        player1.health = self[player1.name]['health']
        player1.this_turn_mana = self[player1.name]['mana']
        player1.heropower = self[player1.name]['heropower']

        player2.intable = self[player2.name]['intable']
        player2.inhands = self[player2.name]['inhands']
        player2.health = self[player2.name]['health']
        player2.this_turn_mana = self[player2.name]['mana']
        player2.heropower = self[player2.name]['heropower']


class Match:

    def __init__(self):
        self.player1 = RandomPlayer(cls=HeroClass.MAGE, name='player1', first_player=True,
                                    fix_deck=mage_fix_deck)  # player 1 plays first
        # self.player2 = RandomPlayer(cls=HeroClass.MAGE, name='player2', first_player=False,
        #                             fix_deck=mage_fix_deck)
        self.player2 = QLearningTabularPlayer(cls=HeroClass.MAGE, name='player2', first_player=False,
                                         fix_deck=mage_fix_deck, gamma=0.95, epsilon=0.2, alpha=0.5)
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1
        self.last_100_player1_win_lose = deque(maxlen=100)

    def play_N_match(self, n):
        for i in range(n):
            self.play_one_match()
        # self.player2.print_qtable()

    def play_one_match(self):
        turn = 0
        self.player1.reset()
        self.player2.reset()
        player = None

        while True:
            turn += 1
            last_player = player
            player = self.player1 if turn % 2 else self.player2
            success_flg = player.turn_begin_init(turn)      # update mana, etc. at the beginning of a turn
            if not success_flg:
                winner, loser = self.match_end(turn=turn, winner=player.opponent, loser=player, reason='no_card_to_drawn')
                break

            print("Turn {0}. {1}".format(turn, player))

            # one action search
            game_world = GameWorld(self.player1, self.player2, turn)
            match_end = False
            while True:
                act = player.search_and_pick_action(game_world)
                if isinstance(act, NullAction):
                    break
                else:
                    act.apply(game_world)
                    game_world.update_player(self.player1, self.player2)
                    print(game_world)
                    match_end = self.check_for_match_end(turn)
                    player.post_action(game_world, match_end)
                    if match_end:
                        winner, loser = match_end[0], match_end[1]
                        break

            # action sequence search
            # all_act_seqs = ActionSequenceCollection()
            # game_world = GameWorld(self.player1, self.player2)
            # player.estimate_all_actions(game_world)
            # player.search_all_actions_in_one_turn(game_world,
            #                                       cur_act_seq=ActionSequence(game_world),
            #                                       all_act_seqs=all_act_seqs)
            # print(all_act_seqs)
            # RandomActionSeqPicker(player).pick_action_seq_and_apply(all_act_seqs, self.player1, self.player2)

            if isinstance(last_player, QLearningTabularPlayer):
                game_world.turn += 1
                game_world[last_player.name]['mana'] = last_player.mana_based_on_turn(game_world.turn)
                last_player.post_action(game_world, match_end)

            if match_end:
                break
            print("")

        self.post_one_match(winner, loser)

    def post_one_match(self, winner, loser):
        print("-------------------------------------------------------------------------------")
        winner.post_match()
        loser.post_match()
        if winner == self.player1:
            self.last_100_player1_win_lose.append(1)
        else:
            self.last_100_player1_win_lose.append(0)
        print("last 100 player 1 win rate: {0}".format(numpy.mean(self.last_100_player1_win_lose)))
        print("-------------------------------------------------------------------------------")

    def match_end(self, turn, winner, loser, reason):
        print("match ends at turn %d. winner=%r, loser=%r, reason=%r" % (turn, winner.name, loser.name, reason))
        return winner, loser

    def check_for_match_end(self, turn):
        """ return winner and loser if the match ends. Otherwise return False """
        if self.player1.health <= 0:
            return self.match_end(turn=turn, winner=self.player2, loser=self.player1, reason="player1 health<=0")
        elif self.player2.health <= 0:
            return self.match_end(turn=turn, winner=self.player1, loser=self.player2, reason="player2 health<=0")
        else:
            return False

