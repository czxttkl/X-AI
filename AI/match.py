from actions import *
from card import *
from player import *
from collections import deque
import numpy
import copy
import logging
import time


logger = logging.getLogger('hearthstone')


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
        t1 = time.time()
        for i in range(n):
            self.play_one_match()
        # self.player2.print_qtable()
        logger.warning('playing %d matches takes %d seconds.' % (n, time.time() - t1))

    def play_one_match(self):
        turn = 0
        self.player1.reset()
        self.player2.reset()
        self.winner = None
        player = None

        while True:
            turn += 1
            player = self.player1 if turn % 2 else self.player2
            logger.info("Turn {0}. {1}".format(turn, player))

            match_end = player.turn_begin_init(turn)      # update mana, etc. at the beginning of a turn
            # match end due to insufficient deck to draw
            if match_end:
                game_world = GameWorld(self.player1, self.player2, turn)
                self.match_end(game_world=game_world, winner=player.opponent, loser=player, reason='no_card_to_drawn')
                break
            else:
                game_world = GameWorld(self.player1, self.player2, turn)
                if turn > 2:
                    # update the last end-turn action's Q value
                    player.post_action(game_world, match_end=False, winner=False)

            # one action search
            while True:
                act = player.search_and_pick_action(game_world)
                if isinstance(act, NullAction):
                    break
                else:
                    act.apply(game_world)
                    game_world.update_player(self.player1, self.player2)
                    logger.info(game_world)
                    match_end = self.check_for_match_end(game_world)
                    if match_end:
                        break
                    else:
                        player.post_action(game_world, match_end=False, winner=False)

            if match_end:
                break

            logger.info("")

        self.post_one_match()

    def post_one_match(self):
        self.player1.post_match()
        self.player2.post_match()
        if self.winner == self.player1:
            self.last_100_player1_win_lose.append(1)
        else:
            self.last_100_player1_win_lose.append(0)
        logger.warning("last 100 player 1 win rate: {0}".format(numpy.mean(self.last_100_player1_win_lose)))
        logger.warning("-------------------------------------------------------------------------------")

    def match_end(self, game_world, winner, loser, reason):
        logger.warning("match ends at turn %d. winner=%r, loser=%r, reason=%r" %
                       (game_world.turn, winner.name, loser.name, reason))
        winner.post_action(game_world, match_end=True, winner=True)
        loser.post_action(game_world, match_end=True, winner=False)
        self.winner = winner

    def check_for_match_end(self, game_world):
        """ return True if the match ends. Otherwise return False """
        if game_world[self.player1.name]['health'] <= 0:
            self.match_end(game_world=game_world, winner=self.player2, loser=self.player1, reason="player1 health<=0")
            return True
        elif game_world[self.player2.name]['health'] <= 0:
            self.match_end(game_world=game_world, winner=self.player1, loser=self.player2, reason="player2 health<=0")
            return True
        else:
            return False

