from player import Player
from typing import Union
import copy
from card import *


class GameWorld:
    def __init__(self, player1, player2, turn):
        self.data = {player1.name: {'intable': player1.intable,
                                    'inhands': player1.inhands,
                                    'health': player1.health,
                                    'mana': player1.this_turn_mana,
                                    'heropower': player1.heropower,
                                    'rem_deck': player1.deck.deck_remain_size,
                                    'used_cards': player1.used_cards},

                     player2.name: {'intable': player2.intable,
                                    'inhands': player2.inhands,
                                    'health': player2.health,
                                    'mana': player2.this_turn_mana,
                                    'heropower': player2.heropower,
                                    'rem_deck': player2.deck.deck_remain_size,
                                    'used_cards': player2.used_cards}}
        self.player1_name = player1.name
        self.player2_name = player2.name
        self.turn = turn
        self.data = copy.deepcopy(self.data)  # make sure game world is a copy of player states
                                              # so altering game world will not really affect player states

    def __repr__(self):
        """ representation of this game world, which looks like a simple game UI """
        s = 'Turn %d\n' % self.turn
        s += '%r. health: %d, mana: %d\n' % \
              (self.player1_name, self.health(self.player1_name), self.mana(self.player1_name))
        s += 'intable: %r\n' % self.intable(self.player1_name)
        s += 'inhands: %r\n' % self.inhands(self.player1_name)
        s += 'used cards: {0}\n'.format(str(self.used_cards(self.player1_name))[28:-2])
        s += "-----------------------------------------------\n"
        s += '%r. health: %d, mana: %d\n' % \
              (self.player2_name, self.health(self.player2_name), self.mana(self.player2_name))
        s += 'intable: %r\n' % self.intable(self.player2_name)
        s += 'inhands: %r\n' % self.inhands(self.player2_name)
        s += 'used cards: {0}\n'.format(str(self.used_cards(self.player2_name))[28:-2])
        return s

    def __getitem__(self, player_name):
        return self.data[player_name]

    def copy(self):
        return copy.deepcopy(self)

    def update_player(self, player1: 'Player', player2: 'Player'):
        """ update player1 and player2 according to this game world
        This represents the real updates, the updates really affect player states """
        player1.intable = self[player1.name]['intable']
        player1.inhands = self[player1.name]['inhands']
        player1.health = self[player1.name]['health']
        player1.this_turn_mana = self[player1.name]['mana']
        player1.heropower = self[player1.name]['heropower']
        player1.used_cards = self[player1.name]['used_cards']

        player2.intable = self[player2.name]['intable']
        player2.inhands = self[player2.name]['inhands']
        player2.health = self[player2.name]['health']
        player2.this_turn_mana = self[player2.name]['mana']
        player2.heropower = self[player2.name]['heropower']
        player2.used_cards = self[player2.name]['used_cards']

    def health(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self[player]['health']

    def dec_health(self, player: Union[Player, str], used_health):
        if isinstance(player, Player):
            player = player.name
        self[player]['health'] -= used_health

    def incr_health(self, player: Union[Player, str], boost_health):
        if isinstance(player, Player):
            player = player.name
        self[player]['health'] += boost_health

    def mana(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self[player]['mana']

    def dec_mana(self, player: Union[Player, str], used_mana):
        if isinstance(player, Player):
            player = player.name
        self[player]['mana'] -= used_mana

    def incr_mana(self, player: Union[Player, str], boost_mana):
        if isinstance(player, Player):
            player = player.name
        self[player]['mana'] += boost_mana

    def rem_deck(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self[player]['rem_deck']

    def hero_power(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self[player]['heropower']

    def hp_used(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self.hero_power(player).used_this_turn

    def inhands(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self[player]['inhands']

    def play_card_from_inhands(self, player: Union[Player, str], card):
        if isinstance(player, Player):
            player = player.name
        # card should be one element from inhands (the object reference should exist in the list)
        self.inhands(player).remove(card)

    def intable(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return self[player]['intable']

    def replace_intable_minion(self, player: Union[Player, str], old_card, new_card):
        target_pawn_idx = Card.find_card_idx(self.intable(player), old_card)
        self.intable(player)[target_pawn_idx] = new_card

    def update_intable(self, player: Union[Player, str], update_intable: list):
        if isinstance(player, Player):
            player = player.name
        self[player]['intable'] = update_intable

    def play_card_to_intable(self, player: Union[Player, str], card):
        if isinstance(player, Player):
            player = player.name
        self.intable(player).append(card)

    def len_intable(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return len(self.intable(player))

    def len_inhands(self, player: Union[Player, str]):
        if isinstance(player, Player):
            player = player.name
        return len(self.inhands(player))

    def inhands_has_card(self, player: Union[Player, str], card_name):
        if isinstance(player, Player):
            player = player.name
        for card in self.inhands(player):
            if card.name == card_name:
                return True
        return False

    def add_count_to_used_cards(self, player, card_name):
        if isinstance(player, Player):
            player = player.name
        self[player]['used_cards'][card_name] += 1

    def used_cards(self, player):
        if isinstance(player, Player):
            player = player.name
        return self[player]['used_cards']