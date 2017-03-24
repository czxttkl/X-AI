import random
from enum import Enum
import string


class Card:
    CARD_DB = {
        'The Coin':
            {'mana_cost':0, 'spell_play_effect': 'this_turn_mana+1', 'is_spell': True, 'is_minion': False},
        'Mage_Hero_Fireblast':
            {'attack': 1, 'mana_cost': 2, 'heropower': True, 'is_minion': False},

        'Sheep':
            {'attack': 1, 'health': 1, 'collectible': False},
        'Mirror Image 0/2 Taunt':
            {'attack': 0, 'health': 2, 'collectible': False},

        'Mirror Image':
            {'mana_cost': 1, 'is_spell': True, 'is_minion': False, 'spell_play_effect': 'summon two 0/2 taunt minions'},
        'Mana Wyrm':
            {'attack': 1, 'health': 3, 'mana_cost': 1, 'last_played_card_effect': 'cast_spell_attack+1'}, # effect is checked after every move

        'Bloodfen Raptor':
            {'attack': 3, 'health': 2, 'mana_cost': 2},
        'Bluegill Warriors':
            {'attack': 2, 'health': 1, 'mana_cost': 2, 'charge': True},
        'River Crocolisk':
            {'attack': 2, 'health': 3, 'mana_cost': 2},
        'Magma Rager':
            {'attack': 5, 'health': 1, 'mana_cost': 3},
        'Wolfrider':
            {'attack': 3, 'health': 1, 'mana_cost': 3, 'charge': True},
        'Chillwind Yeti':
            {'attack': 4, 'health': 5, 'mana_cost': 4},
        'Fireball':
            {'mana_cost': 4, 'is_spell': True, 'is_minion': False, 'spell_play_effect': 'damage_to_a_target_6',
             'spell_require_target': True, 'spell_target_can_be_hero': True},
        'Oasis Snapjaw':
            {'attack': 2, 'health': 7, 'mana_cost': 4},
        'Polymorph':
            {'mana_cost': 4, 'is_spell': True, 'is_minion': False, 'spell_play_effect': 'transform_to_a_1/1sheep',
             'spell_require_target': True, 'spell_target_can_be_hero': False},
        'Stormwind Knight':
            {'attack': 2, 'health': 5, 'mana_cost': 4, 'charge': True},
        'Silvermoon Guardian':
            {'attack': 3, 'health': 3, 'divine': True, 'mana_cost': 4}
    }

    def __init__(self, name=None, attack=None, mana_cost=None, health=None, heropower=False, divine=False, taunt=False,
                 used_this_turn=True, deterministic=True, is_spell=False, is_minion=True, charge=False, summon=None,
                 zone='DECK', spell_play_effect=None, last_played_card_effect=None, spell_require_target=False,
                 spell_target_can_be_hero=False, collectible=True):
        self.cid = ''.join(random.choices(string.printable[:-6], k=20))
        self.name = name
        self.mana_cost = mana_cost
        self.heropower = heropower
        self.used_this_turn = used_this_turn
        self.deterministic = deterministic
        self.collectible = collectible

        # minion
        self.is_minion = is_minion
        self.attack = attack
        self.health = health
        self.charge = charge
        self.summon = summon
        self.divine = divine
        self.taunt = taunt
        # self.zone = zone

        # spell
        self.is_spell = is_spell
        self.spell_play_effect = spell_play_effect
        self.spell_require_target = spell_require_target
        self.spell_target_can_be_hero = spell_target_can_be_hero

        # miscellaneous effects
        self.last_played_card_effect = last_played_card_effect

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.cid == other.cid
        else:
            return False

    @staticmethod
    def init_card(name):
        card_args = Card.CARD_DB[name]
        card_args['name'] = name
        return Card(**card_args)

    @staticmethod
    def find_card(card_list, card):
        """ return the card with the same cid in the card list"""
        for c in card_list:
            if card == c:
                return c

    @staticmethod
    def find_card_idx(card_list, card):
        """ return the index of the card with the same cid in the card list"""
        for i, c in enumerate(card_list):
            if card == c:
                return i

    def __repr__(self):
        if self.is_spell:
            return "SPELL:{0}({1})".format(self.name, self.mana_cost)
        elif self.is_minion:
            return "MINION:{0}({1}, {2}, {3})".format(self.name, self.mana_cost, self.attack, self.health)


mage_fix_deck = [
                 'Mana Wyrm', 'Mirror Image',
                 'Bloodfen Raptor', 'Bloodfen Raptor', 'Bluegill Warriors', 'River Crocolisk', 'River Crocolisk',
                 'Magma Rager', 'Magma Rager', 'Wolfrider', 'Wolfrider',
                 'Chillwind Yeti', 'Chillwind Yeti', 'Fireball', 'Fireball', 'Silvermoon Guardian',
                 'Oasis Snapjaw', 'Oasis Snapjaw', 'Polymorph', 'Polymorph', 'Stormwind Knight', 'Stormwind Knight'
                 ]


class HeroClass(Enum):
    MAGE = 1
    WARRIOR = 2


class DeckInsufficientException(Exception):
    """ Throw when deck is insufficient to draw cards. """
    def __init__(self, k, deck_remain_size):
        print("deck is insufficient to be drawn. k={0}, deck size={1}".format(k, deck_remain_size))


class Deck:

    def __init__(self, fix_deck):
        self.indeck = []
        if fix_deck:
            for card_name in fix_deck:
                card = Card.init_card(card_name)
                self.indeck.append(card)
            print("create fix deck (%d): %r" % (self.deck_remain_size, self.indeck))
        else:
            # random deck
            pass

    def draw(self, k=1):
        """ Draw a number of cards """
        if k > self.deck_remain_size:
            raise DeckInsufficientException(k, self.deck_remain_size)

        idc = random.sample(range(self.deck_remain_size), k=k)           # sample: draw without replacement
        drawn_cards, new_indeck = [], []
        for i, v in enumerate(self.indeck):
            if i in idc:
                drawn_cards.append(v)
            else:
                new_indeck.append(v)
        self.indeck = new_indeck
        return drawn_cards

    @property
    def deck_remain_size(self):
        return len(self.indeck)


