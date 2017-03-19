from enum import Enum
import random


class Card:
    CARD_DB = {
        'The Coin':
            {'mana_cost':0, 'effect': 'this_turn_mana+1'},
        'Mage_Hero_Fireblast':
            {'attack': 1, 'mana_cost': 2, 'heropower': True, 'is_minion': False},
        'Mana Wyrm':
            {'attack': 1, 'health': 3, 'mana_cost': 1, 'effect': 'cast_spell_attack+1'}, # effect is checked after every move
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
            {'attack': 6, 'mana_cost': 4, 'is_spell': True},
        'Oasis Snapjaw':
            {'attack': 2, 'health': 7, 'mana_cost': 4},
        'Polymorph':
            {'is_spell': True, 'mana_cost': 4},
        'Stormwind Knight':
            {'attack': 2, 'health': 5, 'mana_cost': 4, 'charge': True}
    }

    def __init__(self, name=None, attack=None, mana_cost=None, health=None, heropower=False, divine=False, taunt=False,
                 used_this_turn=False, deterministic=True, is_spell=False, is_minion=True, charge=False,
                 zone='DECK', effect=None):
        self.name = name
        self.attack = attack
        self.mana_cost = mana_cost
        self.health = health
        self.heropower = heropower
        self.divine = divine
        self.taunt = taunt
        self.used_this_turn = used_this_turn
        self.deterministic = deterministic
        self.is_spell = is_spell
        self.is_minion = is_minion
        self.charge = charge
        self.zone = zone
        self.effect = effect

    @staticmethod
    def init_card(name):
        card_args = Card.CARD_DB[name]
        card_args['name'] = name
        return Card(**card_args)


mage_fix_deck = {
                 'Mana Wyrm',
                 'Bloodfen Raptor', 'Bloodfen Raptor', 'Bluegill Warriors', 'River Crocolisk', 'River Crocolisk',
                 'Magma Rager', 'Magma Rager', 'Wolfrider', 'Wolfrider',
                 'Chillwind Yeti', 'Chillwind Yeti', 'Fireball', 'Fireball',
                 'Oasis Snapjaw', 'Oasis Snapjaw', 'Polymorph', 'Polymorph', 'Stormwind Knight', 'Stormwind Knight'
                 }



class HeroClass(Enum):
    MAGE = 1
    WARRIOR = 2


class Deck:

    def __init__(self, fix_deck):
        self.indeck = []
        self.graveyard = []  # used cards
        if fix_deck:
            for card_name in fix_deck:
                card = Card.init_card(card_name)
                self.indeck.append(card)
            print("create fix deck: %r" % self.indeck)
        else:
            # random deck
            pass

    def draw(self, k=1):
        """ Draw a number of cards """
        kk = min(len(self.indeck), k)
        if kk < k:
            print("deck is insufficient to be drawn. k={0}, kk={1}, deck size={2}".format(k, kk, self.deck_remain_size))
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


class Player:

    def __init__(self, cls, name, first_player, fix_deck=None):
        self.name = name
        self.cls = cls
        self.health = 30
        self.armor = 0
        self.max_mana = 10
        self.this_turn_mana = 0
        self.init_heropower()
        self.init_deck(fix_deck)
        self.table = []          # cards on self table
        self.first_player = first_player

    def init_deck(self, fix_deck):
        self.deck = Deck(fix_deck)

    def init_heropower(self):
        if self.cls == HeroClass.MAGE:
            self.heropower = Card.init_card('Mage_Hero_Fireblast')

    def __repr__(self):
        return "PLayer %r %r" % (self.cls, self.name)

    def draw_as_first_player(self):
        assert self.first_player
        self.inhands = self.deck.draw(3)    # cards in own hands

    def draw_as_second_player(self):
        assert not self.first_player
        self.inhands = self.deck.draw(4)
        self.inhands.append(Card.init_card('The Coin'))


class Match:

    def __init__(self):
        self.player1 = Player(cls=HeroClass.MAGE, name='player1', first_player=True,
                              fix_deck=mage_fix_deck)  # player 1 plays first
        self.player2 = Player(cls=HeroClass.MAGE, name='player2', first_player=False,
                              fix_deck=mage_fix_deck)
        self.player1.draw_as_first_player()
        self.player2.draw_as_second_player()
        self.turn = 0

    def play_match(self):
        pass
