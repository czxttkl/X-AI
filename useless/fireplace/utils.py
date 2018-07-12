import random
import os.path
from bisect import bisect
from importlib import import_module
from pkgutil import iter_modules
from typing import List
from xml.etree import ElementTree
from hearthstone.enums import CardClass, CardType

# Autogenerate the list of cardset modules
_cards_module = os.path.join(os.path.dirname(__file__), "cards")
CARD_SETS = [cs for _, cs, ispkg in iter_modules([_cards_module]) if ispkg]


class CardList(list):
    def __contains__(self, x):
        for item in self:
            if x is item:
                return True
        return False

    def __getitem__(self, key):
        ret = super().__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(ret)
        return ret

    def __int__(self):
        # Used in Kettle to easily serialize CardList to json
        return len(self)

    def contains(self, x):
        "True if list contains any instance of x"
        for item in self:
            if x == item:
                return True
        return False

    def index(self, x):
        for i, item in enumerate(self):
            if x is item:
                return i
        raise ValueError

    def remove(self, x):
        for i, item in enumerate(self):
            if x is item:
                del self[i]
                return
        raise ValueError

    def exclude(self, *args, **kwargs):
        if args:
            return self.__class__(e for e in self for arg in args if e is not arg)
        else:
            return self.__class__(e for k, v in kwargs.items() for e in self if getattr(e, k) != v)

    def filter(self, **kwargs):
        return self.__class__(e for k, v in kwargs.items() for e in self if getattr(e, k, 0) == v)


def random_draft(card_class: CardClass, exclude=[]):
    """
    Return a deck of 30 random cards for the \a card_class
    """
    from . import cards
    from .deck import Deck

    deck = []
    collection = []
    hero = card_class.default_hero

    for card in cards.db.keys():
        if card in exclude:
            continue
        cls = cards.db[card]
        if not cls.collectible:
            continue
        if cls.type == CardType.HERO:
            # Heroes are collectible...
            continue
        if cls.card_class and cls.card_class != card_class and cls.card_class != CardClass.NEUTRAL:
            continue
        collection.append(cls)

    while len(deck) < Deck.MAX_CARDS:
        card = random.choice(collection)
        if deck.count(card.id) < card.max_count_in_deck:
            deck.append(card.id)

    return deck


def fix_draft(card_class: CardClass, card_ids):
    """
    Return a deck of fixed cards
    """
    from . import cards
    from .deck import Deck

    deck = []
    collection = []
    hero = card_class.default_hero

    for card in cards.db.keys():
        cls = cards.db[card]
        if not cls.collectible:
            continue
        if cls.type == CardType.HERO:
            # Heroes are collectible...
            continue
        if cls.card_class and cls.card_class != card_class and cls.card_class != CardClass.NEUTRAL:
            continue
        collection.append(cls.id)

    for card_id in card_ids:
        # make sure our fixed deck is correct
        assert card_id in collection
        deck.append(card_id)

    return deck


def random_class():
    return CardClass(random.randint(2, 10))


def get_script_definition(id):
    """
    Find and return the script definition for card \a id
    """
    for cardset in CARD_SETS:
        module = import_module("fireplace.cards.%s" % (cardset))
        if hasattr(module, id):
            return getattr(module, id)


def entity_to_xml(entity):
    e = ElementTree.Element("Entity")
    for tag, value in entity.tags.items():
        if value and not isinstance(value, str):
            te = ElementTree.Element("Tag")
            te.attrib["enumID"] = str(int(tag))
            te.attrib["value"] = str(int(value))
            e.append(te)
    return e


def game_state_to_xml(game):
    tree = ElementTree.Element("HSGameState")
    tree.append(entity_to_xml(game))
    for player in game.players:
        tree.append(entity_to_xml(player))
    for entity in game:
        if entity.type in (CardType.GAME, CardType.PLAYER):
            # Serialized those above
            continue
        e = entity_to_xml(entity)
        e.attrib["CardID"] = entity.id
        tree.append(e)

    return ElementTree.tostring(tree)


def weighted_card_choice(source, weights: List[int], card_sets: List[str], count: int):
    """
    Take a list of weights and a list of card pools and produce
    a random weighted sample without replacement.
    len(weights) == len(card_sets) (one weight per card set)
    """

    chosen_cards = []

    # sum all the weights
    cum_weights = []
    totalweight = 0
    for i, w in enumerate(weights):
        totalweight += w * len(card_sets[i])
        cum_weights.append(totalweight)

    # for each card
    for i in range(count):
        # choose a set according to weighting
        chosen_set = bisect(cum_weights, random.random() * totalweight)

        # choose a random card from that set
        chosen_card_index = random.randint(0, len(card_sets[chosen_set]) - 1)

        chosen_cards.append(card_sets[chosen_set].pop(chosen_card_index))
        totalweight -= weights[chosen_set]
        cum_weights[chosen_set:] = [x - weights[chosen_set] for x in cum_weights[chosen_set:]]

    return [source.controller.card(card, source=source) for card in chosen_cards]


def setup_game() -> ".game.Game":
    """
    Original setup
    """
    from .game import Game
    from .player import Player

    deck1 = random_draft(CardClass.MAGE)
    deck2 = random_draft(CardClass.WARRIOR)
    player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
    player2 = Player("Player2", deck2, CardClass.WARRIOR.default_hero)

    game = Game(players=(player1, player2))
    game.start()

    return game


def setup_game_fix_player_fix_deck() -> ".game.Game":
    """
    Setup two fixed players and decks
    """
    from .game import Game
    from .player import Player

    fix_deck_card_ids = [
                          'EX1_277',       # Arcane Missiles x2
                          'EX1_277',
                          'NEW1_012',      # Mana Wyrm x2
                          'NEW1_012',
                          'CS2_027',       # Mirror Image x2
                          'CS2_027',
                          'EX1_066',       # Acidic Swamp Ooze x2
                          'EX1_066',
                          'CS2_172',       # Bloodfen Raptor x2
                          'CS2_172',
                          'CS2_173',       # Bluegill Warrior
                          'EX1_608',       # Sorcerer's Apprentice x2
                          'EX1_608',
                          'EX1_582',       # Dalaran Mage
                          'CS2_124',       # Wolfrider x2
                          'CS2_124',
                          'CS2_182',       # Chillwind Yeti x2
                          'CS2_182',
                          'CS2_029',       # Fireball x2
                          'CS2_029',
                          'CS2_022',       # Polymorph x2
                          'CS2_022',
                          'CS2_155',       # Archmage x2
                          'CS2_155',
                          'CS2_213',       # Reckless Rocketeer
                          'CS2_201',       # Core Hound
                          'CS2_032',       # Flamestrike
                          'CS2_032',
                          'OG_142',        # Eldritch Horror
                          'EX1_279',       # Pyroblast
                      ]

    deck1 = fix_draft(CardClass.MAGE, card_ids=fix_deck_card_ids)
    deck2 = fix_draft(CardClass.MAGE, card_ids=fix_deck_card_ids)

    player1 = Player("Player Mage 1", deck1, CardClass.MAGE.default_hero)
    player2 = Player("Player Mage 2", deck2, CardClass.MAGE.default_hero)

    game = Game(players=(player1, player2))
    game.start()

    return game


def play_turn(game: ".game.Game") -> ".game.Game":
    player = game.current_player

    while True:
        heropower = player.hero.power
        if heropower.is_usable() and random.random() < 0.1:
            if heropower.requires_target():
                heropower.use(target=random.choice(heropower.targets))
            else:
                heropower.use()
            continue

        # iterate over our hand and play whatever is playable
        for card in player.hand:
            if card.id == 'EX1_277':
                print('a')
                card.play(target=None)

            if card.is_playable() and random.random() < 0.5:
                target = None
                if card.must_choose_one:
                    card = random.choice(card.choose_cards)
                if card.requires_target():
                    target = random.choice(card.targets)
                print("Playing %r on %r" % (card, target))
                card.play(target=target)

                if player.choice:
                    choice = random.choice(player.choice.cards)
                    print("Choosing card %r" % (choice))
                    player.choice.choose(choice)

                continue

        # Randomly attack with whatever can attack
        for character in player.characters:
            if character.can_attack():
                character.attack(random.choice(character.targets))

        break

    game.end_turn()
    return game


def play_full_game(game: ".game.Game") -> ".game.Game":

    for player in game.players:
        print("%r can mulligan %r" % (player, player.choice.cards))
        mull_count = random.randint(0, len(player.choice.cards))
        cards_to_mulligan = random.sample(player.choice.cards, mull_count)
        player.choice.choose(*cards_to_mulligan)

    while True:
        play_turn(game)

    return game
