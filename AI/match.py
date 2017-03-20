from enum import Enum
import random
import copy
import string


class Card:
    CARD_DB = {
        'The Coin':
            {'mana_cost':0, 'spell_play_effect': 'this_turn_mana+1', 'is_spell': True},
        'Mage_Hero_Fireblast':
            {'attack': 1, 'mana_cost': 2, 'heropower': True},
        'Sheep':
            {'attack': 1, 'health': 1},
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
            {'mana_cost': 4, 'is_spell': True, 'spell_play_effect': 'damage_to_a_target_6',
             'spell_require_target': True, 'spell_target_can_be_hero': True},
        'Oasis Snapjaw':
            {'attack': 2, 'health': 7, 'mana_cost': 4},
        'Polymorph':
            {'mana_cost': 4, 'is_spell': True, 'spell_play_effect': 'transform_to_a_1/1sheep',
             'spell_require_target': True, 'spell_target_can_be_hero': False},
        'Stormwind Knight':
            {'attack': 2, 'health': 5, 'mana_cost': 4, 'charge': True}
    }

    def __init__(self, name=None, attack=None, mana_cost=None, health=None, heropower=False, divine=False, taunt=False,
                 used_this_turn=False, deterministic=True, is_spell=False, is_minion=True, charge=False,
                 zone='DECK', spell_play_effect=None, last_played_card_effect=None, spell_require_target=False,
                 spell_target_can_be_hero=False):
        self.cid = ''.join(random.choices(string.printable[:-6], k=20))
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

        # miscillaneous effects
        self.last_played_card_effect = last_played_card_effect
        self.spell_play_effect = spell_play_effect
        self.spell_require_target = spell_require_target
        self.spell_target_can_be_hero = spell_target_can_be_hero

    def __eq__(self, other):
        return self.cid == other.cid

    @staticmethod
    def init_card(name):
        card_args = Card.CARD_DB[name]
        card_args['name'] = name
        return Card(**card_args)

    def __repr__(self):
        if self.is_spell:
            return "SPELL:{0}({1})".format(self.name, self.mana_cost)
        elif self.is_minion:
            return "MINION:{0}({1}, {2}, {3})".format(self.name, self.mana_cost, self.attack, self.health)


class Action:
    def apply(self, game_world):
        pass


class MinionPlay(Action):
    """ Play a minion from inhand to intable """
    def __init__(self, src_player, src_card):
        self.src_player = src_player.name
        self.src_card = src_card

    def apply(self, game_world):
        for card in game_world[self.src_player]['inhands']:
            if card == self.src_card:
                game_world[self.src_player]['intable'].append(card)
                game_world[self.src_player]['inhands'].remove(card)
                game_world[self.src_player]['mana'] -= card.mana_cost
                if card.charge:
                    card.used_this_turn = False
                else:
                    card.used_this_turn = True
                break


class SpellPlay(Action):
    def __init__(self, src_player, src_card, target_player=None, target_unit=None):
        self.src_player = src_player.name
        self.src_card = src_card
        if target_player:
            self.target_player = target_player.name
        self.target_unit = target_unit

    def apply(self, game_world):
        for card in game_world[self.src_player]['inhands']:
            if card == self.src_card:
                game_world[self.src_player]['inhands'].remove(card)
                game_world[self.src_player]['mana'] -= card.mana_cost
                self.spell_effect(game_world)
                break

    def spell_effect(self, game_world):
        sp_eff = self.src_card.spell_play_effect
        if sp_eff == 'this_turn_mana+1':
            game_world[self.src_player]['mana'] += 1
        elif sp_eff == 'damage_to_a_target_6':
            if self.target_unit == 'hero':
                game_world[self.target_player]['health'] -= 6
            else:
                for pawn in game_world[self.target_player]['intable']:
                    if pawn == self.target_unit:
                        pawn.health -= 6
                        break
        elif sp_eff == 'transform_to_a_1/1sheep':
            for pawn in game_world[self.target_player]['intable']:
                if pawn == self.target_unit:
                    pawn = Card.init_card('Sheep')


class MinionAttack(Action):
    def __init__(self, src_player, target_player, target_unit, src_card):
        self.src_player = src_player.name
        self.target_player = target_player.name
        self.target_unit = target_unit
        self.src_card = src_card

    def apply(self, game_world):
        assert self.src_card.is_minion
        # need to find src card in the new game world
        for pawn in game_world[self.src_player]['intable']:
            if pawn == self.src_card:
                self.src_card = pawn
                break

        if self.target_unit == 'hero':
            game_world[self.target_player]['health'] -= self.src_card.attack
        else:
            for pawn in game_world[self.target_player]['intable']:
                if pawn == self.target_unit:
                    pawn.health -= self.src_card.attack
                    self.src_card.health -= pawn.attack
                    break

        self.src_card.used_this_turn = True


class HeroPowerAttack(Action):
    def __init__(self, src_player, target_player, target_unit):
        self.src_player = src_player.name
        self.target_player = target_player.name
        self.target_unit = target_unit

    def apply(self, game_world):
        src_card = game_world[self.src_player]['heropower']  # need to find src card in the new game world

        game_world[self.src_player]['mana'] -= src_card.mana_cost
        if self.target_unit == 'hero':
            game_world[self.target_player]['health'] -= src_card.attack
        else:
            for pawn in game_world[self.target_player]['intable']:
                if pawn == self.target_unit:
                    pawn.health -= src_card.attack
                    break


mage_fix_deck = [
                 'Mana Wyrm',
                 'Bloodfen Raptor', 'Bloodfen Raptor', 'Bluegill Warriors', 'River Crocolisk', 'River Crocolisk',
                 'Magma Rager', 'Magma Rager', 'Wolfrider', 'Wolfrider',
                 'Chillwind Yeti', 'Chillwind Yeti', 'Fireball', 'Fireball',
                 'Oasis Snapjaw', 'Oasis Snapjaw', 'Polymorph', 'Polymorph', 'Stormwind Knight', 'Stormwind Knight'
                 ]


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
            print("create fix deck (%d): %r" % (self.deck_remain_size, self.indeck))
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
        self.this_turn_mana = 0
        self.init_heropower()
        self.init_deck(fix_deck)
        self.intable = []          # cards on self table
        self.first_player = first_player
        self.opponent = None       # init in match

    def init_deck(self, fix_deck):
        self.deck = Deck(fix_deck)

    def init_heropower(self):
        if self.cls == HeroClass.MAGE:
            self.heropower = Card.init_card('Mage_Hero_Fireblast')

    def __repr__(self):
        return "%r health:%d, mana:%d, in hands:%r" \
               % (self.name, self.health, self.this_turn_mana, self.inhands)

    def draw_as_first_player(self):
        assert self.first_player
        self.inhands = self.deck.draw(3)    # cards in own hands

    def draw_as_second_player(self):
        assert not self.first_player
        self.inhands = self.deck.draw(4)
        self.inhands.append(Card.init_card('The Coin'))

    def turn_begin_init(self, turn):
        """ works to be done when this turn starts:
            1. update player's mana """
        self.this_turn_mana = min((turn - 1) // 2 + 1, 10)
        for card in self.intable:
            card.used_this_turn = False

    def card_playable_from_hands(self, card, game_world):
        """ whether a card can be played from hands """
        if card.heropower:
            return not card.used_this_turn and game_world[self.name]['mana'] >= card.mana_cost
        elif card.is_minion:
            return game_world[self.name]['mana'] >= card.mana_cost
        elif card.is_spell:
            return game_world[self.name]['mana'] >= card.mana_cost

    def minion_attackable(self, src_card, target_player, target_unit, game_world):
        """ whether a minion can attack the opponent """
        if src_card.used_this_turn:
            return False

        if target_unit == 'hero':
            for oppo_pawn in game_world[target_player.name]['intable']:
                if oppo_pawn.taunt:
                    return False
            return True
        elif target_unit.is_minion:
            if target_unit.taunt:
                return True
            for oppo_pawn in game_world[self.target_player.name]['intable']:
                if oppo_pawn.taunt:
                    return False
            return True

    def search_all_actions(self, game_world, cur_act_seq_and_gw, all_act_seqs_and_gw):
        """ Search all possible actions """
        candidates = []

        if self.card_playable_from_hands(self.heropower, game_world):
            candidates.append(HeroPowerAttack(src_player=self, target_player=self.opponent, target_unit='hero'))
            for oppo_pawn in game_world[self.opponent.name]['intable']:
                candidates.append(HeroPowerAttack(src_player=self, target_player=self.opponent, target_unit=oppo_pawn))

        for card in game_world[self.name]['inhands']:
            if self.card_playable_from_hands(card, game_world):
                if card.is_spell:
                    if card.spell_require_target:
                        if card.spell_target_can_be_hero:
                            candidates.append(SpellPlay(src_player=self, src_card=card, target_player=self.opponent,
                                                        target_unit='hero'))
                        for oppo_pawn in game_world[self.opponent.name]['intable']:
                            candidates.append(SpellPlay(src_player=self, src_card=card, target_player=self.opponent,
                                                        target_unit=oppo_pawn))
                    else:
                        candidates.append(SpellPlay(src_player=self, src_card=card))
                elif card.is_minion:
                    candidates.append(MinionPlay(src_player=self, src_card=card))

        for pawn in game_world[self.name]['intable']:
            if self.minion_attackable(src_card=pawn, target_player=self.opponent,
                                      target_unit='hero', game_world=game_world):
                candidates.append(MinionAttack(src_player=self, src_card=pawn, target_player=self.opponent, target_unit='hero'))
            for oppo_pawn in game_world[self.opponent.name]['intable']:
                if self.minion_attackable(src_card=pawn, target_player=self.opponent,
                                          target_unit=oppo_pawn, game_world=game_world):
                    candidates.append(MinionAttack(src_player=self, src_card=pawn, target_player=self.opponent, target_unit=oppo_pawn))

        # add doing nothing as a choice
        all_act_seqs_and_gw.append(copy.deepcopy(cur_act_seq_and_gw))

        for candidate in candidates:
            # backup
            game_world_new = game_world.copy()
            candidate.apply(game_world)
            # apply, DFS
            cur_act_seq_and_gw.append((candidate, game_world))
            self.search_all_actions(game_world, cur_act_seq_and_gw, all_act_seqs_and_gw)
            # restore
            cur_act_seq_and_gw.pop()
            game_world = game_world_new





class GameWorld:
    def __init__(self, player1, player2):
        self.data = {player1.name: {'intable': player1.intable,
                                    'inhands': player1.inhands,
                                    'health': player1.health,
                                    'mana': player1.this_turn_mana,
                                    'heropower': player1.heropower},

                     player2.name: {'intable': player2.intable,
                                    'inhands': player2.inhands,
                                    'health': player2.health,
                                    'mana': player2.this_turn_mana,
                                    'heropower': player2.heropower}}

    def __getitem__(self, player_name):
        return self.data[player_name]

    def copy(self):
        return copy.deepcopy(self)


class Match:

    def __init__(self):
        self.player1 = Player(cls=HeroClass.MAGE, name='player1', first_player=True,
                              fix_deck=mage_fix_deck)  # player 1 plays first
        self.player2 = Player(cls=HeroClass.MAGE, name='player2', first_player=False,
                              fix_deck=mage_fix_deck)
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1

        self.player1.draw_as_first_player()
        self.player2.draw_as_second_player()
        self.turn = 0

    def play_match(self):
        for i in range(10):
            self.turn += 1
            player = self.player1 if self.turn % 2 else self.player2
            player.turn_begin_init(self.turn)      # update mana, etc. at the beginning of a turn
            print("Turn {0}. {1}".format(self.turn, player))

            all_act_seqs_and_gw = []
            game_world = GameWorld(self.player1, self.player2).copy()
            player.search_all_actions(game_world,
                                      cur_act_seq_and_gw=[(None, game_world)],
                                      all_act_seqs_and_gw=all_act_seqs_and_gw)
            print(player, all_act_seqs_and_gw)


