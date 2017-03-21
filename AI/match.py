import random
import copy
from actions import *
from card import *


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
        return "%r health:%d, mana:%d, in table:%r, in hands:%r" \
               % (self.name, self.health, self.this_turn_mana, self.intable, self.inhands)

    def draw_as_first_player(self):
        assert self.first_player
        self.inhands = self.deck.draw(3)    # cards in own hands

    def draw_as_second_player(self):
        assert not self.first_player
        self.inhands = self.deck.draw(4)
        self.inhands.append(Card.init_card('The Coin'))

    def turn_begin_init(self, turn):
        """ works to be done when this turn starts:
            1. update player's mana
            2. refresh the usability of cards in table
            3. refresh the hero power
            4. draw new card """
        self.this_turn_mana = min((turn - 1) // 2 + 1, 10)
        for card in self.intable:
            card.used_this_turn = False
        self.heropower.used_this_turn = False
        new_card = self.deck.draw(1)
        if new_card:
            self.inhands.extend(new_card)
        else:
            return False   # failure flag because of insufficient deck
        return True

    def card_playable_from_hands(self, card, game_world):
        """ whether a card can be played from hands """
        if card.heropower:
            return not card.used_this_turn and game_world[self.name]['mana'] >= card.mana_cost
        elif card.is_minion:
            return game_world[self.name]['mana'] >= card.mana_cost
        elif card.is_spell:
            if card.spell_require_target and not card.spell_target_can_be_hero:
                if game_world[self.opponent.name]['intable']:
                    return game_world[self.name]['mana'] >= card.mana_cost
                else:
                    return False
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
            for oppo_pawn in game_world[target_player.name]['intable']:
                if oppo_pawn.taunt:
                    return False
            return True

    def search_all_actions(self, game_world, cur_act_seq, all_act_seqs):
        """ Search all possible actions """
        # add actions so far as a choice
        all_act_seqs.add(cur_act_seq)

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

        for candidate in candidates:
            # backup
            game_world_old = game_world.copy()
            candidate.apply(game_world)
            game_world.update_after_action()
            # apply, DFS
            cur_act_seq.update(candidate, game_world)
            self.search_all_actions(game_world, cur_act_seq, all_act_seqs)
            # restore
            game_world = game_world_old
            cur_act_seq.pop(game_world)


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
        self.player1_name = player1.name
        self.player2_name = player2.name

    def __repr__(self):
        str = '%r. health: %d, mana: %d\n' % \
              (self.player1_name, self[self.player1_name]['health'], self[self.player1_name]['mana'])
        str += 'intable: %r\n' % self[self.player1_name]['intable']
        str += 'inhands: %r\n' % self[self.player1_name]['inhands']
        str += "-----------------------------------------------\n"
        str += '%r. health: %d, mana: %d\n' % \
              (self.player2_name, self[self.player2_name]['health'], self[self.player2_name]['mana'])
        str += 'intable: %r\n' % self[self.player2_name]['intable']
        str += 'inhands: %r\n' % self[self.player2_name]['inhands']
        return str

    def update_after_action(self):
        for data in self.data.values():
            for card in data['intable']:
                if card.health <= 0:
                    data['intable'].remove(card)

    def __getitem__(self, player_name):
        return self.data[player_name]

    def copy(self):
        return copy.deepcopy(self)

    def update(self, player1, player2):
        """ update player1 and player2 according to this game world """
        player1.intable = self[player1.name]['intable']
        player1.inhands = self[player1.name]['inhands']
        player1.health = self[player1.name]['health']
        player1.mana = self[player1.name]['mana']
        player1.heropower = self[player1.name]['heropower']

        player2.intable = self[player2.name]['intable']
        player2.inhands = self[player2.name]['inhands']
        player2.health = self[player2.name]['health']
        player2.mana = self[player2.name]['mana']
        player2.heropower = self[player2.name]['heropower']


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
        for i in range(30):
            self.turn += 1
            player = self.player1 if self.turn % 2 else self.player2
            success_flg = player.turn_begin_init(self.turn)      # update mana, etc. at the beginning of a turn
            if not success_flg:
                self.match_end(winner=player.opponent, loser=player, reason='no_card_to_drawn')

            print("Turn {0}. {1}".format(self.turn, player))

            all_act_seqs = ActionSequenceCollection()
            game_world = GameWorld(self.player1, self.player2).copy()
            player.search_all_actions(game_world,
                                      cur_act_seq=ActionSequence(game_world),
                                      all_act_seqs=all_act_seqs)
            print(all_act_seqs)

            RandomActionPicker().pick_action_and_apply(all_act_seqs, self.player1, self.player2)
            self.check_for_match_end()
            print("")

    def match_end(self, winner, loser, reason):
        print("match ends. winner=%r, loser=%r, reason=%r" % (winner.name, loser.name, reason))

    def check_for_match_end(self):
        if self.player1.health <= 0:
            self.match_end(winner=self.player2, loser=self.player1, reason="player1 health<=0")
        elif self.player2.health <= 0:
            self.match_end(winner=self.player1, loser=self.player2, reason="player2 health<=0")


