from match import *
from card import *
import random
import mycopy as copy


class Action:

    def __init__(self, src_player, src_card):
        """
        Any subclass of Action should note:
        src_player should not be entire player object but only his name.
        otherwise, when the action is copied, it will be too costy.
        """
        if src_player is None or isinstance(src_player, str):
            self.src_player = src_player
        else:
            self.src_player = src_player.name
        self.src_card = src_card

    def apply(self, game_world):
        """ apply this action to game world and update this game world.
        Action apply will change the game world (the mirror of player status) but not the real player status """
        self.__apply__(game_world)
        self.update_after_apply(game_world)

    def virtual_apply(self, game_world) -> 'GameWorld':
        """ apply this action to the copy of game_world. """
        new_game_world = game_world.copy()
        self.apply(new_game_world)
        return new_game_world

    def __apply__(self, game_world):
        """ the real action taken place """
        pass

    def update_after_apply(self, game_world: 'GameWorld'):
        """ update this game world after this action is executed.
        1. The updates include clear dead minions on table,
        2. performing last_played_card_effect, etc.
        3. save to player's used cards
        """
        for player in game_world.data.keys():
            new_intable = []
            for card in game_world.intable(player):
                if card.health > 0:
                    new_intable.append(card)
            game_world.update_intable(player, new_intable)

        for pawn in game_world.intable(self.src_player):
            if pawn.last_played_card_effect:
                if isinstance(self, SpellPlay) and pawn.last_played_card_effect == "cast_spell_attack+1":
                    pawn.attack += 1

        if isinstance(self, SpellPlay) or isinstance(self, MinionPlay):
            game_world.add_count_to_used_cards(self.src_player, card_name=self.src_card.name)

    def copy(self):
        return copy.deepcopy(self)


class NullAction(Action):
    """ do nothing as an action """

    def __repr__(self):
        return "End Turn"


class MinionPlay(Action):
    """ Play a minion from inhand to intable """
    def __init__(self, src_card, src_player=None, from_inhands=True):
        if src_player is None or isinstance(src_player, str):
            self.src_player = src_player
        else:
            self.src_player = src_player.name
        self.src_card = src_card
        self.from_inhands = from_inhands

    def __apply__(self, game_world: 'GameWorld'):
        if self.from_inhands:
            new_minion = Card.find_card(game_world.inhands(self.src_player), self.src_card)
            game_world.play_card_from_inhands(self.src_player, new_minion)
            game_world.dec_mana(self.src_player, new_minion.mana_cost)
        else:
            # when MinionPlay is not from inhands, play self.src_card directly
            new_minion = self.src_card

        # a table could have at most 7 minions
        if game_world.len_intable(self.src_player) < 7:
            game_world.play_card_to_intable(self.src_player, new_minion)
            if new_minion.charge:
                new_minion.used_this_turn = False
            else:
                new_minion.used_this_turn = True
            if new_minion.summon:
                for summon in new_minion.summon:
                    MinionPlay(src_player=self.src_player, from_inhands=False, src_card=Card.init_card(summon))\
                        .apply(game_world)

    def __repr__(self):
        return "MinionPlay(%r)" % self.src_card


class SpellPlay(Action):

    def __init__(self, src_player, src_card, target_player=None, target_unit=None):
        if src_player is None or isinstance(src_player, str):
            self.src_player = src_player
        else:
            self.src_player = src_player.name
        if target_player is None or isinstance(target_player, str):
            self.target_player = target_player
        else:
            self.target_player = target_player.name
        self.src_card = src_card
        self.target_unit = target_unit

    def __apply__(self, game_world: 'GameWorld'):
        new_spell = Card.find_card(game_world.inhands(self.src_player), self.src_card)
        game_world.play_card_from_inhands(self.src_player, new_spell)
        game_world.dec_mana(self.src_player, new_spell.mana_cost)
        self.spell_effect(game_world)

    def spell_effect(self, game_world: 'GameWorld'):
        sp_eff = self.src_card.spell_play_effect
        if sp_eff == 'this_turn_mana+1':
            game_world.incr_mana(self.src_player, 1)
        elif sp_eff == 'damage_to_a_target_6':
            if self.target_unit == 'hero':
                game_world.dec_health(self.target_player, 6)
            else:
                target_pawn = Card.find_card(game_world.intable(self.target_player), self.target_unit)
                target_pawn.health -= 6
        elif sp_eff == 'transform_to_a_1/1sheep':
            game_world.replace_intable_minion(player=self.target_player,
                                              old_card=self.target_unit, new_card=Card.init_card('Sheep'))
        elif sp_eff == 'summon two 0/2 taunt minions':
            MinionPlay(src_player=self.src_player, from_inhands=False,
                       src_card=Card.init_card('Mirror Image 0/2 Taunt')).apply(game_world)
            MinionPlay(src_player=self.src_player, from_inhands=False,
                       src_card=Card.init_card('Mirror Image 0/2 Taunt')).apply(game_world)

    def __repr__(self):
        if self.target_player:
            return "SpellPlay(src_card=%r, target_player=%r, target_unit=%r)" % \
                   (self.src_card, self.target_player, self.target_unit)
        else:
            return "SpellPlay(src_card=%r)" % self.src_card


class MinionAttack(Action):

    def __init__(self, src_player, target_player, target_unit, src_card):
        if src_player is None or isinstance(src_player, str):
            self.src_player = src_player
        else:
            self.src_player = src_player.name
        if target_player is None or isinstance(target_player, str):
            self.target_player = target_player
        else:
            self.target_player = target_player.name
        self.target_unit = target_unit
        self.src_card = src_card

    def __apply__(self, game_world: 'GameWorld'):
        assert self.src_card.is_minion
        pawn = Card.find_card(game_world.intable(self.src_player), self.src_card)

        if self.target_unit == 'hero':
            game_world.dec_health(self.target_player, pawn.attack)
        else:
            target_pawn = Card.find_card(game_world.intable(self.target_player), self.target_unit)
            if target_pawn.divine:
                target_pawn.divine = False
            else:
                target_pawn.health -= pawn.attack
            if pawn.divine:
                pawn.divine = False
            else:
                pawn.health -= target_pawn.attack

        pawn.used_this_turn = True

    def __repr__(self):
        return "MinionAttack(source=%r, target_player=%r, target_unit=%r)" \
               % (self.src_card, self.target_player, self.target_unit)


class HeroPowerAttack(Action):

    def __init__(self, src_player, src_card, target_player, target_unit):
        if src_player is None or isinstance(src_player, str):
            self.src_player = src_player
        else:
            self.src_player = src_player.name
        if target_player is None or isinstance(target_player, str):
            self.target_player = target_player
        else:
            self.target_player = target_player.name
        self.src_card = src_card
        self.target_unit = target_unit

    def __apply__(self, game_world: 'GameWorld'):
        heropower = game_world.hero_power(self.src_player)  # need to find src card in the new game world
        game_world.dec_mana(self.src_player, heropower.mana_cost)
        if self.target_unit == 'hero':
            game_world.dec_health(self.target_player, heropower.attack)
        else:
            target_pawn = Card.find_card(game_world.intable(self.target_player), self.target_unit)
            if target_pawn.divine:
                target_pawn.divine = False
            else:
                target_pawn.health -= heropower.attack

        heropower.used_this_turn = True

    def __repr__(self):
        return 'HeroPowerAttack(target_player=%r, target_unit=%r)' % (self.target_player, self.target_unit)


class ActionSequence:
    """ action sequences and the final world state after applying them """
    def __init__(self, game_world):
        self.action_list = [NullAction()]
        self.game_world = game_world

    def update(self, act: Action, game_world: ".match.GameWorld"):
        self.action_list.append(act)
        self.game_world = game_world

    def __len__(self):
        return len(self.action_list)

    def copy(self):
        return copy.deepcopy(self)

    def all(self, action_type):
        """ whether all actions (except NullAction) are of certain action_type """
        if len(self) == 1:
            return True
        for act in self.action_list[1:]:
            if not isinstance(act, action_type):
                return False
        return True

    def no(self, action_type_list):
        """ whether all actions (except NullAction) are not of certain action_types """
        if len(self) == 1:
            return True
        for act in self.action_list[1:]:
            for action_type in action_type_list:
                if isinstance(act, action_type):
                    return False
        return True

    def last(self, action_type):
        """ whether last action is of certain action_type """
        return isinstance(self.action_list[-1], action_type)

    def pop(self, game_world: ".match.GameWorld"):
        """ pop the last action and restore the game world"""
        self.action_list.pop()
        self.game_world = game_world

    def __repr__(self):
        str = ','.join(map(lambda x: '{0}'.format(x), self.action_list))
        return str


class ActionSequenceCollection:
    """ a collection of ActionSequences """
    def __init__(self):
        self.data = []

    def add(self, action_seq: ActionSequence):
        self.data.append(action_seq.copy())

    def __repr__(self):
        str = '\nActionSequenceChoices:\n'
        for i, d in enumerate(self.data):
            str += "Choice %d: %r\n" % (i, d)
        return str


class ActionSeqPicker:
    def __init__(self, player):
        self.player = player.name

    def pick_action(self, act_seq_coll: ActionSequenceCollection):
        pass

    def pick_action_seq_and_apply(self, act_seq_coll: ActionSequenceCollection,
                              player1: ".match.Player", player2: ".match.Player"):
        act_seq = self.pick_action(act_seq_coll)
        act_seq.game_world.update_player(player1, player2)
        print(act_seq.game_world)


class RandomActionSeqPicker(ActionSeqPicker):
    def pick_action(self, act_seq_coll: ActionSequenceCollection):
        """ pick an ActionSequence """
        i = random.choice(range(len(act_seq_coll.data)))
        act_seq = act_seq_coll.data[i]
        print("%r pick Choice %d: %r\n" % (self.player, i, act_seq))
        return act_seq




