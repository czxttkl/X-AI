from match import *
from card import *
import random


class Action:
    def apply(self, game_world):
        """ apply this action to game world and update this game world. """
        self.__apply__(game_world)
        game_world.update_after_action(last_action=self)

    def __apply__(self, game_world):
        """ the real action taken place """
        pass


class NullAction(Action):
    """ do nothing as an action """

    def __init__(self, src_player):
        self.src_player = src_player.name

    def __repr__(self):
        return "End Turn"


class MinionPlay(Action):
    """ Play a minion from inhand to intable """
    def __init__(self, src_card, src_player_name=None, src_player=None, from_inhands=True):
        assert src_player is not None or src_player_name is not None
        if src_player:
            self.src_player = src_player.name
        else:
            self.src_player = src_player_name
        self.src_card = src_card
        self.from_inhands = from_inhands

    def __apply__(self, game_world):
        if self.from_inhands:
            card = Card.find_card(game_world[self.src_player]['inhands'], self.src_card)
            game_world[self.src_player]['inhands'].remove(card)
            game_world[self.src_player]['mana'] -= card.mana_cost
        else:
            # when MinionPlay is not from inhands, play self.src_card directly
            card = self.src_card

        # a table could have at most 7 minions
        if len(game_world[self.src_player]['intable']) < 7:
            game_world[self.src_player]['intable'].append(card)
            if card.charge:
                card.used_this_turn = False
            else:
                card.used_this_turn = True
            if card.summon:
                for summon in card.summon:
                    MinionPlay(src_player_name=self.src_player, from_inhands=False, src_card=Card.init_card(summon))\
                        .apply(game_world)

    def __repr__(self):
        return "MinionPlay(%r)" % self.src_card


class SpellPlay(Action):
    def __init__(self, src_player, src_card, target_player=None, target_unit=None):
        self.src_player = src_player.name
        self.src_card = src_card
        self.target_player = target_player
        self.target_unit = target_unit
        if target_player:
            self.target_player = target_player.name

    def __apply__(self, game_world):
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
            for i, pawn in enumerate(game_world[self.target_player]['intable']):
                if pawn == self.target_unit:
                    game_world[self.target_player]['intable'][i] = Card.init_card('Sheep')
                    break
        elif sp_eff == 'summon two 0/2 taunt minions':
            MinionPlay(src_player_name=self.src_player, from_inhands=False,
                       src_card=Card.init_card('Mirror Image 0/2 Taunt')).apply(game_world)
            MinionPlay(src_player_name=self.src_player, from_inhands=False,
                       src_card=Card.init_card('Mirror Image 0/2 Taunt')).apply(game_world)

    def __repr__(self):
        if self.target_player:
            return "SpellPlay(src_card=%r, target_player=%r, target_unit=%r)" % \
                   (self.src_card, self.target_player, self.target_unit)
        else:
            return "SpellPlay(src_card=%r)" % self.src_card


class MinionAttack(Action):
    def __init__(self, src_player, target_player, target_unit, src_card):
        self.src_player = src_player.name
        self.target_player = target_player.name
        self.target_unit = target_unit
        self.src_card = src_card

    def __apply__(self, game_world):
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
                    if pawn.divine:
                        pawn.divine = False
                    else:
                        pawn.health -= self.src_card.attack
                    self.src_card.health -= pawn.attack
                    break

        if self.src_card.divine:
            self.src_card.divine = False

        self.src_card.used_this_turn = True

    def __repr__(self):
        return "MinionAttack(source=%r, target_player=%r, target_unit=%r)" \
               % (self.src_card, self.target_player, self.target_unit)


class HeroPowerAttack(Action):
    def __init__(self, src_player, target_player, target_unit):
        self.src_player = src_player.name
        self.target_player = target_player.name
        self.target_unit = target_unit

    def __apply__(self, game_world):
        src_card = game_world[self.src_player]['heropower']  # need to find src card in the new game world

        game_world[self.src_player]['mana'] -= src_card.mana_cost
        if self.target_unit == 'hero':
            game_world[self.target_player]['health'] -= src_card.attack
        else:
            for pawn in game_world[self.target_player]['intable']:
                if pawn == self.target_unit:
                    pawn.health -= src_card.attack
                    break

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


class ActionPicker:
    def __init__(self, player):
        self.player = player.name

    def pick_action(self, actions, game_world):
        pass

    def pick_action_and_apply(self, actions, game_world):
        act = self.pick_action(actions)
        act.apply(game_world)


class RandomActionPicker(ActionPicker):

    def pick_action(self, actions, game_world):
        act = random.choice(actions)
        print("%r pick %r\n" % (self.player, act))
        return act

