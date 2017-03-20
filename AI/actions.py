from match import *


class Action:
    def apply(self, game_world):
        pass


class NullAction(Action):
    """ do nothing as an action """

    def __repr__(self):
        return "Null"


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

    def __repr__(self):
        return "MinionAttack(source=%r, target_player=%r, target_unit=%r)" \
               % (self.src_card, self.target_player, self.target_unit)


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

    def copy(self):
        return copy.deepcopy(self)

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
        str = ''
        for i, d in enumerate(self.data):
            str += "Choice %d: %r\n" % (i, d)
        return str



