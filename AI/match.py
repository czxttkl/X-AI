import random
import copy
from actions import *
from card import *
import os.path
import constant
import pickle
from collections import defaultdict, deque
import numpy
import dill
import functools


class Player:

    def __init__(self, cls, name, first_player, fix_deck=None, **kwargs):
        self.name = name
        self.cls = cls
        self.health = 30
        self.armor = 0
        self.this_turn_mana = 0
        self._init_heropower()
        self._init_deck(fix_deck)
        self.intable = []          # cards on self table
        self.first_player = first_player
        self.opponent = None       # this will be set in Match.__init__
        if self.first_player:
            self.draw_as_first_player()
        else:
            self.draw_as_second_player()
        self._init_player(**kwargs)     # any preloading job for the player

    def _init_player(self, **kwargs):
        """ other initialization for this player"""
        pass

    def post_action(self, new_game_world: 'GameWorld', game_end: bool):
        """ other things need to be done after an action is applied. """
        pass

    def post_match(self):
        """ things to do after one match """
        pass

    def _init_deck(self, fix_deck):
        self.deck = Deck(fix_deck)

    def _init_heropower(self):
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
        try:
            new_card = self.deck.draw(1)
            # you can maximally hold 10 cards. otherwise the new drawn card will be wasted.
            if len(self.inhands) < 10:
                self.inhands.extend(new_card)
        except DeckInsufficientException:
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

    def estimate_all_actions(self, game_world):
        """ estimate the number of all actions in a turn. """
        print("estimated actions %d" %
              (len(game_world[self.opponent.name]['intable']) + 1) ** (3+len(game_world[self.name]['intable'])))

    def search_and_pick_action(self, game_world) -> Action:
        all_acts = self.search_one_action(game_world)
        return self.pick_action(all_acts, game_world)

    def pick_action(self, all_acts, game_world) -> Action:
        """ picking action will be determined by child class of Player"""
        pass

    def search_one_action(self, game_world):
        candidates = list()
        candidates.append(NullAction(src_player=self))

        # hero power
        if self.card_playable_from_hands(self.heropower, game_world):
            candidates.append(HeroPowerAttack(src_player=self, target_player=self.opponent, target_unit='hero'))
            for oppo_pawn in game_world[self.opponent.name]['intable']:
                candidates.append(HeroPowerAttack(src_player=self, target_player=self.opponent, target_unit=oppo_pawn))

        # play inhands
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

        # minion attacks
        for pawn in game_world[self.name]['intable']:
            if not pawn.used_this_turn:
                candidates.append(MinionAttack(src_player=self, src_card=pawn,
                                               target_player=self.opponent, target_unit='hero'))
                for oppo_pawn in game_world[self.opponent.name]['intable']:
                    candidates.append(
                        MinionAttack(src_player=self, src_card=pawn,
                                     target_player=self.opponent, target_unit=oppo_pawn))

        for i, candidate in enumerate(candidates):
            print("Choice %d: %r" % (i, candidate))

        return candidates

    def search_all_actions_in_one_turn(self, game_world, cur_act_seq, all_act_seqs):
        """ Search all possible actions in one turn """
        candidates = []

        # hero power
        # heuristic search: if used, hero power should be used first (when cur_act_seq only has NullAction)
        if len(cur_act_seq) == 1 and self.card_playable_from_hands(self.heropower, game_world):
            candidates.append(HeroPowerAttack(src_player=self, target_player=self.opponent, target_unit='hero'))
            for oppo_pawn in game_world[self.opponent.name]['intable']:
                candidates.append(HeroPowerAttack(src_player=self, target_player=self.opponent, target_unit=oppo_pawn))

        # play inhands
        for card in game_world[self.name]['inhands']:
            if self.card_playable_from_hands(card, game_world):
                # heuristic search: spellplay should be executed before MinionPlay and MinionAttack
                if card.is_spell and cur_act_seq.no([MinionPlay, MinionAttack]):
                    if card.spell_require_target:
                        if card.spell_target_can_be_hero:
                            candidates.append(SpellPlay(src_player=self, src_card=card, target_player=self.opponent,
                                                        target_unit='hero'))
                        for oppo_pawn in game_world[self.opponent.name]['intable']:
                            candidates.append(SpellPlay(src_player=self, src_card=card, target_player=self.opponent,
                                                        target_unit=oppo_pawn))
                    else:
                        candidates.append(SpellPlay(src_player=self, src_card=card))
                # heuristic search: minionplay should be executed before MinionAttack
                elif card.is_minion and cur_act_seq.no([MinionAttack]):
                    candidates.append(MinionPlay(src_player=self, src_card=card))

        # minion attack
        oppo_taunt = []
        oppo_divine = []
        for oppo_pawn in game_world[self.opponent.name]['intable']:
            if oppo_pawn.divine:
                oppo_divine.append(oppo_pawn)
            if oppo_pawn.taunt:
                oppo_taunt.append(oppo_pawn)
        # heuristic search: when taunt is present, minion attacks can initiate from any of my minions,
        # to any of taunt opponent minions
        if oppo_taunt:
            for pawn in game_world[self.name]['intable']:
                if not pawn.used_this_turn:
                    for oppo_pawn in oppo_taunt:
                        candidates.append(
                                MinionAttack(src_player=self, src_card=pawn,
                                             target_player=self.opponent, target_unit=oppo_pawn))
        # heuristic search: when divine is present, search minion attacks from any of my minions,
        # to any of opponent minions
        elif oppo_divine:
            for pawn in game_world[self.name]['intable']:
                if not pawn.used_this_turn:
                    candidates.append(MinionAttack(src_player=self, src_card=pawn,
                                                   target_player=self.opponent, target_unit='hero'))
                    for oppo_pawn in game_world[self.opponent.name]['intable']:
                        candidates.append(
                                MinionAttack(src_player=self, src_card=pawn,
                                             target_player=self.opponent, target_unit=oppo_pawn))
        # heuristic search: when no taunt or divine is present in opponent minions, search
        # minion attacks from my first usable minion to any of other minions
        else:
            for pawn in game_world[self.name]['intable']:
                if not pawn.used_this_turn:
                    candidates.append(MinionAttack(src_player=self, src_card=pawn,
                                                   target_player=self.opponent, target_unit='hero'))
                    for oppo_pawn in game_world[self.opponent.name]['intable']:
                        candidates.append(MinionAttack(src_player=self, src_card=pawn,
                                                       target_player=self.opponent, target_unit=oppo_pawn))
                    break

        # add actions so far as a choice
        # heuristic search: has to add all MinionAttack before finishing the turn
        candidate_has_minion_attack = False
        for candidate in candidates:
            if isinstance(candidate, MinionAttack):
                candidate_has_minion_attack = True
                break
        if not candidate_has_minion_attack:
            all_act_seqs.add(cur_act_seq)

        for candidate in candidates:
            # backup
            game_world_old = game_world.copy()
            candidate.apply(game_world)
            # apply, DFS
            cur_act_seq.update(candidate, game_world)
            self.search_all_actions_in_one_turn(game_world, cur_act_seq, all_act_seqs)
            # restore
            game_world = game_world_old
            cur_act_seq.pop(game_world)


class RandomPlayer(Player):
    """ A player always picks a random action from available actions """

    def pick_action(self, all_acts, game_world):
        act = random.choice(all_acts)
        print("%r pick %r\n" % (self.name, act))
        return act


class QLearningTabularPlayer(Player):
    """ A player picks action based on Q-learning tabular method. """

    def _init_player(self, **kwargs):
        self.gamma = kwargs['gamma']        # discounting factor
        self.epsilon = kwargs['epsilon']    # epsilon-greedy
        self.alpha = kwargs['alpha']        #learning rate
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}".\
            format(constant.qltabqvalues, self.gamma, self.epsilon, self.alpha)
        # load q values table
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.gamma, self.epsilon, self.alpha, self.num_match, self.qvalues_tab = pickle.load(f)
        else:
            self.num_match = 0
            dd_int = functools.partial(defaultdict, int)
            defaultdict(dd_int)
            self.qvalues_tab = defaultdict(dd_int)

    def state2str(self, game_world):
        """ convert the game world into a string, which will be used as index in q-value table. """
        state_str = "self h:{0}, m:{1}, oppo h:{2}, turn:{3}".\
            format(game_world[self.name]['health'], game_world[self.name]['mana'],
                   game_world[self.opponent.name]['health'], game_world.turn)
        inhands_str = "self-inhands:" + str(sorted(game_world[self.name]['inhands']))
        intable_str = "self-intable:" + str(sorted(game_world[self.name]['intable']))
        oppo_intable_str = "oppo-intable" + str(sorted(game_world[self.opponent.name]['intable']))
        return state_str + ", " + inhands_str + ", " + intable_str + ", " + oppo_intable_str

    def action2str(self, action):
        return str(action)

    def pick_action(self, all_acts, game_world) -> 'Action':
        if len(all_acts) == 1:
            return all_acts[0]

        state_str = self.state2str(game_world)
        qvalue_tuples = list()
        for i, act in enumerate(all_acts):
            act_str = self.action2str(act)
            qvalue_tuples.append((i, state_str, act_str, self.qvalues_tab[state_str][act_str]))
        choose_i, choose_state_str, choose_act_str, choose_qvalue = self.epsilon_greedy(qvalue_tuples)
        self.last_state_str = choose_state_str
        self.last_act_str = choose_act_str
        choose_act = all_acts[choose_i]
        print("%r pick %r\n" % (self.name, choose_act))
        return choose_act

    def post_action(self, new_game_world: 'GameWorld', game_end: bool):
        """ called when an action is applied.
        update Q values """
        if game_end:
            R = 1
        else:
            R = 0
        new_state_str = self.state2str(new_game_world)
        new_state_qvalues = self.qvalues_tab[new_state_str].values()
        if not new_state_qvalues:
            max_new_state_qvalue = 0
        else:
            max_new_state_qvalue = max(new_state_qvalues)
        self.qvalues_tab[self.last_state_str][self.last_act_str] = \
            (1 - self.alpha) * self.qvalues_tab[self.last_state_str][self.last_act_str] + \
            self.alpha * (R + self.gamma * max_new_state_qvalue)

    def post_match(self):
        self.num_match += 1
        print("total match number: %d" % self.num_match)
        if self.num_match % 500 == 0:
            file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}".\
                format(constant.qltabqvalues, self.gamma, self.epsilon, self.alpha)
            with open(file_name, 'wb') as f:
                pickle.dump((self.gamma, self.epsilon, self.alpha, self.num_match, self.qvalues_tab), f, protocol=4)
            print("save q values to disk")

    def epsilon_greedy(self, qvalue_tuples):
        """ q-values format: a list of tuples (index, state_str, act_str, q-value) """
        max_i, max_state_str, max_act_str, max_qvalue = max(qvalue_tuples, key=lambda x: x[3])
        acts_weights = numpy.full(shape=(len(qvalue_tuples)), fill_value=self.epsilon / (len(qvalue_tuples) - 1))
        acts_weights[max_i] = 1. - self.epsilon
        idx_list = list(range(len(qvalue_tuples)))
        choose_idx = numpy.random.choice(idx_list, 1, replace=False, p=acts_weights)[0]
        return qvalue_tuples[choose_idx]


class GameWorld:
    def __init__(self, player1, player2, turn):
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
        self.turn = 0
        self.last_100_player1_win_lose = deque(maxlen=100)

    def play_N_match(self, n):
        for i in range(n):
            self.play_one_match()

    def play_one_match(self):
        while True:
            self.turn += 1
            player = self.player1 if self.turn % 2 else self.player2
            success_flg = player.turn_begin_init(self.turn)      # update mana, etc. at the beginning of a turn
            if not success_flg:
                winner, loser = self.match_end(winner=player.opponent, loser=player, reason='no_card_to_drawn')
                break

            print("Turn {0}. {1}".format(self.turn, player))

            # one action search
            game_world = GameWorld(self.player1, self.player2, self.turn)
            match_end = False
            while True:
                act = player.search_and_pick_action(game_world)
                if isinstance(act, NullAction):
                    break
                else:
                    act.apply(game_world)
                    game_world.update_player(self.player1, self.player2)
                    print(game_world)
                    match_end = self.check_for_match_end()
                    player.post_action(game_world, match_end)
                    if match_end:
                        winner, loser = match_end[0], match_end[1]
                        break

            # action sequence search
            # all_act_seqs = ActionSequenceCollection()
            # game_world = GameWorld(self.player1, self.player2)
            # player.estimate_all_actions(game_world)
            # player.search_all_actions_in_one_turn(game_world,
            #                                       cur_act_seq=ActionSequence(game_world),
            #                                       all_act_seqs=all_act_seqs)
            # print(all_act_seqs)
            # RandomActionSeqPicker(player).pick_action_seq_and_apply(all_act_seqs, self.player1, self.player2)

            if match_end:
                break
            print("")

        self.post_one_match(winner, loser)

    def post_one_match(self, winner, loser):
        print("-------------------------------------------------------------------------------")
        winner.post_match()
        loser.post_match()
        if winner == self.player1:
            self.last_100_player1_win_lose.append(1)
        else:
            self.last_100_player1_win_lose.append(0)
        print("last 100 player 1 win rate: {0}".format(numpy.mean(self.last_100_player1_win_lose)))
        print("-------------------------------------------------------------------------------")

    def match_end(self, winner, loser, reason):
        print("match ends at turn %d. winner=%r, loser=%r, reason=%r" % (self.turn, winner.name, loser.name, reason))
        return winner, loser

    def check_for_match_end(self):
        """ return True if the match ends """
        if self.player1.health <= 0:
            return self.match_end(winner=self.player2, loser=self.player1, reason="player1 health<=0")
        elif self.player2.health <= 0:
            return self.match_end(winner=self.player1, loser=self.player2, reason="player2 health<=0")
        else:
            return False

