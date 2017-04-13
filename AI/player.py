import numpy
from card import *
from actions import *
import os.path
import constant
import pickle
import random
import logging
import time
import sys
from typing import Union, List


logger = logging.getLogger('hearthstone')


class Player:

    def __init__(self, cls, name, first_player, fix_deck=None, **kwargs):
        self.name = name
        self.cls = cls
        self.health = constant.start_health
        self.armor = 0
        self.this_turn_mana = 0
        self.fix_deck = fix_deck
        self.first_player = first_player

        self._init_heropower()
        self._init_deck(fix_deck)
        self.intable = []          # cards on self table
        self.opponent = None       # this will be set in Match.__init__
        if self.first_player:
            self.draw_as_first_player()
        else:
            self.draw_as_second_player()
        self._init_player(**kwargs)     # any preloading job for the player

    def reset(self):
        """ reset this player as a new match starts """
        self.health = constant.start_health
        self.armor = 0
        self.this_turn_mana = 0
        self._init_heropower()
        self._init_deck(self.fix_deck)
        self.intable = []  # cards on self table
        if self.first_player:
            self.draw_as_first_player()
        else:
            self.draw_as_second_player()

    def _init_player(self, **kwargs):
        """ other initialization for this player"""
        pass

    def post_action(self, new_game_world: 'GameWorld', match_end: bool, winner: bool):
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

    @staticmethod
    def mana_based_on_turn( turn):
        return min((turn - 1) // 2 + 1, 10)

    def turn_begin_init(self, turn):
        """
        Return match_end
        works to be done when this turn starts:
            1. update player's mana
            2. refresh the usability of cards in table
            3. refresh the hero power
            4. draw new card """
        self.this_turn_mana = self.mana_based_on_turn(turn)
        for card in self.intable:
            card.used_this_turn = False
        self.heropower.used_this_turn = False
        try:
            new_card = self.deck.draw(1)
            # you can maximally hold 10 cards. otherwise the new drawn card will be wasted.
            if len(self.inhands) < 10:
                self.inhands.extend(new_card)
        except DeckInsufficientException:
            return True     # match ends because deck is insufficient
        return False

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

    def search_and_pick_action(self, game_world) -> 'Action':
        all_acts = self.search_one_action(game_world)
        return self.pick_action(all_acts, game_world)

    def pick_action(self, all_acts, game_world) -> 'Action':
        """ picking action will be determined by child class of Player"""
        pass

    def search_one_action(self, game_world):
        candidates = list()
        candidates.append(NullAction(src_player=self))

        # return directly if game_world ends
        if game_world[self.opponent.name]['health'] <= 0 or game_world[self.name]['health'] <= 0:
            return candidates

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

        return candidates

    def search_all_actions_in_one_turn(self, game_world, cur_act_seq, all_act_seqs):
        """ Search all possible actions in one turn using DSP """
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
        for i, act in enumerate(all_acts):
            logger.info("Choice %d (%.2f): %r" % (i, 1./len(all_acts), act))
        act = random.choice(all_acts)
        logger.info("%r pick %r\n" % (self.name, act))
        return act


class QValueTabular:
    def __init__(self, player, gamma, epsilon, alpha, annotation):
        self.player = player
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_act_visit_times = 0      # times visiting state-action pairs
        self.num_match = 0                  # number of total matches
        self.annotation = annotation

        file_name = self.file_name()

        # load q values table
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.gamma, self.epsilon, self.alpha, \
                self.num_match, self.state_act_visit_times, self.qvalues_tab = pickle.load(f)
        else:
            # key: state_str, value: dict() with key as act_str and value as (Q(s,a), # of times visiting (s,a)) tuple
            self.qvalues_tab = dict()

    def file_name(self):
        """ file name to associate with this qvalue table """
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_{4}". \
            format(constant.ql_exact_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)
        return file_name

    def state2str(self, game_world):
        """ convert the game world into a short string, which will be used as index in q-value table. """
        player = self.player
        state_str = "self h:{0}, m:{1}, rem_deck:{2}, hp_used: {3}, oppo h:{4}, mana next turn:{5}, rem_deck:{6}".\
            format(game_world[player.name]['health'], game_world[player.name]['mana'],
                   game_world[player.name]['rem_deck'], int(game_world[player.name]['heropower'].used_this_turn),
                   game_world[player.opponent.name]['health'], player.mana_based_on_turn(game_world.turn + 1),
                   game_world[player.opponent.name]['rem_deck'])

        # only use cidx list to represent self inhands cards
        inhands_str = "self-inhands:" + ','.join(map(lambda x: str(x.cidx), sorted(game_world[player.name]['inhands'])))

        # use (cidx, attack, health, divine, taunt) tuple lists to represent intable cards
        intable_str = "self-intable:" + \
                      ','.join(map(lambda x: '({0}, {1}, {2}, {3}, {4})'.
                                   format(x.cidx, x.attack, x.health, int(x.divine), int(x.taunt)),
                                   sorted(game_world[player.name]['intable'])))
        oppo_intable_str = "oppo-intable:" + \
                           ','.join(map(lambda x: '({0}, {1}, {2}, {3}, {4})'.
                                    format(x.cidx, x.attack, x.health, int(x.divine), int(x.taunt)),
                                    sorted(game_world[player.opponent.name]['intable'])))
        return state_str + ", " + inhands_str + ", " + intable_str + ", " + oppo_intable_str

    def action2str(self, action):
        return str(action)

    def __len__(self):
        return len(self.qvalues_tab)

    def __repr__(self):
        # print q-value table
        str = "Q-table:\n"
        for state_str, act_qvalue in self.qvalues_tab.items():
            str += state_str + "\n"
            for act_str, (qvalue, num_sa) in act_qvalue.items():
                str += '\t{0}={1}, {2}\n'.format(act_str, qvalue, num_sa)
        return str

    def post_match(self):
        self.print_qtable_summary()
        # self.print_qtable()
        self.num_match += 1
        # don't save any update during test
        if not self.player.test and self.num_match % constant.ql_exact_save_freq == 0:
            self.save()

    def print_qtable(self):
        """ print q-value table """
        logger.warning(str(self))

    def print_qtable_summary(self):
        """ print qtable summary """
        logger.warning("total match number: %d, qvalue table states  %d, state-action visit total %d"
                       % (self.num_match, len(self), self.state_act_visit_times))

    def save(self):
        t1 = time.time()
        file_name = self.file_name()
        with open(file_name, 'wb') as f:
            pickle.dump((self.gamma, self.epsilon, self.alpha,
                         self.num_match, self.state_act_visit_times, self.qvalues_tab), f, protocol=4)
        logger.warning("save q values to disk in %d seconds" % (time.time() - t1))

    def qvalue(self, state_str: Union[None, str]=None, act_str: Union[None, str]=None,
               game_world: Union[None, 'GameWorld']=None, action: Union[None, 'Action']=None):
        """ Q(s,a) """
        assert (state_str or game_world) and (act_str or action)
        if not state_str:
            state_str = self.state2str(game_world)
        if not act_str:
            act_str = self.action2str(action)
        return self.qvalues_tab.get(state_str, dict()).get(act_str, (0, 0))[0]

    def qvalues(self, state_str: Union[None, str]=None, act_strs: Union[None, List[str]]=None,
                      game_world: Union[None, 'GameWorld']=None, actions: Union[None, List['Action']]=None):
        """ Q(s,a) for all a in act_strs """
        assert (state_str or game_world) and (act_strs or actions)
        if not state_str:
            state_str = self.state2str(game_world)

        if act_strs:
            return list(map(lambda act_str: self.qvalue(state_str=state_str, act_str=act_str), act_strs))
        else:
            return list(map(lambda action: self.qvalue(state_str=state_str, action=action), actions))

    def count(self, state_str: str, act_str: str):
        """ number of visits at (s,a) """
        return self.qvalues_tab.get(state_str, dict()).get(act_str, (0, 0))[1]

    def max_qvalue(self, game_world: 'GameWorld'):
        """ max_a Q(s,a)"""
        state_str = self.state2str(game_world)
        all_acts = self.player.search_one_action(game_world)
        max_state_qvalue = max(map(lambda action: self.qvalue(state_str=state_str, action=action), all_acts))
        return max_state_qvalue

    def set_qvaluetab(self, state_str: str, act_str: str, update_qvalue: float, update_count: int):
        if not self.qvalues_tab.get(state_str):
            self.qvalues_tab[state_str] = dict()
        self.qvalues_tab[state_str][act_str] = (update_qvalue, update_count)

    def update(self, last_state: 'GameWorld', last_act: 'Action', new_game_world: 'GameWorld', R: float):
        """ update Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s',a')] """
        last_state_str = self.state2str(last_state)
        last_act_str = self.action2str(last_act)
        new_state_str = self.state2str(new_game_world)
        old_qvalue = self.qvalue(state_str=last_state_str, act_str=last_act_str)

        # determine max Q(s',a')
        max_new_state_qvalue = self.max_qvalue(new_game_world)

        # not necessary to write to qvalues_tab if
        # R == 0 and  max Q(s',a) == 0 and Q(s,a) == 0
        if R == 0 and max_new_state_qvalue == 0 and old_qvalue == 0:
            return

        update_count = self.count(last_state_str, last_act_str) + 1
        alpha = self.alpha / (update_count ** 0.5)

        update_qvalue = (1 - alpha) * old_qvalue + alpha * (R + self.gamma * max_new_state_qvalue)
        self.set_qvaluetab(last_state_str, last_act_str, update_qvalue, update_count)

        self.state_act_visit_times += 1

        logger.warning("Q-learning update. this state: %r, this action: %r" % (last_state_str, last_act_str))
        logger.warning(
            "Q-learning update. new_state_str: %r, max_new_state_qvalue: %f" % (new_state_str, max_new_state_qvalue))
        logger.warning("Q-learning update. Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s', a')]:   "
                       "{0} <- (1 - {1}) * {2} + {1} * [{3} + {4} * {5}], # of (s,a) visits: {6}".format
                       (update_qvalue, alpha, old_qvalue, R, self.gamma, max_new_state_qvalue, update_count))


class QLearningPlayer(Player):
    """ A player picks action based on Q-learning tabular method. """

    def _init_player(self, **kwargs):
        gamma = kwargs['gamma']               # discounting factor
        epsilon = kwargs['epsilon']           # epsilon-greedy
        alpha = kwargs['alpha']               # learning rate
        test = kwargs.get('test', False)      # whether in test mode
        method = kwargs['method']
        annotation = kwargs['annotation']     # additional note for this player
        self.test = test
        self.epsilon = epsilon
        if method == 'exact':
            self.qvalues_impl = QValueTabular(self, gamma, epsilon, alpha, annotation)
        elif method == 'linear':
            self.qvalues_impl = QValueLinearApprox(self, gamma, epsilon, alpha, annotation)

    def pick_action(self, all_acts, game_world) -> 'Action':
        if len(all_acts) == 1:
            choose_act = all_acts[0]
            logger.info("Choice 0: %r" % choose_act)
        else:
            choose_act = self.epsilon_greedy(game_world, all_acts)

        self.last_state = game_world.copy()
        self.last_act = choose_act.copy()

        logger.info("%r pick %r\n" % (self.name, choose_act))
        return choose_act

    def post_action(self, new_game_world: 'GameWorld', match_end: bool, winner: bool):
        """ called when an action is applied.
        update Q values """
        # determine reward
        if match_end:
            if winner:
                R = 1
            else:
                R = -1
        else:
            R = 0

        # update Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s',a')]
        self.qvalues_impl.update(self.last_state, self.last_act, new_game_world, R)

    def post_match(self):
        """ called when a match finishes """
        self.qvalues_impl.post_match()

    def epsilon_greedy(self, game_world: 'GameWorld', all_acts: List['Action']):
        """ pick actions based on epsilon greedy """
        # (act_idx, a, Q(s,a))
        act_qvalue_tuples = list(zip(range(len(all_acts)),
                                     all_acts,
                                     self.qvalues_impl.qvalues(game_world=game_world, actions=all_acts)))

        for act_idx, act, qvalue in act_qvalue_tuples:
            logger.info("Choice %d (%.2f): %r" % (act_idx, qvalue, act))

        # shuffle qvalue_tuples so that max function will break tie randomly.
        # random.sample is without replacement
        act_qvalue_tuples_shuffled = random.sample(act_qvalue_tuples, len(all_acts))
        max_act_idx, max_act, max_qvalue = max(act_qvalue_tuples_shuffled, key=lambda x: x[2])

        # if in test mode, do not explore, just exploit
        if self.test:
            return max_act
        else:
            acts_weights = numpy.full(shape=len(all_acts), fill_value=self.epsilon / (len(all_acts) - 1))
            acts_weights[max_act_idx] = 1. - self.epsilon
            idx_list = list(range(len(all_acts)))
            # choose_idx is the index of selected action in act_qvalue_tuples before shuffle
            choose_idx = numpy.random.choice(idx_list, 1, replace=False, p=acts_weights)[0]
            return act_qvalue_tuples[choose_idx][1]


class QValueLinearApprox:
    def __init__(self, player, gamma, epsilon, alpha, annotation):
        self.player = player
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_match = 0  # number of total matches
        self.annotation = annotation

        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_{4}". \
            format(constant.ql_linear_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)

        # load q values table
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.gamma, self.epsilon, self.alpha, self.num_match, self.weight = pickle.load(f)
        else:
            # key: state_str, value: dict() with key as act_str and value as (Q(s,a), # of times visiting (s,a)) tuple
            self.weight = numpy.zeros(100)

