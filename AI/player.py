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
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

numpy.set_printoptions(threshold=10)
logger = logging.getLogger('hearthstone')


class Player:

    def __init__(self, cls, name, first_player, start_health, fix_deck=None, **kwargs):
        self.name = name
        self.cls = cls
        self.start_health = start_health
        self.health = self.start_health
        self.armor = 0
        self.this_turn_mana = 0
        self.fix_deck = fix_deck
        self.first_player = first_player

        self._init_heropower()
        self._init_deck(fix_deck)
        self.inhands = []          # cards in self hands
        self.intable = []          # cards on self table
        self.opponent = None       # this will be set in Match.__init__
        if self.first_player:
            self.draw_as_first_player()
        else:
            self.draw_as_second_player()
        self._init_player(**kwargs)     # any preloading job for the player

    def reset(self, test=None):
        """ reset this player as a new match starts """
        self.health = self.start_health
        self.armor = 0
        self.this_turn_mana = 0
        self._init_heropower()
        self._init_deck(self.fix_deck)
        self.inhands = []          # cards in self hands
        self.intable = []          # cards on self table
        if self.first_player:
            self.draw_as_first_player()
        else:
            self.draw_as_second_player()
        if test is not None:
            self.test = test

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
    def max_mana_this_turn(turn):
        return min((turn - 1) // 2 + 1, 10)

    def turn_begin_init(self, turn):
        """
        Return match_end
        works to be done when this turn starts:
            1. update player's mana
            2. refresh the usability of cards in table
            3. refresh the hero power
            4. draw new card """
        self.this_turn_mana = self.max_mana_this_turn(turn)
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

        # key: state_str, value: dict() with key as act_str and value as (Q(s,a), # of times visiting (s,a)) tuple
        self.qvalues_tab = dict()

        # load existing tabular if necessary
        self.load()

    def file_name(self):
        """ file name to associate with this qvalue table """
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_{4}". \
            format(constant.ql_exact_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)
        return file_name

    def state2str(self, game_world: 'GameWorld') -> str:
        """ convert the game world into a short string, which will be used as index in q-value table. """
        player = self.player
        state_str = "self h:{0}, m:{1}, rem_deck:{2}, hp_used: {3}, oppo h:{4}, mana next turn:{5}, rem_deck:{6}".\
            format(game_world[player.name]['health'], game_world[player.name]['mana'],
                   game_world[player.name]['rem_deck'], int(game_world[player.name]['heropower'].used_this_turn),
                   game_world[player.opponent.name]['health'], player.max_mana_this_turn(game_world.turn + 1),
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

    @staticmethod
    def action2str(action: 'Action') -> str:
        return str(action)

    def __len__(self):
        return len(self.qvalues_tab)

    def __repr__(self):
        # print q-value table
        s = "Q-table:\n"
        for state_str, act_qvalue in self.qvalues_tab.items():
            s += state_str + "\n"
            for act_str, (qvalue, num_sa) in act_qvalue.items():
                s += '\t{0}={1}, {2}\n'.format(act_str, qvalue, num_sa)
        return s

    def post_match(self):
        self.num_match += 1
        self.print_qtable_summary()
        # self.print_qtable()
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

    def load(self):
        file_name = self.file_name()
        # load q values table
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.gamma, self.epsilon, self.alpha, self.num_match, self.state_act_visit_times, self.qvalues_tab \
                    = pickle.load(f)

    def save(self):
        t1 = time.time()
        file_name = self.file_name()
        with open(file_name, 'wb') as f:
            pickle.dump((self.gamma, self.epsilon, self.alpha,
                         self.num_match, self.state_act_visit_times, self.qvalues_tab), f, protocol=4)
        logger.warning("save q values to disk in %d seconds" % (time.time() - t1))

    def qvalue(self, state_str: Union[None, str]=None, act_str: Union[None, str]=None,
               game_world: Union[None, 'GameWorld']=None, action: Union[None, 'Action']=None) -> float:
        """ Q(s,a) """
        assert (state_str or game_world) and (act_str or action)
        if not state_str:
            state_str = self.state2str(game_world)
        if not act_str:
            act_str = self.action2str(action)
        return self.qvalues_tab.get(state_str, dict()).get(act_str, (0, 0))[0]

    def qvalues(self, state_str: Union[None, str]=None, act_strs: Union[None, List[str]]=None,
                game_world: Union[None, 'GameWorld']=None, actions: Union[None, List['Action']]=None) \
            -> List[float]:
        """ Q(s,a) for all a in actions """
        assert (state_str or game_world) and (act_strs or actions)
        if not state_str:
            state_str = self.state2str(game_world)

        if act_strs:
            return list(map(lambda act_str: self.qvalue(state_str=state_str, act_str=act_str), act_strs))
        else:
            return list(map(lambda action: self.qvalue(state_str=state_str, action=action), actions))

    def count(self, state_str: str, act_str: str) -> int:
        """ number of visits at (s,a) """
        return self.qvalues_tab.get(state_str, dict()).get(act_str, (0, 0))[1]

    def max_qvalue(self, game_world: 'GameWorld') -> float:
        """ max_a Q(s,a)"""
        state_str = self.state2str(game_world)
        all_acts = self.player.search_one_action(game_world)
        max_state_qvalue = max(map(lambda action: self.qvalue(state_str=state_str, action=action), all_acts))
        return max_state_qvalue

    def set_qvaluetab(self, state_str: str, act_str: str, update_qvalue: float, update_count: int):
        if not self.qvalues_tab.get(state_str):
            self.qvalues_tab[state_str] = dict()
        self.qvalues_tab[state_str][act_str] = (update_qvalue, update_count)

    def update(self, last_state: 'GameWorld', last_act: 'Action', new_game_world: 'GameWorld',
               r: float, match_end: bool, test: bool):
        """ update Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s',a')] """
        last_state_str = self.state2str(last_state)
        last_act_str = self.action2str(last_act)
        new_state_str = self.state2str(new_game_world)
        old_qvalue = self.qvalue(state_str=last_state_str, act_str=last_act_str)

        # determine max Q(s',a')
        if match_end:
            max_new_state_qvalue = 0
        else:
            max_new_state_qvalue = self.max_qvalue(new_game_world)

        # not necessary to write to qvalues_tab if
        # R == 0 and  max Q(s',a) == 0 and Q(s,a) == 0
        if r == 0 and max_new_state_qvalue == 0 and old_qvalue == 0:
            return

        update_count = self.count(last_state_str, last_act_str) + 1
        alpha = self.alpha / (update_count ** 0.5)
        update_qvalue = (1 - alpha) * old_qvalue + alpha * (r + self.gamma * max_new_state_qvalue)
        self.state_act_visit_times += 1

        logger.info("Q-learning update. this state: %r, this action: %r" % (last_state_str, last_act_str))
        logger.info(
            "Q-learning update. new_state_str: %r, max_new_state_qvalue: %f" % (new_state_str, max_new_state_qvalue))
        logger.info("Q-learning update. Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s', a')]:   "
                    "{0} <- (1 - {1}) * {2} + {1} * [{3} + {4} * {5}], # of (s,a) visits: {6}".format
                    (update_qvalue, alpha, old_qvalue, r, self.gamma, max_new_state_qvalue, update_count))

        # only update in training phase
        if not test:
            self.set_qvaluetab(last_state_str, last_act_str, update_qvalue, update_count)

    def determine_r(self, match_end: bool, winner: bool, old_state: 'GameWorld', new_state: 'GameWorld'):
        """ determine reward """
        # there are two schemas of rewards.

        # first, r_bias + relative opponent health decrease
        # if r_bias is set too small (very negative), then player will always choose end_turn to avoid further actions,
        # because every action adds up a negative bias
        # r_bias = -100
        # if r_bias is set too large (very positive), then player will be encouraged to play longer by
        #  using more actions.
        # r_bias = 100
        # when r_bias equals to zero, we simply look at how much damage the action cause.
        # however setting r_bias = 0 is not ideal in test_rd_vs_ql_sh8_all_fireblast_deck() because it encourages to
        # use hero_power very early to cause that one health damage instead of waiting for fireblast
        # r_bias = 0
        # the ideal reward should be set to -1 in the case of test_rd_vs_ql_sh8_all_fireblast_deck()
        # r_bias = -1
        # r = r_bias + \
        #     old_state.get_health(self.player.opponent.name) - new_state.get_health(self.player.opponent.name)

        # second schema, terminal reward, 1 for win, -1 for loss
        r = 0
        if match_end:
            if winner:
                r = 1
            else:
                r = -1
        return r


class QLearningPlayer(Player):
    """ A player picks action based on Q-learning tabular method. """

    def _init_player(self, **kwargs):
        gamma = kwargs['gamma']               # discounting factor
        epsilon = kwargs['epsilon']           # epsilon-greedy
        alpha = kwargs['alpha']               # learning rate
        test = kwargs.get('test', False)      # whether in test mode
        degree = kwargs.get('degree', 1)      # degree for polynomial feature transformation
        hidden_dim = kwargs.get('hidden_dim', 10)
                                              # hidden unit number in DQN
        method = kwargs['method']
        annotation = kwargs['annotation']     # additional note for this player
        self.epsilon = epsilon
        self.test = test
        self.last_state = None
        self.last_act = None

        if method == 'exact':
            self.qvalues_impl = QValueTabular(self, gamma, epsilon, alpha, annotation)
        elif method == 'linear':
            self.qvalues_impl = QValueLinearApprox(self, degree, gamma, epsilon, alpha, annotation)
        elif method == 'dqn':
            self.qvalues_impl = QValueDQNApprox(self, hidden_dim, gamma, epsilon, alpha, annotation)

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
        R = self.qvalues_impl.determine_r(match_end, winner, self.last_state, new_game_world)
        self.qvalues_impl.update(self.last_state, self.last_act, new_game_world, R, match_end, self.test)

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


class QValueFunctionApprox:
    feat_names = [
        'self_h', 'oppo_h',
        'self_mn', 'oppo_mn',       # number of minions of self and oppo intable
        'self_m', 'self_max_m',
        'oppo_next_m',
        'self_hp_used'              # whether self hero power used
    ]

    def __init__(self, player):
        self.player = player
        self.num_match = 0
    # feature names
    # self.feat_names = [
    #                    # feature index 0 - 9
    #                    'turn', 'self_h', 'oppo_h', 'self_m',
    #                    'self_max_m', 'oppo_next_m',         # self maximum mana this turn, oppo mana next turn
    #                    'self_rem_cards', 'oppo_rem_cards',  # number of remaining cards of self and oppo inhands
    #                    'self_mn', 'oppo_mn',                # number of minions of self and oppo intable
    #
    #                    # feature index 10 - 13
    #                    'self_mn_taunt', 'oppo_mn_taunt',    # number of minions in taunt
    #                    'self_mn_divine', 'oppo_mn_divine',  # number of minions in divine
    #
    #                    # feature index 14 - 22
    #                    'self_tt_mn_attkable', 'self_tt_mn_attk',   # self total minion (attackable) attack
    #                    'self_tt_mn_h',   # self total minion health
    #                    'self_tt_mn_taunt_attkable', 'self_tt_mn_taunt_attk', 'self_tt_mn_taunt_h',
    #                    'self_tt_mn_divine_attkable', 'self_tt_mn_divine_attk', 'self_tt_mn_divine_h',
    #
    #                    # feature index 23 - 28
    #                    'oppo_tt_mn_attk', 'oppo_tt_mn_h',
    #                    'oppo_tt_mn_taunt_attk', 'oppo_tt_mn_taunt_h',
    #                    'oppo_tt_mn_divine_attk', 'oppo_tt_mn_divine_h',
    #
    #                    # feature index 29 - 30
    #                    'self_rem_deck', 'oppo_rem_deck',
    #                    ]

    # feature[0] = game_world.turn
    # feature[1] = game_world[self.player.name]['health']
    # feature[2] = game_world[self.player.opponent.name]['health']
    # feature[3] = game_world[self.player.name]['mana']
    # feature[4] = self.player.max_mana_this_turn(game_world.turn)
    # feature[5] = self.player.opponent.max_mana_this_turn(game_world.turn + 1)
    # feature[6] = len(game_world[self.player.name]['inhands'])
    # feature[7] = len(game_world[self.player.opponent.name]['inhands'])
    # feature[8] = len(game_world[self.player.name]['intable'])
    # feature[9] = len(game_world[self.player.opponent.name]['intable'])
    #
    # self_mn_taunt, oppo_mn_taunt, self_mn_divine, oppo_mn_divine, \
    #     self_tt_mn_attkable, self_tt_mn_attk, self_tt_mn_h, \
    #     self_tt_mn_taunt_attkable, self_tt_mn_taunt_attk, self_tt_mn_taunt_h, \
    #     self_tt_mn_divine_attkable, self_tt_mn_divine_attk, self_tt_mn_divine_h, \
    #     oppo_tt_mn_attk, oppo_tt_mn_h, \
    #     oppo_tt_mn_taunt_attk, oppo_tt_mn_taunt_h, \
    #     oppo_tt_mn_divine_attk, oppo_tt_mn_divine_h = [0] * 19
    #
    # self_intable = game_world[self.player.name]['intable']
    # oppo_intable = game_world[self.player.opponent.name]['intable']
    #
    # # process self table
    # for card in self_intable:
    #     self_tt_mn_attk += card.attack
    #     self_tt_mn_h += card.health
    #     if not card.used_this_turn:
    #         self_tt_mn_attkable += card.attack
    #     if card.taunt:
    #         self_mn_taunt += 1
    #         self_tt_mn_taunt_h += card.health
    #         self_tt_mn_taunt_attk += card.attack
    #         if not card.used_this_turn:
    #             self_tt_mn_taunt_attkable += card.attack
    #     if card.divine:
    #         self_mn_divine += 1
    #         self_tt_mn_divine_h += card.health
    #         self_tt_mn_divine_attk += card.attack
    #         if not card.used_this_turn:
    #             self_tt_mn_divine_attkable += card.attack
    #
    # # process oppo intable
    # for card in oppo_intable:
    #     oppo_tt_mn_attk += card.attack
    #     oppo_tt_mn_h += card.health
    #     if card.taunt:
    #         oppo_mn_taunt += 1
    #         oppo_tt_mn_taunt_h += card.health
    #         oppo_tt_mn_taunt_attk += card.attack
    #     if card.divine:
    #         oppo_mn_divine += 1
    #         oppo_tt_mn_divine_h += card.health
    #         oppo_tt_mn_divine_attk += card.attack
    #
    # feature[10] = self_mn_taunt
    # feature[11] = oppo_mn_taunt
    # feature[12] = self_mn_divine
    # feature[13] = oppo_mn_divine
    # feature[14] = self_tt_mn_attkable
    # feature[15] = self_tt_mn_attk
    # feature[16] = self_tt_mn_h
    # feature[17] = self_tt_mn_taunt_attkable
    # feature[18] = self_tt_mn_taunt_attk
    # feature[19] = self_tt_mn_taunt_h
    # feature[20] = self_tt_mn_divine_attkable
    # feature[21] = self_tt_mn_divine_attk
    # feature[22] = self_tt_mn_divine_h
    # feature[23] = oppo_tt_mn_attk
    # feature[24] = oppo_tt_mn_h
    # feature[25] = oppo_tt_mn_taunt_attk
    # feature[26] = oppo_tt_mn_taunt_h
    # feature[27] = oppo_tt_mn_divine_attk
    # feature[28] = oppo_tt_mn_divine_h
    #
    # feature[29] = game_world[self.player.name]['rem_deck']
    # feature[30] = game_world[self.player.opponent.name]['rem_deck']

    def init_and_load(self):
        pass

    def save(self):
        pass

    def file_name(self):
        pass

    def post_match(self):
        pass

    def max_qvalue(self, game_world: 'GameWorld') -> float:
        pass

    def qvalues(self, game_world: 'GameWorld', actions: List['Action']) -> List[float]:
        """ Q(s,a) for a in actions """
        res = map(lambda act: self.qvalue(game_world, act), actions)
        # actually return an iterator
        return res

    def qvalue(self, game_world: 'GameWorld', action: 'Action', return_feature=False):
        pass

    def update(self, last_state: 'GameWorld', last_act: 'Action', new_game_world: 'GameWorld',
               r: float, match_end: bool, test: bool):
        pass

    def state2str(self, game_world: 'GameWorld') -> str:
        """ convert the game world into a short string """
        player = self.player
        oppo = self.player.opponent
        state_str = "self h:{0}, m:{1}, rem_deck:{2}, hp_used: {3}, oppo h:{4}, mana next turn:{5}, rem_deck:{6}". \
                        format(game_world.health(player), game_world.mana(player),
                               game_world.rem_deck(player), game_world.hp_used(player),
                               game_world.health(oppo), player.max_mana_this_turn(game_world.turn + 1),
                               game_world.rem_deck(oppo))

        # only use cidx list to represent self inhands cards
        inhands_str = "self-inhands:" + ','.join(
            map(lambda x: str(x.cidx), sorted(game_world.inhands(player))))

        # use (cidx, attack, health, divine, taunt) tuple lists to represent intable cards
        intable_str = "self-intable:" + \
                      ','.join(map(lambda x: '({0}, {1}, {2}, {3}, {4})'.
                                   format(x.cidx, x.attack, x.health, int(x.divine), int(x.taunt)),
                                   sorted(game_world.intable(player))))
        oppo_intable_str = "oppo-intable:" + \
                           ','.join(map(lambda x: '({0}, {1}, {2}, {3}, {4})'.
                                        format(x.cidx, x.attack, x.health, int(x.divine), int(x.taunt)),
                                        sorted(game_world.intable(oppo))))
        return state_str + ", " + inhands_str + ", " + intable_str + ", " + oppo_intable_str

    @ staticmethod
    def action2str(action: 'Action') -> str:
        return str(action)

    def feature2str(self, feature: 'numpy.ndarray') -> str:
        return 'num_match:' + str(self.num_match) + "," + \
                                                    ','.join(map(lambda name_f: name_f[0] + ':' + str(name_f[1]),
                                                                 zip(self.feat_names, feature)))

    def extract_raw_features(self, game_world: 'GameWorld', action: 'Action') -> numpy.ndarray:
        """
        extract raw features from state-action pair
        if it is an end-turn action, we extract the feature from current game world
        otherwise, we extract the feature from the game world AFTER the action is applied
        """
        if not isinstance(action, NullAction):
            game_world = action.virtual_apply(game_world)
        player = self.player
        oppo = self.player.opponent

        feature = numpy.zeros(len(self.feat_names))
        feature[:] = \
            game_world.health(player), game_world.health(oppo), \
            game_world.len_intable(player), game_world.len_intable(oppo), \
            game_world.mana(player), player.max_mana_this_turn(game_world.turn), \
            oppo.max_mana_this_turn(game_world.turn + 1), \
            game_world.hp_used(player)

        return feature

    def raw_feature_to_full_feature(self, features, action: 'Action') -> numpy.ndarray:
        """
        full features consist of two parts:
        # 1. features of the afterstate if it is a non-end-turn action. (simulate the action and resulting world)
        # 2. features of the current state if it is an end-turn action
        """
        if isinstance(action, NullAction):
            features = numpy.hstack((numpy.zeros_like(features), features))
        else:
            features = numpy.hstack((features, numpy.zeros_like(features)))
        return features

    def to_feature(self, game_world: 'GameWorld', action: 'Action') -> numpy.ndarray:
        """ convert this state-action into a full feature array """
        features = self.extract_raw_features(game_world, action)
        features = self.raw_feature_to_full_feature(features, action)
        return features

    def to_feature_over_acts(self, game_world: 'GameWorld'):
        """ """
        all_acts = self.player.search_one_action(game_world)
        features_over_acts = numpy.array(
            list(map(lambda action: self.to_feature(game_world, action), all_acts)))
        return features_over_acts

    def determine_r(self, match_end: bool, winner: bool, old_state: 'GameWorld', new_state: 'GameWorld'):
        """ determine reward """
        # also see QValueTabular.determine_r for explanation
        r = 0
        if match_end:
            if winner:
                r = 1
            else:
                r = -1
        return r


class QValueDQNApprox(QValueFunctionApprox):
    """ use deep q network for function approximation """
    def __init__(self, player, hidden_dim, gamma, epsilon, alpha, annotation):
        self.player = player
        self.hidden_dim = hidden_dim  # hidden dim of 1-layer deep q network
        self.gamma = gamma            # discount factor
        self.epsilon = epsilon        # epsilon-greedy rate
        self.alpha = alpha            # learning rate
        self.num_match = 0            # number of total matches
        self.annotation = annotation  # annotation added in model file name
        self.model = None             # Keras model. will be initialized in init_and_load()
        self.lag_model = None         # Keras model. sync with self.model every while
                                      # used in max_a' Q(s', a')
        self.train_hist = deque(maxlen=500)
                                      # Keras model train loss value. update after every fit
        self.batch_size = 64
        self.memory_size = 100
        self.memory = deque(maxlen=self.memory_size)
        # features consist of two parts:
        # 1. features of the afterstate if it is a non-end-turn action. (simulate the action and resulting world)
        # 2. features of the current state if it is a end-turn action
        # num of total features. First part for non-end-turn action. Second part for end-turn action.
        self.k = len(self.feat_names) * 2
        self.init_and_load()

    def file_name_pickle(self):
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_dqn_{4}.pickle". \
            format(constant.ql_dqn_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)
        return file_name

    def file_name_h5(self):
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_dqn_{4}.h5". \
            format(constant.ql_dqn_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)
        return file_name

    def init_and_load(self):
        self.model = self.init_weight()
        self.lag_model = self.init_weight()
        # when model is saved, it is saved to two separate files:
        # one for basic information, the other for keras
        if os.path.isfile(self.file_name_pickle()):
            with open(self.file_name_pickle(), 'rb') as f:
                self.gamma, self.epsilon, self.alpha, self.num_match, self.memory = pickle.load(f)
            self.model.load_weights(self.file_name_h5())
        self.sync_lag_model()

    def sync_lag_model(self):
        self.lag_model.set_weights(self.model.get_weights())

    def save(self):
        with open(self.file_name_pickle(), 'wb') as f:
            pickle.dump((self.gamma, self.epsilon, self.alpha, self.num_match, self.memory), f, protocol=4)
        self.model.save_weights(self.file_name_h5())

    def init_weight(self):
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.k, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden_dim, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        print(model.get_weights())
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha), metrics=['accuracy'])
        return model

    def append_memory(self, features, last_act, reward, next_features_over_acts, match_end):
        if reward != 0:
            self.memory.append((features, last_act, reward, next_features_over_acts, match_end))
        else:
            # gradually accept zero reward intermediate state
            thres = 60. / numpy.log(self.num_match + 2)
            seed = numpy.random.random()
            if seed > thres:
                self.memory.append((features, last_act, reward, next_features_over_acts, match_end))

    def post_match(self):
        self.sync_lag_model()
        self.num_match += 1
        logger.warning('QValueDQNApprox post match:' + str(self))
        # don't save any update during test
        if not self.player.test and self.num_match % constant.ql_dqn_save_freq == 0:
            self.save()

    def qvalue(self, game_world: 'GameWorld', action: 'Action', return_feature=False):
        """ Q(s,a)
         feature(s,a) are afterstate, i.e., the state after action is acted on game_world """
        # model.predict only takes 2D array
        features = self.to_feature(game_world, action).reshape((1, -1))
        qvalue = self.model.predict(features)[0, 0]
        if return_feature:
            return qvalue, features
        else:
            return qvalue

    def __repr__(self):
        mem_r = [m[2] for m in self.memory]
        unique, counts = numpy.unique(mem_r, return_counts=True)
        s = 'num_match:{0}, num_memory:{1}, memory rewards: {2}, train_loss_size:{3}, mean:{4} \n'.\
                   format(self.num_match, len(self.memory), dict(zip(unique, counts)),
                          len(self.train_hist), numpy.mean(self.train_hist))
        # print feature weights
        # s += 'model weights: \n{0}\n {1}\n {2} {3}\nlag_model weights: \n{4}\n {5}\n {6} {7}'.\
        #            format(self.model.get_weights()[0], self.model.get_weights()[1],
        #                   self.model.get_weights()[2].flatten(), self.model.get_weights()[3],
        #                   self.lag_model.get_weights()[0], self.lag_model.get_weights()[1],
        #                   self.lag_model.get_weights()[2].flatten(), self.lag_model.get_weights()[3])
        return s

    def update(self, last_state: 'GameWorld', last_act: 'Action', new_game_world: 'GameWorld',
               r: float, match_end: bool, test: bool):
        features = self.to_feature(last_state, last_act)
        next_features_over_acts = self.to_feature_over_acts(new_game_world)
        self.append_memory(features, last_act, r, next_features_over_acts, match_end)

        # train model
        # if memory is not full, continue to collect data
        if len(self.memory) < self.memory_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        features = numpy.zeros((self.batch_size, self.k))
        target = numpy.zeros(self.batch_size)

        for i in range(self.batch_size):
            features[i] = mini_batch[i][0]
            r = mini_batch[i][2]
            next_features_over_acts = mini_batch[i][3]
            match_end = mini_batch[i][4]
            if match_end:
                target[i] = r
            else:
                # in Double DQN, action selection is from model
                max_a_idx = numpy.argmax(self.model.predict(next_features_over_acts))
                # target q(s', a') is from lag_model
                max_q_s_a = self.lag_model.predict(next_features_over_acts)[max_a_idx]
                target[i] = r + self.gamma * max_q_s_a

        # logger.info("Q-learning update. model weight before update: {0}".format(self.model.get_weights()))

        # only update in training phase
        if not test:
            prev_weight = self.model.get_weights()[0]
            prev_train_loss = self.model.evaluate(features, target, verbose=0)[0]
            prev_test_loss = self.evaluate_model_on_memory()[0]
            loss = self.model.fit(features, target, batch_size=self.batch_size, epochs=1, verbose=0).history['loss']
            post_weight = self.model.get_weights()[0]
            post_train_loss = self.model.evaluate(features, target, verbose=0)[0]
            post_test_loss = self.evaluate_model_on_memory()[0]
            self.train_hist.append(loss)
            logger.warning("Q-learning update model weight update change: {0}, prev train loss:{1}, prev memory loss:{2}, post train loss:{3}, post memory loss:{4}".format(
                numpy.sum((post_weight - prev_weight)**2), prev_train_loss, prev_test_loss, post_train_loss, post_test_loss))
            # logger.info("Q-learning update. model weight after update: {0}".format(self.model.get_weights()))

    def evaluate_model_on_memory(self):
        """ evaluate model on whole memory """
        features = numpy.zeros((len(self.memory), self.k))
        target = numpy.zeros(len(self.memory))

        for i, m in enumerate(self.memory):
            features[i] = m[0]
            r = m[2]
            next_features_over_acts = m[3]
            match_end = m[4]
            if match_end:
                target[i] = r
            else:
                # in Double DQN, action selection is from model
                max_a_idx = numpy.argmax(self.model.predict(next_features_over_acts))
                # target q(s', a') is from lag_model
                max_q_s_a = self.lag_model.predict(next_features_over_acts)[max_a_idx]
                target[i] = r + self.gamma * max_q_s_a

        whole_memory_loss, whole_memory_acc = self.model.evaluate(features, target, verbose=0)
        return whole_memory_loss, whole_memory_acc


class QValueLinearApprox(QValueFunctionApprox):
    """ use linear function approximation. however always observe weights grow to infinity """

    def __init__(self, player, degree, gamma, epsilon, alpha, annotation):
        self.player = player
        self.degree = degree          # polynomial degree for feature transforming
        self.gamma = gamma            # discount factor
        self.epsilon = epsilon        # epsilon-greedy rate
        self.alpha = alpha            # learning rate
        self.num_match = 0            # number of total matches
        self.annotation = annotation  # annotation added in model file name
        self.weight = None            # will be initialized in init_and_load()

        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=False)
        # features consist of two parts:
        # 1. features of the afterstate if it is a non-end-turn action. (simulate the action and resulting world)
        # 2. features of the current state if it is a end-turn action
        poly_k = self.poly.fit_transform(numpy.zeros((1, len(self.feat_names)))).shape[1]
        # num of total features. First part for non-end-turn action. Second part for end-turn action.
        self.k = poly_k * 2
        self.init_and_load()

    def init_weight(self):
        return numpy.zeros(self.k)

    def __repr__(self):
        # print feature weights
        return 'num_match:' + str(self.num_match) + "," + \
               ','.join(map(lambda name_w: name_w[0] + ':' + str(name_w[1]), zip(self.feat_names, self.weight)))

    def file_name(self):
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_linear_{4}". \
            format(constant.ql_linear_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)
        return file_name

    def init_and_load(self):
        # load q values weight
        if os.path.isfile(self.file_name()):
            with open(self.file_name(), 'rb') as f:
                self.gamma, self.epsilon, self.alpha, self.num_match, self.weight = pickle.load(f)
        else:
            self.weight = self.init_weight()

    def save(self):
        file_name = self.file_name()
        with open(file_name, 'wb') as f:
            pickle.dump((self.gamma, self.epsilon, self.alpha,
                         self.num_match, self.weight), f, protocol=4)

    def post_match(self):
        """ perform model storage, or output some info """
        self.num_match += 1
        logger.warning('QValueLinearApprox post match:' + str(self))
        # don't save any update during test
        if not self.player.test and self.num_match % constant.ql_linear_save_freq == 0:
            self.save()

    def max_qvalue(self, game_world: 'GameWorld') -> float:
        """ max_a Q(s,a)"""
        all_acts = self.player.search_one_action(game_world)
        max_state_qvalue = max(map(lambda action: self.qvalue(game_world, action), all_acts))
        return max_state_qvalue

    def qvalue(self, game_world: 'GameWorld', action: 'Action', return_feature=False):
        """ Q(s,a)
         feature(s,a) are afterstate, i.e., the state after action is acted on game_world """
        features = self.to_feature(game_world, action)
        qvalue = numpy.dot(features, self.weight)

        if return_feature:
            return qvalue, features
        else:
            return qvalue

    def raw_feature_to_full_feature(self, features, action: 'Action') -> numpy.ndarray:
        """
        full features consist of two parts:
        # 1. features of the afterstate if it is a non-end-turn action. (simulate the action and resulting world)
        # 2. features of the current state if it is an end-turn action
        """
        # Compare to parent implementation, QValueLinearApprox has poly transform features
        features = self.poly.fit_transform(numpy.expand_dims(features, axis=0)).reshape(-1)
        if isinstance(action, NullAction):
            features = numpy.hstack((numpy.zeros_like(features), features))
        else:
            features = numpy.hstack((features, numpy.zeros_like(features)))
        return features

    def update(self, last_state: 'GameWorld', last_act: 'Action', new_game_world: 'GameWorld',
               r: float, match_end: bool, test: bool):
        """ w <- w + alpha * (R + gamma * max_a'Q(s', a') - Q(s,a)) * feature(s,a) """
        # q value is based on linear combination of weights and features
        old_qvalue, old_qvalue_features = self.qvalue(last_state, last_act, return_feature=True)

        # if match end, max_new_state_qvalue eqauls to zero
        if match_end:
            max_new_state_qvalue = 0
        else:
            max_new_state_qvalue = self.max_qvalue(new_game_world)

        predict_qvalue = r + self.gamma * max_new_state_qvalue
        delta = predict_qvalue - old_qvalue
        old_weight = self.weight.copy()
        new_weight = self.alpha * delta * old_qvalue_features + self.weight

        logger.info("Q-learning update. this state: %r, this action: %r" %
                    (self.state2str(last_state), self.action2str(last_act)))
        logger.info("Q-learning update. this state feature: %r" % old_qvalue_features)
        logger.info("Q-learning update. new_state: %r, max_new_state_qvalue: %f" %
                    (self.state2str(new_game_world), max_new_state_qvalue))
        logger.info("Q-learning update. w <- w + alpha * (R + gamma * max_a'Q(s', a') - Q(s,a)) * feature(s,a):\n"
                    "{0} <- \n{1} \n + {2} * ({3} + {4} * {5} - {6}) * {7}"
                    .format(new_weight, old_weight, self.alpha, r, self.gamma,
                            max_new_state_qvalue, old_qvalue, old_qvalue_features))

        # only update in training phase
        if not test:
            self.weight = new_weight


