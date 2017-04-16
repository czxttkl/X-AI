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
        str = "Q-table:\n"
        for state_str, act_qvalue in self.qvalues_tab.items():
            str += state_str + "\n"
            for act_str, (qvalue, num_sa) in act_qvalue.items():
                str += '\t{0}={1}, {2}\n'.format(act_str, qvalue, num_sa)
        return str

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

        logger.info("Q-learning update. this state: %r, this action: %r" % (last_state_str, last_act_str))
        logger.info(
            "Q-learning update. new_state_str: %r, max_new_state_qvalue: %f" % (new_state_str, max_new_state_qvalue))
        logger.info("Q-learning update. Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s', a')]:   "
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
        self.epsilon = epsilon
        self.test = test
        self.last_state = None
        self.last_act = None

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

        # feature names
        self.feat_names = [
                           # feature index 0 - 9
                           'turn', 'self_h', 'oppo_h', 'self_m',
                           'self_max_m', 'oppo_next_m',         # self maximum mana this turn, oppo mana next turn
                           'self_rem_cards', 'oppo_rem_cards',  # number of remaining cards of self and oppo
                           'self_mn', 'oppo_mn',                # number of minions of self and oppo

                           # feature index 10 - 13
                           'self_mn_taunt', 'oppo_mn_taunt',    # number of minions in taunt
                           'self_mn_divine', 'oppo_mn_divine',  # number of minions in divine

                           # feature index 14 - 22
                           'self_tt_mn_attkable', 'self_tt_mn_attk',   # self total minion (attackable) attack
                           'self_tt_mn_h',   # self total minion health
                           'self_tt_mn_taunt_attkable', 'self_tt_mn_taunt_attk', 'self_tt_mn_taunt_h',
                           'self_tt_mn_divine_attkable', 'self_tt_mn_divine_attk', 'self_tt_mn_divine_h',

                           # feature index 23 - 28
                           'oppo_tt_mn_attk', 'oppo_tt_mn_h',
                           'oppo_tt_mn_taunt_attk', 'oppo_tt_mn_taunt_h',
                           'oppo_tt_mn_divine_attk', 'oppo_tt_mn_divine_h',
                           ]

        self.k = len(self.feat_names)         # number of features
        # weights for features:
        self.weight = numpy.zeros(self.k)
        self.load()

    def state2str(self, game_world: 'GameWorld') -> str:
        return 'num_match:' + str(self.num_match) + "," + \
               ','.join(map(lambda name_f: name_f[0] + ':' + str(name_f[1]), zip(self.feat_names, self.extract_features(game_world))))

    @ staticmethod
    def action2str(action: 'Action') -> str:
        return str(action)

    def feature2str(self, feature: 'numpy.ndarray') -> str:
        return 'num_match:' + str(self.num_match) + "," + \
                                                    ','.join(map(lambda name_f: name_f[0] + ':' + str(name_f[1]),
                                                                 zip(self.feat_names, feature)))

    def __repr__(self):
        # print feature weights
        return 'num_match:' + str(self.num_match) + "," + \
               ','.join(map(lambda name_w: name_w[0] + ':' + str(name_w[1]), zip(self.feat_names, self.weight)))

    def file_name(self):
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}_{4}". \
            format(constant.ql_linear_data_path, self.gamma, self.epsilon, self.alpha, self.annotation)
        return file_name

    def load(self):
        file_name = self.file_name()

        # load q values table
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.gamma, self.epsilon, self.alpha, self.num_match, self.weight = pickle.load(f)

    def save(self):
        file_name = self.file_name()
        with open(file_name, 'wb') as f:
            pickle.dump((self.gamma, self.epsilon, self.alpha,
                         self.num_match, self.weight), f, protocol=4)

    def post_match(self):
        self.num_match += 1
        logger.warning(str(self))
        # don't save any update during test
        if not self.player.test and self.num_match % constant.ql_linear_save_freq == 0:
            self.save()

    def max_qvalue(self, game_world: 'GameWorld') -> float:
        """ max_a Q(s,a)"""
        all_acts = self.player.search_one_action(game_world)
        max_state_qvalue = max(map(lambda action: self.qvalue(game_world, action), all_acts))
        return max_state_qvalue

    def qvalues(self, game_world: 'GameWorld', actions: List['Action']) -> List[float]:
        """ Q(s,a) for a in actions """
        res = map(lambda act: self.qvalue(game_world, action=act), actions)
        # actually return an iterator
        return res

    def qvalue(self, game_world: 'GameWorld', action: 'Action', return_feature=False) -> float:
        """ Q(s,a)
         feature(s,a) are afterstate, i.e., the state after action is acted on game_world """
        features = self.extract_features(action.virtual_apply(game_world))
        # linear combination of features and weights
        if return_feature:
            return numpy.dot(features, self.weight), features
        else:
            return numpy.dot(features, self.weight)

    def extract_features(self, game_world: 'GameWorld') -> numpy.ndarray:
        """ extract features from game_world """
        feature = numpy.zeros(self.k)
        feature[0] = game_world.turn
        feature[1] = game_world[self.player.name]['health']
        feature[2] = game_world[self.player.opponent.name]['health']
        feature[3] = game_world[self.player.name]['mana']
        feature[4] = self.player.max_mana_this_turn(game_world.turn)
        feature[5] = self.player.opponent.max_mana_this_turn(game_world.turn + 1)
        feature[6] = len(game_world[self.player.name]['inhands'])
        feature[7] = len(game_world[self.player.opponent.name]['inhands'])
        feature[8] = len(game_world[self.player.name]['intable'])
        feature[9] = len(game_world[self.player.opponent.name]['intable'])

        self_mn_taunt, oppo_mn_taunt, self_mn_divine, oppo_mn_divine, \
            self_tt_mn_attkable, self_tt_mn_attk, self_tt_mn_h, \
            self_tt_mn_taunt_attkable, self_tt_mn_taunt_attk, self_tt_mn_taunt_h, \
            self_tt_mn_divine_attkable, self_tt_mn_divine_attk, self_tt_mn_divine_h, \
            oppo_tt_mn_attk, oppo_tt_mn_h, \
            oppo_tt_mn_taunt_attk, oppo_tt_mn_taunt_h, \
            oppo_tt_mn_divine_attk, oppo_tt_mn_divine_h = [0] * 19

        self_intable = game_world[self.player.name]['intable']
        oppo_intable = game_world[self.player.opponent.name]['intable']

        # process self table
        for card in self_intable:
            self_tt_mn_attk += card.attack
            self_tt_mn_h += card.health
            if not card.used_this_turn:
                self_tt_mn_attkable += card.attack
            if card.taunt:
                self_mn_taunt += 1
                self_tt_mn_taunt_h += card.health
                self_tt_mn_taunt_attk += card.attack
                if not card.used_this_turn:
                    self_tt_mn_taunt_attkable += card.attack
            if card.divine:
                self_mn_divine += 1
                self_tt_mn_divine_h += card.health
                self_tt_mn_divine_attk += card.attack
                if not card.used_this_turn:
                    self_tt_mn_divine_attkable += card.attack

        # process oppo intable
        for card in oppo_intable:
            oppo_tt_mn_attk += card.attack
            oppo_tt_mn_h += card.health
            if card.taunt:
                oppo_mn_taunt += 1
                oppo_tt_mn_taunt_h += card.health
                oppo_tt_mn_taunt_attk += card.attack
            if card.divine:
                oppo_mn_divine += 1
                oppo_tt_mn_divine_h += card.health
                oppo_tt_mn_divine_attk += card.attack

        feature[10] = self_mn_taunt
        feature[11] = oppo_mn_taunt
        feature[12] = self_mn_divine
        feature[13] = oppo_mn_divine
        feature[14] = self_tt_mn_attkable
        feature[15] = self_tt_mn_attk
        feature[16] = self_tt_mn_h
        feature[17] = self_tt_mn_taunt_attkable
        feature[18] = self_tt_mn_taunt_attk
        feature[19] = self_tt_mn_taunt_h
        feature[20] = self_tt_mn_divine_attkable
        feature[21] = self_tt_mn_divine_attk
        feature[22] = self_tt_mn_divine_h
        feature[23] = oppo_tt_mn_attk
        feature[24] = oppo_tt_mn_h
        feature[25] = oppo_tt_mn_taunt_attk
        feature[26] = oppo_tt_mn_taunt_h
        feature[27] = oppo_tt_mn_divine_attk
        feature[28] = oppo_tt_mn_divine_h

        return feature

    def update(self, last_state: 'GameWorld', last_act: 'Action', new_game_world: 'GameWorld', R: float):
        """ update Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s',a')] """
        # q value is based on linear combination of weights and afterstate, the state after last_act
        # is performed on last_state. however, if last_act is NullAction, it is nondeterministic to know
        # what is the next state. In current implementation, NullAction.apply_virtual return the same
        # game world. We use new_game_world applied by EndTurn as the afterstate
        if isinstance(last_act, NullAction):
            old_qvalue, old_qvalue_features = self.qvalue(new_game_world, last_act, return_feature=True)
        else:
            old_qvalue, old_qvalue_features = self.qvalue(last_state, last_act, return_feature=True)

        max_new_state_qvalue = self.max_qvalue(new_game_world)
        predict_qvalue = R + self.gamma * max_new_state_qvalue
        delta = predict_qvalue - old_qvalue
        old_weight = self.weight.copy()
        self.weight += self.alpha * delta * old_qvalue_features

        logger.warning("Q-learning update. this state: %r, this action: %r" %
                       (self.state2str(last_state), self.action2str(last_act)))
        logger.warning("Q-learning update. res  state: %r" %
                       self.feature2str(old_qvalue_features))
        # logger.warning("Q-learning update. new_state: %r, max_new_state_qvalue: %f" %
        #                (self.state2str(new_game_world), max_new_state_qvalue))
        logger.warning("Q-learning update. w <- w + alpha * (R + gamma * max_a'Q(s', a')) * feature(s,a): "
                       "{0} <- {1} + {2} * ({3} + {4} * {5}) * {6}"
                       .format(self.weight, old_weight, self.alpha, R, self.gamma,
                               max_new_state_qvalue, old_qvalue_features))

