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
from collections import namedtuple


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
    def __init__(self, player, gamma, epsilon, alpha):
        self.player = player
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.state_act_visit_times = 0      # times visiting state-action pairs
        self.num_match = 0                  # number of total matches

        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}". \
            format(constant.qltabqvalues, self.gamma, self.epsilon, self.alpha)

        # load q values table
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.gamma, self.epsilon, self.alpha, \
                self.num_match, self.state_act_visit_times, self.qvalues_tab = pickle.load(f)
        else:
            # key: state_str, value: dict() with key as act_str and value as (Q(s,a), # of times visiting (s,a)) tuple
            self.qvalues_tab = dict()

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
        # don't save any update during test
        if not self.player.test and self.num_match % constant.qltab_save_freq == 0:
            self.save()

    def save(self):
        t1 = time.time()
        file_name = "{0}_gamma{1}_epsilon{2}_alpha{3}". \
            format(constant.qltabqvalues, self.gamma, self.epsilon, self.alpha)
        with open(file_name, 'wb') as f:
            pickle.dump((self.gamma, self.epsilon, self.alpha,
                         self.num_match, self.state_act_visit_times, self.qvalues_tab), f, protocol=4)
        logger.warning("save q values to disk in %d seconds" % (time.time() - t1))

    def qvalue(self, state_str, act_str):
        """ Q(s,a) """
        return self.qvalues_tab.get(state_str, dict()).get(act_str, (0, 0))[0]

    def count(self, state_str, act_str):
        """ number of visits at (s,a) """
        return self.qvalues_tab.get(state_str, dict()).get(act_str, (0, 0))[1]

    def max_qvalue(self, game_world):
        """ max_a Q(s,a)"""
        state_str = self.player.state2str(game_world)
        all_acts = self.player.search_one_action(game_world)
        all_acts_strs = list(map(lambda x: self.player.action2str(x), all_acts))
        max_state_qvalue = max(map(lambda act_str: self.qvalue(state_str, act_str), all_acts_strs))
        return max_state_qvalue

    def update(self, last_state_str, last_act_str, new_game_world, R):
        """ update Q(s,a) <- (1-alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s',a')] """
        new_state_str = self.player.state2str(new_game_world)

        # determine max Q(s',a')
        max_new_state_qvalue = self.max_qvalue(new_game_world)

        # not necessary to write to qvalues_tab if
        # R == 0 and  max Q(s',a) == 0 and Q(s,a) == 0
        if R == 0 and max_new_state_qvalue == 0 \
                and self.qvalue(last_state_str, last_act_str) == 0:
            return

        if not self.qvalues_tab.get(last_state_str):
            self.qvalues_tab[last_state_str] = dict()

        update_count = self.count(last_state_str, last_act_str) + 1
        alpha = self.alpha / update_count
        old_qvalue = self.qvalue(last_state_str, last_act_str)
        update_qvalue = \
            (1 - alpha) * self.qvalue(last_state_str, last_act_str) + \
                alpha * (R + self.gamma * max_new_state_qvalue)
        self.qvalues_tab[last_state_str][last_act_str] = (update_qvalue, update_count)

        self.state_act_visit_times += 1
        logger.warning("Q-learning update. this state: %r, this action: %r" % (last_state_str, last_act_str))
        logger.warning(
            "Q-learning update. new_state_str: %r, max_new_state_qvalue: %f" % (new_state_str, max_new_state_qvalue))
        logger.warning("Q-learning update. Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * [R + gamma * max_a' Q(s', a')]:   "
                       "{0} <- (1 - {1}) * {2} + {1} * [{3} + {4} * {5}], # of (s,a) visits: {6}".format
                       (update_qvalue, alpha, old_qvalue, R, self.gamma, max_new_state_qvalue, update_count))


class QLearningTabularPlayer(Player):
    """ A player picks action based on Q-learning tabular method. """

    def _init_player(self, **kwargs):
        gamma = kwargs['gamma']               # discounting factor
        epsilon = kwargs['epsilon']           # epsilon-greedy
        alpha = kwargs['alpha']               # learning rate
        test = kwargs.get('test', False)      # whether in test mode
        self.test = test
        self.epsilon = epsilon
        self.qvalues_tab = QValueTabular(self, gamma, epsilon, alpha)

    def state2str(self, game_world):
        """ convert the game world into a short string, which will be used as index in q-value table. """
        state_str = "self h:{0}, m:{1}, rem_deck:{2}, hp_used: {3}, oppo h:{4}, mana next turn:{5}, rem_deck:{6}".\
            format(game_world[self.name]['health'], game_world[self.name]['mana'], game_world[self.name]['rem_deck'],
                   int(game_world[self.name]['heropower'].used_this_turn),
                   game_world[self.opponent.name]['health'], self.mana_based_on_turn(game_world.turn + 1),
                   game_world[self.opponent.name]['rem_deck'])

        # only use cidx list to represent self inhands cards
        inhands_str = "self-inhands:" + ','.join(map(lambda x: str(x.cidx), sorted(game_world[self.name]['inhands'])))

        # use (cidx, attack, health, divine, taunt) tuple lists to represent intable cards
        intable_str = "self-intable:" + \
                      ','.join(map(lambda x: '({0}, {1}, {2}, {3}, {4})'.
                                   format(x.cidx, x.attack, x.health, int(x.divine), int(x.taunt)),
                                   sorted(game_world[self.name]['intable'])))
        oppo_intable_str = "oppo-intable:" + \
                           ','.join(map(lambda x: '({0}, {1}, {2}, {3}, {4})'.
                                    format(x.cidx, x.attack, x.health, int(x.divine), int(x.taunt)),
                                    sorted(game_world[self.opponent.name]['intable'])))
        return state_str + ", " + inhands_str + ", " + intable_str + ", " + oppo_intable_str

    def action2str(self, action):
        return str(action)

    def pick_action(self, all_acts, game_world) -> 'Action':
        state_str = self.state2str(game_world)

        if len(all_acts) == 1:
            choose_act = all_acts[0]
            choose_act_str = self.action2str(choose_act)
            logger.info("Choice 0: %r" % choose_act)
        else:
            qvalue_tuples = list()
            qvalue_tuple = namedtuple('qvalue_tuple', ['act_idx', 'act_str', 'qvalue'])
            for act_idx, act in enumerate(all_acts):
                act_str = self.action2str(act)
                qvalue = self.qvalues_tab.qvalue(state_str, act_str)
                qvalue_tuples.append(qvalue_tuple(act_idx=act_idx, act_str=act_str, qvalue=qvalue))
                logger.info("Choice %d (%.2f): %r" % (act_idx, qvalue, act))

            choose_i, choose_act_str, choose_qvalue = self.epsilon_greedy(qvalue_tuples)
            choose_act = all_acts[choose_i]

        self.last_state_str = state_str
        self.last_act_str = choose_act_str

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
        self.qvalues_tab.update(self.last_state_str, self.last_act_str, new_game_world, R)

    def post_match(self):
        """ called when a match finishes """
        self.qvalues_tab.post_match()
        self.print_qtable_summary()

    def epsilon_greedy(self, qvalue_tuples):
        """ q-values format: a list of namedtuples: qvalue_tuple(act_idx, act_str, qvalue) """
        # shuffle qvalue_tuples so that max function will break tie randomly
        qvalue_tuples_shuffled = random.sample(qvalue_tuples, len(qvalue_tuples))
        max_act_idx, max_act_str, max_qvalue = max(qvalue_tuples_shuffled, key=lambda x: x.qvalue)
        # if in test mode, do not explore, just exploit
        if self.test:
            return qvalue_tuples[max_act_idx]
        else:
            acts_weights = numpy.full(shape=(len(qvalue_tuples)), fill_value=self.epsilon / (len(qvalue_tuples) - 1))
            acts_weights[max_act_idx] = 1. - self.epsilon
            idx_list = list(range(len(qvalue_tuples)))
            choose_idx = numpy.random.choice(idx_list, 1, replace=False, p=acts_weights)[0]
            return qvalue_tuples[choose_idx]

    def print_qtable(self):
        """ print q-value table """
        logger.warning(str(self.qvalues_tab))

    def print_qtable_summary(self):
        """ print qtable summary """
        logger.warning("total match number: %d, qvalue table states  %d, state-action visit total %d"
                       % (self.qvalues_tab.num_match, len(self.qvalues_tab), self.qvalues_tab.state_act_visit_times))

