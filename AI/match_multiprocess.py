from actions import NullAction
from card import *
import logging
from player.player import Player
from game_world import GameWorld


logger = logging.getLogger('hearthstone')


class Match:

    def __init__(self, player1: 'Player', player2: 'Player'):
        self.player1 = player1
        self.player2 = player2
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1
        self.winner = None
        self.winner_reason = None

    def play_one_match(self, match_idx):
        turn = 0

        while True:
            turn += 1
            player = self.player1 if turn % 2 else self.player2

            match_end = player.turn_begin_init(turn)      # update mana, etc. at the beginning of a turn
            game_world = GameWorld(self.player1, self.player2, turn)
            logger.info("Turn {0}. {1}".format(turn, player))

            # match end due to insufficient deck to draw
            if match_end:
                self.winner = player.opponent
                self.winner_reason = "%r has no card to draw" % player.name
                break

            if turn > 2:
                # update the last end-turn action's Q value
                player.post_action(game_world, match_end=False, winner=False)

            # one action search
            act = player.search_and_pick_action(game_world)
            while not isinstance(act, NullAction):
                act.apply(game_world)
                game_world.update_player(self.player1, self.player2)
                logger.info(game_world)
                match_end = self.check_for_match_end(game_world)
                if match_end:
                    break
                player.post_action(game_world, match_end=False, winner=False)
                act = player.search_and_pick_action(game_world)

            if match_end:
                break

            logger.info("")

        self.post_one_match(game_world, match_idx)
        return self.winner

    def post_one_match(self, game_world, match_idx):
        if self.winner == self.player1:
            self.match_end(game_world=game_world, winner=self.player1, loser=self.player2,
                           match_idx=match_idx, reason=self.winner_reason)
        else:
            self.match_end(game_world=game_world, winner=self.player2, loser=self.player1,
                           match_idx=match_idx, reason=self.winner_reason)
        self.player1.post_match()
        self.player2.post_match()

    def match_end(self, game_world, winner, loser, match_idx, reason):
        logger.warning("%dth match ends at turn %d. winner=%r, loser=%r, reason=%r" %
                       (match_idx, game_world.turn, winner.name, loser.name, reason))
        winner.post_action(game_world, match_end=True, winner=True)
        loser.post_action(game_world, match_end=True, winner=False)
        self.winner = winner

    def check_for_match_end(self, game_world: 'GameWorld') -> bool:
        """ return True if the match ends. Otherwise return False """
        if game_world.health(self.player1) <= 0:
            self.winner = self.player2
            self.winner_reason = "player1 health<=0"
            return True
        elif game_world.health(self.player2) <= 0:
            self.winner = self.player1
            self.winner_reason = "player2 health<=0"
            return True
        else:
            return False
