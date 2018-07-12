from player.player import Player
import random
import logging

logger = logging.getLogger('hearthstone')


class RandomPlayer(Player):
    """ A player always picks a random action from available actions """

    def pick_action(self, all_acts, game_world):
        for i, act in enumerate(all_acts):
            logger.info("Choice %d (%.2f): %r" % (i, 1./len(all_acts), act))
            # cheat by letting random player plays heropower and spell
            # whenever possible in first few turns
            # if game_world.turn == 3 and isinstance(act, HeroPowerAttack):
            #     logger.info("CHEAT: %r pick %r\n" % (self.name, act))
            #     return act
            # elif game_world.turn == 7 and isinstance(act, SpellPlay):
            #     logger.info("CHEAT: %r pick %r\n" % (self.name, act))
            #     return act
        act = random.choice(all_acts)
        logger.info("%r pick %r\n" % (self.name, act))
        return act
