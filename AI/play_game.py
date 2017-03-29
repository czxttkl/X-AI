from match import *
import logging


def main():
    logger = logging.getLogger('hearthstone')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.WARNING)
    match = Match()
    match.play_N_match(n=2000)

if __name__ == "__main__":
    main()
