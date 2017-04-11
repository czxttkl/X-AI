from match import *
import logging


def main():
    logger = logging.getLogger('hearthstone')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    match = Match()
    match.play_N_match(n=3)

if __name__ == "__main__":
    main()
