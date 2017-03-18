import sys

sys.path.append("..")
from fireplace import cards
from fireplace.exceptions import GameOver
from fireplace.utils import play_full_game


def test_full_game():
    try:
        play_full_game()
    except GameOver:
        print("Game completed normally.")


def main():
    cards.db.initialize()
    test_full_game()


if __name__ == "__main__":
    main()
