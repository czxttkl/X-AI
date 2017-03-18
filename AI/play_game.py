import sys

sys.path.append("..")
from fireplace import cards
from fireplace.exceptions import GameOver
from fireplace.utils import play_full_game
from fireplace.utils import setup_game, setup_game_fix_player_fix_deck


def test_full_game():
    try:
        # game = setup_game()
        game = setup_game_fix_player_fix_deck()
        play_full_game(game)
    except GameOver:
        print("Game completed normally.")


def main():
    print("start")
    cards.db.initialize()
    test_full_game()


if __name__ == "__main__":
    main()
