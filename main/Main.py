import os

import deuces

import numpy as np
from deuces import Evaluator
from sklearn.cluster import DBSCAN
import cv2

from main import Cards


def main(img):

    path = os.path.dirname(os.path.abspath(__file__))
    path = path[0: len(path)-4]
    train_ranks = Cards.load_ranks(path + "/card_images/")
    train_suits = Cards.load_suits(path + "/card_images/")

    image = cv2.imread(img, 1)

    image = Cards.resize_image(image)

    pre_proc = Cards.preprocess_image(image)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # ako ne nadje konture, ne radi nista

    if len(cnts_sort) != 0:

        # inicijalizujemo praznu listu karata
        # k predstavlja indeks liste
        cards = []
        k = 0

        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                # kreira card objekat od konture i dodaje ga u listu karata
                # pronalazi konturu sa brojem i znakom karte
                cards.append(Cards.preprocess_card(cnts_sort[i], image))

                # pronalazi najbolji odgovarajuci broj i znak
                cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[
                    k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

                # crta centar karte i rezultat
                image = Cards.draw_results(image, cards[k])
                k = k + 1

        centers = []

        # crta konture karata na slici
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)

                M = cv2.moments(cards[i].contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY])

            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

        # klasterujemo centre karata u tri grupe: prvi igrac, drugi igrac, karte na stolu
        X = np.array(centers)
        clustering = DBSCAN(eps=350, min_samples=1).fit(X)

        groups = {0: [], 1: [], 2: []}

        for card, label in zip(cards, clustering.labels_):
            groups[label].append(card)

        # pravimo prazne liste u koje dodajemo deuces.Card objekte pomocu kojih racunamo snagu ruke oba igraca
        board = []
        player1 = []
        player2 = []

        for board_card in groups[1]:
            board.append(deuces.Card.new(Cards.convert_name(board_card)))

        for board_card in groups[0]:
            player1.append(deuces.Card.new(Cards.convert_name(board_card)))

        for board_card in groups[2]:
            player2.append(deuces.Card.new(Cards.convert_name(board_card)))

        # pomocu deuces.Evaluator racunamo i poredimo snage ruku
        # stampamo obe ruke
        evaluator = Evaluator()
        player1_score = evaluator.evaluate(board, player1)
        player2_score = evaluator.evaluate(board, player2)
        print("Player 1: ", evaluator.class_to_string(evaluator.get_rank_class(player1_score)))
        print(groups[0][0].best_rank_match + " of " + groups[0][0].best_suit_match, ", ", groups[0][1].best_rank_match + " of " + groups[0][1].best_suit_match)
        print("Player 2: ", evaluator.class_to_string(evaluator.get_rank_class(player2_score)))
        print(groups[2][0].best_rank_match + " of " + groups[2][0].best_suit_match, ", ", groups[2][1].best_rank_match + " of " + groups[2][1].best_suit_match)
        if player1_score == player2_score:
            print("Draw")
        elif player1_score < player2_score:
            print("Player 1 wins")
        else:
            print("Player 2 wins")

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='image path')
    parser.add_argument('--img', metavar='path', required=True,
                        help='the path to image')
    args = parser.parse_args()
    main(img=args.img)
