import numpy as np
import cv2

# adaptivni threshold
BKG_THRESH = 60
CARD_THRESH = 30

# sirina i visina coska karata, gde se nalaze broj i znak
CORNER_WIDTH = 50
CORNER_HEIGHT = 120

# dimenzije slika za ucenje broja
RANK_WIDTH = 65
RANK_HEIGHT = 100

# dimenzije slika za ucenje znaka
SUIT_WIDTH = 65
SUIT_HEIGHT = 75

RANK_DIFF_MAX = 4000
SUIT_DIFF_MAX = 5000

CARD_MAX_AREA = 150000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX


class Query_card:

    def __init__(self):
        self.contour = []
        self.width, self.height = 0, 0
        self.corner_pts = []
        self.center = []
        self.warp = []
        self.rank_img = []
        self.suit_img = []
        self.best_rank_match = "Unknown"
        self.best_suit_match = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0


class Train_ranks:

    def __init__(self):
        self.img = []
        self.name = "Placeholder"


class Train_suits:

    def __init__(self):
        self.img = []
        self.name = "Placeholder"


# ucitava slike za treniranje broja
def load_ranks(filepath):

    train_ranks = []
    i = 0

    for Rank in ['Ace', '2', '3', '4', '5', '6', '7',
                 '8', '9', '10', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.png'
        train_ranks[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks


# ucitava slike za treniranje znaka
def load_suits(filepath):
    train_suits = []
    i = 0

    for Suit in ['spades', 'diamonds', 'clubs', 'hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.png'
        train_suits[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits


def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # uzima se jedan pixel u centru vrha slike kako bi se odredio njen intenzitet
    # adaptivni threshold se postavlja za 50 vise od toga, sto mu omogucava da se prilagodi osvetljenju
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def find_cards(thresh_image):

    # pronalazi konture i sortira im indekse po velicini
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    if len(cnts) == 0:
        return [], []

    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # pronalazenje kontura koje su karte tako sto gledamo da li su manje od max velicine karte, vece od min velicine,
    # nemaju roditelje, imaju 4 coska

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


def preprocess_card(contour, image):

    qCard = Query_card()

    qCard.contour = contour

    # pronalazi okvir karte kako bi oznacio coskove
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # pronalazi dimenzije pravougaonika nacrtanog oko karte
    x, y, w, h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # pronalazi centar karte
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # transformise kartu u ravnu 200x300 sliku
    qCard.warp = flattener(image, pts, w, h)

    # pronalazi cosak karte i zumira 4x
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

    # racuna threshold pomocu intenziteta belog piksela
    white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    # deli cosak na dva dela, gornji sa brojem i donji sa znakom
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # pronalazi konturu broja
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

    # menja velicinu konture kako bi odgovarala velicini trening skupa
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1 + h1, x1:x1 + w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # isto se radi i sa znakom
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)

    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2 + h2, x2:x2 + w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard


def match_card(qCard, train_ranks, train_suits):

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):

        # gleda se razlika izmedju pronadjenog broja i svakog broja iz trening skupa,
        # pamti se onaj sa najmanjom razlikom
        for Trank in train_ranks:
            diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img) / 255)

            if rank_diff < best_rank_match_diff:
                best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

        # isto se radi i sa znakom
        for Tsuit in train_suits:

            diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img) / 255)

            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_name = Tsuit.name

    # spajaju se broj i znak, kako bi se odredila karta
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff


def draw_results(image, qCard):

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    cv2.putText(image, (rank_name + ' of'), (x - 60, y - 10), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, (rank_name + ' of'), (x - 60, y - 10), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, suit_name, (x - 60, y + 25), font, 1, (50, 200, 200), 2, cv2.LINE_AA)

    return image


# odradjen pomocu primera sa linka www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def flattener(image, pts, w, h):
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8 * h:
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h:
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8 * h and w < 1.2 * h:
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0]
            temp_rect[1] = pts[0][0]
            temp_rect[2] = pts[3][0]
            temp_rect[3] = pts[2][0]

        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0]
            temp_rect[1] = pts[3][0]
            temp_rect[2] = pts[2][0]
            temp_rect[3] = pts[1][0]

    maxWidth = 200
    maxHeight = 300

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp


# skaliranje slike na 1920x1080
def resize_image(image):
    h, w, _ = image.shape

    if h > w:
        w, h = h, w
        image = np.rot90(image)

    image = cv2.resize(image, (0, 0), fx=1920 / w, fy=1080 / h)
    return image


# izmena imena karte kako bi se iskoristilo u deuces
def convert_name(card):
    name = ''
    if card.best_rank_match == 'Ace': name += 'A'
    elif card.best_rank_match == 'King': name += 'K'
    elif card.best_rank_match == 'Queen': name += 'Q'
    elif card.best_rank_match == 'Jack': name += 'J'
    elif card.best_rank_match == '10': name += 'T'
    else: name += str(card.best_rank_match)
    name += card.best_suit_match[0]
    return name