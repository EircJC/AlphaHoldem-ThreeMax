"""牌的表示：整数 0..51；rank = card // 4 (0..12 = 2..A)，suit = card % 4 (0..3 = s,h,d,c)。"""
from __future__ import annotations
import random

RANKS = 13
SUITS = 4
_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "shdc"


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def card_str(card: int) -> str:
    return _RANK_CHARS[card_rank(card)] + _SUIT_CHARS[card_suit(card)]


class Deck:
    """一副 52 张牌，可发牌。"""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.cards = list(range(52))
        self.rng.shuffle(self.cards)
        self._i = 0

    def deal(self, n: int) -> list[int]:
        out = self.cards[self._i:self._i + n]
        self._i += n
        return out
