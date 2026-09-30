"""7 张牌的牌力评估：返回可比较的元组，越大越强。
类别: 8直花 7四条 6葫芦 5同花 4顺子 3三条 2两对 1一对 0高牌。
实现方式：枚举 C(7,5)=21 组，取最强 5 张。带缓存。"""
from itertools import combinations
from functools import lru_cache
from .cards import card_rank, card_suit


def _rank5(cards5):
    """给 5 张牌打分，返回 (类别, 主排序..., 踢脚...) 元组。"""
    ranks = sorted((card_rank(c) for c in cards5), reverse=True)
    suits = [card_suit(c) for c in cards5]
    is_flush = len(set(suits)) == 1

    # 统计点数出现次数
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # 按 (出现次数, 点数) 降序：先比数量，再比大小
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    count_pattern = tuple(c for _, c in by_count)      # 如 (3,2)=葫芦
    ordered_ranks = tuple(r for r, _ in by_count)      # 对应点数

    # 顺子判定（含 A-5 轮子）
    uniq = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:   # A,5,4,3,2 轮子，A 当低
            straight_high = 3

    if is_flush and straight_high is not None:
        return (8, straight_high)
    if count_pattern == (4, 1):
        return (7,) + ordered_ranks
    if count_pattern == (3, 2):
        return (6,) + ordered_ranks
    if is_flush:
        return (5,) + tuple(ranks)
    if straight_high is not None:
        return (4, straight_high)
    if count_pattern == (3, 1, 1):
        return (3,) + ordered_ranks
    if count_pattern == (2, 2, 1):
        return (2,) + ordered_ranks
    if count_pattern == (2, 1, 1, 1):
        return (1,) + ordered_ranks
    return (0,) + tuple(ranks)


@lru_cache(maxsize=200000)
def _eval_frozen(cards_tuple):
    best = None
    for combo in combinations(cards_tuple, 5):
        s = _rank5(combo)
        if best is None or s > best:
            best = s
    return best


def evaluate7(cards):
    """输入 5~7 张牌（list[int]），返回可比较分数元组，越大越强。"""
    return _eval_frozen(tuple(sorted(cards)))
