"""多路(≥3 人进翻牌)翻后兜底 —— 保守但正确的启发式。

为什么需要:HU 模型假设只有一个对手,多路翻后【完全无效】。松散对手(跟注站/松弱)常把底池打成多路,
所以必须有个兜底策略。多人博弈无 GTO 保证,本模块只做"强启发式":价值导向、不诈唬、面对下注收紧。

核心正确性:必须拿到【真实 to_call / pot / stack】(见 table3_bot 的多路支路重建),
否则会把"面对下注"误判成"无人下注"而拿空气跟注 → 每手倒 ~9bb。

手牌强度分档(_classify):
  - strong:超对 / 顶对 / 两对 / 三条(含 set)—— 可价值下注、可跟注。
  - weak:  中低对(未超对、非顶对)—— 只跟小注,不主动下注。
  - air:   未成对 —— 过牌;面对任何下注弃(多路不追无位置听牌)。

接口:multiway_action(hero_hole, board, to_call, pot, stack) -> ("fold"|"call"|"raise", to_bb or None)
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.cards import card_rank                            # noqa: E402(只读复用,不改)


def _classify(hero_hole, board):
    """英雄两张对当前牌面的成手强度:'strong' / 'weak' / 'air'。
    只看英雄真正参与凑成的牌(避免把纯公共牌成对算成英雄成手)。"""
    if len(board) < 3:
        return "air"
    h = [card_rank(c) for c in hero_hole]
    branks = [card_rank(c) for c in board]
    bmax = max(branks)
    bcount = {r: branks.count(r) for r in set(branks)}
    if h[0] == h[1]:                                   # 口袋对
        if h[0] in bcount:                             # 口袋对踩中公共牌 = 三条(set)
            return "strong"
        return "strong" if h[0] >= bmax else "weak"    # 超对=强,底对/中对=弱
    hit = [r for r in h if r in bcount]                # 击中公共牌的英雄牌
    if len(hit) == 2:                                  # 两张都击中 = 两对
        return "strong"
    if len(hit) == 1:
        r = hit[0]
        if bcount[r] == 2:                             # 公共牌该点已成对 → 英雄三条(trips)
            return "strong"
        return "strong" if r == bmax else "weak"       # 顶对=强,中低对=弱
    return "air"


def multiway_action(hero_hole, board, to_call, pot, stack,
                    value_bet_frac=0.5):
    """多路翻后保守兜底。需传入【真实】to_call / pot / stack。"""
    tier = _classify(hero_hole, board)
    if to_call <= 1e-9:                       # 无人下注(check 到我)
        if tier == "strong":
            bet = min(round(value_bet_frac * pot, 1), stack)
            if bet <= 1e-9:
                return ("call", None)         # 无筹码可下 → 过牌
            return ("raise", bet)             # 价值下注
        return ("call", None)                 # weak/air → 过牌(to_call=0 时 call = check)
    # 面对下注
    if tier == "strong" and to_call <= 0.7 * pot:      # 强牌跟合理注
        return ("call", None)
    if tier == "weak" and to_call <= 0.3 * pot:        # 弱成手只跟小注
        return ("call", None)
    return ("fold", None)                     # 其余(空气/面对大注)→ 弃
