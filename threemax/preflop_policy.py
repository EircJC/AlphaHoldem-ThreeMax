"""3-max 翻前策略:用显式范围表(charts.py)判断,位置感知 + 开池/跟/3bet/4bet 分层。

强度不用"对随机权益百分位"(会低估同花连张/小对),改用**手牌类是否在该位置该动作的范围里**。
这是"位置感知的近似标准 3-max charts",非精确 solver 解(无混合频率),但结构对、可直接用、可调。

接口:make_chart_policy(class_of) -> policy(table, player)->("fold"/"call"/"raise", to_bb?)
  class_of: {seat: 手牌类}(如 "AKs"/"76s"/"QQ";由 table3_bot 用 preflop_guard.hand169 算好传入)。
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charts

EPS = 1e-9


def make_chart_policy(class_of, open_to=2.5, threebet_mult=3.0, fourbet_mult=2.2):
    def policy(table, p):
        hc = class_of.get(p.seat)
        role = p.role
        to_call = table.current_bet - p.street_bet
        raised = table.current_bet > table.bb + EPS           # 有人真加注
        threebet_plus = table.current_bet > open_to * 1.6      # 粗判到 3bet 及以上

        if threebet_plus:                                      # 面对 3bet/4bet
            if charts.in_range(hc, charts.FOURBET):
                return ("raise", round(table.current_bet * fourbet_mult, 1))
            if charts.in_range(hc, charts.CALL_VS_3BET):
                return ("call", None)
            return ("call", None) if to_call <= EPS else ("fold", None)

        if raised:                                             # 面对普通开池
            if charts.in_range(hc, charts.THREEBET):
                return ("raise", round(table.current_bet * threebet_mult, 1))
            defend = charts.CALL_VS_OPEN["BB"] if role == "BB" else charts.CALL_VS_OPEN["OTHER"]
            if charts.in_range(hc, defend):
                return ("call", None)
            return ("call", None) if to_call <= EPS else ("fold", None)

        # 未被加注(只有盲注)→ 可首入开池
        if charts.in_range(hc, charts.OPEN[role]):
            return ("raise", open_to)
        if to_call <= EPS:                                     # BB 免费 → 过牌
            return ("call", None)
        return ("fold", None)                                  # 不够开池范围 → 弃(3-max 不 limp)
    return policy
