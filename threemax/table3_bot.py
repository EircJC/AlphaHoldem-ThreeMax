"""3 人桌决策驱动:把 翻前策略 + 坍缩桥 + HU 决策栈(iter280) + 多路兜底 串成一条链。

给定一个 3 人局面(每条街的动作 + 筹码),返回 hero 该出的动作:
  - 翻前 → 3-max 翻前策略(preflop_policy,当前是占位桩,后续换真 charts)
  - 翻前打完只剩 2 人(含 hero)→ 坍缩成 HU:hu_bridge 造 env(死钱+位置)→ 翻后回放 →
    交 serve.bot.PokerBot.decide_env(iter280 + 库 + 护栏 + 剥削,= HU 那套 +4.8)
  - ≥3 人进翻牌 → multiway 兜底(占位桩)

无 pokerbot 时(torch 不可用)仍能跑路由 + 构造坍缩 env(返回诊断),便于本地自测。

请求 JSON(3 人):
{
  "table_id","hero_seat"(0/1/2),"button"(0/1/2 = BTN/小盲),
  "sb":0.5,"bb":1.0,"start_stacks":[s0,s1,s2],
  "hero_hole":["Ah","Ks"],"board":[...],
  "actions":[{"street","actor","type","to"}...]   # 三人全部动作
}
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT, os.path.join(_ROOT, "serve")):
    sys.path.insert(0, p)

import table3 as T3
import hu_bridge as HB
import preflop_policy as PP
import multiway as MW
import engine_state as es                                     # serve/engine_state:牌解析/桶/街
from config import FOLD, CALL, ALLIN                          # noqa: E402

import preflop_guard as _pg                                   # hand169:两张牌→手牌类(如 AKs/76s/QQ)


def _hole_class(hole_ints):
    """两张底牌(int)→ 规范手牌类(charts 用)。"""
    return _pg.hand169(hole_ints[0], hole_ints[1])


try:
    from config import RAISE_FRACTIONS                        # noqa: E402
    _RB = min(RAISE_FRACTIONS.keys()) if RAISE_FRACTIONS else CALL   # 历史里"加注"用的合法档
except Exception:
    _RB = CALL


def _reconstruct_full(button, starts, sb, bb, actions, cur_street):
    """回放【全部】动作(街≤cur_street)重建当前局面(3 人座位口径):
      committed[3] 各家总投入(含盲注)、folded[3]、当前街 street_bet[3] + current_bet、双活人历史近似 hist。
    用于 mid-hand collapse(翻后才坍缩)与多路兜底,按【当前存活】而非翻前存活分流。"""
    sb_seat = (button + 1) % 3
    bb_seat = (button + 2) % 3
    committed = [0.0, 0.0, 0.0]
    committed[sb_seat] = min(sb, starts[sb_seat])
    committed[bb_seat] = min(bb, starts[bb_seat])
    folded = [False, False, False]
    street_bet = [0.0, 0.0, 0.0]
    street_bet[sb_seat] = committed[sb_seat]
    street_bet[bb_seat] = committed[bb_seat]
    current_bet = bb
    cur = 0
    hist = [[], [], [], []]
    for a in actions:
        st = int(a.get("street", 0))
        if st > cur_street:
            break
        if st != cur:                                         # 进新街:清本街投入
            cur = st
            street_bet = [0.0, 0.0, 0.0]
            current_bet = 0.0
        actor = int(a["actor"])
        t = a.get("type")
        if t == "fold":
            folded[actor] = True
            continue
        if t in ("check", "call"):
            add = max(current_bet - street_bet[actor], 0.0); bk = CALL
        elif t == "allin":
            add = max(starts[actor] - committed[actor], 0.0); bk = ALLIN
        else:                                                 # bet/raise
            target = float(a["to"])
            # 历史档位:翻前用 _RB(与旧 _pf_bucket 一致);翻后用真实池比例档 es._bet_bucket
            # (与旧 _replay_postflop 一致)——否则大 c-bet 被编成最小注,模型误读下注线、翻后决策全乱。
            if st == 0:
                bk = _RB
            else:
                max_to = street_bet[actor] + max(starts[actor] - committed[actor], 0.0)
                bk = es._bet_bucket(target, current_bet, street_bet[actor], sum(committed), max_to)
            add = max(target - street_bet[actor], 0.0)
        committed[actor] += add
        street_bet[actor] += add
        if t not in ("check", "call", "fold"):
            current_bet = max(current_bet, street_bet[actor])
        hist[st].append((actor, bk))
    if cur < cur_street:                                      # 当前街还没人动作(hero 新街先手)→ 无注
        street_bet = [0.0, 0.0, 0.0]
        current_bet = 0.0
    return committed, folded, street_bet, current_bet, hist


def _hist_from_actions(actions):
    """请求 actions([{street,actor,type,to}]) → [(actor,bucket)]×4,供剥削层对手读数更新。
    bucket:raise/bet→2、allin→ALLIN、fold→FOLD、check/call→CALL(与 engine3 的 _hist 一致)。"""
    hist = [[], [], [], []]
    for a in actions:
        t = a.get("type")
        bk = (2 if t in ("raise", "bet") else
              (ALLIN if t == "allin" else (FOLD if t == "fold" else CALL)))
        st = int(a.get("street", 0))
        if 0 <= st <= 3:
            hist[st].append((int(a["actor"]), bk))
    return hist


class ThreeMaxBot:
    def __init__(self, pokerbot=None, norm=500.0, dead_in_pot=True):
        self.pokerbot = pokerbot                              # serve.bot.PokerBot(HU 决策栈);None=只测路由
        self.norm = norm
        # 坍缩时是否把死钱并入模型编码的底池。True=SPR对齐真实(但可能推模型OOD、打太松);
        # False=按纯HU底池(池=两活人投入)喂模型,保持在训练分布内(死钱仍是真实要争的池,只是不喂编码)。
        self.dead_in_pot = dead_in_pot
        self._fed_hands = set()                              # (table_id,hand_id) 已喂读数的手 → /hand_end 去重

    def decide(self, state):
        sb = float(state.get("sb", 0.5)); bb = float(state.get("bb", 1.0))
        button = int(state["button"]); hero = int(state["hero_seat"])
        table_id = state.get("table_id", "default")
        starts = [float(x) for x in state["start_stacks"]]
        hole = [es.str_to_card(c) for c in state["hero_hole"]]
        board = [es.str_to_card(c) for c in state.get("board", [])]
        actions = state.get("actions", [])
        street = es._street_from_board(board)

        # 翻前:3-max 策略(桩)
        if street == 0:
            table = T3.Table3(button, starts, sb, bb)
            for a in actions:                                 # 回放 hero 决策前的所有翻前动作(含 hero 早前的动作)
                if int(a.get("street", 0)) == 0:
                    self._apply_pf(table, a)
            pol = PP.make_chart_policy({hero: _hole_class(hole)})
            act = pol(table, table.P[hero])                   # ("fold"/"call"/"raise", to?)
            return self._pf_response(table, hero, act)

        # 翻后:回放【全部】动作重建当前局面,按【当前存活】分流(支持 mid-hand collapse)
        committed, folded, street_bet, current_bet, hist = _reconstruct_full(
            button, starts, sb, bb, actions, street)
        survivors = [i for i in range(3) if not folded[i]]
        if hero not in survivors:
            return {"action": "fold", "note": "hero 已弃牌,不该被问", "meta": {}}

        if len(survivors) == 1:                                # 只剩 hero(walk):已赢,不该被问
            return {"action": "check", "to": None, "add": 0.0,
                    "meta": {"route": "walk", "note": "只剩hero,无需决策"}}

        if len(survivors) == 2:                               # 坍缩成 HU(翻前或翻后才坍缩皆可)→ 复用 iter330
            env, hero_hu = HB.collapse_state_to_hu(
                button, sb, bb, starts, committed, folded, street_bet, current_bet,
                board, hero, hole, hist, self.norm)
            if not self.dead_in_pot:
                env.dead = 0.0                                # 不把死钱喂进编码(避免 OOD),保持纯 HU 底池
            env.to_act = hero_hu
            if self.pokerbot is None:
                return {"route": "collapsed_hu", "hero_hu": hero_hu,
                        "pot_incl_dead": env.encoding_state(hero_hu)["pot"],
                        "dead": env.dead, "street": env.street,
                        "to_call": env.current_bet - env.street_bet[hero_hu]}
            out = self.pokerbot.decide_env(env, hero_hu, table_id)
            out.setdefault("meta", {})["route"] = "collapsed_hu"
            return out

        # ≥3 人仍在牌(真多路)→ 保守启发式兜底
        to_call = max(current_bet - street_bet[hero], 0.0)
        pot = sum(committed)
        hstack = max(starts[hero] - committed[hero], 0.0)
        act = MW.multiway_action(hole, board, to_call, pot, hstack)
        return {"action": act[0], "to": act[1], "meta": {"route": "multiway_stub"}}

    def hand_end(self, state):
        """一手结束后调用,传本手【完整】动作(三家全部),更新该桌对手读数(剥削层)。
        3 人合并两对手进一个 tracker(与 play_3max 一致);两对手风格不同时是已知近似。"""
        pb = self.pokerbot
        if pb is None or not getattr(pb, "exploit", False):
            return
        hero = int(state["hero_seat"])
        table_id = state.get("table_id", "default")
        hand_id = state.get("hand_id")
        if hand_id is not None:                              # 去重:同一手重复调 /hand_end 不重复计数
            key = (table_id, hand_id)
            if key in self._fed_hands:
                return
            self._fed_hands.add(key)
            if len(self._fed_hands) > 20000:                 # 有界:超量清一半旧的
                self._fed_hands = set(list(self._fed_hands)[10000:])
        hist = _hist_from_actions(state.get("actions", []))
        tr = pb._tracker(table_id)
        for opp in range(3):
            if opp != hero:
                tr.update(hist, opp)

    def reset_table(self, table_id):
        """对手换人/离桌:清掉该桌对手读数 + 该桌已喂手记录。"""
        if self.pokerbot is not None:
            self.pokerbot.reset_table(table_id)
        self._fed_hands = {k for k in self._fed_hands if k[0] != table_id}

    def _apply_pf(self, table, a):
        typ = a["type"]; seat = int(a["actor"])
        if typ == "fold":
            table._apply(seat, ("fold", None))
        elif typ in ("check", "call"):
            table._apply(seat, ("call", None))
        elif typ == "allin":
            table._apply(seat, ("raise", table.P[seat].street_bet + table.P[seat].stack))
        else:
            table._apply(seat, ("raise", float(a["to"])))

    def _pf_response(self, table, hero, act):
        kind = act[0]
        p = table.P[hero]
        if kind == "fold":
            return {"action": "fold", "to": None, "add": 0.0, "meta": {"route": "preflop_chart"}}
        if kind == "call":
            to = table.current_bet
            return {"action": "check" if to - p.street_bet <= 1e-9 else "call",
                    "to": round(to, 2), "add": round(max(to - p.street_bet, 0.0), 2),
                    "meta": {"route": "preflop_chart"}}
        target = float(act[1])
        return {"action": "raise", "to": round(target, 2),
                "add": round(target - p.street_bet, 2), "meta": {"route": "preflop_chart"}}
