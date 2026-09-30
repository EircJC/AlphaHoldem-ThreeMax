"""局面重建:把服务收到的 JSON(每条街的动作 + 筹码)还原成一个 HUNLEnv 快照,
供决策栈(库/护栏/剥削/NN/河解)消费;并把模型输出的 NB bucket 转回具体动作(加注到多少)。

无 torch 依赖,可单独自测。牌用字符串 'Ah'(rank+suit,rank∈23456789TJQKA,suit∈shdc)。

请求 JSON 契约(/decide):
{
  "table_id": "t1",              # 多桌:每桌独立维护对手读数(剥削层)
  "hero_seat": 1,                # 机器人坐哪个座位(0/1)
  "button": 0,                   # 哪个座位是按钮/小盲(翻前先动、翻后有位置)
  "sb": 0.5, "bb": 1.0,          # 盲注(bb),默认 0.5/1.0
  "start_stacks": [200.0, 200.0],# 本手起始筹码(下盲前)[seat0, seat1](bb)
  "hero_hole": ["Ah","Ks"],      # 机器人两张底牌
  "board": ["Qs","Jh","2h"],     # 公共牌 0/3/4/5 张
  "actions": [                   # 本手到目前为止的动作(有序),轮到机器人时调用
     {"street":0,"actor":0,"type":"raise","to":3.0},
     {"street":0,"actor":1,"type":"call"},
     {"street":1,"actor":1,"type":"check"}
  ]
}
type ∈ fold/check/call/bet/raise/allin;bet/raise 需带 "to"=该街累计下注到多少 bb。
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import HUNLEnv                                    # noqa: E402
from config import GameConfig, NB, FOLD, CALL, ALLIN, RAISE_FRACTIONS   # noqa: E402

EPS = 1e-9
_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "shdc"


def str_to_card(s: str) -> int:
    s = s.strip()
    return _RANK_CHARS.index(s[0].upper()) * 4 + _SUIT_CHARS.index(s[1].lower())


def _bet_bucket(target, current_bet, street_bet_actor, committed_total, actor_stack_total):
    """把'本街下注到 target'分到 NB 档(与引擎/ slumbot 同口径)。actor_stack_total=该玩家本街已投+剩余。"""
    if target >= actor_stack_total - EPS:
        return ALLIN
    to_call = current_bet - street_bet_actor
    pot_after_call = committed_total + max(to_call, 0.0)
    raise_extra = target - current_bet
    frac = raise_extra / max(pot_after_call, EPS)
    best_b, best_d = 2, 1e18
    for b, f in RAISE_FRACTIONS.items():
        d = abs(f - frac)
        if d < best_d:
            best_d, best_b = d, b
    return best_b


def _street_from_board(board):
    n = len(board)
    return 0 if n < 3 else (1 if n == 3 else (2 if n == 4 else 3))


def build_env(req, norm=500.0):
    """请求 JSON → HUNLEnv 快照(轮到 hero 决策)。返回 (env, hero_seat)。"""
    sb = float(req.get("sb", 0.5)); bb = float(req.get("bb", 1.0))
    button = int(req["button"]); hero = int(req["hero_seat"])
    starts = [float(x) for x in req["start_stacks"]]
    cfg = GameConfig(small_blind=sb, big_blind=bb, stack_norm=float(norm))
    env = HUNLEnv(cfg)                                         # 不调用 reset(不发牌),手工摆局
    env.button = button
    hero_cards = [str_to_card(c) for c in req["hero_hole"]]
    board = [str_to_card(c) for c in req.get("board", [])]
    used = set(hero_cards) | set(board)
    dummy = [c for c in range(52) if c not in used][:2]        # villain 占位(决策不看,防编码空底牌崩)
    env.holes = [None, None]
    env.holes[hero] = hero_cards
    env.holes[1 - hero] = dummy
    env.board = board
    env.start_stacks = list(starts)
    env.stacks = list(starts)
    env.committed = [0.0, 0.0]
    env.street_bet = [0.0, 0.0]
    env.all_in = [False, False]
    env.acted = [False, False]
    env.folded = None
    env.done = False
    env.winner = None
    env.payoffs = [0.0, 0.0]
    env.history = [[], [], [], []]

    def _put_to(seat, target):
        add = min(target - env.street_bet[seat], env.stacks[seat])
        add = max(add, 0.0)
        env.stacks[seat] -= add
        env.committed[seat] += add
        env.street_bet[seat] += add
        if env.stacks[seat] <= EPS:
            env.all_in[seat] = True

    # 下盲
    sb_seat, bb_seat = button, 1 - button
    _put_to(sb_seat, sb); _put_to(bb_seat, bb)
    env.current_bet = bb
    env.min_raise = bb
    env.aggressor = bb_seat

    cur_street = 0
    for act in req.get("actions", []):
        st = int(act["street"])
        while cur_street < st:                                # 进入新街:清本街下注
            cur_street += 1
            env.street_bet = [0.0, 0.0]
            env.current_bet = 0.0
            env.min_raise = bb
            env.aggressor = None
            env.acted = [False, False]
        actor = int(act["actor"]); typ = act["type"]
        if typ == "fold":
            env.folded = actor
            env.history[st].append((actor, FOLD))
            continue
        if typ in ("check", "call"):
            _put_to(actor, env.current_bet)
            env.acted[actor] = True
            env.history[st].append((actor, CALL))
            continue
        if typ == "allin":
            target = env.street_bet[actor] + env.stacks[actor]
            bk = ALLIN
        else:                                                 # bet / raise
            target = float(act["to"])
            bk = _bet_bucket(target, env.current_bet, env.street_bet[actor],
                             sum(env.committed), env.street_bet[actor] + env.stacks[actor])
        prev = env.current_bet
        _put_to(actor, target)
        if env.street_bet[actor] > prev + EPS:
            env.min_raise = max(env.min_raise, env.street_bet[actor] - prev)
            env.current_bet = env.street_bet[actor]
            env.aggressor = actor
        env.acted[actor] = True
        env.history[st].append((actor, bk))

    # 公共牌已翻但该街还没动作(hero 在新街先手)→ 补进到牌面所示的街,清本街下注
    board_street = _street_from_board(board)
    while cur_street < board_street:
        cur_street += 1
        env.street_bet = [0.0, 0.0]
        env.current_bet = 0.0
        env.min_raise = bb
        env.aggressor = None
        env.acted = [False, False]
    env.street = board_street
    env.to_act = hero
    return env, hero


def bucket_to_action(env, bucket, hero):
    """NB bucket → 给调用方的具体动作。to=本街累计下注到多少bb;add=还需投入多少bb。"""
    to_call = env.current_bet - env.street_bet[hero]
    if bucket == FOLD:
        return {"action": "fold", "to": None, "add": 0.0, "bucket": FOLD}
    if bucket == CALL:
        if to_call <= EPS:
            return {"action": "check", "to": env.street_bet[hero], "add": 0.0, "bucket": CALL}
        add = min(to_call, env.stacks[hero])
        return {"action": "call", "to": env.street_bet[hero] + add, "add": round(add, 2), "bucket": CALL}
    if bucket == ALLIN:
        add = env.stacks[hero]
        return {"action": "allin", "to": round(env.street_bet[hero] + add, 2), "add": round(add, 2), "bucket": ALLIN}
    # 加注档:与 HUNLEnv.step 完全同口径
    frac = RAISE_FRACTIONS[bucket]
    pot_after_call = sum(env.committed) + max(to_call, 0.0)
    total_add = to_call + frac * pot_after_call
    total_add = min(total_add, env.stacks[hero])              # 不够则退化为全下额度
    target = env.street_bet[hero] + total_add
    act = "allin" if env.stacks[hero] - total_add <= EPS else "raise"
    return {"action": act, "to": round(target, 2), "add": round(total_add, 2), "bucket": bucket}


def build_history_only(req):
    """只重建 history(供 /hand_end 更新对手读数):返回 [[(seat,bucket)...] x4]。"""
    env, _ = build_env(req)
    return env.history
