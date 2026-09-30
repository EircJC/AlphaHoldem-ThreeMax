"""坍缩桥 —— 把"翻前坍缩成 2 人"的 3 人局面,构造成一个现有 HU 模型能直接消费的
翻后决策快照(含死钱、位置映射),从而【零改动复用】现有 HU 模型 + 护栏 + 查库。

关键设计:快照式"决策预言机(decision oracle)"
  我们不让这个 HU env 从头跑整手,而是把它摆到【翻牌开始】的正确局面,只用来问模型"下一步打什么"。
  真正的记账、发牌、结算由上层 3 人桌负责。所以:
    - villain 底牌不需要(encoding 只读英雄自己的 hole);不跑内部摊牌 → 无发牌/牌堆冲突问题。
    - 死钱(dead)无法塞进 HU 的 committed(那会污染 to_call/结算),改为【覆写 encoding_state 把死钱并入 pot】,
      这样模型看到的底池/SPR 正确,而合法动作、赔率仍只按两个活人的投入算。

位置映射:翻后位置越靠后越好(SB<BB<BTN)。两存活者中靠后者 = 在位方 = HU 的 button(postflop 最后动)。
  HU 座位:button=0(在位),1-button=1(先动)。to_act 翻后 = 非按钮先动,与 HUNLEnv 一致。

翻前动作历史:把两存活者的翻前动作近似映射进 history[0](call→CALL,raise→RAISE 档,allin→ALLIN;
  已弃牌者的动作丢弃)。这是近似项(翻后决策主要看牌面/底池/筹码),已在 README 标注。
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import HUNLEnv                                    # noqa: E402
from config import NB, FOLD, CALL, ALLIN, RAISE_FRACTIONS, GameConfig   # noqa: E402

# 近似:把翻前一次"加注"记为哪个 NB 档(取中间偏小的加注档,纯为动作历史编码)
_RAISE_BUCKET = min(RAISE_FRACTIONS.keys()) if RAISE_FRACTIONS else CALL


class HUNLEnvResumable(HUNLEnv):
    """可从任意翻后局面"续接"的 HU env;死钱并入 encoding 的 pot。仅作决策预言机用。"""

    def encoding_state(self, player: int):
        s = super().encoding_state(player)
        s["pot"] = s["pot"] + float(getattr(self, "dead", 0.0))   # 死钱并入底池 → SPR/赔率对齐真实局面
        return s


def _pf_bucket(action):
    kind = action[0]
    if kind == "raise":
        return _RAISE_BUCKET
    return CALL          # call/过牌


def collapse_to_hu(table, hero_seat, hero_hole, board, norm=500.0):
    """把翻前已坍缩成 2 人的 3 人桌,构造成 HU 决策快照(翻牌起手)。

    参数:
      table     : 已 run_preflop 且 is_collapsed_hu()==True 的 Table3
      hero_seat : 英雄在【3 人桌】的座位号(必须是存活者之一)
      hero_hole : 英雄两张底牌(engine.cards 的整数编码 list)
      board     : 翻牌三张(整数编码 list);转/河后续由上层逐街追加再重建快照
      norm      : 归一化基准 = 训练筹码上限(与你训模型一致,默认 500)
    返回 (env, hero_hu_seat):env 可直接喂 net_policy / 护栏 / 查库;hero_hu_seat ∈ {0,1}。
    若未坍缩成 HU 返回 None。
    """
    if not table.is_collapsed_hu():
        return None
    survivors = table.survivors()
    assert hero_seat in survivors, f"英雄 {hero_seat} 不在存活者 {survivors}"
    ip = table.in_position_seat(survivors)          # 在位方 = button
    oop = survivors[0] if survivors[1] == ip else survivors[1]

    # 3 人座位 → HU 座位:in-position=0(button),out=1
    hu_of = {ip: 0, oop: 1}
    button = 0
    hero_hu = hu_of[hero_seat]

    cfg = GameConfig(small_blind=table.sb, big_blind=table.bb, stack_norm=float(norm))
    env = HUNLEnvResumable(cfg)

    # ---- 手工摆到"翻牌开始"的 HU 局面 ----
    env.button = button
    # 底牌:只有英雄的会被读到;villain 用不冲突的占位牌兜底(决策不看它,但防误查对手视角时 encode 崩)
    env.holes = [None, None]
    env.holes[hero_hu] = list(hero_hole)
    _used = set(hero_hole) | set(board)
    _dummy = [c for c in range(52) if c not in _used][:2]
    env.holes[1 - hero_hu] = _dummy                 # 决策预言机不跑摊牌 → villain 底牌仅占位
    env.board = list(board)
    # 座位映射后的筹码/投入(各存活者"自己"的投入;死钱单独走 env.dead)
    seat_of_hu = {0: ip, 1: oop}
    env.stacks = [table.P[seat_of_hu[0]].stack, table.P[seat_of_hu[1]].stack]
    env.start_stacks = [table.P[seat_of_hu[0]].start_stack, table.P[seat_of_hu[1]].start_stack]
    env.committed = [table.P[seat_of_hu[0]].committed, table.P[seat_of_hu[1]].committed]
    env.dead = table.dead()                          # 死钱:并入 encoding 的 pot
    env.street_bet = [0.0, 0.0]
    env.current_bet = 0.0
    env.min_raise = table.bb
    env.all_in = [env.stacks[0] <= 1e-9, env.stacks[1] <= 1e-9]
    env.acted = [False, False]
    env.aggressor = None
    env.street = 1                                   # 翻牌
    env.folded = None
    env.done = False
    env.winner = None
    env.payoffs = [0.0, 0.0]
    env.to_act = 1 - button                          # 翻后非按钮(oop)先动

    # 翻前动作历史(近似)→ history[0];按 HU 座位记(两存活者各自的翻前动作串)
    hist0 = []
    for seat in survivors:
        hu = hu_of[seat]
        for a in table.P[seat].pf_actions:
            b = ALLIN if table.P[seat].allin and a[0] == "raise" else _pf_bucket(a)
            hist0.append((hu, b))
    env.history = [hist0, [], [], []]
    return env, hero_hu


_POST_ORDER = {"SB": 0, "BB": 1, "BTN": 2}


def collapse_state_to_hu(button, sb, bb, starts, committed, folded, street_bet, current_bet,
                         board, hero_seat, hero_hole, hist=None, norm=500.0):
    """【mid-hand collapse】从任意街的完整 3 人局面(已重建的记账)直接构造 HU 快照。

    与 collapse_to_hu 的区别:后者只处理"翻前就坍缩",从翻牌起手 + 逐街回放;本函数支持
    **翻后才坍缩**(3 人到翻牌、转/河牌才弃到 2 人),直接摆到当前街,死钱含弃牌者的翻后投入。

    参数(均为 3 人座位口径):committed/folded/street_bet 各 3 项;current_bet=当前街注额;
      board=当前公共牌;hero_seat=英雄 3 人座位;hero_hole=英雄两张(int);hist=[(seat,bucket)]×4(近似历史)。
    返回 (env, hero_hu)。要求恰好 2 人存活且 hero 在内。"""
    survivors = [i for i in range(3) if not folded[i]]
    assert len(survivors) == 2 and hero_seat in survivors, f"需恰好2存活且hero在内: {survivors}"
    role = {button % 3: "BTN", (button + 1) % 3: "SB", (button + 2) % 3: "BB"}
    ip = max(survivors, key=lambda s: _POST_ORDER[role[s]])      # 位置靠后=在位=button
    oop = survivors[0] if survivors[1] == ip else survivors[1]
    hu_of = {ip: 0, oop: 1}
    hero_hu = hu_of[hero_seat]
    seat_of_hu = {0: ip, 1: oop}

    cfg = GameConfig(small_blind=sb, big_blind=bb, stack_norm=float(norm))
    env = HUNLEnvResumable(cfg)
    env.button = 0
    env.holes = [None, None]
    env.holes[hero_hu] = list(hero_hole)
    _used = set(hero_hole) | set(board)
    env.holes[1 - hero_hu] = [c for c in range(52) if c not in _used][:2]   # villain 占位(决策不看)
    env.board = list(board)
    n = len(board)
    env.street = 1 if n == 3 else (2 if n == 4 else 3)
    env.stacks = [max(starts[seat_of_hu[0]] - committed[seat_of_hu[0]], 0.0),
                  max(starts[seat_of_hu[1]] - committed[seat_of_hu[1]], 0.0)]
    env.start_stacks = [starts[seat_of_hu[0]], starts[seat_of_hu[1]]]
    env.committed = [committed[seat_of_hu[0]], committed[seat_of_hu[1]]]
    env.dead = sum(committed[i] for i in range(3) if folded[i])   # 死钱=弃牌者全部投入(含翻后)
    env.street_bet = [street_bet[seat_of_hu[0]], street_bet[seat_of_hu[1]]]
    env.current_bet = max(current_bet, env.street_bet[0], env.street_bet[1])
    env.min_raise = bb
    env.all_in = [env.stacks[0] <= 1e-9, env.stacks[1] <= 1e-9]
    env.acted = [False, False]
    env.aggressor = None
    env.folded = None
    env.done = False
    env.winner = None
    env.payoffs = [0.0, 0.0]
    env.to_act = hero_hu
    H = [[], [], [], []]                                          # 历史近似:两活人动作映射进 hu 座位
    if hist:
        for st in range(min(env.street + 1, 4)):
            for (seat, bk) in hist[st]:
                if seat in hu_of:
                    H[st].append((hu_of[seat], bk))
    env.history = H
    return env, hero_hu


def resume_next_street(env, board):
    """转牌/河牌:上层追加公共牌后,重建同一 env 到新街的起手(新街无人下注)。
    just 更新 board/street/清空本街,history 保留前几街的近似。返回 env。"""
    env.board = list(board)
    n = len(board)
    env.street = 1 if n == 3 else (2 if n == 4 else 3)
    env.street_bet = [0.0, 0.0]
    env.current_bet = 0.0
    env.min_raise = env.cfg.big_blind
    env.acted = [False, False]
    env.aggressor = None
    env.to_act = 1 - env.button
    env.done = False
    return env
