"""翻前护栏(guardrail):按【手牌强度】封顶翻前投入,拦住"垃圾牌重筹送死 / 边缘牌乱跟大注"。
这不是 GTO 表,是一层外科手术式的硬约束——只在【深筹投入】处介入,开池/3bet 不动(留住 +140 的翻前进攻)。

原理:
  - 每手牌有一个"对随机牌权益"排名 → 百分位 pct(AA=1.0 … 42o=0.0),见 HAND_EQ(评估器实测)。
  - 投入越深,要求 pct 越高(GATE 表)。若模型选的动作要投入超过该手牌能承受的额度 → 降级为
    "能承受的最大合法动作"(通常 跟/过);连跟都超额 → 弃。永远只【降级】,不会让模型更激进。

用法:play_vs_bot.py --preflop-guard(只作用于模型 seat1 的翻前决策)。
  python3 preflop_guard.py --selftest
"""
import argparse

# —— 从 config 拿动作编码;拿不到就用默认(与 config 一致)——
try:
    from config import FOLD, CALL, ALLIN, RAISE_FRACTIONS, NB
except Exception:  # pragma: no cover
    FOLD, CALL = 0, 1
    RAISE_FRACTIONS = {2 + i: f for i, f in enumerate([0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])}
    ALLIN = 2 + len(RAISE_FRACTIONS)
    NB = ALLIN + 1

_RANKC = "23456789TJQKA"

# 169 手牌"对随机牌权益"(引擎 evaluate7 实测,N=1200)。用于强度排名。
HAND_EQ = {
    "AA": 0.8721, "KK": 0.8413, "QQ": 0.7954, "JJ": 0.7642, "TT": 0.7446, "99": 0.7225,
    "AKs": 0.6758, "88": 0.6742, "AKo": 0.6708, "AQs": 0.6654, "77": 0.66, "AJs": 0.6592,
    "ATs": 0.6587, "ATo": 0.6483, "KJs": 0.6433, "KQs": 0.6375, "66": 0.6367, "AQo": 0.6308,
    "KJo": 0.6308, "QJo": 0.6296, "AJo": 0.6271, "A8s": 0.6271, "KQo": 0.6246, "K9s": 0.6154,
    "A9s": 0.615, "A8o": 0.6108, "KTs": 0.6104, "A6s": 0.61, "QJs": 0.6083, "A4s": 0.6054,
    "55": 0.6033, "A7s": 0.6029, "A5s": 0.6029, "A7o": 0.6004, "A9o": 0.5883, "QTs": 0.5879,
    "A3s": 0.5858, "K9o": 0.5854, "A4o": 0.5837, "K8s": 0.5833, "A2o": 0.5813, "JTs": 0.5813,
    "Q9s": 0.5808, "A6o": 0.5767, "QTo": 0.5763, "Q8s": 0.5713, "K5s": 0.5708, "44": 0.5708,
    "K6o": 0.5683, "K7s": 0.5679, "KTo": 0.5633, "K4s": 0.5592, "A2s": 0.5558, "K6s": 0.5546,
    "J9s": 0.5537, "Q7s": 0.5508, "Q9o": 0.55, "K4o": 0.5479, "A5o": 0.5446, "JTo": 0.5437,
    "T9s": 0.5413, "T8s": 0.5413, "Q8o": 0.5396, "K8o": 0.5371, "K3s": 0.5333, "J9o": 0.5329,
    "K7o": 0.5325, "J8s": 0.5325, "Q4s": 0.5267, "K2s": 0.5242, "K3o": 0.5225, "A3o": 0.5221,
    "J5s": 0.5208, "T7s": 0.5192, "22": 0.5175, "Q6s": 0.5154, "Q7o": 0.5129, "J8o": 0.5108,
    "T9o": 0.5108, "J7s": 0.51, "33": 0.51, "97s": 0.5071, "K2o": 0.505, "K5o": 0.5046,
    "Q5s": 0.5038, "T8o": 0.5033, "98s": 0.5025, "Q6o": 0.5021, "Q5o": 0.5012, "T6s": 0.5012,
    "J6s": 0.4979, "J4s": 0.4958, "Q4o": 0.4888, "Q2s": 0.4888, "87s": 0.4858, "Q3o": 0.4846,
    "Q3s": 0.4838, "J6o": 0.4829, "T5s": 0.4783, "T2s": 0.4779, "J2s": 0.4775, "98o": 0.4767,
    "T7o": 0.475, "J5o": 0.4746, "J3o": 0.4746, "76s": 0.4713, "96s": 0.4704, "T4s": 0.4658,
    "97o": 0.4637, "86s": 0.4637, "J2o": 0.4596, "T6o": 0.4587, "J7o": 0.4567, "93s": 0.4567,
    "Q2o": 0.4558, "86o": 0.4558, "85s": 0.455, "87o": 0.4546, "J4o": 0.4533, "J3s": 0.45,
    "T3s": 0.4467, "96o": 0.4446, "85o": 0.4433, "75s": 0.4408, "94s": 0.4392, "95s": 0.4375,
    "T4o": 0.4363, "T5o": 0.4358, "65s": 0.4333, "84s": 0.4329, "83s": 0.4288, "76o": 0.4225,
    "T3o": 0.4163, "94o": 0.4163, "74s": 0.415, "95o": 0.4138, "T2o": 0.4113, "75o": 0.4104,
    "92s": 0.4092, "73s": 0.4079, "72s": 0.4029, "82s": 0.3983, "83o": 0.3975, "65o": 0.3971,
    "64o": 0.3962, "64s": 0.39, "43s": 0.3896, "53s": 0.3879, "32s": 0.3879, "62s": 0.3846,
    "63s": 0.3837, "54s": 0.3825, "92o": 0.3821, "93o": 0.3804, "52s": 0.3742, "74o": 0.3708,
    "82o": 0.3704, "62o": 0.3679, "84o": 0.3671, "63o": 0.3625, "54o": 0.36, "42s": 0.3588,
    "53o": 0.3575, "73o": 0.3567, "43o": 0.3471, "52o": 0.3429, "72o": 0.3337, "32o": 0.3233,
    "42o": 0.32,
}

# 手牌 → 百分位(0=最弱 42o,1=最强 AA)
_sorted = sorted(HAND_EQ, key=HAND_EQ.get)
PCT = {h: i / (len(_sorted) - 1) for i, h in enumerate(_sorted)}

# 投入闸(bb 到额, 需要的最低百分位)。只在深筹处介入:
#   ≤13bb(开池/3bet)不设闸 → 保留进攻与 3bet 诈唬;
#   >13bb(4bet 级)需 top~45%;>28bb(5bet 级)需 top~22%;>55bb(巨注/全下)需 top~15%。
GATE = [(13.0, 0.55), (28.0, 0.78), (55.0, 0.85)]


def hand169(c0, c1):
    r0, r1 = c0 // 4, c1 // 4
    s0, s1 = c0 % 4, c1 % 4
    hi, lo = max(r0, r1), min(r0, r1)
    if hi == lo:
        return _RANKC[hi] * 2
    return _RANKC[hi] + _RANKC[lo] + ("s" if s0 == s1 else "o")


def _req(commit):
    """该投入额需要的最低百分位。"""
    reqs = [r for t, r in GATE if commit > t]
    return max(reqs) if reqs else 0.0


def _cap(pct):
    """该手牌(百分位 pct)最多能投入的 bb 额度。"""
    caps = [t for t, r in GATE if pct < r]
    return min(caps) if caps else float("inf")


def _commit_of(env, p, a):
    """动作 a 会让玩家 p 本手翻前投入到多少 bb(到额)。"""
    my_sb = env.street_bet[p]
    to_call = max(env.current_bet - my_sb, 0.0)
    if a == FOLD:
        return 0.0
    if a == CALL:
        return my_sb + to_call
    allin_to = my_sb + env.stacks[p]
    if a == ALLIN:
        return allin_to
    frac = RAISE_FRACTIONS.get(a)
    if frac is None:
        return allin_to
    pot_after_call = sum(env.committed) + to_call
    return min(my_sb + to_call + frac * pot_after_call, allin_to)


def guard(env, seat, a, counters=None):
    """翻前护栏:仅在 street 0 介入。返回(可能被降级的)动作 bucket。只降级、不升级。"""
    if getattr(env, "street", 0) != 0:
        return a
    p = seat
    hole = env.holes[p]
    pct = PCT.get(hand169(hole[0], hole[1]), 0.5)
    ca = _commit_of(env, p, a)
    if pct >= _req(ca):
        return a                                   # 投入在承受范围内 → 放行
    # 降级:只降到 跟/弃,不发明新的加注尺寸(可预测、不会造出 4bet-fold 这类怪线)。
    cap = _cap(pct)
    legal = env.legal_mask()
    call_commit = _commit_of(env, p, CALL)
    if CALL < len(legal) and legal[CALL] and call_commit <= cap + 1e-6 and call_commit <= ca + 1e-6:
        best = CALL                                # 跟得起(投入在承受范围)→ 跟
    else:
        best = FOLD if (FOLD < len(legal) and legal[FOLD]) else CALL   # 连跟都超额 → 弃
    if best != a and counters is not None:
        counters[0] = counters[0] + 1
    return best


# ---------- 河牌近坚果护栏(与 play_vs_bot / slumbot_client 共用;env 为 duck-typed)----------
def river_nut_frac(hole, board):
    """河牌:英雄手在完整 5 张公共牌上,打败对手所有可能两张组合的比例(含阻断;平局算半)。1.0=坚果。
    hole/board 为整数牌(rank*4+suit)。"""
    import itertools
    from engine.evaluator import evaluate7
    dead = set(hole) | set(board)
    rest = [c for c in range(52) if c not in dead]
    hero = evaluate7(list(hole) + list(board))
    win = ties = tot = 0
    for a, b in itertools.combinations(rest, 2):
        o = evaluate7([a, b] + list(board))
        if hero > o:
            win += 1
        elif hero == o:
            ties += 1
        tot += 1
    return (win + 0.5 * ties) / max(tot, 1)


def river_nut_guard_action(env, seat, a, thresh=0.97, counters=None):
    """河牌护栏:模型将【弃掉近坚果】(打败≥thresh 比例对手手)且面对下注时 → 改为跟。只 FOLD→CALL。
    env 需暴露 .street/.holes/.current_bet/.street_bet/.board/.legal_mask()(duck-typed)。"""
    if getattr(env, "street", -1) != 3 or a != FOLD:
        return a
    if env.current_bet - env.street_bet[seat] <= 0:          # 没面对下注(可免费过牌)→ 不管
        return a
    legal = env.legal_mask()
    if not (CALL < len(legal) and legal[CALL]):
        return a
    if river_nut_frac(env.holes[seat], env.board) >= thresh:
        if counters is not None:
            counters[0] = counters[0] + 1
        return CALL
    return a


# ---------- 翻后"底池已承诺/赔率"护栏(与 play_vs_bot / slumbot_client 共用)----------
def equity_vs_random(hole, board, rng, n=400):
    """英雄手对【随机对手手 + 随机后续公共牌】的粗略胜率(蒙特卡洛,平局算半)。
    hole/board=整数牌。翻牌(board=3)/转牌(4)会补发到 5 张;河牌(5)只随机对手手。"""
    import random
    from engine.evaluator import evaluate7
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = max(0, 5 - len(board))
    if rng is None:
        rng = random.Random(0)
    win = tie = 0
    for _ in range(n):
        samp = rng.sample(deck, 2 + need)
        vh, run = samp[:2], samp[2:]
        fb = list(board) + run
        h = evaluate7(list(hole) + fb)
        o = evaluate7(list(vh) + fb)
        if h > o:
            win += 1
        elif h == o:
            tie += 1
    return (win + 0.5 * tie) / max(n, 1)


def equity_vs_strong(hole, board, rng, opp_samples=3, n=400):
    """英雄手对【较强对手范围】的粗略胜率(蒙特卡洛,平局算半)。
    每次迭代:先随机补齐公共牌到 5 张,再抽 opp_samples 个对手手【取最强的一个】——
    等价于假设对手范围 ≈ top-1/opp_samples(opp_samples 越大=对手范围越强,越像大注/全下的价值范围)。
    这样弱牌/只借公共牌的牌不会被高估(对随机手它们不弱,但对价值范围很弱)。hole/board=整数牌。"""
    import random
    from engine.evaluator import evaluate7
    dead0 = set(hole) | set(board)
    need = max(0, 5 - len(board))
    if rng is None:
        rng = random.Random(0)
    win = tie = 0
    for _ in range(n):
        deck = [c for c in range(52) if c not in dead0]
        run = rng.sample(deck, need) if need else []
        fb = list(board) + run
        used = dead0 | set(run)
        pool = [c for c in range(52) if c not in used]
        best_o = None
        for _k in range(max(1, opp_samples)):               # 抽 K 个对手手取最强 ≈ top-1/K 范围
            vh = rng.sample(pool, 2)
            o = evaluate7(list(vh) + fb)
            if best_o is None or o > best_o:
                best_o = o
        h = evaluate7(list(hole) + fb)
        if h > best_o:
            win += 1
        elif h == best_o:
            tie += 1
    return (win + 0.5 * tie) / max(n, 1)


def commit_guard_action(env, seat, a, min_ratio=6.0, opp_samples=3, margin=1.0,
                        rng=None, n=400, counters=None):
    """翻后"底池已承诺/赔率"护栏:面对下注、赔率极大(赢:赌 ≥ min_ratio)、且【对较强范围】胜率够吃赔率时,
    禁止弃牌 → 改为跟。治"已投入大半身价却弃掉高赔率听牌/成手"(如 89bb 池只需再跟 10bb 却弃两头顺听)。
    只 FOLD→CALL,只在翻后(street≥1)。用 equity_vs_strong(不是对随机手)→ 不会错救弱牌/借公共牌的牌。
    - min_ratio: 只在赔率 ≥ 该值时才介入(6:1 ⇒ 只需 ~14% 胜率),避免碰接近 GTO 的边缘弃牌。
    - opp_samples: 对手范围强度(3≈top33%);margin: 胜率需 ≥ margin×所需赔率胜率 才跟。"""
    if getattr(env, "street", -1) < 1 or a != FOLD:
        return a
    to_call = env.current_bet - env.street_bet[seat]
    if to_call <= 1e-9:                                      # 没面对下注 → 不管
        return a
    legal = env.legal_mask()
    if not (CALL < len(legal) and legal[CALL]):
        return a
    pot_before = sum(env.committed)                          # 当前底池(含对手这注)
    ratio = pot_before / to_call                             # 赢:赌
    if ratio < min_ratio:                                    # 赔率没到阈值 → 交给模型/库决定(可能是合理弃牌)
        return a
    req_eq = to_call / (pot_before + to_call)                # 保本所需胜率
    eq = equity_vs_strong(env.holes[seat], env.board, rng, opp_samples, n)
    if eq >= req_eq * margin:                                # 对较强范围仍有足够胜率吃赔率 → 别弃
        if counters is not None:
            counters[0] = counters[0] + 1
        return CALL
    return a


def air_fold_guard_action(env, seat, a, min_bet_frac=0.6, opp_samples=4, safety=1.0,
                          rng=None, n=400, counters=None):
    """翻后"放弃/别拿弱牌接大注"护栏:面对【大注/全下】(to_call ≥ min_bet_frac×底池,或跟注即全下)、
    且【对较强范围】胜率连保本赔率都不到时,把 CALL → FOLD(治"诈唬被抓还拿空气 hero-call 全下")。
    保守:只碰大注(小注一律不弃,避免被诈唬剥削);用 equity_vs_strong(opp_samples 更大=范围更强)。
    只 CALL→FOLD,只在翻后(street≥1)。env 同上需暴露 committed/stacks 等。
    - min_bet_frac: 多大的注才介入(0.6=对手下注≥60%底池 或 全下)。
    - safety: 胜率 < safety×所需赔率胜率 才弃(1.0=只弃明显 -EV;<1 更保守、弃更少)。"""
    if getattr(env, "street", -1) < 1 or a != CALL:
        return a
    to_call = env.current_bet - env.street_bet[seat]
    if to_call <= 1e-9:                                      # 面前是"过牌"而非跟注 → 绝不改成弃
        return a
    legal = env.legal_mask()
    if not (FOLD < len(legal) and legal[FOLD]):
        return a
    pot_before = sum(env.committed)
    my_stack = env.stacks[seat] if hasattr(env, "stacks") else None
    is_allin_call = (my_stack is not None and to_call >= my_stack - 1e-6)
    if (to_call / max(pot_before, 1e-9)) < min_bet_frac and not is_allin_call:
        return a                                             # 只碰大注/全下,小注不弃
    req_eq = to_call / (pot_before + to_call)
    eq = equity_vs_strong(env.holes[seat], env.board, rng, opp_samples, n)
    if eq < req_eq * safety:                                 # 对较强范围连保本都不到 → 放弃
        if counters is not None:
            counters[0] = counters[0] + 1
        return FOLD
    return a


def _selftest():
    class E:  # 极简 env 桩
        def __init__(self, holes, cur, sb, committed, stacks, mask):
            self.street = 0; self.holes = holes; self.current_bet = cur
            self.street_bet = sb; self.committed = committed; self.stacks = stacks
            self._mask = mask
        def legal_mask(self):
            return self._mask

    def card(lbl):  # 'As' → int
        r = _RANKC.index(lbl[0]); s = "shdc".index(lbl[1]); return r * 4 + s

    full = [True] * NB
    # 1) T2o 面对 4bet 到 16(它想 5bet 全下)→ 应降级为弃(连跟16都超额)
    e = E([[card("Td"), card("2c")], [0, 0]], cur=16.0, sb=[6.0, 16.0],
          committed=[16.0, 6.0], stacks=[84.0, 94.0], mask=full)
    out = guard(e, 0, ALLIN)
    print(f"  T2o 面对4bet 想全下 → {out}(应 FOLD={FOLD})"); assert out == FOLD, out
    # 2) 72o 开池(投入 2.5)→ 不设闸,放行
    e2 = E([[card("7d"), card("2c")], [0, 0]], cur=1.0, sb=[0.5, 1.0],
           committed=[0.5, 1.0], stacks=[99.5, 99.0], mask=full)
    a = 5  # 某加注档
    assert guard(e2, 0, a) == a, "开池不该被拦"
    print("  72o 开池 → 放行 ✓")
    # 3) AA 想 5bet 全下 → 放行(顶级牌)
    e3 = E([[card("As"), card("Ah")], [0, 0]], cur=16.0, sb=[6.0, 16.0],
           committed=[16.0, 6.0], stacks=[84.0, 94.0], mask=full)
    assert guard(e3, 0, ALLIN) == ALLIN, "AA 全下该放行"
    print("  AA 面对4bet 全下 → 放行 ✓")
    # 4) A6o 面对 4bet 想全下 → 降级(不够 5bet);但能跟(cap≈28≥16)→ 跟
    e4 = E([[card("Ad"), card("6c")], [0, 0]], cur=16.0, sb=[6.0, 16.0],
           committed=[16.0, 6.0], stacks=[84.0, 94.0], mask=full)
    o4 = guard(e4, 0, ALLIN)
    print(f"  A6o 面对4bet 想全下 → {o4} 投入{_commit_of(e4,0,o4):.0f}bb(应≠全下、投入受限)")
    assert o4 != ALLIN and _commit_of(e4, 0, o4) < 30, o4
    # 5) 已是 FOLD/CALL 且不超额 → 原样
    assert guard(e2, 0, CALL) == CALL
    print(f"  强度百分位抽样: AA={PCT['AA']:.2f} A6o={PCT['A6o']:.2f} T2o={PCT['T2o']:.2f} 72o={PCT['72o']:.2f}")
    # ---- 底池已承诺/赔率护栏 ----
    import random as _rnd
    # 6) 复刻实战:5h4h 翻牌 6h7s7d,已投 89.3bb,只需再跟 10.7bb(赔率~17:1)→ 两头顺听牌禁弃,改跟
    ec = E([[card("5h"), card("4h")], [0, 0]], cur=10.7, sb=[0.0, 10.7],
           committed=[89.3, 100.0], stacks=[10.7, 0.0], mask=full)
    ec.street = 1; ec.board = [card("6h"), card("7s"), card("7d")]
    ct = [0]
    oc = commit_guard_action(ec, 0, FOLD, rng=_rnd.Random(1), counters=ct)
    print(f"  5h4h 89bb已投·跟10.7争190池 → {oc}(应 CALL={CALL});救回计数={ct[0]}")
    assert oc == CALL and ct[0] == 1, oc
    # 7) 赔率不够(跟 10 争 20 池,2:1)→ 不介入,保持弃牌(可能是合理弃)
    ec2 = E([[card("5h"), card("4h")], [0, 0]], cur=10.0, sb=[0.0, 10.0],
            committed=[10.0, 10.0], stacks=[90.0, 90.0], mask=full)
    ec2.street = 1; ec2.board = [card("6h"), card("7s"), card("7d")]
    assert commit_guard_action(ec2, 0, FOLD, rng=_rnd.Random(1)) == FOLD, "赔率不够不该介入"
    print("  赔率不够(2:1) → 不介入、保持弃牌 ✓")
    # 8) 翻前不作用(street=0)
    assert commit_guard_action(ec2, 0, FOLD, rng=_rnd.Random(1)) == FOLD  # ec2.street=1;再测 street=0
    ec2.street = 0
    assert commit_guard_action(ec2, 0, FOLD, rng=_rnd.Random(1)) == FOLD, "翻前不该作用"
    # ---- 放弃/别拿弱牌接大注护栏(air-fold)----
    # 9) 空气 2c3d,河牌 AKQ94,面对底池大小的全下(1:1,需50%)→ CALL 应被改成 FOLD
    ea = E([[card("2c"), card("3d")], [0, 0]], cur=50.0, sb=[0.0, 50.0],
           committed=[50.0, 100.0], stacks=[50.0, 0.0], mask=full)
    ea.street = 3; ea.board = [card("Ah"), card("Kd"), card("Qs"), card("9s"), card("4h")]
    ac = [0]
    oa = air_fold_guard_action(ea, 0, CALL, rng=_rnd.Random(2), counters=ac)
    print(f"  2c3d 面对底池全下(需50%胜率)→ {oa}(应 FOLD={FOLD});弃牌计数={ac[0]}")
    assert oa == FOLD and ac[0] == 1, oa
    # 10) 同样空气但只面对小注(跟5争100池=20:1)→ 不弃(小注不碰,避免被诈唬剥削)
    ea2 = E([[card("2c"), card("3d")], [0, 0]], cur=5.0, sb=[0.0, 5.0],
            committed=[5.0, 100.0], stacks=[95.0, 0.0], mask=full)
    ea2.street = 3; ea2.board = [card("Ah"), card("Kd"), card("Qs"), card("9s"), card("4h")]
    assert air_fold_guard_action(ea2, 0, CALL, rng=_rnd.Random(2)) == CALL, "小注不该弃"
    # 11) 面前是过牌(to_call=0)→ 绝不把 CALL 改成弃
    ea3 = E([[card("2c"), card("3d")], [0, 0]], cur=0.0, sb=[0.0, 0.0],
            committed=[20.0, 20.0], stacks=[80.0, 80.0], mask=full)
    ea3.street = 3; ea3.board = [card("Ah"), card("Kd"), card("Qs"), card("9s"), card("4h")]
    assert air_fold_guard_action(ea3, 0, CALL, rng=_rnd.Random(2)) == CALL, "过牌不该被改弃"
    print("  air-fold: 空气接大注→弃 / 小注不碰 / 过牌不碰 ✓")
    print("[selftest] OK —— 垃圾牌深筹降级/顶级牌放行/开池不拦/CALL兜底/底池已承诺/放弃护栏 全部正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true", help="打印 169 手牌强度百分位(降序)")
    args = ap.parse_args()
    if args.show:
        for h in sorted(PCT, key=PCT.get, reverse=True):
            print(f"{h:>4} eq={HAND_EQ[h]:.3f} pct={PCT[h]:.3f}")
        return
    if args.selftest:
        _selftest(); return
    ap.error("需 --selftest 或 --show")


if __name__ == "__main__":
    main()
