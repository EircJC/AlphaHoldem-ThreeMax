"""终端人机对打——你(人)在【指定深度档】跟你训练的模型单挑。

按档一键起：档位自动对齐【训练时的筹码范围 + 归一化基准(500)】和对应快照，特征口径不错位。
  short 15~60bb / mid 50~210bb / deep 190~300bb / vdeep 290~500bb

用法：
  # 跟 mid 档模型对打(每手在该档范围内随机筹码,贴近训练分布)
  python3 play_vs_bot.py --depth mid
  # 固定 200bb 深筹打，只打 20 手自动结束
  python3 play_vs_bot.py --depth deep --fixed 200 --hands 20
  # 指定快照 / 换深度
  python3 play_vs_bot.py --depth vdeep --model runs/spec_vdeep/model_iter100.pt
  # 让模型的【河牌】用 solver 现解(近似,慢;需 TexasSolver-master)
  python3 play_vs_bot.py --depth mid --river-solve

  # 【自动对打测试】对手bot(seat0) vs 你的模型(seat1),跑完直接报 bb/100,无需手输
  python3 play_vs_bot.py --depth mid --auto --opp call   --hands 2000   # vs 只跟注基线
  python3 play_vs_bot.py --depth mid --auto --opp random --hands 2000   # vs 随机基线
  python3 play_vs_bot.py --depth deep --auto --opp deep_iter140.pt --hands 5000  # vs 另一快照
  python3 play_vs_bot.py --depth mid --auto --opp call --hands 200 --verbose     # 逐动作看细节
  # 【漏洞诊断】按位置/角色/街道/摊牌归因模型输赢,定位 -21bb/100 亏在哪
  python3 play_vs_bot.py --depth mid --model mid-0804.pt --auto --opp bcpoker --hands 5000 --diag

对打时你是 seat 0，每手按钮轮换。动作用编号输入；raise 档会显示实际加到多少 bb。
--auto 模式下 seat0 交给 --opp 指定的对手自动打,结果以【你的模型】视角也会打印。
"""
import argparse
import os
import platform
import random
import time

import torch

from engine import HUNLEnv, card_str
from engine.evaluator import evaluate7
from encoder import encode
from network import AlphaHoldemNet
from config import (GameConfig, TrainConfig, NB, FOLD, CALL, ALLIN, RAISE_FRACTIONS,
                    ACTION_NAMES, pick_device)

# 档位预设：name -> (筹码下限, 上限, 默认快照)。归一化基准统一 500(=训练 --norm 500)。
DEPTHS = {
    "short": (15.0, 60.0, "short_iter210.pt"),
    "mid": (50.0, 210.0, "mid_iter120.pt"),
    "deep": (150.0, 500.0, "deep_iter140.pt"),   # 新阶梯:deep 专精按 150~500bb 训(norm=500 仍够,顶 500/500=1.0)
    "vdeep": (290.0, 500.0, None),          # 本机无 vdeep 快照，需 --model 指定
    # 注意:super-deep(500~1500bb)需 norm=1500,DEPTHS 只存筹码不存 norm →
    #       跑超深筹必须手动 --chips 500,1500 --norm 1500 --model <超深专精>(见下方说明)。
}
NORM = 500.0
STREET_CN = {0: "翻牌前", 1: "翻牌", 2: "转牌", 3: "河牌"}


def load_net(path, device):
    net = AlphaHoldemNet(TrainConfig().num_actions, NB).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def net_policy(net, device, greedy=False):
    def pol(env):
        s = env.encoding_state(env.to_act)
        c, a, sc, legal = encode(s)
        def b(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        idx, _, _ = net.act(b(c), b(a), b(sc), b(legal), greedy=greedy)
        act = int(idx.item())
        return act if legal[act] > 0.5 else CALL
    return pol


def show_options(env):
    """打印人类可选动作，raise 档换算成实际加到多少 bb。返回合法动作编号列表。"""
    p = env.to_act
    to_call = env.current_bet - env.street_bet[p]
    pot_after_call = sum(env.committed) + to_call
    legal = [i for i, v in enumerate(env.legal_mask()) if v]
    lines = []
    for i in legal:
        if i == FOLD:
            lines.append(f"[{i}] 弃牌")
        elif i == CALL:
            lines.append(f"[{i}] {'过牌' if to_call < 1e-9 else f'跟注 {to_call:.1f}bb'}")
        elif i == ALLIN:
            lines.append(f"[{i}] 全下(剩 {env.stacks[p]:.1f}bb)")
        else:
            frac = RAISE_FRACTIONS[i]
            total_add = to_call + frac * pot_after_call
            raise_to = env.street_bet[p] + total_add
            lines.append(f"[{i}] 加注{round(frac*100)}%池 → 加到 {raise_to:.1f}bb(本手共投 +{total_add:.1f})")
    print("  可选: " + "   ".join(lines))
    return legal


def make_river_cfg(args):
    import river_solver as rs
    sd = args.solver_dir
    return {
        "solver_bin": os.path.join(sd, "console_solver"),
        "resource_dir": os.path.join(sd, "resources"),
        "run_prefix": ["arch", "-x86_64"] if platform.machine() == "arm64" else [],
        "ip_range": args.river_range or rs.DEFAULT_RANGE,
        "oop_range": args.river_range or rs.DEFAULT_RANGE,
        "threads": args.river_threads, "iters": args.river_iters,
        "accuracy": 0.5, "timeout": 120,
    }


def preflop_table_bucket(env, seat, table, greedy, rng):
    """翻前查 GTO 表 → 动作 bucket;查不到(手牌/场景不在表)返回 None 交回模型。"""
    import preflop_table as pt
    if env.street != 0:
        return None
    eff = min(env.start_stacks)
    hero = "SB" if seat == env.button else "BB"          # HU:按钮=SB
    facing_to = env.current_bet                          # 当前面对的到额(bb):开池处=大盲1.0
    hole = env.holes[seat]
    r = pt.lookup(table, eff, hero, facing_to, pt.hand169(hole[0], hole[1]), greedy, rng)
    if r is None:
        return None
    act, to_bb = r
    mask = env.legal_mask()
    if act == "fold":
        return FOLD if mask[FOLD] else CALL
    if act == "call":
        return CALL
    if act == "allin":
        return ALLIN if mask[ALLIN] else CALL
    if to_bb is None:                                    # raise 但无到额 → 交回模型
        return None
    to_call = env.current_bet - env.street_bet[seat]
    pot_after_call = sum(env.committed) + to_call
    my_sb = env.street_bet[seat]
    allin_to = my_sb + env.stacks[seat]
    if mask[ALLIN] and to_bb >= allin_to - 1e-6:
        return ALLIN
    best, bd = None, 1e18                                # 到额 → 最近的合法加注档
    for k in RAISE_FRACTIONS:
        if not mask[k]:
            continue
        raise_to = my_sb + to_call + RAISE_FRACTIONS[k] * pot_after_call
        if abs(raise_to - to_bb) < bd:
            best, bd = k, abs(raise_to - to_bb)
    return best if best is not None else (ALLIN if mask[ALLIN] else CALL)


def ai_river_bucket(env, cfg, greedy):
    """AI(seat 1) 在河牌用 solver 现解一个动作 bucket；失败返回 None。"""
    import river_solver as rs
    board5 = [card_str(c) for c in env.board]
    if len(board5) != 5:
        return None
    pot_river = sum(env.committed) - sum(env.street_bet)          # 河牌起始底池
    eff_river = min(env.stacks[0] + env.street_bet[0],
                    env.stacks[1] + env.street_bet[1])            # 河牌起始有效筹码(剩余)
    rh = [(pl == 1, act) for pl, act in env.history[3]]           # hero=AI(seat1)
    hero_is_oop = (env.button == 0)                              # AI 是翻后先手 ⇔ button==0
    hole = [card_str(c) for c in env.holes[1]]
    b = rs.river_action(board5, pot_river, eff_river, rh, hero_is_oop, hole, greedy, cfg)
    reason = getattr(rs, "LAST_REASON", "?")
    if b is not None and not env.legal_mask()[b]:
        return None, "illegal"
    return b, reason


def build_opp(spec, net, device, greedy, args=None):
    """构造 seat0 自动对手策略。spec: self=同模型 / call / random / bcpoker=BCPoker暖场bot /
    gto=GTO顶档(翻前表+翻转查库+兜底) / <路径.pt>=另一模型。返回 (policy, 显示名)。"""
    if spec == "gto":
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data"))
        import board_match, gto_opponent
        if not args or not getattr(args, "gto_lib_dir", None):
            raise SystemExit("--opp gto 需 --gto-lib-dir 指向 GTO 策略库目录(solver_parse --dump-lib 产出)")
        lib_index = board_match.build_multi_index(args.gto_lib_dir)   # 多深度:按有效筹码就近选 short/mid/deep
        prng = random.Random((args.seed if args else 0) + 999)
        ptable = None
        if args and getattr(args, "preflop_table", None):
            import preflop_table as pt
            ptable = pt.load(args.preflop_table)
        preflop_fn = (lambda env: preflop_table_bucket(env, env.to_act, ptable, greedy, prng)) if ptable else None
        fallback = net_policy(net, device, greedy=greedy)     # 河牌 + 未命中一律回退基座(里程碑先不接河牌现解)
        pol = gto_opponent.make_gto_policy(0, lib_index, preflop_fn, None, fallback, greedy, prng)
        bands = getattr(lib_index, "summary", lambda: {})()
        band_str = " ".join(f"{d}bb:{n}" for d, n in sorted(bands.items())) if bands else f"{len(lib_index)}板"
        return pol, f"GTO顶档(深度带 {band_str}"+(",翻前查表" if ptable else "")+")"
    if spec == "call":
        return (lambda env: CALL), "call基线"
    if spec == "random":
        return (lambda env: random.choice([i for i, v in enumerate(env.legal_mask()) if v])), "random基线"
    if spec == "self":
        return net_policy(net, device, greedy=greedy), "同模型(自对弈)"
    if spec == "bcpoker":
        from bcpoker_opp import make_bcpoker_policy
        return make_bcpoker_policy(seat=0)            # BCPoker 引擎当 seat0 对手(需 node)
    if not os.path.exists(spec):
        raise SystemExit(f"--opp 无法识别: {spec}(应为 self/call/random/bcpoker 或 .pt 路径)")
    onet = load_net(spec, device)
    return net_policy(onet, device, greedy=greedy), os.path.basename(spec)


def _diag_acc(diag, dim, key, payoff):
    d = diag.setdefault(dim, {})
    s, c = d.get(key, (0.0, 0))
    d[key] = (s + payoff, c + 1)


_CAT_NAMES = ("高牌", "一对", "两对", "三条", "顺子", "同花", "葫芦", "四条", "同花顺")


def _diag_record(diag, env):
    """一手结束后,按模型(seat1)视角把这手的收益归到各维度。"""
    pf = env.payoffs[1]
    role = "def"
    for pl, a in env.history[0]:                      # 翻前最后一个自己的动作定角色
        if pl == 1:
            role = "pfa" if a >= 2 else "def"
    ip = (env.button == 1)                            # seat1 是按钮 → 翻后 IP
    street = {0: "翻前", 1: "翻牌", 2: "转牌", 3: "河牌"}[env.street]
    end = "弃牌结束" if env.folded is not None else "摊牌"
    _diag_acc(diag, "按位置", "IP" if ip else "OOP", pf)
    _diag_acc(diag, "按翻前角色", "进攻pfa" if role == "pfa" else "防守def", pf)
    _diag_acc(diag, "按结束街道", street, pf)
    _diag_acc(diag, "按结束方式", end, pf)
    _diag_acc(diag, "角色×街道", f"{'进攻' if role == 'pfa' else '防守'}-{street}", pf)
    if len(env.board) >= 3:                            # 按模型最终成手牌型统计盈亏
        cat = evaluate7(list(env.holes[1]) + list(env.board))[0]
        _diag_acc(diag, "按牌型", _CAT_NAMES[cat] if 0 <= cat < 9 else f"cat{cat}", pf)
    else:
        _diag_acc(diag, "按牌型", "翻前结束", pf)


def _value_check(diag, env, seat, action):
    """价值加注漏检:模型持【强牌(顺子+)】、面对下注(to_call>0)且可加注时,选了加注/只跟/弃?
    evaluate7 类别:4顺子 5同花 6葫芦 7四条 8同花顺;<4 不算强牌。"""
    if len(env.board) < 3:
        return
    to_call = env.current_bet - env.street_bet[seat]
    if to_call <= 1e-9:                                   # 只看"面对下注、有得加"的价值点
        return
    mask = env.legal_mask()
    if not (any(mask[k] for k in RAISE_FRACTIONS) or mask[ALLIN]):
        return
    hand_rank = evaluate7(list(env.holes[seat]) + list(env.board))
    if len(env.board) >= 5 and hand_rank <= evaluate7(list(env.board)):
        return                                           # 打公共牌(手牌没改善)→ 无价值可加,不算
    cat = hand_rank[0]
    if cat < 4:
        return
    tier = "葫芦+(近坚果)" if cat >= 6 else "顺子/同花"
    d = diag.setdefault("__value__", {}).setdefault(
        tier, {"spots": 0, "raised": 0, "called": 0, "folded": 0})
    d["spots"] += 1
    d["raised" if action >= 2 else ("called" if action == 1 else "folded")] += 1


def _diag_print(diag, hands):
    print(f"\n===== 漏洞诊断(模型 seat1 视角,负=模型亏,共 {hands} 手)=====")
    for dim in ("按位置", "按翻前角色", "按结束方式", "按结束街道", "角色×街道", "按牌型"):
        d = diag.get(dim, {})
        print(f"\n[{dim}]")
        for key, (s, c) in sorted(d.items(), key=lambda kv: kv[1][0]):   # 最亏的排前
            print(f"   {key:10s} n={c:5d} 占{100*c/max(hands,1):4.0f}%   "
                  f"贡献 {s:+8.1f}bb   {s/max(c,1)*100:+8.1f} bb/100")
    vd = diag.get("__value__")
    if vd:
        print("\n[价值加注漏检] 模型持强牌、面对下注且可加注时怎么选(只跟=漏价值):")
        for tier, d in vd.items():
            s = max(d["spots"], 1)
            print(f"   {tier:12s} {d['spots']:4d}次   加注{d['raised']}({100*d['raised']/s:3.0f}%)  "
                  f"只跟{d['called']}({100*d['called']/s:3.0f}%)  弃{d['folded']}({100*d['folded']/s:3.0f}%)")


def _eta(t0, done, total):
    """按已用时线性外推预计完成时钟时间 + 剩余分钟。"""
    if not total or done <= 0:
        return "?"
    el = time.time() - t0
    rem = (total - done) * (el / done)
    return time.strftime("%H:%M:%S", time.localtime(time.time() + rem)) + f"(剩{rem/60:.1f}分)"


def _hitmark(seen, hit):
    """本手各街命中标记:✓=命中GTO(查表/查库/现解/护栏干预);×=轮到但走了模型;—=本手没打到该街。"""
    lab = ("翻前", "翻牌", "转牌", "河牌")
    return " ".join(lab[i] + ("✓" if hit[i] else ("×" if seen[i] else "—")) for i in range(4))


def _log_big_hand(logf, hands, env, opp_name, trace, model_cum):
    """把一手大额牌写入日志——格式对齐 slumbot_client 的"复盘"风格:
    开始分隔线 → 复盘头 → 双方手牌 → 公共牌 → 逐步动作(含理由) → 结果 → 结束分隔线(本手/累计)。"""
    names = {0: "翻前", 1: "翻牌", 2: "转牌", 3: "河牌"}
    pf1 = env.payoffs[1]                              # 模型(seat1)本手输赢
    role = "def"
    for pl, a in env.history[0]:
        if pl == 1:
            role = "pfa" if a >= 2 else "def"
    model_btn = (env.button == 1)
    end = "弃牌" if env.folded is not None else "摊牌"
    bar = "━" * 30
    L = [f"\n{bar} 第 {hands} 手 开始 {bar}",
         f"===== 第 {hands} 手 复盘 =====",
         f"模型手牌：{' '.join(card_str(c) for c in env.holes[1])}"
         f"    {opp_name}手牌：{' '.join(card_str(c) for c in env.holes[0])}",
         f"位置：模型 {'按钮IP' if model_btn else '大盲OOP'}   翻前角色：{'进攻pfa' if role == 'pfa' else '防守def'}"
         f"   结束方式：{end}",
         f"起始筹码：模型 {env.start_stacks[1]:.1f} / {opp_name} {env.start_stacks[0]:.1f}bb"
         f"   有效筹码：{min(env.start_stacks):.1f}bb",
         f"最终公共牌：{' '.join(card_str(c) for c in env.board) or '—'}"
         + ("   ⚠河牌前全下→按【期望权益】结算(不发runout,收益=有效栈×(2×权益−1),故非整数)"
            if (env.all_in[0] and env.all_in[1] and len(env.board) < 5 and env.folded is None) else "")]
    for actor, street, a, note in trace:                 # 逐步动作(自然语言复盘)
        who = "模型  " if actor == 1 else opp_name
        extra = f"   « {note} »" if note else ""
        L.append(f"  [{names[street]}] {who}：{ACTION_NAMES[a]}{extra}")
    L.append(f"结果：模型 {pf1:+.1f}bb")
    L.append(f"{'━' * 27} 第 {hands} 手 结束（本手 {pf1:+.1f}bb，累计 {model_cum:+.1f}bb）{'━' * 27}\n")
    logf.write("\n".join(L) + "\n")
    logf.flush()


def play(net, device, game_cfg, river_cfg, greedy, seed, max_hands=0,
         seat0_pol=None, verbose=True, opp_name="对手bot", diag=None,
         log_path=None, big_frac=0.7, ptable=None, pf_guard=False, flopturn_fn=None,
         river_nut_guard=False, river_nut_thresh=0.97,
         commit_guard=False, commit_min_ratio=6.0,
         air_fold_guard=False, air_fold_min_bet=0.6, exploit=False, defer_summary=False):
    pol = net_policy(net, device, greedy=greedy)
    env = HUNLEnv(game_cfg, random.Random(seed))
    prng = random.Random(seed + 777)                 # 翻前查表采样用(可复现)
    cg_rng = random.Random(seed + 999)               # 底池已承诺护栏 equity 蒙特卡洛(可复现)
    button, bankroll, hands = 0, 0.0, 0
    river_used = [0, 0]
    river_reasons = {}                      # 河牌现解各结果原因计数(诊断 0% 用)
    ft_used = [0, 0]                         # 翻/转查库 [命中, 回退模型]
    ft_reasons = {}                          # 翻/转查库各结果原因
    nutguard_ct = [0]                        # 河牌近坚果护栏:救回的弃牌次数
    ptable_used = [0, 0]                     # 翻前查表 [命中, 回退模型](诊断表覆盖度)
    guard_ct = [0]                           # 翻前护栏降级次数
    commit_ct = [0]                          # 底池已承诺护栏:救回的弃牌次数
    airfold_ct = [0]                         # 放弃护栏:拿弱牌接大注被改弃的次数
    gto_seen = [0, 0, 0, 0]                  # 四街命中汇总:模型在各街(翻前/翻牌/转牌/河牌)轮到决策的手数
    gto_hit = [0, 0, 0, 0]                   # 各街命中GTO(查表/查库/现解/护栏干预)的手数
    result = {}                              # play() 返回给调用方(如 play_vs_styles)的本轮小结
    exploit_ct = {}                          # 剥削层触发计数(station_cutbluff/maniac_uncall)
    opp_tracker = None
    if pf_guard or river_nut_guard or commit_guard or air_fold_guard or exploit:
        import preflop_guard
    if exploit:
        import exploit_layer as _exmod
        opp_tracker = _exmod.OpponentTracker()   # 只跟踪对手(seat0);逐手 update
    auto = seat0_pol is not None            # seat0_pol=None → 人;否则 seat0 由该策略自动打
    seat0_name = opp_name if auto else "你"
    goal = f"共 {max_hands} 手" if max_hands else "无限(q 退出)"
    mode = f"【自动对打】{seat0_name}(seat0) vs 你的模型(seat1)" if auto else "开始对打。你是 seat 0"
    print(f"\n{mode}；目标手数: {goal}\n")
    t0 = time.time()
    logf = open(log_path, "w", encoding="utf-8") if (log_path and auto) else None
    if logf:
        logf.write(f"# 大额手牌日志  |模型输赢| ≥ {big_frac:.0%}×有效筹码  对手={opp_name}\n")
        logf.flush()

    def finish(interrupted=False):
        """结算:控制台 + 日志都输出统计小结(正常结束/中断 Ctrl+C 都走这里)。"""
        final = bankroll / max(hands, 1) * 100
        persp = f"{seat0_name}(seat0)相对你的模型" if auto else "你相对模型"
        head = "已中断(Ctrl+C)" if interrupted else "结束"
        L = [f"{head}。共 {hands} 手，用时 {time.time()-t0:.1f}s",
             f"  {persp}: {bankroll:+.1f}bb 累计  ~{final:+.1f} bb/100"
             + (f"  ⇒ 你的模型 {-final:+.1f} bb/100" if auto else "")]
        serving = (flopturn_fn is not None or ptable is not None or river_cfg is not None
                   or pf_guard or river_nut_guard)
        if serving:                                       # 四街命中汇总(常驻):命中=查表/查库/现解/护栏干预
            lab = ("翻前", "翻牌", "转牌", "河牌")
            parts = [f"{lab[i]} {gto_hit[i]}/{gto_seen[i]}({100*gto_hit[i]/max(gto_seen[i],1):.0f}%)"
                     for i in range(4)]
            L.append("命中汇总(命中/轮到手, 命中率): " + "   ".join(parts))
        if flopturn_fn is not None:
            tot = sum(ft_used)
            L.append(f"翻/转查库: 命中 {ft_used[0]} / 回退模型 {ft_used[1]}"
                     f"（成功率 {100*ft_used[0]/max(tot,1):.0f}%;覆盖不足=库板/深度带不够或导航未命中）")
            top = sorted(ft_reasons.items(), key=lambda kv: -kv[1])[:5]
            if top:
                L.append("  翻/转未命中原因 Top: " + " ".join(f"{r}={n}" for r, n in top))
        if river_cfg is not None:
            tot = sum(river_used)
            L.append(f"河牌现解: 用solver {river_used[0]} / 回退模型 {river_used[1]}"
                     f"（成功率 {100*river_used[0]/max(tot,1):.0f}%）")
        cov = getattr(seat0_pol, "counters", None)          # GTO顶档对手:覆盖率(库命中 vs 回退)
        if cov is not None:
            for phase in ("preflop", "flopturn", "river"):
                hit, fb = cov[phase]; t = hit + fb
                if t:
                    L.append(f"GTO覆盖·{phase}: 命中 {hit} / 回退 {fb}（{100*hit/t:.0f}%）")
            top = sorted(cov["reasons"].items(), key=lambda kv: -kv[1])[:5]
            if top:
                L.append("  翻/转未命中原因 Top: " + " ".join(f"{r}={n}" for r, n in top))
            for ex in cov.get("nav_dbg", [])[:15]:
                L.append("    [导航失败样本] " + ex)
        if ptable is not None:
            tt = sum(ptable_used)
            L.append(f"翻前查表: 命中 {ptable_used[0]} / 回退模型 {ptable_used[1]}"
                     f"（共{tt}次翻前决策,命中率 {100*ptable_used[0]/max(tt,1):.0f}%;"
                     f"回退多=表覆盖不足,防守点走了基座）")
        if pf_guard:
            L.append(f"翻前护栏: 降级 {guard_ct[0]} 次(垃圾牌深筹投入被降为跟/弃)")
        if river_nut_guard:
            L.append(f"河牌近坚果护栏: 救回弃牌 {nutguard_ct[0]} 次(近坚果 FOLD→CALL)")
        if commit_guard:
            L.append(f"底池已承诺护栏: 救回弃牌 {commit_ct[0]} 次(高赔率有胜率 FOLD→CALL)")
        if air_fold_guard:
            L.append(f"放弃护栏: 拿弱牌接大注被改弃 {airfold_ct[0]} 次(对强范围无赔率 CALL→FOLD)")
        if exploit and opp_tracker is not None:
            rd = opp_tracker.read()
            L.append(f"剥削层: 对手读作【{rd['label']}】(loose={rd['loose']} pfr={rd['pfr']} af={rd['af']});"
                     f"站砍诈唬 {exploit_ct.get('station_cutbluff',[0])[0]} 次 / 疯子放宽抓诈 {exploit_ct.get('maniac_uncall',[0])[0]} 次")
        final = bankroll / max(hands, 1) * 100
        result.update({
            "hands": hands, "bankroll": bankroll,
            "model_bb100": (-final if auto else final),      # 模型视角 bb/100
            "gto_seen": list(gto_seen), "gto_hit": list(gto_hit),
            "guard_ct": guard_ct[0], "nutguard_ct": nutguard_ct[0],
            "commit_ct": commit_ct[0], "airfold_ct": airfold_ct[0],
            "ft_used": list(ft_used), "interrupted": interrupted,
            "opp_label": (opp_tracker.read()["label"] if opp_tracker is not None else None),
            "exploit_cutbluff": exploit_ct.get("station_cutbluff", [0])[0],
            "exploit_uncall": exploit_ct.get("maniac_uncall", [0])[0],
            "summary_text": "\n".join(L),                    # 供 play_vs_styles 统一在结尾输出
        })
        if not defer_summary:                                # defer=True:不立即打结算块(调用方在结尾统一打)
            print("\n" + "\n".join(L))
        if river_cfg is not None:
            print("  失败/结果原因 Top:")
            for reason, n in sorted(river_reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f"    {n:5d}  {reason}")
        if diag is not None:
            _diag_print(diag, hands)
        if logf is not None:
            logf.write(f"\n# ==== 统计小结{'（中断）' if interrupted else ''} ====\n")
            for ln in L:
                logf.write(ln.lstrip() + "\n")
            logf.flush()
            logf.close()
            print(f"日志(大额手牌+统计小结)已写入: {log_path}")

    try:
      while True:
        env.reset(button)
        hands += 1
        _hand_t0 = time.time()                        # 本手计时(诊断:看是每手都慢还是只有河牌手慢)
        trace = []                                    # 本手逐步决策(actor,街道,动作,理由)
        hand_river = 0                                # 本手模型河牌现解成功次数
        seen = [False, False, False, False]           # 本手模型在各街(翻前/翻牌/转牌/河牌)是否轮到决策
        hit = [False, False, False, False]            # 各街是否命中GTO(查表/查库/现解/护栏干预)
        if verbose:
            print(f"==== 第 {hands} 手  ({seat0_name}按钮位: {'是' if button == 0 else '否'}  "
                  f"起始筹码 {env.start_stacks[0]:.0f} / {env.start_stacks[1]:.0f}bb) ====")
        last_street = -1
        while not env.done:
            if verbose and env.street != last_street:
                last_street = env.street
                print(f"-- {STREET_CN[env.street]}  公共牌: {[card_str(c) for c in env.board] or '—'}")
            p = env.to_act
            to_call = env.current_bet - env.street_bet[p]
            if p == 0 and not auto:                       # 人类走这支
                print(f"  你的手牌 {[card_str(c) for c in env.holes[0]]}  "
                      f"底池 {sum(env.committed):.1f}bb  你剩 {env.stacks[0]:.1f}bb  需跟 {to_call:.1f}bb")
                legal = show_options(env)
                try:
                    a = int(input("  你的动作> ").strip())
                except (ValueError, EOFError):
                    a = CALL
                if a not in legal:
                    print("  (非法,默认 过牌/跟注)")
                    a = CALL
                env.step(a)
            elif p == 0:                                  # seat0 自动对手
                a = seat0_pol(env)
                trace.append((0, env.street, a, getattr(seat0_pol, "last_note", None)))
                if verbose:
                    print(f"  {seat0_name}: {ACTION_NAMES[a]}")
                env.step(a)
            else:                                         # seat1 = 你的模型(翻前可查表/护栏,翻转可查库,河牌可现解)
                a, from_solver, from_table, from_lib = None, False, False, False
                seen[env.street] = True                    # 本手模型在这条街轮到了决策
                opp_read = opp_tracker.read() if opp_tracker is not None else None
                # 读作【疯子】时:关掉查库+air-fold,让重训后的模型自己打(它对疯子的反打比通用GTO库/保守弃牌更强)
                _is_maniac = bool(exploit and opp_read and opp_read["label"] == "maniac")
                if ptable is not None and env.street == 0:
                    pb = preflop_table_bucket(env, 1, ptable, greedy, prng)
                    if pb is not None:
                        a, from_table = pb, True
                        ptable_used[0] += 1                   # 命中查表
                    else:
                        ptable_used[1] += 1                   # 未命中 → 回退基座
                if a is None and flopturn_fn is not None and env.street in (1, 2) and not _is_maniac:
                    fb, freason = flopturn_fn(env)            # 翻/转查 GTO 库(返回 档或None, 原因)
                    ft_reasons[freason] = ft_reasons.get(freason, 0) + 1
                    if fb is not None:
                        a, from_lib, ft_used[0] = fb, True, ft_used[0] + 1
                    else:
                        ft_used[1] += 1
                if a is None and river_cfg is not None and env.street == 3:
                    rb, reason = ai_river_bucket(env, river_cfg, greedy)
                    river_reasons[reason] = river_reasons.get(reason, 0) + 1
                    if rb is not None:
                        a, from_solver, river_used[0] = rb, True, river_used[0] + 1
                    else:
                        river_used[1] += 1
                if a is None:
                    a = pol(env)
                gc0 = guard_ct[0]
                # 读作疯子时也关翻前护栏:重训模型学会对疯子加宽 3bet/4bet 反打,护栏会把这些"深筹投入"误降级
                if pf_guard and env.street == 0 and not _is_maniac:   # 翻前护栏:按手牌强度封顶投入,只降级
                    a = preflop_guard.guard(env, 1, a, guard_ct)
                nc0 = nutguard_ct[0]
                if river_nut_guard and env.street == 3:   # 河牌护栏:近坚果绝不弃(FOLD→CALL)
                    a = preflop_guard.river_nut_guard_action(env, 1, a, river_nut_thresh, nutguard_ct)
                cc0 = commit_ct[0]
                if commit_guard and env.street >= 1:      # 底池已承诺/赔率护栏:高赔率有胜率则禁弃(FOLD→CALL)
                    a = preflop_guard.commit_guard_action(env, 1, a, commit_min_ratio,
                                                          rng=cg_rng, counters=commit_ct)
                af0 = airfold_ct[0]
                # 放弃护栏:拿弱牌接大注/全下且对强范围没赔率 → 弃(CALL→FOLD)。
                # 仅在 commit 护栏本步【未救牌】时才跑;读作疯子时关掉(交给重训模型,别对疯子空气大注乱弃)。
                if air_fold_guard and env.street >= 1 and commit_ct[0] == cc0 and not _is_maniac:
                    a = preflop_guard.air_fold_guard_action(env, 1, a, air_fold_min_bet,
                                                            rng=cg_rng, counters=airfold_ct)
                # 剥削层:站砍诈唬照常;但【疯子】让位给重训模型(它裸测已赢 G,再放宽抓诈=过度买单其价值)
                if exploit and not _is_maniac:            # 最后一步、有最终决定权
                    a = _exmod.exploit_adjust(env, 1, a, opp_read, rng=cg_rng, counters=exploit_ct)
                # 本手各街命中标记:查表/查库/现解命中,或护栏实际干预了动作,都算命中GTO
                if from_table or guard_ct[0] > gc0:
                    hit[0] = True
                if from_lib and env.street in (1, 2):
                    hit[env.street] = True
                if commit_ct[0] > cc0 or airfold_ct[0] > af0:   # 底池已承诺/放弃护栏干预了这条街
                    hit[env.street] = True
                if from_solver or nutguard_ct[0] > nc0:
                    hit[3] = True
                if from_solver:
                    hand_river += 1
                if diag is not None:
                    _value_check(diag, env, 1, a)         # 价值加注漏检(决策前局面)
                trace.append((1, env.street, a,
                              "翻前查表" if from_table else ("翻转查库" if from_lib else ("河牌现解" if from_solver else None))))
                if verbose:
                    tagv = "  [翻转查库]" if from_lib else ("  [河牌现解]" if from_solver else "")
                    print(f"  模型: {ACTION_NAMES[a]}" + tagv)
                env.step(a)
        if diag is not None:
            _diag_record(diag, env)
        if opp_tracker is not None:                       # 逐手累加对手读数(供下一手的剥削判断)
            opp_tracker.update(env.history, 0)
        res = env.payoffs[0]                              # 始终以 seat0 视角记账
        bankroll += res
        hands_bb100 = bankroll / hands * 100
        if logf is not None and abs(env.payoffs[1]) >= big_frac * min(env.start_stacks):
            _log_big_hand(logf, hands, env, opp_name, trace, -bankroll)   # 大额手牌落盘(累计=模型视角)
        gto_mark = _hitmark(seen, hit)                    # 本手各街命中情况:翻前✓ 翻牌✓ 转牌× 河牌—
        for _i in range(4):                               # 累加到四街命中汇总
            gto_seen[_i] += 1 if seen[_i] else 0
            gto_hit[_i] += 1 if hit[_i] else 0
        if verbose:
            who = seat0_name
            print(f"  本手: {who}{'赢' if res > 0 else ('输' if res < 0 else '平')} {res:+.1f}bb   "
                  f"模型手牌 {[card_str(c) for c in env.holes[1]]}   命中[{gto_mark}]   "
                  f"累计 {bankroll:+.1f}bb  (~{hands_bb100:+.1f} bb/100)\n")
        elif auto:                                        # 自动模式:每手显示本手输赢 + 命中情况 + 累计 + 预计完成
            pf1 = env.payoffs[1]
            print(f"  [{hands}/{max_hands or '∞'}] 模型本手 {pf1:+6.1f}bb | 命中[{gto_mark}] | "
                  f"本手 {time.time()-_hand_t0:.2f}s | "
                  f"累计 {-bankroll:+.1f} ~{-hands_bb100:+.1f} bb/100 | 预计完成 {_eta(t0, hands, max_hands)}")
        else:                                             # 人类非 verbose:每手也报本手+累计 bb/100
            print(f"  第{hands}手: 你 {res:+.1f}bb | 累计 {bankroll:+.1f}bb ~{hands_bb100:+.1f} bb/100")
        button ^= 1
        if max_hands and hands >= max_hands:
            if verbose:
                print(f"已打满 {max_hands} 手,结束。")
            break
        if not auto and input("再来一手? [回车=是 / q=退出] ").strip().lower() == "q":
            break
    except KeyboardInterrupt:
        print("\n[已中断 Ctrl+C] 正在统计并写入日志...")
        finish(interrupted=True)
        return result
    finish(interrupted=False)
    return result


def _add_bool(ap, name, default, help):
    """默认可开关的布尔参数:--name 开 / --no-name 关。优先用 BooleanOptionalAction(py3.9+),
    老 Python 回退到 store_true/store_false 互斥组(兼容解算机可能的旧 Python)。"""
    dest = name.replace("-", "_")
    boa = getattr(argparse, "BooleanOptionalAction", None)
    if boa is not None:
        ap.add_argument(f"--{name}", action=boa, default=default, help=help)
    else:
        g = ap.add_mutually_exclusive_group()
        g.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help)
        g.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"关闭 --{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", choices=list(DEPTHS), help="深度档:short/mid/deep/vdeep")
    ap.add_argument("--model", default=None, help="覆盖默认快照路径")
    ap.add_argument("--chips", default=None, help="自定义筹码范围 'lo,hi'(覆盖档位默认)")
    ap.add_argument("--fixed", type=float, default=None, help="固定筹码(bb),如 200。覆盖范围")
    ap.add_argument("--norm", type=float, default=NORM, help=f"归一化基准(训练=500)。默认{NORM:.0f}")
    ap.add_argument("--hands", type=int, default=0, help="对打手数;0=无限(打到 q 退出)")
    ap.add_argument("--auto", action="store_true",
                    help="自动对打(无需手输):对手bot(seat0) vs 你的模型(seat1),跑完报 bb/100")
    ap.add_argument("--opp", default="call",
                    help="自动模式下 seat0 对手:self=同模型 / call / random / bcpoker=BCPoker暖场bot / <快照.pt路径>")
    ap.add_argument("--verbose", action="store_true", help="自动模式也逐动作打印(默认只报进度+结果)")
    ap.add_argument("--diag", action="store_true",
                    help="自动模式:按位置/角色/街道/摊牌等维度归因模型输赢,定位漏洞在哪")
    ap.add_argument("--log", default=None,
                    help="自动模式:把大额手牌(见 --big)的双方牌/公共牌/完整决策+理由写入该日志文件")
    ap.add_argument("--big", type=float, default=0.7,
                    help="大额阈值=|模型输赢|≥该比例×有效筹码(默认 0.7,即赢/输超过 70%% 筹码的手)")
    ap.add_argument("--greedy", action="store_true", help="AI 确定性打法(默认按概率采样)")
    ap.add_argument("--ev-allin", action="store_true",
                    help="河牌前全下按【期望权益】结算(低方差,训练口径);默认关=按实际发牌算真实输赢")
    ap.add_argument("--preflop-table", default=None,
                    help="翻前 GTO 查表 json(preflop_table.py build 生成):翻前查表出招,翻后走模型")
    ap.add_argument("--gto-lib-dir", default=None,
                    help="GTO 顶档策略库目录(solver_parse --dump-lib 产出);配合 --opp gto 用。翻/转查库、翻前配 --preflop-table、其余回退基座")
    ap.add_argument("--flopturn-db", default=None,
                    help="【推荐】让模型(seat1)翻/转查 per-node SQLite 库(lib_to_sqlite.py 建的 gto_lib.db):单节点查询,几乎零常驻内存,大库不再 OOM/卡。与 --flopturn-lib 二选一(它优先)")
    ap.add_argument("--flopturn-lib", default=None,
                    help="旧:整板内存库根目录(按有效筹码就近选 short/mid/deep)。大库会把整块 60MB JSON 反复读进内存→OOM/慢,建议改用 --flopturn-db")
    _add_bool(ap, "river-nut-guard", True,
              "河牌护栏:模型将弃掉【近坚果】(打败≥阈值比例的对手手)且面对下注时,强制改为跟(治现解/NN 弃坚果)。【默认开】,--no-river-nut-guard 关闭")
    ap.add_argument("--river-nut-thresh", type=float, default=0.97,
                    help="判'近坚果'的胜过比例阈值(默认0.97=打败97%对手手才护;越高越保守只救铁坚果)")
    _add_bool(ap, "commit-guard", True,
              "翻后底池已承诺/赔率护栏:面对下注、赔率极大(赢:赌≥阈值)且胜率吃得下时,禁止弃牌→改跟(治'已投大半身价却弃高赔率听牌')。【默认开】,--no-commit-guard 关闭")
    ap.add_argument("--commit-min-ratio", type=float, default=6.0,
                    help="commit 护栏触发的最小赔率(赢:赌;默认6=只需~14%胜率才介入,避免碰接近GTO的边缘弃牌)")
    _add_bool(ap, "air-fold-guard", True,
              "翻后放弃护栏:拿弱牌接【大注/全下】(≥air-fold-min-bet×底池)且对较强范围连保本赔率都不到时,CALL→FOLD(治'诈唬被抓还拿空气跟全下')。【默认开】,--no-air-fold-guard 关闭")
    ap.add_argument("--air-fold-min-bet", type=float, default=0.6,
                    help="air-fold 只碰≥该比例底池的大注(默认0.6);小注一律不弃,避免被诈唬剥削")
    ap.add_argument("--preflop-guard", action="store_true",
                    help="翻前护栏:按手牌强度封顶翻前投入,垃圾牌深筹动作降为跟/弃(拦 5bet 送死类炸弹;开池/3bet不动)")
    ap.add_argument("--exploit", action="store_true",
                    help="剥削层:在线读对手(AFq/VPIP)→ 对【跟注站】砍诈唬、对【疯子】放宽抓诈少弃;只在高置信判为站/疯子时介入,tag/nit/未知不动(保稳健)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default=None)
    # 河牌现解(可选)
    ap.add_argument("--river-solve", action="store_true", help="AI 河牌用 solver 现解(近似,慢)")
    ap.add_argument("--solver-dir", default="TexasSolver-v0.2.0-MacOs",
                    help="TexasSolver 目录(含 console_solver 与 resources)")
    ap.add_argument("--river-range", default=None)
    ap.add_argument("--river-iters", type=int, default=80)
    ap.add_argument("--river-threads", type=int, default=4)
    args = ap.parse_args()

    if not args.depth and not (args.model and (args.chips or args.fixed)):
        ap.error("需 --depth，或同时给 --model 和 --chips/--fixed")

    lo, hi, default_model = (DEPTHS[args.depth] if args.depth else (None, None, None))
    if args.chips:
        parts = [p for p in args.chips.replace(" ", "").split(",") if p]
        if len(parts) != 2:
            ap.error("--chips 需 '下限,上限',如 50,210(固定用 --fixed)")
        lo, hi = sorted(float(p) for p in parts)
    if args.fixed:
        lo = hi = args.fixed
    model_path = args.model or default_model
    if not model_path:
        ap.error(f"{args.depth} 档本机无默认快照,请用 --model 指定")
    if not os.path.exists(model_path):
        ap.error(f"找不到模型: {model_path}")

    device = args.device or pick_device()
    game_cfg = GameConfig(random_stacks=True, min_stack=lo, max_stack=hi, stack_norm=args.norm,
                          allin_ev=args.ev_allin)   # 默认 False=按实际发牌结算真实输赢(非期望)
    river_cfg = make_river_cfg(args) if args.river_solve else None

    print(f"[档位] {args.depth or '自定义'}  筹码 {lo:.0f}~{hi:.0f}bb "
          f"{'(固定)' if lo == hi else '(每手随机)'}  归一化 {args.norm:.0f}")
    print(f"[模型] {model_path}   设备 {device}   打法 {'greedy' if args.greedy else '采样'}"
          + ("   [河牌现解 开]" if river_cfg else ""))
    net = load_net(model_path, device)

    seat0_pol, verbose, hands, opp_name = None, True, args.hands, "对手bot"
    if args.auto:
        seat0_pol, opp_name = build_opp(args.opp, net, device, args.greedy, args)
        verbose = args.verbose
        if hands <= 0:
            hands = 200                                   # 自动模式默认 200 手,避免无限
        print(f"[自动] seat0 对手 = {opp_name}   手数 = {hands}")
    diag = {} if (args.diag and args.auto) else None
    ptable = None
    if args.preflop_table:
        import preflop_table as pt
        ptable = pt.load(args.preflop_table)
        print(f"[翻前查表] 已加载 {args.preflop_table}  深度桶={sorted(int(d) for d in ptable)}")
    flopturn_fn = None
    if args.flopturn_db:                                     # 优先:per-node SQLite 库(单节点查询,几乎零常驻内存)
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data"))
        import gto_lib_db as _gdb
        _db = _gdb.LibDB(args.flopturn_db)
        _ftrng = random.Random(args.seed + 555)
        flopturn_fn = lambda env: _gdb.flopturn_bucket_db(_db, env, 1, args.greedy, _ftrng)
        print(f"[翻转查库·DB] 已打开 {args.flopturn_db}  档(depth_bb)={_db.band_names()}  节点 {_db.count()}")
    elif args.flopturn_lib:                                  # 旧:整板内存库(大库会 OOM/慢,建议改用 --flopturn-db)
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_data"))
        import board_match as _bm, gto_opponent as _go
        import collections as _c
        _ftidx = _bm.build_multi_index(args.flopturn_lib)
        _ftcache = _c.OrderedDict()                           # LRU:防满库整块缓存堆积 OOM
        _ftrng = random.Random(args.seed + 555)
        flopturn_fn = lambda env: _go.flopturn_bucket(_ftidx, env, 1, args.greedy, _ftrng, _ftcache)
        _bands = getattr(_ftidx, "summary", lambda: {})()
        print(f"[翻转查库] 已加载 {args.flopturn_lib}  深度带={_bands}"
              f"  ⚠大库建议改用 --flopturn-db(单节点查询,不整板进内存)")
    play(net, device, game_cfg, river_cfg, args.greedy, args.seed, hands,
         seat0_pol=seat0_pol, verbose=verbose, opp_name=opp_name, diag=diag,
         log_path=(args.log if args.auto else None), big_frac=args.big, ptable=ptable,
         pf_guard=args.preflop_guard, flopturn_fn=flopturn_fn,
         river_nut_guard=args.river_nut_guard, river_nut_thresh=args.river_nut_thresh,
         commit_guard=args.commit_guard, commit_min_ratio=args.commit_min_ratio,
         air_fold_guard=args.air_fold_guard, air_fold_min_bet=args.air_fold_min_bet,
         exploit=args.exploit)


if __name__ == "__main__":
    main()
