"""河牌按需现解:给定河牌局面 → 调 TexasSolver 解河牌子博弈 → 返回你这手牌的 GTO 动作档(bucket)。
供 slumbot_client.py 的 --river-solve 用(对比测试:模型出河牌 vs solver 出河牌)。

【硬近似,务必知悉】
  · 河牌 GTO 需要双方【到达河牌的范围】。我们不知道 Slumbot 的策略 → 只能【假设】范围(默认一套 SRP
    宽范围)。所以这是"假设双方按标准范围打到河"的 GTO 参照,不是针对 Slumbot 真实范围的精确解。
  · 用【翻前 SRP 范围】近似河牌到达范围,忽略了翻/转的收窄 → 偏宽。粗糙但可用于 A/B 对比。
  · 任何失败(解算出错/超时/牌不在范围/导航失败)→ 返回 None,由调用方用模型动作兜底,不影响对局继续。

依赖 gto_feedback 的树导航;config 的档位;console_solver 二进制。
"""
import json
import os
import subprocess
import tempfile

from config import RAISE_FRACTIONS, FOLD, CALL, ALLIN

_FR = sorted(RAISE_FRACTIONS.items(), key=lambda kv: kv[1])


def find_children(node):
    for k in ("childrens", "dealcards", "deal_cards", "children"):
        if isinstance(node.get(k), dict):
            return node[k]
    return {}


def node_strategy(node):
    strat = node.get("strategy", {}) or {}
    return strat.get("actions") or node.get("actions") or [], strat.get("strategy", {})


def parse_label(label):
    parts = label.strip().split()
    return parts[0].upper(), (float(parts[1]) if len(parts) > 1 else 0.0)

# 默认到达范围(SRP 宽范围;可在 slumbot_client 用 --river-range 覆盖)
# 【第二层·对手范围真实度】默认改用重档 66% 范围(与 solver_gen 灌库同一套,已校验),对手建模更贴近
# 河牌实际到达范围;配合 river_action 里的 hero 手牌注入(第一层),= 对手宽范围 + hero 必被覆盖。
# 旧窄档 40% 留作 _NARROW(需回退/图快时可 --river-range 传它;窄=解得快但对手假设失真)。
DEFAULT_RANGE = ("AA,KK,QQ,99,JJ,TT,77,88,55,66,33,44,22,AKs,AQs,AJs,AKo,ATs,KQs,QJs,AQo,A9s,JTs,KJs,A8s,"
                 "AJo,QTs,KTs,A7s,KQo,A6s,J9s,Q9s,T9s,ATo,K9s,QJo,KJo,98s,A5s,A9o,K8s,J8s,Q8s,T8s,A4s,JTo,"
                 "A8o,QTo,K7s,KTo,87s,97s,Q7s,A3s,T7s,A7o,J7s,K6s,T9o,J9o,Q6s,A2s,Q9o,A6o,76s,K5s,J6s,K9o,"
                 "86s,96s,Q5s,A5o,T6s,K4s,98o,K8o,J5s,T8o,65s,J8o,Q4s,Q8o,A4o,T5s,75s,85s,K3s,K7o,95s,J4s,"
                 "Q3s,87o,Q7o,T4s,97o,A3o,K2s,T7o,K6o,54s,J3s,J7o,64s,94s,74s,Q2s,Q6o,T3s,84s,A2o,K5o,76o,"
                 "J2s,J6o,86o,93s,96o,Q5o,43s,T2s,83s,T6o,53s,K4o")
_DEFAULT_RANGE_NARROW = ("22,33,44,55,66,77,88,99,TT,JJ,QQ,KK,AA,A2s,A3s,A4s,A5s,A6s,A7s,A8s,A9s,ATs,AJs,AQs,AKs,"
                 "K2s,K3s,K4s,K5s,K6s,K7s,K8s,K9s,KTs,KJs,KQs,Q6s,Q7s,Q8s,Q9s,QTs,QJs,J7s,J8s,J9s,JTs,"
                 "T7s,T8s,T9s,97s,98s,86s,87s,76s,65s,54s,A2o,A3o,A4o,A5o,A6o,A7o,A8o,A9o,ATo,AJo,AQo,AKo,"
                 "K9o,KTo,KJo,KQo,Q9o,QTo,QJo,J9o,JTo,T9o,98o")


def _frac_to_bucket(frac):
    if frac >= 3.5:
        return ALLIN
    return min(_FR, key=lambda kv: abs(kv[1] - frac))[0]


def _label_to_bucket(label, pot_before, to_call, my_contrib):
    kind, amt = parse_label(label)
    if kind in ("CHECK", "CALL"):
        return CALL
    if kind == "FOLD":
        return FOLD
    if kind == "ALLIN":
        return ALLIN
    if kind == "BET":
        return _frac_to_bucket(amt / max(pot_before, 1e-9))
    if kind == "RAISE":
        return _frac_to_bucket(max(amt - my_contrib, 0) / max(pot_before + max(to_call, 0), 1e-9))
    return CALL


def _kind(label):
    return parse_label(label)[0]


def _bucket_kind(bucket):
    if bucket == FOLD:
        return "FOLD"
    if bucket == CALL:
        return "CALL"      # check/call 同类
    if bucket == ALLIN:
        return "ALLIN"
    return "BET"           # 加注/下注档统称 BET(下注或加注)


def _child_by_kind(children, kind):
    """按动作【类型】选子节点(下注史里的精确尺寸不一定在树里,按类型近似匹配)。
    kind: CALL(含CHECK) / FOLD / BET(含RAISE) / ALLIN。返回 (label, child) 或 None。"""
    # 先找完全同类;BET 类优先非全下的下注/加注,再退全下
    cand = []
    for lab, ch in children.items():
        k = _kind(lab)
        if kind == "CALL" and k in ("CHECK", "CALL"):
            return lab, ch
        if kind == "FOLD" and k == "FOLD":
            return lab, ch
        if kind == "ALLIN" and k in ("ALLIN",):
            return lab, ch
        if kind == "BET" and k in ("BET", "RAISE"):
            cand.append((lab, ch))
    if kind == "BET" and cand:
        return cand[0]
    if kind == "ALLIN":         # 树里全下可能写成 BET/RAISE 全额
        big = None
        for lab, ch in children.items():
            if _kind(lab) in ("BET", "RAISE"):
                amt = parse_label(lab)[1]
                if big is None or amt > parse_label(big[0])[1]:
                    big = (lab, ch)
        return big
    return None


def _build_input(out_json, board5, pot_bb, eff_bb, ip_range, oop_range, threads, iters, accuracy):
    L = [f"set_pot {pot_bb:.2f}", f"set_effective_stack {eff_bb:.2f}",
         f"set_board {board5}", f"set_range_ip {ip_range}", f"set_range_oop {oop_range}"]
    for pos in ("oop", "ip"):
        L += [f"set_bet_sizes {pos},river,bet,33,75",
              f"set_bet_sizes {pos},river,raise,50",
              f"set_bet_sizes {pos},river,allin"]
    L += ["set_allin_threshold 0.67", "build_tree", f"set_thread_num {threads}",
          f"set_accuracy {accuracy}", f"set_max_iteration {iters}", "set_print_interval 50",
          "set_use_isomorphism 0", "start_solve", "set_dump_rounds 1", f"dump_result {out_json}"]
    return "\n".join(L) + "\n"


# 最近一次 river_action 的结果原因(诊断用):调用方可读 river_solver.LAST_REASON。
# 取值:ok / solver_no_output / solver_exc:<msg> / nav_no_children / nav_no_match /
#       pos_mismatch / node_no_actions / hole_not_in_range:<key样例> / exc:<msg>
LAST_REASON = None


_RVAL = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def _hole_class(hole):
    """['Qc','Js'] → 规范式类别 'QJo' / 'QJs' / 'QQ'(高牌在前)。用于判断 hero 手是否已在范围内。"""
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if _RVAL.get(r1, 0) >= _RVAL.get(r2, 0) else (r2, r1)
    return f"{hi}{lo}" + ("s" if s1 == s2 else "o")


def river_action(board5, pot_bb, eff_bb, river_hist, hero_is_oop, hole, greedy, cfg):
    """核心:河牌现解 → 返回你这手牌的动作 bucket(失败返回 None)。
    board5: ['Qs','Jh','2h','Td','8d'](5张字符串);pot_bb/eff_bb: 河牌起始底池/有效筹码(bb)。
    river_hist: 本街已发生动作 [(actor_is_hero(bool), bucket)],按先后;空=你先手。
    hero_is_oop: 你是否 OOP(先手)。hole: ['Ah','Ks']。cfg: dict(solver_bin,resource_dir,run_prefix,
       ip_range,oop_range,threads,iters,accuracy,timeout)。
    失败原因写入 river_solver.LAST_REASON(诊断用)。"""
    global LAST_REASON
    try:
        board_str = ",".join(board5)
        tmpd = tempfile.mkdtemp(prefix="riversolve_")
        out_json = os.path.join(tmpd, "r.json")
        inp = os.path.join(tmpd, "in.txt")
        # 【范围优化】把 hero 的实际底牌强制并入其所在位置的范围,保证 solved 策略表里一定有这手,
        # 消除头号未命中 hole_not_in_range(静态范围常不覆盖 hero 经具体线到达河牌的实际手牌)。+1组合,几乎不增成本。
        ip_rng, oop_rng = cfg["ip_range"], cfg["oop_range"]
        if hole and len(hole) == 2:
            _cls = _hole_class(hole)                       # 类别记法 65o/AKs/66
            _hero_rng = oop_rng if hero_is_oop else ip_rng
            _toks = set(t.strip() for t in _hero_rng.split(","))
            # 用【类别】注入,不用具体组合:TexasSolver 范围只认 AA/AKs/AKo 类记法,
            # 具体组合 '6s5h' 会被判 "range str len not valid" 而崩(exit-6)。类不在范围才补,避免与已有类重叠。
            if _cls not in _toks:
                if hero_is_oop:
                    oop_rng = f"{oop_rng},{_cls}"
                else:
                    ip_rng = f"{ip_rng},{_cls}"
        with open(inp, "w") as f:
            f.write(_build_input(out_json, board_str, pot_bb, eff_bb,
                                 ip_rng, oop_rng,
                                 cfg["threads"], cfg["iters"], cfg["accuracy"]))
        cmd = (cfg.get("run_prefix", []) +
               [cfg["solver_bin"], "--input_file", inp, "--resource_dir", cfg["resource_dir"], "--mode", "holdem"])
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=cfg.get("timeout", 120), check=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", "ignore").strip().replace("\n", " ")[:160]
            LAST_REASON = f"solver_exit{e.returncode}:{err or '(no stderr)'}"
            return None
        except FileNotFoundError:
            LAST_REASON = f"solver_missing:{cmd[0]} / {cfg.get('solver_bin')}"
            return None
        if not os.path.exists(out_json) or os.path.getsize(out_json) == 0:
            LAST_REASON = "solver_no_output"
            return None
        with open(out_json) as f:
            tree = json.load(f)

        # 导航:河牌树 root = OOP 先手(actor 0)。逐步下降,同时【跟踪底池/双方投入】,交替行动者。
        node = tree
        pot = pot_bb
        contrib = [0.0, 0.0]        # [oop, ip] 本街投入
        actor = 0                   # 0=OOP 先手,每动一步交替
        for _actor_is_hero, bucket in river_hist:
            children = find_children(node)
            if not children:
                LAST_REASON = "nav_no_children"
                return None
            pick = _child_by_kind(children, _bucket_kind(bucket))
            if not pick:
                LAST_REASON = "nav_no_match"
                return None
            label, node = pick
            kind, amt = parse_label(label)
            opp = 1 - actor
            if kind == "CALL":
                add = max(contrib[opp] - contrib[actor], 0.0); contrib[actor] += add; pot += add
            elif kind == "BET":
                contrib[actor] += amt; pot += amt
            elif kind == "RAISE":
                add = max(amt - contrib[actor], 0.0); contrib[actor] = amt; pot += add
            # CHECK / FOLD / ALLIN: 不改(ALLIN 罕见,近似)
            actor = opp
        # 一致性校验:到达的决策节点应轮到 hero(位置对不上=史/树不一致 → 回退)
        if actor != (0 if hero_is_oop else 1):
            LAST_REASON = "pos_mismatch"
            return None
        # 到达你的决策节点
        actions, combo_strat = node_strategy(node)
        if not actions:
            LAST_REASON = "node_no_actions"
            return None
        probs = combo_strat.get(hole[0] + hole[1]) or combo_strat.get(hole[1] + hole[0])
        if probs is None:
            sample = list(combo_strat.keys())[:4]
            LAST_REASON = f"hole_not_in_range:{hole[0]+hole[1]} vs键样例{sample}"
            return None
        if greedy:
            idx = max(range(len(actions)), key=lambda i: probs[i])
        else:
            import random as _r
            x = _r.random(); acc = 0.0; idx = len(actions) - 1
            for i, p in enumerate(probs):
                acc += p
                if x <= acc:
                    idx = i; break
        # 用【真实的当前底池/待跟额/我方已投入】把选中的标签映射回 bucket
        to_call = max(contrib[1 - actor] - contrib[actor], 0.0)
        LAST_REASON = "ok"
        return _label_to_bucket(actions[idx], pot, to_call, contrib[actor])
    except Exception as e:
        LAST_REASON = f"exc:{type(e).__name__}:{str(e)[:140]}"
        return None
