"""翻前 GTO 查表 —— 把 preflop_solve.py 的解压成一张近GTO策略表,推理时【翻前直接查表出招】。

为什么:翻前状态空间极小(169手 × 位置 × 深度 × 加注轮次),可以完全枚举查表,
不必让神经网络去"学"。这直接治好"翻前太被动/PFR过低"的头号漏洞,且确定、可审计。

表结构(按【深度桶 × 位置 × 面对到额(facing size) × 169手】聚合):
  table[depth_bb][hero("SB"/"BB")][facing_to]["AKs"] = {"s":{fold,call,raise,allin 频率}, "rt":加注到额bb}
  facing_to = hero 面对的"到额"(bb)=seq 里最后一个加注的到额;seq空(SB开池)=1.0(大盲)。
    SB@1.0=开池  BB@~2.5=vs开池  SB@~11=vs3bet  BB@~25=vs4bet …(按尺寸分键,不同 open size 不混平均)

用法:
  # 1) 先用 preflop_solve 出解(在你训练机上;各深度对齐你的4档:见下方命令)
  # 2) 压成表:
  python3 preflop_table.py build --jsonl preflop_solve.jsonl --out preflop_table.json
  # 3) 推理:play_vs_bot / slumbot_client 加 --preflop-table preflop_table.json
  python3 preflop_table.py selftest      # 合成数据自检(不需真解)
"""
import argparse
import json
import random

RANKC = "23456789TJQKA"


def hand169(c0, c1):
    """两张 int 牌(card=rank*4+suit) → 169 规范标签,如 'AA'/'AKs'/'T9o'(与 preflop_solve 一致)。"""
    r0, r1 = c0 // 4, c1 // 4
    s0, s1 = c0 % 4, c1 % 4
    hi, lo = max(r0, r1), min(r0, r1)
    if hi == lo:
        return RANKC[hi] * 2
    return RANKC[hi] + RANKC[lo] + ("s" if s0 == s1 else "o")


def _parse_strategy(strat):
    """solve 的 strategy dict → ({fold,call,allin,raise 概率}, 加注平均到额bb 或 None)。"""
    probs = {"fold": 0.0, "call": 0.0, "allin": 0.0, "raise": 0.0}
    rt_num = 0.0
    for k, p in strat.items():
        p = float(p)
        if k in ("fold", "call", "allin"):
            probs[k] += p
        elif k.startswith("raise:"):
            probs["raise"] += p
            try:
                rt_num += p * float(k.split(":", 1)[1])
            except ValueError:
                pass
    rt = (rt_num / probs["raise"]) if probs["raise"] > 1e-9 else None
    return probs, rt


def _facing_to(seq):
    """hero 面对的"到额"(bb):seq 里最后一个 raise/allin 的到额;seq 空(SB 开池)= 1.0(大盲)。"""
    ft = 1.0
    for a in seq:
        if a[1] in ("raise", "allin"):
            ft = float(a[2])
    return round(ft * 2) / 2                           # 就近到 0.5bb,合并近似尺寸、减少键


def build(jsonl, out, force=False):
    acc = {}                                          # (depth,hero,facing_to,hand) -> [(probs, rt, w)]
    for_check = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            depth = int(round(float(r["depth_bb"])))
            hero = r["hero"]
            hand = r["hand"]
            ft = _facing_to(r["seq"])
            probs, rt = _parse_strategy(r["strategy"])
            w = float(r.get("w", 1.0))                # 到达权重(旧jsonl无此字段→退回等权1.0)
            acc.setdefault((depth, hero, ft, hand), []).append((probs, rt, w))
    table = {}
    for (depth, hero, ft, hand), lst in acc.items():
        agg = {"fold": 0.0, "call": 0.0, "allin": 0.0, "raise": 0.0}
        rt_num = rt_w = wtot = 0.0
        for probs, rt, w in lst:                      # 按到达权重加权平均:稀有/未收敛节点自动近乎不计
            for k in agg:
                agg[k] += w * probs[k]
            if rt is not None:
                rt_num += w * probs["raise"] * rt
                rt_w += w * probs["raise"]
            wtot += w
        if wtot <= 1e-12:                             # 全零权重(理论上不会)→ 退回等权
            wtot = len(lst)
            for probs, rt, _w in lst:
                for k in agg:
                    agg[k] += probs[k]
        for k in agg:
            agg[k] /= wtot
        s = sum(agg.values())
        if s > 1e-9:
            for k in agg:
                agg[k] /= s
        entry = {"s": {k: round(v, 4) for k, v in agg.items()}, "w": round(wtot, 4)}
        if rt_w > 1e-9:
            entry["rt"] = round(rt_num / rt_w, 2)
        (table.setdefault(str(depth), {}).setdefault(hero, {})
              .setdefault(f"{ft:.1f}", {})[hand]) = entry
    ok = _sanity_check(table)
    if not ok and not force:
        raise SystemExit(f"[build] 收敛自检未通过 → 拒绝写表 {out}。多半是解没收敛(迭代不够)或抽象有误,"
                         f"别用这张表。加大 --iters 重解,或看上面 [自检] 明细。加 --force 可强制写出(仅供实测验证)。")
    if not ok and force:
        print("[build] ⚠ 自检未过,但 --force 强制写出 —— 仅供对打实测,勿直接当成品用。")
    with open(out, "w") as f:
        json.dump(table, f, ensure_ascii=False)
    depths = sorted(int(d) for d in table)
    print(f"[build] 深度桶={depths}  按 位置×面对到额×手牌 分键(到达权重加权) → {out}")


def _cont(node):
    """续牌频率 = 1 - 弃牌频率(call+raise+allin 之和)。"""
    s = node["s"]
    tot = sum(s.values())
    return 0.0 if tot <= 1e-9 else 1.0 - s.get("fold", 0.0) / tot


def _agg(node):
    """归一化后的 (fold, call, raise, allin) 频率。"""
    s = node["s"]; t = sum(s.values())
    if t <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0
    return (s.get("fold", 0.0) / t, s.get("call", 0.0) / t,
            s.get("raise", 0.0) / t, s.get("allin", 0.0) / t)


def _sanity_check(table):
    """铁律锚点自检:抓未收敛/反转(垃圾牌激进、强牌被动)这类坏表。不过→返回False,build 拒绝写表。
    面对3bet+(facing≥5bb),按到达权重 w 加权,四条铁律必须同时成立:
      (a) AA 续牌率 显著高于 72o(≥0.20);
      (b) AA 价值加注(raise+allin) 不能被压没(≥0.08)—— 抓"强牌只会跟"的被动坍缩;
      (c) 72o 主动加注(raise+allin) 必须很低(≤0.18)—— 抓"垃圾牌反加/诈全下"的反转;
      (d) 72o 弃牌率 必须够高(≥0.45)—— 抓"垃圾牌乱跟 3bet"。
    另加:AA 开池/无面对 几乎不弃。旧表无 w → 退回等权(严格)。"""
    print("[自检] 铁律锚点检查(按到达权重;面对3bet+)...")
    den = 0.0
    sep_n = aar_n = tzr_n = tzf_n = badsep = 0.0
    aa_open = []
    for depth, htbl in table.items():
        for hero, ftbl in htbl.items():
            for ft, hands in ftbl.items():
                aa, trash = hands.get("AA"), hands.get("72o")
                if float(ft) <= 1.6 and aa is not None:
                    aa_open.append((aa["s"].get("fold", 0.0), aa.get("w", 1.0), depth, hero, ft))
                if float(ft) >= 5.0 and aa is not None and trash is not None:
                    w = aa.get("w", 1.0)
                    af, ac, ar, aj = _agg(aa)
                    tf, tc, tr, tj = _agg(trash)
                    sep = (1 - af) - (1 - tf)
                    den += w
                    sep_n += w * sep
                    aar_n += w * (ar + aj)            # AA 价值加注
                    tzr_n += w * (tr + tj)            # 72o 主动加注(反转信号)
                    tzf_n += w * tf                   # 72o 弃牌
                    if sep < 0.20:
                        badsep += w
    ok = True
    if den > 1e-12:
        wsep, waar, wtzr, wtzf = sep_n/den, aar_n/den, tzr_n/den, tzf_n/den
        print(f"   加权: AA续牌-72o续牌分离={wsep:+.2f}  AA价值加注={waar:.2f}  "
              f"72o主动加注={wtzr:.2f}  72o弃牌={wtzf:.2f}  低分离权重占比={badsep/den:.0%}")
        if wsep < 0.20:
            print("   ✗ (a) AA 与 72o 续牌不分——未收敛"); ok = False
        if waar < 0.08:
            print("   ✗ (b) AA 价值加注被压没(强牌只会跟)——反转/未收敛"); ok = False
        if wtzr > 0.18:
            print("   ✗ (c) 72o 主动加注过高(垃圾牌反加/诈全下)——范围反转"); ok = False
        if wtzf < 0.45:
            print("   ✗ (d) 72o 面对3bet 弃牌太少(乱跟)——未收敛"); ok = False
        if ok:
            print("   ✓ 四条铁律全过:强牌敢加、垃圾牌该弃、无范围反转")
    else:
        print("   (无 facing≥5 样本可查,跳过)")
    real = [x for x in aa_open if x[1] > 1e-9]
    if real:
        mxw = max(x[1] for x in real)
        worst = max(real, key=lambda x: x[0] * min(x[1], 1.0))
        wf, ww, dp, he, f = worst
        if wf > 0.10 and ww > 0.5 * mxw:
            print(f"   ✗ AA 开池弃{wf:.0%}(深度{dp} {he} facing{f} 权重{ww:.2f},应≈0)"); ok = False
        else:
            print(f"   ✓ AA 开池不弃(高频节点最高弃{max(x[0] for x in real if x[1] > 0.5*mxw):.0%})")
    return ok


def load(path):
    with open(path) as f:
        return json.load(f)


def _nearest_depth(table, eff_bb):
    return min((int(d) for d in table), key=lambda d: abs(d - eff_bb))


def lookup(table, eff_bb, hero, facing_to, hand, greedy=True, rng=None):
    """→ ('fold'|'call'|'allin', None) 或 ('raise', 到额bb);查不到返回 None。
    facing_to = hero 当前面对的到额(bb):开池处=1.0(大盲),否则=对手最近一次加注的到额。"""
    d = _nearest_depth(table, eff_bb)
    hero_tbl = table.get(str(d), {}).get(hero, {})
    if not hero_tbl:
        return None
    fk = min(hero_tbl, key=lambda k: abs(float(k) - facing_to))   # 面对尺寸就近匹配
    node = hero_tbl[fk].get(hand)
    if not node:
        return None
    s = node["s"]
    acts = ["fold", "call", "raise", "allin"]
    ps = [s.get(a, 0.0) for a in acts]
    if sum(ps) <= 1e-9:
        return None
    if greedy:
        act = acts[max(range(4), key=lambda i: ps[i])]
    else:
        rng = rng or random
        x = rng.random() * sum(ps)
        acc = 0.0
        act = acts[-1]
        for a, p in zip(acts, ps):
            acc += p
            if x <= acc:
                act = a
                break
    if act == "raise":
        return ("raise", node.get("rt"))
    return (act, None)


def _selftest():
    import os
    tmp = "/tmp/_pf_solve.jsonl"
    rows = [
        # 100bb SB 开池(nraise0):AA 全加注(raise到2.5), 72o 全弃
        {"depth_bb": 100, "hero": "SB", "seq": [], "hand": "AA", "strategy": {"raise:2.5": 1.0}},
        {"depth_bb": 100, "hero": "SB", "seq": [], "hand": "72o", "strategy": {"fold": 1.0}},
        # 100bb BB vs开池(nraise1):AA 3bet(raise到11), KK 混合
        {"depth_bb": 100, "hero": "BB", "seq": [["SB", "raise", 2.5]], "hand": "AA", "strategy": {"raise:11": 0.9, "call": 0.1}},
        {"depth_bb": 100, "hero": "BB", "seq": [["SB", "raise", 2.5]], "hand": "72o", "strategy": {"fold": 0.95, "call": 0.05}},
        # 100bb SB vs3bet(nraise2):AA 4bet全下
        {"depth_bb": 100, "hero": "SB", "seq": [["SB", "raise", 2.5], ["BB", "raise", 11]], "hand": "AA", "strategy": {"allin": 1.0}},
    ]
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    build(tmp, "/tmp/_pf_table.json")
    t = load("/tmp/_pf_table.json")
    # AA 开池(面对大盲1.0)→ raise 到 2.5
    assert lookup(t, 100, "SB", 1.0, "AA") == ("raise", 2.5), lookup(t, 100, "SB", 1.0, "AA")
    # 72o 开池 → fold
    assert lookup(t, 100, "SB", 1.0, "72o") == ("fold", None)
    # AA vs开池(面对 2.5)→ raise 到 11
    assert lookup(t, 100, "BB", 2.5, "AA") == ("raise", 11.0), lookup(t, 100, "BB", 2.5, "AA")
    # AA vs3bet(面对 11)→ allin
    assert lookup(t, 100, "SB", 11.0, "AA") == ("allin", None)
    # 面对尺寸就近:面对 2.6 → 匹配 2.5 键
    assert lookup(t, 100, "BB", 2.6, "AA")[0] == "raise"
    # 深度就近:eff=95 → 落到 100 桶
    assert lookup(t, 95, "SB", 1.0, "AA")[0] == "raise"
    # 不在表 → None
    assert lookup(t, 100, "SB", 1.0, "AKs") is None
    # hand169 编码
    assert hand169(12 * 4 + 0, 12 * 4 + 1) == "AA"
    assert hand169(12 * 4 + 0, 11 * 4 + 0) == "AKs"    # 同花
    assert hand169(12 * 4 + 0, 11 * 4 + 1) == "AKo"    # 非同花
    print("[selftest] OK —— build/lookup/深度就近/编码 全部正确")
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="preflop_solve.jsonl → 查表 json")
    b.add_argument("--jsonl", required=True)
    b.add_argument("--out", default="preflop_table.json")
    b.add_argument("--force", action="store_true", help="自检未过也强制写表(仅供对打实测)")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.jsonl, args.out, args.force)
    else:
        _selftest()


if __name__ == "__main__":
    main()
