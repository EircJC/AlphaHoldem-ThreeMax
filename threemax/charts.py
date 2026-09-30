"""3-max 翻前范围表(显式手牌类,非"对随机权益"阈值——后者会低估同花连张/小对)。

这些是**合理的近似标准 3-max 范围**(不是某求解器精确解、无混合频率),但结构=真 charts:
按 开池位 / 面对开池(3bet·跟) / 面对3bet(4bet·跟) 分。要更准就按公开 charts 调这些字符串。

用 expand() 展开简写:
  "22+"   → 22,33,...,AA
  "A2s+"  → A2s,A3s,...,AKs(定高牌 A,低牌递增到 K)
  "K9o+"  → K9o,KTo,KJo,KQo
  单牌     → 原样(如 "76s","54s")
"""
from __future__ import annotations

_R = "23456789TJQKA"                                          # 2..A,索引 0..12
_RIDX = {c: i for i, c in enumerate(_R)}


def _expand_token(tok):
    tok = tok.strip()
    if not tok:
        return []
    if tok.endswith("+"):
        base = tok[:-1]
        if len(base) == 2 and base[0] == base[1]:             # 对子 "NN+"
            i = _RIDX[base[0]]
            return [_R[j] + _R[j] for j in range(i, 13)]
        hi, lo, so = base[0], base[1], base[2]                # "XYs+"/"XYo+"
        hii, loi = _RIDX[hi], _RIDX[lo]
        return [hi + _R[j] + so for j in range(loi, hii)]     # 低牌从 lo 递增到 hi-1
    return [tok]


def expand(spec):
    """逗号/空格分隔的简写 → 手牌类集合。"""
    out = set()
    for part in spec.replace("\n", " ").replace(",", " ").split():
        for h in _expand_token(part):
            out.add(h)
    return out


# ---- 开池范围(未被加注时)。3-max 只 2 家在后,开池偏宽,尤其 BTN/SB 偷盲。----
OPEN = {
    "BTN": expand("22+ A2s+ K2s+ Q3s+ J5s+ T6s+ 95s+ 84s+ 74s+ 63s+ 53s+ 43s "
                  "A2o+ K7o+ Q8o+ J8o+ T8o+ 97o+ 87o 76o"),                  # ~58%
    "SB":  expand("22+ A2s+ K3s+ Q6s+ J7s+ T7s+ 96s+ 85s+ 75s+ 64s 54s "
                  "A4o+ K9o+ Q9o+ J9o+ T9o 98o"),                            # ~48%(SB 折到 BB,可宽偷)
    "BB":  expand("22+ A2s+ K6s+ Q8s+ J8s+ T8s+ 97s+ 86s+ 76s 65s "
                  "A8o+ KTo+ QTo+ JTo"),                                     # BB 领打(vs limp)用
}

# ---- 面对开池:3bet(价值+诈唬) / 跟注。盲位防守放宽:别再弃盲送钱。----
THREEBET = expand("99+ AJs+ KQs AQo+ A5s A4s A3s KJs QJs")                   # 价值 + 同花A/K 诈唬
CALL_VS_OPEN = {
    "BB":  expand("22+ A2s+ K2s+ Q5s+ J7s+ T7s+ 96s+ 86s+ 75s+ 65s 54s "
                  "A2o+ K8o+ Q9o+ J9o+ T9o 98o"),                           # BB 有价格+closing → 防守很宽(~50%)
    "OTHER": expand("22+ A8s+ K9s+ QTs+ J9s+ T9s 98s 87s 76s A9o+ KJo+ QJo"),  # SB 面对开池(OOP/挤压风险)中等宽
}
# ---- 面对 3bet:4bet / 跟 ----
FOURBET = expand("QQ+ AKs AKo A5s")                                          # 价值 + A5 诈唬
CALL_VS_3BET = expand("99+ AJs+ KQs AQo")


def in_range(hand_class, rng):
    return hand_class in rng
