"""threemax —— 3 人桌启发式引擎(独立模块,零改动复用现有 HU 模型)。

模块:
  table3.py         3 人翻前下注引擎 → 存活/投入/底池/死钱/坍缩判定
  hu_bridge.py      坍缩桥:2人局面 → HU 决策快照(死钱并入底池 + 位置映射),交现有模型
  preflop_policy.py 【占位桩】3-max 翻前策略(下一步接 charts)
  multiway.py       【占位桩】多路翻后兜底(下一步接多路启发式)
  selftest.py       自测 + 可选真实模型接通
"""
