"""全局配置与设备选择（Mac Apple Silicon 优先 MPS，其次 CUDA，最后 CPU）。"""
from dataclasses import dataclass, field


def pick_device():
    """自动选择计算设备：MPS(Apple) > CUDA(N卡) > CPU。"""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ===== 动作离散化（数据驱动：改这里的档位，NB / 索引 / 标签会自动跟着变）=====
# 结构：0=fold，1=check/call，2..(2+n-1)=按底池比例加注，最后一个=all-in
# 下注比例 = 作用于“跟注后的底池”的倍数（0.25 = 1/4 池，1.25 = 1.25 池）
_RAISE_LIST = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

FOLD = 0
CALL = 1
RAISE_FRACTIONS = {2 + i: f for i, f in enumerate(_RAISE_LIST)}
ALLIN = 2 + len(_RAISE_LIST)
NB = ALLIN + 1          # = 2(fold/call) + 加注档数 + 1(all-in)


def action_name(a: int) -> str:
    """动作编号 → 可读标签（evaluate.py / agent.py 共用）。"""
    if a == FOLD:
        return "fold(弃牌)"
    if a == CALL:
        return "check/call(过牌或跟注)"
    if a == ALLIN:
        return "all-in(全下)"
    return f"raise {round(RAISE_FRACTIONS[a] * 100)}% pot"


ACTION_NAMES = {a: action_name(a) for a in range(NB)}

# ===== 动作历史编码维度（单街步数，这是唯一开关）=====
# 每条街最多记录的动作步数。同一条街的 3bet/4bet/5bet 都各占一步。
# 想支持更长的加注战就把 MAX_STEP 调大（如 8）；编码器/网络会自动跟随。
# ⚠️ 改完必须【重新训练 + 重新导出 ONNX】，并把 Java 端 StateEncoder.STEPS_PER_STREET 同步。
MAX_STEP = 8
ACTION_CHANNELS = 4 * MAX_STEP   # 动作张量通道数 = 4 街 × 每街步数

# ===== 【论文之上的实战扩展】可变筹码 + 有效筹码/SPR 标量特征 =====
# 论文是固定 200bb 深筹；实战对手筹码深度不定。这里让训练随机化起始筹码，
# 并把"有效筹码/SPR"作为额外标量喂进网络融合层，使模型能按筹码深度自适应。
# 仍是两人零和，不影响论文的自我对弈收敛逻辑。
STACK_NORM = 200.0        # 筹码归一化基准(bb)
SPR_NORM = 20.0           # SPR 归一化基准(截断上限)
NUM_SCALAR = 4            # 标量特征数：[我剩余, 对手剩余, 底池, SPR]（均已归一化）

# 标量里的筹码/底池特征用【对数尺度】还是【线性尺度】：
#   False(默认，兼容旧模型) = 线性 x/norm。浅筹被压到 0 附近(20/500=0.04)，分辨率差。
#   True                    = log1p(x)/log1p(norm)。把浅筹展开(20bb→~0.49)，短/中筹分辨率大增。
# ⚠️ 这是【特征语义】改动：改了必须【重新训练】，且 Python 全部工具 + Java StateEncoder
#    必须用【同一设置】，否则推理特征错位。不能和用另一设置训练的旧模型混用。
LOG_STACK_FEATURES = False


@dataclass
class GameConfig:
    """单挑无限注德扑（HUNL）规则，单位=大盲(bb)。"""
    start_stack: float = 200.0   # 固定筹码模式下的每人起始筹码(bb)，论文 200bb 深筹
    small_blind: float = 0.5
    big_blind: float = 1.0
    # 可变筹码（实战扩展）：开启后每手为两人各随机一个起始筹码，覆盖不同深度/不对称
    random_stacks: bool = True
    min_stack: float = 20.0
    max_stack: float = 200.0
    # 特征归一化基准：0=自动用 max_stack(训练默认)。评估异于训练范围的深度时，
    # 显式设为【训练时的筹码上限】，使 stack_norm 与训练一致，避免特征错位。
    stack_norm: float = 0.0
    # ===== 【Stage 1：低方差价值目标】all-in 精确/MC 权益结算 =====
    # True(默认)：两家全下且还有公共牌没发时，不再发【一次】随机 runout 决胜负，
    #   而是【枚举/采样所有剩余公共牌】算期望权益，用 matched·(2·equity−1) 作为期望收益。
    #   → 消掉 runout 方差(HUNL 方差最大来源)，让价值目标可学。fold/正常摊牌分支不受影响。
    # False：回退到旧的"发一次 runout"行为(用于对照/复现旧模型)。
    allin_ev: bool = True
    # 翻前全下(需补 5 张、组合数≈170万无法精确枚举)时的 MC 采样次数；翻牌/转牌走精确枚举，不受此值影响。
    allin_ev_mc_samples: int = 500


def make_game_cfg(chips, norm=0.0):
    """把 '20,200' 解析成 GameConfig(每手随机筹码)；None/空 → 默认 20,200。
    norm: 归一化基准，0=自动用 max_stack；评估时应设为【训练筹码上限】以对齐训练口径。
    返回 (GameConfig, lo, hi)。lo==hi 即固定筹码。"""
    lo, hi = 20.0, 200.0
    if chips:
        try:
            parts = str(chips).replace(" ", "").split(",")
            lo, hi = float(parts[0]), float(parts[1])
            if lo > hi:
                lo, hi = hi, lo
        except Exception:
            lo, hi = 20.0, 200.0
    return GameConfig(random_stacks=True, min_stack=lo, max_stack=hi,
                      stack_norm=float(norm or 0.0)), lo, hi


@dataclass
class TrainConfig:
    # 网络
    num_actions: int = NB
    # 奖励缩放：把回报除以此值，使回报落在 [-1,1]、价值损失保持 O(0.1)。
    # 【train.py 会自动把它设成 --chips 上限】，无需手改；此默认值仅在不经 train.py 时用作回退。
    reward_scale: float = 500.0
    # 回报归一化方式：
    #   "global"   —— 除以固定 reward_scale(=筹码上限)。深筹每手回报量级远大于浅筹，
    #                 梯度被深筹主导 → 深筹强、浅筹弱(默认，复现论文口径)。
    #   "effstack" —— 除以【本手有效筹码=两人起始筹码较小者】。让各深度每手回报都落在
    #                 ±1 附近、梯度分量相当 → 各深度均衡学习，能把短/中筹也练强。
    reward_norm: str = "global"
    # PPO
    gamma: float = 1.0            # 扑克按“每手”结算，单步无折扣
    gae_lambda: float = 0.95
    clip_eps: float = 0.2         # 标准 PPO 裁剪 ε
    clip_delta1: float = 3.0      # Trinal-Clip：负优势时比率上限 δ1
    value_coef: float = 0.5
    entropy_coef: float = 0.016  # 熵正则力度（在甜区附近微调）
    lr: float = 2e-4             # actor 学习率（1e-4 学得太温和/陷被动，提到 2e-4 更有力）
    # 特权评论家(Stage 2)单独用【更高】学习率：诊断显示 lr=2e-4 时 critic 学得太慢、
    # 每轮才 4 个 epoch 跟不上 → 留出 EV 停在 0。提到 1e-3 让 critic 快速追上策略。
    critic_lr: float = 1e-3
    # 特权评论家【回放缓冲】(治"每轮小缓冲过拟合":内EV高/外EV≈0)。
    #   critic_replay_iters：保留最近多少轮的决策一起训 critic(更大更多样→泛化);
    #   critic_epochs：每轮在回放集上训 critic 的遍数。
    critic_replay_iters: int = 8
    critic_epochs: int = 2
    # 【BC 锚定正则】从 bc_actor 微调时,给损失加一项 KL(π_当前 ‖ π_BC锚定),把策略拴在解算器克隆附近,
    # 防止 RL 把 BC 的纪律腐蚀成乱侵略。0=关;从 BC 微调建议 0.3~1.0(越大越贴 BC、越不敢偏离)。
    bc_anchor_coef: float = 0.0
    # 【分街道锚定】翻前/翻后用不同锚定力度:诊断显示腐蚀主要是"翻前大注开太勤",而翻前恰是 BC 的弱点。
    # 理想是【翻后拴紧(保住纪律)、翻前放松(留出空间补强)】。None=沿用全局 bc_anchor_coef(不分街道)。
    # 典型:翻后 0.8、翻前 0.2 → RL 能重学翻前、又不敢把翻后打乱。
    bc_anchor_coef_pre: float = None
    bc_anchor_coef_post: float = None
    # 【奖励塑形:把护栏"学进权重"】训练时,凡是护栏会否决的动作(深筹翻前非强牌大额投入 / 翻后空气大额投入),
    # 在 PPO 目标里给它减去 shape_coef 个"优势单位",把这类动作从策略里压下去。0=关;建议 1.0~3.0。
    # 目的:让"别拿空气/垃圾投一大截身家"逐步长进权重,最终少依赖甚至脱离推理期护栏。
    shape_coef: float = 0.0
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 512
    # 采样
    hands_per_iter: int = 2048    # 每轮自我对弈手数
    iterations: int = 200         # 训练总轮数
    # K-Best 对手池
    pool_size: int = 30           # 池中保留的历史模型数
    snapshot_every: int = 5       # 每多少轮把当前模型存入池
    opp_latest_prob: float = 0.2  # 采样对手时“用最新自己”的概率，其余从池里抽（0.2 更抗打转）
    # 混入"原型对手"(规则策略:跟注站/紧凶/疯子/随机)的概率，增加对手多样性、修补自我对弈盲区。
    # 0=纯自我对弈(默认)；0.2~0.3 可教模型对大侵略弃牌、别乱诈唬。太高会偏离均衡(去剥削规则bot)。
    opp_archetype_prob: float = 0.0
    # 其它
    eval_every: int = 10
    eval_hands: int = 2000
    ckpt_dir: str = "checkpoints"
    seed: int = 42
