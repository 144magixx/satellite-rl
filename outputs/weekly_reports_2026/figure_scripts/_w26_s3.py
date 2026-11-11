# -*- coding: utf-8 -*-
"""第三阶段（s3lab_20261029）原始记录读取。主矩阵 runs/main/rows.jsonl，取舍曲线 runs/tradeoff/rows.jsonl。

每行是一份第二阶段方案（输入）× 一种功率方法 × 一个求解器种子的结果。
  W0/worst10_0/F0  细调前的需求满足率/最差10%飞机/供需匹配程度
  a_W/a_worst10/a_SGM  “保SGM接入”之后的结果（SGM不低于细调前）
  tier  random随机 / fixed固定规则 / greedy容量感知贪心 / ptr块指针头T-SAC
  set   old旧确认集287景 / new新确认集300景
"""
import json
import os
import pandas as pd
from _w2526_common import EXP, S3

COLS = ["set", "plan", "tier", "idx", "key", "stratum", "method", "seed", "W0", "F0", "worst10_0",
        "a_W", "a_SGM", "a_worst10", "a_power_w", "t_native", "t_avail", "delta", "snaps", "status"]


def load(sub="main"):
    path = os.path.join(EXP, S3, "runs", sub, "rows.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["method"] == "bounds":
                continue
            rows.append({k: d.get(k) for k in COLS})
    df = pd.DataFrame(rows)
    df["dW"] = df["a_W"] - df["W0"]
    return df


METHODS = [  # 方法名（原始记录里的名字）, 中文名
    ("sca", "SCA（本研究）"),
    ("aravanis", "两段式"),
    ("wmmse", "WMMSE"),
    ("slsqp", "SLSQP"),
    ("target_cap", "目标容量迭代"),
    ("choi_chan", "注水"),
    ("proportional", "按需求比例"),
    ("maxmin", "最大最小公平"),
    ("equal", "等功率"),
]
TIERS = [("ptr", "块指针头\nT-SAC"), ("greedy", "容量感知\n贪心"), ("fixed", "固定\n规则"), ("random", "随机\n分配")]
