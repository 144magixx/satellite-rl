# -*- coding: utf-8 -*-
"""W12_训练过程_聚类加PPO微调.png：报告现成图改文字并裁出两个小图，不是重画。
原图是 first_stage_s1_01_20260930/curves/C2_training_diagnostics.png（聚类基础布局＋PPO微调的训练过程），
改动见 _w12_train_patch.py 的说明。运行：python W12_训练过程_聚类加PPO微调.py
"""
from _w12_train_patch import patch_training

patch_training("C2", "聚类基础布局＋PPO微调：验证场景上的覆盖和负载随训练的变化", "W12_训练过程_聚类加PPO微调.png")
