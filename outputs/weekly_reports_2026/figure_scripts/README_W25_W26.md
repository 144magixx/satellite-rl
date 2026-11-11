# W25、W26 画图脚本说明

运行：在仓库根目录 `python outputs\weekly_reports_2026\figure_scripts\<脚本名>.py`，图写到 `outputs/weekly_reports_2026/figures/`。
原始数据只读，在 `D:\paper\outputs\experiments\`。公共函数在 `_w2526_common.py`（字体、读逐景评价、保存真 PNG）和 `_w26_s3.py`（第三阶段记录）。

## 有原始记录、从 csv/jsonl 重画的图

| 脚本 | 数据 |
| --- | --- |
| W25_整条链对照 | litcmp_20261018/analysis/stage23_table.csv（d432 档） |
| W25_第一轮主结果、第一轮训练曲线、第一轮逐场景对比 | s2lab_20261024/eval/confirm/*.csv、eval/dev/rule-greedy.csv、runs/*/dev.jsonl |
| W26_块指针对比 | s2lab_round4 的 r4-ptr 三个种子 + 各规则方法 csv（数字与原图一致，只是把 SGM、W 改成中文名；原图备份为 W26_块指针对比_原图备份.png） |
| W26_四轮最好组对比、第2轮训练曲线、第3轮消融、第3轮第4轮训练曲线、第4轮各组对比 | s2lab_20261024、s2lab_round2/3/4 的 eval_confirm_best_sgm.csv、dev.jsonl、eval_c2/confirm |
| W26_大波束分到的资源 | s2lab_round4_20261028/beamdiag/b{1,2,3}_beams.csv |
| W26_第三阶段九种功率方法、细调前后 | s3lab_20261029/runs/main/rows.jsonl |

## 缺逐景原始记录、按报告表格数字重画的图（原图本机只剩 png，没有 results.jsonl）

实验目录 `s2_orig_compare_20261022` 在本机只有报告引用的 png，没有 `main/results.jsonl`、`diagnostics.jsonl`、`cross/*.jsonl`。
下面三张图的数字取自报告 `docs/Claude维护/原版框架复现/20_原版框架上八种方法不重训对比报告.md` 的表格，脚本里写明了取自哪一节：

- W25_原版环境八种方法：第2节“结果”表（满足率、最差10%波束、单景耗时），没有误差线。
- W25_原版环境诊断：第3节“事后诊断”表；“T-SAC直接输出”一行取报告里诊断图上的数（96.0%、60.0%）。
- W25_分散摆放规则新环境：第4节“顺带的发现”表。

报告里原来的现成图（f1 到 f6）图例带实验代号（I0），没有直接用。原版实验的逐景散点图（f2）没法重画，没有放。
