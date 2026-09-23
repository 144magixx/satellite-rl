# Git 仓库初始化

- 用户请求：建一个 git 仓库。
- 回答日期：2026-09-24。

## 结论与范围

在 `D:/paper` 初始化本地 Git 仓库，主分支为 `main`。首次版本记录纳入当前源码、测试、项目配置、文档、问答、输入数据及原预训练权重。

沿用根目录 `.gitignore`：忽略 Python 缓存、虚拟环境、IDE 配置、安装构建产物、`outputs/` 与 `models/runs/`。这些文件仍保留在本地；Git 历史不包含被忽略的训练日志、评估结果及新训练权重，实验产物仍需单独保存。

初始化未配置远程仓库，后续如需推送须指定远程地址。创建 Git 仓库不涉及算法、环境或学习行为变化，按 `satellite-rl-change-eval` 的轻量分支检查版本管理范围、忽略规则、文档引用与仓库状态，无需启动训练。

## 依据与复查

- 分支和状态：`git branch --show-current`、`git status --short`。
- 版本记录：`git log -1 --oneline`。
- 文件范围：`git ls-files`。
- 忽略规则：`git check-ignore -v outputs/ models/runs/ .idea/`。
- 远程配置：`git remote -v`。

此前问答中“没有 Git 元数据”的表述描述当时状态；自本次初始化后，后续改动可使用 Git 提交记录作为源码参照。
