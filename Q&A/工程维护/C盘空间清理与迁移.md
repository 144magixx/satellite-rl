# C盘空间清理与迁移

来源：cc

## 2026-10-24 问题

C 盘的 `paper_archives` 还用吗、能删吗？C 盘有什么能删、什么能迁到 D/E 盘？

## 结论

**`C:\paper_archives` 不占 C 盘空间，不要删。** 它是 10 月 14 日迁移时留下的目录链接，真实数据（实验原始归档，约 55 GB）在 `E:\paper_archives`，报告和几个归档脚本仍按 C 盘路径引用它。旁边的 `paper_archives_empty_20261014` 是空壳，可以删。

按用户确认清理后，**C 盘可用空间从 17.6 GB 增到 73 GB**：

| 做了什么 | 释放（约） |
| --- | --- |
| 软件旧版本（夸克、WPS、VS Code 扩展、Codex）和各类缓存（显卡着色器、网易云、Edge 离线缓存、G HUB、临时文件等） | 21 GB |
| 卸载 Docker Desktop、CUDA Toolkit 12.4（含手放的 cuDNN）、Nsight 三件、雷电模拟器，删 DiskGenius WinPE | 16 GB |
| Windows 安装监视日志、传递优化缓存 | 6 GB |
| 基础 Anaconda 整体移到 `E:\anaconda3`，原路径留链接 | 约 6 GB |
| 桌面"废品集中"移到 `E:\废品集中` | 3 GB |

## 对项目的影响

- `C:\Users\Admin\anaconda3` 路径不变（现为指向 E 盘的链接），`conda run -n DQN` 实测 PyTorch 2.3.1+cu121 在 RTX 4070 SUPER 上矩阵乘和卷积正常。PyTorch 自带 CUDA 和 cuDNN，不依赖已卸载的系统 CUDA 工具包；以后要自己编译 CUDA 代码才需要重装。
- Windows 组件存储清理（DISM）在提交阶段被拒绝访问，没生效；重启后可在"设置 → 系统 → 存储 → 临时文件"里重试"Windows 更新清理"。
- Windows 安装监视驱动仍在写日志（`C:\Windows\Panther\monitor`，每天一两个 16 MB 文件），会慢慢再涨，隔几个月清一次即可。

清理脚本、扫描结果和启动配置备份在 `outputs/maintenance/disk_scan_20261024/`。
