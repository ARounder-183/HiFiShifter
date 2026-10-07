# Task 11 FINDINGS：拉伸与倒放

本文件是 2026-10-04 在 REAPER 7.81 隔离实例中的实测归一化记录；原始插件日志保留在
同目录 `task11-plugin.log`。

`task11-stretch-reverse.json` 中 region 1 的 `durationInModificationTime=2.0`、
`durationInPlaybackTime=1.0`，且 transformation flags 为 `1`（Timestretch），因此
U2 通过。region 2 通过官方 action 41051 倒放，宿主 section reader 确认 `reversed=true`；
它仍使用同一个 source persistentID，两个时长坐标与普通 region 相同，flags 仍为 `1`。
ARA reader 读取共享源首 16 个样本，与文件正向样本的最大差为 1.40624999978023e-8。

U1 是当前映射的方向表达缺口；宿主是否在插件处理器外部处理反向播放尚需输出实验。
本体侧保留 `Clip.reversed` 只是可能的退路，还需要可信方向通道，不能宣称倒放已支持。

JSON 是 `verify_task11_capture.ps1` 从原始日志和 WAV 自动生成的归一化记录，
不是完整 ARA 模型 dump。早期 `B_REVERSED` setter 的观察已废弃。
