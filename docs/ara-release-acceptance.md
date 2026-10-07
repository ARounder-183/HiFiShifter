# ARA 插件交付与验收记录

中文交付记录，2026-10-05。当前分支为 `feature/ara-plugin`，开发工作区为
`E:\code\HiFiShifter\.worktrees\ara-plugin`；主工作区 `develop` 未修改。

## 验收结论

用户使用 `release-delivery-02` 的双轨工程副本后确认“验收通过”，随后再次确认
“我测试过了，ok”。以用户实际操作作为集中 GUI 验收证据，不冒称代理逐项复跑了
所有宿主矩阵。保留原独立 App，插件使用原 GUI，同工程多轨共享工作区、逐轨输出。

本次交付包含跨轨曲线/气声/音高自动应用、宿主渐变显示、播放控制与播放头同步、
编辑状态保存恢复、正向线性拉伸及 HiFiGAN 缓存/分块处理。相关源码与模型证据见
`probe/ara/EDITOR-SYNC-FINDINGS.md`、`SOURCE-AUTHORITY-FINDINGS.md`、
`NEURAL-CACHE-FINDINGS.md`、`MULTITRACK-RESOURCE-FINDINGS.md`。

用户已要求停止重复全量测试/新增 review。本次整理与推送不重新执行测试或构建。
最后一次集中回归已完成 frontend 2729、kernel App配置587/插件配置585、App245及
plugin156项；后续Verify流水线被中止，不能称整条Verify成功。

## 交付位置

已构建的 Release 位于 `.build-tmp/deliveries/release-delivery-02/`：

- 独立应用：`app/HiFiShifter.exe`，及模型/运行依赖。
- 插件：整个 `HiFiShifter.vst3` 文件夹，包括 loader、engine、原GUI和模型。
- 构建身份：`build-manifest.json`，构建源码commit为 `5ea3df47`，共45个产物文件。

产物与用户工程位于ignored目录，不随Git分支推送。manifest记录的是构建时状态，
`verificationRequested=false`、`nativeAcceptance=false`不回写篡改；后续用户验收
结论由本文记录。没有重建或替换用户当前REAPER加载的模块。

## 已确认的边界

- 只做正向线性拉伸；倒放、非线性stretch markers不在本次范围。
- 用户允许HiFiShifter自己的渐变示意曲线；普通渐变声音仍由REAPER应用一次。
- HiFiGAN分块，HNSEP整段处理。三分钟HNSEP曾实测瞬时峰值约13.3GB，已有资源预检；
  不能保证普通16GB机器无条件运行长素材，也不能把PCM额度当模型总内存。
- GPU VRAM、任意自定义模型的资源严格上界及其它宿主矩阵没有完整测量证据。
- Linux/macOS插件本次不要求实现；独立App原跨平台路径保留。

## 后续使用与开发

用户操作见用户手册的插件章节（[简体中文](i18n/USERMANUAL.md#八vst3--ara-插件reaper)，
另有英/日/韩/繁中四语）；构建入口见 `tools/build-hifishifter.ps1`，打包见
`tools/package-vst3.ps1` 与 `scripts/pack-portable.ps1`。只维护同一份frontend/kernel，
宿主权限与生命周期留在适配层。
用户已明确授权把当前分支改名为 `feature/ara-plugin` 并推送origin；不改名或删除其它
远程分支，不强推，不自动创建PR。
