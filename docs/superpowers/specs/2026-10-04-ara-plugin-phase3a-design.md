# HiFiShifter ARA Phase 3a 输出链路设计

> 中文设计说明：本批只证明宿主 PCM 到 VST3 输出的链路，不宣称修音功能完成。
> 日期：2026-10-04；分支 `codex/ara-plugin`；按用户授权在本会话内分批执行。
> 上位：[v1 设计](2026-10-04-ara-plugin-v1-design.md) / [产品设计](2026-10-04-ara-bridge-design.md)。

## 1. 目标与边界

交付可测的 VST3 缓冲 ABI、每处理器区域分配、宿主供源的简单播放快照，以及 REAPER
导出的波形证据。只用普通和裁切 region 验证 PCM 保真；拉伸与倒放单独判定。
不把原来的文件型 `render_timeline` 当成宿主渲染，不修改 app / frontend / SDK 检出。

## 2. 选择与理由

选择“模型线程读宿主源，离线准备不可变快照，音频线程只读快照”。另两条路径不采用：
在 `process` 里调用 reader 会阻塞且越过访问 scope；按 persistentID 解码文件会绕开
宿主权威源。Phase 3a 不引入 ONNX，先把输出错误与推理错误分开。

VST3 的 `AudioBusBuffers` / `ProcessData` / `ProcessContext` / `ProcessSetup` 用 `repr(C)`，
以锁定 SDK 的 C++ `sizeof/alignof/offsetof` 可执行程序交叉验证，不靠抄常量猜布局。
本批只支持 `kSample32`、一进一出 stereo；未实现的组合明确拒绝，不能返回成功。

## 3. 所有权和数据流

`CreateContext::realtime_key()` 已是 region 的 model-ref 地址键；不是 RawHandle 下标。
在创建 region 时登记 `key -> document/region`，销毁时撤销。每个 renderer 单独接收
add/remove 分配事件，按分配集合取 region。三个处理器不得各播放整张文档。

本地 vendored `ExtensionBinding` 添加模型线程 assignment observer；callback 发布
不可变键集合，不在音频线程读取其 Mutex 集合。扩展 owner 由入口 adapter 的 builder
闭包捕获，保持到 native entry 最后一个 COM 引用释放，不能仅跟随组件 `terminate`。
document 先销毁或 companion 先销毁均需 tombstone 测试。已存在本地补丁，不修改 SDK。

源 reader 只能在 `HostContentScope` 内使用。先导夹具最长 30 秒、44100/48000Hz、1/2
通道；PCM 与快照总预算 512 MiB，超限非实时明确拒绝并记录。源撤销/更新后撤销旧版本
发布。音频回调不能持有 reader、日志器或模型锁，也不能回收最后一个大快照引用。
先导版退役快照保留到 renderer owner 析构，预算内拒绝后续发布；长期回收策略属于 3b。

快照按 `projectTimeSamples` 随机访问，负时间、静音间隙与越界补零；不能靠内部累计
playhead，否则 seek/loop 会漂移。没有 context 则输出零并计数，不推断播放位置。
失效、未就绪和未知方向不能静默当成功输出；非实时汇总诊断必须区分这些原因。

## 4. 验收与杀死判据

1. 原生 SDK 布局与 Rust 完全一致；缓冲尾哨兵、零样本、空总线与非法格式测试通过。
2. 分配/remove/销毁/双文档测试证明无重复输出和串文档；callback 无锁、无分配、无 IO。
3. 宿主授予访问时读取的 PCM 能播放；revocation 和版本变更后旧输出不能复用。
4. 隔离 REAPER 输出普通/裁切/seek 与 oracle 最大绝对差 <= 1e-6，无双倍幅度。
5. 使用不对称波形、action 41051 与宿主 section 结果验证真正倒放，再比较最终输出。
   若仍正向且无可信方向输入，明确停止“全 v1 支持”声称；不能凭 `Clip.reversed` 猜。

首块和 miss 允许在未完成阶段有显式零输出，但不是最终供音通过。Phase 3a 通过也只
证明输出链路，不关闭 A3/U3/U4；完整修音与 R4 压力测试在 Phase 3b。实测无法解决
方向或连续供音时写 ledger 并报告技术阻塞，用户的“继续”不等于允许伪造成功。
