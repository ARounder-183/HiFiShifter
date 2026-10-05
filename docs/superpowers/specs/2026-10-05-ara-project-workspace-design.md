# ARA 工程级多轨原 GUI 工作区设计

范围更新：完整clip变换/HiFiGAN缓存/长源最终门见`2026-10-05-ara-complete-integration-design.md`。
本文工作区设计继续实施，旧unsupported和先导资源边界不能替代新增目标。

中文设计，2026-10-05。针对用户提出的“多个轨道都在一个窗口里”；沿用用户已要求的
自主分批推进、少做重复测试/review。本文补充内嵌编辑器设计，不取消原有完成门。
实施前必须阅读 `2026-10-05-ara-embedded-editor-design.md`；其安全/预算/平台边界仍适用。

## 目标与窗口语义

每条需要修音的 REAPER 轨道仍挂 ARA 插件。打开其中任意一个 FX 编辑器，原 HiFiShifter
多轨时间线同时显示同一 ARA 文档内已接入的轨道；选择任何一轨即可编辑/分析/自动应用。
不需要逐轨开窗口、另开独立 app、手动提交，原独立 app 的多轨模式保留。

VST3 FX 顶层窗口由宿主拥有。“一个窗口可编辑全部轨道”不等于强行关闭其它宿主窗口：
若用户同时打开多个 FX 面板，它们是同一工作区的同步视图，不是多套独立编辑权威。
不改父窗口、不搬动外部 app、不按全局“最近实例”选择工程。

## 已知事实与方案取舍

目前 `EditorSession` 弱引用单个 `ExtensionOwner`，Snapshot 由 `assigned_timeline`
过滤成该实例的轨道。`DocumentSession` 已持有完整宿主图、共享 edits、时钟、region
身份及 renderer 弱租约，因此并非前端缺少多轨布局，而是 GUI 会话范围刻意较窄。

选定文档级共享 actor：所有视图使用同一 TimelineState、参数权威、history、分析缓存与
自动应用代次。原 processor/controller 真实路由仍是进入文档的授权入口。
不采用“前端拼接多个实例”：跨轨撤销和部分提交失败容易不一致；不采用单个 master
插件混全部轨道：会改变 REAPER 路由、把声部加到错误轨道，并扩大宿主区域权限。

## 身份与可编辑范围

以已登记的真实 ARA controller / DocumentId 区分工程。仅聚合同一仍存活文档的 renderer
实际 assigned regions 的并集，去重后按宿主轨道顺序显示。同名轨道、共享 audioSource /
modification 不合并为一条轨道；不同工程即使源 persistentID 相同也不能互见。
零分配不意味着整张文档授权。未挂本插件/未被宿主分配的区域不冒充可编辑内容。

完整图仍来自 ARA 模型；workspace 仅是其授权投影。移除最后一个负责某区域的实例后，
它从可编辑范围退出；新增/删除/复制区域的稳定模型事件刷新视图，真实 pending 冲突
保留曲线并明确提示。不能把未知/销毁的 region 或任意本地路径交给 UI 读取。

## 编辑、历史与音频

EditorSession 改为弱引用 DocumentSession，文档惰性持有唯一 Arc<EditorSession>。
保留有界 FIFO、私有 namespace、原共享参数命令、原 DockRoot/参数编辑器及事件协议。
同文档多个 view 的请求串行执行，timeline/history 变化通知全部视图。打开新 FX 不重置
已有选轨、历史或 pending 编辑；不再因另一个视图请求首载而制造第二套权威。

工作区 history 按用户操作顺序跨轨撤销：A60、B67 后撤销 B 只恢复 B，A60 不变。
重做恢复 B67；一次跨轨批量操作是一条原有 undo group。宿主稳定几何刷新后旧几何
history 失效，不能通过撤销偷偷修改宿主 clip 位置。

全范围参数在一个文档事务内校验/接受；每个 renderer 仍仅合成自己的 assigned regions，
process 只读本实例快照。不得因为 GUI 聚合而向每条轨道重复输出全工程混音。
150ms 合并、过期任务拒绝、WORLD/PCM/退役快照预算以及 unsupported 边界全部保留。

## 保存与生命周期

组件 getState 先 flush 文档共享编辑请求，再只序列化本组件实际范围；不能让每个组件
保存全工程并在重开时相互覆盖。v2 原持久身份/限定范围恢复仍保留，并接受旧组件 state。
重开在宿主明确分配后合入文档权威，GUI 未打开时也可恢复供音。

关闭一个 view 仅撤销该 view 的回信/事件；释放某个 processor 不停止文档共享 actor。
关闭文档撤销所有租约、停止/join actor 与分析/渲染服务，再释放模型和 PCM。不能形成
document→editor→document 的 Arc 循环；不存在可用 renderer 时不保持后台无限重试。
视图每次请求仍核对原真实组件路由，已释放组件的残留 view 不能借其它实例继续访问。

## 主线程耗时前置风险

Task30 调试 bundle 出现 REAPER 长时间未响应，后来恢复，并非已证实死锁。源码确认
`enable_audio_source_samples_access → refresh_source_pcm → prepare_renderers → prepare`
在宿主模型回调同步执行 edited `render_edits`，存在重复重合成与主线程等待风险。
release 对照只用于分辨优化因素，不能替代线程边界修复。后续模型回调只能登记版本/
撤销输出/排准备任务，合成不持有阻塞宿主的长事务；发布前重新核对模型、参数和分配。
不可用时显示 pending/错误，不伪报已应用。Task31 先处理此门，再做多轨最终 GUI 验收。

## 验收门

1. 同工程两条同源轨道，仅开一个原 GUI 显示两轨/四段；在该窗口分别设 A60/B67。
2. 宿主逐轨 Solo 导出并用独立音高 oracle 验证 A≈261.63Hz、B≈392Hz、布局/gap 正确。
   B 改动/撤销/重做后 A 的 PCM 不变，不能只验证音频“有所变化”。
3. B 撤销恢复64、重做67；另开任意 FX 视图看到同一曲线/history，不能恢复旧本地副本。
4. 关闭所有 FX 继续供音；保存、正常退出、冷重开，未开 GUI 时两轨各自 PCM 一致。
5. 关闭作为入口的实例窗口不影响其它轨；宿主复制/移除实例与两工程隔离回归。
6. 已有 BPM/停播/seek/自动几何同步、44100/48000 mono/stereo、≤30秒与独立 app
   导入编辑门仍在；真实 pending 冲突/鼠标笔画没有证据时不得关闭目标。

当前只完成方案，尚未实现/实测工程级工作区；单实例或一个轨道输出不能代替上述门。
