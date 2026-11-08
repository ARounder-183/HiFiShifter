# 参数线区间块与 ARA 运行模型

## 数据权威

音符块是根参数线上的可操作时间区间。块保存 ID 和整数帧半开边界
`[startFrame, endFrameExclusive)`，不保存独立音高、纵向中心或 Clip 父子关系。
纵向显示取当前参数线的代表值；拖动块时修改对应范围的参数线。

共享内核的 `ParamLineBlockSet` 保存规范块列表、修订号、用户修改范围及 GAME 初始分段。
区间必须有正长度、ID 唯一且互不重叠。未初始化与明确保存的空列表是不同状态：删除全部块后，
后续刷新不能重新派生出已删除的块。

App、ARA 和普通 VST Capture 使用同一模型。旧 Clip/Take 分段及 merge/interval edit
字段仅用于兼容读取和迁移，不再作为第二套编辑权威。

主要入口：

- [共享区间模型](../backend/hifishifter-kernel/src/param_line_blocks.rs)
- [Clip 几何映射](../backend/hifishifter-kernel/src/param_line_blocks/geometry.rs)
- [前端规范布局](../frontend/src/components/layout/pianoRoll/paramLineBlocks.ts)

## 编辑与保存

手动创建和 GAME 生成的块进入同一列表，使用相同的拖动、延展、切分、合并、删除及保存路径。
GAME 初始分段单独保留，供选区内的“音符块还原”使用；还原只恢复边界和分段，不恢复参数值。
迟到分析需复核 Clip、Take、源身份及用户修改范围，不能覆盖分析期间的新编辑。

创建与延展始终从按下时的固定布局重新计算。覆盖其他块时保留其未覆盖部分；覆盖中间可留下
左右两段，反向拉回时原有未覆盖区间重新出现。合法的短块不能因显示宽度而被删掉。

曲线补丁和块布局在同一事务提交，同时检查布局修订、曲线修订及几何签名。过期操作整体拒绝，
不能只保存曲线或只保存边界。连续手势串行提交，旧回执只能确认或撤销自己的预览。
纯边界编辑及其撤销不触发音频重新生成。

Clip 移动、伸缩、复制及裁剪沿用参数线的时间映射，同时作用于块和 GAME 初始分段。
不完整宿主清单、分割后的短暂重认领及暂时没有样本访问权，不能被当作删除块的依据。

相关入口：

- [App 原子提交](../backend/src-tauri/src/commands/param_line_blocks.rs)
- [插件命令准入](../backend/hifishifter-plugin/src/editor/commands.rs)
- [前端提交与交接](../frontend/src/components/layout/pianoRoll/useParamLineBlockPersistence.ts)

## 预览与交互

块、曲线和采样窗口通过同一份 `ParamEditorRenderFrame` 发布。手势冻结当前可见底图，
临时补丁只在声明的影响区间生效；无关块不能因为另一份窗口、stride 或迟到回包而移动。
手势内的临时代表值不进入工程保存。权威窗口接手时需核对作用域及实际内容，不能仅靠请求完成
或窗口 key 相同撤下预览。

右键拖动建立贯穿参数编辑区纵向的时间选区，并选择其中的规范块；单击空白处取消选区。
默认物理 Backquote / Shift+Backquote 在音高绘制和块编辑间切换，退出块编辑时切回 pitch。
输入框、可编辑文本及按住重复沿用快捷键焦点规则，IME 的 Process / 229 事件按物理键匹配。
显隐只控制绘制，不删除块数据；标签切换和浮窗重开仍读取相同的已提交状态。

相关入口：

- [渲染帧](../frontend/src/components/layout/pianoRoll/paramEditorRenderFrame.ts)
- [创建事务](../frontend/src/components/layout/pianoRoll/paramBlockCreationTransaction.ts)
- [物理键匹配](../frontend/src/features/keybindings/keybindingMatch.ts)

## REAPER / ARA 调度边界

REAPER 可为每个 Item 创建播放组件，因此一条轨上的一个可见插件不等于一个原生组件。
未收到区间或序列分配的组件保持空闲；收到分配后按需创建准备队列。显式离线准备独立排队，
空图仍输出正确静音。组件关闭后不能被迟到请求重新启动。

宿主主线程任务泵与心跳各使用时间和数量预算，多个组件、视图按游标轮转。预算是协作式的，
不能抢占单个长 SDK 调用；主线程不能通过排空全部队列或 sleep 等待后台完成。

大音源在 SDK 允许的模型回调中创建 reader，由两个有界 worker 分段读取。每个 reader 串行
使用，结果发布复核取消状态和源代次，最后的 reader 引用在合法 SDK 作用域销毁。
源指纹、验证和 WAV 物化由后台准备，编辑器先接收元数据；文件就绪后从最新文档重新投影，
保留准备期间的编辑、选择和历史。同一不可变 PCM 引用可复用文件，换 PCM 必须重新准备。

模型登记幂等，插件启动不复制模型库。插件的相关 ONNX 会话按可用核心限制推理线程，最多
4 个；独立 App 保留自身策略。实例启动阶段及慢调用限流记录，用于区分宿主绑定、资源准备、
WebView 附着和 React 首帧，不能把页面桥接就绪当作整个插件启动完成。

锁顺序统一为先取播放区域、再取 Item 映射，快照取得后释放两把锁再访问 PCM。
关闭准备队列时先从所属槽位取出队列并释放锁，再取消和等待 worker；不能持槽位锁 join。

相关入口：

- [主线程预算](../backend/hifishifter-plugin/src/host/work_budget.rs)
- [授权样本读取](../backend/hifishifter-plugin/src/render/source_reads.rs)
- [后台文件准备](../backend/hifishifter-plugin/src/editor/source_files.rs)
- [编辑会话](../backend/hifishifter-plugin/src/editor/session.rs)

## 验证与发布

回归覆盖区间避让、连续创建、乱序回执、空布局、撤销重做、GAME 局部更新、Clip 几何、
保存恢复、准备期间编辑、源撤销及非阻塞宿主回调。主线程和模型生命周期约束由原生集成测试
验证；绘制连接与作用域约束由前端测试验证。

Windows bundle 发布使用 [加载与资源门禁](../scripts/vst3/windows/verify.ps1)，同时检查
原生工厂、组件创建、插件/浮窗 HTML 以及本地 JS/CSS 引用。模型和运行库必须与实际导入一致，
App 与插件分别构建，避免 Cargo feature 合并引入额外 DLL。

构建、日志、临时复现工程及安装备份放在忽略的 `temp/`，备份不放在 VST 扫描目录。
真实工程的冷启动、分析期间宿主响应、多实例及保存重开仍需宿主验收；单元测试通过和模块
加载成功不能替代性能测量。
