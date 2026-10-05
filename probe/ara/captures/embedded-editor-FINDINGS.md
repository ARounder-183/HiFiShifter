# 二期内嵌 GUI 首次宿主门：通过显示与关联，未完成修音验收

中文实测记录，2026-10-05，隔离REAPER7.81，非用户工程。

## 实测

- 初始前端tsc-b与生产构建成功，Rust引擎构建正常exit0。薄入口首轮CP936误读UTF-8
  中文注释后报语法错误；显式utf8后只剩CRT terminate同名问题，改engine_exit后
  薄入口编译exit0。未因此重跑全部Rust/前端测试。
- 在项目scratch目录CWD启动，未加任何PATH目录，REAPER加载规范bundle并识别ARA。
  原日志有`factory countClasses ->3`、`mapped`、两区域快照ready与`fx=0`。
- 原日志三组processor/controller消息握手均`route sent result=0`、`route requested
  result=0`，`controller editor route bound to actual processor`。不是选“最近实例”的推断。
- `IEditController::createView -> native editor`与WebView固定本地域导航已运行。
  Windows accessibility明确在REAPER的`VST3: HiFiShifter ... Track1`对话框下面包含
  `HiFiShifter ARA`、`https://hifishifter.invalid/plugin.html`、原文件/编辑/视图菜单、
  原时间线、参数编辑器、WORLD算法、自动应用栏`编辑0/音频0`和宿主轨道名称。
  正常跳过隔离profile设备提示，不修改用户设备设置。
- 只读进程检查仅REAPER46132，无HiFiShifter.exe。GUI是宿主内WebView，不是外部app。
- 一次性脚本已导出`embedded-editor-baseline.wav`，1323736 bytes；未编辑基线不能
  证明自动修音。尚未执行edited/reopened频率/PCM验证。
  基线SHA256：`E4E12F6BCA92C3497F4CAA69E94050A5EDB8ABF6AB83FF9DD9FCAAB7928F1A6D`。

## 当前问题与限制

Computer Use截图有捕获前景Codex而非REAPER的已知不一致；本轮以实际REAPER accessibility
层级与插件日志记录显示事实，**没有将错误截图保存成GUI证据**。初次主窗口句柄过期已
重新选取；提示框元素点击有bounds mismatch，使用returned dialog的Raise及快捷键正常
跳过设备向导。评估/About窗口随后仍在，继续输入前需要刷新焦点，不能重复用旧index。

首次真实UI日志暴露`has_timeline_clipboard`的时间轴轮询，以及`read_system_clipboard_object`
和`transliterate`还未接线。时间轴几何剪贴板轮询已在源码按插件模式关闭，**运行bundle
仍是修正前版本**。参数剪贴板和转写还需复用/明确能力适配，不能伪造成功来消除日志。

源码已追加退役快照非实时回收：实时计数+SeqCst指针建立读区，只有无读者时回收非current
Box；发布失败保留旧音频。新增连续1000次/活跃读者回归未运行，当前加载引擎也尚未包含
这项新代码。最终需要并发与真实process无分配/释放验证，不能只据算法分析宣称安全通过。
最新源码cargo check --tests正常exit0，包括分配前非实时collect_retired；这不是行为回归通过。

下一步：保留当前一次性工程，继续完整原GUI参数动作接线，完成自动pitch导出、关闭FX/
保存重开、独立app与资源矩阵的集中验收。当前没有技术不可实现证据，不关闭目标。

## 本批接线与集中回归

原生剪贴板已抽为`hifishifter-clipboard`共享crate，沿用原Windows/macOS/Linux自定义
格式、争用重试与旧信封兼容；app路径是再导出，plugin实际读写参数JSON。文字转写/匹配
整体迁到kernel/search，app原路径再导出，插件调用真正transliterate_batch，不回传假数据。
这些源修改未热替换当前运行bundle。

插件首轮55 lib测试54通过/1失败：算法枚举的serde(other)保留Unknown，typed deserialize
成功不等于算法合法。插件边界拒绝Unknown/vslib（不改app的旧工程兼容）后完整重跑：
lib55、mapping13、依赖树1、native renderer5、exports1，共75，正常exit0。
包括actor真正getState尾块屏障、无效patch原子性、持续1000次快照回收和process零分配/
零释放。它们不代替真实宿主音高/保存验收。

共享kernel editor12、search39、HfsPeaks7均正常exit0。独立app cargo check exit0。
前端先跑相关23通过；全量首次2698通过/12失败，涉及Radix Text/字号规则、独立窗口
事件异步初始化及布线门只识别Tauri。修正为原排版角色、独立app保留原直达Tauri加载
时序、插件分支才用hostEvents；布线门检查真实actor match而非手工例外。六个失败文件
集中定向61通过，tsc-b通过；修正后的全量仍待最终一次复跑，不写“全前端绿”。

当前桌面只读诊断为Default（不是锁屏）。Sky fresh observation能读FX及About对话框，
但即使重置JS观察器，最新index1的Raise仍报`no cached secondary actions for reaper.exe`。
停止盲目输入，不用PowerShell UIA/猜窗口句柄绕过，也不以脚本参数提交冒充手绘。
Lua原有save命令已把当前未编辑一次性工程正常保存为`.build-tmp/embedded-probe/embedded-editor.RPP`
（5056 bytes，log project saved）。隔离REAPER46132仍运行，未强杀。

## 后续源码修复与回归（2026-10-05）

- 模型资源随规范bundle复制到 `Contents/Resources/models`；隔离REAPER日志确认FCPE DirectML会话创建并通过smoke test。
- 插件会话现在按实例持有可取消、可join的分析worker，消费ClipPitchReady并重组原线；关闭FX不会取消其它实例，也不会留下跨会话游离线程。
- 真实actor回归覆盖“先画pitch、分析完成后自动应用、无需GUI轮询”，独立自相关测得329.104Hz（MIDI64目标329.63Hz）。缺原线时自动应用保持pending并保留可保存曲线。
- 新bundle已重建并用隔离RPP重开；原时间线、钢琴卷帘、WORLD算法和自动应用栏再次出现。
- Computer Use对REAPER父窗口坐标drag返回WebView子窗口目标不匹配；未猜HWND、未用PowerShell UIA或Lua提交替代手绘，因此真实手绘、导出及重开PCM仍是open项。
- 追加验证：WebView accessibility树可读到绘制工具，但对同一索引执行click仍返回“element is not available in cached app state”；secondary action没有Invoke。短暂实现的同进程输入代理会使所有插件IPC请求超时，已撤回，未进入bundle。

## 用户试用反馈及高优先级修复（2026-10-05）

用户真实操作确认“基本可以”，但报告停播光标落在clip内时持续重复音频、两边光标不一致、复制多轨道后无法使用。截图和实际日志显示第二轨plugin_refresh失败，文档为2个sequence/6个region但1个source，恢复报persistent identity ambiguous。这不是仅Computer Use的输入限制。

本批源码修复：realtime停播清零（offline导出继续供音）；修正base/position重复相加，并允许前端在宿主停播seek时更新游标；单组件保存/恢复改为host-assignment限定范围，同源复制轨道可分别编辑；跨轨revision变化只在本实例投影变化时产生Conflict。

验证：同源别名lib测试62 passed、exit0（含原实时零分配/释放、WORLD独立输出oracle、共享身份复制/恢复及双actor连续编辑）；前端新增停播seek与原playheadGuard共15 passed、exit0；frontend tsc/生产build和规范bundle build exit0。未重复全前端及app全套测试；既有警告保留。

新bundle：`.build-tmp/embedded-feedback-01/HiFiShifter.vst3`，没有替换用户仍在运行的`.build-tmp/embedded-vst3`。真实REAPER三项回归还没验证，不能将源码测试写成宿主通过。播放/暂停控制REAPER（中优）和clip改动自动刷新（低优）仍待下一批；正常保存并退出REAPER后才能升级加载版本。

## Task28：BPM实测及宿主控制接线（2026-10-05）

本批已实现标准ARA宿主播放请求租约，原ActionBar播放/暂停/停止在宿主提供可选接口时开放；native主线程发送，actor/音频线程不调用。租约原线程、撤销和有限位置回归1 passed，plugin lib65 passed（新增首载Unsupported回归首轮失败，修复后65/65 exit0）。请求确认包只代表送达，不伪造宿主开始播放；前端相关18 passed，tsc/build成功。

稳定host model变更且无pending时actor自动刷新，tempo从VST3 kTempoValid取得，GUI经版本事件/轮询同步；有真实冲突继续保留未应用曲线。不是完整Tempo Map/时间拉伸支持。

真实UI验证使用`.build-tmp/embedded-transport-probe/embedded-editor.RPP`副本和独立profile，原`.build-tmp/embedded-probe/embedded-editor.RPP`始终SHA256 `4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3`。副本在REAPER原生Project Settings由120改150，插件BPM显示150。默认Beats时间基准同时使媒体PLAYRATE1.25；回到120并仅在副本改Time基准后再次改150，媒体保持PLAYRATE1，插件仍显示150。截图`bpm-sync-150.jpg`可评审；保存副本5096 bytes，RPP含TEMPO150/PLAYRATE1。

按用户提示先鼠标点插件标题栏，accessibility实际焦点进入VST3 FX；随后点网页播放按钮，仍被工具返回“point ... is over msedgewebview2.exe ... not target window reaper.exe”。激活/新截图后仅重试一次，同样拒绝。没有用脚本或隐藏入口伪造按钮验收，实际播放/暂停和GUI保持打开时自动重载仍未测。测试副本已保存并正常退出，原工程未改。

启动脚本现清空旧command.txt，并支持独立ScratchName；否则上次残留save命令会在新实例重放而覆盖旧测试工程。此修正只作用一次性采集工具。

最后一次集中审查发现：首次Unsupported必须存入可见状态；get_playback_state必须先于timeline加载门禁，否则失败几何下不能观察真实停播。两项已修并纳入回归。该回归进一步实测发现Snapshot序列化只保留take权威，扁平倍率反序列化为1，检查前必须normalize_takes；不放宽时间拉伸边界。完整lib65正常exit0，重建含最新前端和引擎的bundle exit0，既有Rust/Vite警告保留。

新版隔离副本的连续UI验证：插件保持打开，原生Project Settings将150改180，BPM框自动变180、媒体仍x1，截图bpm-sync-live-180.jpg。原生Media Item Properties仅选第一段，将Item position从0:00.000改0:01.000，插件原时间线和参数区自动右移，未点击重新载入，日志clipStartsSec=[1.000000,3.000000]；截图host-geometry-auto-refresh.jpg。停播宿主光标位于1秒时GUI也显示0:1.000，未重复相加。没有本地pending编辑，不能据此宣称冲突交互通过。网页按钮/手绘/全流程PCM仍未验收。

原生Duplicate tracks复制测试轨道，日志sources=1/modifications=1/regionSequences=2/playbackRegions=4，两个原GUI均为已应用0/0且只显示各自两段，截图copied-track-two-editors.jpg。新采集段未出现旧identity ambiguous阻塞；模型变更过程中仍有两条get_param_frames unknown host track日志，尚未定位，不宣称双实例曲线/PCM全流程通过。副本正常保存8356 bytes，TEMPO180、两轨POSITION1/3和PLAYRATE1；原用户工程不变。

最终一次前端全量：`npm test -- --reporter=dot`，311 files/2715 tests passed，exit0，28.79秒。既有Canvas/act环境警告仍在；通过不代表Windows原GUI手绘音频验收。没有重复运行内核/app全套。
