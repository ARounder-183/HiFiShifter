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
