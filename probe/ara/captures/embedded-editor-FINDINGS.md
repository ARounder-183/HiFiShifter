# 二期内嵌 GUI 实测记录：键盘单轨链路通过，完整多轨工作区未完成

中文实测记录，2026-10-05，隔离REAPER7.81，非用户工程。

## 当前权威摘要（Task30-31）

## Task36/38最新源码（尚未完成最终宿主门）

用户目标已扩展为原独立App、同工程多轨原GUI/独立输出、完整位置/拉伸/渐变、HiFiGAN缓存，
最终总纲为ara-complete-integration设计/计划。时间拉伸用户再次明确“必须”；倒放暂不做。
旧30秒/stretch/fades unsupported边界是先导，不作为整goal完成定义。

纯editor角色已按SDK透传，实时/停播/离线、in-place/同bus、inactive/silence/非法输入/
尾哨兵与零分配新回归通过。playback角色仍替换自己的区域；后台只准备playback实例，
GUI汇总实际音频准备状态。首轮完整lib76正常exit0；测试fixture改为真正有playback职责的
组合角色，不再用editor-only错误地证明歌曲供音。没有重复前端全量。

原生隔离ROLE版引擎DB8A2F0CE406D5A61A3B722FB318B0E5A06490B92A4CCB52A89508AACA83C604真实加载，
日志8819行后采集。原GUI Space实际Playing时，在没有手工seek的首个采样段累计82次
回跳，realtime4不变、prefetch11832→12208、writer9070→9430。证据
transport-prefetch-diagnostic.json；不能按主/子截图不同拍摄时刻推算精确偏差。
不能排除所有prefetch，本机REAPER主要用此模式更新。官方SDK已核对到commit
c0eafe87863b2bf69c5c822760f1b32a753b211b，parent(3)为所属project、GetPlayPositionEx
是延迟补偿实际听到位置；现已typed适配并在活view UI timer发布独立原子元组，预取/
其它clip停止不再覆盖这个UI权威。fake host真实QI/线程/引用回归1与transport3通过，
不是实际REAPER调用成功/游标已修好。后续native门仍open。

正向线性时长比已放行到原kernel，plain mixer继续拒绝“重采样冒充拉伸”，没有改原
kernel算法/独立App语义。0.5秒220Hz源→1秒的实际内核输出44100/48000完整长度，前/
后窗口有声、独立自相关约220Hz；GUI真实take倍率0.5/目标长度回归通过。stretch过滤
3 passed/exit0。倒放仍明确拒绝，非线性marker/tempo映射、渐变完整数据与HiFiGAN缓存未完成。

最新host-stretch-01规范release包含更新前端、REAPER getter与线性拉伸：tsc/Vite/
Rust/native loader全部exit0，release2m01s，引擎SHA
2F4FD9EC8B443DBA8A7225A740EB5506C2EC7071AEBBB2AC880438044D951C38。
用户物理Esc停止Computer Use后不继续操作/重开窗口；该包未新宿主验收，没有热覆盖。
原用户RPP仍4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3。

该源批最后完整plugin lib80/0失败正常exit0（10.65s），包含实际REAPER typed adapter的
QI/所属project/暂停与停止/线程/引用平衡，以及host权威不受预取污染/后退seek、线性
stretch真实PCM与GUI状态、原实时边界/Doc恢复回归。现有Rust/Vite及测试raw-pointer
unused-assignment警告保留；不是四项目标或native最终矩阵通过。

Task31最新源码：宿主prepare只排有界后台任务，原参数合成抽为冻结RenderInput，自动应用
和外部提交在计算时不持document.transaction；发布核对model/edit/render_epoch/分配。
首轮lib70、收尾render26均正常exit0。一次集中review指出冷恢复过期无重排/分配与发布
竞态，已补统一恢复、事务序列化和过期重排。最新定向bound_tests11通过/exit0；
总用例73没有重复全跑，尚未做最终native性能/最终review。新冷恢复测试实际验证两轨
供音，无GUI；原生assignment测试必须在绑定模型线程驱动，不能在另一线程触发桥接
线程拒绝后误判为序列化失败。用户要求之后只在最终做一次review。

隔离preparation-01是修正review竞态前的release构建，exit0/1m51s，引擎SHA
55BC4BEEA7DD64E30DE6173F2ABE75A126C6A642A084458CA031832D810CAA09。
用新的BundleDirectory参数和fresh embedded-preparation-probe启动，日志8362行后出现
background snapshot ready和原GUI分析；不代表最终修正源码音频通过。未向已有实例
发送脚本，未把默认embedded-vst3热覆盖，未抢用户正在操作的窗口。

用户新增播放头跳动/渐变/拉伸反馈：源码共享clock多writer/prefetch更新是候选，不是
根因实测。锁定ARA SDK明确纯editor renderer必须透传，现audio_process两角色皆替换
输入，职责错误需修；是否实际覆盖REAPER fade需导出确认。时长比已映射但GUI/快照
仍主动拒绝stretch；普通fade形状不在ARAPlaybackRegionProperties字段中，不能把
content-based fade标志冒充普通fade数据。详见新plan Task36-37；这些项仍未完成。

以下历史段落保留当时状态，不能把早期“尚未修音”当作当前状态，也不能把单轨通过
当作全部完成。Task29原GUI键盘编辑/自动修音/关闭FX/冷重开已实测；本批新增第一轨60
的真实GUI编辑/宿主独奏输出，4窗口220.5→260.94674556213016Hz，gap0，baseline布局
maxdiff5.960464477539063e-8。WAV gui-dual-track1-60.wav，报告gui-dual-track1-output.json，
fresh verifier exit0；未用脚本写音高、未启动独立HiFiShifter。

第一轮REAPER进程后来不存在，文件hash表明本批编辑未保存，不推断其退出原因；确认
进程缺失后重新启动同一隔离scratch，原GUI再次设60，save命令完成。RPP归档
gui-dual-partial-60-and64.RPP，SHA256 DD1A87B5C591EBA71D5F12D385D57205FEE4BE87F396112E1B72E98F7D4096DD。
第二轨仍是上一批64；本批没有第二轨67/撤销/重做/冷恢复证据。归档只是部分编辑工程，
不能从“保存完成”推出重开PCM通过。

REAPER3904保存后连续数次未响应，plugin_get_apply_state/get_playback_state各超时；随后
Responding恢复True，CPU累计约248秒，**不是已证实死锁**。源码的宿主源授权回调会同步
prepare_renderers→prepare→edited render_edits，并在计算期间持文档transaction，存在
主线程/重复合成风险；无真实栈/profile时不将全部卡顿归因于此。工具报告窗口内检测到
用户输入后停止继续发键鼠，不抢焦点/强杀/热覆盖。Release对照已在新目录构建exit0，尚未
证明宿主耗时改善。原加载引擎hash仍6F066ADFD512C2722BCFC3AA44469AA76DD3A6D9A2B200D94F691F7D3CF424E8。

用户明确希望多轨同窗口：已增加工程级共享原GUI设计/计划，底层renderer/state仍逐轨
隔离，完整双轨验收转到新的Task35。当前还没有实现该工作区；鼠标笔画、pending冲突、
资源矩阵与独立app真实导入编辑仍open。原用户RPP hash仍4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3。

Release命令 `build_embedded_editor.ps1 -SkipFrontend -Release -BundleDirectory embedded-release-probe`
正常exit0，cargo optimized 6m00s。该脚本默认debug不变，MSVC后仍重设干净TEMP/TMP和两个
SDK目录，未重建前端、未跑旧全套测试。bundle与backend/target/release引擎SHA一致：
FA52774D93DC8F7226699AAE5991D8531B8F66B79D539368822CA43D6D424D78。
thin模块SHA 8DB2C7B5EBD5BCF6B59177B025A158359E4393F55277B8FF541D3CDA20249891，dumpbin实际
导出GetPluginFactory/InitDll/ExitDll；动态依赖DirectML.dll/SoundTouchDLL.dll在邻接目录。
ort-sys当前构建output为static=onnxruntime，没有load-dynamic，不虚构缺少onnxruntime.dll。
front插件入口/models存在；未真实加载release插件，不据exports外推REAPER性能通过。
重复raw edited WAV的SHA与gui-dual-track1-60.wav相同，移至ignored
.build-tmp/embedded-dual-edit-probe/raw-captures可恢复，未删除/未把profile主题加入Git。

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

## Task29：真实原GUI键盘编辑与自动音高输出

插件首帧错误来源为独立app默认track_main；lazy初始化在插件清空轨道/选择，独立app保留原默认。新双轨重开日志（6165行之后）未出现unknown host track/identity unresolved/Invoke failed。回归旧版1失败/1通过，新版相关19通过，tsc通过。

真实Tab先只循环宿主工具栏，自有HWND增加WS_TABSTOP，WM_SETFOCUS使用标准WebView2 MoveFocus，IPlugView onFocus转交受UI线程/token校验的焦点，不改父窗口、不代理按键，COM/Win32调用前释放锁与借用。完整lib66 exit0；新HWND回归覆盖首个旧焦点null、跨线程及关闭后拒绝。真实Shift+Tab进入HTML，Tab执行原工具切换，Space发GUI Playback requested并令REAPER实际Playing。为避免短素材自然结束假装暂停，仅在副本临时开Repeat；GUI再次Space主动停在2.414秒，两GUI和宿主秒数一致并保持静止，之后恢复Repeat Off。播放中的多截图不是同时采集，不拿其时间差证明游标不一致。

原GUI使用F7选择工具、Ctrl+A选本实例素材、Ctrl+Shift+A转参数选区。原逻辑只切logical surface，画布局部Ctrl+0收不到事件；补scroller DOM focus后真实“音高设置到...”对话框打开，输入MIDI64，Shift+Tab到原确定按钮并Enter激活。状态从编辑1/音频0自动到1/1；另一轨保持0/0。没有外部app，没有手工Submit，也没有用脚本/IPC代替曲线编辑。

第二轨由宿主原生Solo独奏，隔离Lua仅负责DAW导出：gui-keyboard-baseline.wav与gui-keyboard-edited.wav，第一段实际起点1秒、第二裁切段3秒。独立验证baseline源/布局maxdiff=5.960464477539063e-8，4窗口220.5→329.1044776119403Hz，edited RMS0.122872205262133、平均差0.1530956955642657，gap0。报告gui-keyboard-output.json。验证器新增显式FirstClipStartSec参数/保留旧重载，不移动真实PCM来凑原布局；6旧+3移动布局回归exit0。

正常关闭两个FX窗口而不卸载效果，确认无GUI后重新导出gui-keyboard-closed-editor.wav；PCM相对已编辑输出maxdiff0，报告gui-keyboard-closed-editor-output.json、截图gui-keyboard-closed-editors.jpg。WAV整体hash不同来自REAPER元数据，PCM未变。隔离RPP保存40645 bytes，完整REAPER重开输出/曲线恢复待本轮继续验证。

最终正常退出REAPER、构建最终源版本（含review单素材焦点补齐）、冷重开40653-byte测试RPP。在第二轨编辑器未打开时导出gui-keyboard-reopened.wav，独立PCM比较maxdiff0，329.104Hz仍在。再从宿主原生FX按钮打开第二轨，原参数区恢复MIDI64曲线，截图gui-keyboard-restored-curves.jpg；会话代次重置0/0是新会话，不代表曲线被清空。Lua使用既有source-changed标签仅作这次导出命名，没有改源、更没有写参数；最终报告gui-keyboard-output.json的reopen_checked=true对应此次真正REAPER冷重开。

最终前端全量`npm test -- --reporter=json --outputFile=<worktree>/.build-tmp/embedded-final-frontend-tests.json --silent`，312文件/2718测试/0失败/success=true、exit0；tsc及生产bundle build exit0。引擎SHA256 `6F066ADFD512C2722BCFC3AA44469AA76DD3A6D9A2B200D94F691F7D3CF424E8`。既有Canvas/Rust/Vite警告保留，未跑独立app全部测试。原用户RPP保持SHA256 `4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3`。

最终鼠标笔画再检查：先点击原FX标题，再在画布drag；工具仍报告point over msedgewebview2.exe Chrome Legacy Window/not target reaper.exe。激活/新截图仅重试一次，同样拒绝，未写入新笔画。不绕过工具检查、不增加输入代理；真实鼠标手绘仍open，但键盘原GUI编辑/自动合成/保存重开已经实测，二者不能混为同一结果。

终态：正常保存/退出隔离REAPER，副本40656 bytes，归档gui-keyboard-edited.RPP（开发路径快照，不是可移植发行工程）。原始4份重复导出WAV校验与命名归档SHA一致后移到ignored `.build-tmp/embedded-transport-probe/raw-captures`，不删除；Git只stage明确命名证据。最终再执行当前同源测试exe，完整lib66/0失败/exit0。

一次集中review仅Important为单素材selectClipParamRange也需同样DOM focus，已补齐；无其它Critical/Important。多参数面板广播时最后监听器获焦点记Minor残余。真实鼠标手绘/双轨均编辑/资源矩阵/独立app实测仍未关闭，不能把此键盘正向链路当二期全部验收。
