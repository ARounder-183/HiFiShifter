# ARA 产品开发当前交接

中文工作记录，更新于 2026-10-05。此文件记录产品分支，历史探针仍见 HANDOFF.md。

## 2026-10-06 用户最新部署要求（覆盖下文历史“不安装”约束）

以后每次插件打包完成，都把整个 `HiFiShifter.vst3` 同步到 `D:\VST\HiFiShifter.vst3`。
先确认 REAPER 已退出；运行时不热替换，保留新包待正常退出后安装。替换前将旧版备份
到此 worktree 的 ignored `.build-tmp/vst-install-backups/`，不动 `D:\VST` 的其它插件，
不自动 push。2026-10-06 用户已确认 Ctrl+V/默认弯曲渐变这一批“ok的，我测过了”；
交付为 `.build-tmp/deliveries/keyboard-fade-fix-01/HiFiShifter.vst3`。渐变只改显示，
短辅音静音问题仍按用户要求暂不处理。

## 当前续接：四项目标仍未全部完成

当前权威是`docs/superpowers/specs/2026-10-05-ara-complete-integration-design.md`及同名plan。
下方单实例/旧unsupported/旧构建段落为历史，不覆盖新范围：用户最新明确只做线性拉伸，
固定倍率/tempo-timebase整段线性变化/曲线重投影必达，非线性markers/坡度/段内tempo warp
不再是完成门；倒放暂缓。保留独立App，同工程共享多轨原GUI和逐轨输出，
HiFiGAN缓存与正常长人声也在最终门中。后续仅最终一次review。

`ffa8266a`完成有界异步准备/冷恢复与原子发布；`ddbb5a97`完成editor角色透传、typed
REAPER所属project时钟和正向线性保调路径，plugin lib80通过。新规范release bundle
`.build-tmp/embedded-host-stretch-01/HiFiShifter.vst3`构建exit0，引擎SHA256
`2F4FD9EC8B443DBA8A7225A740EB5506C2EC7071AEBBB2AC880438044D951C38`，尚未真实加载验收。
`dfe8f4e`是Task32授权工作区基础（两真实scope回归exit0）；`2cdf48ff`已接文档唯一原
actor/跨轨history/真实queued租约授权/全局solo独立输出/基础doc关闭与共享flush。
批末plugin lib89、相关frontend74、tsc与实际document apply定向回归均正常exit0；Task34
在`a5dfa654`完成12定向源码门：旧RPP原v2 bytes、实际COM/Weak/stream、关窗尾笔及undo保存、
未开GUI双轨WORLD后台publisher冷恢复（A60≈260.947/B67≈390.265，undo后B64≈329.104Hz）。
本批修即时组件撤销、view移除后尾笔被丢、旧actor缓存滞留，v2形状未改。
Task38a在6b8fb67c/b5572a4e完成typed直接take/逐getter重入授权、project counter稳定检查、
独立transport/geometry API及短COM初始化保活/终止失败清理（19新定向+6旧相关exit0）。
工厂只广告线性TIMESTRETCH，REFLECT_TEMPO/CONTENT_FADES已撤回，不等于实现二者。
fe731313有秒域正逆/分段映射基础，4回归exit0；随后单GUI入口驱动隐藏playback元数据，
稳定project/model/scope缓存避免每tick读全marker，变化/关闭撤销回归共11/exit0。
本批ParameterAtlas已接原接受事务/模型稳定投影/每clip独立kernel输入/v3限定范围保存恢复，
源basis不因宿主移动/拉伸反复回写插值，原GUI仍用项目网格。6新回归RED→GREEN；批末
128首轮123/5失败，修正后的失败5、后续8定向、旧归档v2各正常exit0，未重跑新全量。
不含atlas仍v2；含basis为v3，新state不可给旧引擎读。raw marker保留诊断但本轮不实现非线性。
重叠区域GUI编辑选择、跨不同算法轨移动、旧v2首载后移动、实际source内容换源/基线失效、
拆分/冷恢复完整PCM与原GUI真机组合门仍需收尾；当前不能宣称完整线性支持交付。
新源改动不包含在上述bundle里，不能混用证据，native多轨未验收。
短actor filter曾摘要后退出挂住，完整lib正常exit；确切退出根因未知，未通过禁用模型或
强杀冒充通过。自有测试PID/候选FCPE链和生命周期证据在Task33本地report，历史49800未碰。

完整marker/fade/tempo接口源码调查见`HOST-GEOMETRY-FINDINGS.md`；标准ARA缺普通fade/
marker数组，REAPER直接take绑定尚未取得实测。原kernel stretch_markers目前只保存未消费，
轨级曲线缺源坐标锚点；当前runtime仅广告TIMESTRETCH，旧已构建bundle仍是历史版本。
完整拉伸/共享GUI/HiFiGAN缓存/长源/独立App/native最终门全部仍open；元数据缓存不是神经缓存。

用户此前物理Esc停止Computer Use，本次只继续源码，不自动恢复CU，不启动/关闭用户应用。
不强杀、不热替换，不覆盖已有用户/测试工程；全程只ara-plugin worktree，本地提交不push。

## 最新状态：一期核心真实验收通过，二期改为内嵌GUI与自动应用

Task29当前：插件初始化空宿主态，不再请求虚构track_main；标准Win32/WebView2焦点进入原HTML，Space实测控制宿主播放/主动暂停，秒位置一致2.414。素材选区转参数画布补DOM焦点后，真实原Ctrl+0对话框输入MIDI64并确认，第二轨自动1/1、第一轨0/0。真实独奏导出四窗口220.5→329.104Hz、gap0；关闭两个FX窗口后PCM maxdiff0，正常退出REAPER冷重开40653-byte测试RPP后、第二轨GUI尚未打开时供音PCM同样maxdiff0，之后正常打开FX显示恢复曲线。无外部HiFiShifter进程、没有手工提交或脚本写pitch。最终前端312文件/2718测试及tsc/生产bundle exit0。证据captures/gui-keyboard-*与gui-host-*-keyboard.jpg；旧“无法网页操作”仅剩鼠标工具命中限制，不再代表所有GUI输入都无法验收。真实鼠标手绘、双轨均编辑及资源/独立app验收仍未闭合。

本批最终前端全量311 files/2715 tests通过(exit0)。隔离副本已保存8356 bytes：BPM180、Time基准、两轨各两段POSITION1/3、PLAYRATE1；真实Duplicate tracks后两个GUI都显示已应用，截图copied-track-two-editors.jpg。模型过渡仍有两条get_param_frames unknown host track日志，未定位；双轨真实曲线与PCM验收仍open，不以显示通过代替编辑通过。

复现入口：`.\probe\ara\start_embedded_editor.ps1 -Reopen -ScratchName embedded-transport-probe`，仅在REAPER正常退出后启动。默认embedded-probe仍是原用户测试工程，SHA256保持4AD35908AA2D252D9171A9B6F423E4D9BFEE4DEDBE29861B7665B0FE485897A3，不覆盖它。

最终测试副本重新保存为40656 bytes，并归档`captures/gui-keyboard-edited.RPP`（含绝对开发路径，仅一次性证据）。原始重复导出4份WAV已按SHA一致性校验后移到`.build-tmp/embedded-transport-probe/raw-captures`，没有删除；可评审的命名WAV仍在captures。隔离REAPER已正常退出，未push。下一批继续未闭合验收，不标整个目标完成。

Task28新增：标准ARA宿主播放租约/原ActionBar能力开放、模型自动同步、有效宿主tempo同步已实现；plugin lib65/租约1/前端18/tsc与build通过。真实UI在独立副本中将BPM120改150，HiFiShifter显示150；副本Time基准下PLAYRATE1保持，截图captures/bpm-sync-150.jpg。进一步在GUI不关闭时150→180立即同步；宿主单素材属性起点0→1秒后原时间线和参数区自动跟随，未点重载，日志clipStartsSec=[1,3]，见bpm-sync-live-180.jpg及host-geometry-auto-refresh.jpg。原工程hash未变。按钮真实点击虽先确认FX标题栏焦点，仍被Computer Use跨进程检查拒绝，键盘Tab未确认触发网页控件，不能标播放控制/全流程验收通过。最新构建回到embedded-vst3；自动同步有pending时仍保留冲突，完整Tempo Map/time stretch未支持。

2026-10-05用户已真实试用内嵌GUI并确认基本可编辑；新反馈三项高优缺陷正在修复，不再将所有未验证行为归为Computer Use。当前新源码修停播process发声、绝对时间重复相加、同源复制轨道的组件state作用域和跨轨假Conflict。62插件lib回归与15前端光标回归正常exit0；新bundle在`.build-tmp/embedded-feedback-01/HiFiShifter.vst3`。用户仍在修改旧隔离REAPER，绝不关闭或热替换。下一批先待用户正常保存退出后部署高优修复，再实现标准ARA宿主播放控制（可选能力/主线程调用）和无待应用编辑时的宿主模型自动同步；真实验证未完成，不能标二期完成。

原GUI手绘音高提交r1/m2，REAPER真实输出四个窗口220.5Hz→393.75Hz，gap=0。
正常关GUI、保存/关REAPER后无GUI重开，输出PCM maxdiff=0；恢复GUI连接r2/m3，
用户确认曲线显示。证据见`captures/forward-gui-output.json`和同名三份WAV。
首次重开126加载失败已定位flat模块依赖搜索受项目CWD影响，启动脚本改为进程局部PATH；
失败WAV保留。完整规范bundle、同路径改源/seek/采样率声道边界尚未全部宿主复测。

用户新目标是整个原GUI嵌入REAPER插件、不再另开app或手动提交，同时保留独立app。
当前源码`IEditController::createView`仍返回null；一期成果不能证明二期已完成。
二期规格/计划为`2026-10-05-ara-embedded-editor`，替代一期A5中“禁止任何WebView”的
UI边界，仅允许插件原生WebView2，不引入Tauri app/事件循环/设备音频。下文旧状态保留
历史排错意义，最新状态以本节为准。

本轮起点只读进程检查无REAPER主进程/HiFiShifter，仍有两个reaper_host32辅助进程；
未终止它们。所有变更仍仅在ara-plugin worktree，不push。用户要求集中最终测试，
不逐小改动重跑验证；必要编译检查不冒称行为验收。

二期当前源码检查点：plugin原生IPlugView/WebView2（createView不再null）、前端pluginHost
命令与hostEvents事件适配、独立app原params完整实现迁入kernel/editor，AppState薄hook。
native cargo check --tests exit0（含CSP/token/防同步重入和参数迁移），前端通信迁移
tsc-b exit0，迁移后独立app cargo check exit0。最新模块pin增量未重新编译，新增测试未运行，不能宣称宿主
已经显示或能用。WebView当前只接受ping/日志，实际编辑请求明确报session未绑定。
下一步Task22：正确processor/controller连接、实例会话、宿主PCM分析/波形及原API分发；
随后原App能力适配、自动应用与state保存、bundle部署、最后集中REAPER实测。
前端入口是plugin.html，bundle必须带它（不是独立app的index.html）；暂未生成发行包。

二期最新源进度覆盖上一段“仅ping”的旧检查点：已用标准IConnectionPoint/宿主IMessage
建立真实processor路线，拒绝PID/过期令牌；原生窗口排队给实例EditorSession actor，
原参数/原波形与能力描述符/共享history已接线。150ms自动渲染及状态事件、保存尾块屏障
源码已写，已接受参数先入组件state，音频稍后发布；process仅额外原子写宿主时钟。
native字段不保留Tauri app句柄、不运行设备。UI响应由窗口timer主线程回传COM。
新增路由/真实actor-state编码/关UI尾块/无效track patch回归已编译，未执行（遵用户要求）。
最新plugin cargo check --tests exit0（含模块pin、clock/event router、actor回归），
独立app cargo check exit0（共享波形/PCM/history后，随后仅去掉新增unused导入/变量）。
还没frontend能力适配、部署/REAPER内嵌真实操作与音频验收。不要让“编译成功”替代这些。
未决重点：retired snapshot预算下长期自动编辑、安全回收；doc transaction持有期间的
宿主UI等待；私有PCM文件生命周期。下一步先读二期plan末尾进度并执行Task23/25，
最终集中测试与REAPER验收。当前未启动REAPER或用户app，未改用户工程。

## 一期历史排错记录

最新集中回归/接线：原系统剪贴板搬到共享hifishifter-clipboard，纯搜索转写到kernel/search；
插件参数clipboard/transliterate真实实现，app原入口再导出。native算法Unknown的serde兼容
值必须在命令边界拒绝，原失败用例已修。完整plugin75、kernel editor12/search39/peaks7
正常exit0；独立app cargo check正常exit0。前端全量首次2698/12fail，定向修复六文件61
正常通过，tsc通过；最新全量未重跑，不能报告全前端绿。SDK/main仍未改，没有push。
当前仍未取得手绘自动pitch和重开PCM输出。Sky reset后对最新FX index1 Raise仍报无cache，
screenshots有前景Codex不一致；只读InputDesktop为Default，不能说桌面锁了。
已走原有一次性Lua save，RPP5056 bytes且log project saved；REAPER46132继续运行。
源修改未部署旧窗口，不强杀/热替换；后续先处理可靠FX输入或正常关闭后升级，然后真正
原参数编辑器绘制/自动应用/导出。不要把UI helper问题写成“ARA无法实现”。

二期最新实测（覆盖上文“未部署”历史）：已构建规范bundle，薄Win32入口以绝对路径/
DLL_LOAD_DIR加载同进程Rust engine，项目CWD启动且不加PATH。隔离REAPER PID46132已
加载并ARA绑定，三组标准消息关联result0；createView返回真实native editor。REAPER
accessibility的FX对话框下明确存在原GUI菜单/时间线/参数编辑器/WORLD和自动应用栏。
没有HiFiShifter.exe。只取得显示/关联与未编辑baseline，不是自动修音/重开验收。
当前REAPER一次性实例仍运行，有评估/About窗口；不要重建替换正在加载的bundle或强杀。
新入口为build/start_embedded_editor.ps1，scratch `.build-tmp/embedded-probe`。
FINDINGS见captures/embedded-editor-FINDINGS.md。源码额外快照回收与时间轴剪贴板轮询
修正尚未部署、回归未运行；参数系统剪贴板/转写当前日志仍Unsupported。

最新源码已补齐插件模型登记和实例级可取消分析worker；真实actor的自动WORLD回归在“先画曲线、分析完成后自动应用、无GUI轮询”路径测得329.104Hz（MIDI64目标329.63Hz），缺分析时保持pending。新bundle已重建并以隔离RPP重开，原GUI层级和FCPE DirectML日志存在。Computer Use对WebView子窗口drag返回目标窗口不匹配，未用脚本伪造手绘；真实手绘、导出、重开PCM仍未关闭。

用户明确要求本轮先不做倒放，继续到原 GUI 正向全流程可用。此授权覆盖下方历史方向停止条件，
不代表倒放已修复。当前规格/计划为 `2026-10-04-ara-gui-forward`。

工作树仍是 `E:/code/HiFiShifter/.worktrees/ara-plugin`，`codex/ara-plugin`。
Task16 PCM注入口提交 c836bd65；Task18 原GUI客户端及嵌入dist/Low临时目录修正为
82942a25、49bda828、6da16345。Task17 plugin/IPC基础检查点为4800705。
最新修复为8eba3ae5/0b8f4dd4，中文doc8c6cb394。controller新鲜验证plugin68/IPC4/
kernel28/app ARA18/params10/project21/frontend8通过；前端生产构建也通过。
四项集中审查已修，首轮窄复审仅F2的false-checkpoint并发dirty留P1，现已定向修正，
最终窄复审F2/R1为ADDRESSED，无新问题。最终独立target GUI构建38.55秒正常exit0，
产物92661760 bytes。真实导出/重开仍未完成，不能称完整链路验收通过。

本机WebView2创建0x800700AA：隔离WEBVIEW2_USER_DATA_FOLDER后实际GUI正常显示；
独立debug app需要custom-protocol feature。Low GUI到Medium REAPER的管道已仅在本应用
对象设置Low标签，保留同用户DACL/token/remote拒绝；大PCM必须16KiB分块，回归已通过。
未改系统设置或用户原profile，未停止其他应用。

真实GUI已下载宿主PCM并显示两片段及原音高编辑器。用户截图记录手绘曲线提交时
`Conflict: host model changed; refresh`；最新成功Snapshot之后仅有samples_access=false，
而clear_renderers无条件revision++，已定位为访问开关与模型版本混淆。已分离版本，
实际授权回调/同快照提交/真实源改变拒绝均有回归。相关报告在本plan的SDD目录。
随后用户自行操作产生真实`GUI commit ready revision=1 model=8`；这仅证明一次提交被接受，
未作REAPER音频导出/重开比对，不能代替修音验收。

用户用物理Escape停止了Computer Use。当前GUI/隔离REAPER仍运行且可能有用户未保存曲线，
不得为重建/部署直接终止它们、刷新覆盖或发送脚本到已有实例。新GUI只在独立target构建，
startup脚本在当前窗口还活着时明确拒绝部署启动。最终验收仍待继续。
旧开发版非空v1 state没有稳定轨道身份，新v2明确拒绝猜测迁移；保留现有GUI编辑。

## 历史检查点：完整 v1 被真实倒放输出阻塞

Task13真实controller身份/销毁、editor sequence通知与展开已补齐。Task14已实现普通/
裁切的host PCM -> VST3 process链路，绝不按persistentID读源文件。真实REAPER输出
正常/裁切maxdiff=5.960464e-8、间隙0；官方倒放action/section已确认，最终输出仍为正向，
反向oracle差0.5003815。遵照Task15停止条件，**不进入Phase3b/4/5，不宣称完整修音/GUI联动**。
下一轮先读 `captures/phase3a-FINDINGS.md`，旧“下一步按顺序”部分仅作历史，不能跳过方向阻塞。

用户要求减少review，本批只一次集中审查；两个P2（96k协商/sequence迁移）已补红绿回归。
验证器6条正确/变异测试通过，真实倒放采集依旧应FAIL。源/快照共享512MiB硬预算，retired
快照保留到owner释放；time stretch/fades先导明确拒绝。日志/源码/输出WAV/JSON/截图均保留。
真实host PCM首轮缺ARA绑定是误将controllerRef当instance，已按锁定header/shim纠正，
旧失败产物保留 `.build-tmp`。隔离REAPER已关闭。main workspace不改，SDK不改，不push。

最终验证：构建成功，55条插件测试通过，两套验证器回归各6条通过，diff检查通过。
真实输出验证仍exit 1（倒放输出为正向）；这不是完整Phase3a/v1通过。内核/app全套
测试本批未重跑。后续必须先收敛方向契约，不能直接执行下方历史待办。

## 工作位置与授权

- 仅 `E:\code\HiFiShifter\.worktrees\ara-plugin`，分支 `codex/ara-plugin`。
- 用户授权按建议持续分批推进，无需中途选择执行方式；不 push，不 git add -A。
- 主工作区 develop、用户 REAPER 工程均不改；SDK 检出不修改。
- 当前 v1 是原本独立 app 图形界面联动插件，不是把完整 Tauri UI 嵌入 REAPER FX 窗口。

## 已完成

Phase 1 已抽内核并保持 app/内核测试合计。Phase 2 Task 10/11 提交 `a9d80b06`：
真实 REAPER UI 移动/切片、实际拉伸、官方 action 41051 倒放并读 ARA PCM；原始证据
已进 git（两个 *.log 用 explicit force-stage）。U1 仍是当前映射的方向表达缺口。

Task 12 提交 `e36889be`：完整 VST3 音频 ABI 原生布局 oracle、安全输出初始化、非法
输入/格式/布局拒绝、process/setProcessing 日志守卫。当前 process 只输出零。
上位设计已更正不存在的 storeAudioSourceContent，head/tail 查询不是预渲染调度。

Task 13 的部分检查点：RegionOwners、真实 region key 注册/撤销、扩展 add/remove
通知、entry builder Arc owner（无 Box::leak）、Processor 工厂初始 COM 引用修复。
完整插件构建及 38 测试通过；采集验证器 6 通过。审查无部分检查点阻塞，**并未批准
Task 13 全部完成或产品发布**。新 DLL 尚未部署到 REAPER。

## 下一步，按顺序

1. 阅读 Phase 3a spec/plan 的 Task 13 “当前部分检查点”。实际 destroy_document 只撤销
   RegionOwners，没有通知 ExtensionControllerLease 或清空 ExtensionOwner.assignments。
   必须找真实 controller -> document -> owner 的关联，不使用“最后一个文档”猜测。
2. 补 bound-extension teardown、observer 重入，以及 editor sequence assignment 通知。
   独立 FFI teardown 测试销毁手动 lease，不能证明产品接线；native entry lifetime
   测试目前是不带文档绑定的 entry。
3. Task 13 完成后才做 Task 14：scope 内宿主 PCM、不可变快照、项目 sample 时间读取。
   不直接按 persistentID 打开文件，不在 process 读 host reader/推理/IO/等待锁。
4. Task 15 在隔离 REAPER 输出波形验证普通/裁切/seek 与真实倒放；未测输出不关闭 U1。
5. 其后 Phase 3b 完整内核注入式修音与供音/cache 风险，Phase 4 原 app 参数通道及持久化，
   Phase 5 打包。不要宣称插件已经能修音或 GUI 联动。

## 环境与验证

每个 PowerShell 独立构建环境：点源 tools/msvc-env.ps1，之后设置 TEMP/TMP 到
.build-tmp/cl；两 SDK 变量指向 probe/ara/rust-path/.third-party 下对应检出。
全套命令见 Phase 3a plan；cargo 均 --offline --jobs 1。
产物为 backend/target/debug/hifishifter_plugin.dll，不是 crate 自己的 target 目录。

最后已停止 task10-clean 隔离 REAPER。重启前检查进程与命令行；不向已有用户实例发
脚本，不使用 -nonewinst。REAPER 会补扫系统 VST3；用已有隔离扫描缓存避免激活弹窗。
38 条插件测试与 6 条验证脚本回归无失败；本批不改 kernel/app，没重跑其完整测试，
四条既有 Windows /tmp 失败不修。既有未用 macro/mut/sequence index 警告保留。
