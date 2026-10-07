# ARA 产品开发当前交接

## 当前工作位置与交付（2026-10-07，优先于下方历史记录）

用户允许修改主工作区，并要求合并 develop、增加 VST3 Actions 与快速打包支持，
随后移除当前工作树。代码已合并并推送：feature/ara-plugin 的源码/打包提交为
bf96b4d0 / 61b07319，主工作区 develop 的合并提交为 e834ef54；后续 CI 路径修正和
诊断增强已推送。当前在 `E:\code\HiFiShifter` 的 develop 工作，
`E:\code\HiFiShifter\.worktrees\ara-plugin` 已移除，不再使用旧目录。

最新本地统一交付迁入 `.build-tmp/deliveries/develop-vst3-packaging-06`，05 包和最新
VST 安装备份也已迁入主工作区；一次性诊断源码保存在 `.build-tmp/retired-ara-plugin`。
主工作区原有 `.dsh-plugin-inspect/` 保留。`dist` 现有完整 VST3 ZIP、NSIS setup.exe、
各自 SHA256 和同批 App 便携 ZIP。规范插件安装内容仍是完整 .vst3 目录；安装器默认
系统 VST3 目录，可改 D:\VST，且写入前检查 REAPER 和插件宿主已退出。

快速入口 `pack-portable.bat` / `scripts/pack-portable.ps1` 支持 PackageTarget
App/Plugin/All、SkipBuild、NoZip、Installer、指定 DeliveryDirectory；All 复用同批
App/插件交付，不混入插件或 App 专用运行库。说明见 docs/VST3-BUILD.md。

本地前端全量 365 文件/3260 项通过，插件全量 208 passed/2 ignored，lint 无错误。
本地 SDK 从空目录下载/身份校验及 ZIP/NSIS 生成、摘要和目录结构已验证。
GitHub 首次 frontend 失败为旧 state.rs 源码扫描路径，已改为共享 kernel model 并
复跑通过。云端插件测试又失败，正在通过新增日志/公开错误摘要定位；当前 run：
https://github.com/ARounder-183/HiFiShifter/actions/runs/37574960116 。
Actions 定义已经启用，云端首次完整成功尚未确认，不能写成 CI 已全部通过。

## 最新目标：延迟／插件私有参数分组／App与插件共用折叠

用户明确父子轨仅在插件内建立，不同步REAPER folder，新进轨道根级。本轮源码已接
TrackGroups GUID父级/排序、原App拖动命令、分组根参数delta向物理成员展开、根合成
算法/开关继承（含空父轨），组件v5存轻量共享关系、源atlas继续按renderer范围裁剪。
媒体GUI回执改为真实清单结构确认，不再等待ARA/PCM，host_audio_pending明确区分音频
未就绪；选中新占位不等物化，后台补音频保留选择。折叠按钮接入两模式共用ActionBar，
Dock隐藏/恢复Timeline、保证参数面板可见，不改session。
Rust定向3项及前端首轮16项/tsc通过，未做实际鼠标操作或真实延迟测量，统一包待构建。
完整方案、证据和未完成门见PRIVATE-GROUPS-AND-LATENCY-2026-10-07.md，取代旧folder评估。

## 最新：粘贴宿主成功但当前窗口未刷新（覆盖下方旧状态）

用户确认feedback-undo-copy-01能在REAPER创建新clip，HFS需重开UI才出现。新增清单独立
同步，在ensure_loaded模型缓存早退前处理inventory版本，不把结构通知依赖于ARA fade
投影成功；媒体写完强制重新枚举清单、结束native Undo请求，回执只读继续核对ARA音频。
同一session的inventory-only新增/删除、曲线/历史保留Rust用例通过，前端回流8项通过。
首轮Rust测试缺fake I_GROUPID字段，补齐后通过。feedback-paste-refresh-02 Release已构建，
确认REAPER退出后安装D:\VST，35文件SHA256一致；旧包可恢复备份
`.build-tmp/vst-install-backups/feedback-paste-refresh-02-5194dd09`，GUI尚未实测。
详情以CLIP-FEEDBACK-2026-10-07.md最新节为准，不重做创建、不用强制reload掩盖。

## 最新用户实测失败（取代下文旧“仅缺验收”）

参数Ctrl+Z、Ctrl拖动复制、clip复制粘贴仍不生效；父子轨拖动仅低优可行性评估。
详见CLIP-FEEDBACK-2026-10-07.md。已修源码：clipboard_kind识别clips、decode从
Take恢复源起点/倍率、接duplicate_clips_bulk宿主路径和前端读取代次；4项解码/
路由回归及tsc通过。通知Undo tick前flush dirty，并记录宿主支持/返回值。
未出新包，D:\VST仍editor-parity-05。参数Undo还未闭合，不重复声称只有验收缺口。

最新反馈源码批次补充：App的copy/cut/paste最终通道检查漏op，已修；参数面板Ctrl+Z/
Redo按焦点走HFS参数历史专用命令，只恢复参数与轨道控制不恢复clip几何，轨道/全局
仍宿主历史（替换旧全入口REAPER栈的语义，见CLIP-FEEDBACK）。分组曲线Undo/Redo
actor用例通过。Ctrl拖动duplicate已接native，同轨/映射/新轨span路径俱有；新轨不用
App add_track，原初始轨道索引错误clipId键改trackId。tsc/cargo check通过，未出包。

最新实际安装包为`feedback-undo-copy-01`：Release构建exit0，REAPER实际退出后35文件
整包部署D:\VST并逐项SHA256一致，旧05备份`.build-tmp/vst-install-backups/feedback-undo-copy-01`。
前三项源码/定向回归已修，但真实GUI新反馈验收尚未进行；父子轨仅评估，不修改folder。

## 最新目标：编辑问题排查 + clip操作恢复

用户把目标扩展为恢复clip绝大部分操作（包括复制粘贴）。参数面板右键正常，
轨道头允许禁用，轨道空白菜单必须保留。只恢复显示不算功能实现。
新设计/计划：`docs/superpowers/{specs,plans}/2026-10-06-ara-editor-parity*.md`。
四项排查详情在GUI-EDIT-DIAGNOSTICS.md；真实宿主Undo录入/回流仍未闭合。
空白菜单源码已恢复（6项定向前端测试），没有新Release或安装；粘贴/关闭间隙
暂时禁用，后续必须按新范围接通，不把临时门禁当最终交付。
原生item state捕获/GUID安全重建、创建/删除、结构化剪贴板与原GUI copy/cut/paste/
delete路由已连接源码；新clipClipboard能力门，旧插件不误发媒体命令。剪切先复制
成功再删除；粘贴保留相对位置/轨道关系，按新GUID保存曲线seed并等待ARA/GUI回流。
新增host_clipboard.rs，复制经过actor写入屏障但不在UI等待DSP；仍可能等待已运行
的同步DSP，需后续渲染解耦。媒体Undo请求在回流完成或错误之后才收尾。
3项parser/seed检查通过；新增2项解码测试修正fixture缺name后通过；前端13项/tsc
通过。未验证真实REAPER媒体API和完整GUI流程，没有新安装包。静音复制波形/冷恢复、
混合Undo、其它clip菜单和整轨曲线问题仍待做，不把这一批称全部完成。
当前REAPER6252仍开着，不热替换D:\VST。
后续按五批计划推进，不再重复已完成的短元音EOF或默认算法验收，不自动push。

### 2026-10-07 曲线批次 2A

ParameterAtlas新增按项目绝对时间保存的稀疏空白层gaps，clip仍使用源basis。
显示合并仅复制真正覆盖的网格点；源曲线首尾外侧哨兵夹到真实边界，避免默认零。
新增连续60/非网格边界、移动后空白保留/原clip不留假源曲线、旧布局迁移检查。
state scope过滤/根映射包含gaps，旧v3/v4缺字段可读；state encode新增传输预算检查。
插件同scope paramsEpoch刷新不先清空曲线，切参数/轨道与独立App维持原规则。
Atlas模块首轮11项通过、1项恢复测试漏真实rebind步骤；补上rebind后3个新增gap用例
通过，前端刷新策略/笔画推迟9项及tsc通过。未出包或实机验收。
同步DSP仍在actor内：参数读取和媒体屏障可能等待渲染，下一步需解耦。完全空轨与
静音素材的编辑/冷恢复矩阵仍未闭合，不能把此批称完整整轨生命周期已验证。

### 2026-10-07 曲线批次 2B：DSP与actor解耦

自动apply移到单个hfs-editor-dsp任务；原线缓存组装仍在actor，DSP线程只持文档/
版本票据，完成结果由actor收取。新写入/历史/宿主变化取消旧任务，进度和发布检查
ticket/generation/edit/model/epoch/scope；关闭先取消，再join actor与DSP，不遗留卸载线程。
原同步apply仅保留cfg(test)供已有诊断使用。2项受控调度/取消、1项历史屏障、2项
实际PCM与持续只读请求检查通过。最后补了瞬时superseded静默重试、启动失败结束
progress及state.pending包含DSP任务；这些小收尾分支随最终统一构建验证，不复跑全套。
未出包、未进行真实GUI神经渲染或宿主Undo验收。下一步统一历史/其余clip操作。

### 2026-10-07 撤销批次 3A

修复 `editor/commands.rs` 的分组入口提前return：现在begin_undo_group会进入已加载
分支并写入真实本地基线checkpoint，组内操作由suppress_history合并；end只关闭组。
分组/非分组本地undoDepth与state往返定向检查通过。REAPER宿主Undo写块仍走
HostUndo；IEditController::setComponentHandler尚未保存handler/调用IComponentHandler2
setDirty，真实宿主历史录入和冷恢复仍未闭合，未出包。

## 2026-10-06 当前待安装批次：元音EOF / 分割 / clip音量 / HiFiGAN默认

最新用户目标是排查丢音、插件分割、clip音量；另明确新默认算法用HiFiGAN。
用户现场确认ka/n是元音；独立App粘贴有声，旁路HFS FX有声，启用无声。
已定位并复现完整ARA RenderInput的源尾门禁拒绝：ka 8073帧却窗口约8074，n 8084帧却
窗口约8085。按源网格round允许最多一帧零尾，起点在EOF/越界更多仍拒绝，几何不改。
直接kernel诊断绕过该门禁，因此之前“kernel有声”不足以排除插件问题。

本批已实现S/精简右键/多选分割，原生item两GUID+源域参数继承，Undo仍走REAPER；
clip增益徽章拖动/双击/多选写item D_VOL，读真实item/take音量，take极性不改，GUI
gain不交给kernel二次烘焙。新宿主轨道默认HiFiGAN，已有明确覆盖仍通过edits恢复。
Snapshot新增可选只读diagnostics，含准备错误/版本和前8clip的实际发布PCM RMS/峰值。

统一Release包：`.build-tmp/deliveries/vowel-eof-split-gain-01/HiFiShifter.vst3`，构建正常。
sourceFingerprint=`77F890FA0790347EA72474AC178A19EF838237E9905F76094766E90DA1F9C346`。
EOF2项、gain2项、split3项、默认算法1项、前端13项及tsc通过；不跑全量/review。
当前REAPER PID19916仍运行且工程未保存，**尚未安装D:\VST**，不热替换。已请求用户
保存并完全退出；退出后用ignored install-sync-fix.ps1部署整包并核验SHA256/保留备份。
manifest nativeAcceptance=false。仍需一次真实ka/n有声、分割Undo/Redo、gain声音只施加一次
及保存重开验收，goal不能标complete。

本机MCP普通TEMP写入/锁权限被拒。固定Python服务端tempfile到Lua TEMP能避免错误地
回落cwd，但ipc.mutex依然拒绝；没有改ACL或删锁。用户普通终端可运行只读
`probe/ara/live_ka_diagnostic.py`。当前现场只读读取用既有认证插件Snapshot，明确不是MCP。
重启后capture_live_ka.py要传实际REAPER PID；旧19916仅为默认一次性目标。
这次错误TEMP落在worktree产生的两小文件已移至ignored failed-mcp-root-20261006，可恢复。
详情见SHORT-CLIP-FINDINGS.md及新增split design末节。源码未push，原有未跟踪文件未动。

## 2026-10-06 当前最终源码与安装包（覆盖下方历史状态）

本批合并静音clip/空轨独立GUI清单、GUID参数归属、250ms兜底同步、live空轨曲线
保留、真实目录选择/只读浏览和原生File拖入；另修mute已有显示波形丢失、导入旧
读取覆盖新clip及seek ABAB。新spec/plan为2026-10-06-ara-ui-file-import-fixes.md。
波形缓存仅GUI：按item/take保留已授权生成的元信息/私有路径，不送入DSP/分析；
take更换/删除失效，冷启从未供源的静音clip仍不能直接读原文件造波形。

最终包`.build-tmp/deliveries/waveform-import-seek-fix-01/HiFiShifter.vst3`构建exit0，
sourceFingerprint=`EB999C760AE56224187B654FA82F111141FADA9F95B2C3C1903CF316CEC334FD`。
当前实际无REAPER后已安装D:\VST\HiFiShifter.vst3，35文件SHA256一致；旧包备份于
`.build-tmp/vst-install-backups/waveform-import-seek-fix-01-5a7bb330`。
本批目录授权1项、GUI清单/身份/空成员3项、工作区9项、mute显示波形1项、前端乱序6项
及tsc通过，不运行新全量/review。01中间包实机见两段灰色mute clip/空第三轨，同时
发现两项额外错误后修正。完整GUI仍未验收，manifest nativeAcceptance=false。
此前CU被物理Esc中断，停止该轮输入；不将源码检查或安装当交互验收通过。

用户询问以后维护App/插件：主体同源frontend与hifishifter-kernel，两种宿主外壳及
文件/设备/走带接口分别适配；本批改动均plugin能力分支或plugin crate，独立App保留。
普通共享新功能通常只改一份，但必须检查插件能力限制，不能承诺所有App功能自动开放。
用户要求设新目标，create_goal因旧未完成目标被拒绝；未为重建目标伪报旧goal complete。

中文工作记录，更新于 2026-10-05。此文件记录产品分支，历史探针仍见 HANDOFF.md。

## 2026-10-06 用户最新部署要求（覆盖下文历史“不安装”约束）

以后每次插件打包完成，都把整个 `HiFiShifter.vst3` 同步到 `D:\VST\HiFiShifter.vst3`。
先确认 REAPER 已退出；运行时不热替换，保留新包待正常退出后安装。替换前将旧版备份
到此 worktree 的 ignored `.build-tmp/vst-install-backups/`，不动 `D:\VST` 的其它插件，
不自动 push。2026-10-06 用户已确认 Ctrl+V/默认弯曲渐变这一批“ok的，我测过了”；
交付为 `.build-tmp/deliveries/keyboard-fade-fix-01/HiFiShifter.vst3`。渐变只改显示，
短辅音静音问题仍按用户要求暂不处理。

## 2026-10-06 最新修正：移动闪回与渐变fallback

用户报告HFS移动A→B会闪回A再回B，随后允许渐变声音走REAPER，但HFS渐变区宽度
必须同时写入REAPER。覆盖下方旧“实际HFS渐变”的声音归属要求，停止包络补偿研究。
当前修复为前端交互/写请求读取代次守卫、原生确切clip几何回流完成门；宽度patch
带未改shape时去除冗余字段。插件自定义形状/曲率暂关闭，弯曲示意与独立App保持。
最终同源Release仍安装D:\VST，真实验证与源码合同分开记录，不自动push。

本批最终包为`.build-tmp/deliveries/bidirectional-sync-fix-03/HiFiShifter.vst3`，Release构建
exit0，sourceFingerprint=`5BEC84084B0FDB4BEBBDD0C2B9B07A65D0A49B3F140FB794F8C03299BB4FDB23`。
已在REAPER正常退出后安装D:\VST\HiFiShifter.vst3，35文件数量与manifest SHA256全部
一致。旧包可恢复于`.build-tmp/vst-install-backups/bidirectional-sync-fix-03-01513c3a`。
实机03冷加载/count API/REAPER→HFS淡出0.25→0.75自动同步通过；nativeAcceptance仍false，
鼠标移动及反向宽度被CU的WebView2子进程保护拦截（已按指示点标题/激活/刷新重试一次）。
完整矩阵仍待用户集中验收，不再反复自动跑测试。详情BIDIRECTIONAL-SYNC-FINDINGS.md。

续批最新安装为`.build-tmp/deliveries/bidirectional-completion-02/HiFiShifter.vst3`，补齐
非零吸附偏移回流与裁切源窗口/倍率旧请求预检；35文件SHA256一致，上一包备份于
`.build-tmp/vst-install-backups/bidirectional-completion-02-0468ed66`。原生host_edit9项及
负偏移保留后的单项护栏通过。真实隔离33144关闭全部GUI后的未静音/静音/solo覆盖/
解除四份PCM通过，静音全0，后两份与基线逐样本一致；实例已正常退出。
这仅闭合离线有效item mute输出门，不声称实时/UI/完整双向GUI验收。最终仍需一次
原GUI导入→移动/裁切→线性拉伸→渐变宽度→混合Undo/Redo→保存冷重开验证，不重复
自动全量测试或鼠标工具重试。manifest nativeAcceptance仍false，目标保持active。

最终审计见BIDIRECTIONAL-COMPLETION-AUDIT.md：同一GUI交互工具阻塞已连续三轮，
现无可替代完整原GUI验收的安全自动化路径，将目标标blocked而非complete，等用户
一次集中验收结果。已开临时双轨工程user-final-bidirectional/embedded-editor.RPP，
PID11212，包与安装版逐文件相同，原GUI显示双轨/已应用；保留实例，不继续发脚本。
REAPER更新提示可能需用户先关闭。启动器单路径检查原会把string[0]当首字符，
已修为外层数组并实际启动成功；只改probe工具，不改产品包或再次打包。

## 2026-10-06 新活跃目标：双向片段编辑与实际HFS渐变（历史声音要求）

用户要求在HFS导入/编辑/线性拉伸/渐变clip并同步REAPER，确认REAPER item mute应
同步；进一步指定REAPER只保留fade长度和默认方式，自定义渐变形状与声音由HFS负责。
新权威为docs/superpowers/specs/2026-10-06-ara-bidirectional-editing-design.md及同名
plans下ara-bidirectional-editing.md。旧几何只读/渐变只显示不再是目标边界；先验证
ARA渐变委托以避免双重淡化，不清零宿主fade长度或反算未知包络冒充实现。

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

2026-10-07撤销补充：分组checkpoint提前return已修复，分组/非分组本地历史与state
往返回归通过；宿主handler桥已保存IComponentHandler2引用，actor置dirty、UI timer
调用setDirty并在terminate/drop释放；真实REAPER历史条目与Undo后setState冷恢复仍
未闭合。本轮 focused frontend 16项、tsc、plugin cargo check通过，未打包或安装。

2026-10-07 `editor-parity-01` Release 构建成功，完整35文件部署到 `D:\VST\HiFiShifter.vst3`，
旧包备份在 `.build-tmp/vst-install-backups/editor-parity-01`，逐文件 SHA256 零差异。
隔离 REAPER smoke PID36088 扫描/加载插件并输出 GetPluginFactory、3 classes、host
extension available 日志，随后已关闭；这只证明包能被扫描加载，不是GUI交互或Undo验收。
随后同包隔离实例可见打开VST3 HiFiShifter窗口，UI树确认菜单、播放控制、两段clip
时间轴和nsf-hifigan；WebView子区域右键输入被Computer Use目标保护拒绝，按规则停止
重试，故未宣称空白/clip菜单实机通过。隔离PID20844已关闭。

2026-10-07 `editor-parity-02` 已重新 Release 构建并部署到 `D:\VST\HiFiShifter.vst3`，
包含 REAPER I_GROUPID 编组/解组桥；旧包备份在 `.build-tmp/vst-install-backups/editor-parity-02`，
35文件 SHA256 零差异。该包尚未重新做交互验收。

2026-10-07 `editor-parity-03` 又重新构建并部署，包含关闭间隙到宿主 `move_clips` 的桥；
旧包备份在 `.build-tmp/vst-install-backups/editor-parity-03`，35文件 SHA256 零差异。

随后修正编组回流receipt等待 `group_id` 后，构建 `editor-parity-04` 并重新部署；旧包备份
在 `.build-tmp/vst-install-backups/editor-parity-04`，35文件 SHA256 零差异。

最新 `editor-parity-05` 增加 active Take 重命名（REAPER P_NAME setter），重新构建并部署
到 `D:\VST\HiFiShifter.vst3`；旧包备份在 `.build-tmp/vst-install-backups/editor-parity-05`，
35文件 SHA256 零差异。
