# 原GUI编辑问题排查（2026-10-06）

最新目标仅为排查Ctrl+Z、轨道面板右键菜单、绘制后曲线暂时消失、连续曲线尖跳/断裂。
随后用户把目标扩展为第5项：恢复clip绝大部分操作，包括复制粘贴。四项排查证据
保留，新增实现按docs/superpowers/specs/2026-10-06-ara-editor-parity-design.md与
对应plan推进，不再以“菜单显示出来”作为操作恢复完成标准。
用户已澄清参数面板没有问题；右键范围为轨道头、轨道空白区和clip。
用户最新允许轨道头禁用，要求保留空白区菜单；轨道头不再计为待修缺陷。
前期仅排查；该澄清后已单独恢复空白菜单源码，未打包或替换当前安装版本。
上一批EOF/分割/clip gain/默认HiFiGAN
用户已实测通过；用户随后指出Undo及本文件的问题，不将它们计作已修。

## 1. Ctrl+Z只撤REAPER，不撤HFS编辑

实测（用户）：快捷键能撤销REAPER操作，但HFS参数及clip调整没有撤销。
源码确认：App把快捷键派发到undoRemote；native WebView在有project_history_host时
强制调用REAPER历史，绕过actor本地history。故不是简单“Ctrl+Z没识别”。

早期源码缺口：vst3.rs的edit_controller_set_component_handler曾完全忽略this/handler，
直接返回K_RESULT_OK。现已改为查询并保留IComponentHandler2引用，dirty由actor置位、
UI timer调用setDirty；官方ivsteditcontroller.h明确该接口为UI-thread/Connected的
私有非自动化参数状态通知。现在仍不能用fixture证明真实REAPER已经将HFS私有编辑
录入FX历史，真实冷恢复待验收。

仍需确认：具体状态记录/回流失败点，以及纯宿主几何/gain调整的Undo组是否收尾。
不是把所有Undo错误都凭空归于setDirty；接口缺口是源码事实，真实host效果尚未闭合。

排除项：flush的Barrier虽然只排FIFO，但ParamHost.publish_timeline已经立即调用
accept_workspace_edits保存文档控制数据。因此不能把“必须等音频渲染才保存参数”作为
Undo根因，也不能把此前的屏障怀疑当已证结论。

补充实测（一次性真实actor分派对照，不是宿主历史验收）：commands.rs前置的
begin_undo_group先设置suppress_history并return，后面的同名checkpoint分支不可达。
同一set_track_state操作非分组时本地undoDepth大于0，分组时为0；定向诊断通过。
记录在probe/ara/undo_group_diagnostic.rs及ignored .build-tmp/undo-group-diagnostic.log。
这证明本地分组检查点丢失，不证明REAPER具体在哪一步漏记/漏恢复私有FX状态。
该段描述的是修复前证据；2026-10-07删除前置return后，分组检查点已可达，分组/非分组
回归测试均要求undoDepth大于0并通过。

进一步实测：新增diagnostic_grouped_state_round_trip_without_host_history，通过真实
actor分组改轨道volume，自动DSP暂缓时encode_state已保存0.5；restore_state恢复编辑前
和编辑后state，两次get_timeline_state分别回到原值和0.5。单项测试通过（0.01秒），
日志ignored .build-tmp/undo-state-round-trip-diagnostic.log。保存/恢复使用与组件
getState/setState相同的owner入口，不是直接改timeline；仍未模拟REAPER真实历史录入。
因此在此已就绪、单组件、轨道参数场景可以排除“分组更改没有写入组件state”与
“组件恢复后actor必然仍返回新值”，剩余重点是宿主录入/回调及真实几何Undo。

2026-10-07修复并复测分组检查点：删除commands.rs前置begin/end的提前return，让
ensure_loaded之后的checkpoint分支真正执行。原来的分组对照改为要求undoDepth>0；
非分组/分组与state往返两项均通过。分组现在先保存一个基线检查点，组内尾笔仍由
suppress_history合并为同一历史步。宿主历史仍由REAPER写块负责，真实宿主回调/冷
恢复尚未实测；IEditController::setComponentHandler仍是明确接口缺口，不能把本地
修复写成REAPER历史已验收。

原生菜单只读采集未取得Undo标签：Computer Use先遇到输入坐标缓存/元素缓存错误，
刷新后报告用户输入，随后新观察出现用户打开的媒体对象属性窗口。没有点击撤销或
关闭用户窗口，也没有改工程；停止该轮输入，不将未读到标签当作没有HFS历史条目。

## 2. 轨道空白区和clip的右键菜单缺失（轨道头允许禁用）

用户确认：参数面板在选择模式下正常，之前“所有右键都不出”是描述误差；轨道面板
clip截图只有“在播放头处分割”。不再将参数面板当作故障或重复做其焦点测试。

逐项源码确认：

- 轨道头：TrackList.tsx:1463的onContextMenu在isPluginMode时preventDefault并return，
  不设置trackCtxMenu。
- 轨道空白区：TimelinePanel.tsx:4021在没有命中clip时插件模式直接return，
  不设置trackAreaMenu；clipSplitting未开放时内核整个普通菜单入口也不提供。
- clip：ClipContextMenu.tsx:398插件分支提前return，单独只渲染分割MenuItem，
  不进入后面的原App完整菜单。

结论：这三处是插件能力适配时主动裁剪菜单，并非已证的DOM右键总体失效。
用户最新决定：轨道头插件早退保持；轨道空白区不能整体消失。已删除空白分支的
插件早退与普通菜单入口的clipSplitting总门禁。TrackAreaContextMenu保留原三项：
粘贴、在播放头处分割、关闭间隙；插件分割按真实能力启用，未接宿主写口的粘贴/
关闭间隙显示禁用并解释宿主管理，回调再校验准入。独立App三个动作保持原逻辑。
单文件6项定向前端测试通过，包含两种插件能力场景；尚未打包或实机验收，不改变
参数面板和轨道头，不将恢复菜单显示说成插件粘贴/关闭间隙功能已实现。
恢复时须按真实宿主能力逐项开放：轨道增删/重排/改名等仍有插件写入门禁，不能仅
删除isPluginMode判断而调用App私有几何修改，制造只改HFS、不改REAPER的双权威。
其它三项仍仅排查；clip完整菜单尚未恢复，参数面板未变更。

## 3. 各参数绘制后消失，渲染完成才回来

源码链路：after_write通知timeline变化 →前端paramsEpoch变化 →usePianoRollData在同一
参数/轨道scope下仍setParamView(null)、清叠加并强制取数。笔画期间有推迟机制，但松手
后恢复强制刷新就会清空窗口，已提交预览无法独立支撑画面。

后端EditorSession.run在同一个actor中同步执行自动apply/神经渲染，随后才消费下一条
get_param_frames。因此UI清空后，读取可能排在渲染之后，和用户观察一致。
这是通用参数读取链，不限张力。需要保留同scope已提交可见数据，并将DSP计算与只读
参数命令解耦；切参数/切轨和真正Undo仍必须废弃错误scope，不能一律不刷新。

## 4. 连续绘制变成clip外回退、边界尖跳

已确认：ParameterAtlas当前按clip捕获源域曲线。project_roots先clear_curves，再只复制
每个clip的frame_range；空白处回落pad（hifigan_tension为0）。用户判断“只存clip”正确。

边界附加问题：frame_range对clip首帧floor、末帧ceil；SourceTimeMap严格拒绝clip范围外
的项目时刻。SourceCurve.project于是把向外取整的首尾格点留为pad=0，project_roots又
把这些默认格点覆盖到整轨，形成细尖跳，重叠区域还可能覆盖其他clip已有值。

实测一次性诊断（直接include产品ParameterAtlas/SourceTimeMap/Budget，不是Python仿真，
不调用声码器）：输入801帧全部为60的连续张力线，使用真实ha等非网格起点，回流为
3.200s=0、3.205s=60、3.290s=60、3.295s=0、空白3.350s=0；下一clip首尾同样0/60/0。
输出存ignored .build-tmp/curve-projection-diagnostic.log。

所需权威应同时支持整轨（含空白）的用户编辑、各clip随宿主移动/线性拉伸的源basis、
按真正覆盖的网格合并及明确的重叠选择规则。不能靠连线显示隐藏真实数据丢失。

## 排查状态

2026-10-07实现更新：新增稀疏项目时间空白层，与clip源basis并存；显示合并仅复制
真实覆盖格点，外侧源曲线哨兵夹到边界。旧归档按保存布局分離空白层并做scope
过滤/根映射。常量60连续线、非网格边界、clip移动后空白保留与新gap delta、旧布局
迁移已在产品函数定向测试中通过。首轮1个恢复测试漏rebind，补真实步骤后通过。
插件同scope刷新不先清窗；刷新策略/笔画推迟9项及tsc通过，独立App规则不变。
未打包、未实机验收，完全空轨/静音编辑冷恢复未验证；同步DSP阻塞actor仍存在。

Release后可见隔离UI smoke已确认新包打开原生VST3窗口、显示两段clip和宿主播放控件；
WebView子区域右键被Computer Use跨进程目标保护拒绝，未绕过或重复点击，因此菜单、
撤销和媒体回流仍以源码/fixture证据为主，隔离实例随后关闭。

新增操作实现：已整理完整操作清单、五批计划；原生捕获/身份重建/创建/删除、
独立剪贴板数据和GUID参数seed、GUI copy/cut/paste/delete与空白粘贴源码链已连接。
Native先经过actor写入屏障，再在主线程操作；媒体修改完成回流前不收尾其Undo请求。
未静音新item必须真正取得ARA授权/source，不能仅GUI清单存在就成功。创建/删除回执
按新GUID/旧GUI身份与位置/长度/源起点/倍率/目标轨道确认，重试只读不重复写入。
3项parser/跨身份seed检查通过，新增2项解码检查修正fixture缺name后通过，前端13项
及tsc通过；这些不证明真实宿主媒体API/Undo已正常。静音复制波形/冷恢复仍需后续
闭合，其它操作仍待实现。没有新安装包，没有声称全流程已可用。

能力准入补充：插件模式下copy/cut/paste/delete现在必须同时具备clipClipboard能力；
旧插件不会再把这些操作送入独立App actor。该门禁与空白区菜单的禁用提示一致，
不代表当前安装包已包含新宿主媒体实现。

曲线尖跳/空白数据丢失已有产品函数复现；消失的清窗/队列链已定位。
轨道面板三类菜单裁剪已逐项定位，参数面板排除；无需继续验证“所有区域事件失效”。
Undo接口/路由及本地分组检查点缺口已定位，但真实宿主录入/恢复失败点仍待闭合。
未宣告全部目标完成，未调用goal complete。

2026-10-07已补宿主handler桥：setComponentHandler查询并保留IComponentHandler2的
COM引用；actor/DSP只置dirty旗标，WebView UI timer调用setDirty，terminate/drop释放
引用。cargo check通过。真实REAPER历史条目和Undo后setState冷恢复仍需集中实机验证，
不能把源码检查写成宿主验收。
