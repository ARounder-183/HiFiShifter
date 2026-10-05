# ARA 二期：REAPER 内嵌原 GUI 与自动应用设计

中文设计，2026-10-05。用户授权按建议持续实施、不逐步询问，并要求测试集中到最后。
工作树`E:/code/HiFiShifter/.worktrees/ara-plugin`，分支`codex/ara-plugin`。

## 目标与完成定义

打开REAPER中HiFiShifter的FX编辑器，直接显示原React时间线、波形、钢琴卷帘与参数
编辑器；不启动HiFiShifter.exe、不选择外部实例、不手动点提交。修改支持的参数后，
后台自动预渲染并在REAPER播放/导出；保存重开恢复曲线与输出。原独立Tauri app仍可
启动、导入本地音频和编辑，不变成只能连接DAW的客户端。

二期不是重写简化GUI，不用模拟编辑器替代原功能，不以一个能显示的空WebView完成验收。
先实施Windows x64/REAPER7.81，沿用正向普通/裁切、mono/stereo、44100/48000Hz、
每源≤30秒、有界PCM/快照预算。倒放仍不做，stretch/content fades尚不支持，显式提示。
“整个做成插件”指完整原编辑工作区、支持的参数/分析/合成链路及宿主生命周期；DAW模式
不接管独立音频设备，不允许GUI改变宿主clip几何或另行打开文件冒充ARA源。项目文件
打开/保存、录音和设备选择在插件模式明确禁用/由宿主接管，不伪报成功。全功能硬件/
算法矩阵和macOS插件不是本轮证据，独立app跨平台行为必须保留。

## 已有证据与缺口

一期真实220.5→393.75Hz、关GUI重开PCM maxdiff0和用户恢复曲线确认已记录在
`probe/ara/captures/forward-gui-FINDINGS.md`。缺口在`vst3.rs::create_view`返回null、
原前端事件/窗口API直连Tauri，以及编辑后需要显式ara_submit。源权威、非实时WORLD
渲染、版本冲突和v2持久身份复用，不另造音频实现。

## 路线选择

1. 采用原生VST3 IPlugView子窗口+WebView2，复用原前端，编辑命令由插件内无Tauri
   编辑会话执行。宿主提供主线程消息循环，UI异步初始化，不跑自己的run/消息泵。
2. 不采用SetParent外部Tauri app：跨进程焦点/析构不可靠，而且仍另开应用。
3. 不把Tauri App runtime直接链接插件：其主事件循环、进程级state/设备可能与宿主
   冲突。不以另写原生简化钢琴卷帘替代已有完整工作区。

一期A5改为“插件不依赖Tauri、wry、app crate或cpal；允许原生webview2-com”。
UI只在编辑器打开时创建，关闭不丢曲线或停供音；WebView2浏览器辅助进程不等于
另开的HiFiShifter app。WebView创建失败在插件窗口和日志可见，宿主不能被panic终止。

## 窗口与生命周期

`IPlugView`按锁定iplugview.h定义，包括FUnknown、平台、attach/remove、键鼠、尺寸、
焦点、frame与constraint。只在Windows接受HWND；只创建/销毁自己的WS_CHILD窗口，
不更改REAPER父窗口、DPI策略或主COM apartment。COM初始化仅在UI线程，平衡自己
成功的CoInitializeEx；RPC_E_CHANGED_MODE显式失败，不重新初始化宿主。

view与controller/processor使用真实IConnectionPoint关联；不查“最后一个实例”、
不按轨道名猜归属。view保有实例会话弱关联，关闭窗口/销毁controller使请求取消，
迟到的WebView异步回调检查取消generation，不能复活窗口或访问已释放裸指针。
setFrame持有/释放COM引用，removed可重复调用。尺寸仅onSize改变，keyboard仅在
真实转发给编辑器且被处理时返回true，不能吞掉REAPER快捷键。多实例并存隔离。

WebView2环境创建没有取消API；窗口weak失效只能取消效果，不能保证迟到的COM函数
不进入DLL。当前Windows路线在首次浏览器初始化时只将本模块固定到REAPER进程退出，
仍及时释放窗口/COM浏览器与worker，不让异步callback执行已卸载代码。代价是宿主
进程内不能热卸载/更新插件DLL，升级须正常退出REAPER；不能把它写成支持热重载。

## 前端宿主通信

新增`services/pluginHost.ts`，原`invoke.ts`继续位置参数→命名参数映射，再分发插件
bridge；独立Tauri/pywebview分支保留。插件bootstrap只注入`__HFS_PLUGIN_BOOTSTRAP__`
含`version:1,viewId`，不伪造`__TAURI__`。新`services/hostEvents.ts`统一事件订阅，
插件走本地bridge，独立app懒加载既有Tauri事件。窗口/拖放/分离面板能力由宿主模式
显式判断，不在插件加载Tauri window API。

UI协议：request `{version:1,viewId,id,command,args}`；response
`{version:1,viewId,id,ok,value?,error?}`；event
`{version:1,viewId,event,payload}`。最多128未完成请求，每个请求30秒超时，关闭全部
reject，viewId/版本不符或未知id不接收。原源PCM不经UI消息流往返，留插件会话；
UI只取有界波形/参数。命令侧类型校验、宿主来源校验、预算和明确unsupported，
不能使用任意JS调用来执行shell/开放路径读取。

bundle自带frontend资产，以固定虚拟HTTPS域映射只读资源目录；禁止远端导航、弹窗
与非授权资源，IPC只接受固定本地origin消息。WebView数据目录单独按插件实例隔离，
不复用本机发生0x800700AA的独立app profile；不删除用户profile、不改系统安全设置。

## 共享编辑命令与原GUI

内核新增`editor`模块：实例级TimelineState/undo/redo、事件出口、选区与曲线读写。
从现有params/common/core命令提取无Tauri逻辑，app命令是薄适配；不复制一套语义
相似但不同的pitch/动态/平滑实现。音高分析、waveform、renderer描述符复用既有
内核；插件UI状态由`EditorSession`拥有，源访问由ARA会话拥有。无cpal初始化。
独立app默认文件入口、原菜单/窗口、显式ARA客户端模式仍在。

原App在插件模式使用原DockRoot和参数面板，隐藏外部连接/提交栏，显示自动应用状态。
宿主几何只读；不支持的命令拒绝并禁用入口。改变pitch/gain/renderer参数、undo/redo
均使自动渲染generation前进；选区/滚动/主题变化不触发音频重渲染。

## 自动应用与持久化

实例后台worker合并连续编辑（150ms quiet period），只渲染最新代次；分块写入尚未
完成时保留dirty状态。不可变音频快照只在generation与host model_revision仍匹配时
发布。后台失败保留已发布音频，界面明确“尚未应用/失败”，绝不显示假同步。实际宿主
内容/几何变化保持Conflict保护；纯sampleAccess开关不是几何变更。

用户编辑权威立即入组件state，不等待预渲染完成才可保存；非实时getState序列化已
接受参数，重开按v2宿主持久身份恢复后重新预渲染。save/关闭UI时flush最新编辑到权威，
不能因debounce漏掉最后一笔。处理器只读取原子快照，IO/IPC/WebView/推理/锁等待
不在process或setProcessing。全局内核缓存不等于实例编辑权威，事件不串实例。

## 部署与集中验收

规范Windows VST3 bundle含x86_64-win模块、frontend资产和必要依赖；不依赖REAPER
CWD/系统PATH，launcher只能用于隔离采集，不能替代发行部署。动态加载邻接依赖，
插件目录由当前模块路径解析，不用主仓库或开发机绝对路径。

最终一次集中执行：共享editor/app/frontend/plugin测试、build、依赖树与native ABI；
修发现后只定向重测。真实REAPER打开内嵌原GUI，拖尺寸、画pitch、自动应用、导出
音高差异，关闭FX后供音，重开FX/RPP恢复、双实例不串、快速编辑最后一笔、undo/redo、
真实宿主改源/几何失效、seek、44100/48000 mono/stereo与30秒矩阵；独立app导入回归。
记录命令/退出码/日志/PCM/截图；未实测项保持open，禁止把源码实现或空窗口当完整通过。

## 用户试用补充（2026-10-05）

高优：宿主停播实时process必须静音，offline导出不受kPlaying门禁影响；原GUI位置采用一次绝对项目时间，停播seek同样跟随宿主。多轨道/复制插件合法共享ARA source/modification，持久化恢复范围由该组件实际assigned regions确定，不再把共享对象等同归属歧义。每组件只保存本范围参数，未限定范围的歧义恢复继续拒绝。另一轨编辑不能制造本轨假Conflict，同轨旧写入仍拒绝。

后续中优：原GUI播放/暂停请求通过可选ARAPlaybackControllerInterface在宿主主线程执行；不启动独立设备、不用REAPER脚本或快捷键模拟控制。不支持的宿主保持明确disabled/unsupported。后续低优：稳定host model变更且没有待应用本地编辑时自动同步；存在真实冲突仍保留曲线，禁止静默丢最后一笔。

BPM补充：VST3 ProcessContext的kTempoValid字段作为当前宿主BPM权威，无有效字段时保留最后值；actor只同步标量tempo，不宣称完整Tempo Map。REAPER项目Timebase为Beats(position,length,rate)时改BPM会同时变更clip倍率，这属于尚未支持的time stretch，不由BPM数字同步“修复”。纯BPM先导验证使用测试副本的Time时间基准，绝不替用户更改原项目默认值。
