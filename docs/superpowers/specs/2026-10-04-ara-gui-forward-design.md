# ARA 正向 GUI 全流程设计

中文设计记录，2026-10-04。用户已授权不做倒放，继续完成原 GUI 全流程；沿用已批准
v1 的独立 app 客户端方案，自主执行，不重复设计审批。此增量覆盖历史方向停止条件，
不改写失败证据，也不声称反向 take 会被正确处理。

## 范围与验收

Windows x64 / REAPER 7.81。正向 mono/stereo、44100/48000Hz、每源最多30秒，
普通/裁切片段；现有先导资源预算继续生效。倒放不支持，GUI连接面板明确显示该限制。
本轮不承诺时间拉伸/宿主内容淡化，已有拒绝行为保持。独立app仍可照常导入编辑。

验收：在隔离REAPER挂插件，原HiFiShifter GUI列出实例并连接，显示宿主clip和波形，
可使用既有参数编辑器修改音高/音量，显式提交到REAPER；实际导出与未编辑不同；
关闭GUI、保存并重开一次性REAPER工程，编辑曲线及输出保持。A5无Tauri/WebView入插件。
不把“代码/单测通过”当宿主验收，不把“GUI可连接”当修音验收。

## 组件与权威

1. 内核新增注入PCM的mixdown入口。独立app默认文件入口不变。注入路径必须缺源即失败，
   不回退读persistentID；禁止复用文件域缓存，宿主全PCM内容哈希影响源指纹。
2. 插件文档保存当前映射TimelineState；renderer仅渲染实际分配的clip，编辑状态不改
   DAW几何。显式提交在非实时线程运行现有WORLD/HiFiGAN内核，再发布不可变快照。
   未编辑仍用已实测原声快照，提交失败保留旧状态或明确撤销，不伪报成功。
3. `hifishifter-ara-ipc`共享长度前缀JSON协议，Windows本机命名管道，拒绝远端。
   发现目录默认为LOCALAPPDATA/HiFiShifter/ara-instances，可用HIFISHIFTER_ARA_INSTANCE_DIR
   指到隔离目录。每条记录包含实例ID/进程ID/管道/名称/心跳；连接检查协议版本和会话token。
   有界帧64MiB，读写有超时，后台服务器析构可退出，不泄漏DLL执行线程。
4. Snapshot返回revision/model_revision/当前timeline/授权PCM。GUI将收到的PCM写为
   自有临时WAV供既有波形与分析管线使用，绝不按宿主persistentID打开原源。导入前
   不覆盖未保存工程；需要用户在GUI正常确认替换或新建干净工程。
5. Commit带base_revision/model_revision和GUI的timeline。插件只接受曲线和轨道合成
   参数；忽略/拒绝客户端几何变更。旧revision或宿主布局变化返回Conflict，不静默覆盖。
   按钮显式提交与刷新，不引入自动同步造成用户编辑被轮询覆盖。
6. VST3组件state保存有版本的编辑状态。setState使revision单调前进，触发已有宿主PCM
   重建。源PCM不塞进工程state，由ARA重新授权供应。空旧state合法，损坏/超限state拒绝。

## 文件边界

内核注入：`backend/hifishifter-kernel/src/mixdown.rs`及其测试。
协议：`backend/hifishifter-ara-ipc/{Cargo.toml,src/lib.rs,src/transport.rs}`。
插件：`src/state_channel.rs`、`src/state_stream.rs`、render/document/extension、ara/model、vst3。
app：`src/ara_bridge.rs`、commands门面与注册；frontend新增`features/ara/AraConnectionPanel.tsx`。
不改SDK/registry源，不push，不git add -A，中文文件头和关键函数doc。

## 验证与风险

TDD覆盖PCM权威/缺源、独立文件入口、修音差异、IPC真实往返、冲突/坏帧/关闭、state真实
IBStream短读写、GUI连接和提交错误。最终构建插件与app，frontend build，重跑插件/IPC/
相关内核和app测试。已有四条Windows/tmp失败不处理。
集中审查后，隔离REAPER真实GUI修改参数并导出，记录曲线、源/输出哈希、截图、原始日志，
再关GUI重开项目比对输出。全PCM指纹避免离线改源命中过期文件缓存。
残余：退役快照预算耗尽可能拒绝继续提交；30秒限制是已知先导能力，不称完整发布。
