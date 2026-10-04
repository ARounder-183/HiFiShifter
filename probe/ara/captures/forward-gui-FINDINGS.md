# 正向原 GUI 全流程：进行中的实测记录

2026-10-04；一次性隔离REAPER工程，非用户原工程。验收未完成。

## 已实测

- 独立app启用custom-protocol、嵌入现有frontend/dist后构建成功。插件另一次cargo单独构建
  成功（避免workspace feature统一带入vslib）；dumpbin依赖无vslib/Tauri/WebView。
- 设置worktree专属WEBVIEW2_USER_DATA_FOLDER后原HiFiShifter GUI持续打开，不是闪退。
  此证据只覆盖当前隔离profile，未证明用户原WebView2 profile已修复。
- 最新GUI连接到真实REAPER实例，显示0秒起2秒片段及3秒起1秒裁切片段、波形与音高编辑器。
  快照为宿主供应的1个源、2个片段；无按persistentID读源文件的回退。
- plugin60测试、IPC4测试新鲜重跑全部通过；大PCM真实管道测试全套共0.08秒。
  历史大帧一次写入超时保留；16KiB分块修复不是小消息通过的推断。

## 用户截图暴露的真实冲突

GUI r0/m4手绘后提交，显示`Conflict: host model changed; refresh`。
插件原日志成功Snapshot后只有两次samples_access=false，没有begin_editing/片段变更。
源码refresh_source_pcm无条件clear_renderers，后者无条件revision++，使单纯reader授权状态
变更错误地令GUI模型版本过期。不能直接去掉所有冲突保护；真实内容/几何版本仍必须守住。
源码已修复并回归：8eba3ae5分离访问撤销和真实模型版本；同一Snapshot在纯授权切换后
Commit仍成功，真实源content/geometry更新后仍拒绝。controller新鲜运行plugin68+IPC4
正常通过。新DLL未部署到当前用户窗口，真实宿主复验未完成；禁止自动刷新覆盖手绘曲线。

用户随后自行操作出现`GUI commit ready revision=1 model=8`；只读IPC检查也返回revision1、
2clips、1source。它证明一次提交已被插件接受，不证明REAPER导出的音频真的改变或持久化。

## 未完成

独立输出验证器`verify_forward_gui_output.ps1`已增加：仅gain/未变/非有限/间隙漏音/重开变音
均拒绝，真实音高变化接受；6条行为回归通过。实际24bit基线与float32源经独立RIFF解析比对，
普通/裁切/间隙maxdiff=5.96046447753906e-8。频率oracle使用该周期夹具的独立自相关，
不读取app音高分析值；仅适用于此次合成夹具，不冒称对任意人声通用的pitch验证器。
尚无edited/reopened实际导出文件，所以此验证器测试通过不代表GUI修音或保存验收通过。

集中审查：局部renderer共享编辑、unsupported修改/dirty、临时轨道ID恢复、compose关闭
手绘pitch四项已实现定向修复。首轮复审F1/F3/F4及授权误冲突通过；F2的非checkpoint并发
尾块另以0b8f4dd4修正实际Commit参数相等检查与四个真实参数写入的dirty记账。
controller新鲜运行ARA18+params10，均正常exit0；移除本P1修复的真实5条RED正常exit1。
不造额外version或undo。最终窄复审F2/R1为ADDRESSED，未发现新问题；五项源码修复均通过
定向检查，但不代替真实宿主验收。完整报告在本plan的SDD目录。

controller还重跑kernel mixdown28、app项目21、前端8及frontend build；全部正常通过。
WORLD运行有既有ONNX初始化回退诊断，不把它描述为无警告；真实WORLD音频仍产生差异。
部分早期真实参数测试虽输出通过摘要却没退出，不算GREEN。根因是fixture缺省Nsf算法
触发FCPE异步预热；改为显式no-device/None算法测试fixture后取得正常0/1退出，产品DSP/
AudioEngine生命周期没改。该fixture只验证命令/dirty契约，真实DSP另由WORLD音频oracle验证。

构建产物：GUI隔离构建到backend/target/ara-fix-app/debug/HiFiShifter.exe，插件在
backend/target/debug/hifishifter_plugin.dll；二者分开构建，插件imports无vslib/Tauri/WebView。
启动脚本选择两份本worktreeGUI产物中最新版本，并在旧GUI/REAPER运行时拒绝替换，未强杀。
最终嵌入dist GUI生产构建正常exit0（38.55秒），产物92661760 bytes；新DLL/GUI未热替换当前窗口。

真实REAPER导出音高差异、关闭GUI后保存/重开仍保持曲线与输出、同路径改源不能命中过期缓存
尚无证据。倒放不做；stretch/content fades仍不承诺；每源30秒、mono/stereo、44.1/48k限制不变。

用户用Escape停止Computer Use；保留当前GUI/隔离REAPER与用户编辑，不为部署强制终止。
